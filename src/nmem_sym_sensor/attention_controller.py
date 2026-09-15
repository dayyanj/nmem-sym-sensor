"""Attention Controller — knowledge-driven three-phase attention state machine.

Replaces the threshold-based familiarity gate with a system that transitions
based on what the system KNOWS, not how much time has passed.

Three phases:
  EXPLORE  — "What's here?" Full processing, build scene model.
  INSPECT  — "What's interesting?" Focus on unexamined detail.
  MONITOR  — "Has anything changed?" Minimal compute, deviation detection.

Transitions are driven by:
  - Prediction accuracy (EmbeddingPredictor.running_error)
  - Foveal coverage (fraction of salient regions visited)
  - New candidate rate (are new things still appearing?)
  - Surprise rate (are predictions failing?)
  - Scene recognition (is this a return visit?)

Frame rate independent — all signals are rates or ratios, not frame counts.
"""
import logging
import time
from dataclasses import dataclass
from enum import Enum

from nmem_sym_sensor.debug import is_enabled, trace

log = logging.getLogger(__name__)


class AttentionPhase(Enum):
    EXPLORE = "explore"
    INSPECT = "inspect"
    MONITOR = "monitor"


@dataclass
class AttentionBudget:
    """Processing budget for the current frame."""
    phase: AttentionPhase = AttentionPhase.EXPLORE
    max_saccades: int = 8
    edge_follow_steps: int = 6
    run_intelligence: bool = True
    frame_familiarity: float = 0.0
    is_return_visit: bool = False
    cluster_cache: object = None       # ClusterCache for object-level matching
    promoted_visual_count: int = 0     # set after analysis


@dataclass
class AttentionSignals:
    """Signals fed to the controller each frame.

    All rates are per-second, all ratios are 0-1.
    Frame rate independent.
    """
    prediction_accuracy: float = 0.0    # 1 - running_error (0=bad, 1=perfect)
    candidate_coverage: float = 0.0     # fraction of salient area visited by fovea
    new_candidate_count: int = 0        # new candidates this frame
    surprise: float = 0.0              # current frame surprise (0-1)
    is_return_visit: bool = False       # scene recognised by chronoception
    scene_changed: bool = False         # scene transition detected
    promoted_visual_count: int = 0      # visual nodes promoted this frame


class AttentionController:
    """Knowledge-driven attention state machine.

    The brain spends full effort exploring a new scene, shifts to
    inspecting details once the broad strokes are learned, then drops
    to monitoring for changes. A triangle enters INSPECT in 2 seconds.
    A busy street stays in EXPLORE for 30 seconds.
    """

    def __init__(
        self,
        # EXPLORE → INSPECT thresholds
        explore_prediction_threshold: float = 0.7,
        explore_coverage_threshold: float = 0.6,
        explore_no_new_candidates_s: float = 2.0,
        # INSPECT → MONITOR thresholds
        inspect_surprise_threshold: float = 0.1,
        inspect_stable_duration_s: float = 5.0,
        # MONITOR → re-engagement thresholds
        monitor_surprise_threshold: float = 0.3,
        monitor_major_surprise: float = 0.5,
        # Processing budgets per phase
        explore_saccades: int = 8,
        explore_edge_steps: int = 6,
        inspect_saccades: int = 5,
        inspect_edge_steps: int = 3,
        monitor_saccades: int = 3,
        monitor_edge_steps: int = 1,
        # Return visit acceleration
        return_visit_explore_coverage: float = 0.3,
    ):
        self.phase = AttentionPhase.EXPLORE
        self._phase_entered_at = time.monotonic()

        # Transition thresholds
        self._explore_pred_thresh = explore_prediction_threshold
        self._explore_cov_thresh = explore_coverage_threshold
        self._explore_no_new_s = explore_no_new_candidates_s
        self._inspect_surprise_thresh = inspect_surprise_threshold
        self._inspect_stable_s = inspect_stable_duration_s
        self._monitor_surprise_thresh = monitor_surprise_threshold
        self._monitor_major_surprise = monitor_major_surprise
        self._return_visit_cov = return_visit_explore_coverage

        # Budget per phase
        self._budgets = {
            AttentionPhase.EXPLORE: (explore_saccades, explore_edge_steps),
            AttentionPhase.INSPECT: (inspect_saccades, inspect_edge_steps),
            AttentionPhase.MONITOR: (monitor_saccades, monitor_edge_steps),
        }

        # Tracking
        self._last_new_candidate_at = time.monotonic()
        self._stable_since: float | None = None  # when surprise dropped below threshold
        self._is_return_visit = False

        # Stats
        self.phase_transitions = 0
        self.frames_in_phase = {p: 0 for p in AttentionPhase}

    def update(self, signals: AttentionSignals) -> AttentionBudget:
        """Called each frame. Check transitions, return processing budget.

        Args:
            signals: Current frame's attention signals.

        Returns:
            AttentionBudget with phase-appropriate processing parameters.
        """
        now = time.monotonic()

        # Track new candidate timing
        if signals.new_candidate_count > 0:
            self._last_new_candidate_at = now

        # Track surprise stability
        if signals.surprise < self._inspect_surprise_thresh:
            if self._stable_since is None:
                self._stable_since = now
        else:
            self._stable_since = None

        # Scene change → reset to EXPLORE
        if signals.scene_changed:
            self._is_return_visit = signals.is_return_visit
            self._transition(AttentionPhase.EXPLORE, "scene_change",
                             return_visit=signals.is_return_visit)

        # Check phase transitions
        self._check_transitions(signals, now)

        # Build budget
        saccades, edge_steps = self._budgets[self.phase]
        self.frames_in_phase[self.phase] += 1

        budget = AttentionBudget(
            phase=self.phase,
            max_saccades=saccades,
            edge_follow_steps=edge_steps,
            run_intelligence=True,  # always run — it's cheap
            frame_familiarity=signals.prediction_accuracy,
            is_return_visit=self._is_return_visit,
        )

        if is_enabled("attention"):
            trace("attention", "frame",
                  phase=self.phase.value,
                  pred_acc=signals.prediction_accuracy,
                  coverage=signals.candidate_coverage,
                  surprise=signals.surprise,
                  new_cands=signals.new_candidate_count,
                  saccades=saccades,
                  promoted=signals.promoted_visual_count)

        return budget

    def _check_transitions(self, signals: AttentionSignals, now: float):
        """Evaluate transition conditions for current phase."""

        if self.phase == AttentionPhase.EXPLORE:
            self._check_explore_to_inspect(signals, now)

        elif self.phase == AttentionPhase.INSPECT:
            self._check_inspect_to_monitor(signals, now)
            self._check_inspect_to_explore(signals, now)

        elif self.phase == AttentionPhase.MONITOR:
            self._check_monitor_to_inspect(signals, now)
            self._check_monitor_to_explore(signals, now)

    def _check_explore_to_inspect(self, signals: AttentionSignals, now: float):
        """EXPLORE → INSPECT when the scene is broadly understood."""
        # Return visits need less coverage before transitioning
        coverage_threshold = (
            self._return_visit_cov if self._is_return_visit
            else self._explore_cov_thresh
        )

        time_since_new = now - self._last_new_candidate_at
        conditions = {
            "prediction_ok": signals.prediction_accuracy >= self._explore_pred_thresh,
            "coverage_ok": signals.candidate_coverage >= coverage_threshold,
            "no_new_candidates": time_since_new >= self._explore_no_new_s,
        }

        if all(conditions.values()):
            self._transition(
                AttentionPhase.INSPECT,
                "explore_complete",
                pred=round(signals.prediction_accuracy, 3),
                cov=round(signals.candidate_coverage, 3),
                no_new_s=round(time_since_new, 1),
                return_visit=self._is_return_visit,
            )

    def _check_inspect_to_monitor(self, signals: AttentionSignals, now: float):
        """INSPECT → MONITOR when everything has been examined."""
        if self._stable_since is None:
            return

        stable_duration = now - self._stable_since
        if (stable_duration >= self._inspect_stable_s
                and signals.prediction_accuracy >= 0.85):
            self._transition(
                AttentionPhase.MONITOR,
                "inspection_complete",
                stable_s=round(stable_duration, 1),
                pred=round(signals.prediction_accuracy, 3),
            )

    def _check_inspect_to_explore(self, signals: AttentionSignals, now: float):
        """INSPECT → EXPLORE if major surprise (something big changed)."""
        if signals.surprise > self._monitor_major_surprise:
            self._transition(
                AttentionPhase.EXPLORE,
                "major_surprise_in_inspect",
                surprise=round(signals.surprise, 3),
            )

    def _check_monitor_to_inspect(self, signals: AttentionSignals, now: float):
        """MONITOR → INSPECT when something specific changes."""
        if signals.surprise > self._monitor_surprise_thresh:
            self._transition(
                AttentionPhase.INSPECT,
                "change_detected",
                surprise=round(signals.surprise, 3),
            )

    def _check_monitor_to_explore(self, signals: AttentionSignals, now: float):
        """MONITOR → EXPLORE on major surprise (scene-level change)."""
        if signals.surprise > self._monitor_major_surprise:
            self._transition(
                AttentionPhase.EXPLORE,
                "major_surprise_in_monitor",
                surprise=round(signals.surprise, 3),
            )

    def _transition(self, new_phase: AttentionPhase, reason: str, **kwargs):
        """Execute a phase transition."""
        old_phase = self.phase
        if new_phase == old_phase:
            return

        duration = time.monotonic() - self._phase_entered_at
        self.phase = new_phase
        self._phase_entered_at = time.monotonic()
        self.phase_transitions += 1

        # Reset phase-specific state
        if new_phase == AttentionPhase.EXPLORE:
            self._last_new_candidate_at = time.monotonic()
        elif new_phase == AttentionPhase.INSPECT:
            self._stable_since = None

        log.info("Attention: %s → %s (%s) after %.1fs",
                 old_phase.value, new_phase.value, reason, duration)

        trace("attention", "phase_change",
              old=old_phase.value, new=new_phase.value,
              reason=reason, duration_s=round(duration, 2), **kwargs)

    @property
    def phase_duration_s(self) -> float:
        """How long we've been in the current phase."""
        return time.monotonic() - self._phase_entered_at

    def reset(self):
        """Full reset — return to EXPLORE."""
        self.phase = AttentionPhase.EXPLORE
        self._phase_entered_at = time.monotonic()
        self._last_new_candidate_at = time.monotonic()
        self._stable_since = None
        self._is_return_visit = False

    def stats(self) -> dict:
        return {
            "phase": self.phase.value,
            "phase_duration_s": round(self.phase_duration_s, 1),
            "phase_transitions": self.phase_transitions,
            "is_return_visit": self._is_return_visit,
            "frames_per_phase": {p.value: c for p, c in self.frames_in_phase.items()},
        }
