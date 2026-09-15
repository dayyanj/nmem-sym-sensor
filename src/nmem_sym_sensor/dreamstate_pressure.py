"""Demand-driven dreamstate — pressure-triggered consolidation.

The brain doesn't consolidate on a timer. It consolidates when there's
PRESSURE to do so: too many prediction failures, too many unprocessed
associations, or nothing new to observe. Each dreamstate function has
its own trigger condition — a measurable pressure that builds during
observation and releases during idle periods.

This module tracks all pressure signals and determines which dreamstate
functions need to run at any given moment. The dreamstate becomes a
collection of independently-triggered operations rather than a monolithic
cycle.

Integration:
- SensorGraph.process_frame() feeds surprise and co-occurrence signals
- AttentionController phase transitions feed the idle tracker
- sensory_dreamstate reads what_needs_running() to decide which steps to execute
"""
import logging
import time

log = logging.getLogger(__name__)


# ── Signal Trackers ───────────────────────────────────────


class SurpriseAccumulator:
    """Rolling average of surprise scores.

    High sustained surprise means the internal model is significantly
    wrong — triggers full consolidation.
    """

    def __init__(self, window_s: float = 30.0, threshold: float = 0.6, min_observations: int = 10):
        self.observations: list[tuple[float, float]] = []  # (timestamp, surprise)
        self.window_s = window_s
        self.threshold = threshold
        self.min_observations = min_observations

    def observe(self, timestamp: float, surprise: float):
        self.observations.append((timestamp, surprise))
        cutoff = timestamp - self.window_s
        self.observations = [(t, s) for t, s in self.observations if t > cutoff]

    @property
    def pressure(self) -> float:
        if not self.observations:
            return 0.0
        return sum(s for _, s in self.observations) / len(self.observations)

    @property
    def should_trigger(self) -> bool:
        return self.pressure > self.threshold and len(self.observations) >= self.min_observations

    def reset(self):
        self.observations.clear()


class CooccurrenceVolumeTracker:
    """Counts new co-occurrences since last decay cycle.

    Too many unprocessed associations means noise is accumulating —
    triggers homeostatic decay.
    """

    def __init__(self, threshold: int = 500):
        self.count_since_last_decay = 0
        self.threshold = threshold

    def observe(self, count: int = 1):
        self.count_since_last_decay += count

    @property
    def pressure(self) -> float:
        return min(self.count_since_last_decay / max(self.threshold, 1), 2.0)

    @property
    def should_trigger(self) -> bool:
        return self.count_since_last_decay >= self.threshold

    def reset(self):
        self.count_since_last_decay = 0


class AccuracyTracker:
    """Rolling prediction accuracy.

    Sustained low accuracy means stored associations are wrong —
    triggers self-play to test and weaken failing associations.
    """

    def __init__(self, window_s: float = 20.0, threshold: float = 0.4, min_observations: int = 5):
        self.observations: list[tuple[float, float]] = []  # (timestamp, accuracy)
        self.window_s = window_s
        self.threshold = threshold
        self.min_observations = min_observations

    def observe(self, timestamp: float, accuracy: float):
        self.observations.append((timestamp, accuracy))
        cutoff = timestamp - self.window_s
        self.observations = [(t, a) for t, a in self.observations if t > cutoff]

    @property
    def pressure(self) -> float:
        if not self.observations:
            return 0.0
        avg = sum(a for _, a in self.observations) / len(self.observations)
        # Invert: low accuracy = high pressure
        return max(0.0, 1.0 - avg / max(self.threshold, 0.01)) if avg < self.threshold else 0.0

    @property
    def should_trigger(self) -> bool:
        if len(self.observations) < self.min_observations:
            return False
        avg = sum(a for _, a in self.observations) / len(self.observations)
        return avg < self.threshold

    def reset(self):
        self.observations.clear()


class IdleTracker:
    """Tracks how long the attention controller has been in MONITOR phase.

    Nothing new happening = natural time to consolidate. This is the
    most important trigger: it naturally balances observation and
    consolidation like daydreaming when bored.
    """

    def __init__(self, idle_threshold_s: float = 60.0):
        self.monitor_entered_at: float | None = None
        self.idle_threshold_s = idle_threshold_s

    def on_phase_change(self, phase: str, timestamp: float | None = None):
        ts = timestamp or time.monotonic()
        if phase == "monitor":
            if self.monitor_entered_at is None:
                self.monitor_entered_at = ts
        else:
            self.monitor_entered_at = None

    @property
    def idle_duration(self) -> float:
        if self.monitor_entered_at is None:
            return 0.0
        return time.monotonic() - self.monitor_entered_at

    @property
    def pressure(self) -> float:
        return min(self.idle_duration / max(self.idle_threshold_s, 1.0), 2.0)

    @property
    def should_trigger_deep(self) -> bool:
        return self.idle_duration > self.idle_threshold_s

    def reset(self):
        self.monitor_entered_at = None


class ConflictTracker:
    """Tracks number of visual nodes with competing voice bindings.

    Multiple sounds fighting for the same visual = unresolved competition.
    Triggers Darwinian competition + focused decay.
    """

    def __init__(self, threshold: int = 10):
        self.conflicted_nodes = 0
        self.threshold = threshold

    def update(self, count: int):
        self.conflicted_nodes = count

    @property
    def pressure(self) -> float:
        return min(self.conflicted_nodes / max(self.threshold, 1), 2.0)

    @property
    def should_trigger(self) -> bool:
        return self.conflicted_nodes >= self.threshold


class SceneNoveltyTracker:
    """Tracks new scenes that need integration.

    First visit to a complex scene = lots to consolidate.
    """

    def __init__(self, member_threshold: int = 10):
        self.pending_scenes: list[int] = []  # scene IDs needing consolidation
        self.member_threshold = member_threshold

    def observe_new_scene(self, scene_id: int, member_count: int):
        if member_count >= self.member_threshold and scene_id not in self.pending_scenes:
            self.pending_scenes.append(scene_id)

    @property
    def pressure(self) -> float:
        return min(len(self.pending_scenes), 5) / 5.0

    @property
    def should_trigger(self) -> bool:
        return len(self.pending_scenes) > 0

    def reset(self):
        self.pending_scenes.clear()


# ── Dreamstate Function Names ─────────────────────────────

# These match the step names used in sensory_dreamstate.py
DECAY = "decay"
SELF_PLAY = "self_play"
CONCEPT_LINKING = "concept_linking"
RECOMBINATION = "recombination"
COMPETITION = "competition"
COMPOSITION = "composition"
SCENE_CONSOLIDATION = "scene_consolidation"
NODE_DECAY = "node_decay"
DEDUP = "dedup"
REACTIVATION = "reactivation"
MENTAL_ROTATION = "mental_rotation"


# ── Main Pressure Controller ──────────────────────────────


class DreamstatePressure:
    """Tracks all pressure signals and determines what needs running.

    Each signal independently triggers specific dreamstate functions.
    Multiple signals can fire simultaneously — the system runs all
    needed functions in one cycle.

    The dreamstate never interrupts active observation. Pressure builds
    during EXPLORE/INSPECT and releases during MONITOR idle periods.
    """

    def __init__(self):
        self.surprise = SurpriseAccumulator(window_s=30.0, threshold=0.6)
        self.cooc_volume = CooccurrenceVolumeTracker(threshold=500)
        self.prediction_accuracy = AccuracyTracker(window_s=20.0, threshold=0.4)
        self.idle = IdleTracker(idle_threshold_s=60.0)
        self.conflicts = ConflictTracker(threshold=10)
        self.scene_novelty = SceneNoveltyTracker(member_threshold=10)
        self.hypothesis_backlog = 0

        # Track when we last ran each function
        self._last_run: dict[str, float] = {}
        # Cooldown: don't re-trigger the same function within N seconds
        self._cooldowns: dict[str, float] = {
            DECAY: 10.0,
            SELF_PLAY: 30.0,
            CONCEPT_LINKING: 60.0,
            RECOMBINATION: 60.0,
            COMPETITION: 30.0,
            COMPOSITION: 60.0,
            DEDUP: 120.0,
            MENTAL_ROTATION: 60.0,
        }

    def what_needs_running(self) -> list[str]:
        """Returns deduplicated list of dreamstate functions that should run now."""
        now = time.monotonic()
        needs: set[str] = set()

        # Signal 1: High sustained surprise -> full consolidation
        if self.surprise.should_trigger:
            needs.update([DECAY, SELF_PLAY, CONCEPT_LINKING, RECOMBINATION])

        # Signal 2: Co-occurrence volume -> decay + self-play
        if self.cooc_volume.should_trigger:
            needs.add(DECAY)
            needs.add(SELF_PLAY)

        # Signal 3: Prediction accuracy drop -> self-play
        if self.prediction_accuracy.should_trigger:
            needs.add(SELF_PLAY)

        # Signal 4: Hypothesis backlog -> recombination
        if self.hypothesis_backlog > 20:
            needs.add(RECOMBINATION)

        # Signal 5: Conflicting bindings -> competition + decay
        if self.conflicts.should_trigger:
            needs.update([COMPETITION, DECAY])

        # Signal 6: New complex scene -> scene consolidation
        if self.scene_novelty.should_trigger:
            needs.add(SCENE_CONSOLIDATION)

        # Signal 7: Deep idle -> expensive operations
        if self.idle.should_trigger_deep:
            needs.update([
                RECOMBINATION, MENTAL_ROTATION, CONCEPT_LINKING,
                COMPOSITION, DEDUP, REACTIVATION,
            ])

        # Signal 8: Staleness — functions that haven't run in a while
        staleness_triggers = {
            SELF_PLAY: 120.0,
            CONCEPT_LINKING: 300.0,
        }
        for fn, max_gap_s in staleness_triggers.items():
            last = self._last_run.get(fn, 0)
            if now - last > max_gap_s:
                needs.add(fn)

        # Apply cooldowns — don't re-trigger functions that ran recently
        cooled: list[str] = []
        for fn in needs:
            last = self._last_run.get(fn, 0)
            cooldown = self._cooldowns.get(fn, 0)
            if now - last >= cooldown:
                cooled.append(fn)

        return cooled

    def mark_ran(self, function_name: str):
        """Record that a dreamstate function just ran."""
        self._last_run[function_name] = time.monotonic()

        # Reset relevant trackers after their functions run
        if function_name == DECAY:
            self.cooc_volume.reset()
        elif function_name == RECOMBINATION:
            pass  # hypothesis_backlog is set externally
        elif function_name == SELF_PLAY:
            self.prediction_accuracy.reset()

    def any_pressure(self) -> bool:
        """Is there any pressure that needs releasing?"""
        return len(self.what_needs_running()) > 0

    def stats(self) -> dict:
        """Current pressure state for dashboard/logging."""
        needs = self.what_needs_running()
        return {
            "surprise_pressure": round(self.surprise.pressure, 3),
            "surprise_trigger": self.surprise.should_trigger,
            "cooc_volume": self.cooc_volume.count_since_last_decay,
            "cooc_trigger": self.cooc_volume.should_trigger,
            "accuracy_pressure": round(self.prediction_accuracy.pressure, 3),
            "accuracy_trigger": self.prediction_accuracy.should_trigger,
            "idle_duration_s": round(self.idle.idle_duration, 1),
            "idle_deep_trigger": self.idle.should_trigger_deep,
            "conflicted_nodes": self.conflicts.conflicted_nodes,
            "conflict_trigger": self.conflicts.should_trigger,
            "hypothesis_backlog": self.hypothesis_backlog,
            "scene_novelty_pending": len(self.scene_novelty.pending_scenes),
            "needs_running": needs,
            "any_pressure": len(needs) > 0,
        }
