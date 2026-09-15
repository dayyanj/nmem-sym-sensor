"""
Compute Budget Allocator — Thalamic Gating.

Distributes a fixed compute budget across sensors and internal
processing based on novelty bids. Biology: the thalamus gates
sensory input to the cortex, routing compute where it's needed.

No sensor is ever fully off — minimum 5% budget ensures change
detection. Unused sensory budget flows to internal processing
(dreamstate, speech practice, imagery).

Phase 1: Fixed equal weights, proportional to novelty.
Phase 2+: Learned weights from experience (plug-in compatible).

References:
- Sherman & Guillery (2006) "Exploring the Thalamus" — thalamic relay
- Kanai et al. (2015) "Cerebral hierarchies" — predictive processing budget
"""
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


TOTAL_BUDGET = 100
MIN_BUDGET = 5


# Budget level thresholds — what each level enables
# These are documented per-sensor but the allocator is sensor-agnostic.
# The consuming code interprets levels, not the allocator.
LEVEL_MINIMAL = 15     # change detection only
LEVEL_REDUCED = 30     # single pass, no intelligence
LEVEL_STANDARD = 50    # full processing
LEVEL_ELEVATED = 70    # full + extended analysis


@dataclass
class SensorBid:
    """A sensor's bid for compute budget based on novelty."""
    sensor: str       # "visual", "auditory", "olfactory", "internal"
    novelty: float    # 0.0 (nothing new) to 1.0 (completely novel)


@dataclass
class BudgetAllocation:
    """Allocated compute budget per sensor/consumer."""
    allocations: dict[str, int] = field(default_factory=dict)

    def get(self, sensor: str) -> int:
        """Get allocated budget for a sensor (default: MIN_BUDGET)."""
        return self.allocations.get(sensor, MIN_BUDGET)

    @property
    def visual(self) -> int:
        return self.get("visual")

    @property
    def auditory(self) -> int:
        return self.get("auditory")

    @property
    def internal(self) -> int:
        return self.get("internal")

    def above(self, sensor: str, level: int) -> bool:
        """Check if a sensor's budget meets a threshold level."""
        return self.get(sensor) >= level


class BudgetAllocator:
    """Thalamic gate: distributes compute budget across sensors.

    Each timestep:
    1. Sensors report cheap novelty scores (bids)
    2. Allocator distributes budget proportional to bids
    3. Each sensor processes at allocated depth
    4. Unallocated budget flows to internal processing

    Usage::

        allocator = BudgetAllocator()

        # Per frame:
        budget = allocator.allocate([
            SensorBid("visual", visual_novelty),
            SensorBid("auditory", auditory_novelty),
            SensorBid("internal", internal_pressure),
        ])

        if budget.above("visual", LEVEL_STANDARD):
            # Full saccade analysis
        elif budget.above("visual", LEVEL_REDUCED):
            # Single fixation
        else:
            # Scene change check only
    """

    def __init__(
        self,
        total_budget: int = TOTAL_BUDGET,
        min_budget: int = MIN_BUDGET,
    ):
        self.total_budget = total_budget
        self.min_budget = min_budget
        # Phase 2+: learned weights per sensor (default 1.0 = equal)
        self._weights: dict[str, float] = {}
        # Running stats for monitoring
        self._alloc_count = 0
        self._alloc_sums: dict[str, float] = {}
        # Previous frame for visual novelty
        self._prev_frame_small: np.ndarray | None = None

    def allocate(self, bids: list[SensorBid]) -> BudgetAllocation:
        """Distribute budget proportional to novelty bids.

        Each sensor gets at least min_budget. Remaining budget is
        distributed proportional to weighted novelty bids.
        """
        n = len(bids)
        if n == 0:
            return BudgetAllocation()

        # Guarantee minimum per sensor
        reserved = self.min_budget * n
        remaining = max(0, self.total_budget - reserved)

        # Apply learned weights (Phase 2+, default 1.0)
        weighted_bids = []
        for bid in bids:
            w = self._weights.get(bid.sensor, 1.0)
            weighted_bids.append(bid.novelty * w)

        total_weighted = sum(weighted_bids)

        allocations = {}
        if total_weighted < 0.01:
            # All sensors quiet — route remaining budget to internal processing.
            # Biology: when nothing demands attention, the brain thinks.
            for bid in bids:
                if bid.sensor == "internal":
                    allocations[bid.sensor] = self.min_budget + remaining
                else:
                    allocations[bid.sensor] = self.min_budget
        else:
            for bid, wb in zip(bids, weighted_bids):
                share = int(remaining * wb / total_weighted)
                allocations[bid.sensor] = self.min_budget + share

        # Track stats
        self._alloc_count += 1
        for sensor, budget in allocations.items():
            self._alloc_sums[sensor] = self._alloc_sums.get(sensor, 0) + budget

        return BudgetAllocation(allocations=allocations)

    def visual_novelty(self, frame: np.ndarray) -> float:
        """Cheap visual change detection (~1ms).

        Downscales to 64x64 grayscale, computes mean absolute difference
        from previous frame. Returns 0.0 (identical) to 1.0 (scene change).
        """
        import cv2

        small = cv2.resize(frame, (64, 64))
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        small = small.astype(np.float32) / 255.0

        if self._prev_frame_small is None:
            self._prev_frame_small = small
            return 1.0  # first frame is always novel

        delta = float(np.mean(np.abs(small - self._prev_frame_small)))
        self._prev_frame_small = small

        # Map delta to novelty: <0.02 = 0, >0.15 = 1.0, linear between
        novelty = max(0.0, min(1.0, (delta - 0.02) / 0.13))
        return round(novelty, 3)

    def auditory_novelty(
        self,
        has_onset: bool = False,
        speech_surprise: float = 0.0,
        has_speech: bool = False,
    ) -> float:
        """Cheap audio novelty from existing signals (~0ms).

        Combines onset detection, speech prediction surprise,
        and speech presence into a single novelty score.
        """
        score = 0.0
        if has_onset:
            score += 0.4
        if has_speech:
            score += 0.3
        score += speech_surprise * 0.3
        return min(1.0, round(score, 3))

    def internal_pressure(
        self,
        pressure_tracker=None,
    ) -> float:
        """Internal processing demand from dreamstate pressure (~0ms).

        Uses existing DreamstatePressure signals if available.
        """
        if pressure_tracker is None:
            return 0.1  # baseline internal activity

        # Sum normalised pressure from all trackers
        signals = []
        if hasattr(pressure_tracker, 'surprise'):
            signals.append(min(1.0, pressure_tracker.surprise.pressure))
        if hasattr(pressure_tracker, 'cooc_volume'):
            signals.append(min(1.0, pressure_tracker.cooc_volume.pressure))
        if hasattr(pressure_tracker, 'accuracy'):
            signals.append(min(1.0, pressure_tracker.accuracy.pressure))
        if hasattr(pressure_tracker, 'idle'):
            signals.append(min(1.0, pressure_tracker.idle.pressure))

        if not signals:
            return 0.1

        return min(1.0, round(max(signals), 3))

    def reset_visual(self):
        """Reset visual state (e.g., on video change)."""
        self._prev_frame_small = None

    @property
    def stats(self) -> dict:
        """Average allocation stats for monitoring."""
        if self._alloc_count == 0:
            return {}
        return {
            sensor: round(total / self._alloc_count, 1)
            for sensor, total in self._alloc_sums.items()
        }

    def reset_stats(self):
        """Reset allocation tracking stats."""
        self._alloc_count = 0
        self._alloc_sums.clear()
