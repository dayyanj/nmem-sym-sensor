# Demand-Driven Dreamstate — Pressure-Triggered Consolidation

## Problem

The dreamstate currently runs on a fixed schedule — every N videos or at the end of
a batch. This is like sleeping on a timer rather than sleeping when you're tired.
The brain enters consolidation when there's PRESSURE to do so, not on a clock.

## Design Principle

Each dreamstate function should have its own **trigger condition** — a measurable
pressure signal that indicates "this function needs to run." The dreamstate becomes
a collection of independently-triggered operations, each firing when their specific
pressure crosses a threshold.

This also means the dreamstate doesn't need to run all functions every cycle. A cycle
might only run decay (because co-occurrences are piling up) without running recombination
(because there are no unvalidated hypotheses). Or it might run a deep self-play cycle
because prediction accuracy has been dropping, without touching decay at all.

## Pressure Signals

### 1. Surprise Accumulation → Full Consolidation
**Signal**: Rolling average of surprise scores from the intelligence cycle.
**Threshold**: surprise_avg > 0.6 sustained for > 30 seconds.
**Triggers**: Full dreamstate — decay, self-play, concept linking, recombination.
**Why**: High sustained surprise means the internal model is significantly wrong.
Everything needs updating. This is the "I just experienced something that changes
how I understand the world" signal.

```python
class SurpriseAccumulator:
    def __init__(self, window_s=30.0, threshold=0.6):
        self.observations = []  # (timestamp, surprise_value)
        self.window_s = window_s
        self.threshold = threshold

    def observe(self, timestamp: float, surprise: float):
        self.observations.append((timestamp, surprise))
        # Prune old
        cutoff = timestamp - self.window_s
        self.observations = [(t, s) for t, s in self.observations if t > cutoff]

    @property
    def pressure(self) -> float:
        if not self.observations:
            return 0.0
        return sum(s for _, s in self.observations) / len(self.observations)

    @property
    def should_trigger(self) -> bool:
        return self.pressure > self.threshold and len(self.observations) > 10
```

### 2. Co-occurrence Volume → Decay Cycle
**Signal**: Number of new co-occurrences since last decay.
**Threshold**: new_cooccurrences > 500 since last decay cycle.
**Triggers**: Homeostatic decay (Ebbinghaus) only.
**Why**: Lots of new associations means the weak/noisy ones need pruning.
Without decay, the system accumulates noise that drowns out signal.

```python
class CooccurrenceVolumeTracker:
    def __init__(self, threshold=500):
        self.count_since_last_decay = 0
        self.threshold = threshold

    def observe(self):
        self.count_since_last_decay += 1

    def reset(self):
        self.count_since_last_decay = 0

    @property
    def should_trigger(self) -> bool:
        return self.count_since_last_decay > self.threshold
```

### 3. Hypothesis Backlog → Recombination
**Signal**: Number of unvalidated speculative hypotheses (from nmem-sym bridge).
**Threshold**: speculative_hypotheses > 20.
**Triggers**: Recombination + hypothesis evaluation.
**Why**: Too many untested beliefs means the system is generating ideas faster
than it's validating them. Consolidation prioritises testing the most plausible ones.

### 4. Prediction Accuracy Drop → Self-Play
**Signal**: Rolling prediction accuracy from the intelligence cycle.
**Threshold**: accuracy < 0.4 sustained for > 20 seconds (was previously > 0.6).
**Triggers**: Self-play round-trip testing.
**Why**: The system's predictions are failing — its stored associations are wrong
or outdated. Self-play tests each one and weakens the ones that fail the round-trip.

### 5. Scene Novelty → Scene Consolidation
**Signal**: New scene created by chronoception that has > 10 member clusters.
**Threshold**: scene.visit_count == 1 AND scene.member_count > 10.
**Triggers**: Scene-specific consolidation — spatial graph building, cluster
association strengthening.
**Why**: A complex new environment needs integrating into memory. First visit
to a busy scene = lots to process.

### 6. Conflicting Co-occurrences → Competition + Decay
**Signal**: Number of visual nodes with 3+ competing voice units.
**Threshold**: conflicted_nodes > 10.
**Triggers**: Darwinian competition + focused decay on conflicted pairs.
**Why**: Multiple sound units fighting for the same visual node means the system
hasn't resolved which word goes with which shape. Competition selects the strongest,
decay removes the losers.

### 7. MONITOR Phase Idle → Deep Dreamstate
**Signal**: Attention controller has been in MONITOR phase for > 60 seconds.
**Threshold**: monitor_duration > 60s with surprise_rate < 0.1.
**Triggers**: Deep dreamstate — recombination, mental rotation, concept linking,
compound concept formation. The expensive operations that need time.
**Why**: Nothing new is happening. The system is idle. This is the natural time
to consolidate — like daydreaming or falling asleep when bored. Use the idle
compute for deep processing.

This is the most important trigger because it naturally balances observation
and consolidation. When there's lots to see → EXPLORE/INSPECT → no dreamstate.
When things are quiet → MONITOR → dream.

```python
class IdleTracker:
    def __init__(self, idle_threshold_s=60.0, surprise_threshold=0.1):
        self.monitor_entered_at = None
        self.idle_threshold_s = idle_threshold_s
        self.surprise_threshold = surprise_threshold

    def on_phase_change(self, phase: str, timestamp: float):
        if phase == "monitor":
            self.monitor_entered_at = timestamp
        else:
            self.monitor_entered_at = None

    @property
    def idle_duration(self) -> float:
        if self.monitor_entered_at is None:
            return 0.0
        return time.monotonic() - self.monitor_entered_at

    @property
    def should_trigger_deep(self) -> bool:
        return self.idle_duration > self.idle_threshold_s
```

### 8. Curiosity Pressure → Hypothesis-Seeking Behaviour
**Signal**: High-plausibility unvalidated hypotheses that match current scene context.
**Threshold**: plausible_hypothesis.plausibility > 0.5 AND matches_current_scene.
**Triggers**: NOT dreamstate — triggers attention bias instead. The fovea
biases toward objects/regions that could validate the hypothesis.
**Why**: This is awake-state curiosity, not sleep-state consolidation. But it's
the bridge between dreamstate (which generated the hypothesis) and observation
(which validates it). Listed here because it's part of the demand-driven system.

## Integration Architecture

```python
class DreamstatePressure:
    """Tracks all pressure signals and determines what needs running."""

    def __init__(self):
        self.surprise = SurpriseAccumulator(window_s=30, threshold=0.6)
        self.cooc_volume = CooccurrenceVolumeTracker(threshold=500)
        self.prediction_accuracy = AccuracyTracker(window_s=20, threshold=0.4)
        self.idle = IdleTracker(idle_threshold_s=60)
        self.hypothesis_backlog = 0
        self.conflicted_nodes = 0

    def what_needs_running(self) -> list[str]:
        """Returns list of dreamstate functions that should run now."""
        needs = []

        if self.surprise.should_trigger:
            needs.extend(["decay", "self_play", "concept_linking", "recombination"])

        if self.cooc_volume.should_trigger:
            needs.append("decay")

        if self.prediction_accuracy.should_trigger:
            needs.append("self_play")

        if self.hypothesis_backlog > 20:
            needs.append("recombination")

        if self.conflicted_nodes > 10:
            needs.extend(["competition", "decay"])

        if self.idle.should_trigger_deep:
            needs.extend(["recombination", "mental_rotation", "concept_linking",
                          "compound_formation"])

        return list(set(needs))  # deduplicate

    def any_pressure(self) -> bool:
        return len(self.what_needs_running()) > 0
```

## Connection to Attention Controller

The attention controller's phase transitions naturally feed the pressure system:

- EXPLORE → high surprise → surprise pressure builds → triggers consolidation
  at next idle period
- INSPECT → moderate surprise → predictions tested → accuracy tracked
- MONITOR → low surprise, idle → deep dreamstate triggered after 60s
- Scene change → resets idle timer, new scene novelty pressure

```
Attention Phase    Pressure Effect           Dreamstate
──────────────     ─────────────────         ──────────
EXPLORE            surprise ↑, coocs ↑       deferred (too busy)
INSPECT            accuracy tracked           deferred (still observing)
MONITOR (early)    idle timer starts          not yet
MONITOR (60s+)     idle threshold crossed     DEEP DREAMSTATE
Scene change       idle reset, novelty ↑      scene consolidation
```

The dreamstate never interrupts active observation. It only runs when the
system is idle. This is biologically correct — you can't dream while you're
actively perceiving. The pressure builds during observation and releases
during idle periods.

## Connection to nmem-sym Temporal Pressures

nmem-sym's `temporal.py` already has 4 pressure signals:
1. Event silence (nothing happening)
2. Prediction deadlines (answers needed)
3. Knowledge staleness (information aging)
4. Processing backlog (too much queued)

The sensory pressure system mirrors this but for perceptual processing:
1. Surprise accumulation (too many prediction failures)
2. Co-occurrence volume (too many associations to prune)
3. Hypothesis backlog (too many speculative beliefs)
4. Idle time (nothing to observe)

Both systems should share a common pressure interface so that:
- Sensory pressure can escalate to nmem-sym ("I need help reasoning about this")
- nmem-sym pressure can direct sensory attention ("go look for X to validate Y")

## Implementation Order

1. Implement `DreamstatePressure` class with all signal trackers
2. Wire `SurpriseAccumulator` to intelligence cycle output
3. Wire `CooccurrenceVolumeTracker` to `cooc_store.observe()`
4. Wire `IdleTracker` to attention controller phase changes
5. Replace fixed-schedule dreamstate calls with `pressure.what_needs_running()`
6. Implement selective dreamstate — only run functions that are needed
7. Add pressure stats to dashboard (visualise what's building up)
8. Test: verify dreamstate fires on surprise spikes, not on timer
9. Test: verify deep dreamstate fires during idle, not during active observation
10. Wire to nmem-sym temporal pressure system via bridge

## Success Criteria

- Dreamstate never interrupts active EXPLORE/INSPECT phases
- Dreamstate fires within 10s of MONITOR phase entering idle
- High-surprise observation sequences trigger consolidation at next idle
- Quiet periods (no new content) naturally produce deep consolidation
- Dashboard shows pressure building and releasing in real-time
