# Compute Budget Allocator — Thalamic Gating

## The Problem

Processing every sensor at full capacity every timestep is wasteful.
At 2-5 FPS with full visual analysis + audio + intelligence cycles,
we process at 0.4x real-time. Biology solves this with the thalamus —
a gate that routes compute based on novelty, not by skipping sensors.

## Biological Model

The brain has a fixed metabolic budget (~20W). It doesn't "turn off"
the visual cortex when you close your eyes — it reallocates that
compute to internal processing (imagination, memory consolidation,
planning). Key principles:

1. **Fixed budget per timestep** — not infinite, must be allocated
2. **Sensors bid for compute** — based on novelty, not importance
3. **Unused budget flows inward** — dreamstate, practice, imagery
4. **No sensor is ever fully off** — even habituated sensors run at
   minimum (1-5% budget) to detect change
5. **Budget prediction is learned** — a musician gives more to audio,
   a driver more to visual. This is experience-dependent.

## Architecture

```
                    ┌─────────────────────────┐
                    │   Compute Budget Pool    │
                    │   (100 units/timestep)   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Budget Allocator       │
                    │   (thalamic gate)        │
                    │                         │
                    │   Inputs:               │
                    │   - sensor novelty bids │
                    │   - scene context       │
                    │   - learned weights     │
                    │     (future)            │
                    └───┬───┬───┬───┬────────┘
                        │   │   │   │
              ┌─────────┘   │   │   └──────────┐
              ▼             ▼   ▼              ▼
        ┌──────────┐ ┌──────┐ ┌──────┐ ┌────────────┐
        │ Visual   │ │Audio │ │Olfac │ │ Internal   │
        │ Pipeline │ │Pipe  │ │Pipe  │ │ Processing │
        │          │ │      │ │      │ │            │
        │ budget:  │ │budget│ │budget│ │ - dream    │
        │ 5-40     │ │5-40  │ │5-40  │ │ - practice │
        │          │ │      │ │      │ │ - imagery  │
        └──────────┘ └──────┘ └──────┘ └────────────┘
```

## Sensor Novelty Bidding

Each sensor computes a cheap novelty score (0.0 - 1.0) BEFORE any
expensive processing. This is the "bid" for compute budget.

### Visual Novelty (cost: ~1ms)
- Downscale frame to 64x64 grayscale
- Mean absolute difference from previous frame
- Scene change detection (already exists in attention controller)
- Score: 0.0 (identical) to 1.0 (scene change)

### Auditory Novelty (cost: ~1ms)
- RMS energy change from previous audio window
- Onset detection (already exists: audio_has_onset)
- Speech prediction surprise (from SpeechPredictor)
- Score: 0.0 (silence or repetitive) to 1.0 (new speech/sound)

### Olfactory Novelty (future, cost: ~0ms)
- Chemical sensor delta from previous reading
- Score: 0.0 (same smell) to 1.0 (new chemical)

### Internal Pressure (cost: ~0ms)
- DreamstatePressure accumulated signals (already exists)
- Syllable practice queue depth
- Unprocessed consolidation backlog
- Score: 0.0 (nothing pending) to 1.0 (urgent consolidation)

## Budget Allocation Formula

### Phase 1: Fixed Weights (implement now)

```python
TOTAL_BUDGET = 100  # abstract compute units per timestep
MIN_BUDGET = 5      # minimum per sensor (never fully off)

# Each sensor bids with novelty 0-1
bids = {
    "visual": visual_novelty,      # 0.0 - 1.0
    "auditory": auditory_novelty,  # 0.0 - 1.0
    "olfactory": olfactory_novelty,# 0.0 - 1.0
    "internal": internal_pressure, # 0.0 - 1.0
}

# Guarantee minimum allocation
remaining = TOTAL_BUDGET - MIN_BUDGET * len(bids)

# Distribute remaining proportional to bids
total_bids = sum(bids.values()) or 1.0
allocation = {
    sensor: MIN_BUDGET + int(remaining * bid / total_bids)
    for sensor, bid in bids.items()
}
```

### Phase 2: Learned Weights (future)

```python
# Scene context modulates base weights
# e.g., "in kitchen scenes, olfactory gets 2x weight"
weights = budget_predictor.predict_weights(scene_context)

weighted_bids = {
    sensor: bid * weights.get(sensor, 1.0)
    for sensor, bid in bids.items()
}
# Then distribute as above using weighted_bids
```

The budget_predictor would be trained from experience:
- Track which sensor allocations led to learning (new grounding events)
- Reinforce allocations that produced useful co-occurrences
- Weaken allocations that produced no learning signal
- This is essentially attention learning — the system learns WHERE
  to pay attention based on what has been rewarding in the past

## What Budget Controls

Budget doesn't control WHETHER a sensor runs — it controls HOW DEEPLY:

### Visual Pipeline Budget Levels
- **5-15 (minimal)**: Scene change detection only. No saccades, no
  edge analysis, no encoding. Just "did anything change?"
- **15-30 (reduced)**: Single foveal fixation on highest saliency
  point. One crop encoded. No intelligence cycle.
- **30-50 (standard)**: Full saccade pattern (3-5 fixations). Edge
  analysis. Buffer observation. Co-activation edges.
- **50-100 (elevated)**: Full analysis + intelligence cycle +
  structural predictions + embedding prediction.

### Auditory Pipeline Budget Levels
- **5-15 (minimal)**: Environmental novelty check only. No STT,
  no syllable processing.
- **15-30 (reduced)**: TABULA2 voice detection. Sound unit matching.
  No sequence recording.
- **30-50 (standard)**: Full TABULA2 processing. Sound unit
  observation. Sequence recording. Speech prediction.
- **50-100 (elevated)**: Full processing + extended context window +
  cross-modal binding with visual context.

### Internal Processing Budget Levels
- **5-15 (minimal)**: Ebbinghaus decay only (cheap).
- **15-30 (reduced)**: Decay + self-play (10 pairs).
- **30-50 (standard)**: Full dreamstate cycle (all functions).
- **50-100 (elevated)**: Extended dreamstate + syllable practice +
  recombination + mental rotation.

## Implementation

### New File: `budget_allocator.py`

```python
@dataclass
class SensorBid:
    sensor: str       # "visual", "auditory", "olfactory", "internal"
    novelty: float    # 0.0 - 1.0
    cost_ms: float    # how long the novelty check took

@dataclass
class BudgetAllocation:
    visual: int       # 5-100
    auditory: int     # 5-100
    internal: int     # 5-100
    # future: olfactory, tactile, etc.

class BudgetAllocator:
    """Thalamic gate: distributes compute budget across sensors."""

    TOTAL_BUDGET = 100
    MIN_BUDGET = 5

    def allocate(self, bids: list[SensorBid]) -> BudgetAllocation:
        """Distribute budget proportional to novelty bids."""

    def visual_novelty(self, frame, prev_frame) -> float:
        """Cheap visual change detection (~1ms)."""

    def auditory_novelty(self, has_onset, speech_surprise) -> float:
        """Cheap audio change detection (~0ms, uses existing signals)."""

    def internal_pressure(self, dreamstate_pressure) -> float:
        """Internal processing demand (~0ms, reads pressure tracker)."""
```

### Integration in video.py

Replace the per-frame processing depth decisions with budget allocation:

```python
allocator = BudgetAllocator()

for i, frame in enumerate(frames):
    # 1. Cheap novelty bids from each sensor
    v_novelty = allocator.visual_novelty(frame.image, prev_image)
    a_novelty = allocator.auditory_novelty(audio_has_onset, speech_surprise)
    i_pressure = allocator.internal_pressure(sensor_graph._pressure)

    # 2. Allocate budget
    budget = allocator.allocate([
        SensorBid("visual", v_novelty, 1.0),
        SensorBid("auditory", a_novelty, 0.5),
        SensorBid("internal", i_pressure, 0.1),
    ])

    # 3. Visual processing at allocated depth
    if budget.visual >= 30:
        # Full saccade analysis
        visual_ids, analysis = await sg.ingest_frame(frame.image, ...)
    elif budget.visual >= 15:
        # Single fixation only
        visual_ids, analysis = await sg.ingest_frame_minimal(frame.image, ...)
    else:
        # Scene change check only
        visual_ids = []

    # 4. Audio processing at allocated depth
    if budget.auditory >= 30:
        # Full TABULA2 + sequence recording
        ...
    elif budget.auditory >= 15:
        # Voice detection only
        ...

    # 5. Internal processing with remaining budget
    if budget.internal >= 30:
        # Full dreamstate functions
        ...
    elif budget.internal >= 15:
        # Decay + self-play only
        ...
```

### Integration with Existing Systems

- **AttentionController**: Budget allocation replaces the
  run_intelligence flag. EXPLORE/INSPECT phases naturally produce
  high visual novelty bids. MONITOR produces low bids.
- **DreamstatePressure**: Already tracks internal pressure signals.
  Feed directly as internal bid.
- **SpeechPredictor**: Already produces surprise scores. Feed as
  auditory novelty component.
- **FamiliarityGate**: Budget levels replace the binary familiar/novel
  decision. A familiar scene gets budget 5-15, not "skip".

## Phases

### Phase 1 (implement now)
- BudgetAllocator with fixed equal weights
- Visual novelty via cheap pixel delta
- Auditory novelty from existing onset/surprise signals
- Internal pressure from existing DreamstatePressure
- Budget controls intelligence cycle frequency
- Budget controls number of foveal saccades

### Phase 2 (after learning data exists)
- Budget controls depth of each sensor pipeline
- ingest_frame_minimal() for low-budget visual
- Reduced TABULA2 processing for low-budget audio

### Phase 3 (learned weights)
- Scene context → weight prediction
- Train from co-occurrence success signal
- "This scene context rewards auditory attention"
- Per-sensor weight history with Ebbinghaus decay

### Phase 4 (cross-sensor reallocation)
- Freed visual budget flows to active dreamstate
- Eyes-closed → full internal processing mode
- "Daydreaming" — imagery runs on freed visual budget
- Sleep mode — all sensor budgets → internal

## Key Design Principles

1. **No sensor is ever off** — minimum 5% budget always allocated.
   Even habituated sensors must detect sudden change.
2. **Budget is zero-sum** — more visual = less internal. Forces
   efficient allocation, prevents unbounded processing.
3. **Novelty drives allocation** — not importance, not type. A loud
   noise in a quiet room gets budget regardless of sensor type.
4. **Internal processing is a first-class consumer** — dreaming,
   practice, consolidation compete for the same budget as sensors.
5. **Weights are learnable** — Phase 1 uses fixed weights. Phase 3
   learns from experience. The allocator interface stays the same.
6. **Sensor-agnostic** — adding olfactory or tactile sensors is just
   adding another bid to the allocator. No architectural changes.

## Expected Impact

With a static flashcard showing for 10 seconds:
- **Current**: 20-50 frames × 1.7s/frame = 34-85s processing
- **Phase 1**: Frame 1 gets full budget (visual novelty=1.0).
  Frames 2-50 get minimal visual (novelty≈0.02), budget flows to
  audio + internal. Processing drops to ~0.1s/frame for static
  frames. Total: ~7s instead of 85s. ~12x speedup on static content.
- **Overall**: Mixed content (some static, some dynamic) should see
  3-5x speedup, bringing us to 1-2x real-time.
