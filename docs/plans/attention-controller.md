# Attention Controller — Knowledge-Driven Phase Transitions

## The Problem

The familiarity gate used static thresholds (frame similarity > 0.95 → reduce processing). This failed because:
1. Grayscale embeddings made everything look the same
2. Even color embeddings can't differentiate within-scene variations
3. A fixed threshold doesn't account for scene complexity (triangle = 2s to explore, busy street = 30s)
4. First visit vs return visit need completely different behaviour
5. Moving objects should keep exploration alive regardless of elapsed time

## The Solution: Three-Phase Attention State Machine

Phase transitions are driven by **knowledge state**, not time. You move from exploration to inspection when you've learned enough, not when a timer fires.

```
                    ┌─────────────────────┐
                    │                     │
                    ▼                     │ scene change OR
┌──────────────────────────┐              │ major surprise
│   EXPLORE                │              │
│   "What's here?"         │──────────────┘
│                          │
│   Full saccades          │    coverage high AND
│   Edge-following         │    predictions accurate AND
│   Build scene model      │    no new candidates
│   All visual processing  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   INSPECT                │
│   "What's interesting?"  │
│                          │──── surprise on specific
│   Reduced saccades       │     element → EXPLORE that
│   Focus on unexamined    │     element only (not whole
│   Detail within shapes   │     scene reset)
│   Adjective learning     │
└────────────┬─────────────┘
             │ everything examined,
             │ predictions stable
             ▼
┌──────────────────────────┐
│   MONITOR                │
│   "Has anything changed?"│
│                          │──── surprise fires →
│   Minimal saccades       │     INSPECT the changed
│   Prediction-only        │     element (not full
│   Deviation detection    │     scene EXPLORE)
│   Low compute            │
└──────────────────────────┘
```

## Phase Transition Signals

All signals already exist in the system — we just need to consume them.

### EXPLORE → INSPECT
Transition when the system has "gotten a handle on the scene":
- `prediction_accuracy > 0.7` — the predictor is anticipating frames correctly
- `candidate_coverage > 0.8` — fovea has visited 80%+ of salient regions
- `new_candidate_rate < 1 per 2s` — not finding new things to look at
- ALL of these must hold simultaneously

A simple triangle might hit this in 1-2 seconds. A busy street scene with moving cars might never leave EXPLORE because new candidates keep appearing (movement resets the counter).

### INSPECT → MONITOR
Transition when details have been examined:
- `surprise_rate < 0.1` for sustained period — nothing unexpected
- `unexamined_detail_ratio < 0.2` — most detail regions inspected
- `prediction_accuracy > 0.85` — confident about this scene

### MONITOR → EXPLORE (scene change)
Full reset when the world changes:
- Scene change detected (chronoception) → new scene, fresh EXPLORE
- Major surprise event (surprise > 0.5) → something significant happened

### MONITOR → INSPECT (local change)
Partial re-engagement when something specific changes:
- A single element moves or changes → INSPECT that element only
- The rest of the scene stays in MONITOR (cached model)
- This is the key insight: you don't re-explore the whole room when one thing moves

## First Visit vs Return Visit

### First visit to a scene
```
EXPLORE (full) → INSPECT → MONITOR
```
Exploration runs until coverage is achieved. No shortcuts.

### Return visit (scene recognised by chronoception)
```
EXPLORE (abbreviated) → INSPECT (verify) → MONITOR
```
Scene snapshot loaded → cached model activated → brief verification pass
("is everything where I expect it?") → jump to MONITOR.

The verification pass is a fast EXPLORE with the expectation that everything matches. If it does → straight to MONITOR. If something doesn't match → focused INSPECT on the discrepancy.

## Per-Phase Processing Budget

### EXPLORE
- Saccades: `max_saccades` (8)
- Edge following: `max_edge_follow` (6)
- Intelligence: ON (generating predictions, building model)
- Sound binding: ON (learning what things sound like)
- Promotion threshold: normal (3 observations)

### INSPECT
- Saccades: `reduced_saccades` (4-6) — focused, not broad
- Edge following: reduced (2-3) — already traced outlines
- Intelligence: ON (refining predictions)
- Sound binding: ON (learning details)
- Fovea behaviour: skip habituated candidates, seek unexamined detail within known shapes
- This is Phase 2 of the fovea plan: "I know this shape, what's different about THIS one?"

### MONITOR
- Saccades: `min_saccades` (2-3) — just checking
- Edge following: 0 — not tracing anything
- Intelligence: periodic (every 3rd frame) — just verifying predictions
- Sound binding: ON for new sounds only — familiar sounds skip
- Frame subsampling: process every 2nd-3rd frame when scene is completely stable

## Implementation

### AttentionPhase enum
```python
class AttentionPhase(Enum):
    EXPLORE = "explore"    # Full processing, building scene model
    INSPECT = "inspect"    # Detail-focused, reduced broad scanning
    MONITOR = "monitor"    # Minimal, deviation detection only
```

### AttentionController class
```python
class AttentionController:
    """Knowledge-driven attention state machine.

    Replaces the threshold-based familiarity gate with a three-phase
    system driven by prediction accuracy, foveal coverage, and surprise.
    """

    phase: AttentionPhase = EXPLORE
    phase_entered_at: float  # monotonic time

    # Transition signals (updated each frame)
    prediction_accuracy: float  # from EmbeddingPredictor.running_error
    candidate_coverage: float   # from CandidateTracker
    new_candidate_rate: float   # new candidates per second
    surprise_rate: float        # from surprise.py

    # Scene knowledge
    is_return_visit: bool       # from chronoception scene recognition
    scene_model_loaded: bool    # cached model available

    def update(self, signals: dict) -> AttentionBudget:
        """Called each frame. Evaluate transition conditions, return budget."""

    def _check_transitions(self):
        """State machine transition logic."""

    def _on_surprise(self, surprise_level: float, element_id: int | None):
        """Handle surprise — may trigger phase transition."""
```

### Integration points
- `api.py`: AttentionController replaces SceneMemory.compute_familiarity()
- `visual_attention.py`: CandidateTracker reports coverage + new candidate rate
- `embedding_predictor.py`: provides prediction_accuracy (already exists)
- `surprise.py`: provides surprise_rate (already exists)
- `chronoception.py`: provides is_return_visit + scene model

### Signals we need to add
1. **candidate_coverage** — CandidateTracker needs to report what fraction of the saliency map has been visited. Simple: track visited (x,y) regions, compare to total salient area.
2. **new_candidate_rate** — CandidateTracker already creates new candidates. Track the rate (new per second).
3. **unexamined_detail_ratio** — during INSPECT, what fraction of the attended region has been deeply analysed. Needs a "detail_examined" flag on candidates.

## Key Design Principles

1. **Knowledge drives transitions, not time.** A triangle enters INSPECT in 2 seconds. A busy street stays in EXPLORE for 30 seconds. Both are correct.
2. **Moving objects sustain EXPLORE.** New candidates keep appearing → exploration continues regardless of elapsed time.
3. **Return visits skip EXPLORE.** Scene recognised → load model → verify briefly → MONITOR.
4. **Local changes don't reset everything.** One thing moved → INSPECT that thing. Everything else stays cached.
5. **Sound binding always runs.** Even in MONITOR, new sounds need to be processed. Familiar sounds can be skipped.
6. **Frame rate independent.** All signals are rates (per second) or ratios, not counts.

## Files to modify
- **MODIFY** `familiarity.py` → replace SceneMemory.compute_familiarity() with AttentionController.update()
- **MODIFY** `visual_attention.py` → CandidateTracker reports coverage + new candidate rate
- **MODIFY** `api.py` → wire AttentionController, feed it signals from predictor + tracker + surprise
- **MODIFY** `chronoception.py` → feed is_return_visit to controller
- **MODIFY** `config.py` → transition thresholds (configurable, eventually adaptive)

## Verification
1. **Triangle video**: should enter INSPECT within 1-3 seconds (simple scene, fast coverage)
2. **YouTube video**: should stay in EXPLORE longer (complex, moving elements)
3. **Same video on re-watch**: should skip to abbreviated EXPLORE → MONITOR quickly
4. **Scene change mid-video**: should reset to EXPLORE on the new scene
5. **All phases produce visual nodes**: no phase should produce 0 visual (the current bug)
