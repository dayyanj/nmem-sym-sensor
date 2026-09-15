# Intelligence Loops

The predict-observe-compare-update cycle that makes the system learn, not just record. Without this, the system is a filing cabinet. With it, the system genuinely learns from experience.

## Modules

| Module | Purpose |
|--------|---------|
| `temporal.py` | Object tracking: embedding + spatial matching across frames, velocity estimation, disappearance detection. Provides temporal predictions |
| `sensory_prediction.py` | Three-source prediction engine (structural, temporal, symbolic) + verification + LTP/LTD feedback on confirmed/refuted predictions |
| `surprise.py` | Continuous surprise signal (0-1): modulates attention dwell, memory thresholds, decomposition triggers, drive pressure |
| `sensory_bridge.py` | Bidirectional symbol-sensory bridge: symbol -> sensory expectations (top-down), sensory -> symbol events (bottom-up) |
| `sensory_dreamstate.py` | Offline consolidation: prediction expiry, expectation decay, binding LTP/LTD, cluster dedup, salience decay |

## The Cognitive Loop

```
       PREDICT
       "I expect to see X"
           |
           v
       OBSERVE
       sensory input (frame/audio)
           |
           v
       COMPARE
       prediction vs reality
       surprise = mismatch
           |
           v
       UPDATE
       confirmed -> LTP (strengthen path)
       refuted -> LTD (weaken path)
       novel -> create new tracking
```

## Three Prediction Sources

1. **Structural**: composition.py says "eye" -> expect "nose" and "mouth" nearby
2. **Temporal**: object tracker says cup was at (200,300) -> predict at (205,300) next frame
3. **Symbolic**: nmem-sym activates "cup" -> predict cylinder + handle visual cluster

## Surprise Effects

| Surprise Level | Attention | Memory | Decomposition | Drives |
|---------------|-----------|--------|---------------|--------|
| HIGH (>0.7) | Dwell 2x longer | Promotion threshold halved | Trigger deeper analysis | Uncertainty +0.15 |
| MODERATE (0.4-0.7) | Dwell 1.3x | Threshold 0.8x | Normal | Uncertainty +0.05 |
| LOW (<0.2) | Move on quickly | Threshold 1.2x (don't waste memory) | Skip | Uncertainty -0.05 |

## Links to Other Components

- **api.py** (Public API): `run_intelligence_cycle()` orchestrates the full loop per frame
- **visual.py / audio.py** (Perception): observations come from sensory analysis
- **graph.py** (Sensory Graph): active nodes drive structural predictions
- **composition.py** (Composition): structural expectations generate sibling predictions
- **sensory_bridge.py** (Bridge): surprise events feed nmem-sym drive pressures
- **consolidation.py** (Consolidation): dreamstate runs as final consolidation step

## Database Tables

| Table | Purpose |
|-------|---------|
| `sensory_predictions` | Stored predictions with status (pending/confirmed/refuted/expired) |

## Key Config

```
INTELLIGENCE_LOOPS_ENABLED = 0|1
SURPRISE_HIGH = 0.7
SURPRISE_LOW = 0.2
DREAMSTATE_EVERY_VIDEOS = 5
DREAMSTATE_MAX_DURATION_S = 30.0
```
