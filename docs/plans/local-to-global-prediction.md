# Local-to-Global Prediction — System-Level Part-to-Whole Inference

## Concept

Two levels of local-to-global learning:

1. **Encoder level** (training time): JEPA student sees foveal crop, teacher sees wider
   context. The encoder learns to produce embeddings that imply what's nearby.

2. **System level** (observation time): Fovea sees one part, co-occurrence graph predicts
   what other parts should appear nearby, fovea moves to verify, prediction accuracy
   drives learning.

Level 1 is a training technique. Level 2 is a cognitive capability that uses the
co-occurrence graph, spatial edges, structural expectations, and the prediction system.

## System-Level Flow

```
Fovea sees handle curve
  → Encoder: 512-dim embedding (geometric + JEPA)
  → Co-occurrence lookup: "this embedding pattern co-occurs with..."
    → cylinder body (strength 450, spatial relation: NEAR, LEFT)
    → rim circle (strength 380, spatial relation: NEAR, ABOVE)
    → brown surface (strength 500, relation: ON)
  → Sensory predictions generated:
    → "Expect cylinder body to the LEFT within 50px"
    → "Expect rim ABOVE within 30px"
  → Fovea bias: next saccade targets predicted location (LEFT)
  → Observe: straight vertical edge found at predicted location
  → Surprise: LOW (prediction confirmed)
  → Strengthen: handle ↔ body co-occurrence + spatial edge
  → After 3-4 saccades: all parts confirmed → "this is a cup"
```

## Existing Infrastructure

Already built:
- `sensory_prediction.py` — 3-source predictions (structural/temporal/symbolic)
- `surprise.py` — modulates attention based on prediction accuracy
- `cooccurrence.py` — co-occurrence store with Ebbinghaus decay
- `scene_spatial_edges` — allocentric spatial relationships
- `structural_expectations` — "handle is part_of cup" learned patterns
- `attention_controller.py` — fovea direction influenced by prediction

Needs connecting:
- Co-occurrence recall → sensory prediction source
  "This embedding has strong co-occurrences with X, Y, Z → predict X nearby"
- Spatial edges → prediction location
  "X is usually LEFT of this → predict X at (fovea_x - 50, fovea_y)"
- Prediction → fovea bias
  "I predict X to the LEFT → bias next saccade leftward"
- Verification → co-occurrence + spatial edge reinforcement
  "Prediction confirmed → strengthen both the co-occurrence and the spatial edge"

## Connection to Dreamstate

The recombination engine tests these predictions in novel contexts:
- "Handle predicts cup body. What if I place handle in kitchen scene without a cup?
  Does spreading activation still predict cup body? → hypothesis generated"
- Confirmed hypotheses become structural expectations
- This is how the system learns "handles belong to cups" as a general rule,
  not just "this specific handle belongs to this specific cup"

## Connection to Curiosity

Unverified predictions create curiosity:
- "I predicted a rim above this handle but haven't looked yet"
- The attention controller in INSPECT phase biases toward unverified predictions
- This naturally drives systematic object exploration:
  see part → predict other parts → verify each → complete understanding

## Implementation Steps

1. Add co-occurrence-based prediction source to `sensory_prediction.py`
2. Add spatial-edge-based location prediction (where to look next)
3. Wire prediction locations into fovea candidate selection (bias saccades)
4. Wire prediction outcomes into co-occurrence reinforcement
5. Add prediction-driven exploration to INSPECT phase of attention controller
6. Test: show cup, verify system predicts parts and moves fovea to verify
