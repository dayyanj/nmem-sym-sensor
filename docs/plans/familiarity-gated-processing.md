# Familiarity-Gated Processing

## The Problem We're Solving

The system currently spends identical processing effort on familiar vs novel content. Watch 6 of the same video burns the same cycles as watch 1. A well-known shape gets the same foveal tracing as a never-seen shape. This is biologically implausible and computationally wasteful.

## The Human Model

Consider sitting in a familiar cafe typing on a laptop:

- **Visual periphery** fills in chairs and tables without active processing — you know where they are from prior visits. Your brain maintains a spatial model and runs on it.
- **Auditory scene** is parsed without looking: music to the right (radio — confirmed by prior visual), voices to the left (cafe owners — recognised by voice alone, no visual needed).
- **Multi-modal confirmation** strengthens the model: you hear the radio AND you've seen it there → high confidence.
- **Surprise drives attention**: if the furniture moved, you'd notice instantly — not because you're actively scanning, but because the prediction failed.
- **Novel elements get full processing**: a new person enters → attention shifts, full visual analysis. Everything else runs on the cached model.

This is multiple systems working simultaneously: spatial model (persistent map), auditory source localisation + identity, predictive coding (run the model, update on deviation), and multi-modal confirmation.

## Design Principle

**Familiarity reduces processing. Novelty demands it.**

The brain's energy flows from the familiar toward the novel. Processing budget, attention allocation, memory consolidation, dreamstate replay — all biased toward what's new and unexpected. The familiar runs on cached models with minimal verification.

## Architecture: Three-Level Filter

The three levels are not sequential gates — they are **parallel signals** that combine into an attention budget for each frame.

```
Input (frame/audio)
  │
  ├──→ Level 1: Frame Gate ──→ "Have I seen this scene before?"
  │    Coarse embedding vs scene memory + EmbeddingPredictor.running_error
  │    High familiarity → skip frames, reduce saccades
  │
  ├──→ Level 2: Object Gate ──→ "Do I know what this thing is?"
  │    Candidate embedding vs grounded cluster centroids
  │    Known object → habituate faster, trace outline less, inspect details more
  │    Unknown object → full exploration, edge-following, maximum dwell
  │
  └──→ Level 3: Scene Model ──→ "Has anything changed in my world?"
       Persistent spatial map: (position, object_id, last_confirmed)
       Deviation-only processing: only update what moved/appeared/disappeared
       This is the "I know this cafe" level
  │
  ▼
  AttentionBudget (combined from all three signals)
  │
  ▼
  Existing pipeline (with gated compute)
```

## Familiarity State

Per-frame assessment that feeds into the attention budget:

```python
@dataclass
class FamiliarityState:
    frame_familiarity: float      # 0.0 = completely novel, 1.0 = identical to memory
    scene_stability: float        # rolling window of frame-to-frame similarity
    prediction_accuracy: float    # EmbeddingPredictor.running_error inverted
    familiar_object_ratio: float  # fraction of candidates matching known clusters
```

**Key insight**: `EmbeddingPredictor.running_error` is already computed for free. It IS a familiarity signal — low error means the system predicted this frame accurately, meaning it's familiar. No new computation needed for the primary signal.

## Attention Budget

The combined output that controls per-frame compute allocation:

```python
@dataclass
class AttentionBudget:
    max_saccades: int          # 1-8 (fewer for familiar scenes)
    edge_follow_steps: int     # 0-6 (0 for myelinated objects)
    periphery_enabled: bool    # False for familiar scenes
    run_intelligence: bool     # skip prediction cycle for stable scenes
    analyze_full_frame: bool   # True only for novel scenes
    consolidation_fast_path: bool  # skip full consolidation for familiar
```

## Implementation Phases

### Phase 1: Frame-Level Gate ("Have I seen this before?")

The coarsest gate. Before deep analysis, compare the frame's signature against recent scene memory.

**SceneMemory**: Rolling buffer of coarse frame embeddings (32x32 grayscale → 1024-dim vector). Computed from grayscale already available in `visual_attention.py` — effectively free.

**Gating logic**:
- `frame_familiarity > 0.95` → skip frame entirely (audio still processes)
- `frame_familiarity 0.80-0.95` → reduce saccades to 50%, disable periphery
- `frame_familiarity < 0.80` → full processing, all saccades, full periphery

**Dynamic frame subsampling**: When `scene_stability > 0.9` for 5+ consecutive frames, drop to every 2nd frame. At `> 0.95` for 10+ frames, every 3rd. Audio always runs — sounds need continuous monitoring even in familiar visual scenes. A phone ringing in a familiar room still needs to be heard.

**Integration**: Modify `FovealAttention.process_frame()` to accept an `AttentionBudget`. Store `SceneMemory` on `SensorGraph`, compute familiarity before calling the fovea.

**Validation**: Same video 3x — third watch should process 30-50% fewer frames with no loss in recognition quality (clusters already exist).

### Phase 2: Object-Level Gate ("I know what this is")

Per-candidate familiarity within a frame. This is the transition from "What is this?" to "What's different about THIS one?"

**Cluster centroid cache**: Load all grounded cluster centroids at `connect()` time. Refresh during `consolidate()`. Dict of `{cluster_id: (centroid, label, confidence)}`, typically < 100 entries. No per-frame DB queries.

**Familiar candidate behaviour**:
- Habituate 3x faster (the fovea doesn't need to dwell on known objects)
- Edge-follow 2 steps instead of 6 (partial contour is enough — DeepFovea insight: the brain fills in from sparse signals + learned statistics)
- Skip full edge analysis — use cached cluster embedding as the primitive
- Shift attention to **what's novel within the object**: colour difference, size difference, rotation. This is Phase 2 of foveal development.

**Novel candidate behaviour**:
- Full edge-following, maximum dwell time, all saccade points
- Lower habituation threshold (keep looking longer)
- Priority for sound-visual binding (novel objects get more binding opportunities)

**This connects to the existing recognition engine**: if `RecognitionEngine` has a cache hit for the candidate's cluster, the object is "known" and gets reduced processing.

### Phase 3: Intelligence Loop Gating

When the scene is familiar, most predictions will be correct. Generating and verifying predictions for a stable scene is pure waste.

- When `frame_familiarity > 0.85` AND no novel objects detected, skip prediction generation entirely
- Still run the cycle periodically (every 5th familiar frame) to catch slow drifts
- Always observe the embedding (keep `EmbeddingPredictor` EMA current even on skipped frames)
- Surprise signal from a skipped check = instant full processing on next frame

### Phase 4: Scene Model ("I know this cafe")

The most complex level. Deferred until Levels 1-2 prove effective.

**Persistent spatial map**: Hash map of `(normalised_position, cluster_id, last_confirmed)` tuples. When a frame arrives, each detected object's position + cluster is compared against the map.

- Objects in expected positions = no surprise, skip processing
- Objects missing = surprise signal → search/investigate
- Objects in wrong positions = surprise signal → full analysis
- New objects not in map = high surprise → maximum attention

**Cross-session persistence**: Scene snapshots stored in DB. Re-entering a familiar scene (same video, or returning to same webcam view) activates the cached model instantly — no re-learning needed.

**Scene change detection** (already exists in `extract_frames()`) resets the model. Camera cut = new scene = fresh start.

```sql
CREATE TABLE scene_snapshots (
    id SERIAL PRIMARY KEY,
    scene_hash BIGINT,
    spatial_map JSONB,              -- {cluster_id: {x, y, size, last_seen}}
    coarse_embedding vector(1024),
    observation_count INT DEFAULT 1,
    first_seen_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ
);
```

## Connection to Dreamstate Decay + Myelination

### Homeostatic Decay (IMPLEMENTED — `sensory_dreamstate.py`)

Each dreamstate cycle, ALL unmyelinated sound-visual co-occurrences decay by a small factor (×0.97). Strong recent ones barely notice. Old unreinforced ones fade. Associations don't die — they go dormant (count → 0). Fresh evidence reactivates them.

Three mechanisms:
1. **Global decay** — multiply all unmyelinated counts by 0.97
2. **Recency penalty** — not reinforced in last 24h → additional ×0.94
3. **Archive, not delete** — dormant associations (count=0) remain in DB, can be reactivated

### Myelination

Associations that survive enough dreamstate challenges AND self-play confirmations get locked in:
- `dreamstate_challenges >= 10` AND `self_play_confirmations >= 5` AND `count >= 50`
- Myelinated = exempt from decay, always in recognition cache
- Failed predictions from myelinated associations → de-myelinate (force re-examination)

### Dreamstate Recombination (PLANNED — Phase 3 of dreamstate evolution)

Inspired by actual dream behaviour: the brain takes recent unprocessed events, strips them from their original context, and recombines them with other memories to create coherent test scenarios. Not random — it selects emotionally salient/unprocessed material and borrows context from other memories to fill gaps.

**Three purposes:**
1. **Generalise** — extract abstract patterns from specific events ("this type of situation" not "this exact situation")
2. **Simulate** — run patterns through different scene contexts ("what if I saw this triangle in the kitchen?")
3. **Evaluate** — test if associations hold in novel contexts (universal vs context-specific)

**Algorithm:**
1. Pick a recently observed cluster (high salience, low dreamstate replay count)
2. Activate a DIFFERENT scene context (not the one it was observed in)
3. Place the cluster in that scene's spatial graph via spreading activation
4. Check coherence: do the activated neighbours make sense together?
5. Coherent → strengthen the generalised pattern (this concept is universal)
6. Conflict → flag the association as context-specific (only works in original scene)

**What it discovers:**
- "Triangle" activates coherently in any scene → universal concept
- "The red mug on my desk" only activates in the office scene → context-specific
- A sound-visual binding that works in one scene but conflicts in another → weak/noisy association

**Builds on:** Scene model (alternative contexts), active set (clusters to recombine), spreading activation (coherence testing), allocentric spatial graph (plausibility checking)

**Archive-not-delete is essential here:** Dormant associations can be reactivated during recombination dreams. An association that was archived because it was weak in the original context might be valid in a different context — the dream discovers this.

### Familiarity × Dreamstate

- **Familiar objects get less dream replay** — bias replay toward novel experiences (the brain consolidates new learning more than routine)
- **Unfamiliar-context associations decay faster** — if an association was only observed under high surprise (unstable scene), it's less reliable
- **Consistently familiar clusters get groundedness boost** but lower replay priority — the system is confident about them, doesn't need to re-examine

## The Developmental Arc

This system mirrors stages of infant visual development:

1. **0-3 months** (current): Everything is novel. Full processing on every frame. Fovea follows saliency and edges. Sound binds to whatever is attended. ← We are here.

2. **3-6 months** (Phase 1+2): Familiarity emerges. Known objects processed faster. The fovea transitions from "What is this?" (outline tracing) to "What's different?" (detail inspection). Habituation drives novelty-seeking within familiar shapes.

3. **6-12 months** (Phase 3+4): Scene models form. The infant knows their room, their toys, their parents' faces. Surprise fires on spatial changes. Joint attention begins (learning to follow pointing/gaze to the object being named). Processing budget heavily biased toward the novel.

4. **12+ months** (future): Language-guided attention. Words direct the fovea. "Look at the triangle" → fovea seeks triangular shapes in the scene. The system queries its own model ("where have I seen triangles?") rather than bottom-up scanning.

## Key Design Constraints

- Must work for both **pre-recorded video AND live camera** — familiarity gate uses only past state, never future frames
- Must be **computationally cheap** — the whole point is spending LESS compute on familiar content
- **Integrates with existing systems**, doesn't duplicate: prediction/surprise, habituation, cluster matching
- **Audio always processes** — sounds need continuous monitoring even in familiar visual scenes
- Cluster cache refreshed at consolidation, **not per-frame DB queries**
- **Gradual transitions** — familiarity is a continuum (0-1), not a binary switch. Processing scales smoothly from full (novel) to minimal (myelinated).
