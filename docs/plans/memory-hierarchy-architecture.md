# Sensory Memory Hierarchy & Scene-Gated Activation Architecture

## Context

The nmem-sym-sensor system learns visual and auditory concepts through observation. As knowledge grows (potentially billions of nodes over months/years of live camera input), we need an activation architecture that scales — not by making search faster, but by searching less.

The brain doesn't load every memory when you walk into a room; it activates a context-dependent subset. This document designs the memory hierarchy for sensory processing, integrating with existing nmem (cognitive) and nmem-sym (symbolic) tiers, and introduces scene-gated activation as the scaling solution.

**Triggered by:** ClusterCache currently loads ALL grounded clusters (125 today, but unbounded). At billions of nodes, even pgvector HNSW queries become expensive if scanning broadly. The activation architecture IS the scaling architecture.

## Existing Memory Hierarchy (What We Have)

### nmem (Cognitive Memory — 6 tiers)
- **Working Memory** — in-process, ~20 slots, session-scoped
- **Journal** — 30-day buffer, importance-scored, auto-promotes at importance >= 7 or access >= 5
- **LTM** — permanent, versioned, salience-decayed (multiplicative decay over days)
- **Shared** — cross-agent canonical facts
- **Entity** — per-entity dossiers
- **Policy** — governance rules, read-only

Key patterns: budget-aware briefing (loads relevant subset within token budget), recency-weighted search (half-life decay), recognition signals (KNOWN/FAMILIAR/UNCERTAIN), parallel hybrid search (vector + FTS).

### nmem-sym (Symbolic Graph)
- Symbol nodes with spreading activation (seed → threshold-gated traversal → activated subgraph)
- Two modes: QUERY (fast, 50 nodes, 100ms) and EXPLORATION (deep, 100 nodes, dreamstate)
- Myelination: ltp_score >= 0.8 → fast path, skip threshold check
- LTP/LTD Hebbian plasticity, co-activation bonding (fired >= 3 times → shortcut edge)

### nmem-sym-sensor (Sensory — current)
- **Iconic Buffer** — 5s decay, 256 capacity, in-memory
- **Short-Term** — 1h decay, DB-backed, promotes at 10 co-occurrences
- **Long-Term** — permanent, clusters form at 5+ members, coherence threshold 0.5
- **Grounded Concepts** — clusters linked to nmem-sym symbols

## Proposed: Three-Tier Runtime Activation

Like the brain: long-term storage is vast, working memory is tiny, context determines what's loaded. Like a transformer: attention window (active), KV cache (primed), model weights (everything).

```
┌─────────────────────────────────────────────────────┐
│  Tier 1: ACTIVE SET (in-process, nanoseconds)       │
│  50-200 clusters in dict or FAISS IndexFlatIP       │
│  Scene-gated: loaded when entering a context        │
│  Hot embeddings with decay from prior scene         │
│  = Transformer's attention window                   │
│  = nmem's Working Memory                            │
│  = nmem-sym's QUERY activation (50 nodes, 100ms)    │
│  Memory: ~400KB for 200 × 512-dim vectors           │
└──────────────────┬──────────────────────────────────┘
                   │ miss → query
┌──────────────────▼──────────────────────────────────┐
│  Tier 2: PRIMED SET (pgvector HNSW, milliseconds)   │
│  10K-100K clusters from related scenes              │
│  Pre-warmed by scene associations + recency         │
│  = Transformer's KV cache                           │
│  = nmem's Journal + recent LTM                      │
│  = nmem-sym's EXPLORATION activation (100 nodes)    │
└──────────────────┬──────────────────────────────────┘
                   │ miss → deep search (dreamstate only)
┌──────────────────▼──────────────────────────────────┐
│  Tier 3: DEEP STORAGE (cold DB, seconds)            │
│  Billions of nodes, never queried in real-time      │
│  Only during dreamstate or genuine novelty          │
│  = Transformer's model weights                      │
│  = nmem's LTM + Shared Knowledge                    │
│  = nmem-sym's full graph                            │
└─────────────────────────────────────────────────────┘
```

### Transformer Context Analogy

| Transformer | Sensory System | nmem Equivalent |
|-------------|---------------|-----------------|
| Attention window | Active Set (Tier 1) | Working Memory |
| KV cache | Primed Set (Tier 2) | Journal + recent LTM |
| Model weights | Deep Storage (Tier 3) | Full LTM + Shared |
| Positional encoding | Temporal decay | Recency-weighted search |
| Attention scores | Activation levels | Relevance ranking |
| Context window limit | Budget constraint | Token budget in briefing |

### Scene Model as Activation Context

A scene is a retrieval cue that primes what's relevant — not just "I've been here" but "here's what I expect to find."

```python
class SceneContext:
    scene_id: int
    scene_embedding: np.ndarray       # coarse 32x32 signature
    associated_clusters: set[int]     # clusters seen in this scene
    spatial_map: dict                 # {cluster_id: (x, y, size)}
    visit_count: int                  # familiarity level
    last_visited: datetime
```

### Scene Transition: Decay Bridge

When moving bedroom → living room, don't dump bedroom clusters instantly. Fade them with exponential decay. If something from the bedroom is relevant (you carried an object), the cluster stays active because it's being reinforced. Shared objects (doors, light switches) stay hot because they appear in both scenes.

```
Enter bedroom:
  → Load bedroom clusters into Tier 1 (activation = 1.0)
  → Spatial map loaded → deviation-only attention

Move to living room:
  → Bedroom clusters begin decaying (×0.9 per frame)
  → Living room clusters load at activation = 1.0
  → Shared clusters (door, walls) stay active (reinforced by both)
  → After ~20 frames, bedroom-only clusters fall below threshold → demote to Tier 2

Return to bedroom:
  → Scene match → instant reload from Tier 2/3
  → Skip re-learning phase, just verify spatial map
```

### Active Set Implementation

```python
class ActiveSet:
    """Tier 1: Scene-gated clusters with temporal decay."""

    _clusters: dict[int, ActiveCluster]
    # Each ActiveCluster has:
    #   activation: float (0-1, decays each frame)
    #   decaying: bool (True when scene deactivated)
    #   centroid: np.ndarray (512-dim for matching)
    #   label: str

    def activate_scene(self, scene: SceneContext):
        """Load scene's clusters, boost activation to 1.0"""

    def deactivate_scene(self, scene: SceneContext):
        """Begin decay for this scene's clusters"""

    def tick(self, decay_rate=0.9):
        """Per-frame: decay flagged clusters, remove below threshold"""

    def reinforce(self, cluster_id):
        """Object seen → boost to 1.0, cancel decay"""

    def match(self, embedding) -> (cluster_id, label, similarity):
        """Fast lookup against active clusters only"""
```

### FAISS Strategy

- **< 500 clusters**: dict loop (current ClusterCache), no FAISS needed
- **500-10K clusters**: FAISS IndexFlatIP, rebuild on scene transitions (~1ms for 10K vectors)
- **10K+**: Tiered — FAISS for Tier 1, pgvector HNSW for Tier 2 fallback
- FAISS can't add/remove dynamically — rebuild index when active set changes
- Memory: 2KB per 512-dim vector, so 10K clusters = 20MB (trivial)
- Hot-swap pattern: pre-build per-scene indexes, atomic pointer swap

### DB Schema

```sql
CREATE TABLE scene_snapshots (
    id SERIAL PRIMARY KEY,
    scene_embedding vector(1024),    -- coarse 32x32 signature
    spatial_map JSONB,               -- {cluster_id: {x, y, size, confidence}}
    observation_count INT DEFAULT 1,
    myelinated BOOLEAN DEFAULT FALSE,
    first_seen_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ
);

CREATE TABLE scene_cluster_associations (
    scene_id INT REFERENCES scene_snapshots(id),
    cluster_id INT REFERENCES sensory_clusters(id),
    strength FLOAT DEFAULT 1.0,
    observation_count INT DEFAULT 1,
    PRIMARY KEY (scene_id, cluster_id)
);

CREATE INDEX idx_scene_embedding ON scene_snapshots
    USING hnsw (scene_embedding vector_cosine_ops);
```

## Implementation Phases

**Phase 1 (DONE):** Frame-level familiarity gate + ClusterCache
- SceneMemory with coarse embeddings ✓
- AttentionBudget gates foveal processing ✓
- ClusterCache brute-force (fine for < 500 clusters) ✓
- Object-level familiarity tagging on primitives ✓

**Phase 2 (DONE):** Homeostatic decay in dreamstate
- Global decay of unmyelinated co-occurrences (×0.97/cycle) ✓
- Recency penalty for stale associations ✓
- Archive-not-delete (dormant associations reactivatable) ✓
- Myelination after N challenges + M self-play confirmations ✓

**Phase 3 (Next):** Scene snapshot persistence
- Save/load scene embeddings + cluster associations to DB
- Scene recognition on entry (match against snapshots)
- Scene-gated cluster loading into ActiveSet

**Phase 4:** Active Set with decay bridge
- Replace ClusterCache with ActiveSet
- Scene transitions: decay old, load new, keep shared
- Per-frame tick() for activation decay

**Phase 5:** FAISS integration (when clusters > 500)
- FAISS IndexFlatIP for Tier 1 matching
- Rebuild on scene transitions
- Fallback to pgvector Tier 2 on miss

**Phase 6:** Tiered query cascade
- Tier 1 miss → Tier 2 pgvector HNSW
- Tier 2 miss → Tier 3 cold scan (dreamstate only)
- Metrics: track miss rates per tier

**Phase 7:** Cross-system integration
- Scene context feeds into nmem briefing
- nmem-sym symbols activate sensory clusters (word → visual recall)
- Sensory clusters ground nmem-sym symbols (visual → word)

## Key Design Principles

1. **Queries only flow downward on failure.** Familiar scene → Tier 1 only.
2. **Activation is the scaling solution.** Billions of nodes; active set is always < 200.
3. **Scenes are retrieval cues.** They prime expectations, spatial maps, cluster associations.
4. **Decay bridges, not hard cuts.** Scene transitions fade rather than purge.
5. **Myelination prevents re-learning.** Stable associations fire instantly.
6. **Same patterns as nmem/nmem-sym.** Budget-aware, recency-weighted, recognition-tiered.
7. **Context determines working memory.** Like a transformer's attention window — finite, relevant, dynamically loaded.

## Files to Create/Modify

- **NEW** `scene_model.py` — SceneContext, SceneSnapshot persistence, spatial maps
- **MODIFY** `familiarity.py` — ActiveSet replaces ClusterCache, decay bridge logic
- **MODIFY** `api.py` — scene detection in ingest_frame, active set management
- **MODIFY** `consolidation.py` — scene snapshot creation, cluster association updates
- **MODIFY** `sensory_dreamstate.py` — Tier 3 deep search during dreamstate
- **MODIFY** `config.py` — tier thresholds, decay rates, scene matching thresholds

## Verification

1. **Scene recognition:** Process 3 different video dirs. Each forms a scene snapshot. Re-entering a dir triggers scene reload with reduced processing.
2. **Decay bridge:** Switch between dirs. Old scene clusters decay over 10-20 frames. Shared clusters (background colors) stay active.
3. **Scaling:** Synthetically create 10K clusters. Active set stays at 50-200. Query latency constant.
4. **Quality preservation:** Same recognition accuracy with scene gating as without.
