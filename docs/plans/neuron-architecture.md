# Neuron Architecture — Fixed Neurons, Edge-Bound Memory

## Problem

The current system creates ~600 nodes after 2 days of learning 328 videos. Nodes accumulate
thousands of observations via EMA merging, becoming broad averages that represent categories
rather than specific experiences. "Pink" has 17,546 observations — every pink thing ever seen
is one node. This prevents:

- Meaningful myelination (strong associations to vague constructs)
- Concept link formation (nothing specific to link between)
- Rich edge structure (no edges needed when everything is one mega-node)
- Proper clustering (clusters ARE the category layer, but nodes already are categories)

## Biological Principle

Neurons don't merge. They fire or don't fire. Memory is the PATTERN of simultaneous
activation, not the content of a single neuron.

A telephone number isn't stored as one neuron. It's stored as:
- Neuron for "3" (from childhood letterbox)
- Neuron for "04" (birth month)
- Neuron for jingle pattern (from a commercial)
- Temporal binding edges connecting them in sequence

"Dayyan's red cup" isn't one mega-node. It's:
- [specific red hue neuron] + [curved surface neuron] + [handle shape neuron]
- Bound by spatial edges (handle attached to body)
- Bound to [Dayyan] and [morning] and [kitchen] through co-occurrence edges

## Architecture Change

### Current (Category Nodes)
```
observation → decompose → find similar node (0.75+) → merge into it
                                                    → increment obs count
                                                    → EMA features
Result: 600 fat nodes, few edges, no structure
```

### New (Fixed Neurons)
```
observation → decompose into primitives → for each primitive:
    → search for matching neuron (0.95+ cosine)
    → FOUND: activate it (record activation, don't modify)
    → NOT FOUND: create new neuron (embedding fixed at birth)
    → record edges between all simultaneously active neurons
Result: many small neurons, rich edge structure, clusters emerge naturally
```

## Key Rules

1. **Neuron embeddings are immutable after creation.** No EMA, no merging, no drift.
2. **Observation count = activation count.** How many times this neuron fired, not how many things it absorbed.
3. **Features are set at birth.** No `|| COALESCE` merging of new features.
4. **Matching threshold is 0.95+** for visual, 0.90+ for audio. Only truly identical primitives share a neuron.
5. **Edges are the knowledge structure.** Co-activation, spatial, temporal, cross-modal — all through edges.
6. **Clusters group neurons that fire together.** This is the category layer (what "cup" means).
7. **Myelination applies to EDGES**, not just co-occurrences. A myelinated edge = permanent association between specific neurons.

## Implementation

### Files to Change

#### 1. `graph.py` — Core Changes

**`upsert_node()` → `activate_or_create_neuron()`**

```python
async def activate_or_create_neuron(
    pool: asyncpg.Pool,
    embedding: list[float],
    modality: str,
    node_type: str,
    label: str,           # diagnostic only
    features: dict,       # set at creation, never merged
    frame_id: str = None,
) -> tuple[int, bool]:
    """Activate an existing neuron or create a new one.

    Returns (node_id, is_new).

    Search: cosine similarity against all neurons of same type+modality.
    If best match >= NEURON_MATCH_THRESHOLD: activate (increment count, record time).
    Otherwise: create new neuron with fixed embedding and features.

    NEVER modifies embedding or features of existing neurons.
    """
```

**`promote_from_iconic()` — Simplified**

Current: groups buffer entries by similarity → merges group into one node.
New: each buffer entry independently either activates an existing neuron or creates one.
No grouping needed — the threshold handles dedup naturally.

```python
# For each unexpired buffer entry:
node_id, is_new = await activate_or_create_neuron(
    pool, entry["embedding"], entry["modality"],
    entry["node_type"], entry["label"], entry["features"],
)
promoted_ids.append(node_id)
# Record edges between all promoted_ids from this frame
```

**Recognition pass removed** — it's now the same operation as promotion.
There's no distinction between "recognise existing" and "promote new" — both go through
`activate_or_create_neuron()`.

#### 2. `config.py` — New Constants

```python
# Neuron matching thresholds (only near-identical share a neuron)
NEURON_VISUAL_THRESHOLD = float(os.environ.get("NMEM_SENSOR_NEURON_VIS_THRESH", "0.95"))
NEURON_AUDIO_THRESHOLD = float(os.environ.get("NMEM_SENSOR_NEURON_AUD_THRESH", "0.90"))
```

#### 3. `api.py` — Edge Recording

After processing a frame, all active neuron IDs from that frame get co-activation edges:

```python
# All neurons active in this frame
active_ids = [node_id for node_id in promoted_ids if node_id is not None]

# Record co-activation edges between all pairs
for i in range(len(active_ids)):
    for j in range(i + 1, len(active_ids)):
        await upsert_edge(pool, active_ids[i], active_ids[j],
                         "co_activated", confidence=0.5)
```

This replaces the current spatial-relation-only edge creation. Every frame produces
edges between everything active in that frame.

#### 4. `consolidation.py` — No Changes Needed

Clustering already reads node embeddings as immutable and computes centroids on clusters.
With more nodes (thousands instead of hundreds), clustering becomes MORE effective because
there are actually distinct things to group.

#### 5. `familiarity.py` — No Changes Needed

Already matches against cluster centroids, not node embeddings.

#### 6. `sensory_dreamstate.py` — Minor Updates

Self-play round-trip testing works on co-occurrences, not nodes directly. No change needed.
Mental rotation already uses node features read-only.

#### 7. `binding.py` — No Changes Needed

Edge creation between nodes. Already works with node IDs.

### Schema Changes

```sql
-- Add activation tracking columns
ALTER TABLE sensory_nodes ADD COLUMN IF NOT EXISTS last_activated_at TIMESTAMPTZ;
ALTER TABLE sensory_nodes ADD COLUMN IF NOT EXISTS activation_count INTEGER DEFAULT 0;

-- Drop the normalized_label unique constraint (no longer needed for dedup)
-- Dedup is now purely embedding-based
DROP INDEX IF EXISTS idx_sensory_nodes_normalized_label_type_modality;

-- Add HNSW index for fast similarity search (if not exists)
CREATE INDEX IF NOT EXISTS idx_nodes_visual_hnsw
    ON sensory_nodes USING hnsw (visual_embedding vector_cosine_ops)
    WHERE visual_embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nodes_audio_hnsw
    ON sensory_nodes USING hnsw (audio_embedding vector_cosine_ops)
    WHERE audio_embedding IS NOT NULL;
```

### Edge Explosion Mitigation

With co-activation edges between all simultaneously active neurons per frame,
edge count could explode (N neurons per frame → N*(N-1)/2 edges).

Mitigations:
- **Only edge between neurons of different types.** A colour neuron doesn't need an edge
  to another colour neuron from the same frame — they're alternatives, not associations.
- **Attention-gated edges.** Only neurons in the foveal attention region get edges to each other.
  Background neurons don't bind to foreground.
- **Edge strength via co-occurrence store.** Don't create an edge on first co-activation.
  Use the existing co-occurrence system — edges only materialise after repeated co-activation
  exceeds a threshold. This is what we already have.
- **Temporal proximity weighting.** Neurons active within 500ms of each other get stronger
  binding than those 2s apart.

### Expected Outcomes

After the same 328 videos with the new architecture:
- **Nodes**: 5,000-50,000 (vs 598) — many small specific neurons
- **Edges**: 50,000-500,000 (vs 3,376) — rich binding structure
- **Clusters**: More meaningful — grouping genuinely distinct neurons that co-activate
- **Myelination**: Specific associations ("this exact red + this exact curve = this cup")
- **Concept links**: Should finally form (specific visual cluster ↔ specific sound unit)
- **New node creation rate**: 10-100 per video (vs 1-7 per hour)

### What Stays the Same

- Co-occurrence store (unchanged — tracks cross-modal bindings)
- Sound language system (sound units already work like neurons — clustering by similarity)
- Speech hierarchy (syllables, prediction, practice — all unchanged)
- Dreamstate (self-play, recombination, rotation — all operate on co-occurrences and edges)
- TABULA2 audio processing (unchanged)
- Dual-stream visual encoder (unchanged)
- Foveal attention (unchanged — still produces crops)
- Ebbinghaus decay (unchanged — applies to co-occurrences)

### Migration

1. Wipe DB (fresh start — old mega-nodes are useless)
2. Update `graph.py` (activate_or_create_neuron replaces upsert_node)
3. Update `config.py` (new thresholds)
4. Update `api.py` (edge recording from co-activation)
5. Remove iconic buffer grouping logic (each entry promotes independently)
6. Run full curriculum learning

### Verification

1. **Node count growth**: Should see 50-200 new neurons per video (not 1-7)
2. **Edge density**: Should see edges between co-active neurons from same frame
3. **Cluster formation**: Clusters should contain 5-20 distinct neurons (not 1 mega-node)
4. **Myelination**: Should myelinate specific associations ("this red" + "this cup shape")
5. **Concept links**: Should form between visual clusters and sound units
6. **No embedding drift**: Node embeddings should never change after creation

## Open Questions

1. **Should we keep observation_count?** It's now "activation count" — how many times this
   neuron fired. Useful for salience/decay but semantically different from before.
   → YES, keep it. Rename mentally to "activation_count" but column name can stay.

2. **Should edges have types beyond co_activated?** Currently we have: part_of, contains,
   adjacent_to, bound_to, co_occurs_with. Add "co_activated" for frame-level binding.
   → YES. "co_activated" is the base edge. Others (spatial, cross-modal) are specialisations.

3. **How to handle the iconic buffer?** Currently groups entries before promoting. With
   fixed neurons, each entry promotes independently.
   → SIMPLIFY. Buffer still prevents noise (5s decay), but promotion is just
   activate_or_create per entry. No grouping pass needed.

4. **What about colour nodes?** A pure red pixel is always the same — should genuinely be
   one neuron. The 0.95 threshold handles this: truly identical colours match, slightly
   different ones don't.
   → CORRECT. Two pixels of rgb(255,0,0) activate the same neuron. rgb(255,10,0) might
   create a new one. Both get clustered into "red family" by consolidation.

5. **Scale concerns?** 50K nodes × 500K edges in pgvector.
   → pgvector HNSW handles 100K+ vectors easily. Edges are indexed by (source_id, target_id).
   The real scale solution is the existing tier system (active set, primed set, deep storage).
