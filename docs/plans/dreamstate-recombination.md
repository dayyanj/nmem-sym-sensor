# Dreamstate Recombination — Creative Consolidation

## Origin

During a nap, the user observed their dream taking a real morning event, stripping it
from its original context, and recombining it with unrelated content to create a coherent
test scenario. The dream was exploring how the event would play out in a different setting —
testing whether the associations held, what emotions arose, and how the outcome changed.

This is not random hallucination. It is **purposeful recombination** — the brain's mechanism
for discovering which memories are universal truths vs context-specific associations.

## Purpose

Three cognitive functions:

1. **Generalise** — extract abstract patterns from specific events.
   "I saw a triangle on a blue background" → "triangles can exist on any background."

2. **Simulate** — run patterns through different contexts to explore outcomes.
   "What if this triangle appeared in the kitchen scene? Does it still activate 'shape'?"

3. **Evaluate** — test if associations hold outside their original context.
   "The red mug always appears on my desk. Does 'red mug' still make sense in the garden?"
   If yes → universal association. If no → context-bound association.

## Prerequisites

All in place:

- **Scene model** (`chronoception.py`) — provides alternative contexts (scene snapshots
  with associated clusters and spatial graphs)
- **Co-occurrence store** (`cooccurrence.py`) — tracks what goes with what, with
  strength/half-life for each association
- **Allocentric spatial graph** (`scene_spatial_edges`) — where things are relative
  to each other within a scene
- **Active set** — cluster loading/unloading with decay bridge
- **Ebbinghaus decay** — per-pair strength that can be modified by recombination outcomes
- **Self-play** — existing round-trip testing infrastructure

## Algorithm

### Phase 1: Selection — "What needs processing?"

Pick a **recombination candidate**: a cluster that is:
- Recently observed (last_reinforced within last N hours)
- NOT myelinated (already consolidated = skip)
- Has moderate-to-high reinforcement count (enough signal to test)
- Has NOT been recombined recently (avoid re-testing the same thing)

```sql
SELECT sc.id, sc.grounded_label, sc.total_observations,
       MAX(co.last_reinforced) as last_active
FROM sensory_clusters sc
JOIN sensory_cooccurrences co ON
    (co.unit_a_id = sc.id AND co.modality_a = 'visual')
    OR (co.unit_b_id = sc.id AND co.modality_b = 'visual')
WHERE NOT co.myelinated
  AND co.reinforcement_count >= 5
  AND co.last_reinforced > NOW() - INTERVAL '24 hours'
ORDER BY co.last_reinforced DESC
LIMIT 10
```

From these 10 candidates, pick one at random (not always the most recent —
varied selection produces more diverse recombination).

### Phase 2: Context Switch — "Where else could this appear?"

Pick a **target scene** that is:
- Different from the candidate's home scene
- Has some overlap in associated cluster types (not completely alien)
- OR is randomly selected (exploratory recombination — test in unfamiliar territory)

Strategy: 70% similar scene (shared modalities/types), 30% random scene.

Similar scene selection:
```python
# Find scenes that share some cluster types but not this specific cluster
candidate_modalities = get_cooccurrence_modalities(candidate_id)
target_scenes = find_scenes_with_modality_overlap(
    candidate_modalities,
    exclude_scenes=[candidate_home_scene],
    min_overlap=0.3,
)
```

### Phase 3: Placement — "Does it fit here?"

Check spatial plausibility: can the candidate cluster exist in the target scene?

1. **Load target scene's spatial graph** — what's already there and where?
2. **Find compatible positions** — based on allocentric relations the candidate
   typically has (e.g., "this object is usually ON something" → find a surface)
3. **Check conflicts** — does placing it here violate any spatial constraints?
   (e.g., "two objects can't occupy the same NEAR relation to the same anchor")

Score: `spatial_compatibility = compatible_relations / total_relations`

If spatial_compatibility < 0.2 → skip this combination (too implausible).
Even dreams have some coherence — complete nonsense doesn't test anything useful.

### Phase 4: Activation Spread — "What lights up?"

With the candidate "placed" in the target scene:

1. **Activate candidate's co-occurrences** — what does this cluster usually appear with?
   ```python
   coocs = await cooc_store.recall(candidate_id, "visual", pool)
   # Returns: {"voice": [...], "motor": [...], "environmental": [...]}
   ```

2. **Activate target scene's existing associations** — what's already expected here?
   ```python
   scene_clusters = await load_scene_members(target_scene_id, pool)
   scene_coocs = {c: await cooc_store.recall(c, "visual", pool) for c in scene_clusters}
   ```

3. **Compare activation patterns:**
   - **Reinforcing**: candidate's voice associations also appear in scene context
     → "this word applies here too" → strengthen as universal
   - **Conflicting**: candidate's associations contradict scene's associations
     → "this word only applies in the original context" → flag as context-specific
   - **Novel**: candidate brings associations the scene has never seen
     → "new information" → neither strengthen nor weaken, just note

### Phase 5: Evaluation — "What did we learn?"

Score each co-occurrence of the candidate:

```python
for cooc in candidate_cooccurrences:
    if cooc in reinforcing_set:
        # Universal — works across contexts
        cooc.strength *= 1.05  # small boost
        cooc.context_independence += 1
    elif cooc in conflicting_set:
        # Context-specific — only works in original scene
        cooc.strength *= 0.95  # small penalty
        cooc.context_dependence += 1
    else:
        # Novel — no evidence either way
        pass
```

The cumulative effect across many recombination cycles:
- Universal associations get slightly stronger each time → eventually myelinate
- Context-specific associations get slightly weaker → eventually go dormant
- This is MUCH gentler than Darwinian displacement — it's gradual discovery, not competition

### Phase 6: Record — "Remember what we dreamed"

Store the recombination result for future reference:

```sql
CREATE TABLE IF NOT EXISTS dreamstate_recombinations (
    id BIGSERIAL PRIMARY KEY,
    candidate_cluster_id INT NOT NULL,
    candidate_home_scene_id INT,
    target_scene_id INT NOT NULL,
    spatial_compatibility FLOAT,
    n_reinforcing INT DEFAULT 0,
    n_conflicting INT DEFAULT 0,
    n_novel INT DEFAULT 0,
    coherence_score FLOAT,  -- reinforcing / (reinforcing + conflicting)
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Over time, this table reveals:
- Which clusters are highly context-independent (always coherent in new scenes)
- Which clusters are tightly bound to specific scenes (always conflicting)
- Which scene pairs are "similar" (high cross-coherence)

## New Co-occurrence Columns

Add to `sensory_cooccurrences`:

```sql
ALTER TABLE sensory_cooccurrences ADD COLUMN
    context_independence INT DEFAULT 0;  -- times confirmed across contexts
ALTER TABLE sensory_cooccurrences ADD COLUMN
    context_dependence INT DEFAULT 0;    -- times conflicted across contexts
```

The ratio `context_independence / (context_independence + context_dependence)` gives
a **universality score** for each association. High universality → this association
is a general truth. Low universality → this association is scene-specific.

## Integration with Existing Dreamstate

Recombination runs as Step 11 in the dreamstate cycle, after:
- Step 8: Homeostatic decay (Ebbinghaus)
- Step 9: Self-play round-trip
- Step 10: Concept linking

Budget: 5-10 seconds per dreamstate cycle. One recombination per cycle is enough —
the effects accumulate over many cycles.

```python
# In sensory_dreamstate.py run_sensory_dreamstate():
if budget_remaining() > 5:
    recom_stats = await _dreamstate_recombination(pool, max_candidates=1)
    stats["recombination"] = recom_stats
```

## Connection to Mental Rotation

Recombination tests context-independence. Mental rotation tests orientation-independence.
Both are consolidation operations that discover invariances:

- Recombination: "does this concept work in a different PLACE?"
- Mental rotation: "does this concept work at a different ANGLE?"

They can share the same evaluation framework:
1. Take a stored representation
2. Transform it (change scene / rotate)
3. Check if associations still hold
4. Strengthen universal, weaken specific

## Gap Filling — The Creative Substrate

The dream didn't just place the morning event in a new context — it **filled in the gaps**
with plausible content. The cafe scenario had tables, people, sounds that were coherent
but not from the original memory.

In our system, gap filling happens naturally through spreading activation:
- Place triangle in kitchen scene
- Kitchen scene activates: countertop, sink, window, warm lighting
- Triangle + kitchen context = "triangle-shaped object in a kitchen" → maybe a pizza slice,
  a cheese wedge, a door stopper
- The gap filling IS the spreading activation — the scene provides the context,
  the co-occurrence graph provides the content

This is how the system will eventually generate novel combinations it has never seen —
not through a generative model, but through **combinatorial activation** of existing memories
in new arrangements. The diffusion decoder can then visualise what this combination
would look like.

## Hypothesis Generation — The Missing Output

The current plan only tests existing associations. But the most powerful output
of recombination is **new hypotheses** — predictions that didn't exist before
the dream.

### The Chocolate Insight

"I dreamt that if I put chocolate on the counter it might melt in the sun."

This is NOT recalling a memory. It's **creating a prediction** from combining:
- chocolate (object) + counter (surface) + sun (heat source) = melting (outcome)

The system has never observed this. The dream INVENTED it by activating
spreading associations through a novel context. This hypothesis now needs
validation through future observation.

### Hypothesis Lifecycle

```
DREAMED → SPECULATIVE → PRIMED → CONFIRMED/REFUTED → INTEGRATED/PRUNED
```

**Stage 1: Dreamed**
Recombination produces a novel activation path — co-occurrences that don't
exist yet but are plausible given the spreading activation.

```python
novel_paths = []
for cooc in candidate_activations:
    for scene_cooc in scene_activations:
        combined = cooc.target + scene_cooc.target
        if not exists_in_cooccurrence_store(combined):
            novel_paths.append({
                "source_a": cooc,
                "source_b": scene_cooc,
                "predicted_association": combined,
                "plausibility": cooc.strength * scene_cooc.strength,
            })
```

**Stage 2: Hand off to nmem-sym**
The sensory system discovers novel paths. It does NOT store or manage hypotheses.
That's nmem-sym's job — it already has the full infrastructure:

- `nmem_sym.hypothesis.find_novel_paths()` — discovers novel activation paths
- `nmem_sym.hypothesis.score_plausibility()` — rates likelihood
- `nmem_sym.hypothesis.store_hypotheses()` — persists in nmem-sym DB
- `nmem_sym.hypothesis.update_hypothesis_status()` — confirm/refute lifecycle
- `nmem_sym.prediction.PredictionPlugin` — causal forward activation, deliberation

The sensory bridge (`sensory_bridge.py`) passes the novel paths upstream:

```python
# In sensory dreamstate recombination:
novel_paths = discover_novel_paths(candidate, target_scene, activations)

# Hand off to nmem-sym for hypothesis management
for path in novel_paths:
    await sensory_bridge.emit_hypothesis_candidate({
        "source": "dreamstate_recombination",
        "recombination_id": recom_id,
        "unit_a": path["source_a"],
        "unit_b": path["source_b"],
        "modality_a": path["modality_a"],
        "modality_b": path["modality_b"],
        "plausibility": path["plausibility"],
        "context": {
            "original_scene": candidate_home_scene,
            "target_scene": target_scene_id,
            "spatial_compatibility": spatial_score,
        },
    })
```

**Stage 3: nmem-sym manages the hypothesis lifecycle**
nmem-sym's existing pipeline handles:
- Plausibility scoring (via graph structure and causal reasoning)
- Storage (in nmem-sym's hypothesis tables)
- Status tracking (speculative → primed → confirmed/refuted)
- Curiosity drive (hypotheses create prediction expectations)

**Stage 4: Validation flows back to sensory system**
When nmem-sym confirms a hypothesis through observation, it signals back
to the sensory system to create a real co-occurrence:

```python
# nmem-sym confirms hypothesis, signals sensory bridge:
await sensory_bridge.emit_confirmed_hypothesis({
    "unit_a": h.unit_a_id,
    "modality_a": h.modality_a,
    "unit_b": h.unit_b_id,
    "modality_b": h.modality_b,
    "source": "hypothesis_confirmed",
})

# Sensory system receives and creates a real co-occurrence:
await cooc_store.observe(
    h.unit_a_id, h.modality_a,
    h.unit_b_id, h.modality_b,
    pool,
    attention=0.8,   # dream-confirmed = high quality encoding
    surprise=0.3,    # expected (we predicted it)
    proximity=0.5,   # moderate (inferred, not directly co-observed)
)
```

**Stage 5: Clean separation of concerns**

```
nmem-sym-sensor (sensory)          nmem-sym (intelligence)
─────────────────────────          ─────────────────────────
Observes the world                 Reasons about observations
Records co-occurrences             Generates hypotheses
Builds scene models                Scores plausibility
Runs dreamstate recombination      Tracks hypothesis lifecycle
  → discovers novel paths ──────→ find_novel_paths()
                                   store_hypotheses()
                                   predict() with causal reasoning
                                   curiosity drive
During observation:
  sensory_bridge ←───────────────  confirmed_hypothesis callback
  creates real co-occurrence       update_hypothesis_status()
```

This means NO hypothesis tables in sensory_memory DB. The sensory system's
job is to discover and report. nmem-sym's job is to reason and validate.

### New Concept Formation

The most powerful case: recombination discovers a **compound concept**.

"Chocolate + sun + counter = melting" isn't just three co-occurrences.
It's a new concept: "heat-sensitive materials change state on warm surfaces."

Detection: when multiple hypotheses from the same recombination session all
get confirmed, they might represent a new compound concept. The system should:

1. Detect co-confirmed hypothesis clusters (3+ hypotheses confirmed together)
2. Create a new cluster node representing the compound concept
3. Link it to all participating clusters via co-occurrences
4. The compound concept can then participate in FUTURE recombinations

This is how abstract concepts emerge from concrete observations — not programmed,
but discovered through the dream → hypothesis → validation → integration loop.

### Curiosity Drive

Unvalidated hypotheses create a natural curiosity signal:

```python
async def get_curiosity_targets(pool):
    """What does the system want to investigate?"""
    return await pool.fetch("""
        SELECT * FROM dreamstate_hypotheses
        WHERE status = 'speculative'
          AND plausibility > 0.3
          AND created_at > NOW() - INTERVAL '7 days'
        ORDER BY plausibility DESC
        LIMIT 10
    """)
```

These curiosity targets could influence:
- **Fovea direction**: bias attention toward scenes/objects that might validate hypotheses
- **Scene selection**: prefer visiting scenes where hypotheses are testable
- **Active exploration**: when in INSPECT phase, look for evidence of dream predictions

This closes the loop: dreams generate hypotheses → hypotheses create curiosity →
curiosity drives observation → observation confirms/refutes → confirmed hypotheses
become knowledge → knowledge informs future dreams.

## Implementation Order

### Phase A: Recombination Engine (test existing associations)
1. Add `context_independence` / `context_dependence` columns to co-occurrences
2. Create `dreamstate_recombinations` table
3. Implement `_select_recombination_candidate()`
4. Implement `_select_target_scene()`
5. Implement `_check_spatial_compatibility()`
6. Implement `_evaluate_recombination()` — strengthen/weaken existing
7. Wire into dreamstate cycle as Step 11
8. Test with primitive scenes

### Phase B: Hypothesis Bridge (sensory → nmem-sym)
9. Add `emit_hypothesis_candidate()` to sensory_bridge.py
10. Implement `_discover_novel_paths()` in sensory dreamstate — find novel
    activation paths from recombination spreading
11. Wire: recombination novel paths → bridge → nmem-sym `find_novel_paths()`
12. Add `emit_confirmed_hypothesis()` callback from nmem-sym → sensory bridge
13. Wire: confirmed hypothesis → `cooc_store.observe()` to create real co-occurrence
14. Test: run recombination, verify hypotheses reach nmem-sym, verify confirmation
    creates co-occurrence back in sensory system

### Phase C: Curiosity Drive (nmem-sym → sensory attention)
15. nmem-sym exposes `get_curiosity_targets()` via bridge
16. Sensory attention controller queries curiosity targets during INSPECT phase
17. Fovea biases toward hypothesis-relevant objects/scenes
18. Test: verify system actively seeks hypothesis-validating observations

### Phase D: Compound Concept Formation (nmem-sym)
19. nmem-sym detects co-confirmed hypothesis clusters (3+ confirmed together)
20. Auto-creates compound concept nodes in symbolic graph
21. Signals sensory bridge to link compound concept to participating clusters
22. Test: verify compound concepts participate in future recombination

## Success Criteria

After 100+ dreamstate cycles with recombination:
- Color associations (red, blue, black) should have HIGH universality scores
  (they work in any scene — a shape can be red anywhere)
- Shape-background associations should have LOW universality scores
  (pentagon on cyan background is specific to that video, not universal)
- The system should naturally discover that "triangle" is universal
  but "triangle on blue background" is context-specific
