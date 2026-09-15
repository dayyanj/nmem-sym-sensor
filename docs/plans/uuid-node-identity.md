# UUID Node Identity — Decouple Labels from Learning

## Problem

Node identity currently depends on text labels in three critical paths:

1. **Iconic buffer promotion** (`promote_from_iconic`): `GROUP BY label, modality, node_type`
   - Two visually different objects with the same label merge into one observation group
   - Two visually identical objects with different labels create separate groups

2. **Node upsert** (`upsert_node`): `ON CONFLICT (normalized_label, node_type, modality)`
   - Color/texture nodes: fully label-dependent (always hit the unique constraint)
   - Shape nodes: partially fixed — uses embedding similarity first, but falls back to label with discriminator suffix (`rectangle-gray-matte#1`)

3. **Edge creation** (`_ingest_visual_analysis`): looks up nodes by `normalized_label`
   - If a shape got a discriminator suffix, the lookup fails (searching for `rectangle-gray-matte` but node is `rectangle-gray-matte#1`)
   - Spatial relations silently lost

### What we want

The same pattern as sound units: identity by embedding similarity, text label for diagnostics only.

```
Current:  image → label("red-hexagon-noisy") → GROUP BY label → promote → upsert ON CONFLICT label
Proposed: image → embedding(512-dim) → GROUP BY embedding similarity → promote → upsert by embedding match
```

## Design

### Core change: UUID as node identifier

Replace `normalized_label` as the identity key with a UUID. The unique constraint moves from `(normalized_label, node_type, modality)` to `(id)` (already exists as PK).

Node matching becomes exclusively embedding-based:
- Buffer entries store embeddings (already do)
- Promotion groups by embedding cluster, not label
- Upsert finds existing node by embedding similarity (already works for shapes)
- Labels stored as `diagnostic_label` on features — never used for matching

### Phase 1: Buffer grouping by embedding (replace GROUP BY label)

**Current** (`promote_from_iconic`):
```sql
GROUP BY label, modality, node_type HAVING COUNT(*) >= threshold
```

**Proposed**:
1. Fetch all non-expired buffer entries with embeddings
2. Group by embedding similarity (cosine > 0.85) in Python — same approach as `SoundLanguage.observe_sound()`
3. Each embedding cluster that meets the threshold gets promoted as one node
4. Entries without embeddings (rare) fall back to label grouping

This means 5 observations of a red hexagon from different angles (slightly different labels but similar embeddings) count as 5 observations of the same thing.

**Files**: `graph.py:promote_from_iconic()`

### Phase 2: Unified embedding-based upsert (all node types)

**Current**: Shape nodes use `_find_similar_shape_node()` + discriminator fallback. Color/texture nodes use label match only.

**Proposed**: ALL node types use embedding similarity for dedup.
- Shape: already works (just remove the discriminator fallback)
- Color: embedding-based — two different reds with different embeddings get separate nodes
- Texture: embedding-based — same logic

The `normalized_label` column becomes diagnostic only. The unique constraint `(normalized_label, node_type, modality)` needs to either:
- **Option A**: Drop and replace with a looser constraint, or
- **Option B**: Keep but make `normalized_label` a UUID string (guaranteed unique per creation)

**Option B is safer** — existing queries that reference the constraint don't break, UUIDs are guaranteed unique so the constraint never fires, and we don't need a migration to drop/recreate the index.

```python
import uuid
norm = str(uuid.uuid4())  # replaces normalize_label(label)
```

**Files**: `graph.py:upsert_node()`, `graph.py:normalize_label()`, `graph.py:_find_similar_shape_node()`

### Phase 3: Edge creation by node ID (replace label lookup)

**Current**: After ingestion, looks up `normalized_label` to find node IDs for edge creation.

**Proposed**: The ingest path tracks `{buffer_entry_id → node_id}` from promotion, and `{primitive_index → buffer_entry_id}` from buffer observation. Edge creation uses these mappings directly — no label lookup needed.

```python
# Current
label_to_ids: dict[str, int] = {}
label_to_ids[prim.label] = entry_id

# Proposed
prim_to_node: dict[int, int] = {}  # primitive index → node_id
```

For spatial relations between primitives, we already have the node IDs from promotion — just wire them through.

**Files**: `api.py:_ingest_visual_analysis()`

### Phase 4: Extend embedding match to color/texture

Colors are currently categorical (`hue=27.6, saturation=0.352, value=0.702`) with no 512-dim embedding. Options:

1. **Use the dual-stream encoder on color crops too** — produces a 512-dim embedding that captures the actual color appearance, not just HSV stats. Two "red" objects with different textures get different embeddings.
2. **Synthesize a mini-embedding from color features** — concatenate (hue, saturation, value, dominant_hue_angle) into a short vector, L2 normalise, use for matching. Cheaper but less discriminative.
3. **Keep label-based for colors, UUID for shapes** — simplest, acknowledges that color is inherently categorical.

**Recommendation**: Option 3 for now. Color nodes ARE categorical — "red" is "red" regardless of texture. Shape nodes need embedding identity because two "irregular" shapes can be completely different objects. We can revisit if color discrimination becomes a problem.

For color nodes specifically, keep the `normalized_label` match but derive it from the color features (hue bucket + saturation bucket) rather than the text label:
```python
# Color identity from features, not from label text
color_id = f"color_{int(hue/30)}_{int(sat*10)}_{int(val*10)}"
```

### DB Migration

```sql
-- 1. Add diagnostic_label column (stores the old text label for inspection)
ALTER TABLE sensory_nodes ADD COLUMN IF NOT EXISTS diagnostic_label TEXT;

-- 2. Copy existing labels to diagnostic_label
UPDATE sensory_nodes SET diagnostic_label = label WHERE diagnostic_label IS NULL;

-- 3. For shape nodes, replace normalized_label with UUID
-- (Do this in Python migration script, not raw SQL, to generate UUIDs)

-- 4. Drop the old unique constraint and create new one
-- Only after verifying all code paths use embedding matching
```

**Note**: This migration is BACKWARDS COMPATIBLE. The `normalized_label` column stays, it just contains UUIDs for shapes and color-fingerprints for colors. Old queries still work — they just won't match anything new by the old label format.

## Implementation Order

1. **`graph.py:promote_from_iconic()`** — embedding-grouped promotion
2. **`graph.py:upsert_node()`** — UUID normalized_label for shapes, remove discriminator hack
3. **`api.py:_ingest_visual_analysis()`** — track prim→node mapping by index, not label
4. **`graph.py:_find_similar_shape_node()`** — extend to color/texture if needed
5. **Migration script** — backfill `diagnostic_label`, convert existing `normalized_label` to UUIDs
6. **Dashboard/describe** — use `diagnostic_label` for human-readable output

## Risks

1. **Embedding instability**: if the dual-stream encoder produces slightly different embeddings for the same object across frames, grouping might fragment. Mitigated by the 0.85 similarity threshold (already proven for shapes).

2. **Color fragmentation**: without embeddings, colors could over-split (two "reds" that are visually identical but have slightly different HSV). Mitigated by using hue/sat/val buckets instead of raw values.

3. **Migration complexity**: 430+ existing nodes need normalised_label updated. A migration script handles this safely.

4. **Edge lookup during transition**: while old code lookups by label and new code uses IDs, there could be a brief period where edges fail to create. The fix is to deploy all changes together in one restart.

## Verification

1. **Dedup quality**: ingest a video with the same object from different angles. Should produce ONE node (embedding match), not multiple (label mismatch).
2. **Separation**: ingest a video with two different objects that happen to share a label. Should produce TWO nodes (embedding difference).
3. **Edges**: verify spatial relations (part_of, adjacent_to) still recorded between nodes after the change.
4. **Cluster formation**: verify clusters still form correctly from embedding-matched nodes.
5. **Grounding**: verify word→visual grounding still works (it uses co-occurrence IDs, not labels — should be unaffected).
