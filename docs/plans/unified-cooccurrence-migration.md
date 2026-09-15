# Unified Co-occurrence Table — Migration Plan

## Current State

### Tables being replaced
| Table | Columns | Rows | Write paths | Read paths |
|-------|---------|------|-------------|------------|
| `sound_visual_cooccurrences` | sound_unit_id, visual_node_id, count, last_seen, myelinated, dreamstate_challenges, self_play_confirmations | ~1K | language.py observe_sound | selection.py (displacement, competition, spread), dreamstate (decay, self-play, concept linking), dashboard, imagery, dream_inspector |
| `saccade_sound_cooccurrences` | saccade_pattern_id, sound_unit_id, count, last_seen | ~8K | oculomotor.py observe_trajectory | dreamstate (saccade self-play), oculomotor recall |
| `saccade_visual_cooccurrences` | saccade_pattern_id, visual_node_id, count, last_seen | ~22K | oculomotor.py observe_trajectory | oculomotor recall_by_visual |
| `word_visual_cooccurrences` | word (text!), node_id, count, first_seen, last_seen | ~93 | text_bridge.py | dashboard, dream inspector |

### New table (already created)
```sql
sensory_cooccurrences (
    id BIGSERIAL PRIMARY KEY,
    unit_a_id BIGINT NOT NULL,
    modality_a TEXT NOT NULL,
    unit_b_id BIGINT NOT NULL,
    modality_b TEXT NOT NULL,
    count INT NOT NULL DEFAULT 1,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    myelinated BOOLEAN NOT NULL DEFAULT FALSE,
    dreamstate_challenges INT NOT NULL DEFAULT 0,
    self_play_confirmations INT NOT NULL DEFAULT 0,
    UNIQUE (unit_a_id, modality_a, unit_b_id, modality_b)
)
```

### New API (already built and tested)
`cooccurrence.py` — `CooccurrenceStore` with `observe()`, `query()`, `recall()`, `displace()`, `stats()`

### Already migrated
- `language.py` — observe_sound writes to unified table via `cooc_store.observe()`
- `oculomotor.py` — observe_trajectory writes to unified table
- `sensory_dreamstate.py` — homeostatic decay + self-play + concept linking use unified table

### NOT yet migrated
- `selection.py` — still references old column names (sound_unit_id, visual_node_id) via a broken sed replace
- `dashboard.py` — still queries old tables
- `ingest/text_bridge.py` — still writes to word_visual_cooccurrences
- `imagery.py` — still queries old tables
- `dream_inspector.py` — still queries old tables

## Scalability Considerations

### Row growth projections
| Scenario | Modality pairs | Estimated rows | Growth rate |
|----------|---------------|----------------|-------------|
| Current (video learning) | voice↔visual, motor↔visual, motor↔voice, text↔visual | ~30K | ~1K/hour |
| + Kinect depth | + depth↔visual, depth↔voice, depth↔motor | ~100K | ~5K/hour |
| + Olfactory (BME690) | + olfactory↔visual, olfactory↔voice, olfactory↔depth | ~200K | ~10K/hour |
| + Compass/Accelerometer | + compass↔visual, compass↔olfactory, ... | ~500K | ~20K/hour |
| Long-term (months) | All combinations | 10M-1B+ | continuous |

### Query patterns that must be fast

**Pattern 1: Observe (INSERT/UPSERT)** — called per-frame, multiple times
```sql
INSERT INTO sensory_cooccurrences (unit_a_id, modality_a, unit_b_id, modality_b, count)
VALUES ($1, $2, $3, $4, 1)
ON CONFLICT ... DO UPDATE SET count = count + 1
```
- Index: UNIQUE constraint on (unit_a_id, modality_a, unit_b_id, modality_b)
- Scale concern: UNIQUE index becomes large. B-tree on 4 columns.
- At 1B rows: INSERT latency could grow. Partitioning by modality_pair might help.

**Pattern 2: Query one unit's partners (fan-out)**
```sql
-- "What visual nodes does sound unit 42 bind to?"
SELECT ... WHERE unit_a_id = $1 AND modality_a = $2 AND modality_b = $3
UNION ALL
SELECT ... WHERE unit_b_id = $1 AND modality_b = $2 AND modality_a = $3
```
- Needs TWO index lookups (unit could be on either side)
- Index: idx_cooc_a (unit_a_id, modality_a) + idx_cooc_b (unit_b_id, modality_b)
- Scale concern: UNION ALL doubles the work. At 1B rows each index scan is still O(log n).

**Pattern 3: Darwinian displacement (UPDATE competitors)**
```sql
-- "Weaken all OTHER voice units that bind to visual node 100"
UPDATE sensory_cooccurrences SET count = GREATEST(0, count - $1)
WHERE (unit_a_id = $target AND modality_a = 'visual' AND modality_b = 'voice' AND unit_b_id != $winner)
   OR (unit_b_id = $target AND modality_b = 'visual' AND modality_a = 'voice' AND unit_a_id != $winner)
```
- This is the most expensive query — scans for competitors on a specific node
- Index: (unit_a_id, modality_a, modality_b) and (unit_b_id, modality_b, modality_a)
- Scale concern: OR clauses can't use a single index. Two separate UPDATE statements (already done in cooc_store.displace()) is better.

**Pattern 4: Dreamstate decay (bulk UPDATE)**
```sql
UPDATE sensory_cooccurrences SET count = GREATEST(1, count * 0.97)
WHERE NOT myelinated AND count > 1 AND last_seen < NOW() - '1 hour'
```
- Full table scan, filtered by myelinated + count + last_seen
- Partial index: idx_cooc_active (last_seen) WHERE count > 0 — already created
- Scale concern: At 1B rows this becomes slow. Partitioning by myelinated helps (separate myelinated partition never scanned).

**Pattern 5: Concept linking (find units with shared partners)**
```sql
-- "Which voice units share visual partners?"
SELECT uid, array_agg(vid) FROM (
    SELECT unit_a_id as uid, unit_b_id as vid FROM sensory_cooccurrences
    WHERE modality_a = 'voice' AND modality_b = 'visual' AND count >= 3
    ...
) GROUP BY uid
```
- Needs modality_pair index: idx_cooc_modpair (modality_a, modality_b)
- Scale concern: At 1B rows, filtering by modality pair first is essential.

### Index strategy

```sql
-- Already created
CREATE INDEX idx_cooc_a ON sensory_cooccurrences (unit_a_id, modality_a);
CREATE INDEX idx_cooc_b ON sensory_cooccurrences (unit_b_id, modality_b);
CREATE INDEX idx_cooc_modpair ON sensory_cooccurrences (modality_a, modality_b);
CREATE INDEX idx_cooc_active ON sensory_cooccurrences (last_seen) WHERE count > 0;

-- Additional for displacement queries
CREATE INDEX idx_cooc_target_a ON sensory_cooccurrences (unit_a_id, modality_a, modality_b) WHERE count > 0;
CREATE INDEX idx_cooc_target_b ON sensory_cooccurrences (unit_b_id, modality_b, modality_a) WHERE count > 0;
```

### Future partitioning (when rows > 10M)
```sql
-- Partition by modality pair for query isolation
CREATE TABLE sensory_cooccurrences (
    ...
) PARTITION BY LIST (modality_a || '↔' || modality_b);

-- Each modality pair gets its own partition
CREATE TABLE cooc_visual_voice PARTITION OF sensory_cooccurrences
    FOR VALUES IN ('visual↔voice');
CREATE TABLE cooc_motor_visual PARTITION OF sensory_cooccurrences
    FOR VALUES IN ('motor↔visual');
-- etc.
```
This keeps each modality pair's data physically together, improving scan performance for Pattern 4 (bulk decay) and Pattern 5 (concept linking).

## Migration Plan

### Phase 1: Fix selection.py (revert sed, use CooccurrenceStore)

Revert the broken sed replace. Instead of patching SQL queries, refactor selection.py to use `CooccurrenceStore` methods:

**`displace_sound_competitors()`** → `cooc_store.displace()`
- Already implemented and tested in cooccurrence.py
- Replace the entire function body with a single call

**`run_sound_competition()`** → New method on CooccurrenceStore
```python
async def find_competitive_pairs(self, modality_a, modality_b, pool, min_count=5):
    """Find target nodes with 2+ competing units from the same modality."""
```

**`prune_sound_spread()`** → New method on CooccurrenceStore
```python
async def find_spread_units(self, modality, pool, spread_threshold=5):
    """Find units that bind to too many targets (promiscuous binding)."""
```

### Phase 2: Migrate text_bridge.py

The `word_visual_cooccurrences` table uses `word` (TEXT) as the key, not an integer ID. Options:
1. **Hash the word**: `unit_a_id = hashtext(word)`, `modality_a = 'text'` — simple but loses the word string
2. **Create text_units table**: Like sound_units but for text tokens. Each word gets an ID.
3. **Keep word_visual_cooccurrences**: It's diagnostic only, low volume. Not worth migrating.

**Recommendation: Option 3** — keep word_visual as-is. It's diagnostic (STT transcription reference), low volume (93 rows), and doesn't need myelination/decay. The unified table handles the primary learning paths (voice, visual, motor, olfactory). Text stays as a diagnostic sidecar.

### Phase 3: Migrate dashboard.py

Replace all old-table queries with `CooccurrenceStore` calls:
- Overview stats → `cooc_store.stats(pool)`
- Bindings page → `cooc_store.query()` per modality pair
- Recall page → `cooc_store.recall()`
- Concept links → unchanged (already uses concept_links table)

### Phase 4: Migrate imagery.py and dream_inspector.py

These have 2 refs each — straightforward `cooc_store.query()` replacements.

### Phase 5: Add displacement indexes

```sql
CREATE INDEX idx_cooc_target_a ON sensory_cooccurrences (unit_a_id, modality_a, modality_b) WHERE count > 0;
CREATE INDEX idx_cooc_target_b ON sensory_cooccurrences (unit_b_id, modality_b, modality_a) WHERE count > 0;
```

### Phase 6: Drop old tables (after validation)

Only after a full learning run on the unified table confirms everything works:
```sql
DROP TABLE sound_visual_cooccurrences;
DROP TABLE saccade_sound_cooccurrences;
DROP TABLE saccade_visual_cooccurrences;
-- Keep word_visual_cooccurrences (diagnostic)
```

## Files to modify

| File | Refs | Approach | Effort |
|------|------|----------|--------|
| `selection.py` | 10 | Revert sed, refactor to use CooccurrenceStore methods | High — new methods on store |
| `dashboard.py` | 11 | Replace queries with cooc_store calls | Medium |
| `ingest/text_bridge.py` | 11 | Keep word_visual table (diagnostic) | None |
| `imagery.py` | 2 | cooc_store.query() | Low |
| `dream_inspector.py` | 2 | cooc_store.query() | Low |

## Verification

1. **Write test**: observe() through all modality pairs, verify canonical ordering
2. **Query test**: query() from both sides returns same results
3. **Displacement test**: competitors weakened, winner + myelinated protected
4. **Decay test**: homeostatic decay affects all modalities equally, grace period works
5. **Self-play test**: round-trip across any modality pair
6. **Selection test**: competition + spread pruning work through CooccurrenceStore
7. **Dashboard test**: all pages render correctly
8. **Full learning run**: 6x primitives, verify accuracy >= previous runs
9. **Scale test**: Insert 100K synthetic rows, verify query latency stays under 10ms
