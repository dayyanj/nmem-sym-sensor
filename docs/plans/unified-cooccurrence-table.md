# Unified N-to-N Sensory Co-occurrence Table

## Problem

Currently we have 4 separate co-occurrence tables:

| Table | A side | B side | Rows |
|-------|--------|--------|------|
| `sound_visual_cooccurrences` | sound_unit_id | visual_node_id | 1,172 |
| `saccade_sound_cooccurrences` | saccade_pattern_id | sound_unit_id | 8,450 |
| `saccade_visual_cooccurrences` | saccade_pattern_id | visual_node_id | 21,605 |
| `word_visual_cooccurrences` | word (text) | node_id | 93 |

Adding new sensors (olfactory, compass, accelerometer, depth) means N² tables. Smell alone adds smell↔visual, smell↔sound, smell↔motor. Each with its own schema, queries, decay logic, and Darwinian displacement code.

## Solution: Single Unified Table

```sql
CREATE TABLE sensory_cooccurrences (
    id BIGSERIAL PRIMARY KEY,

    -- Side A
    unit_a_id BIGINT NOT NULL,
    modality_a TEXT NOT NULL,       -- visual, voice, environmental, motor,
                                    -- olfactory, depth, compass, text, ...

    -- Side B
    unit_b_id BIGINT NOT NULL,
    modality_b TEXT NOT NULL,

    -- Co-occurrence strength
    count INT NOT NULL DEFAULT 1,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Dreamstate / myelination
    myelinated BOOLEAN NOT NULL DEFAULT FALSE,
    dreamstate_challenges INT NOT NULL DEFAULT 0,
    self_play_confirmations INT NOT NULL DEFAULT 0,

    -- Canonical ordering: modality_a < modality_b lexicographically,
    -- or if same modality, unit_a_id < unit_b_id.
    -- This prevents duplicate (A,B) and (B,A) entries.
    UNIQUE (unit_a_id, modality_a, unit_b_id, modality_b)
);

-- Indexes for fast lookup from either side
CREATE INDEX idx_cooc_a ON sensory_cooccurrences (unit_a_id, modality_a);
CREATE INDEX idx_cooc_b ON sensory_cooccurrences (unit_b_id, modality_b);
CREATE INDEX idx_cooc_modality_pair ON sensory_cooccurrences (modality_a, modality_b);
CREATE INDEX idx_cooc_last_seen ON sensory_cooccurrences (last_seen) WHERE count > 0;
```

## Canonical Ordering

To prevent duplicates, entries are always stored with modality_a <= modality_b
(lexicographic). If same modality, unit_a_id < unit_b_id.

```python
def canonical_order(id_a, mod_a, id_b, mod_b):
    """Ensure consistent ordering to prevent (A,B) + (B,A) duplicates."""
    if mod_a > mod_b or (mod_a == mod_b and id_a > id_b):
        return id_b, mod_b, id_a, mod_a
    return id_a, mod_a, id_b, mod_b
```

Example: `("voice", 42)` + `("visual", 17)` → stored as `(17, "visual", 42, "voice")`
because "visual" < "voice" lexicographically.

## Modality Registry

Rather than a fixed enum, modalities are strings. Known modalities:

| Modality | Unit type | Source |
|----------|-----------|--------|
| `visual` | sensory_nodes.id | Edge analysis, MoE encoder |
| `voice` | sound_units.id (modality='voice') | TABULA2 disentangler |
| `environmental` | sound_units.id (modality='environmental') | TABULA2 noise stream |
| `motor` | saccade_patterns.id | Oculomotor system |
| `text` | hash(word) or sensory_nodes.id | Whisper STT (diagnostic) |
| `olfactory` | olfactory_units.id (future) | BME690 + Grove gas sensors |
| `depth` | depth_nodes.id (future) | Kinect v1 |
| `compass` | compass_readings.id (future) | Accelerometer/compass |
| `proprioceptive` | motor_programs.id (future) | Body position awareness |

New sensors just register a new modality string. No schema changes.

## Migration Path

### Phase 1: Create new table alongside old ones
- Create `sensory_cooccurrences`
- Add helper functions that write to BOTH old and new tables
- New code reads from new table, falls back to old

### Phase 2: Migrate existing data
```sql
-- sound_visual → unified
INSERT INTO sensory_cooccurrences (unit_a_id, modality_a, unit_b_id, modality_b, count, last_seen, myelinated, dreamstate_challenges, self_play_confirmations)
SELECT visual_node_id, 'visual', sound_unit_id, 'voice', count, last_seen, myelinated, dreamstate_challenges, self_play_confirmations
FROM sound_visual_cooccurrences
WHERE count > 0;

-- saccade_sound → unified
INSERT INTO sensory_cooccurrences (unit_a_id, modality_a, unit_b_id, modality_b, count, last_seen)
SELECT saccade_pattern_id, 'motor', sound_unit_id, 'voice', count, last_seen
FROM saccade_sound_cooccurrences;

-- saccade_visual → unified
INSERT INTO sensory_cooccurrences (unit_a_id, modality_a, unit_b_id, modality_b, count, last_seen)
SELECT saccade_pattern_id, 'motor', visual_node_id, 'visual', count, last_seen
FROM saccade_visual_cooccurrences;

-- word_visual → unified (text modality uses hash of word as unit_id)
INSERT INTO sensory_cooccurrences (unit_a_id, modality_a, unit_b_id, modality_b, count, last_seen)
SELECT hashtext(word), 'text', node_id, 'visual', count, last_seen
FROM word_visual_cooccurrences;
```

### Phase 3: Update all code to use unified table
Files to modify (by reference count):
- `selection.py` (20 refs) — Darwinian displacement queries
- `sensory_dreamstate.py` (18 refs) — decay, self-play, concept linking
- `dashboard.py` (11 refs) — display queries
- `language.py` (8 refs) — observe_sound co-occurrence recording
- `oculomotor.py` (6 refs) — saccade co-occurrence recording
- `ingest/text_bridge.py` (11 refs) — word-visual recording
- `ingest/video.py` — binding context queries
- `imagery.py` (2 refs) — cross-modal recall
- `dream_inspector.py` (2 refs) — dream visualisation

### Phase 4: Drop old tables
Once all code migrated and verified.

## Unified API

```python
class CooccurrenceStore:
    """Unified N-to-N sensory co-occurrence storage.

    Any sensor modality can bind to any other. The store handles
    canonical ordering, upsert, decay, and query.
    """

    async def observe(
        self,
        unit_a: int, modality_a: str,
        unit_b: int, modality_b: str,
        pool: asyncpg.Pool,
    ):
        """Record a co-occurrence between two sensory units."""
        a_id, a_mod, b_id, b_mod = canonical_order(
            unit_a, modality_a, unit_b, modality_b
        )
        await pool.execute("""
            INSERT INTO sensory_cooccurrences
                (unit_a_id, modality_a, unit_b_id, modality_b, count)
            VALUES ($1, $2, $3, $4, 1)
            ON CONFLICT (unit_a_id, modality_a, unit_b_id, modality_b)
            DO UPDATE SET count = sensory_cooccurrences.count + 1,
                          last_seen = NOW()
        """, a_id, a_mod, b_id, b_mod)

    async def query(
        self,
        unit_id: int, modality: str,
        target_modality: str | None = None,
        pool: asyncpg.Pool,
        min_count: int = 1,
    ) -> list[dict]:
        """Find all co-occurrences for a unit, optionally filtered by target modality.

        "What visual nodes co-occur with sound unit 42?"
        → query(42, "voice", target_modality="visual")

        "What co-occurs with visual node 17 across ALL modalities?"
        → query(17, "visual")  # returns voice, motor, olfactory, ...
        """
        # Search both sides (unit could be A or B)
        ...

    async def recall(
        self,
        unit_id: int, modality: str,
        pool: asyncpg.Pool,
    ) -> dict[str, list]:
        """Full cross-modal recall: given one unit, what lights up across all senses?

        "I see a red circle" → returns:
        {
            "voice": [(sound_unit_42, 150), ...],   # "circle" sound
            "motor": [(saccade_7, 80), ...],          # circular eye movement
            "olfactory": [],                           # no smell associations yet
            "environmental": [(env_unit_3, 20), ...],  # associated ambient sound
        }
        """
        ...
```

## Query Patterns

### "What do I see when I hear 'triangle'?" (sound → visual recall)
```sql
SELECT unit_a_id as visual_id, count
FROM sensory_cooccurrences
WHERE unit_b_id = $1 AND modality_b = 'voice' AND modality_a = 'visual'
  AND count > 0
ORDER BY count DESC;
```

### "What's associated with this visual node?" (cross-modal fan-out)
```sql
-- Find everything that co-occurs with visual node 17
SELECT unit_b_id, modality_b, count
FROM sensory_cooccurrences
WHERE unit_a_id = 17 AND modality_a = 'visual' AND count > 0
UNION ALL
SELECT unit_a_id, modality_a, count
FROM sensory_cooccurrences
WHERE unit_b_id = 17 AND modality_b = 'visual' AND count > 0
ORDER BY count DESC;
```

### "Do these two smells co-occur?" (within-modality)
```sql
SELECT count FROM sensory_cooccurrences
WHERE unit_a_id = $1 AND modality_a = 'olfactory'
  AND unit_b_id = $2 AND modality_b = 'olfactory';
```

### Dreamstate decay (all modalities, unified)
```sql
UPDATE sensory_cooccurrences
SET count = GREATEST(1, CAST(count * 0.97 AS INT)),
    dreamstate_challenges = dreamstate_challenges + 1
WHERE NOT myelinated AND count > 1
  AND last_seen < NOW() - INTERVAL '1 hour';
```

## Olfactory Sensor Design (BME690 + Grove)

### Sensor readings
- BME690 (×8 via Shuttle Board): temperature, humidity, pressure, gas_resistance (VOC)
- Grove Multichannel: CO, NO2, C2H5OH (ethanol), H2, NH3, CH4

### Olfactory embedding (hand-crafted, like depth)
```
BME690 array:    8 sensors × 4 readings = 32-dim
Grove channels:  6 gas types × 1 reading = 6-dim
Temporal delta:  rate of change for each = 38-dim
Normalised:      76-dim total → pad to 128-dim
```

### Olfactory units (like sound units)
- Cluster similar olfactory embeddings into units
- Each unit represents a "smell type" (coffee, cooking, fresh air, etc.)
- STT-equivalent labelling: no automatic labels initially, human can label via dashboard

### Binding
```python
# When processing a frame with olfactory data:
await cooc_store.observe(visual_node_id, "visual", smell_unit_id, "olfactory", pool)
await cooc_store.observe(sound_unit_id, "voice", smell_unit_id, "olfactory", pool)
# Coffee smell + brown visual + grinding sound = coffee concept
```

## Key Design Principles

1. **Any sensor to any sensor.** No hardcoded modality pairs.
2. **Canonical ordering prevents duplicates.** (A,B) and (B,A) are the same entry.
3. **Same decay/myelination for all modalities.** Unified dreamstate processing.
4. **New sensors = new modality string.** No schema changes, no code changes to the store.
5. **Cross-modal recall is a single query.** "What do all my senses know about this?"
6. **Backward compatible.** Migration can be gradual — old tables work alongside new.

## Verification

1. **Migration test:** Migrate existing data, verify counts match old tables
2. **Observe test:** Record voice↔visual, motor↔voice, then query both sides
3. **Canonical ordering:** Verify (A,B) and (B,A) produce same row
4. **Decay test:** Run homeostatic decay on unified table, verify grace period works
5. **Self-play:** Round-trip test works across unified table
6. **Dashboard:** All pages work with unified queries
7. **Future sensor:** Add a mock "olfactory" modality, verify it binds to visual+sound
