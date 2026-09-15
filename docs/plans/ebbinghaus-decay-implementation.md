# Ebbinghaus Decay Implementation Plan

## Overview

Replace the flat 3% decay with a biologically-inspired forgetting model where:
- Each co-occurrence has its own decay rate based on reinforcement history
- Observation quality (attention, surprise, proximity) modulates encoding strength
- Half-life increases logarithmically with reinforcement count
- Myelination emerges naturally when half-life exceeds dreamstate cycle interval

## Equations

### Observation (Encoding)

Observation weight:
    w_i = alpha_i * sigma_i * pi_i

Where:
- alpha_i = attention level [0,1] — foveal=1.0, peripheral=0.1, monitoring=0.05
- sigma_i = surprise [0,1] — novel=1.0, predicted=0.0
- pi_i = proximity factor [0,1] — in hull=1.0, near=exp(-d/200), distant=0.0

Temporal proximity (soft binding window):
    pi_temporal = exp(-|dt| / tau_bind)    where tau_bind ~ 2s

Combined spacetime proximity:
    pi_i = pi_visual * pi_temporal

Strength update on observation:
    S_new = S_old * D(dt) + w_i

Encoding quality (running average):
    q_new = ((n-1) * q_old + w_i) / n

### Decay (Forgetting)

Half-life:
    tau = tau_0 * (1 + ln(1 + n)) * (0.5 + q_bar)

    tau_0 = 300s base half-life
    n = reinforcement count
    q_bar = mean encoding quality

Decay function:
    D(dt) = 2^(-dt / tau)

Strength at time t:
    S(t) = S(t_last) * 2^(-dt / tau)

### Myelination

Natural emergence when half-life exceeds maximum cycle interval:
    myelinated = (tau > 86400) AND (n > N_min)

Computationally: skip decay calculation for myelinated pairs.

## Schema Changes

### Add columns to sensory_cooccurrences

```sql
ALTER TABLE sensory_cooccurrences
    ADD COLUMN strength FLOAT NOT NULL DEFAULT 1.0,
    ADD COLUMN reinforcement_count INT NOT NULL DEFAULT 1,
    ADD COLUMN encoding_quality FLOAT NOT NULL DEFAULT 0.5,
    ADD COLUMN half_life FLOAT NOT NULL DEFAULT 300.0,
    ADD COLUMN last_reinforced TIMESTAMPTZ NOT NULL DEFAULT NOW();
```

### Migrate existing data

Existing rows have `count` (integer observations) but no quality/strength data.
Bootstrap from count:

```sql
UPDATE sensory_cooccurrences SET
    strength = count::float,
    reinforcement_count = count,
    encoding_quality = 0.5,
    half_life = 300.0 * (1 + ln(1 + count)) * (0.5 + 0.5),
    last_reinforced = last_seen;
```

The `count` column remains as a simple diagnostic counter.

## Code Changes

### Phase 1: Schema + CooccurrenceStore.observe()

**File: cooccurrence.py**

1. Update `observe()` signature:
   ```python
   async def observe(
       self,
       unit_a: int, modality_a: str,
       unit_b: int, modality_b: str,
       pool: asyncpg.Pool,
       attention: float = 1.0,    # alpha
       surprise: float = 0.5,     # sigma
       proximity: float = 1.0,    # pi
   ):
   ```

2. Compute observation weight: `w = attention * surprise * proximity`

3. Update SQL to use strength-based model:
   ```sql
   INSERT INTO sensory_cooccurrences
       (unit_a_id, modality_a, unit_b_id, modality_b,
        count, strength, reinforcement_count, encoding_quality, half_life, last_reinforced)
   VALUES ($1, $2, $3, $4,
        1, $5, 1, $5, $6, NOW())
   ON CONFLICT (unit_a_id, modality_a, unit_b_id, modality_b)
   DO UPDATE SET
       count = sensory_cooccurrences.count + 1,
       -- Apply decay to existing strength, then add new evidence
       strength = sensory_cooccurrences.strength
                  * POWER(2, -EXTRACT(EPOCH FROM NOW() - sensory_cooccurrences.last_reinforced)
                          / GREATEST(sensory_cooccurrences.half_life, 1))
                  + $5,
       reinforcement_count = sensory_cooccurrences.reinforcement_count + 1,
       encoding_quality = ((sensory_cooccurrences.reinforcement_count * sensory_cooccurrences.encoding_quality) + $5)
                          / (sensory_cooccurrences.reinforcement_count + 1),
       -- Recompute half-life from updated reinforcement count and quality
       half_life = $7 * (1 + LN(1 + sensory_cooccurrences.reinforcement_count + 1))
                   * (0.5 + ((sensory_cooccurrences.reinforcement_count * sensory_cooccurrences.encoding_quality) + $5)
                          / (sensory_cooccurrences.reinforcement_count + 1)),
       last_reinforced = NOW(),
       last_seen = NOW()
   ```

   Parameters: $5 = weight, $6 = initial_half_life, $7 = tau_0

4. Update `observe_batch()` to pass through quality parameters.

5. Add `observe_weighted()` convenience method that takes raw alpha/sigma/pi.

### Phase 2: Callers pass quality signals

**File: language.py — observe_sound()**

Currently calls `cooc_store.observe(unit_id, "voice", vid, "visual", pool)`

Change to pass attention and surprise:
```python
await cooc_store.observe(
    unit_id, "voice", vid, "visual", pool,
    attention=attention_level,    # from attention controller phase
    surprise=current_surprise,   # from surprise module
    proximity=temporal_proximity, # exp(-|dt| / 2.0)
)
```

Sources for each signal:
- attention_level: from AttentionController.phase (EXPLORE=1.0, INSPECT=0.7, MONITOR=0.3)
- surprise: from SurpriseSignal.current_surprise (already computed per frame)
- temporal_proximity: exp(-|burst_time - frame_time| / 2.0)

**File: oculomotor.py — observe_trajectory()**

Motor co-occurrences. Attention = 1.0 (saccades are always intentional).
Surprise = novelty of the trajectory pattern.

**File: ingest/video.py**

Thread attention_level and surprise through to the observe calls.
The frame loop already has:
- `frame_analysis.fixations` → attention level available
- `surprise.current_surprise` → surprise available
- `burst_time - t` → temporal proximity available

### Phase 3: Dreamstate decay update

**File: sensory_dreamstate.py — _homeostatic_decay()**

Replace flat multiplier with per-row Ebbinghaus decay:

```sql
UPDATE sensory_cooccurrences
SET strength = strength * POWER(2, -$1 / GREATEST(half_life, 1))
WHERE NOT myelinated AND strength > 0;
```

Where $1 = elapsed seconds since last dreamstate cycle.

Archive when strength drops below threshold:
```sql
UPDATE sensory_cooccurrences
SET count = 0
WHERE strength < 0.01 AND NOT myelinated;
```

Check myelination:
```sql
UPDATE sensory_cooccurrences
SET myelinated = TRUE
WHERE half_life > 86400
  AND reinforcement_count > 50
  AND NOT myelinated;
```

### Phase 4: Query updates

**File: cooccurrence.py — query()**

Change `ORDER BY count DESC` to `ORDER BY strength DESC`.
The strength reflects both observation count AND recency — a pair seen
100 times last week ranks higher than one seen 200 times last month.

**File: dashboard.py**

Show strength alongside count. Add half-life and encoding quality to
the bindings view.

## Testing Plan

### Unit tests (cooccurrence.py)

1. Fresh observation: strength = weight, reinforcement_count = 1
2. Second observation: strength = decayed_old + new_weight
3. Observation after long gap: old strength heavily decayed
4. Observation after short gap: old strength barely decayed
5. High attention obs produces higher strength than low attention
6. Half-life increases with reinforcement count
7. Verify canonical ordering preserved

### Decay tests (dreamstate)

1. Row with reinforcement_count=1: decays fast (half-life ~300s)
2. Row with reinforcement_count=100: decays slowly (half-life ~2000s)
3. Row with reinforcement_count=500 + high quality: barely decays
4. Myelination triggers at correct thresholds
5. Myelinated rows skipped by decay
6. Dormant archival at strength < 0.01

### Integration tests

1. Run 2 watches of primitives
2. Verify strength values increase with repeated observations
3. Verify half-lives differ between frequently-seen and rarely-seen pairs
4. Run dreamstate manually — verify differential decay
5. Compare ranking: strength-ordered vs count-ordered recall

### Backward compatibility

- `count` column preserved (simple diagnostic counter)
- Old code paths that read `count` still work
- Dashboard shows both count and strength

## Migration for existing data

Existing rows have integer counts but no strength/quality data.
The bootstrap migration sets:
- strength = count (treat each past observation as weight 1.0)
- reinforcement_count = count
- encoding_quality = 0.5 (assume moderate quality for historical data)
- half_life = computed from count and quality
- last_reinforced = last_seen

This means existing strong associations (count=2000) get long half-lives
and resist decay naturally. Weak associations (count=2) get short
half-lives and decay quickly. The system self-corrects from the existing
data without needing a full restart.

## Constants

```python
TAU_0 = 300.0              # Base half-life (seconds of dreamstate time)
TAU_MYELINATE = 86400.0     # Half-life threshold for myelination (1 day)
N_MIN_MYELINATE = 50        # Minimum reinforcements for myelination
STRENGTH_ARCHIVE = 0.01     # Archive threshold
TAU_BIND = 2.0              # Temporal binding window (seconds)
```

## Execution Order

1. Schema migration (add columns, bootstrap existing data)
2. CooccurrenceStore.observe() — strength-based with quality params
3. Callers (language.py, oculomotor.py, video.py) — pass quality signals
4. Dreamstate decay — Ebbinghaus curve
5. Query ordering — strength-based ranking
6. Dashboard — show new fields
7. Full learning run + recall test
