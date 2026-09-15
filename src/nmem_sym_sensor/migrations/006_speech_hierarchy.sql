-- nmem-sym-sensor migration 006: speech hierarchy (syllable chunking)
--
-- Folds the tables previously lazy-created by language.init_speech_hierarchy() into a versioned
-- migration, so the schema has a single authoritative source (H3a of the hardening plan). These are
-- self-contained (no cross-table FK outside this pair), so they migrate cleanly now.
--
-- The audio motor-production tables (motor_programs) and the core `sound_units` table it references
-- are NOT folded here: `sound_units` has no DDL anywhere in the repo yet (the audio path is dormant —
-- no agent has audio). Defining sound_units + migrating motor_programs is audio bring-up (H3b).
--
-- language.init_speech_hierarchy() remains as an idempotent belt for standalone use without the nmem
-- migration runner; this migration is the source of truth. DDL is byte-identical to that function.

CREATE TABLE IF NOT EXISTS speech_syllables (
    id                      SERIAL PRIMARY KEY,
    unit_ids                INTEGER[] NOT NULL,
    exemplar_frames         BYTEA,
    exemplar_duration_ms    FLOAT,
    confidence              FLOAT DEFAULT 0.0,
    half_life               FLOAT DEFAULT 300.0,
    self_play_confirmations INTEGER DEFAULT 0,
    myelinated              BOOLEAN DEFAULT FALSE,
    stt_label               TEXT,
    observation_count       INTEGER DEFAULT 0,
    total_duration_ms       FLOAT DEFAULT 0.0,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_syllables_unit_ids
    ON speech_syllables (unit_ids);
CREATE INDEX IF NOT EXISTS idx_syllables_first_unit
    ON speech_syllables ((unit_ids[1]));
CREATE INDEX IF NOT EXISTS idx_syllables_myelinated
    ON speech_syllables (myelinated) WHERE myelinated = TRUE;

CREATE TABLE IF NOT EXISTS syllable_sequences (
    id               SERIAL PRIMARY KEY,
    from_syllable_id INTEGER REFERENCES speech_syllables(id) ON DELETE CASCADE,
    to_syllable_id   INTEGER REFERENCES speech_syllables(id) ON DELETE CASCADE,
    count            INTEGER DEFAULT 1,
    confidence       FLOAT DEFAULT 0.0,
    gap_ms_avg       FLOAT DEFAULT 0.0,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (from_syllable_id, to_syllable_id)
);
