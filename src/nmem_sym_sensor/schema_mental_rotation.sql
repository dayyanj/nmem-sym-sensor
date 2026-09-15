-- Dreamstate Mental Rotation schema additions.
-- Run after schema.sql to add rotation consolidation support.

-- Record each rotation test and its outcome
CREATE TABLE IF NOT EXISTS dreamstate_rotations (
    id                    BIGSERIAL PRIMARY KEY,
    cluster_id            INT NOT NULL,
    member_a_id           BIGINT NOT NULL,   -- node with orientation A
    member_b_id           BIGINT NOT NULL,   -- node with orientation B
    orientation_a         TEXT,               -- vertical/horizontal/square
    orientation_b         TEXT,
    embedding_similarity  FLOAT,             -- cosine sim of raw embeddings
    rotated_similarity    FLOAT,             -- cosine sim after virtual rotation
    similarity_gain       FLOAT,             -- rotated - raw (positive = rotation helps)
    verdict               TEXT NOT NULL,      -- 'invariant' | 'sensitive' | 'inconclusive'
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drot_cluster
    ON dreamstate_rotations (cluster_id);
CREATE INDEX IF NOT EXISTS idx_drot_created
    ON dreamstate_rotations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_drot_verdict
    ON dreamstate_rotations (verdict);

-- Track rotation invariance per co-occurrence pair
ALTER TABLE sensory_cooccurrences
    ADD COLUMN IF NOT EXISTS orientation_independence INT DEFAULT 0;
ALTER TABLE sensory_cooccurrences
    ADD COLUMN IF NOT EXISTS orientation_dependence INT DEFAULT 0;
