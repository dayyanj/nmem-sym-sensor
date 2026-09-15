-- Dreamstate Recombination schema additions.
-- Run after schema.sql to add recombination support.

-- New table: record each recombination attempt and its outcome
CREATE TABLE IF NOT EXISTS dreamstate_recombinations (
    id                    BIGSERIAL PRIMARY KEY,
    candidate_cluster_id  INT NOT NULL,
    candidate_home_scene_id INT,
    target_scene_id       INT NOT NULL,
    spatial_compatibility FLOAT,
    n_reinforcing         INT DEFAULT 0,
    n_conflicting         INT DEFAULT 0,
    n_novel               INT DEFAULT 0,
    coherence_score       FLOAT,  -- reinforcing / (reinforcing + conflicting)
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drecom_candidate
    ON dreamstate_recombinations (candidate_cluster_id);
CREATE INDEX IF NOT EXISTS idx_drecom_created
    ON dreamstate_recombinations (created_at DESC);

-- New columns on sensory_cooccurrences: track context universality
ALTER TABLE sensory_cooccurrences
    ADD COLUMN IF NOT EXISTS context_independence INT DEFAULT 0;
ALTER TABLE sensory_cooccurrences
    ADD COLUMN IF NOT EXISTS context_dependence INT DEFAULT 0;
