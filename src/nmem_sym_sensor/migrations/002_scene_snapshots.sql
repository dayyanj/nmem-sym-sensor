-- nmem-sym-sensor migration 002: Chronoception scene memory
-- PostgreSQL + pgvector
--
-- The Chronoception subsystem (scene-gated activation, return-visit detection) reads/writes three
-- `scene_*` tables that existed only in design docs — no schema or migration ever created them, so
-- FAMILIARITY_ENABLED (default on) silently failed on every frame (`relation "scene_snapshots"
-- does not exist`, caught + logged). This migration creates them so scene recognition works —
-- which is what visual-memory Phase 2 read-back leans on ("I have seen a screen like this before").
--
-- Columns/constraints mirror the ACTUAL queries in chronoception.py (the design-doc DDL is stale:
-- different column names + a 1024-d vector). The scene embedding is the coarse 8x8 COLOR embedding
-- (COARSE_SIZE=8 → 8*8*3 = 192-d, L2-normalized) produced by SceneMemory._coarse_embedding.

CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- SCENE SNAPSHOTS: a recognizable whole-frame "place"
-- ============================================================
CREATE TABLE IF NOT EXISTS scene_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    embedding       vector(192),                     -- coarse 8x8 color scene embedding (normalized)
    visit_count     INT NOT NULL DEFAULT 1,
    first_seen      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- KNN recognition: ORDER BY embedding <=> $1 LIMIT 1 (cosine). Mirrors the sensory_nodes index style.
CREATE INDEX IF NOT EXISTS idx_scene_snapshots_emb
    ON scene_snapshots USING hnsw (embedding vector_cosine_ops);


-- ============================================================
-- SCENE MEMBERS: which grounded clusters live in a scene
-- ============================================================
CREATE TABLE IF NOT EXISTS scene_members (
    scene_id            BIGINT NOT NULL REFERENCES scene_snapshots(id) ON DELETE CASCADE,
    cluster_id          BIGINT NOT NULL,             -- FK to sensory_clusters.id (not enforced; may prune independently)
    observation_count   INT NOT NULL DEFAULT 1,
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scene_id, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_scene_members_scene ON scene_members (scene_id);


-- ============================================================
-- SCENE SPATIAL EDGES: allocentric relations between clusters in a scene
-- ============================================================
CREATE TABLE IF NOT EXISTS scene_spatial_edges (
    scene_id            BIGINT NOT NULL REFERENCES scene_snapshots(id) ON DELETE CASCADE,
    cluster_a_id        BIGINT NOT NULL,
    cluster_b_id        BIGINT NOT NULL,
    relation            TEXT NOT NULL,               -- allocentric only (near, above, below, on, contains, …)
    confidence          REAL NOT NULL DEFAULT 0.5,
    observation_count   INT NOT NULL DEFAULT 1,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scene_id, cluster_a_id, cluster_b_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_scene_spatial_edges_scene ON scene_spatial_edges (scene_id);
