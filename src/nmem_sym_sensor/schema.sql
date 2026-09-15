-- ⚠️ SUPERSEDED by migrations/001_initial_schema.sql (this file's content) — do NOT hand-run.
-- Schema is now migration-managed via nmem.migrate (project 'nmem-sym-sensor'); the appliance
-- runs it at DB init. Kept only as reference for the legacy schema_{mental_rotation,recombination}.sql
-- extensions until those are folded into numbered migrations.
-- nmem-sym-sensor: Sensory Memory Graph Schema
-- PostgreSQL + pgvector
--
-- Three-tier sensory memory: iconic (buffer) → short-term → long-term nodes.
-- Visual and audio primitives share the same node/edge tables with typed
-- embeddings per modality.

CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- SENSORY NODES: Visual and audio primitives
-- ============================================================
CREATE TABLE IF NOT EXISTS sensory_nodes (
    id              BIGSERIAL PRIMARY KEY,

    -- Identity
    label           TEXT NOT NULL,                  -- auto-generated descriptor: "red-circle", "500hz-burst"
    node_type       TEXT NOT NULL,                  -- shape | color | texture | spatial_rel | frequency | rhythm | ...
    modality        TEXT NOT NULL,                  -- 'visual' | 'audio'
    normalized_label TEXT NOT NULL,                 -- lowercase, stripped — for dedup

    -- Embeddings (per modality — not all columns populated)
    text_embedding  vector(384),                    -- label embedding (same space as nmem-sym)
    visual_embedding vector(512),                   -- visual feature vector (NULL for audio nodes)
    audio_embedding vector(128),                    -- audio feature vector (NULL for visual nodes)
    embedder_id     TEXT,                           -- vector-space tag (e.g. dualstream_geo320_jepa192v2);
                                                    -- similarity is only valid WITHIN one embedder_id (JEPA on/off, model swaps differ)

    -- Provenance
    groundedness    INTEGER NOT NULL DEFAULT 0,     -- observation count
    source_refs     JSONB NOT NULL DEFAULT '[]',    -- [{frame_id, timestamp, source}]

    -- Memory tier lifecycle
    memory_tier     TEXT NOT NULL DEFAULT 'iconic', -- iconic | short_term | long_term
    tier_promoted_at TIMESTAMPTZ,                   -- when promoted to current tier
    observation_count INTEGER NOT NULL DEFAULT 1,   -- raw observation count (for promotion decisions)

    -- Activation dynamics
    salience        FLOAT NOT NULL DEFAULT 1.0,
    activation_count INTEGER NOT NULL DEFAULT 0,
    last_activated  TIMESTAMPTZ,
    firing_threshold FLOAT NOT NULL DEFAULT 0.15,   -- lower than nmem-sym: sensory nodes fire easily

    -- Feature descriptors (modality-specific, stored as JSONB)
    features        JSONB NOT NULL DEFAULT '{}',    -- e.g. {"area": 0.12, "aspect_ratio": 1.3, "hue": 0.6}

    -- Lifecycle
    archived        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ                     -- iconic nodes expire quickly
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_label ON sensory_nodes (normalized_label);
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_type ON sensory_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_modality ON sensory_nodes (modality);
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_tier ON sensory_nodes (memory_tier) WHERE NOT archived;
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_salience ON sensory_nodes (salience DESC) WHERE NOT archived;
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_expires ON sensory_nodes (expires_at) WHERE expires_at IS NOT NULL;

-- Embedding indexes (per modality)
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_text_emb
    ON sensory_nodes USING hnsw (text_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_visual_emb
    ON sensory_nodes USING hnsw (visual_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE visual_embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_audio_emb
    ON sensory_nodes USING hnsw (audio_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE audio_embedding IS NOT NULL;

-- Dedup: same label + type + modality = same node
CREATE UNIQUE INDEX IF NOT EXISTS idx_sensory_nodes_unique
    ON sensory_nodes (normalized_label, node_type, modality);

-- Active nodes only
CREATE INDEX IF NOT EXISTS idx_sensory_nodes_active
    ON sensory_nodes (id) WHERE NOT archived;


-- ============================================================
-- SENSORY EDGES: Relationships between primitives
-- ============================================================
CREATE TABLE IF NOT EXISTS sensory_edges (
    id              BIGSERIAL PRIMARY KEY,

    source_id       BIGINT NOT NULL REFERENCES sensory_nodes(id) ON DELETE CASCADE,
    target_id       BIGINT NOT NULL REFERENCES sensory_nodes(id) ON DELETE CASCADE,
    edge_type       TEXT NOT NULL,                  -- see config.SENSORY_EDGE_TYPES

    -- Activation dynamics
    weight          FLOAT NOT NULL DEFAULT 0.5,
    ltp_score       FLOAT NOT NULL DEFAULT 0.0,
    ltd_score       FLOAT NOT NULL DEFAULT 0.0,
    myelinated      BOOLEAN NOT NULL DEFAULT FALSE,

    -- Provenance
    confidence      FLOAT NOT NULL DEFAULT 0.5,
    groundedness    INTEGER NOT NULL DEFAULT 0,
    source_refs     JSONB NOT NULL DEFAULT '[]',

    -- Lifecycle
    last_traversed  TIMESTAMPTZ,
    traversal_count INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Traversal indexes
CREATE INDEX IF NOT EXISTS idx_sensory_edges_source ON sensory_edges (source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_sensory_edges_target ON sensory_edges (target_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_sensory_edges_weight ON sensory_edges (weight DESC);
CREATE INDEX IF NOT EXISTS idx_sensory_edges_myelinated ON sensory_edges (source_id) WHERE myelinated;
CREATE UNIQUE INDEX IF NOT EXISTS idx_sensory_edges_unique ON sensory_edges (source_id, target_id, edge_type);


-- ============================================================
-- CO-OCCURRENCES: Temporal binding candidates
-- ============================================================
-- Tracks how often two sensory primitives appear together within
-- the binding window. Drives cross-modal binding and cluster formation.
CREATE TABLE IF NOT EXISTS sensory_cooccurrences (
    node_a      BIGINT NOT NULL REFERENCES sensory_nodes(id) ON DELETE CASCADE,
    node_b      BIGINT NOT NULL REFERENCES sensory_nodes(id) ON DELETE CASCADE,
    count       INTEGER NOT NULL DEFAULT 1,
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (node_a, node_b)
);


-- ============================================================
-- SENSORY CLUSTERS: Consolidated concept candidates
-- ============================================================
-- When primitives consistently co-occur, they form a cluster.
-- Clusters that reach sufficient groundedness become concepts
-- eligible for grounding to nmem-sym symbolic nodes.
CREATE TABLE IF NOT EXISTS sensory_clusters (
    id              BIGSERIAL PRIMARY KEY,

    -- Identity
    label           TEXT,                           -- NULL until named (via grounding or manual label)
    modality        TEXT NOT NULL,                  -- 'visual' | 'audio' | 'cross_modal'
    cluster_type    TEXT NOT NULL DEFAULT 'proto',  -- proto | stable | grounded

    -- Taxonomy: parent-child hierarchy for concept refinement
    -- "blue" is parent of "sky blue" and "navy blue"
    parent_cluster_id BIGINT REFERENCES sensory_clusters(id) ON DELETE SET NULL,
    generation      INTEGER NOT NULL DEFAULT 0,     -- 0 = root, 1 = first split, etc.

    -- Centroid embedding (average of member embeddings)
    visual_centroid vector(512),
    embedder_id     TEXT,                           -- vector-space tag (must match member nodes' embedder_id)
    audio_centroid  vector(128),
    text_embedding  vector(384),                    -- set when grounded to a label

    -- Statistics
    member_count    INTEGER NOT NULL DEFAULT 0,
    total_observations INTEGER NOT NULL DEFAULT 0,  -- sum of member observation counts
    coherence       FLOAT NOT NULL DEFAULT 0.0,     -- intra-cluster similarity (0-1)
    coherence_at_formation FLOAT,                   -- coherence when cluster was created (drift detector)

    -- Grounding to nmem-sym (NULL until grounded)
    grounded_symbol_id  BIGINT,                     -- FK to nmem-sym symbol_nodes.id (not enforced, cross-db)
    grounded_label      TEXT,                       -- cached label from symbol graph
    grounding_confidence FLOAT,

    -- Rechallenge tracking
    observations_at_grounding INTEGER NOT NULL DEFAULT 0,   -- total_observations when label was assigned
    observations_at_last_challenge INTEGER NOT NULL DEFAULT 0, -- total_observations when last rechallenged
    challenge_count     INTEGER NOT NULL DEFAULT 0,         -- how many times this label has been challenged
    challenge_failures  INTEGER NOT NULL DEFAULT 0,         -- how many times LLM proposed a different label
    grounding_centroid_snapshot TEXT,                        -- centroid embedding at time of grounding (for drift)
    grounding_status    TEXT NOT NULL DEFAULT 'ungrounded',  -- ungrounded | speculative | confident | disputed

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sensory_clusters_type ON sensory_clusters (cluster_type);
CREATE INDEX IF NOT EXISTS idx_sensory_clusters_modality ON sensory_clusters (modality);

-- Cluster membership
CREATE TABLE IF NOT EXISTS sensory_cluster_members (
    cluster_id  BIGINT NOT NULL REFERENCES sensory_clusters(id) ON DELETE CASCADE,
    node_id     BIGINT NOT NULL REFERENCES sensory_nodes(id) ON DELETE CASCADE,
    role        TEXT NOT NULL DEFAULT 'member',     -- member | prototype (best exemplar)
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (cluster_id, node_id)
);


-- ============================================================
-- ICONIC BUFFER: Ultra-short-term sensory memory
-- ============================================================
-- Transient table for raw frame observations before graph insertion.
-- Rows expire after ICONIC_DECAY_SECONDS. A background sweep promotes
-- frequently-seen entries to sensory_nodes.
CREATE TABLE IF NOT EXISTS sensory_iconic_buffer (
    id              BIGSERIAL PRIMARY KEY,
    modality        TEXT NOT NULL,                  -- 'visual' | 'audio'
    node_type       TEXT NOT NULL,
    label           TEXT NOT NULL,
    features        JSONB NOT NULL DEFAULT '{}',
    visual_embedding vector(512),
    audio_embedding vector(128),
    embedder_id     TEXT,                           -- vector-space tag (carried to sensory_nodes on promote)
    frame_id        TEXT,                           -- source frame/window identifier
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL             -- NOW() + iconic_decay_seconds
);

CREATE INDEX IF NOT EXISTS idx_iconic_buffer_expires ON sensory_iconic_buffer (expires_at);
CREATE INDEX IF NOT EXISTS idx_iconic_buffer_label ON sensory_iconic_buffer (label, modality);


-- ============================================================
-- GROUNDING LOG: Links between sensory clusters and nmem-sym nodes
-- ============================================================
CREATE TABLE IF NOT EXISTS sensory_grounding_log (
    id              BIGSERIAL PRIMARY KEY,
    cluster_id      BIGINT NOT NULL REFERENCES sensory_clusters(id) ON DELETE CASCADE,
    symbol_node_id  BIGINT NOT NULL,                -- nmem-sym symbol_nodes.id (cross-db, not FK)
    symbol_label    TEXT NOT NULL,
    confidence      FLOAT NOT NULL,
    grounding_type  TEXT NOT NULL DEFAULT 'label_match',
        -- label_match | temporal_cooccurrence | manual | rechallenge_confirmed | rechallenge_revised | rechallenge_revoked
    previous_label  TEXT,                           -- for rechallenge: what label was replaced
    rechallenge_reason TEXT,                        -- drift | observation_interval | confidence_decay | manual
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_grounding_cluster ON sensory_grounding_log (cluster_id);
CREATE INDEX IF NOT EXISTS idx_grounding_symbol ON sensory_grounding_log (symbol_node_id);


-- ============================================================
-- LABEL CANDIDATES: Competing label hypotheses per cluster
-- ============================================================
-- Each time a speech label is heard for a cluster, it gets tallied here.
-- The incumbent label decays slightly on each challenge. When a challenger
-- accumulates enough evidence (tally >= 3, more than incumbent), it overtakes.
-- ============================================================
-- WORD-VISUAL CO-OCCURRENCES: Statistical grounding evidence
-- ============================================================
-- Every time a word is heard while visual nodes are active, a
-- co-occurrence is recorded. Over time, the word that most
-- consistently co-occurs with a visual node becomes its label.
-- No single observation creates a grounding — only accumulated evidence.
CREATE TABLE IF NOT EXISTS word_visual_cooccurrences (
    word        TEXT NOT NULL,
    node_id     BIGINT NOT NULL REFERENCES sensory_nodes(id) ON DELETE CASCADE,
    count       INTEGER NOT NULL DEFAULT 1,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (word, node_id)
);

CREATE INDEX IF NOT EXISTS idx_wvc_word ON word_visual_cooccurrences (word);
CREATE INDEX IF NOT EXISTS idx_wvc_node ON word_visual_cooccurrences (node_id);
CREATE INDEX IF NOT EXISTS idx_wvc_count ON word_visual_cooccurrences (count DESC);


CREATE TABLE IF NOT EXISTS sensory_label_candidates (
    cluster_id  BIGINT NOT NULL REFERENCES sensory_clusters(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    tally       INTEGER NOT NULL DEFAULT 1,         -- how many times this label was heard
    total_confidence FLOAT NOT NULL DEFAULT 0.0,    -- sum of STT confidence across hearings
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (cluster_id, label)
);
