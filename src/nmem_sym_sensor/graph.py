"""
Graph operations: neuron activation, edge creation, iconic buffer management.

NEURON ARCHITECTURE: Nodes are fixed neurons. Their embeddings are immutable
after creation — they represent ONE specific thing observed at birth. Memory
is the PATTERN of simultaneous activation (edges), not the content of a single
neuron. Complex concepts emerge from edge structure, not mega-nodes.

Rules:
  1. Neuron embeddings never change after creation
  2. Features are set at birth, never merged
  3. Observation count = activation count (how many times this neuron fired)
  4. Matching threshold is tight (0.95 visual, 0.90 audio) — only truly
     identical primitives share a neuron
  5. Edges bind co-active neurons — this IS the knowledge structure

Shared by encoders, consolidation, and activation modules.
"""
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

import asyncpg
import numpy as np

from nmem_sym_sensor import config

log = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dumps(obj):
    """json.dumps with numpy type support."""
    return json.dumps(obj, cls=_NumpyEncoder)


def normalize_label(label: str) -> str:
    """Kept for backward compat in bridge.py — returns lowercase label.
    NOT used for node identity anymore (UUID instead)."""
    return label.strip().lower()


# ── Iconic buffer ────────────────────────────────────────

async def buffer_observation(
    pool: asyncpg.Pool,
    modality: str,
    node_type: str,
    label: str,
    features: dict,
    embedding: list[float] | None = None,
    frame_id: str | None = None,
    embedder_id: str | None = None,
) -> int:
    """Insert a raw sensory observation into the iconic buffer.

    ``embedder_id`` tags the vector's space (which visual encoder produced it) and is carried
    verbatim onto the promoted ``sensory_nodes`` row — so cosine comparison only ever happens
    WITHIN one space. Returns the buffer entry ID.
    """
    expires_at = datetime.now(UTC) + timedelta(seconds=config.ICONIC_DECAY_SECONDS)

    emb_col = "visual_embedding" if modality == "visual" else "audio_embedding"
    emb_val = _dumps(embedding) if embedding else None

    row = await pool.fetchrow(
        f"""
        INSERT INTO sensory_iconic_buffer
            (modality, node_type, label, features, {emb_col}, frame_id, embedder_id, expires_at)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8)
        RETURNING id
        """,
        modality,
        node_type,
        label,
        _dumps(features),
        emb_val,
        frame_id,
        embedder_id,
        expires_at,
    )
    return row["id"]


async def sync_cluster_counts(pool: asyncpg.Pool) -> int:
    """Recount cluster members and delete empty clusters.

    Call after any operation that deletes from sensory_cluster_members
    without updating the parent cluster's member_count.

    Returns number of empty clusters deleted.
    """
    # Recount all clusters in one pass
    await pool.execute("""
        UPDATE sensory_clusters c SET member_count = sub.cnt
        FROM (
            SELECT cluster_id, COUNT(*) as cnt
            FROM sensory_cluster_members
            GROUP BY cluster_id
        ) sub
        WHERE c.id = sub.cluster_id AND c.member_count != sub.cnt
    """)

    # Delete clusters with zero members (not in the subquery above)
    result = await pool.execute("""
        DELETE FROM sensory_clusters
        WHERE id NOT IN (SELECT DISTINCT cluster_id FROM sensory_cluster_members)
    """)
    deleted = int(result.split()[-1])
    if deleted > 0:
        log.info("Pruned %d empty clusters", deleted)
    return deleted


async def flush_expired_iconic(pool: asyncpg.Pool) -> int:
    """Delete expired iconic buffer entries. Returns count deleted."""
    result = await pool.execute(
        "DELETE FROM sensory_iconic_buffer WHERE expires_at < NOW()"
    )
    count = int(result.split()[-1])
    if count > 0:
        log.debug("Flushed %d expired iconic entries", count)
    return count


async def promote_from_iconic(pool: asyncpg.Pool) -> list[int]:
    """Activate or create neurons from iconic buffer entries.

    Each buffer entry independently either:
    - Activates an existing neuron (tight threshold match)
    - Creates a new neuron (no match found)

    No grouping, no merging. Each observation becomes a neuron activation.
    Returns list of activated/created node IDs.
    """
    # Fetch all non-expired buffer entries
    entries = await pool.fetch(
        """
        SELECT id, label, modality, node_type, embedder_id,
               visual_embedding::text as v_emb,
               audio_embedding::text as a_emb,
               features::text as features_text
        FROM sensory_iconic_buffer
        WHERE expires_at >= NOW()
        ORDER BY id
        """
    )

    if not entries:
        return []

    activated_ids = []
    buf_ids_to_delete = []

    for e in entries:
        emb_str = e["v_emb"] if e["modality"] == "visual" else e["a_emb"]
        emb = None
        if emb_str:
            try:
                emb_str = emb_str.strip("[]")
                if emb_str:
                    emb = np.array([float(x) for x in emb_str.split(",")], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb = emb / norm
            except (ValueError, TypeError):
                emb = None

        if emb is None:
            buf_ids_to_delete.append(e["id"])
            continue

        feat = {}
        if e["features_text"]:
            try:
                feat = json.loads(e["features_text"])
            except (ValueError, TypeError):
                pass

        node_id, is_new = await activate_or_create_neuron(
            pool,
            embedding=emb.tolist(),
            modality=e["modality"],
            node_type=e["node_type"],
            label=e["label"],
            features=feat,
            embedder_id=e["embedder_id"],
        )
        activated_ids.append(node_id)
        buf_ids_to_delete.append(e["id"])

    # Clean up processed buffer entries
    if buf_ids_to_delete:
        await pool.execute(
            "DELETE FROM sensory_iconic_buffer WHERE id = ANY($1)",
            buf_ids_to_delete,
        )

    return activated_ids


# ── Sensory neurons ────────────────────────────────────────

async def activate_or_create_neuron(
    pool: asyncpg.Pool,
    embedding: list[float],
    modality: str,
    node_type: str,
    label: str,
    features: dict | None = None,
    embedder_id: str | None = None,
) -> tuple[int, bool]:
    """Activate an existing neuron or create a new one.

    Returns (node_id, is_new). ``embedder_id`` tags a NEWLY-created neuron's vector space (an
    activated existing neuron keeps its birth tag — embeddings are immutable).

    Search: cosine similarity against all neurons of same type+modality.
    If best match >= threshold: ACTIVATE (increment count, record time).
    Otherwise: CREATE new neuron with fixed embedding and features.

    NEVER modifies embedding or features of existing neurons.
    Embeddings are immutable after creation. Features are set at birth.
    """
    features = dict(features or {})
    if "description" not in features:
        features["description"] = label

    emb_col = "visual_embedding" if modality == "visual" else "audio_embedding"
    emb_val = _dumps(embedding)

    # Search for matching neuron
    threshold = config.NEURON_VISUAL_THRESHOLD if modality == "visual" else config.NEURON_AUDIO_THRESHOLD

    # Only ever match WITHIN one vector space: a tag matches the same tag, and legacy NULL matches
    # NULL — a differently-tagged (JEPA on/off, model swap) neuron is NOT a valid neighbour even if
    # cosine-close, since the two 512-vecs aren't comparable. `IS NOT DISTINCT FROM` gives exactly
    # that (NULL=NULL true; tag=other-tag / tag=NULL false).
    match = await pool.fetchrow(
        f"""
        SELECT id, archived, 1 - ({emb_col} <=> $1::vector) as similarity
        FROM sensory_nodes
        WHERE node_type = $2
          AND modality = $3
          AND {emb_col} IS NOT NULL
          AND embedder_id IS NOT DISTINCT FROM $4
        ORDER BY {emb_col} <=> $1::vector
        LIMIT 1
        """,
        emb_val,
        node_type,
        modality,
        embedder_id,
    )

    if match and match["similarity"] >= threshold:
        # ACTIVATE existing neuron — only increment count + timestamp
        await pool.execute(
            """UPDATE sensory_nodes SET
                observation_count = observation_count + 1,
                archived = FALSE,
                memory_tier = CASE WHEN archived THEN 'short_term' ELSE memory_tier END,
                updated_at = NOW()
               WHERE id = $1""",
            match["id"],
        )
        if match["archived"]:
            log.debug("Reactivated archived neuron %d (sim=%.3f)", match["id"], match["similarity"])
        return match["id"], False

    # No match — create new neuron with fixed embedding
    node_uuid = str(uuid.uuid4())

    row = await pool.fetchrow(
        f"""
        INSERT INTO sensory_nodes
            (label, node_type, modality, normalized_label,
             {emb_col}, features, groundedness, observation_count,
             memory_tier, tier_promoted_at, source_refs, embedder_id)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, 1, 1, 'short_term', NOW(), '[]'::jsonb, $7)
        RETURNING id
        """,
        label,              # $1
        node_type,          # $2
        modality,           # $3
        node_uuid,          # $4
        emb_val,            # $5
        _dumps(features),   # $6
        embedder_id,        # $7
    )
    return row["id"], True


# Keep upsert_node as a thin wrapper for backward compatibility (bridge.py etc)
async def upsert_node(
    pool: asyncpg.Pool,
    label: str,
    node_type: str,
    modality: str,
    features: dict | None = None,
    embedding: list[float] | None = None,
    observation_count: int = 1,
    memory_tier: str = "short_term",
    source_ref: dict | None = None,
) -> int:
    """Backward-compatible wrapper around activate_or_create_neuron."""
    if embedding is not None:
        node_id, _ = await activate_or_create_neuron(
            pool, embedding, modality, node_type, label, features,
        )
        return node_id

    # No embedding — create with UUID (for non-sensory nodes like bridge symbols)
    features = dict(features or {})
    if "description" not in features:
        features["description"] = label
    node_uuid = str(uuid.uuid4())
    refs = _dumps([source_ref]) if source_ref else "[]"

    row = await pool.fetchrow(
        """
        INSERT INTO sensory_nodes
            (label, node_type, modality, normalized_label,
             features, groundedness, observation_count,
             memory_tier, tier_promoted_at, source_refs)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $6, $7, NOW(), $8::jsonb)
        RETURNING id
        """,
        label, node_type, modality, node_uuid,
        _dumps(features), observation_count, memory_tier, refs,
    )
    return row["id"]


async def upsert_edge(
    pool: asyncpg.Pool,
    source_id: int,
    target_id: int,
    edge_type: str,
    confidence: float = 0.5,
    source_ref: dict | None = None,
) -> int:
    """Insert or update a sensory edge. Returns edge ID."""
    refs = _dumps([source_ref]) if source_ref else "[]"

    row = await pool.fetchrow(
        """
        INSERT INTO sensory_edges
            (source_id, target_id, edge_type, weight, confidence, groundedness, source_refs)
        VALUES ($1, $2, $3, $4, $5, 1, $6::jsonb)
        ON CONFLICT (source_id, target_id, edge_type)
        DO UPDATE SET
            weight = GREATEST(sensory_edges.weight, $4),
            confidence = GREATEST(sensory_edges.confidence, $5),
            groundedness = sensory_edges.groundedness + 1,
            source_refs = sensory_edges.source_refs || $6::jsonb,
            updated_at = NOW()
        RETURNING id
        """,
        source_id, target_id, edge_type,
        confidence, confidence, refs,
    )
    return row["id"]


async def record_cooccurrence(
    pool: asyncpg.Pool,
    node_a: int,
    node_b: int,
) -> int:
    """Record a temporal co-occurrence between two sensory nodes.
    Returns updated count.
    """
    # Canonical ordering to avoid duplicates
    a, b = min(node_a, node_b), max(node_a, node_b)
    row = await pool.fetchrow(
        """
        INSERT INTO sensory_cooccurrences (node_a, node_b, count, last_seen)
        VALUES ($1, $2, 1, NOW())
        ON CONFLICT (node_a, node_b)
        DO UPDATE SET
            count = sensory_cooccurrences.count + 1,
            last_seen = NOW()
        RETURNING count
        """,
        a, b,
    )
    return row["count"]


# ── Threshold management ─────────────────────────────────

async def update_firing_thresholds(pool: asyncpg.Pool):
    """Recalculate firing thresholds based on node degree.

    Sensory nodes use lower base thresholds than nmem-sym since
    primitive pattern matching should be fast and permissive.
    """
    await pool.execute(f"""
        UPDATE sensory_nodes SET firing_threshold = LEAST(
            {config.SENSORY_MAX_THRESHOLD},
            {config.SENSORY_BASE_THRESHOLD} + (
                SELECT COUNT(*) FROM sensory_edges
                WHERE source_id = sensory_nodes.id OR target_id = sensory_nodes.id
            )::float / 100.0
        )
    """)


# ── Stats ────────────────────────────────────────────────

async def graph_stats(pool: asyncpg.Pool) -> dict:
    """Return current sensory graph statistics."""
    nodes = await pool.fetchval(
        "SELECT COUNT(*) FROM sensory_nodes WHERE NOT archived")
    edges = await pool.fetchval(
        "SELECT COUNT(*) FROM sensory_edges")
    iconic = await pool.fetchval(
        "SELECT COUNT(*) FROM sensory_iconic_buffer WHERE expires_at >= NOW()")
    clusters = await pool.fetchval(
        "SELECT COUNT(*) FROM sensory_clusters")
    grounded = await pool.fetchval(
        "SELECT COUNT(*) FROM sensory_clusters WHERE cluster_type = 'grounded'")

    tier_counts = await pool.fetch("""
        SELECT memory_tier, COUNT(*) as cnt
        FROM sensory_nodes WHERE NOT archived
        GROUP BY memory_tier
    """)
    tiers = {r["memory_tier"]: r["cnt"] for r in tier_counts}

    modality_counts = await pool.fetch("""
        SELECT modality, COUNT(*) as cnt
        FROM sensory_nodes WHERE NOT archived
        GROUP BY modality
    """)
    modalities = {r["modality"]: r["cnt"] for r in modality_counts}

    return {
        "nodes": nodes,
        "edges": edges,
        "iconic_buffer": iconic,
        "clusters": clusters,
        "grounded_clusters": grounded,
        "tiers": tiers,
        "modalities": modalities,
    }
