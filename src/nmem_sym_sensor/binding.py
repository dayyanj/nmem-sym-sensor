"""
Cross-modal temporal binding.

When visual and audio events occur within the same temporal window,
they become binding candidates. Repeated co-occurrence creates
cross-modal edges (bound_to) that link visual and audio primitives.

This is how the system learns that a "clinking" sound goes with
"cylindrical-glass-smooth" shapes — not through labeled data,
but through experiencing them together repeatedly.
"""
import logging
from datetime import UTC, datetime

import asyncpg

from nmem_sym_sensor import config
from nmem_sym_sensor.graph import record_cooccurrence, upsert_edge

log = logging.getLogger(__name__)


async def bind_frame_audio(
    pool: asyncpg.Pool,
    visual_node_ids: list[int],
    audio_node_ids: list[int],
    timestamp: datetime | None = None,
) -> list[int]:
    """Record temporal co-occurrence between visual and audio nodes.

    Called when visual and audio observations happen within the
    binding window (BINDING_WINDOW_MS). Each pair gets a co-occurrence
    bump. When the count reaches BINDING_MIN_COOCCURRENCES, a
    cross-modal edge is created.

    Args:
        pool: Database connection pool.
        visual_node_ids: Sensory node IDs from visual analysis.
        audio_node_ids: Sensory node IDs from audio analysis.
        timestamp: When the binding occurred. Defaults to now.

    Returns:
        List of newly created cross-modal edge IDs.
    """
    ts = timestamp or datetime.now(UTC)
    new_edges = []

    for v_id in visual_node_ids:
        for a_id in audio_node_ids:
            count = await record_cooccurrence(pool, v_id, a_id)

            if count >= config.BINDING_MIN_COOCCURRENCES:
                # Check if edge already exists
                existing = await pool.fetchval(
                    """
                    SELECT id FROM sensory_edges
                    WHERE source_id = $1 AND target_id = $2 AND edge_type = 'bound_to'
                    """,
                    v_id, a_id,
                )

                if existing is None:
                    # Create cross-modal binding edge
                    confidence = min(count / (config.BINDING_MIN_COOCCURRENCES * 3), 1.0)
                    edge_id = await upsert_edge(
                        pool,
                        source_id=v_id,
                        target_id=a_id,
                        edge_type="bound_to",
                        confidence=confidence,
                        source_ref={"type": "temporal_binding", "timestamp": ts.isoformat()},
                    )
                    new_edges.append(edge_id)
                    log.info("Cross-modal binding: visual %d ↔ audio %d (count=%d)",
                            v_id, a_id, count)

    return new_edges


async def bind_within_modality(
    pool: asyncpg.Pool,
    node_ids: list[int],
    timestamp: datetime | None = None,
) -> list[int]:
    """Record intra-modal co-occurrence for nodes observed in the same frame.

    Visual primitives from the same frame, or audio primitives from the
    same window, co-occur temporally. This drives within-modality
    clustering (shapes + colors + textures that appear together).

    Args:
        pool: Database connection pool.
        node_ids: Sensory node IDs from a single frame/window.
        timestamp: When the observation occurred.

    Returns:
        List of newly created co_occurs_with edge IDs.
    """
    ts = timestamp or datetime.now(UTC)
    new_edges = []

    for i, a_id in enumerate(node_ids):
        for b_id in node_ids[i + 1:]:
            count = await record_cooccurrence(pool, a_id, b_id)

            if count >= config.BINDING_MIN_COOCCURRENCES:
                existing = await pool.fetchval(
                    """
                    SELECT id FROM sensory_edges
                    WHERE source_id = $1 AND target_id = $2 AND edge_type = 'co_occurs_with'
                    """,
                    min(a_id, b_id), max(a_id, b_id),
                )

                if existing is None:
                    confidence = min(count / (config.BINDING_MIN_COOCCURRENCES * 3), 1.0)
                    edge_id = await upsert_edge(
                        pool,
                        source_id=min(a_id, b_id),
                        target_id=max(a_id, b_id),
                        edge_type="co_occurs_with",
                        confidence=confidence,
                        source_ref={"type": "frame_cooccurrence", "timestamp": ts.isoformat()},
                    )
                    new_edges.append(edge_id)

    return new_edges


async def get_binding_candidates(
    pool: asyncpg.Pool,
    min_count: int | None = None,
) -> list[dict]:
    """Return co-occurrence pairs that are close to binding threshold.

    Useful for diagnostics: shows what associations are forming
    but haven't yet solidified into edges.
    """
    threshold = min_count or config.BINDING_MIN_COOCCURRENCES

    rows = await pool.fetch(
        """
        SELECT
            co.node_a, co.node_b, co.count, co.last_seen,
            a.label as label_a, a.modality as modality_a,
            b.label as label_b, b.modality as modality_b
        FROM sensory_cooccurrences co
        JOIN sensory_nodes a ON a.id = co.node_a
        JOIN sensory_nodes b ON b.id = co.node_b
        WHERE co.count >= $1
        ORDER BY co.count DESC
        LIMIT 100
        """,
        max(1, threshold - 2),  # show near-threshold pairs too
    )

    return [
        {
            "node_a": r["node_a"],
            "node_b": r["node_b"],
            "label_a": r["label_a"],
            "label_b": r["label_b"],
            "modality_a": r["modality_a"],
            "modality_b": r["modality_b"],
            "count": r["count"],
            "last_seen": r["last_seen"].isoformat(),
            "is_cross_modal": r["modality_a"] != r["modality_b"],
            "above_threshold": r["count"] >= threshold,
        }
        for r in rows
    ]
