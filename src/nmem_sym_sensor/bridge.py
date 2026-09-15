"""
nmem-sym grounding bridge.

Links stable sensory clusters to the nmem-sym symbolic graph.
This is where sensory experience connects to language — a cluster
of visual primitives (cylinder + handle + smooth + brown) gets
grounded to the symbolic concept "cup" when they co-occur with
that text label enough times.

Two grounding mechanisms:
1. Label matching: Text embeddings of cluster member labels are compared
   to symbol node embeddings. High similarity → grounding edge.
2. Temporal co-occurrence: When a text mention (via nmem LTM) and a
   sensory cluster are active simultaneously, binding strengthens.

Zero import coupling with nmem-sym — uses duck-typing and cross-DB
references (symbol_node_id stored as plain BIGINT, not FK).
"""
import logging
from typing import TYPE_CHECKING

import asyncpg

from nmem_sym_sensor import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


async def ground_clusters_by_label(
    pool: asyncpg.Pool,
    sym_pool: asyncpg.Pool,
    embedder: "SentenceTransformer",
    limit: int = 50,
) -> list[dict]:
    """Ground stable sensory clusters to nmem-sym symbols via label similarity.

    For each stable (ungrounded) cluster, generate a text description from
    its member labels, embed it, and search for similar symbol nodes.

    Args:
        pool: Connection pool for sensory DB.
        sym_pool: Connection pool for nmem-sym DB.
        embedder: SentenceTransformer for text embedding.
        limit: Max clusters to process per call.

    Returns:
        List of grounding results.
    """
    # Fetch ungrounded stable clusters
    clusters = await pool.fetch(
        """
        SELECT c.id, c.modality, c.member_count, c.total_observations, c.coherence
        FROM sensory_clusters c
        WHERE c.cluster_type = 'stable'
          AND c.grounded_symbol_id IS NULL
        ORDER BY c.total_observations DESC
        LIMIT $1
        """,
        limit,
    )

    results = []

    for cluster in clusters:
        # Get member labels to build text description
        members = await pool.fetch(
            """
            SELECT n.label, n.node_type, n.features
            FROM sensory_cluster_members cm
            JOIN sensory_nodes n ON n.id = cm.node_id
            WHERE cm.cluster_id = $1
            ORDER BY n.groundedness DESC
            """,
            cluster["id"],
        )

        if not members:
            continue

        # Build a natural language description of what this cluster "looks like"
        descriptions = []
        for m in members:
            if m["node_type"] == "shape":
                descriptions.append(m["label"])
            elif m["node_type"] == "color":
                descriptions.append(f"{m['label']} colored")
            elif m["node_type"] == "texture":
                descriptions.append(f"{m['label']} surface")
            elif m["node_type"] in ("frequency", "timbre"):
                descriptions.append(m["label"])
            else:
                descriptions.append(m["label"])

        cluster_text = ", ".join(descriptions[:10])  # cap description length
        cluster_emb = embedder.encode(cluster_text).tolist()

        # Search nmem-sym for similar symbol nodes
        matches = await sym_pool.fetch(
            """
            SELECT id, label, node_type,
                   1 - (embedding <=> $1::vector) as similarity
            FROM symbol_nodes
            WHERE NOT archived
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT 5
            """,
            str(cluster_emb),
        )

        best_match = None
        for match in matches:
            if match["similarity"] >= config.GROUNDING_LABEL_SIMILARITY:
                best_match = match
                break

        # If no matching symbol exists and creation is enabled, create one
        if not best_match and config.GROUNDING_CREATE_SYMBOLS:
            if (cluster["total_observations"] >= config.GROUNDING_CREATE_MIN_OBSERVATIONS
                    and cluster["coherence"] >= config.GROUNDING_CREATE_MIN_COHERENCE):
                # Build a concise label from the cluster description
                symbol_label = _derive_symbol_label(descriptions)
                if symbol_label:
                    new_id = await _create_symbol_node(
                        sym_pool, symbol_label, cluster["modality"],
                        cluster_emb, cluster["id"],
                    )
                    if new_id:
                        best_match = {
                            "id": new_id,
                            "label": symbol_label,
                            "node_type": "entity",
                            "similarity": config.GROUNDING_MIN_CONFIDENCE,
                        }
                        log.info("Created new symbol node #%d '%s' from sensory cluster #%d",
                                new_id, symbol_label, cluster["id"])

        if best_match and best_match["similarity"] >= config.GROUNDING_MIN_CONFIDENCE:
            # Ground the cluster
            await pool.execute(
                """
                UPDATE sensory_clusters
                SET grounded_symbol_id = $1,
                    grounded_label = $2,
                    grounding_confidence = $3,
                    cluster_type = 'grounded',
                    text_embedding = $4,
                    updated_at = NOW()
                WHERE id = $5
                """,
                best_match["id"],
                best_match["label"],
                round(float(best_match["similarity"]), 4),
                str(cluster_emb),
                cluster["id"],
            )

            # Log the grounding
            await pool.execute(
                """
                INSERT INTO sensory_grounding_log
                    (cluster_id, symbol_node_id, symbol_label, confidence, grounding_type)
                VALUES ($1, $2, $3, $4, 'label_match')
                """,
                cluster["id"],
                best_match["id"],
                best_match["label"],
                round(float(best_match["similarity"]), 4),
            )

            result = {
                "cluster_id": cluster["id"],
                "modality": cluster["modality"],
                "cluster_description": cluster_text,
                "symbol_id": best_match["id"],
                "symbol_label": best_match["label"],
                "confidence": round(float(best_match["similarity"]), 4),
            }
            results.append(result)
            log.info("Grounded cluster #%d (%s) → symbol '%s' (%.3f)",
                    cluster["id"], cluster_text,
                    best_match["label"], best_match["similarity"])

    return results


async def ground_by_temporal_cooccurrence(
    pool: asyncpg.Pool,
    sym_pool: asyncpg.Pool,
    cluster_id: int,
    text_label: str,
    embedder: "SentenceTransformer",
) -> dict | None:
    """Ground a cluster when a text label co-occurs with its activation.

    Called by the application layer when a text input mentions a concept
    while sensory primitives from a cluster are currently active. This
    is the runtime grounding path — the system hears "that's a cup"
    while looking at a cluster of cylinder+handle+smooth primitives.

    Args:
        pool: Sensory DB pool.
        sym_pool: Symbol DB pool.
        cluster_id: The sensory cluster to ground.
        text_label: The text label heard/read at the same time.
        embedder: SentenceTransformer for embedding the label.

    Returns:
        Grounding result dict or None if no match found.
    """
    # Check if cluster exists and is ungrounded (or weakly grounded)
    cluster = await pool.fetchrow(
        """
        SELECT id, modality, grounding_confidence
        FROM sensory_clusters
        WHERE id = $1
        """,
        cluster_id,
    )
    if not cluster:
        return None

    # Find or create the symbol node for this label
    label_emb = embedder.encode(text_label).tolist()

    symbol = await sym_pool.fetchrow(
        """
        SELECT id, label, 1 - (embedding <=> $1::vector) as similarity
        FROM symbol_nodes
        WHERE NOT archived AND embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT 1
        """,
        str(label_emb),
    )

    if not symbol or symbol["similarity"] < config.GROUNDING_MIN_CONFIDENCE:
        log.debug("No matching symbol for '%s' (best=%.3f)",
                 text_label, symbol["similarity"] if symbol else 0)
        return None

    # Only overwrite if this grounding is stronger
    existing = cluster["grounding_confidence"]
    new_confidence = float(symbol["similarity"])
    if existing and existing >= new_confidence:
        return None

    # Ground it
    await pool.execute(
        """
        UPDATE sensory_clusters
        SET grounded_symbol_id = $1,
            grounded_label = $2,
            grounding_confidence = $3,
            cluster_type = 'grounded',
            text_embedding = $4,
            updated_at = NOW()
        WHERE id = $5
        """,
        symbol["id"],
        symbol["label"],
        round(new_confidence, 4),
        str(label_emb),
        cluster_id,
    )

    await pool.execute(
        """
        INSERT INTO sensory_grounding_log
            (cluster_id, symbol_node_id, symbol_label, confidence, grounding_type)
        VALUES ($1, $2, $3, $4, 'temporal_cooccurrence')
        """,
        cluster_id,
        symbol["id"],
        symbol["label"],
        round(new_confidence, 4),
    )

    result = {
        "cluster_id": cluster_id,
        "symbol_id": symbol["id"],
        "symbol_label": symbol["label"],
        "confidence": round(new_confidence, 4),
        "grounding_type": "temporal_cooccurrence",
    }
    log.info("Grounded cluster #%d → '%s' via temporal co-occurrence (%.3f)",
            cluster_id, symbol["label"], new_confidence)
    return result


async def augment_symbol_query(
    pool: asyncpg.Pool,
    symbol_node_id: int,
) -> str | None:
    """Augment a symbol graph query with sensory context.

    When nmem-sym activates a concept node, this function returns
    sensory descriptions of what that concept looks/sounds like,
    enriching the LLM context.

    Args:
        pool: Sensory DB pool.
        symbol_node_id: The nmem-sym symbol node being queried.

    Returns:
        Formatted sensory context string, or None if no sensory grounding.
    """
    cluster = await pool.fetchrow(
        """
        SELECT c.id, c.modality, c.grounded_label, c.coherence,
               c.member_count, c.total_observations
        FROM sensory_clusters c
        WHERE c.grounded_symbol_id = $1
          AND c.cluster_type = 'grounded'
        ORDER BY c.grounding_confidence DESC
        LIMIT 1
        """,
        symbol_node_id,
    )

    if not cluster:
        return None

    members = await pool.fetch(
        """
        SELECT n.label, n.node_type, n.features, cm.role
        FROM sensory_cluster_members cm
        JOIN sensory_nodes n ON n.id = cm.node_id
        WHERE cm.cluster_id = $1
        ORDER BY cm.role DESC, n.groundedness DESC
        LIMIT 8
        """,
        cluster["id"],
    )

    if not members:
        return None

    lines = [f"Sensory context for '{cluster['grounded_label']}':"]
    for m in members:
        prefix = "(prototype) " if m["role"] == "prototype" else ""
        lines.append(f"  {prefix}{m['node_type']}: {m['label']}")

    lines.append(f"  [{cluster['modality']}, {cluster['total_observations']} observations, "
                f"coherence={cluster['coherence']:.2f}]")

    return "\n".join(lines)


# ── Symbol creation helpers ──────────────────────────────


def _derive_symbol_label(descriptions: list[str]) -> str | None:
    """Derive a concise symbol label from cluster member descriptions.

    Prioritizes shape labels, then combines with dominant color.
    Returns None if no meaningful label can be derived.
    """
    shapes = [d for d in descriptions if not d.endswith("colored") and not d.endswith("surface")]
    colors = [d.replace(" colored", "") for d in descriptions if d.endswith("colored")]

    if shapes:
        label = shapes[0]
        if colors:
            label = f"{colors[0]} {label}"
        return label
    elif colors:
        return colors[0]
    elif descriptions:
        return descriptions[0]
    return None


async def _create_symbol_node(
    sym_pool: asyncpg.Pool,
    label: str,
    modality: str,
    embedding: list[float],
    source_cluster_id: int,
) -> int | None:
    """Create a new symbol node in nmem-sym from sensory grounding.

    Uses the same schema as nmem-sym's graph.upsert_node but with
    a sensory source reference instead of an LTM source.
    """
    import json

    norm = label.strip().lower()
    node_type = "entity"  # sensory-derived concepts are entities
    source_ref = [{"tier": "sensory", "cluster_id": source_cluster_id, "modality": modality}]

    try:
        row = await sym_pool.fetchrow(
            """
            INSERT INTO symbol_nodes (label, node_type, normalized_label, embedding, groundedness, source_ids)
            VALUES ($1, $2, $3, $4::vector, 1, $5::jsonb)
            ON CONFLICT (normalized_label, node_type)
            DO UPDATE SET
                groundedness = symbol_nodes.groundedness + 1,
                source_ids = symbol_nodes.source_ids || $5::jsonb,
                updated_at = NOW()
            RETURNING id
            """,
            label,
            node_type,
            norm,
            json.dumps(embedding),
            json.dumps(source_ref),
        )
        return row["id"]
    except asyncpg.DataError as e:
        log.error("Type mismatch creating symbol node '%s': %s", label, e)
        return None
    except asyncpg.IntegrityError as e:
        log.error("Constraint violation creating symbol node '%s': %s", label, e)
        return None
    except Exception as e:
        log.error("Failed to create symbol node '%s': %s", label, e)
        return None
