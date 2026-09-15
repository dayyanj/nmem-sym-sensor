"""Dreamstate Mental Rotation — orientation consolidation.

Shepard & Metzler (1971) showed the brain literally rotates mental images
to compare them. Response time scales linearly with angular difference.

At observation time, the encoder produces orientation-aware embeddings
(cup upright != cup sideways). Orientation metadata is stored in node
features. In dreamstate, we discover which concepts are rotation-invariant
and which are orientation-sensitive:

- A ball looks the same from any angle -> rotation-invariant
- Text upside-down is unreadable -> orientation-sensitive
- A triangle pointing up vs down might be the same shape or different

This is learned through consolidation, not assumed. The process:
1. Find clusters with members at different orientations
2. Compare embeddings directly (raw similarity)
3. Virtually rotate one embedding to match the other's orientation
4. If rotated similarity >> raw similarity -> rotation helps -> orientation-sensitive concept
5. If raw similarity is already high -> rotation-invariant concept
6. Adjust co-occurrence strengths accordingly

Over many cycles:
- Rotation-invariant associations strengthen (they survive rotation tests)
- Orientation-sensitive associations are flagged (orientation matters for them)
"""
import logging
import random

import asyncpg
import numpy as np

log = logging.getLogger(__name__)

# Thresholds
HIGH_SIM = 0.85          # Already similar without rotation -> invariant
ROTATION_GAIN = 0.10     # Minimum gain from rotation to call it "sensitive"
MIN_MEMBERS = 3          # Cluster needs at least this many members to test
REINFORCE_FACTOR = 1.03  # Gentler than recombination (rotation is subtler)
WEAKEN_FACTOR = 0.97

# Orientation categories and their 90-degree rotation mappings
ORIENTATION_MAP = {
    "vertical": "horizontal",
    "horizontal": "vertical",
    "square": "square",
}


async def dreamstate_mental_rotation(
    pool: asyncpg.Pool,
    max_tests: int = 3,
) -> dict:
    """Run one mental rotation consolidation cycle.

    Finds clusters with orientation diversity, tests whether the concept
    is rotation-invariant or orientation-sensitive, and adjusts
    co-occurrence strengths accordingly.

    Returns stats dict.
    """
    stats = {
        "candidates_found": 0,
        "tests_run": 0,
        "invariant": 0,
        "sensitive": 0,
        "inconclusive": 0,
    }

    candidates = await _find_rotation_candidates(pool)
    stats["candidates_found"] = len(candidates)

    if not candidates:
        return stats

    # Sample up to max_tests candidates
    selected = random.sample(candidates, min(max_tests, len(candidates)))

    for cand in selected:
        result = await _test_rotation(pool, cand)
        if result is None:
            continue

        stats["tests_run"] += 1
        stats[result["verdict"]] += 1

        # Record the rotation test
        await _record_rotation(pool, result)

        # Apply co-occurrence adjustments
        await _apply_rotation_feedback(pool, cand["cluster_id"], result)

        log.info(
            "Mental rotation: cluster %d ('%s') members %d↔%d | "
            "raw=%.3f rotated=%.3f gain=%.3f -> %s",
            cand["cluster_id"], cand.get("label", "?"),
            result["member_a_id"], result["member_b_id"],
            result["embedding_similarity"], result["rotated_similarity"],
            result["similarity_gain"], result["verdict"],
        )

    return stats


async def _find_rotation_candidates(pool: asyncpg.Pool) -> list[dict]:
    """Find clusters whose members span multiple orientations.

    A good candidate has members with different orientation values
    in their features JSONB. This means the same concept was observed
    at different angles — exactly what we want to test.
    """
    rows = await pool.fetch(
        """
        SELECT sc.id as cluster_id,
               sc.grounded_label as label,
               sc.member_count,
               COUNT(DISTINCT n.features->>'orientation') as orientation_diversity
        FROM sensory_clusters sc
        JOIN sensory_cluster_members cm ON cm.cluster_id = sc.id
        JOIN sensory_nodes n ON n.id = cm.node_id
        WHERE sc.cluster_type IN ('stable', 'grounded')
          AND sc.modality = 'visual'
          AND sc.member_count >= $1
          AND n.features->>'orientation' IS NOT NULL
          AND n.visual_embedding IS NOT NULL
          AND sc.id NOT IN (
              SELECT cluster_id FROM dreamstate_rotations
              WHERE created_at > NOW() - INTERVAL '2 hours'
          )
        GROUP BY sc.id
        HAVING COUNT(DISTINCT n.features->>'orientation') >= 2
        ORDER BY sc.total_observations DESC
        LIMIT 20
        """,
        MIN_MEMBERS,
    )

    return [dict(r) for r in rows]


async def _test_rotation(
    pool: asyncpg.Pool,
    candidate: dict,
) -> dict | None:
    """Test rotation invariance for a single cluster.

    Picks two members with different orientations, compares their
    embeddings raw and after virtual rotation. If rotation brings
    them closer, the concept is orientation-sensitive.
    """
    cluster_id = candidate["cluster_id"]

    # Get members grouped by orientation (categorical or numeric angle)
    members = await pool.fetch(
        """
        SELECT n.id, n.features->>'orientation' as orientation,
               n.features->>'orientation_angle' as orientation_angle,
               n.visual_embedding::text as emb,
               n.features->>'elongation' as elongation
        FROM sensory_cluster_members cm
        JOIN sensory_nodes n ON n.id = cm.node_id
        WHERE cm.cluster_id = $1
          AND n.features->>'orientation' IS NOT NULL
          AND n.visual_embedding IS NOT NULL
        """,
        cluster_id,
    )

    if len(members) < 2:
        return None

    # Group by orientation
    by_orient: dict[str, list] = {}
    for m in members:
        orient = m["orientation"]
        by_orient.setdefault(orient, []).append(m)

    # Pick two members with different orientations
    orientations = list(by_orient.keys())
    if len(orientations) < 2:
        return None

    orient_a, orient_b = random.sample(orientations, 2)
    member_a = random.choice(by_orient[orient_a])
    member_b = random.choice(by_orient[orient_b])

    # Parse embeddings
    emb_a = _parse_embedding(member_a["emb"])
    emb_b = _parse_embedding(member_b["emb"])
    if emb_a is None or emb_b is None:
        return None

    # Raw cosine similarity (no rotation)
    raw_sim = float(np.dot(emb_a, emb_b))

    # Virtual rotation: prefer numeric angles when available
    angle_a = float(member_a["orientation_angle"]) if member_a["orientation_angle"] else None
    angle_b = float(member_b["orientation_angle"]) if member_b["orientation_angle"] else None
    if angle_a is not None and angle_b is not None:
        rotated_a = _virtual_rotate_numeric(emb_a, angle_a, angle_b)
    else:
        rotated_a = _virtual_rotate(emb_a, orient_a, orient_b)
    rotated_sim = float(np.dot(rotated_a, emb_b))

    gain = rotated_sim - raw_sim

    # Verdict
    if raw_sim >= HIGH_SIM:
        verdict = "invariant"
    elif gain >= ROTATION_GAIN:
        verdict = "sensitive"
    else:
        verdict = "inconclusive"

    return {
        "cluster_id": cluster_id,
        "member_a_id": member_a["id"],
        "member_b_id": member_b["id"],
        "orientation_a": orient_a,
        "orientation_b": orient_b,
        "embedding_similarity": round(raw_sim, 4),
        "rotated_similarity": round(rotated_sim, 4),
        "similarity_gain": round(gain, 4),
        "verdict": verdict,
    }


def _parse_embedding(emb_text: str) -> np.ndarray | None:
    """Parse a vector text representation into a normalized numpy array."""
    try:
        stripped = emb_text.strip("[]")
        if not stripped:
            return None
        vec = np.array([float(x) for x in stripped.split(",")], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    except (ValueError, AttributeError):
        return None


def _virtual_rotate_numeric(
    embedding: np.ndarray,
    from_angle: float,
    to_angle: float,
) -> np.ndarray:
    """Rotate embedding by a numeric angular difference (degrees).

    For the dual-stream 512-dim encoder, the first 320 dims are geometric
    with quadrant-aware HoGC (4 quadrants x 10 bins = 40 dims at the start).
    Rotating means cyclically shifting the quadrant bins by the angular
    difference. A 90-degree rotation shifts by 1 quadrant.

    The JEPA portion (dims 320-511) is left unchanged — JEPA learned some
    rotation invariance from augmented training data.
    """
    diff = (to_angle - from_angle) % 360
    if diff < 5 or diff > 355:
        return embedding.copy()

    rotated = embedding.copy()
    dim = len(rotated)

    if dim >= 512:
        # Dual-stream: geometric portion has quadrant HoGC in first 40 dims
        # 4 quadrants x 10 bins: [Q0_10, Q1_10, Q2_10, Q3_10]
        # 90° rotation = shift by 1 quadrant, 180° = shift by 2, 270° = shift by 3
        q_bins = 10
        n_quads = 4
        q_block = rotated[:n_quads * q_bins].copy()
        shift = round(diff / 90) % n_quads
        if shift > 0:
            reshaped = q_block.reshape(n_quads, q_bins)
            rotated[:n_quads * q_bins] = np.roll(reshaped, shift, axis=0).ravel()

        # Also rotate quadrant colour features (next block of 4 x 4 = 16 dims)
        qc_start = n_quads * q_bins  # 40
        qc_bins = 4
        if dim > qc_start + n_quads * qc_bins:
            qc_block = rotated[qc_start:qc_start + n_quads * qc_bins].copy()
            reshaped_c = qc_block.reshape(n_quads, qc_bins)
            rotated[qc_start:qc_start + n_quads * qc_bins] = np.roll(
                reshaped_c, shift, axis=0,
            ).ravel()

    # Re-normalize
    norm = np.linalg.norm(rotated)
    if norm > 0:
        rotated /= norm
    return rotated


def _virtual_rotate(
    embedding: np.ndarray,
    from_orient: str,
    to_orient: str,
) -> np.ndarray:
    """Virtually rotate an embedding from one orientation to another.

    The rotation operates on the geometric dimensions of the embedding.
    For hand-crafted embeddings (256-dim), dims 40-47 encode geometry
    (area, aspect_ratio). For learned embeddings (512-dim), we apply
    a dimension-swap transform on spatial feature dimensions.

    The key insight: we don't need a perfect rotation matrix. We just
    need to transform the embedding enough that orientation-invariant
    concepts become more similar and orientation-sensitive concepts
    don't. Even an approximate transform reveals the invariance structure.

    Strategy:
    - vertical <-> horizontal: swap aspect ratio encoding + mirror spatial dims
    - square -> anything: no transform needed (already symmetric)
    """
    if from_orient == to_orient or from_orient == "square" or to_orient == "square":
        return embedding.copy()

    rotated = embedding.copy()
    dim = len(rotated)

    if dim <= 256:
        # Hand-crafted embedding: dim 41 = aspect_ratio
        # Vertical->horizontal means inverting aspect ratio encoding
        rotated[41] = 1.0 - rotated[41]

        # Position dims if present (pos_x/pos_y would swap)
        # Dims 42-47 may encode spatial info
        if dim > 43:
            rotated[42], rotated[43] = rotated[43], rotated[42]
    else:
        # Learned embedding (512-dim): apply a block-swap transform.
        # The learned encoder encodes spatial structure throughout the
        # vector. We approximate a 90-degree rotation by swapping
        # complementary halves of spatial-sensitive dimension blocks.
        #
        # This is intentionally imprecise — we're testing whether the
        # concept survives transformation, not producing a perfect rotation.
        # If the concept is truly rotation-invariant, even this rough
        # transform won't change the similarity much.
        quarter = dim // 4

        # Swap the 2nd and 3rd quarters (spatial structure lives here
        # in typical ViT / CNN encoder outputs)
        tmp = rotated[quarter:2*quarter].copy()
        rotated[quarter:2*quarter] = rotated[2*quarter:3*quarter]
        rotated[2*quarter:3*quarter] = tmp

    # Re-normalize
    norm = np.linalg.norm(rotated)
    if norm > 0:
        rotated /= norm

    return rotated


async def _record_rotation(pool: asyncpg.Pool, result: dict) -> None:
    """Persist the rotation test result."""
    await pool.execute(
        """
        INSERT INTO dreamstate_rotations
            (cluster_id, member_a_id, member_b_id,
             orientation_a, orientation_b,
             embedding_similarity, rotated_similarity,
             similarity_gain, verdict)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        result["cluster_id"],
        result["member_a_id"], result["member_b_id"],
        result["orientation_a"], result["orientation_b"],
        result["embedding_similarity"], result["rotated_similarity"],
        result["similarity_gain"], result["verdict"],
    )


async def _apply_rotation_feedback(
    pool: asyncpg.Pool,
    cluster_id: int,
    result: dict,
) -> None:
    """Adjust co-occurrence strengths based on rotation verdict.

    Rotation-invariant: the concept works at any angle.
    Boost its co-occurrences and increment orientation_independence.

    Orientation-sensitive: angle matters for this concept.
    Slightly weaken co-occurrences and increment orientation_dependence.
    This doesn't destroy the association — it flags it so the system
    knows orientation is part of the concept's identity.
    """

    if result["verdict"] == "invariant":
        factor = REINFORCE_FACTOR
        col = "orientation_independence"
    elif result["verdict"] == "sensitive":
        factor = WEAKEN_FACTOR
        col = "orientation_dependence"
    else:
        return  # inconclusive — don't adjust

    # Get all co-occurrences for this cluster's member nodes.
    # Co-occurrences use node IDs, so join through cluster_members.
    coocs = await pool.fetch(
        """
        SELECT DISTINCT co.unit_a_id, co.modality_a, co.unit_b_id, co.modality_b
        FROM sensory_cooccurrences co
        JOIN sensory_cluster_members cm ON
            (co.unit_a_id = cm.node_id AND co.modality_a = 'visual')
            OR (co.unit_b_id = cm.node_id AND co.modality_b = 'visual')
        WHERE cm.cluster_id = $1
          AND NOT co.myelinated
          AND co.count > 0
        """,
        cluster_id,
    )

    for row in coocs:
        await pool.execute(
            f"""
            UPDATE sensory_cooccurrences
            SET strength = strength * $1,
                {col} = {col} + 1
            WHERE unit_a_id = $2 AND modality_a = $3
              AND unit_b_id = $4 AND modality_b = $5
              AND NOT myelinated
            """,
            factor,
            row["unit_a_id"], row["modality_a"],
            row["unit_b_id"], row["modality_b"],
        )
