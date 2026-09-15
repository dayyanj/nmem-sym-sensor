"""
Darwinian selection pressure for sensory memory.

Implements competitive displacement at three levels:

1. **Word-visual co-occurrence competition**: When word A strengthens on
   a visual node, competing words on that same node weaken. "Red" gaining
   evidence actively suppresses "orange" on the same color node. The
   strongest association displaces its competitors rather than just
   passively outgrowing them.

2. **Cluster member competition**: A node can't meaningfully belong to
   50 clusters. The most coherent cluster for a node should pull it away
   from less coherent ones. Finite membership = competitive pressure.

3. **Label competition**: When a cluster's grounded label is confirmed,
   alternative labels decay. When a label is challenged, the incumbent
   decays. Only labels backed by persistent evidence survive.

The math is the same at all levels: when A strengthens by delta,
competitors weaken by delta * displacement_rate. This creates a
zero-sum dynamic where the total "belief budget" is finite.

Designed as a reusable pattern that can be lifted into nmem-sym
for symbolic graph competition (edge LTD, hypothesis displacement).
"""
import logging

import asyncpg
import numpy as np

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────

# How much competitors weaken when a winner strengthens.
# 0.1 = competitors lose 10% of the winner's gain.
# Higher = more aggressive selection, faster convergence, risk of
# prematurely killing correct associations.
WORD_DISPLACEMENT_RATE = float(__import__("os").environ.get(
    "NMEM_SENSOR_WORD_DISPLACEMENT_RATE", "0.15"))

# Minimum co-occurrence count below which a record gets pruned entirely.
# Prevents the table from growing unboundedly with near-zero noise entries.
WORD_PRUNE_THRESHOLD = 2

# Maximum cluster memberships per node. If a node belongs to more
# clusters than this, the weakest memberships are pruned.
MAX_CLUSTER_MEMBERSHIPS = 5

# Label decay rate when a competing label gains evidence
LABEL_DISPLACEMENT_RATE = 0.1


# ── Word-visual competitive displacement ─────────────────

async def displace_word_competitors(
    pool: asyncpg.Pool,
    word: str,
    node_id: int,
    gain: int = 1,
) -> int:
    """When word gains co-occurrence with node, competitors weaken.

    Called after recording a new word-visual co-occurrence. All OTHER
    words on the same node lose a fraction of the gain.

    Args:
        pool: Database pool.
        word: The word that just strengthened.
        node_id: The visual node it strengthened on.
        gain: How much the word gained (typically 1 per observation).

    Returns:
        Number of competitor records weakened.
    """
    displacement = max(1, int(gain * WORD_DISPLACEMENT_RATE))

    # Weaken all other words on this node
    result = await pool.execute(
        """
        UPDATE word_visual_cooccurrences
        SET count = GREATEST(count - $1, 0)
        WHERE node_id = $2 AND word != $3 AND count > 0
        """,
        displacement, node_id, word,
    )
    weakened = int(result.split()[-1])

    # Prune entries that have decayed to near-zero (exclude the word that just gained)
    pruned_result = await pool.execute(
        """
        DELETE FROM word_visual_cooccurrences
        WHERE node_id = $1 AND word != $3 AND count < $2
        """,
        node_id, WORD_PRUNE_THRESHOLD, word,
    )
    pruned = int(pruned_result.split()[-1])

    if pruned > 0:
        log.debug("Pruned %d dead co-occurrences on node %d after '%s' displaced",
                 pruned, node_id, word)

    return weakened


async def run_word_competition(pool: asyncpg.Pool) -> dict:
    """Run competitive displacement across all word-visual co-occurrences.

    For each visual node, the dominant word (highest count) displaces
    competitors. Called during consolidation cycles.

    This is the periodic "survival of the fittest" sweep — individual
    displacements happen per-observation via displace_word_competitors(),
    but this sweep catches any that were missed and applies bulk pressure.

    Returns stats dict.
    """
    # Find nodes where there's meaningful competition
    # (at least 2 words with count >= 5)
    competitive_nodes = await pool.fetch(
        """
        SELECT node_id, COUNT(*) as n_words
        FROM word_visual_cooccurrences
        WHERE count >= 5
        GROUP BY node_id
        HAVING COUNT(*) >= 2
        ORDER BY COUNT(*) DESC
        """
    )

    total_weakened = 0
    total_pruned = 0

    for row in competitive_nodes:
        node_id = row["node_id"]

        # Find the dominant word
        dominant = await pool.fetchrow(
            """
            SELECT word, count FROM word_visual_cooccurrences
            WHERE node_id = $1
            ORDER BY count DESC
            LIMIT 1
            """,
            node_id,
        )

        if not dominant:
            continue

        # Displacement proportional to dominance gap
        # If dominant has 100 and runner-up has 80, displacement is small.
        # If dominant has 100 and runner-up has 10, displacement is large.
        runner_up = await pool.fetchval(
            """
            SELECT count FROM word_visual_cooccurrences
            WHERE node_id = $1 AND word != $2
            ORDER BY count DESC
            LIMIT 1
            """,
            node_id, dominant["word"],
        )

        if runner_up is None:
            continue

        dominance_ratio = dominant["count"] / max(runner_up, 1)
        if dominance_ratio < 1.5:
            continue  # too close to call — no displacement

        # Displacement strength scales with dominance
        displacement = max(1, int(dominance_ratio * WORD_DISPLACEMENT_RATE))

        result = await pool.execute(
            """
            UPDATE word_visual_cooccurrences
            SET count = GREATEST(count - $1, 0)
            WHERE node_id = $2 AND word != $3 AND count > 0
            """,
            displacement, node_id, dominant["word"],
        )
        total_weakened += int(result.split()[-1])

        # Prune dead entries
        pruned = await pool.execute(
            """
            DELETE FROM word_visual_cooccurrences
            WHERE node_id = $1 AND count < $2
            """,
            node_id, WORD_PRUNE_THRESHOLD,
        )
        total_pruned += int(pruned.split()[-1])

    stats = {
        "competitive_nodes": len(competitive_nodes),
        "records_weakened": total_weakened,
        "records_pruned": total_pruned,
    }

    if total_weakened > 0 or total_pruned > 0:
        log.info("Word competition: %d nodes, %d weakened, %d pruned",
                len(competitive_nodes), total_weakened, total_pruned)

    return stats


# ── Cluster membership competition ───────────────────────

async def compete_cluster_memberships(pool: asyncpg.Pool) -> dict:
    """Prune excess cluster memberships per node.

    A node that belongs to 50 clusters is effectively unclassified —
    it's in everything and therefore means nothing. Limit to the
    MAX_CLUSTER_MEMBERSHIPS most coherent clusters.

    Returns stats dict.
    """
    # Find nodes with too many memberships
    overcrowded = await pool.fetch(
        """
        SELECT node_id, COUNT(*) as n_clusters
        FROM sensory_cluster_members
        GROUP BY node_id
        HAVING COUNT(*) > $1
        """,
        MAX_CLUSTER_MEMBERSHIPS,
    )

    total_pruned = 0

    for row in overcrowded:
        node_id = row["node_id"]

        # Keep the top N clusters by coherence, prune the rest
        keep = await pool.fetch(
            """
            SELECT cm.cluster_id
            FROM sensory_cluster_members cm
            JOIN sensory_clusters c ON c.id = cm.cluster_id
            WHERE cm.node_id = $1
            ORDER BY c.coherence DESC, c.total_observations DESC
            LIMIT $2
            """,
            node_id, MAX_CLUSTER_MEMBERSHIPS,
        )
        keep_ids = [r["cluster_id"] for r in keep]

        if not keep_ids:
            continue

        result = await pool.execute(
            """
            DELETE FROM sensory_cluster_members
            WHERE node_id = $1 AND cluster_id != ALL($2)
            """,
            node_id, keep_ids,
        )
        pruned = int(result.split()[-1])
        total_pruned += pruned

    # Sync counts and prune empty clusters after membership deletions
    if total_pruned > 0:
        from nmem_sym_sensor.graph import sync_cluster_counts
        await sync_cluster_counts(pool)

    stats = {
        "overcrowded_nodes": len(overcrowded),
        "memberships_pruned": total_pruned,
    }

    if total_pruned > 0:
        log.info("Cluster competition: %d overcrowded nodes, %d memberships pruned",
                len(overcrowded), total_pruned)

    return stats


# ── Label competition ────────────────────────────────────

async def compete_labels(pool: asyncpg.Pool) -> dict:
    """Decay weak label candidates when strong ones gain evidence.

    For each cluster with multiple label candidates, the strongest
    label's tally displaces weaker ones.

    Returns stats dict.
    """
    # Find clusters with competing labels
    contested = await pool.fetch(
        """
        SELECT cluster_id, COUNT(*) as n_labels
        FROM sensory_label_candidates
        GROUP BY cluster_id
        HAVING COUNT(*) >= 2
        """
    )

    total_decayed = 0
    total_pruned = 0

    for row in contested:
        cluster_id = row["cluster_id"]

        # Get dominant label
        dominant = await pool.fetchrow(
            """
            SELECT label, tally FROM sensory_label_candidates
            WHERE cluster_id = $1
            ORDER BY tally DESC
            LIMIT 1
            """,
            cluster_id,
        )

        if not dominant:
            continue

        # Decay competitors
        displacement = max(1, int(dominant["tally"] * LABEL_DISPLACEMENT_RATE))
        result = await pool.execute(
            """
            UPDATE sensory_label_candidates
            SET tally = GREATEST(tally - $1, 0)
            WHERE cluster_id = $2 AND label != $3
            """,
            displacement, cluster_id, dominant["label"],
        )
        total_decayed += int(result.split()[-1])

        # Prune dead candidates
        pruned = await pool.execute(
            """
            DELETE FROM sensory_label_candidates
            WHERE cluster_id = $1 AND tally <= 0
            """,
            cluster_id,
        )
        total_pruned += int(pruned.split()[-1])

    stats = {
        "contested_clusters": len(contested),
        "labels_decayed": total_decayed,
        "labels_pruned": total_pruned,
    }

    if total_decayed > 0:
        log.info("Label competition: %d contested, %d decayed, %d pruned",
                len(contested), total_decayed, total_pruned)

    return stats


# ── Full selection cycle ─────────────────────────────────

# ── Coherence-based word spread pruning ──────────────────

# Words that bind to fewer nodes than this aren't checked — they're
# either specific concepts or haven't accumulated enough evidence yet.
WORD_SPREAD_THRESHOLD = int(__import__("os").environ.get(
    "NMEM_SENSOR_WORD_SPREAD_THRESHOLD", "5"))

# Coherence below this → noise. Prune weakest bindings.
WORD_NOISE_COHERENCE = float(__import__("os").environ.get(
    "NMEM_SENSOR_WORD_NOISE_COHERENCE", "0.35"))

# Coherence above this → potential category. Keep all bindings.
WORD_CATEGORY_COHERENCE = float(__import__("os").environ.get(
    "NMEM_SENSOR_WORD_CATEGORY_COHERENCE", "0.50"))

# For noise words, keep only the top N strongest bindings.
WORD_NOISE_KEEP_TOP = int(__import__("os").environ.get(
    "NMEM_SENSOR_WORD_NOISE_KEEP_TOP", "3"))


async def prune_word_spread(pool: asyncpg.Pool) -> dict:
    """Prune words that bind promiscuously to many unrelated nodes.

    A word that binds to 30+ different nodes is either:
    - A CATEGORY concept (its nodes are similar: "animal" → dog, cat, fish)
    - NOISE (its nodes are random: "wow" → triangle, red, octopus)

    We distinguish them by computing the coherence (mean pairwise cosine
    similarity) of the bound nodes' visual embeddings.

    High coherence → keep (potential category for future hierarchy)
    Low coherence → prune weakest bindings, keep only the strongest

    Returns stats dict.
    """
    # Find words with high spread
    spread_words = await pool.fetch(
        """
        SELECT word, COUNT(DISTINCT node_id) as spread, SUM(count) as total
        FROM word_visual_cooccurrences
        GROUP BY word
        HAVING COUNT(DISTINCT node_id) >= $1
        ORDER BY COUNT(DISTINCT node_id) DESC
        """,
        WORD_SPREAD_THRESHOLD,
    )

    stats = {
        "words_checked": len(spread_words),
        "noise_words_pruned": 0,
        "records_pruned": 0,
        "category_words_found": 0,
    }

    for sw in spread_words:
        word = sw["word"]
        spread = sw["spread"]

        # Fetch visual embeddings of bound nodes
        rows = await pool.fetch(
            """
            SELECT wv.node_id, wv.count, n.visual_embedding::text as emb
            FROM word_visual_cooccurrences wv
            JOIN sensory_nodes n ON n.id = wv.node_id
            WHERE wv.word = $1
              AND n.visual_embedding IS NOT NULL
            ORDER BY wv.count DESC
            """,
            word,
        )

        if len(rows) < 2:
            continue

        # Parse embeddings and compute coherence
        vecs = []
        for r in rows:
            emb_str = r["emb"].strip("[]")
            if emb_str:
                vecs.append(np.array(
                    [float(x) for x in emb_str.split(",")], dtype=np.float32,
                ))

        if len(vecs) < 2:
            continue

        emb_matrix = np.array(vecs)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = emb_matrix / norms
        sim_matrix = normed @ normed.T

        n = len(vecs)
        upper = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        coherence = float(np.mean(upper))

        if coherence >= WORD_CATEGORY_COHERENCE:
            # Potential category — leave all bindings intact
            stats["category_words_found"] += 1
            log.debug(
                "Word '%s' (spread=%d, coherence=%.3f) → CATEGORY, keeping",
                word, spread, coherence,
            )
            continue

        if coherence < WORD_NOISE_COHERENCE:
            # Noise — keep only the top N strongest bindings, prune the rest
            keep_ids = [r["node_id"] for r in rows[:WORD_NOISE_KEEP_TOP]]

            if not keep_ids:
                continue

            result = await pool.execute(
                """
                DELETE FROM word_visual_cooccurrences
                WHERE word = $1
                  AND node_id != ALL($2::bigint[])
                """,
                word, keep_ids,
            )
            pruned = int(result.split()[-1])

            if pruned > 0:
                stats["noise_words_pruned"] += 1
                stats["records_pruned"] += pruned
                log.info(
                    "Pruned '%s' (spread=%d→%d, coherence=%.3f): "
                    "removed %d weak bindings",
                    word, spread, min(spread, WORD_NOISE_KEEP_TOP),
                    coherence, pruned,
                )

        # Ambiguous (between thresholds) — weaken but don't prune yet
        # The per-node competition will handle gradual displacement

    if stats["records_pruned"] > 0 or stats["category_words_found"] > 0:
        log.info(
            "Word spread pruning: checked %d words, %d noise pruned "
            "(%d records), %d categories found",
            stats["words_checked"], stats["noise_words_pruned"],
            stats["records_pruned"], stats["category_words_found"],
        )

    return stats


# ── Sound-visual competitive displacement ─────────────────


async def displace_sound_competitors(
    pool: asyncpg.Pool,
    sound_unit_id: int,
    visual_node_id: int,
    gain: int = 1,
) -> int:
    """When a sound unit strengthens on a visual node, competing sounds weaken.

    Uses unified co-occurrence table. Modality-scoped: voice competes with
    voice, environmental with environmental.
    """
    from nmem_sym_sensor.cooccurrence import cooc_store

    # Get this unit's modality
    mod = await pool.fetchval(
        "SELECT modality FROM sound_units WHERE id = $1", sound_unit_id,
    )
    if not mod:
        return 0

    displacement = max(1, int(gain * WORD_DISPLACEMENT_RATE))
    await cooc_store.displace(
        sound_unit_id, mod, visual_node_id, "visual", pool, displacement,
    )
    return displacement


async def run_sound_competition(pool: asyncpg.Pool) -> dict:
    """Run competitive displacement for sound-visual co-occurrences.

    Uses unified co-occurrence table via CooccurrenceStore.
    """
    from nmem_sym_sensor.cooccurrence import cooc_store

    # Find visual nodes with 2+ competing voice units
    competitive = await cooc_store.find_competitive_targets(
        "voice", "visual", pool, min_count=5,
    )

    total_weakened = 0
    total_pruned = 0

    for comp in competitive:
        target_id = comp["target_id"]
        dominant_id = comp["dominant_id"]
        runner_up_count = comp["runner_up_count"]
        dominant_count = comp["dominant_count"]

        dominance_ratio = dominant_count / max(runner_up_count, 1)
        if dominance_ratio < 1.5:
            continue

        displacement = max(1, int(dominance_ratio * WORD_DISPLACEMENT_RATE))
        await cooc_store.displace(
            dominant_id, "voice", target_id, "visual", pool, displacement,
        )
        total_weakened += 1

    stats = {
        "competitive_nodes": len(competitive),
        "records_weakened": total_weakened,
        "records_pruned": total_pruned,
    }

    if total_weakened > 0:
        log.info("Sound competition: %d nodes, %d weakened",
                 len(competitive), total_weakened)

    return stats


async def prune_sound_spread(pool: asyncpg.Pool) -> dict:
    """Prune sound units that bind promiscuously to many unrelated visual nodes.

    Uses unified co-occurrence table via CooccurrenceStore.
    """
    from nmem_sym_sensor.cooccurrence import cooc_store

    spread_units = await cooc_store.find_spread_units(
        "voice", "visual", pool, spread_threshold=WORD_SPREAD_THRESHOLD,
    )

    stats = {
        "units_checked": len(spread_units),
        "noise_units_pruned": 0,
        "records_pruned": 0,
        "category_units_found": 0,
    }

    for su in spread_units:
        unit_id = su["unit_id"]
        spread = su["spread"]

        rows = await cooc_store.get_unit_bindings(
            unit_id, "voice", "visual", pool,
        )

        if len(rows) < 2:
            continue

        vecs = []
        for r in rows:
            emb_str = r["emb"]
            if emb_str:
                emb_str = emb_str.strip("[]")
                if emb_str:
                    vecs.append(np.array(
                        [float(x) for x in emb_str.split(",")], dtype=np.float32,
                    ))

        if len(vecs) < 2:
            continue

        emb_matrix = np.array(vecs)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = emb_matrix / norms
        sim_matrix = normed @ normed.T

        n = len(vecs)
        upper = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        coherence = float(np.mean(upper))

        if coherence >= WORD_CATEGORY_COHERENCE:
            stats["category_units_found"] += 1
            continue

        if coherence < WORD_NOISE_COHERENCE:
            keep_ids = [r["target_id"] for r in rows[:WORD_NOISE_KEEP_TOP]]
            if not keep_ids:
                continue

            pruned = await cooc_store.archive_spread(
                unit_id, "voice", keep_ids, "visual", pool,
            )

            if pruned > 0:
                stats["noise_units_pruned"] += 1
                stats["records_pruned"] += pruned
                log.info(
                    "Pruned sound unit #%d (spread=%d→%d, coherence=%.3f): removed %d weak bindings",
                    unit_id, spread, min(spread, WORD_NOISE_KEEP_TOP), coherence, pruned,
                )

    if stats["records_pruned"] > 0 or stats["category_units_found"] > 0:
        log.info(
            "Sound spread pruning: checked %d units, %d noise pruned (%d records), %d categories",
            stats["units_checked"], stats["noise_units_pruned"],
            stats["records_pruned"], stats["category_units_found"],
        )

    return stats


async def run_selection_pressure(pool: asyncpg.Pool) -> dict:
    """Run all competitive displacement mechanisms.

    Call this during or after consolidation. The order matters:
    1. Word competition (cleans text co-occurrence table per-node) — DIAGNOSTIC
    2. Word spread pruning (removes promiscuous noise words) — DIAGNOSTIC
    3. Sound competition (cleans symbol co-occurrence table) — PRIMARY
    4. Sound spread pruning (removes promiscuous noise sounds) — PRIMARY
    5. Cluster membership competition (prunes overcrowded nodes)
    6. Label competition (resolves naming conflicts)

    Returns combined stats.
    """
    # Each mechanism is INDEPENDENT — guard per-step so one subsystem's failure (e.g. the audio/
    # sound co-occurrence path on a VISUAL-ONLY deployment, which has no audio data) can't abort the
    # rest, in particular the visual-relevant cluster/label competition that follows the sound steps.
    stats: dict = {}
    for name, fn in (
        ("word_competition", run_word_competition),
        ("word_spread", prune_word_spread),
        ("sound_competition", run_sound_competition),
        ("sound_spread", prune_sound_spread),
        ("cluster_competition", compete_cluster_memberships),
        ("label_competition", compete_labels),
    ):
        try:
            stats[name] = await fn(pool)
        except Exception as e:  # noqa: BLE001
            log.warning("[selection] %s failed (non-fatal, continuing): %s", name, e)
            stats[name] = {"error": str(e)}
    return stats
