"""
Sensory Dreamstate: offline consolidation during idle periods.

Like sleep consolidation in the brain — replays, restructures, and
strengthens sensory knowledge when the system isn't actively perceiving.

Triggered by nmem-sym's dreamstate cycle (registered as post-dreamstate
hook) or by the sensory coherence drive when pressure builds.

What it does:
1. Replay: re-examine clusters with current knowledge
2. Schema extraction: find recurring compositions → structural schemas
3. Prediction grounding: check old predictions — did outcomes occur?
4. Cross-modal consolidation: strengthen consistent audio-visual bindings
5. Cluster hygiene: merge duplicates, split incoherent clusters
6. Abstraction: promote consistent compositions to prototype templates
7. Expectation refinement: update structural expectations with evidence
"""
import logging
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from nmem_sym_sensor.vocal_tract import VocalTract

log = logging.getLogger(__name__)


async def run_sensory_dreamstate(
    pool: asyncpg.Pool,
    sym_pool: asyncpg.Pool | None = None,
    max_duration_s: float = 30.0,
    only_functions: set[str] | None = None,
    vocal_tract: "VocalTract | None" = None,
) -> dict:
    """Run a sensory dreamstate consolidation cycle.

    This is the sensory system's equivalent of sleep — offline
    processing that restructures and strengthens learned knowledge.

    Can be called:
    - As a post-dreamstate hook (via nmem-sym bridge)
    - Directly when the system is idle
    - By the drive system when coherence pressure builds
    - By the pressure controller with selective functions

    Args:
        pool: Sensory DB pool.
        sym_pool: nmem-sym DB pool (optional, for cross-system grounding).
        max_duration_s: Maximum time to spend (budget-aware).
        only_functions: If set, only run these specific functions.
            Uses names from dreamstate_pressure module constants:
            "decay", "self_play", "concept_linking", "recombination",
            "competition", "composition", "dedup", "reactivation",
            "node_decay", "scene_consolidation".
            If None, runs all functions (backward compatible).

    Returns:
        Stats dict with "ran_functions" key listing what actually executed.
    """
    import time
    start = time.monotonic()
    stats = {}
    ran_functions: list[str] = []

    def budget_remaining():
        return max_duration_s - (time.monotonic() - start)

    def should_run(fn_name: str) -> bool:
        """Check if this function should run (selective or full cycle)."""
        if only_functions is None:
            return True
        return fn_name in only_functions

    # Steps 1-5: Lightweight housekeeping — always run (cheap, no pressure gating)
    # These are fast single-UPDATE operations that keep the system healthy.

    # 1. Expire stale predictions
    if budget_remaining() > 0:
        expired = await pool.execute(
            """
            UPDATE sensory_predictions
            SET status = 'expired', resolved_at = NOW()
            WHERE status = 'pending'
              AND created_at < NOW() - INTERVAL '1 hour'
            """
        )
        stats["predictions_expired"] = int(expired.split()[-1])

    # 2. Prediction accuracy stats
    if budget_remaining() > 0:
        accuracy = await pool.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'confirmed') as confirmed,
                COUNT(*) FILTER (WHERE status = 'refuted') as refuted,
                COUNT(*) FILTER (WHERE status = 'partial') as partial,
                COUNT(*) FILTER (WHERE status = 'expired') as expired,
                COUNT(*) as total
            FROM sensory_predictions
            WHERE resolved_at > NOW() - INTERVAL '24 hours'
            """
        )
        if accuracy and accuracy["total"] > 0:
            stats["prediction_accuracy_24h"] = {
                "confirmed": accuracy["confirmed"],
                "refuted": accuracy["refuted"],
                "partial": accuracy["partial"],
                "expired": accuracy["expired"],
                "accuracy": round(accuracy["confirmed"] / max(accuracy["total"], 1), 3),
            }

    # 3. Structural expectation refinement
    if budget_remaining() > 0:
        decayed = await pool.execute(
            """
            UPDATE structural_expectations
            SET confidence = GREATEST(0.1, confidence - 0.01)
            WHERE updated_at < NOW() - INTERVAL '7 days'
              AND confidence > 0.1
            """
        )
        stats["expectations_decayed"] = int(decayed.split()[-1])

        pruned = await pool.execute(
            """
            DELETE FROM structural_expectations
            WHERE confidence < 0.1
              AND observation_count < 5
            """
        )
        stats["expectations_pruned"] = int(pruned.split()[-1])

    # 4. Cross-modal binding consolidation
    if budget_remaining() > 0:
        strengthened = await pool.execute(
            """
            UPDATE sensory_edges
            SET weight = LEAST(1.0, weight + 0.02),
                ltp_score = LEAST(1.0, ltp_score + 0.01)
            WHERE edge_type = 'bound_to'
              AND groundedness >= 10
              AND weight < 0.9
            """
        )
        stats["bindings_strengthened"] = int(strengthened.split()[-1])

        weakened = await pool.execute(
            """
            UPDATE sensory_edges
            SET weight = GREATEST(0.1, weight - 0.01),
                ltd_score = LEAST(1.0, ltd_score + 0.01)
            WHERE edge_type = 'bound_to'
              AND groundedness < 3
              AND last_traversed < NOW() - INTERVAL '7 days'
            """
        )
        stats["bindings_weakened"] = int(weakened.split()[-1])

    # 5. Part-of edge consolidation
    if budget_remaining() > 0:
        await pool.execute(
            """
            UPDATE sensory_edges
            SET weight = LEAST(1.0, weight + 0.01),
                ltp_score = LEAST(1.0, ltp_score + 0.005)
            WHERE edge_type = 'part_of'
              AND groundedness >= 5
            """
        )

    # Steps 6+: Pressure-gated operations — only run when triggered or full cycle.

    # 6. Synaptic homeostasis — global decay of unmyelinated associations.
    # Track real wall-clock time between decay calls (not hardcoded 30s).
    if budget_remaining() > 0 and should_run("decay"):
        import time as _time
        _last = getattr(run_sensory_dreamstate, '_last_decay_time', None)
        _now = _time.monotonic()
        _elapsed = min(_now - _last, 300.0) if _last is not None else 30.0
        run_sensory_dreamstate._last_decay_time = _now
        decay_stats = await _homeostatic_decay(pool, elapsed_s=_elapsed)
        stats["homeostatic_decay"] = decay_stats
        ran_functions.append("decay")

    # 7. Self-play: round-trip consistency testing
    if budget_remaining() > 2 and should_run("self_play"):
        self_play_stats = await _self_play_round_trip(pool)
        stats["self_play"] = self_play_stats
        ran_functions.append("self_play")

    # 8. Concept linking — discover that different sounds mean the same thing
    if budget_remaining() > 2 and should_run("concept_linking"):
        concept_stats = await _discover_concept_links(pool)
        stats["concept_links"] = concept_stats
        ran_functions.append("concept_linking")

    # 9. Node salience decay (forget unimportant observations)
    if budget_remaining() > 0 and should_run("node_decay"):
        decayed_nodes = await pool.execute(
            """
            UPDATE sensory_nodes
            SET salience = GREATEST(0.1, salience * 0.95)
            WHERE NOT archived
              AND last_activated < NOW() - INTERVAL '7 days'
              AND salience > 0.2
            """
        )
        stats["nodes_salience_decayed"] = int(decayed_nodes.split()[-1])
        ran_functions.append("node_decay")

    # 10. Compositional analysis (discovers new part-whole patterns)
    if budget_remaining() > 2 and should_run("composition"):
        from nmem_sym_sensor.composition import run_compositional_analysis
        comp_stats = await run_compositional_analysis(pool)
        stats["composition"] = comp_stats
        ran_functions.append("composition")

    # 11. Archived node reactivation — dreaming mind revisits old traces
    if budget_remaining() > 1 and should_run("reactivation"):
        reactivated = await _dream_reactivate_archived(pool)
        stats["dream_reactivated"] = reactivated
        ran_functions.append("reactivation")

    # 12. Cluster deduplication (merge near-identical clusters)
    if budget_remaining() > 2 and should_run("dedup"):
        merged = await _merge_duplicate_clusters(pool)
        stats["clusters_merged"] = merged
        ran_functions.append("dedup")

    # 13. Dreamstate recombination — creative consolidation
    if budget_remaining() > 5 and should_run("recombination"):
        from nmem_sym_sensor.recombination import dreamstate_recombination
        recom_stats = await dreamstate_recombination(pool, max_candidates=1)
        stats["recombination"] = recom_stats
        ran_functions.append("recombination")

    # 14. Mental rotation — orientation consolidation
    # Tests whether concepts are rotation-invariant (ball) or
    # orientation-sensitive (face). Shares evaluation framework with
    # recombination: transform -> test -> strengthen/weaken.
    if budget_remaining() > 3 and should_run("mental_rotation"):
        from nmem_sym_sensor.mental_rotation import dreamstate_mental_rotation
        rotation_stats = await dreamstate_mental_rotation(pool, max_tests=3)
        stats["mental_rotation"] = rotation_stats
        ran_functions.append("mental_rotation")

    # 15. Syllable practice — speech motor consolidation via TABULA2
    if budget_remaining() > 3 and should_run("syllable_practice"):
        syl_stats = await _practice_syllables(pool, vocal_tract=vocal_tract, max_syllables=10)
        stats["syllable_practice"] = syl_stats
        ran_functions.append("syllable_practice")

    elapsed = time.monotonic() - start
    stats["duration_s"] = round(elapsed, 2)
    stats["ran_functions"] = ran_functions
    stats["selective"] = only_functions is not None

    log.info("Sensory dreamstate complete (%.1fs, %d functions): %s",
             elapsed, len(ran_functions), stats)
    return stats


async def _discover_concept_links(
    pool: asyncpg.Pool,
    min_cooccurrences: int = 3,
    min_overlap: float = 0.3,
    max_pairs: int = 100,
) -> dict:
    """Discover that different sound units refer to the same concept.

    Two sound units that co-occur with the same visual nodes (or saccade
    patterns) are likely the same concept spoken differently. This is how
    speaker invariance emerges — not from acoustic similarity but from
    shared meaning.

    For each pair of sound units with visual co-occurrences:
    1. Compute Jaccard overlap of their visual binding sets
    2. If overlap exceeds threshold, create a concept link
    3. Strengthen existing links that continue to hold
    4. Weaken links where overlap has decreased

    Also checks saccade co-occurrence overlap as additional evidence.
    """
    stats = {
        "pairs_checked": 0,
        "links_created": 0,
        "links_strengthened": 0,
        "links_weakened": 0,
    }

    # Find voice units with enough visual co-occurrences to compare
    # Uses unified table: voice units that bind to visual nodes
    active_units = await pool.fetch(
        """
        SELECT uid as sound_unit_id, array_agg(vid) as visual_set
        FROM (
            SELECT unit_a_id as uid, unit_b_id as vid
            FROM sensory_cooccurrences
            WHERE modality_a = 'voice' AND modality_b = 'visual' AND reinforcement_count >= $1
            UNION ALL
            SELECT unit_b_id as uid, unit_a_id as vid
            FROM sensory_cooccurrences
            WHERE modality_b = 'voice' AND modality_a = 'visual' AND reinforcement_count >= $1
        ) pairs
        GROUP BY uid
        HAVING COUNT(*) >= 2
        ORDER BY COUNT(*) DESC
        LIMIT $2
        """,
        min_cooccurrences,
        max_pairs,
    )

    if len(active_units) < 2:
        return stats

    # Build visual binding sets per unit
    unit_visuals: dict[int, set[int]] = {}
    for row in active_units:
        unit_visuals[row["sound_unit_id"]] = set(row["visual_set"])

    # Also gather saccade co-occurrence sets if available
    unit_saccades: dict[int, set[int]] = {}
    try:
        sac_rows = await pool.fetch(
            """
            SELECT uid as sound_unit_id, array_agg(sid) as sac_set
            FROM (
                SELECT unit_a_id as uid, unit_b_id as sid
                FROM sensory_cooccurrences
                WHERE modality_a = 'voice' AND modality_b = 'motor' AND reinforcement_count >= 2
                UNION ALL
                SELECT unit_b_id as uid, unit_a_id as sid
                FROM sensory_cooccurrences
                WHERE modality_b = 'voice' AND modality_a = 'motor' AND reinforcement_count >= 2
            ) pairs
            GROUP BY uid
            """,
        )
        for row in sac_rows:
            unit_saccades[row["sound_unit_id"]] = set(row["sac_set"])
    except Exception:
        pass

    # Compare all pairs
    unit_ids = list(unit_visuals.keys())
    for i in range(len(unit_ids)):
        for j in range(i + 1, len(unit_ids)):
            uid_a, uid_b = unit_ids[i], unit_ids[j]
            stats["pairs_checked"] += 1

            vis_a = unit_visuals[uid_a]
            vis_b = unit_visuals[uid_b]

            # Jaccard overlap of visual bindings
            intersection = len(vis_a & vis_b)
            union = len(vis_a | vis_b)
            visual_overlap = intersection / max(union, 1)

            # Saccade overlap (bonus evidence)
            sac_overlap = 0.0
            if uid_a in unit_saccades and uid_b in unit_saccades:
                sac_a = unit_saccades[uid_a]
                sac_b = unit_saccades[uid_b]
                sac_inter = len(sac_a & sac_b)
                sac_union = len(sac_a | sac_b)
                sac_overlap = sac_inter / max(sac_union, 1)

            # Combined overlap score (visual primary, saccade bonus)
            combined = visual_overlap * 0.7 + sac_overlap * 0.3

            if combined >= min_overlap:
                # Ensure canonical ordering (smaller ID first)
                a, b = min(uid_a, uid_b), max(uid_a, uid_b)

                result = await pool.execute(
                    """
                    INSERT INTO concept_links
                        (unit_a_id, unit_b_id, overlap_score, strength,
                         modality_evidence)
                    VALUES ($1, $2, $3, 1.0, $4)
                    ON CONFLICT (unit_a_id, unit_b_id)
                    DO UPDATE SET
                        overlap_score = $3,
                        strength = LEAST(concept_links.strength + 0.5, 10.0),
                        updated_at = NOW()
                    """,
                    a, b,
                    round(combined, 4),
                    "visual+saccade" if sac_overlap > 0 else "visual",
                )

                if "INSERT" in result:
                    stats["links_created"] += 1
                else:
                    stats["links_strengthened"] += 1

    # Weaken links that no longer hold (overlap decreased below threshold)
    weakened = await pool.execute(
        """
        UPDATE concept_links
        SET strength = GREATEST(strength - 0.2, 0),
            updated_at = NOW()
        WHERE updated_at < NOW() - INTERVAL '1 hour'
          AND strength > 0
        """,
    )
    stats["links_weakened"] = int(weakened.split()[-1])

    # Prune dead links
    pruned = await pool.execute(
        "DELETE FROM concept_links WHERE strength <= 0"
    )
    stats["links_pruned"] = int(pruned.split()[-1])

    total_links = await pool.fetchval("SELECT COUNT(*) FROM concept_links")
    stats["total_links"] = total_links

    if stats["links_created"] > 0 or stats["links_strengthened"] > 0:
        log.info(
            "Concept linking: checked %d pairs, %d new links, %d strengthened, %d total",
            stats["pairs_checked"], stats["links_created"],
            stats["links_strengthened"], total_links,
        )

    return stats


async def _homeostatic_decay(
    pool: asyncpg.Pool,
    elapsed_s: float = 30.0,
) -> dict:
    """Ebbinghaus decay: per-pair forgetting based on reinforcement history.

    Each co-occurrence decays at its own rate:
        strength *= 2^(-elapsed / half_life)

    Pairs with many reinforcements (long half-life) barely move.
    Pairs with few reinforcements (short half-life) fade fast.
    Myelinated pairs are exempt.

    Myelination emerges naturally when half_life exceeds the threshold
    AND the pair has enough reinforcements.
    """
    from nmem_sym_sensor.cooccurrence import cooc_store

    decay_stats = await cooc_store.decay_all(elapsed_s, pool)

    myelinated_count = await pool.fetchval(
        "SELECT COUNT(*) FROM sensory_cooccurrences WHERE myelinated"
    )
    decay_stats["myelinated_count"] = myelinated_count

    # Increment dreamstate_challenges for all active unmyelinated pairs
    await pool.execute("""
        UPDATE sensory_cooccurrences
        SET dreamstate_challenges = dreamstate_challenges + 1
        WHERE NOT myelinated AND strength > 0
    """)

    if decay_stats["decayed"] > 0 or decay_stats["archived"] > 0 or decay_stats["newly_myelinated"] > 0:
        log.info(
            "Ebbinghaus decay (%.0fs): %d decayed, %d archived, "
            "%d newly myelinated (%d total)",
            elapsed_s, decay_stats["decayed"], decay_stats["archived"],
            decay_stats["newly_myelinated"], myelinated_count,
        )

    return decay_stats


async def _self_play_round_trip(
    pool: asyncpg.Pool,
    reinforce_bonus: int = 2,
    weaken_penalty: int = 1,
    max_units: int = 50,
) -> dict:
    from nmem_sym_sensor.cooccurrence import TAU_0
    """Self-play: test associations by round-tripping between modalities.

    For each unit with co-occurrences in the unified table:
    1. Find its strongest partner in another modality (A → B)
    2. From that partner, find its strongest back (B → A)
    3. If round-trip returns original → reinforce (LTP)
    4. If returns different → weaken (LTD)

    Tests ALL modality pairs automatically (voice↔visual, motor↔voice, etc).
    """
    from nmem_sym_sensor.cooccurrence import canonical_order, cooc_store

    stats = {
        "tested": 0,
        "round_trips_passed": 0,
        "round_trips_failed": 0,
        "reinforced": 0,
        "weakened": 0,
    }

    # Modality-balanced sampling: ensure voice↔visual pairs get tested,
    # not crowded out by environmental/motor pairs.
    modality_quotas = [("voice", 15), ("visual", 15), ("motor", 10), ("environmental", 10)]
    units = []
    for _mod, _quota in modality_quotas:
        rows = await pool.fetch(
            """
            SELECT uid, mod FROM (
                SELECT DISTINCT unit_a_id as uid, modality_a as mod FROM sensory_cooccurrences
                WHERE modality_a = $2 AND reinforcement_count >= 3
                UNION
                SELECT DISTINCT unit_b_id as uid, modality_b as mod FROM sensory_cooccurrences
                WHERE modality_b = $2 AND reinforcement_count >= 3
            ) sub
            ORDER BY RANDOM() LIMIT $1
            """,
            _quota, _mod,
        )
        units.extend(rows)

    for row in units:
        start_id = row["uid"]
        start_mod = row["mod"]
        stats["tested"] += 1

        # Forward: find strongest cross-modal partner.
        # For voice, prefer visual partner. For visual, prefer voice.
        # This ensures voice↔visual round-trips actually get tested.
        _target_mod = None
        if start_mod == "voice":
            _target_mod = "visual"
        elif start_mod == "visual":
            _target_mod = "voice"
        forward = await cooc_store.query(start_id, start_mod, pool,
            target_modality=_target_mod, min_count=3, limit=1)
        # Fallback: if no cross-modal partner found, try any modality
        if not forward and _target_mod:
            forward = await cooc_store.query(start_id, start_mod, pool, min_count=3, limit=1)
        if not forward:
            continue

        partner_id = forward[0]["unit_id"]
        partner_mod = forward[0]["modality"]

        # Reverse: from partner, find strongest back in original modality
        reverse = await cooc_store.query(
            partner_id, partner_mod, pool,
            target_modality=start_mod, min_count=1, limit=1,
        )
        if not reverse:
            continue

        # Canonical form for the UPDATE
        a_id, a_mod, b_id, b_mod = canonical_order(
            start_id, start_mod, partner_id, partner_mod,
        )

        if reverse[0]["unit_id"] == start_id:
            # Round-trip passed — mutually consistent.
            # Bump count AND recompute half_life with self-play multiplier.
            # Self-play confirmations are the primary path to myelination.
            stats["round_trips_passed"] += 1
            await pool.execute(
                """
                UPDATE sensory_cooccurrences
                SET count = count + $1,
                    self_play_confirmations = self_play_confirmations + 1,
                    half_life = $6 * (
                        1 + LN(1 + reinforcement_count)
                    ) * (0.5 + encoding_quality)
                    * (1.0 + (self_play_confirmations + 1) / 10.0)
                WHERE unit_a_id = $2 AND modality_a = $3
                  AND unit_b_id = $4 AND modality_b = $5
                """,
                reinforce_bonus, a_id, a_mod, b_id, b_mod,
                TAU_0,
            )
            stats["reinforced"] += 1
        else:
            # Round-trip failed — noise binding
            stats["round_trips_failed"] += 1
            await pool.execute(
                """
                UPDATE sensory_cooccurrences
                SET count = GREATEST(count - $1, 0)
                WHERE unit_a_id = $2 AND modality_a = $3
                  AND unit_b_id = $4 AND modality_b = $5
                """,
                weaken_penalty, a_id, a_mod, b_id, b_mod,
            )
            stats["weakened"] += 1

    # Archive negative counts (safety)
    archived = await pool.execute(
        "UPDATE sensory_cooccurrences SET count = 0 WHERE count < 0 AND NOT myelinated"
    )
    stats["archived"] = int(archived.split()[-1])

    if stats["round_trips_passed"] > 0 or stats["round_trips_failed"] > 0:
        total = stats["round_trips_passed"] + stats["round_trips_failed"]
        consistency = stats["round_trips_passed"] / max(total, 1)
        log.info(
            "Self-play: %d tested, %d passed (%.0f%%), %d reinforced, %d weakened, %d pruned",
            stats["tested"],
            stats["round_trips_passed"],
            consistency * 100,
            stats["reinforced"],
            stats["weakened"],
            stats.get("pruned", 0),
        )

    return stats


async def _practice_syllables(
    pool: asyncpg.Pool,
    vocal_tract: "VocalTract | None" = None,
    max_syllables: int = 10,
) -> dict:
    """Dreamstate syllable practice: select syllables and practice producing them.

    Selection priority: high observation_count + low self_play_confirmations
    + not myelinated + has exemplar frames.

    Calls vocal_tract.practice_syllable() for each. If vocal_tract is None,
    only does selection (DB bookkeeping, no decode/encode).
    """
    stats = {"selected": 0, "practiced": 0, "reinforced": 0, "myelinated": 0, "skipped": 0}

    # Check if speech_syllables table exists
    table_exists = await pool.fetchval("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'speech_syllables'
        )
    """)
    if not table_exists:
        return stats

    # Select candidates with rotation — cycle through different syllables each call
    import random
    total_eligible = await pool.fetchval("""
        SELECT COUNT(*) FROM speech_syllables
        WHERE NOT myelinated AND exemplar_frames IS NOT NULL
    """)
    offset = random.randint(0, max(0, total_eligible - max_syllables)) if total_eligible > max_syllables else 0

    candidates = await pool.fetch(
        """
        SELECT id, unit_ids, observation_count, self_play_confirmations
        FROM speech_syllables
        WHERE NOT myelinated
          AND exemplar_frames IS NOT NULL
        ORDER BY observation_count DESC, self_play_confirmations ASC
        LIMIT $1 OFFSET $2
        """,
        max_syllables, offset,
    )

    stats["selected"] = len(candidates)

    if not candidates or vocal_tract is None:
        stats["skipped"] = len(candidates) if vocal_tract is None else 0
        return stats

    for c in candidates:
        try:
            result = await vocal_tract.practice_syllable(c["id"])
            if result.get("error"):
                continue
            stats["practiced"] += 1
            if result.get("reinforced"):
                stats["reinforced"] += 1
            if result.get("myelinated"):
                stats["myelinated"] += 1
        except Exception as e:
            log.debug("Syllable practice failed for %d: %s", c["id"], e)

    if stats["practiced"] > 0:
        log.info("Syllable practice: %d/%d practiced, %d reinforced, %d myelinated",
                 stats["practiced"], stats["selected"], stats["reinforced"], stats["myelinated"])

    return stats


async def _merge_duplicate_clusters(pool: asyncpg.Pool) -> int:
    """Merge clusters that have converged to be near-identical.

    Two clusters with >90% member overlap and similar centroids
    should be merged — they represent the same concept discovered
    from different starting points.
    """
    # Find pairs of clusters in the same modality with high member overlap
    # Uses subquery-based set intersection/union instead of intarray operators
    pairs = await pool.fetch(
        """
        WITH cluster_sizes AS (
            SELECT cluster_id, COUNT(*) as size
            FROM sensory_cluster_members
            GROUP BY cluster_id
        ),
        overlap AS (
            SELECT a.cluster_id as id_a, b.cluster_id as id_b,
                   COUNT(*) as shared
            FROM sensory_cluster_members a
            JOIN sensory_cluster_members b
              ON a.node_id = b.node_id AND a.cluster_id < b.cluster_id
            JOIN sensory_clusters ca ON ca.id = a.cluster_id
            JOIN sensory_clusters cb ON cb.id = b.cluster_id
            WHERE ca.modality = cb.modality
              AND ca.cluster_type IN ('stable', 'grounded')
              AND cb.cluster_type IN ('stable', 'grounded')
            GROUP BY a.cluster_id, b.cluster_id
        )
        SELECT o.id_a, o.id_b,
               sa.size as size_a, sb.size as size_b
        FROM overlap o
        JOIN cluster_sizes sa ON sa.cluster_id = o.id_a
        JOIN cluster_sizes sb ON sb.cluster_id = o.id_b
        WHERE o.shared::float / GREATEST(sa.size + sb.size - o.shared, 1) > 0.8
        LIMIT 10
        """
    )

    merged = 0
    for pair in pairs:
        # Keep the larger/older cluster, merge the smaller into it
        keep_id = pair["id_a"] if pair["size_a"] >= pair["size_b"] else pair["id_b"]
        merge_id = pair["id_b"] if keep_id == pair["id_a"] else pair["id_a"]

        # Move members (skip any that already exist in the target cluster)
        await pool.execute(
            """
            UPDATE sensory_cluster_members
            SET cluster_id = $1
            WHERE cluster_id = $2
              AND node_id NOT IN (
                  SELECT node_id FROM sensory_cluster_members WHERE cluster_id = $1
              )
            """,
            keep_id, merge_id,
        )
        # Delete remaining duplicates from merged cluster
        await pool.execute(
            "DELETE FROM sensory_cluster_members WHERE cluster_id = $1",
            merge_id,
        )

        # Update stats on the surviving cluster
        await pool.execute(
            """
            UPDATE sensory_clusters
            SET member_count = (SELECT COUNT(*) FROM sensory_cluster_members WHERE cluster_id = $1),
                updated_at = NOW()
            WHERE id = $1
            """,
            keep_id,
        )

        # Delete the now-empty merged cluster
        await pool.execute(
            "DELETE FROM sensory_clusters WHERE id = $1",
            merge_id,
        )

        merged += 1
        log.debug("Merged cluster #%d into #%d", merge_id, keep_id)

    return merged


async def _dream_reactivate_archived(pool: asyncpg.Pool) -> int:
    """Reactivate archived nodes that are similar to active cluster centroids.

    During dreamstate, the system "revisits" archived traces and checks
    if they fit patterns that have emerged since they were archived.

    An archived node that is very similar (cosine >= 0.8) to a grounded
    cluster's centroid probably represents the same concept — it was seen
    before the concept solidified. Reactivate it and add it to the cluster.

    Returns count of reactivated nodes.
    """
    import numpy as np

    # Get grounded clusters with visual centroids
    clusters = await pool.fetch(
        """
        SELECT id, grounded_label, visual_centroid::text as centroid
        FROM sensory_clusters
        WHERE cluster_type = 'grounded'
          AND visual_centroid IS NOT NULL
          AND modality = 'visual'
        """
    )

    if not clusters:
        return 0

    # Get archived visual nodes with embeddings
    archived = await pool.fetch(
        """
        SELECT id, label, visual_embedding::text as emb, observation_count
        FROM sensory_nodes
        WHERE archived = TRUE
          AND modality = 'visual'
          AND visual_embedding IS NOT NULL
        LIMIT 200
        """
    )

    if not archived:
        return 0

    # Parse cluster centroids
    cluster_data = []
    for c in clusters:
        emb_str = c["centroid"].strip("[]")
        if emb_str:
            vec = np.array([float(x) for x in emb_str.split(",")], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                cluster_data.append((c["id"], c["grounded_label"], vec / norm))

    if not cluster_data:
        return 0

    reactivated = 0
    threshold = 0.8

    for node in archived:
        emb_str = node["emb"].strip("[]")
        if not emb_str:
            continue
        vec = np.array([float(x) for x in emb_str.split(",")], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        vec_norm = vec / norm

        # Find best matching cluster
        best_sim = 0.0
        best_cluster = None
        best_label = None
        for cid, clabel, cvec in cluster_data:
            sim = float(np.dot(vec_norm, cvec))
            if sim > best_sim:
                best_sim = sim
                best_cluster = cid
                best_label = clabel

        if best_sim >= threshold and best_cluster is not None:
            # Reactivate the node
            await pool.execute(
                """
                UPDATE sensory_nodes
                SET archived = FALSE,
                    memory_tier = 'long_term',
                    updated_at = NOW()
                WHERE id = $1
                """,
                node["id"],
            )

            # Add to the matching cluster
            await pool.execute(
                """
                INSERT INTO sensory_cluster_members (cluster_id, node_id, role)
                VALUES ($1, $2, 'member')
                ON CONFLICT (cluster_id, node_id) DO NOTHING
                """,
                best_cluster, node["id"],
            )

            # Update cluster member count
            await pool.execute(
                """
                UPDATE sensory_clusters
                SET member_count = member_count + 1, updated_at = NOW()
                WHERE id = $1
                """,
                best_cluster,
            )

            reactivated += 1
            log.info("DREAM REACTIVATION: archived node %d ('%s') → "
                     "cluster '%s' (similarity=%.3f, obs=%d)",
                     node["id"], node["label"], best_label,
                     best_sim, node["observation_count"])

    return reactivated
