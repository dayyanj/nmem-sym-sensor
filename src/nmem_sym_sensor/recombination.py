"""Dreamstate Recombination — creative consolidation through context switching.

Inspired by how dreams strip memories from their original context and
recombine them with unrelated content to discover which associations
are universal truths vs context-specific bindings.

Three cognitive functions:
1. Generalise — extract abstract patterns from specific events
2. Simulate — run patterns through different contexts
3. Evaluate — test if associations hold outside their original context

Runs as a step in the sensory dreamstate cycle. One recombination per
cycle is enough — effects accumulate over many cycles.

Over many cycles:
- Universal associations get slightly stronger -> eventually myelinate
- Context-specific associations get slightly weaker -> eventually go dormant
"""
import logging
import random
from typing import Any

import asyncpg

from nmem_sym_sensor.cooccurrence import cooc_store

log = logging.getLogger(__name__)

# Strength multipliers (gentle — accumulate over many cycles)
REINFORCE_FACTOR = 1.05
CONFLICT_FACTOR = 0.95
# Minimum spatial compatibility to proceed (even dreams have some coherence)
MIN_SPATIAL_COMPAT = 0.2
# Scene selection: probability of picking a random (exploratory) scene
RANDOM_SCENE_PROB = 0.30


async def dreamstate_recombination(
    pool: asyncpg.Pool,
    max_candidates: int = 1,
    sym_pool: asyncpg.Pool | None = None,
    bridge: Any | None = None,
) -> dict:
    """Run one recombination cycle.

    Orchestrates: select candidate -> pick target scene -> check spatial
    compatibility -> spread activation -> evaluate -> record.

    Args:
        pool: Sensory DB pool.
        max_candidates: How many candidates to process per cycle.
        sym_pool: nmem-sym DB pool (for hypothesis handoff).
        bridge: nmem-sym SymbolBridge instance (for hypothesis handoff).

    Returns stats dict.
    """
    stats = {
        "candidates_tried": 0,
        "recombinations_completed": 0,
        "skipped_no_scene": 0,
        "skipped_spatial": 0,
        "total_reinforcing": 0,
        "total_conflicting": 0,
        "total_novel": 0,
    }

    for _ in range(max_candidates):
        candidate = await _select_recombination_candidate(pool)
        if candidate is None:
            break

        stats["candidates_tried"] += 1
        cluster_id = candidate["cluster_id"]
        home_scene_id = candidate["home_scene_id"]

        # Phase 2: pick a target scene
        target_scene_id = await _select_target_scene(
            pool, cluster_id, home_scene_id,
        )
        if target_scene_id is None:
            stats["skipped_no_scene"] += 1
            continue

        # Phase 3: two-stage spatial compatibility check
        spatial_score = await _check_spatial_compatibility(
            pool, cluster_id, target_scene_id,
        )
        if spatial_score < MIN_SPATIAL_COMPAT:
            stats["skipped_spatial"] += 1
            continue

        # Phase 4: spreading activation
        reinforcing, conflicting, novel = await _spread_activation(
            pool, cluster_id, target_scene_id,
        )

        # Phase 5: evaluate — adjust strengths and context counters
        await _evaluate_recombination(
            pool, cluster_id, reinforcing, conflicting,
        )

        # Phase 6: record
        n_r, n_c, n_n = len(reinforcing), len(conflicting), len(novel)
        total = n_r + n_c
        coherence = n_r / max(total, 1)

        rec_id = await _record_recombination(
            pool,
            candidate_cluster_id=cluster_id,
            candidate_home_scene_id=home_scene_id,
            target_scene_id=target_scene_id,
            spatial_compatibility=spatial_score,
            n_reinforcing=n_r,
            n_conflicting=n_c,
            n_novel=n_n,
            coherence_score=coherence,
        )

        # Phase 7: discover novel paths and emit hypotheses to nmem-sym
        novel_paths = await _discover_novel_paths(
            pool, cluster_id, target_scene_id,
            reinforcing, novel,
        )
        if novel_paths:
            from nmem_sym_sensor.sensory_bridge import emit_hypothesis_candidate
            emitted = 0
            for path in novel_paths:
                path["recombination_id"] = rec_id
                ok = await emit_hypothesis_candidate(sym_pool, bridge, path)
                if ok:
                    emitted += 1
            stats["hypotheses_emitted"] = stats.get("hypotheses_emitted", 0) + emitted

        stats["recombinations_completed"] += 1
        stats["total_reinforcing"] += n_r
        stats["total_conflicting"] += n_c
        stats["total_novel"] += n_n

        log.info(
            "Recombination: cluster %d (scene %s) -> scene %d | "
            "spatial=%.2f reinforcing=%d conflicting=%d novel=%d coherence=%.2f",
            cluster_id, home_scene_id, target_scene_id,
            spatial_score, n_r, n_c, n_n, coherence,
        )

    return stats


# ── Phase 1: Selection ────────────────────────────────────


async def _select_recombination_candidate(
    pool: asyncpg.Pool,
) -> dict | None:
    """Pick a cluster that needs recombination testing.

    Criteria:
    - Recently active co-occurrences (last 24h)
    - NOT myelinated (already consolidated = skip)
    - Moderate reinforcement count (enough signal)
    - Not recombined in the last hour (avoid re-testing)

    Returns {cluster_id, home_scene_id} or None.
    """
    # Co-occurrences use node IDs, clusters use their own IDs.
    # Join through cluster_members to bridge the ID spaces.
    rows = await pool.fetch(
        """
        SELECT sc.id as cluster_id, sc.grounded_label,
               sc.total_observations,
               MAX(co.last_reinforced) as last_active
        FROM sensory_clusters sc
        JOIN sensory_cluster_members cm ON cm.cluster_id = sc.id
        JOIN sensory_cooccurrences co ON
            (co.unit_a_id = cm.node_id AND co.modality_a = 'visual')
            OR (co.unit_b_id = cm.node_id AND co.modality_b = 'visual')
        WHERE NOT co.myelinated
          AND co.reinforcement_count >= 5
          AND co.last_reinforced > NOW() - INTERVAL '24 hours'
          AND sc.cluster_type IN ('stable', 'grounded')
          AND sc.id NOT IN (
              SELECT candidate_cluster_id
              FROM dreamstate_recombinations
              WHERE created_at > NOW() - INTERVAL '1 hour'
          )
        GROUP BY sc.id
        ORDER BY MAX(co.last_reinforced) DESC
        LIMIT 10
        """
    )

    if not rows:
        return None

    # Random pick from top 10 (varied selection = more diverse recombination)
    chosen = random.choice(rows)
    cluster_id = chosen["cluster_id"]

    # Find home scene: the scene where this cluster has the most observations
    home_scene_id = await pool.fetchval(
        """
        SELECT scene_id FROM scene_members
        WHERE cluster_id = $1
        ORDER BY observation_count DESC
        LIMIT 1
        """,
        cluster_id,
    )

    return {"cluster_id": cluster_id, "home_scene_id": home_scene_id}


# ── Phase 2: Context Switch ──────────────────────────────


async def _select_target_scene(
    pool: asyncpg.Pool,
    candidate_id: int,
    home_scene_id: int | None,
) -> int | None:
    """Pick a target scene different from the candidate's home.

    Strategy: 70% similar scene (shared cluster types), 30% random.
    """
    # Need at least 2 scenes to do recombination
    scene_count = await pool.fetchval("SELECT COUNT(*) FROM scene_snapshots")
    if scene_count < 2:
        return None

    exclude_ids = [home_scene_id] if home_scene_id is not None else []

    if random.random() < RANDOM_SCENE_PROB:
        # Exploratory: random scene (excluding home)
        return await _pick_random_scene(pool, exclude_ids)

    # Similar: find scenes that share some cluster modalities but not
    # this specific cluster
    candidate_modalities = await pool.fetch(
        """
        SELECT DISTINCT
            CASE WHEN co.unit_a_id = $1 THEN co.modality_b
                 ELSE co.modality_a END as mod
        FROM sensory_cooccurrences co
        WHERE (co.unit_a_id = $1 OR co.unit_b_id = $1)
          AND co.count > 0
        """,
        candidate_id,
    )
    mods = [r["mod"] for r in candidate_modalities]

    if mods:
        # Find scenes whose members have co-occurrences in the same modalities
        similar = await pool.fetch(
            """
            SELECT sm.scene_id, COUNT(DISTINCT sm.cluster_id) as overlap
            FROM scene_members sm
            JOIN sensory_cooccurrences co ON
                (co.unit_a_id = sm.cluster_id OR co.unit_b_id = sm.cluster_id)
                AND co.count > 0
            WHERE sm.scene_id != ALL($1::int[])
              AND (co.modality_a = ANY($2::text[]) OR co.modality_b = ANY($2::text[]))
            GROUP BY sm.scene_id
            HAVING COUNT(DISTINCT sm.cluster_id) >= 2
            ORDER BY COUNT(DISTINCT sm.cluster_id) DESC
            LIMIT 5
            """,
            exclude_ids,
            mods,
        )
        if similar:
            chosen = random.choice(similar)
            return chosen["scene_id"]

    # Fallback: random scene
    return await _pick_random_scene(pool, exclude_ids)


async def _pick_random_scene(
    pool: asyncpg.Pool,
    exclude_ids: list[int],
) -> int | None:
    """Pick a random scene, excluding given IDs."""
    row = await pool.fetchrow(
        """
        SELECT id FROM scene_snapshots
        WHERE id != ALL($1::int[])
        ORDER BY RANDOM()
        LIMIT 1
        """,
        exclude_ids,
    )
    return row["id"] if row else None


# ── Phase 3: Spatial Compatibility ────────────────────────


async def _check_spatial_compatibility(
    pool: asyncpg.Pool,
    candidate_id: int,
    target_scene_id: int,
) -> float:
    """Two-stage spatial plausibility check.

    Stage 1 (cheap): does the target scene have any anchor clusters
    (i.e. any spatial edges at all)? If the scene is spatially empty,
    score 0.5 (neutral — no evidence for or against).

    Stage 2 (full): load the candidate's typical spatial relations and
    the target scene's spatial graph. Score = fraction of the candidate's
    relations that can be satisfied in the target scene.
    """
    # Stage 1: does target scene have any spatial structure?
    edge_count = await pool.fetchval(
        "SELECT COUNT(*) FROM scene_spatial_edges WHERE scene_id = $1",
        target_scene_id,
    )
    if edge_count == 0:
        # No spatial graph yet — neutral score (don't block early scenes)
        return 0.5

    # Stage 2: full relation matching
    # Get the candidate's typical spatial relations (from its home scenes)
    candidate_relations = await pool.fetch(
        """
        SELECT DISTINCT relation
        FROM scene_spatial_edges
        WHERE cluster_a_id = $1 OR cluster_b_id = $1
        """,
        candidate_id,
    )
    if not candidate_relations:
        # Candidate has no spatial history — neutral
        return 0.5

    candidate_rels = {r["relation"] for r in candidate_relations}

    # Get the target scene's available relation types
    target_relations = await pool.fetch(
        """
        SELECT DISTINCT relation
        FROM scene_spatial_edges
        WHERE scene_id = $1
        """,
        target_scene_id,
    )
    target_rels = {r["relation"] for r in target_relations}

    # Score: what fraction of the candidate's typical relations exist
    # in the target scene? e.g. candidate usually "on" something —
    # does target scene have any "on" relations?
    if not candidate_rels:
        return 0.5

    compatible = len(candidate_rels & target_rels)
    score = compatible / len(candidate_rels)

    return score


# ── Phase 4: Activation Spread ────────────────────────────


async def _spread_activation(
    pool: asyncpg.Pool,
    candidate_id: int,
    target_scene_id: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Spread activation from candidate into target scene context.

    Activates candidate's co-occurrences and the target scene's existing
    associations. Classifies each candidate co-occurrence as:
    - reinforcing: same (unit_id, modality) appears in scene context
    - conflicting: same modality slot but different unit
    - novel: modality not present in scene context at all

    Returns (reinforcing, conflicting, novel) lists of co-occurrence dicts.
    """
    # Candidate's co-occurrences (what it usually appears with)
    candidate_coocs = await cooc_store.recall(candidate_id, "visual", pool)

    # Target scene's cluster IDs
    scene_member_rows = await pool.fetch(
        "SELECT cluster_id FROM scene_members WHERE scene_id = $1",
        target_scene_id,
    )
    scene_cluster_ids = {r["cluster_id"] for r in scene_member_rows}

    # Build the scene's activation map: for each scene cluster, gather
    # its co-occurrences. Result: {modality: set of unit_ids}
    scene_activation: dict[str, set[int]] = {}
    for sc_id in scene_cluster_ids:
        sc_coocs = await cooc_store.recall(sc_id, "visual", pool, min_count=2)
        for mod, entries in sc_coocs.items():
            scene_activation.setdefault(mod, set())
            for entry in entries:
                scene_activation[mod].add(entry["unit_id"])

    # Classify each of the candidate's co-occurrences
    reinforcing: list[dict] = []
    conflicting: list[dict] = []
    novel: list[dict] = []

    for mod, entries in candidate_coocs.items():
        if mod not in scene_activation:
            # This modality doesn't exist in the scene at all — novel
            for e in entries:
                novel.append({"unit_id": e["unit_id"], "modality": mod, **e})
            continue

        scene_units = scene_activation[mod]
        for e in entries:
            info = {"unit_id": e["unit_id"], "modality": mod, **e}
            if e["unit_id"] in scene_units:
                # Same unit in same modality exists in scene — reinforcing
                reinforcing.append(info)
            else:
                # Same modality but different unit — conflicting
                conflicting.append(info)

    return reinforcing, conflicting, novel


# ── Phase 5: Evaluation ──────────────────────────────────


async def _evaluate_recombination(
    pool: asyncpg.Pool,
    candidate_id: int,
    reinforcing: list[dict],
    conflicting: list[dict],
) -> None:
    """Apply gentle strength adjustments based on recombination outcome.

    Universal associations (reinforcing) get a small boost.
    Context-specific associations (conflicting) get a small penalty.
    Novel associations are left untouched.
    """
    from nmem_sym_sensor.cooccurrence import canonical_order

    for cooc in reinforcing:
        a_id, a_mod, b_id, b_mod = canonical_order(
            candidate_id, "visual", cooc["unit_id"], cooc["modality"],
        )
        await pool.execute(
            """
            UPDATE sensory_cooccurrences
            SET strength = strength * $1,
                context_independence = context_independence + 1
            WHERE unit_a_id = $2 AND modality_a = $3
              AND unit_b_id = $4 AND modality_b = $5
              AND NOT myelinated
            """,
            REINFORCE_FACTOR, a_id, a_mod, b_id, b_mod,
        )

    for cooc in conflicting:
        a_id, a_mod, b_id, b_mod = canonical_order(
            candidate_id, "visual", cooc["unit_id"], cooc["modality"],
        )
        await pool.execute(
            """
            UPDATE sensory_cooccurrences
            SET strength = strength * $1,
                context_dependence = context_dependence + 1
            WHERE unit_a_id = $2 AND modality_a = $3
              AND unit_b_id = $4 AND modality_b = $5
              AND NOT myelinated
            """,
            CONFLICT_FACTOR, a_id, a_mod, b_id, b_mod,
        )


# ── Phase 6: Record ──────────────────────────────────────


async def _record_recombination(
    pool: asyncpg.Pool,
    *,
    candidate_cluster_id: int,
    candidate_home_scene_id: int | None,
    target_scene_id: int,
    spatial_compatibility: float,
    n_reinforcing: int,
    n_conflicting: int,
    n_novel: int,
    coherence_score: float,
) -> int:
    """Persist the recombination result for future analysis."""
    rec_id = await pool.fetchval(
        """
        INSERT INTO dreamstate_recombinations
            (candidate_cluster_id, candidate_home_scene_id, target_scene_id,
             spatial_compatibility, n_reinforcing, n_conflicting, n_novel,
             coherence_score)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """,
        candidate_cluster_id, candidate_home_scene_id, target_scene_id,
        spatial_compatibility, n_reinforcing, n_conflicting, n_novel,
        coherence_score,
    )
    return rec_id


# ── Phase 7: Novel Path Discovery ────────────────────────


async def _discover_novel_paths(
    pool: asyncpg.Pool,
    candidate_id: int,
    target_scene_id: int,
    reinforcing: list[dict],
    novel: list[dict],
    max_paths: int = 5,
    min_plausibility: float = 0.1,
) -> list[dict]:
    """Discover novel activation paths from recombination spreading.

    When a candidate is placed in a new scene, its co-occurrences activate
    alongside the scene's co-occurrences. Some combinations don't exist as
    co-occurrences yet — these are novel paths (hypothesis candidates).

    The most interesting paths come from combining:
    - A reinforcing co-occurrence of the candidate (confirmed to work here)
    - A novel co-occurrence the candidate brings (new to this scene)

    Plausibility = product of the constituent strengths (normalized).

    Returns list of hypothesis candidate dicts ready for emit_hypothesis_candidate.
    """
    from nmem_sym_sensor.cooccurrence import canonical_order

    if not reinforcing or not novel:
        return []

    # For each reinforcing entry paired with each novel entry,
    # check whether that specific pair already exists as a co-occurrence.
    # If not, it's a novel path — a dream-generated hypothesis.
    paths: list[dict] = []

    for r_entry in reinforcing:
        for n_entry in novel:
            r_id, r_mod = r_entry["unit_id"], r_entry["modality"]
            n_id, n_mod = n_entry["unit_id"], n_entry["modality"]

            if r_id == n_id and r_mod == n_mod:
                continue

            # Check if this pair already exists
            a_id, a_mod, b_id, b_mod = canonical_order(r_id, r_mod, n_id, n_mod)
            existing = await pool.fetchval(
                """
                SELECT count FROM sensory_cooccurrences
                WHERE unit_a_id = $1 AND modality_a = $2
                  AND unit_b_id = $3 AND modality_b = $4
                  AND strength > 0
                """,
                a_id, a_mod, b_id, b_mod,
            )
            if existing:
                continue  # Already known — not novel

            # Plausibility from constituent strengths
            r_strength = r_entry.get("strength", 0.5)
            n_strength = n_entry.get("strength", 0.5)
            plausibility = (r_strength * n_strength) ** 0.5  # geometric mean

            if plausibility < min_plausibility:
                continue

            paths.append({
                "source": "dreamstate_recombination",
                "unit_a": r_id,
                "modality_a": r_mod,
                "unit_b": n_id,
                "modality_b": n_mod,
                "plausibility": round(plausibility, 4),
                "context": {
                    "candidate_cluster": candidate_id,
                    "target_scene": target_scene_id,
                },
            })

            if len(paths) >= max_paths:
                break
        if len(paths) >= max_paths:
            break

    if paths:
        log.info(
            "Novel paths discovered: %d (from %d reinforcing x %d novel)",
            len(paths), len(reinforcing), len(novel),
        )

    return paths
