"""Unified N-to-N sensory co-occurrence store with Ebbinghaus decay.

Any sensor modality can bind to any other — visual↔voice, voice↔olfactory,
motor↔visual, etc. Single table, canonical ordering, strength-based encoding
with per-pair half-lives that increase with reinforcement.

Equations:
    Observation weight:  w = attention * surprise * proximity
    Strength update:     S_new = S_old * 2^(-dt/tau) + w
    Half-life:           tau = tau_0 * (1 + ln(1+n)) * (0.5 + q_bar)
    Encoding quality:    q_bar = running average of observation weights
    Myelination:         tau > TAU_MYELINATE and n > N_MIN_MYELINATE

Usage:
    store = CooccurrenceStore()
    await store.observe(vis_id, "visual", snd_id, "voice", pool,
                        attention=0.9, surprise=0.6, proximity=0.8)
    results = await store.query(vis_id, "visual", pool)
    recall = await store.recall(vis_id, "visual", pool)
"""
import logging
import math

import asyncpg

log = logging.getLogger(__name__)

# Constants
TAU_0 = 300.0               # Base half-life (seconds of dreamstate time)
TAU_MYELINATE = 86400.0      # Half-life threshold for myelination (1 day)
N_MIN_MYELINATE = 50         # Minimum reinforcements for myelination
STRENGTH_ARCHIVE = 0.01      # Archive below this strength
TAU_BIND = 2.0               # Temporal binding window (seconds)


def canonical_order(
    id_a: int, mod_a: str, id_b: int, mod_b: str,
) -> tuple[int, str, int, str]:
    """Ensure consistent (A,B) ordering to prevent duplicates.

    Rules:
    - modality_a <= modality_b (lexicographic)
    - If same modality, unit_a_id < unit_b_id
    """
    if mod_a > mod_b or (mod_a == mod_b and id_a > id_b):
        return id_b, mod_b, id_a, mod_a
    return id_a, mod_a, id_b, mod_b


def compute_half_life(
    reinforcement_count: int,
    encoding_quality: float,
    self_play_confirmations: int = 0,
) -> float:
    """Compute half-life from reinforcement history and dreamstate testing.

    More reinforcements and higher quality observations = longer half-life.
    Self-play confirmations provide a multiplier — associations that survive
    dreamstate round-trip testing grow half-life much faster. This is the
    primary path to myelination: not just frequent exposure, but *proven*
    consistency through dreaming.

    Returns seconds.
    """
    base = TAU_0 * (1 + math.log(1 + reinforcement_count)) * (0.5 + encoding_quality)
    # Self-play multiplier: 10 confirmations = 2x, 50 = 6x, 100 = 11x
    sp_mult = 1.0 + self_play_confirmations / 10.0
    return base * sp_mult


class CooccurrenceStore:
    """Unified N-to-N sensory co-occurrence storage with Ebbinghaus decay.

    Each co-occurrence tracks:
    - strength: weighted sum of observations, decayed over time
    - reinforcement_count: raw observation count
    - encoding_quality: running average of observation weights
    - half_life: computed from reinforcement history
    - myelinated: protected from decay when half-life exceeds threshold
    """

    TABLE = "sensory_cooccurrences"

    # Lightweight counter for pressure system — incremented on every observe()
    observations_since_reset: int = 0

    async def observe(
        self,
        unit_a: int, modality_a: str,
        unit_b: int, modality_b: str,
        pool: asyncpg.Pool,
        attention: float = 1.0,
        surprise: float = 0.5,
        proximity: float = 1.0,
    ):
        """Record a co-occurrence with quality-weighted encoding.

        The observation weight w = attention * surprise * proximity modulates
        how strongly this observation is encoded. High attention during
        surprising events near the observer produces stronger memories.

        On update: existing strength decays by elapsed time, then new
        evidence adds on top. Half-life is recomputed from updated history.
        """
        a_id, a_mod, b_id, b_mod = canonical_order(
            unit_a, modality_a, unit_b, modality_b,
        )
        weight = max(attention * surprise * proximity, 0.01)
        initial_hl = compute_half_life(1, weight)

        await pool.execute(
            f"""
            INSERT INTO {self.TABLE}
                (unit_a_id, modality_a, unit_b_id, modality_b,
                 count, strength, reinforcement_count, encoding_quality,
                 half_life, last_reinforced)
            VALUES ($1, $2, $3, $4,
                    1, $5, 1, $5, $6, NOW())
            ON CONFLICT (unit_a_id, modality_a, unit_b_id, modality_b)
            DO UPDATE SET
                count = {self.TABLE}.count + 1,
                strength = CASE
                    WHEN EXTRACT(EPOCH FROM NOW() - {self.TABLE}.last_reinforced)
                         / GREATEST({self.TABLE}.half_life, 1.0) > 20.0
                    THEN $5
                    ELSE {self.TABLE}.strength
                        * POWER(2.0, -EXTRACT(EPOCH FROM NOW() - {self.TABLE}.last_reinforced)
                                / GREATEST({self.TABLE}.half_life, 1.0))
                        + $5
                END,
                reinforcement_count = {self.TABLE}.reinforcement_count + 1,
                encoding_quality = (
                    ({self.TABLE}.reinforcement_count * {self.TABLE}.encoding_quality) + $5
                ) / ({self.TABLE}.reinforcement_count + 1),
                half_life = $7 * (
                    1 + LN(1 + {self.TABLE}.reinforcement_count + 1)
                ) * (
                    0.5 + (
                        ({self.TABLE}.reinforcement_count * {self.TABLE}.encoding_quality) + $5
                    ) / ({self.TABLE}.reinforcement_count + 1)
                ) * (1.0 + COALESCE({self.TABLE}.self_play_confirmations, 0) / 10.0),
                last_reinforced = NOW(),
                last_seen = NOW()
            """,
            a_id, a_mod, b_id, b_mod,
            weight,         # $5
            initial_hl,     # $6
            TAU_0,          # $7
        )
        self.observations_since_reset += 1

    async def observe_batch(
        self,
        unit_id: int, modality: str,
        targets: list[tuple[int, str]],
        pool: asyncpg.Pool,
        attention: float = 1.0,
        surprise: float = 0.5,
        proximity: float = 1.0,
    ):
        """Record co-occurrences between one unit and multiple targets."""
        for t_id, t_mod in targets:
            await self.observe(
                unit_id, modality, t_id, t_mod, pool,
                attention=attention, surprise=surprise, proximity=proximity,
            )

    async def query(
        self,
        unit_id: int, modality: str,
        pool: asyncpg.Pool,
        target_modality: str | None = None,
        min_count: int = 1,
        limit: int = 50,
    ) -> list[dict]:
        """Find co-occurrences for a unit, ranked by strength (not count).

        Returns list of {unit_id, modality, count, strength, half_life, myelinated}.
        """
        if target_modality:
            rows = await pool.fetch(
                f"""
                SELECT other_id as unit_id, other_mod as modality,
                       count, strength, half_life, myelinated
                FROM (
                    SELECT unit_b_id as other_id, modality_b as other_mod,
                           count, strength, half_life, myelinated
                    FROM {self.TABLE}
                    WHERE unit_a_id = $1 AND modality_a = $2 AND reinforcement_count >= $3
                    UNION ALL
                    SELECT unit_a_id as other_id, modality_a as other_mod,
                           count, strength, half_life, myelinated
                    FROM {self.TABLE}
                    WHERE unit_b_id = $1 AND modality_b = $2 AND reinforcement_count >= $3
                ) combined
                WHERE other_mod = $5
                ORDER BY strength DESC
                LIMIT $4
                """,
                unit_id, modality, min_count, limit, target_modality,
            )
        else:
            rows = await pool.fetch(
                f"""
                SELECT other_id as unit_id, other_mod as modality,
                       count, strength, half_life, myelinated
                FROM (
                    SELECT unit_b_id as other_id, modality_b as other_mod,
                           count, strength, half_life, myelinated
                    FROM {self.TABLE}
                    WHERE unit_a_id = $1 AND modality_a = $2 AND reinforcement_count >= $3
                    UNION ALL
                    SELECT unit_a_id as other_id, modality_a as other_mod,
                           count, strength, half_life, myelinated
                    FROM {self.TABLE}
                    WHERE unit_b_id = $1 AND modality_b = $2 AND reinforcement_count >= $3
                ) combined
                ORDER BY strength DESC
                LIMIT $4
                """,
                unit_id, modality, min_count, limit,
            )
        return [dict(r) for r in rows]

    async def recall(
        self,
        unit_id: int, modality: str,
        pool: asyncpg.Pool,
        min_count: int = 1,
    ) -> dict[str, list[dict]]:
        """Full cross-modal recall: given one unit, what lights up across all senses?

        Returns {modality: [{unit_id, count, strength, myelinated}, ...]}.
        Ranked by strength within each modality.
        """
        all_coocs = await self.query(unit_id, modality, pool, min_count=min_count, limit=200)

        by_modality: dict[str, list[dict]] = {}
        for r in all_coocs:
            mod = r["modality"]
            by_modality.setdefault(mod, []).append(r)

        return by_modality

    async def displace(
        self,
        winner_id: int, winner_modality: str,
        target_id: int, target_modality: str,
        pool: asyncpg.Pool,
        displacement: int = 1,
    ):
        """Darwinian displacement: weaken competitors on a target node.

        Reduces strength (not just count) of competing associations.
        Only displaces within the same modality.
        """
        disp_strength = float(displacement)
        # Competitors where target is on the A side
        await pool.execute(
            f"""
            UPDATE {self.TABLE}
            SET count = GREATEST(0, count - $1),
                strength = GREATEST(0, strength - $6)
            WHERE strength > 0 AND NOT myelinated
              AND unit_a_id = $2 AND modality_a = $3
              AND modality_b = $4 AND unit_b_id != $5
            """,
            displacement,
            target_id, target_modality,
            winner_modality, winner_id,
            disp_strength,
        )
        # Competitors where target is on the B side
        await pool.execute(
            f"""
            UPDATE {self.TABLE}
            SET count = GREATEST(0, count - $1),
                strength = GREATEST(0, strength - $6)
            WHERE strength > 0 AND NOT myelinated
              AND unit_b_id = $2 AND modality_b = $3
              AND modality_a = $4 AND unit_a_id != $5
            """,
            displacement,
            target_id, target_modality,
            winner_modality, winner_id,
            disp_strength,
        )

    async def decay_all(
        self,
        elapsed_s: float,
        pool: asyncpg.Pool,
    ) -> dict:
        """Apply Ebbinghaus decay to all unmyelinated co-occurrences.

        Each pair decays at its own rate based on half_life:
            strength *= 2^(-elapsed / half_life)

        Pairs with long half-lives (heavily reinforced) barely move.
        Pairs with short half-lives (barely seen) fade fast.

        Returns stats dict.
        """
        # Apply per-row decay
        result = await pool.execute(
            f"""
            UPDATE {self.TABLE}
            SET strength = CASE
                WHEN ($1::float8) / GREATEST(half_life, 1.0) > 20.0 THEN 0
                ELSE strength * POWER(2.0, -($1::float8) / GREATEST(half_life, 1.0))
            END
            WHERE NOT myelinated AND strength > 0
            """,
            float(elapsed_s),
        )
        decayed = int(result.split()[-1])

        # Archive pairs that dropped below threshold — zero strength but
        # KEEP count and reinforcement_count intact. These are historical
        # observation records that should not be reset on archival.
        result2 = await pool.execute(
            f"""
            UPDATE {self.TABLE}
            SET strength = 0
            WHERE strength < $1 AND strength > 0 AND NOT myelinated
            """,
            STRENGTH_ARCHIVE,
        )
        archived = int(result2.split()[-1])

        # Check for myelination candidates
        result3 = await pool.execute(
            f"""
            UPDATE {self.TABLE}
            SET myelinated = TRUE
            WHERE half_life > $1
              AND reinforcement_count > $2
              AND NOT myelinated
              AND strength > 0
            """,
            TAU_MYELINATE, N_MIN_MYELINATE,
        )
        myelinated = int(result3.split()[-1])

        return {
            "decayed": decayed,
            "archived": archived,
            "newly_myelinated": myelinated,
            "elapsed_s": elapsed_s,
        }

    async def find_competitive_targets(
        self,
        source_modality: str,
        target_modality: str,
        pool: asyncpg.Pool,
        min_count: int = 5,
    ) -> list[dict]:
        """Find target units with 2+ competing source units."""
        rows = await pool.fetch(
            f"""
            SELECT target_id, COUNT(*) as n_competitors,
                   MAX(strength) as max_strength
            FROM (
                SELECT unit_a_id as target_id, unit_b_id as source_id, strength
                FROM {self.TABLE}
                WHERE modality_a = $1 AND modality_b = $2 AND reinforcement_count >= $3
                UNION ALL
                SELECT unit_b_id as target_id, unit_a_id as source_id, strength
                FROM {self.TABLE}
                WHERE modality_b = $1 AND modality_a = $2 AND reinforcement_count >= $3
            ) pairs
            GROUP BY target_id
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC
            """,
            target_modality, source_modality, min_count,
        )

        results = []
        for r in rows:
            target_id = r["target_id"]
            competitors = await self.query(
                target_id, target_modality, pool,
                target_modality=source_modality, min_count=1, limit=2,
            )
            if len(competitors) < 2:
                continue
            results.append({
                "target_id": target_id,
                "n_competitors": r["n_competitors"],
                "dominant_id": competitors[0]["unit_id"],
                "dominant_count": competitors[0].get("reinforcement_count", competitors[0].get("count", 1)),
                "dominant_strength": competitors[0]["strength"],
                "runner_up_count": competitors[1].get("reinforcement_count", competitors[1].get("count", 1)),
                "runner_up_strength": competitors[1]["strength"],
            })

        return results

    async def find_spread_units(
        self,
        source_modality: str,
        target_modality: str,
        pool: asyncpg.Pool,
        spread_threshold: int = 5,
    ) -> list[dict]:
        """Find units that bind to too many targets (promiscuous binding)."""
        rows = await pool.fetch(
            f"""
            SELECT source_id as unit_id, COUNT(DISTINCT target_id) as spread,
                   SUM(strength) as total_strength
            FROM (
                SELECT unit_b_id as source_id, unit_a_id as target_id, strength
                FROM {self.TABLE}
                WHERE modality_b = $1 AND modality_a = $2 AND strength > 0
                UNION ALL
                SELECT unit_a_id as source_id, unit_b_id as target_id, strength
                FROM {self.TABLE}
                WHERE modality_a = $1 AND modality_b = $2 AND strength > 0
            ) pairs
            GROUP BY source_id
            HAVING COUNT(DISTINCT target_id) >= $3
            ORDER BY COUNT(DISTINCT target_id) DESC
            """,
            source_modality, target_modality, spread_threshold,
        )
        return [dict(r) for r in rows]

    async def get_unit_bindings(
        self,
        unit_id: int, unit_modality: str,
        target_modality: str,
        pool: asyncpg.Pool,
    ) -> list[dict]:
        """Get all bindings for a unit to a specific modality with embeddings."""
        rows = await pool.fetch(
            f"""
            SELECT target_id, count, strength, n.visual_embedding::text as emb
            FROM (
                SELECT unit_b_id as target_id, count, strength
                FROM {self.TABLE}
                WHERE unit_a_id = $1 AND modality_a = $2 AND modality_b = $3 AND strength > 0
                UNION ALL
                SELECT unit_a_id as target_id, count, strength
                FROM {self.TABLE}
                WHERE unit_b_id = $1 AND modality_b = $2 AND modality_a = $3 AND strength > 0
            ) pairs
            JOIN sensory_nodes n ON n.id = pairs.target_id
            WHERE n.visual_embedding IS NOT NULL
            ORDER BY strength DESC
            """,
            unit_id, unit_modality, target_modality,
        )
        return [dict(r) for r in rows]

    async def archive_spread(
        self,
        unit_id: int, unit_modality: str,
        keep_target_ids: list[int],
        target_modality: str,
        pool: asyncpg.Pool,
    ) -> int:
        """Archive all bindings for a unit EXCEPT the strongest ones."""
        r1 = await pool.execute(
            f"""
            UPDATE {self.TABLE} SET count = 0, strength = 0
            WHERE unit_a_id = $1 AND modality_a = $2 AND modality_b = $3
              AND unit_b_id != ALL($4::bigint[])
              AND strength > 0 AND NOT myelinated
            """,
            unit_id, unit_modality, target_modality, keep_target_ids,
        )
        r2 = await pool.execute(
            f"""
            UPDATE {self.TABLE} SET count = 0, strength = 0
            WHERE unit_b_id = $1 AND modality_b = $2 AND modality_a = $3
              AND unit_a_id != ALL($4::bigint[])
              AND strength > 0 AND NOT myelinated
            """,
            unit_id, unit_modality, target_modality, keep_target_ids,
        )
        return int(r1.split()[-1]) + int(r2.split()[-1])

    async def stats(self, pool: asyncpg.Pool) -> dict:
        """Overview stats for the co-occurrence table."""
        row = await pool.fetchrow(f"""
            SELECT
                count(*) as total_pairs,
                count(*) FILTER (WHERE strength > 0) as active_pairs,
                count(*) FILTER (WHERE count = 0) as dormant_pairs,
                count(*) FILTER (WHERE myelinated) as myelinated_pairs,
                coalesce(sum(count) FILTER (WHERE strength > 0), 0) as total_obs,
                coalesce(avg(strength) FILTER (WHERE strength > 0), 0) as avg_strength,
                coalesce(avg(half_life) FILTER (WHERE strength > 0), 0) as avg_half_life,
                coalesce(max(half_life) FILTER (WHERE strength > 0), 0) as max_half_life
            FROM {self.TABLE}
        """)
        modalities = await pool.fetch(f"""
            SELECT modality_a || '↔' || modality_b as pair,
                   count(*) as pairs,
                   coalesce(sum(count), 0) as obs,
                   coalesce(avg(strength), 0) as avg_strength,
                   coalesce(avg(half_life), 0) as avg_half_life
            FROM {self.TABLE}
            WHERE strength > 0
            GROUP BY modality_a, modality_b
            ORDER BY sum(strength) DESC
        """)
        return {
            "total_pairs": row["total_pairs"],
            "active_pairs": row["active_pairs"],
            "dormant_pairs": row["dormant_pairs"],
            "myelinated_pairs": row["myelinated_pairs"],
            "total_obs": row["total_obs"],
            "avg_strength": round(float(row["avg_strength"]), 2),
            "avg_half_life": round(float(row["avg_half_life"]), 0),
            "max_half_life": round(float(row["max_half_life"]), 0),
            "by_modality": [dict(m) for m in modalities],
        }


# Module-level singleton for convenience
cooc_store = CooccurrenceStore()
