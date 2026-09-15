"""
Oculomotor system: saccade trajectory motor memory.

When the fovea scans a shape, the sequence of fixation points forms a
motor pattern — the eye trace IS how the system "knows" the shape.
A triangle produces 3 fixation points in a triangular arrangement.
A circle produces fixation points in a circular arrangement.

This module captures, encodes, clusters, and retrieves these saccade
trajectories. It follows the same EMA centroid clustering pattern as
SoundLanguage (language.py) for sound units.

The saccade trajectory is one dimension of the mind's eye:
  - Motor replay: the eye movement pattern (this module)
  - Visual sensation: what was seen at each fixation (visual embeddings)
  - Sound association: what was heard (formant embeddings / sound units)
"""
import logging

import asyncpg
import numpy as np

from nmem_sym_sensor import config

log = logging.getLogger(__name__)


class SaccadeMemory:
    """Stores and clusters saccade trajectories as motor memory.

    Each trajectory is encoded as a 24-dim vector (8 fixation points ×
    3 values: x, y, saliency). Trajectories are clustered by cosine
    similarity using EMA centroid updates — repeated viewings of
    triangles converge toward a canonical "triangle scan pattern."
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        similarity_threshold: float | None = None,
        max_fixations: int | None = None,
    ):
        self.pool = pool
        self.similarity_threshold = (
            similarity_threshold or config.SACCADE_SIMILARITY_THRESHOLD
        )
        self.max_fixations = max_fixations or config.SACCADE_MAX_FIXATIONS
        self.embed_dim = self.max_fixations * 3

    # ── Encoding ─────────────────────────────────────────

    @staticmethod
    def encode_trajectory(
        fixations: list[tuple[float, float, float]],
        max_points: int = 8,
    ) -> np.ndarray:
        """Encode a variable-length fixation sequence into a fixed-dim vector.

        Steps:
        1. Take up to max_points fixations in temporal order
        2. Normalize: centroid → origin, scale max extent → 1.0
        3. Saliency values sum to 1.0
        4. Zero-pad unused slots
        5. L2-normalize final vector

        Args:
            fixations: List of (x, y, saliency) in temporal order.
            max_points: Maximum fixation points to encode.

        Returns:
            L2-normalized numpy array of shape (max_points * 3,).
        """
        dim = max_points * 3
        vec = np.zeros(dim, dtype=np.float32)

        if not fixations:
            return vec

        pts = fixations[:max_points]
        n = len(pts)

        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        sals = np.array([p[2] for p in pts], dtype=np.float32)

        # Translate centroid to origin
        cx, cy = xs.mean(), ys.mean()
        xs -= cx
        ys -= cy

        # Scale max extent to 1.0
        max_extent = max(np.abs(xs).max(), np.abs(ys).max(), 1e-6)
        xs /= max_extent
        ys /= max_extent

        # Normalize saliency to sum to 1.0
        sal_sum = sals.sum()
        if sal_sum > 0:
            sals /= sal_sum

        # Pack into vector (temporal order preserved)
        for i in range(n):
            vec[i * 3] = xs[i]
            vec[i * 3 + 1] = ys[i]
            vec[i * 3 + 2] = sals[i]

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    # ── Observation ──────────────────────────────────────

    async def observe_trajectory(
        self,
        fixations: list[tuple[float, float, float]],
        timestamp: float,
        active_visual_ids: list[int] | None = None,
        active_sound_unit_ids: list[int] | None = None,
        active_words: list[str] | None = None,  # DEPRECATED, kept for backward compat
    ) -> int | None:
        """Observe a saccade trajectory and match or create a pattern.

        1. Encode fixations to embedding
        2. Find nearest pattern by cosine similarity (pgvector)
        3. Match → EMA centroid update
        4. Novel → create new pattern
        5. Record visual and sound symbol co-occurrences

        Returns pattern ID or None if fixations are empty.
        """
        if not fixations or len(fixations) < 1:
            return None

        embedding = self.encode_trajectory(fixations, self.max_fixations)

        # Skip if encoding is all zeros (degenerate case)
        if np.linalg.norm(embedding) < 0.01:
            return None

        emb_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

        # Find nearest existing pattern
        match = await self.pool.fetchrow(
            """
            SELECT id, centroid::text as centroid, total_observations,
                   shape_variance, fixation_count,
                   1 - (centroid <=> $1::vector) as similarity
            FROM saccade_patterns
            ORDER BY centroid <=> $1::vector
            LIMIT 1
            """,
            emb_str,
        )

        pattern_id = None

        if match and match["similarity"] >= self.similarity_threshold:
            # Merge into existing pattern via EMA centroid update
            pattern_id = match["id"]
            n = match["total_observations"]
            alpha = min(0.3, 1.0 / max(n, 1))

            # Parse existing centroid
            old_str = match["centroid"].strip("[]")
            old_centroid = np.array(
                [float(x) for x in old_str.split(",")], dtype=np.float32,
            )

            # EMA update
            new_centroid = (1 - alpha) * old_centroid + alpha * embedding
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid /= norm

            # Update shape variance (running variance of cosine distances)
            cos_dist = 1.0 - float(np.dot(old_centroid, embedding))
            old_var = match["shape_variance"]
            new_var = old_var + (cos_dist - old_var) / max(n, 1)

            new_centroid_str = (
                "[" + ",".join(f"{v:.6f}" for v in new_centroid) + "]"
            )

            await self.pool.execute(
                """
                UPDATE saccade_patterns
                SET centroid = $1::vector,
                    total_observations = total_observations + 1,
                    shape_variance = $2,
                    fixation_count = $3,
                    updated_at = NOW()
                WHERE id = $4
                """,
                new_centroid_str,
                round(new_var, 6),
                len(fixations),
                pattern_id,
            )
        else:
            # Create new pattern
            label = self.auto_label(len(fixations), embedding)
            row = await self.pool.fetchrow(
                """
                INSERT INTO saccade_patterns
                    (centroid, fixation_count, label)
                VALUES ($1::vector, $2, $3)
                RETURNING id
                """,
                emb_str,
                len(fixations),
                label,
            )
            pattern_id = row["id"]
            log.debug(
                "New saccade pattern #%d: %s (%d fixations)",
                pattern_id, label, len(fixations),
            )

        # Record co-occurrences (unified table)
        if pattern_id:
            from nmem_sym_sensor.cooccurrence import cooc_store

            if active_visual_ids:
                for vid in active_visual_ids[:10]:
                    await cooc_store.observe(pattern_id, "motor", vid, "visual", self.pool)

            if active_sound_unit_ids:
                for sid in active_sound_unit_ids[:10]:
                    await cooc_store.observe(pattern_id, "motor", sid, "voice", self.pool)

        # DEPRECATED: text co-occurrences (kept for diagnostic dashboard)
        if pattern_id and active_words:
            for word in active_words[:5]:
                w = word.strip().lower()
                if w:
                    try:
                        await self.pool.execute(
                            """
                            INSERT INTO saccade_word_cooccurrences
                                (saccade_pattern_id, word)
                            VALUES ($1, $2)
                            ON CONFLICT (saccade_pattern_id, word)
                            DO UPDATE SET count = saccade_word_cooccurrences.count + 1,
                                          last_seen = NOW()
                            """,
                            pattern_id, w,
                        )
                    except Exception:
                        pass  # table may not exist in new installs

        return pattern_id

    # ── Retrieval (for imagination) ──────────────────────

    async def recall_by_visual(
        self, visual_node_ids: list[int],
    ) -> list[dict]:
        """Recall saccade patterns associated with visual concepts."""
        if not visual_node_ids:
            return []

        rows = await self.pool.fetch(
            """
            SELECT sp.id, sp.centroid::text as centroid, sp.fixation_count,
                   sp.label, sp.total_observations, sp.shape_variance,
                   SUM(sc.count) as cooccurrence
            FROM saccade_patterns sp
            JOIN sensory_cooccurrences sc
              ON (sc.unit_a_id = sp.id AND sc.modality_a = 'motor' AND sc.modality_b = 'visual' AND sc.unit_b_id = ANY($1))
              OR (sc.unit_b_id = sp.id AND sc.modality_b = 'motor' AND sc.modality_a = 'visual' AND sc.unit_a_id = ANY($1))
            WHERE sc.count > 0
            GROUP BY sp.id
            ORDER BY cooccurrence DESC
            LIMIT 5
            """,
            visual_node_ids,
        )

        return [self._row_to_dict(r) for r in rows]

    async def recall_by_sound(self, sound_unit_id: int) -> list[dict]:
        """Recall saccade patterns associated with a sound symbol."""
        from nmem_sym_sensor.cooccurrence import cooc_store
        results = await cooc_store.query(sound_unit_id, "voice", self.pool, target_modality="motor")
        if not results:
            return []
        pattern_ids = [r["unit_id"] for r in results]
        rows = await self.pool.fetch(
            """
            SELECT id, centroid::text as centroid, fixation_count,
                   label, total_observations, shape_variance
            FROM saccade_patterns WHERE id = ANY($1)
            """,
            pattern_ids,
        )
        return [self._row_to_dict(r) for r in rows]

    async def recall_by_label(self, label: str) -> list[dict]:
        """Convenience: look up sound unit by STT label, then recall saccade.

        This is the human-facing API — "what does the eye do when hearing
        this word?" Internally it resolves the word to a sound unit first.
        """
        unit = await self.pool.fetchrow(
            "SELECT id FROM sound_units WHERE stt_label = $1 ORDER BY total_observations DESC LIMIT 1",
            label.strip().lower(),
        )
        if not unit:
            return []
        return await self.recall_by_sound(unit["id"])

    async def recall_by_word(self, word: str) -> list[dict]:
        """DEPRECATED: Recall via text co-occurrences. Use recall_by_label() instead."""
        try:
            rows = await self.pool.fetch(
                """
                SELECT sp.id, sp.centroid::text as centroid, sp.fixation_count,
                       sp.label, sp.total_observations, sp.shape_variance,
                       swc.count as cooccurrence
                FROM saccade_patterns sp
                JOIN saccade_word_cooccurrences swc ON swc.saccade_pattern_id = sp.id
                WHERE swc.word = $1
                ORDER BY swc.count DESC
                LIMIT 5
                """,
                word.strip().lower(),
            )
            return [self._row_to_dict(r) for r in rows]
        except Exception:
            return []  # table may not exist

    async def recall_by_embedding(
        self, trajectory_embedding: np.ndarray, limit: int = 5,
    ) -> list[dict]:
        """Find similar saccade patterns by trajectory embedding."""
        emb_str = "[" + ",".join(f"{v:.6f}" for v in trajectory_embedding) + "]"
        rows = await self.pool.fetch(
            """
            SELECT id, centroid::text as centroid, fixation_count,
                   label, total_observations, shape_variance,
                   1 - (centroid <=> $1::vector) as similarity
            FROM saccade_patterns
            ORDER BY centroid <=> $1::vector
            LIMIT $2
            """,
            emb_str, limit,
        )
        return [self._row_to_dict(r) for r in rows]

    # ── Consolidation ────────────────────────────────────

    async def consolidate(self) -> dict:
        """Merge converged patterns and prune orphans."""
        stats = {"merged": 0, "pruned": 0}

        # Prune patterns with only 1 observation and older than 1 hour
        result = await self.pool.execute(
            """
            DELETE FROM saccade_patterns
            WHERE total_observations = 1
              AND updated_at < NOW() - INTERVAL '1 hour'
            """,
        )
        stats["pruned"] = int(result.split()[-1])

        total = await self.pool.fetchval("SELECT COUNT(*) FROM saccade_patterns")
        stats["total_patterns"] = total

        return stats

    # ── Auto-labelling ───────────────────────────────────

    @staticmethod
    def auto_label(fixation_count: int, embedding: np.ndarray) -> str:
        """Generate a descriptive label based on trajectory shape."""
        if fixation_count <= 1:
            return "point_fixation"
        elif fixation_count == 2:
            return "linear_scan"
        elif fixation_count == 3:
            return "triangular_scan"
        elif fixation_count == 4:
            return "quadrilateral_scan"
        else:
            return f"complex_scan_{fixation_count}pt"

    # ── Helpers ──────────────────────────────────────────

    @staticmethod
    def decode_trajectory(
        centroid_str: str, max_points: int = 8,
    ) -> list[tuple[float, float, float]]:
        """Decode a centroid string back into fixation points."""
        vals = [float(x) for x in centroid_str.strip("[]").split(",")]
        fixations = []
        for i in range(0, len(vals), 3):
            x, y, sal = vals[i], vals[i + 1], vals[i + 2]
            if abs(x) < 1e-8 and abs(y) < 1e-8 and abs(sal) < 1e-8:
                break  # zero-padded slot
            fixations.append((x, y, sal))
        return fixations

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        if "centroid" in d and d["centroid"]:
            d["fixations"] = SaccadeMemory.decode_trajectory(d["centroid"])
        return d
