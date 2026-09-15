"""
Emergent language acquisition from sound-visual observation.

Language is learned the way a child learns it: by hearing sounds while
seeing things. No grammar rules, no text corpus, no STT required.
Recurring sound patterns cluster into "sound units" (phoneme/word-like).
Their sequences reveal structure ("the" always precedes nouns).
Their visual bindings reveal meaning ("elephant" sound + elephant image).

Three layers:
  1. Sound units: speaker-fuzzy audio clusters (same word from man/woman/child → same unit)
  2. Sound sequences: which units follow which (emergent grammar)
  3. Sound-visual bindings: which sounds mean which things (grounded semantics)

STT is optional bootstrapping — it labels sound units with text when
available, but the units exist and learn independently of text.
"""
import io
import logging
import zlib

import asyncpg
import numpy as np

from nmem_sym_sensor import config

log = logging.getLogger(__name__)


def _compress_frames(frames: dict[str, np.ndarray]) -> bytes:
    """Compress a dict of numpy arrays to zlib-compressed bytes."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **frames)
    return zlib.compress(buf.getvalue(), level=1)


def _decompress_frames(data: bytes) -> dict[str, np.ndarray]:
    """Decompress bytes back to a dict of numpy arrays."""
    raw = zlib.decompress(data)
    npz = np.load(io.BytesIO(raw))
    return dict(npz)


async def init_speech_hierarchy(pool: asyncpg.Pool):
    """Idempotent belt for the speech-hierarchy tables.

    The authoritative schema is migration 006_speech_hierarchy.sql (run by
    SensorGraph.connect() before this is called). This remains only for standalone
    use without the nmem migration runner; keep its DDL byte-identical to 006.
    """
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS speech_syllables (
            id                      SERIAL PRIMARY KEY,
            unit_ids                INTEGER[] NOT NULL,
            exemplar_frames         BYTEA,
            exemplar_duration_ms    FLOAT,
            confidence              FLOAT DEFAULT 0.0,
            half_life               FLOAT DEFAULT 300.0,
            self_play_confirmations INTEGER DEFAULT 0,
            myelinated              BOOLEAN DEFAULT FALSE,
            stt_label               TEXT,
            observation_count       INTEGER DEFAULT 0,
            total_duration_ms       FLOAT DEFAULT 0.0,
            created_at              TIMESTAMPTZ DEFAULT NOW(),
            updated_at              TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    await pool.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_syllables_unit_ids
            ON speech_syllables (unit_ids)
    """)
    await pool.execute("""
        CREATE INDEX IF NOT EXISTS idx_syllables_first_unit
            ON speech_syllables ((unit_ids[1]))
    """)
    await pool.execute("""
        CREATE INDEX IF NOT EXISTS idx_syllables_myelinated
            ON speech_syllables (myelinated) WHERE myelinated = TRUE
    """)
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS syllable_sequences (
            id               SERIAL PRIMARY KEY,
            from_syllable_id INTEGER REFERENCES speech_syllables(id) ON DELETE CASCADE,
            to_syllable_id   INTEGER REFERENCES speech_syllables(id) ON DELETE CASCADE,
            count            INTEGER DEFAULT 1,
            confidence       FLOAT DEFAULT 0.0,
            gap_ms_avg       FLOAT DEFAULT 0.0,
            created_at       TIMESTAMPTZ DEFAULT NOW(),
            updated_at       TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (from_syllable_id, to_syllable_id)
        )
    """)
    log.info("Speech hierarchy tables initialized")


class SoundLanguage:
    """Emergent language learning from sound-visual observation.

    Usage::

        lang = SoundLanguage(pool)

        # During video processing, for each audio window:
        unit_id = await lang.observe_sound(
            audio_embedding=embedding,
            timestamp=t,
            active_visual_ids=[42, 43],
            stt_text="elephant",  # optional
        )

        # Track sequences (which sounds follow which)
        await lang.record_sequence(prev_unit_id, unit_id, gap_ms=150)

        # Periodically consolidate (merge, prune, discover phrases)
        stats = await lang.consolidate()
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        similarity_threshold: float | None = None,
        max_units: int = 5000,
    ):
        self.pool = pool
        self.similarity_threshold = (
            similarity_threshold or config.SOUND_SIMILARITY_THRESHOLD
        )
        self.max_units = max_units
        self._global_centroid: np.ndarray | None = None
        self._global_count: int = 0

    async def observe_sound(
        self,
        audio_embedding: list[float],
        timestamp: float,
        active_visual_ids: list[int] | None = None,
        stt_text: str | None = None,
        stt_confidence: float = 0.0,
        modality: str = "voice",
        sub_features: dict[str, list[float]] | None = None,
        attention: float = 1.0,
        surprise: float = 0.5,
        proximity: float = 1.0,
        voice_frames: np.ndarray | None = None,
        sub_feature_frames: dict[str, np.ndarray] | None = None,
        burst_energy: float = 0.0,
    ) -> int | None:
        """Observe a sound and match/create a sound unit.

        1. Speaker-normalize embedding (subtract global centroid)
        2. Find nearest sound unit by embedding similarity
        3. If similar enough, merge into existing unit (EMA centroid)
        4. If novel, create new sound unit
        5. Record visual co-occurrence if visual nodes active
        6. Optionally attach STT label (diagnostic only, not a learning signal)

        Returns sound_unit_id or None if embedding is empty.
        """
        if audio_embedding is None:
            return None

        emb = np.array(audio_embedding, dtype=np.float32).flatten()
        if emb.shape[0] == 0:
            log.warning("observe_sound: empty embedding, skipping")
            return None
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return None
        emb = emb / norm  # L2 normalise for cosine search

        # Speaker normalization: subtract running global centroid to remove
        # speaker-specific bias. "triangle" from a man and a woman produce
        # different raw embeddings but similar speaker-normalized embeddings.
        if config.SOUND_SPEAKER_NORMALIZE:
            self._global_count += 1
            if self._global_centroid is None:
                self._global_centroid = emb.copy()
            else:
                alpha_g = 1.0 / self._global_count
                self._global_centroid = (
                    (1 - alpha_g) * self._global_centroid + alpha_g * emb
                )
            emb_norm = emb - self._global_centroid
            norm2 = np.linalg.norm(emb_norm)
            if norm2 > 1e-8:
                emb_norm = emb_norm / norm2
            else:
                emb_norm = emb  # fallback if centroid cancels out
        else:
            emb_norm = emb

        # Safety: ensure embedding is 1-D and non-empty before DB query
        if emb_norm.ndim != 1 or emb_norm.shape[0] == 0:
            log.warning("observe_sound: bad emb_norm shape %s (input shape was %s), skipping",
                        emb_norm.shape, np.array(audio_embedding).shape)
            return None

        # Warmup: merge aggressively at first (broad categories), tighten later
        total_units = await self.pool.fetchval(
            "SELECT COUNT(*) FROM sound_units",
        )
        if total_units < config.SOUND_WARMUP_UNTIL:
            threshold = config.SOUND_WARMUP_THRESHOLD
        else:
            threshold = self.similarity_threshold

        # Find nearest existing unit (filtered by modality)
        unit_id, similarity = await self._find_nearest_unit(emb_norm, modality)

        if unit_id is not None and similarity >= threshold:
            # Merge into existing unit — EMA centroid update
            row = await self.pool.fetchrow(
                "SELECT centroid::text as c, total_observations, speaker_variance "
                "FROM sound_units WHERE id = $1",
                unit_id,
            )
            if row and row["c"]:
                old_centroid = np.array(
                    [float(x) for x in row["c"].strip("[]").split(",")],
                )
                n = row["total_observations"]
                alpha = min(0.3, 1.0 / max(n, 1))
                new_centroid = (1 - alpha) * old_centroid + alpha * emb_norm
                new_centroid = new_centroid / max(
                    np.linalg.norm(new_centroid), 1e-8,
                )

                # Running speaker variance
                dist = float(1.0 - np.dot(old_centroid, emb_norm))
                old_var = row["speaker_variance"] or 0.0
                new_var = (1 - alpha) * old_var + alpha * dist

                # Build sub-feature update clause (inline values)
                sub_updates = ""
                sub_col_map = {
                    "pitch_embedding": "pitch_centroid",
                    "harmonic_embedding": "harmonic_centroid",
                    "formant_embedding": "formant_centroid",
                    "vad_embedding": "vad_centroid",
                    "spectral_embedding": "spectral_centroid",
                }
                if sub_features:
                    for feat_key, col_name in sub_col_map.items():
                        if feat_key in sub_features:
                            feat_arr = np.array(sub_features[feat_key], dtype=np.float32)
                            vec_str = str(feat_arr.tolist())
                            sub_updates += f", {col_name} = '{vec_str}'::vector"

                await self.pool.execute(
                    f"""
                    UPDATE sound_units
                    SET centroid = $1::vector,
                        total_observations = total_observations + 1,
                        speaker_variance = $2,
                        updated_at = NOW()
                        {sub_updates}
                    WHERE id = $3
                    """,
                    str(new_centroid.tolist()), round(new_var, 6), unit_id,
                )
        else:
            # Create new sound unit with sub-feature centroids
            # Build SET clauses for sub-features (inline values to avoid
            # asyncpg parameter type inference issues with vector columns)
            sub_col_map = {
                "pitch_embedding": "pitch_centroid",
                "harmonic_embedding": "harmonic_centroid",
                "formant_embedding": "formant_centroid",
                "vad_embedding": "vad_centroid",
                "spectral_embedding": "spectral_centroid",
            }
            sub_cols = ""
            sub_vals = ""
            if sub_features:
                for feat_key, col_name in sub_col_map.items():
                    if feat_key in sub_features:
                        feat_arr = np.array(sub_features[feat_key], dtype=np.float32)
                        vec_str = str(feat_arr.tolist())
                        sub_cols += f", {col_name}"
                        sub_vals += f", '{vec_str}'::vector"

            unit_id = await self.pool.fetchval(
                f"""
                INSERT INTO sound_units (centroid, member_count, total_observations, modality{sub_cols})
                VALUES ($1::vector, 1, 1, $2{sub_vals})
                RETURNING id
                """,
                str(emb_norm.tolist()),
                modality,
            )
            log.debug("New sound unit #%d (threshold=%.2f, nearest_sim=%.3f)",
                      unit_id, threshold, similarity or 0.0)

        # Attach STT label if available
        if stt_text and stt_confidence > 0.3 and unit_id:
            clean_text = stt_text.strip().lower().rstrip(".,!?")
            if len(clean_text) > 1:
                await self.pool.execute(
                    """
                    UPDATE sound_units
                    SET stt_label = CASE
                            WHEN stt_observations IS NULL OR stt_observations = 0 THEN $1
                            WHEN $2 > COALESCE(stt_confidence, 0) THEN $1
                            ELSE stt_label
                        END,
                        stt_confidence = GREATEST(COALESCE(stt_confidence, 0), $2),
                        stt_observations = COALESCE(stt_observations, 0) + 1,
                        updated_at = NOW()
                    WHERE id = $3
                    """,
                    clean_text, stt_confidence, unit_id,
                )

        # Store exemplar frame sequence if this observation has higher energy
        if voice_frames is not None and unit_id and burst_energy > 0:
            current_energy = await self.pool.fetchval(
                "SELECT exemplar_energy FROM sound_units WHERE id = $1", unit_id,
            )
            if current_energy is None or burst_energy > current_energy:
                frames_dict = {"voice": voice_frames}
                if sub_feature_frames:
                    frames_dict.update(sub_feature_frames)
                compressed = _compress_frames(frames_dict)
                await self.pool.execute(
                    """UPDATE sound_units
                       SET exemplar_frames = $1,
                           exemplar_n_frames = $2,
                           exemplar_energy = $3,
                           updated_at = NOW()
                       WHERE id = $4""",
                    compressed, voice_frames.shape[0],
                    round(burst_energy, 4), unit_id,
                )

        # Record sound-visual co-occurrence + Darwinian displacement
        if active_visual_ids and unit_id:
            from nmem_sym_sensor.cooccurrence import cooc_store

            for vid in active_visual_ids:
                await cooc_store.observe(
                    unit_id, modality, vid, "visual", self.pool,
                    attention=attention, surprise=surprise, proximity=proximity,
                )
                # Competing sound units on this visual node weaken
                await cooc_store.displace(unit_id, modality, vid, "visual", self.pool)

        return unit_id

    async def record_sequence(
        self,
        prev_unit_id: int,
        curr_unit_id: int,
        gap_ms: float,
    ):
        """Record that sound unit B followed sound unit A.

        Builds sequential co-occurrence statistics. Over thousands
        of observations, recurring patterns emerge:
        - "the" sound always precedes noun sounds
        - "is" sound appears between noun and adjective sounds
        - Certain pairs form tight bigrams (compound words)
        """
        if prev_unit_id == curr_unit_id:
            return  # skip self-loops

        await self.pool.execute(
            """
            INSERT INTO sound_sequences (from_unit_id, to_unit_id, direction, gap_ms_avg, count)
            VALUES ($1, $2, 'follows', $3, 1)
            ON CONFLICT (from_unit_id, to_unit_id, direction)
            DO UPDATE SET
                count = sound_sequences.count + 1,
                gap_ms_avg = (sound_sequences.gap_ms_avg * (sound_sequences.count - 1) + $3)
                             / sound_sequences.count,
                confidence = LEAST(1.0, sound_sequences.count::float / 20.0),
                updated_at = NOW()
            """,
            prev_unit_id, curr_unit_id, gap_ms,
        )

    async def _find_nearest_unit(
        self,
        embedding: np.ndarray,
        modality: str = "voice",
    ) -> tuple[int | None, float]:
        """Find the nearest existing sound unit by cosine similarity.

        Uses pgvector HNSW index for efficient search.
        Filters by modality so voice and environmental units don't mix.
        Returns (unit_id, similarity) or (None, 0).
        """
        row = await self.pool.fetchrow(
            """
            SELECT id, 1 - (centroid <=> $1::vector) as similarity
            FROM sound_units
            WHERE modality = $2
            ORDER BY centroid <=> $1::vector
            LIMIT 1
            """,
            str(embedding.tolist()),
            modality,
        )
        if row:
            return row["id"], float(row["similarity"])
        return None, 0.0

    async def discover_phrases(
        self,
        min_sequence_count: int = 10,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Discover recurring sound sequences (emergent words/phrases).

        Finds bigrams with high count and consistent timing, then
        chains them into longer sequences where possible.

        Returns list of discovered phrases with:
        - unit_ids: list of sound unit IDs in sequence
        - labels: STT labels for each unit (if available)
        - count: how many times this sequence was observed
        - avg_gap_ms: average timing between units
        - visual_bindings: visual nodes bound to these units
        """
        # Find strong bigrams
        bigrams = await self.pool.fetch(
            """
            SELECT s.from_unit_id, s.to_unit_id, s.count, s.gap_ms_avg, s.confidence,
                   a.stt_label as label_a, b.stt_label as label_b,
                   a.total_observations as obs_a, b.total_observations as obs_b
            FROM sound_sequences s
            JOIN sound_units a ON a.id = s.from_unit_id
            JOIN sound_units b ON b.id = s.to_unit_id
            WHERE s.count >= $1
              AND s.confidence >= $2
            ORDER BY s.count DESC
            LIMIT 500
            """,
            min_sequence_count, min_confidence,
        )

        phrases = []
        for b in bigrams:
            phrase = {
                "unit_ids": [b["from_unit_id"], b["to_unit_id"]],
                "labels": [b["label_a"], b["label_b"]],
                "count": b["count"],
                "avg_gap_ms": round(b["gap_ms_avg"], 1),
                "confidence": round(b["confidence"], 3),
            }

            # Find visual bindings for both units via unified table
            from nmem_sym_sensor.cooccurrence import cooc_store
            bindings = []
            for uid in [b["from_unit_id"], b["to_unit_id"]]:
                coocs = await cooc_store.query(uid, "voice", self.pool,
                                                target_modality="visual", min_count=3, limit=3)
                for c in coocs:
                    lbl = await self.pool.fetchval("SELECT label FROM sensory_nodes WHERE id=$1", c["unit_id"])
                    bindings.append({"sound_unit_id": uid, "visual_node_id": c["unit_id"],
                                     "count": c["count"], "visual_label": lbl or "?"})
            phrase["visual_bindings"] = [
                {"visual": bind["visual_label"], "count": bind["count"]}
                for bind in bindings
            ]
            phrases.append(phrase)

        return phrases

    async def _build_syllable_exemplar(
        self, unit_ids: list[int],
    ) -> tuple[bytes | None, float]:
        """Concatenate exemplar frames from constituent sound units.

        Fetches exemplar_frames for each unit, decompresses, concatenates
        the 'voice' arrays along time axis with inter-unit gaps derived
        from sound_sequences.gap_ms_avg.

        Returns (compressed_bytes, total_duration_ms) or (None, 0.0).
        """
        all_frames = []
        total_ms = 0.0
        frame_rate = 200  # TABULA2 frame rate (5ms per frame)

        for i, uid in enumerate(unit_ids):
            row = await self.pool.fetchrow(
                "SELECT exemplar_frames, exemplar_n_frames FROM sound_units WHERE id = $1",
                uid,
            )
            if row is None or row["exemplar_frames"] is None:
                return None, 0.0

            frames_dict = _decompress_frames(row["exemplar_frames"])
            voice = frames_dict.get("voice")
            if voice is None or len(voice) == 0:
                return None, 0.0

            all_frames.append(voice)
            total_ms += len(voice) / frame_rate * 1000

            # Add inter-unit gap (silence frames) between units
            if i < len(unit_ids) - 1:
                gap_row = await self.pool.fetchrow(
                    """SELECT gap_ms_avg FROM sound_sequences
                       WHERE from_unit_id = $1 AND to_unit_id = $2""",
                    uid, unit_ids[i + 1],
                )
                gap_ms = gap_row["gap_ms_avg"] if gap_row else 150.0
                gap_frames = max(1, int(gap_ms / 1000 * frame_rate))
                dim = voice.shape[1] if voice.ndim == 2 else voice.shape[0]
                silence = np.zeros((gap_frames, dim), dtype=voice.dtype)
                all_frames.append(silence)
                total_ms += gap_ms

        concatenated = np.concatenate(all_frames, axis=0)
        compressed = _compress_frames({"voice": concatenated})
        return compressed, total_ms

    async def discover_syllables(self) -> list[dict]:
        """Discover syllable-level chunks from phoneme transition chains.

        Scans sound_sequences for chains of 2-4 phonemes where every
        consecutive transition has confidence >= threshold and the chain
        has been observed at least min_count times. Total duration must
        fall within 50-500ms.

        Creates or updates speech_syllables records.
        Returns list of discovered syllable dicts.
        """
        min_count = config.SYLLABLE_MIN_CHAIN_COUNT
        min_conf = config.SYLLABLE_MIN_CONFIDENCE
        min_ms = config.SYLLABLE_MIN_DURATION_MS
        max_ms = config.SYLLABLE_MAX_DURATION_MS
        max_len = config.SYLLABLE_MAX_LENGTH

        # Fetch high-confidence transitions
        bigrams = await self.pool.fetch(
            """
            SELECT from_unit_id, to_unit_id, count, gap_ms_avg, confidence
            FROM sound_sequences
            WHERE confidence >= $1 AND count >= $2
            ORDER BY count DESC
            LIMIT 200
            """,
            min_conf, min_count,
        )

        if not bigrams:
            return []

        # Build directed adjacency graph
        graph: dict[int, list[tuple[int, int, float, float]]] = {}
        for b in bigrams:
            src = b["from_unit_id"]
            graph.setdefault(src, []).append((
                b["to_unit_id"], b["count"],
                b["gap_ms_avg"], b["confidence"],
            ))

        # Find chains of length 2..max_len
        candidates: list[tuple[list[int], int, float]] = []
        for start in graph:
            # BFS-like chain extension
            stack = [([start], float("inf"), 0.0)]
            while stack:
                chain, chain_count, chain_ms = stack.pop()
                if len(chain) >= 2:
                    if min_ms <= chain_ms <= max_ms:
                        candidates.append((chain[:], chain_count, chain_ms))
                if len(chain) >= max_len + 1:
                    continue
                last = chain[-1]
                for to_id, cnt, gap_ms, conf in graph.get(last, []):
                    if to_id in chain:  # no cycles
                        continue
                    new_count = min(chain_count, cnt)
                    new_ms = chain_ms + gap_ms
                    if new_ms > max_ms:
                        continue
                    stack.append((chain + [to_id], new_count, new_ms))

        if not candidates:
            return []

        # Deduplicate and sort by chain_count descending
        seen = set()
        unique = []
        for chain, cnt, ms in sorted(candidates, key=lambda x: -x[1]):
            key = tuple(chain)
            if key not in seen:
                seen.add(key)
                unique.append((chain, cnt, ms))

        # UPSERT into speech_syllables (top 50)
        results = []
        for chain, cnt, ms in unique[:200]:
            # Get STT labels for diagnostic
            labels = []
            for uid in chain:
                lbl = await self.pool.fetchval(
                    "SELECT stt_label FROM sound_units WHERE id = $1", uid,
                )
                labels.append(lbl)
            stt_label = "-".join(lab or "?" for lab in labels)

            # Build exemplar
            exemplar_bytes, exemplar_ms = await self._build_syllable_exemplar(chain)

            row = await self.pool.fetchrow(
                """
                INSERT INTO speech_syllables (unit_ids, observation_count, total_duration_ms,
                    stt_label, exemplar_frames, exemplar_duration_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (unit_ids) DO UPDATE SET
                    observation_count = speech_syllables.observation_count + $2,
                    total_duration_ms = $3,
                    stt_label = COALESCE($4, speech_syllables.stt_label),
                    exemplar_frames = COALESCE($5, speech_syllables.exemplar_frames),
                    exemplar_duration_ms = COALESCE($6, speech_syllables.exemplar_duration_ms),
                    updated_at = NOW()
                RETURNING id, observation_count
                """,
                chain, cnt, ms, stt_label, exemplar_bytes, exemplar_ms,
            )
            results.append({
                "syllable_id": row["id"],
                "unit_ids": chain,
                "stt_label": stt_label,
                "observation_count": row["observation_count"],
                "total_duration_ms": round(ms, 1),
                "has_exemplar": exemplar_bytes is not None,
            })

        if results:
            log.info("Syllable discovery: %d syllables found", len(results))

        return results

    async def consolidate(self) -> dict:
        """Consolidation cycle for sound language.

        1. Merge sound units with centroids too close (converged)
        2. Prune orphan units (no sequences, no bindings, low observations)
        3. Discover phrase patterns
        4. Discover syllable chunks
        5. Update stats
        """
        stats = {}

        # 1. Merge converged units (centroids > 0.90 similarity)
        merge_pairs = await self.pool.fetch(
            """
            SELECT a.id as id_a, b.id as id_b,
                   1 - (a.centroid <=> b.centroid) as similarity,
                   a.total_observations as obs_a, b.total_observations as obs_b
            FROM sound_units a
            JOIN sound_units b ON a.id < b.id
            WHERE 1 - (a.centroid <=> b.centroid) > 0.90
            LIMIT 20
            """,
        )

        merged = 0
        for pair in merge_pairs:
            # Keep the one with more observations
            keep_id = pair["id_a"] if pair["obs_a"] >= pair["obs_b"] else pair["id_b"]
            merge_id = pair["id_b"] if keep_id == pair["id_a"] else pair["id_a"]

            # Move sequences
            await self.pool.execute(
                "UPDATE sound_sequences SET from_unit_id = $1 WHERE from_unit_id = $2",
                keep_id, merge_id,
            )
            await self.pool.execute(
                "UPDATE sound_sequences SET to_unit_id = $1 WHERE to_unit_id = $2",
                keep_id, merge_id,
            )
            # Move co-occurrences from merged unit to keeper (unified table)
            # Transfer entries where merge_id is on either side
            modality = pair.get("modality", "voice")
            for side_col, id_col in [("unit_a_id", "modality_a"), ("unit_b_id", "modality_b")]:
                await self.pool.execute(
                    f"""
                    UPDATE sensory_cooccurrences SET {side_col} = $1
                    WHERE {side_col} = $2 AND {id_col} = $3
                      AND NOT EXISTS (
                          SELECT 1 FROM sensory_cooccurrences sc2
                          WHERE sc2.unit_a_id = CASE WHEN '{side_col}' = 'unit_a_id' THEN $1 ELSE sensory_cooccurrences.unit_a_id END
                            AND sc2.modality_a = sensory_cooccurrences.modality_a
                            AND sc2.unit_b_id = CASE WHEN '{side_col}' = 'unit_b_id' THEN $1 ELSE sensory_cooccurrences.unit_b_id END
                            AND sc2.modality_b = sensory_cooccurrences.modality_b
                      )
                    """,
                    keep_id, merge_id, modality,
                )
            # Delete remaining (duplicates that couldn't transfer)
            await self.pool.execute(
                """DELETE FROM sensory_cooccurrences
                   WHERE (unit_a_id = $1 AND modality_a = $2) OR (unit_b_id = $1 AND modality_b = $2)""",
                merge_id, modality,
            )
            # Delete duplicate sequences
            await self.pool.execute(
                """
                DELETE FROM sound_sequences a
                USING sound_sequences b
                WHERE a.id > b.id
                  AND a.from_unit_id = b.from_unit_id AND a.to_unit_id = b.to_unit_id
                  AND a.direction = b.direction
                """,
            )
            # Delete merged unit
            await self.pool.execute("DELETE FROM sound_units WHERE id = $1", merge_id)
            merged += 1
            log.debug("Merged sound unit #%d into #%d (sim=%.3f)",
                      merge_id, keep_id, pair["similarity"])

        stats["units_merged"] = merged

        # 2. Prune orphans (low observations, no connections)
        pruned = await self.pool.execute(
            """
            DELETE FROM sound_units
            WHERE total_observations < 3
              AND id NOT IN (SELECT from_unit_id FROM sound_sequences)
              AND id NOT IN (SELECT to_unit_id FROM sound_sequences)
              AND id NOT IN (SELECT unit_a_id FROM sensory_cooccurrences WHERE modality_a = 'voice' UNION SELECT unit_b_id FROM sensory_cooccurrences WHERE modality_b = 'voice')
              AND created_at < NOW() - INTERVAL '1 hour'
            """,
        )
        stats["units_pruned"] = int(pruned.split()[-1])

        # 3. Discover phrases
        phrases = await self.discover_phrases()
        stats["phrases_discovered"] = len(phrases)
        if phrases:
            for p in phrases[:3]:
                labels = " → ".join(lab or "?" for lab in p["labels"])
                log.info("Sound phrase: [%s] (count=%d, gap=%.0fms)",
                         labels, p["count"], p["avg_gap_ms"])

        # 4. Syllable chunking
        syllables = await self.discover_syllables()
        stats["syllables_discovered"] = len(syllables)

        # 5. Stats
        total_units = await self.pool.fetchval("SELECT COUNT(*) FROM sound_units")
        total_sequences = await self.pool.fetchval("SELECT COUNT(*) FROM sound_sequences")
        total_bindings = await self.pool.fetchval("SELECT COUNT(*) FROM sensory_cooccurrences")
        labelled = await self.pool.fetchval(
            "SELECT COUNT(*) FROM sound_units WHERE stt_label IS NOT NULL")
        stats["total_units"] = total_units
        stats["total_sequences"] = total_sequences
        stats["total_bindings"] = total_bindings
        stats["labelled_units"] = labelled

        log.info("Sound language consolidation: %d units (%d labelled), "
                 "%d sequences, %d visual bindings, %d merged, %d pruned",
                 total_units, labelled, total_sequences, total_bindings,
                 merged, stats["units_pruned"])

        return stats
