"""
Vocal tract: sound production via TABULA2 decoder + self-monitoring.

The system's "mouth" — takes a sound unit embedding (512-dim voice
centroid from TABULA2), decodes it to a waveform via TABULA2's voice
decoder, then re-encodes to verify it matches the target.

Motor programs are refined through self-play: generate → listen → compare
→ adjust. Like a baby learning to control their voice by babbling and
hearing the results.

Requires:
  - TABULA2Disentangler with decoder loaded (for voice reconstruction)
"""
import logging

import asyncpg
import numpy as np

from nmem_sym_sensor import config

log = logging.getLogger(__name__)


class VocalTract:
    """Sound production with self-monitoring feedback loop.

    Uses TABULA2's voice decoder for embedding → waveform reconstruction.
    The disentangler's encode path provides self-monitoring (re-encode
    the produced waveform and compare to the target centroid).

    Usage::

        tract = VocalTract(pool, tabula2=disentangler)
        await tract.init_db()

        # Produce a sound
        waveform, quality = await tract.produce(sound_unit_id=42)

        # Practice (self-play refinement)
        result = await tract.practice(sound_unit_id=42, iterations=50)

        # Speak a concept (string multiple sounds together)
        waveform, log = await tract.speak(visual_node_ids=[101, 102])
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        tabula2=None,
        sample_rate: int = 16000,
        voice_profile_path: str | None = None,
    ):
        self.pool = pool
        self.sample_rate = sample_rate
        self._tabula2 = tabula2
        self._voice_profile = None

        # Load voice profile for consistent speaker identity in production
        if voice_profile_path is None:
            import os
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "checkpoints",
                "voice_profile_dayyan_subfeatures.npz",
            )
            if os.path.exists(default_path):
                voice_profile_path = default_path

        if voice_profile_path:
            try:
                data = np.load(voice_profile_path)
                self._voice_profile = {
                    "pitch": data["pitch"],       # (96,)
                    "formant": data["formant"],   # (80,)
                }
                log.info("Voice profile loaded: %s", voice_profile_path)
            except Exception as e:
                log.warning("Voice profile load failed: %s", e)

    async def init_db(self):
        """Create motor_programs table and ensure sound_units has exemplar columns."""
        await self.pool.execute("""
            CREATE TABLE IF NOT EXISTS motor_programs (
                id              BIGSERIAL PRIMARY KEY,
                sound_unit_id   BIGINT NOT NULL REFERENCES sound_units(id),
                production_embedding vector(512),
                quality_score   FLOAT DEFAULT 0.0,
                practice_count  INTEGER DEFAULT 0,
                myelinated      BOOLEAN DEFAULT FALSE,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(sound_unit_id)
            )
        """)
        # Exemplar frame columns on sound_units (for speech production)
        for col, dtype in [
            ("exemplar_frames", "BYTEA"),
            ("exemplar_sub_frames", "BYTEA"),
            ("exemplar_n_frames", "INTEGER DEFAULT 0"),
            ("exemplar_energy", "FLOAT DEFAULT 0.0"),
        ]:
            await self.pool.execute(
                f"ALTER TABLE sound_units ADD COLUMN IF NOT EXISTS {col} {dtype}"
            )

    def _apply_voice_profile(self, frames: np.ndarray) -> np.ndarray:
        """Recentre pitch and formant features on voice profile.

        Preserves temporal dynamics (intonation, stress) but shifts the
        baseline to the target speaker. Like transposing a melody to a
        different key — same rhythm, different register.

        TABULA2 voice layout: [pitch:96 | harmonic:96 | formant:80 | vad:48 | spectral:192]
        """
        if self._voice_profile is None:
            return frames

        adapted = frames.copy()
        pitch_prof = self._voice_profile["pitch"]
        formant_prof = self._voice_profile["formant"]

        if frames.ndim == 1:
            # Single embedding — direct replacement blended
            adapted[0:96] = 0.5 * frames[0:96] + 0.5 * pitch_prof
            adapted[192:272] = 0.5 * frames[192:272] + 0.5 * formant_prof
        else:
            # Temporal sequence — recentre dynamics on profile
            pitch_mean = frames[:, 0:96].mean(axis=0)
            pitch_delta = frames[:, 0:96] - pitch_mean[np.newaxis, :]
            adapted[:, 0:96] = pitch_prof[np.newaxis, :] + pitch_delta

            formant_mean = frames[:, 192:272].mean(axis=0)
            formant_delta = frames[:, 192:272] - formant_mean[np.newaxis, :]
            adapted[:, 192:272] = formant_prof[np.newaxis, :] + formant_delta

        return adapted

    def decode(
        self,
        embedding: np.ndarray,
        n_frames: int = 50,
        sub_features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Decode a 512-dim voice centroid to a waveform via TABULA2.

        Falls back to expanding centroid to n identical frames.
        Returns float32 mono waveform at 16kHz, or None.
        """
        if self._tabula2 is None:
            return None

        embedding = self._apply_voice_profile(embedding)
        return self._tabula2.decode_voice(
            embedding, n_frames=n_frames, sub_features=sub_features,
        )

    def decode_sequence(
        self,
        voice_frames: np.ndarray,
        sub_feature_frames: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Decode a temporal frame sequence to a waveform via TABULA2.

        This is the proper path — real temporal variation instead of
        expanded centroids. Like replaying a memory of hearing the word.

        Args:
            voice_frames: (T, 512) temporal voice frame sequence.
            sub_feature_frames: {key: (T, D)} matching sub-feature sequences.
                Keys: 'pitch', 'harmonic', 'formant', 'vad', 'spectral'.

        Returns float32 mono waveform at 16kHz, or None.
        """
        if self._tabula2 is None:
            return None

        # Apply voice profile (recentre pitch/formant on target speaker)
        voice_frames = self._apply_voice_profile(voice_frames)

        # decode_voice handles (T, 512) input directly
        return self._tabula2.decode_voice(
            voice_frames, sub_features=sub_feature_frames,
        )

    def encode(self, waveform: np.ndarray) -> np.ndarray | None:
        """Re-encode a waveform to a 512-dim voice embedding (self-monitoring).

        Runs the waveform through TABULA2's encoder and returns the mean
        voice embedding. Used to compare produced output against the target.
        """
        if self._tabula2 is None:
            return None

        try:
            output = self._tabula2.process(waveform, sample_rate=self.sample_rate)
            # Mean-pool voiced frames, or all frames if no speech detected
            if output.vad_mask.any():
                embedding = output.voice_embeddings[output.vad_mask].mean(axis=0)
            else:
                embedding = output.voice_embeddings.mean(axis=0)
            # L2 normalise
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            log.debug("Self-monitoring encode failed: %s", e)
            return None

    async def produce(self, sound_unit_id: int) -> tuple[np.ndarray | None, float]:
        """Produce a waveform for a sound unit.

        Uses the motor program if available, otherwise falls back
        to the unit's centroid embedding.

        Returns (waveform, quality_score). Quality is the cosine
        similarity between re-encoded output and target centroid.
        """
        # Get target centroid + exemplar frames
        unit = await self.pool.fetchrow(
            """SELECT centroid::text as c, stt_label,
                      pitch_centroid::text as pitch,
                      harmonic_centroid::text as harmonic,
                      formant_centroid::text as formant,
                      vad_centroid::text as vad,
                      spectral_centroid::text as spectral,
                      exemplar_frames, exemplar_n_frames
               FROM sound_units WHERE id = $1""",
            sound_unit_id,
        )
        if not unit or not unit["c"]:
            return None, 0.0

        def _parse_vec(s):
            if not s:
                return None
            return np.array([float(x) for x in s.strip("[]").split(",")])

        target = _parse_vec(unit["c"])

        # Try exemplar frame sequence first (temporal, much better for decoder)
        if unit["exemplar_frames"] is not None:
            from nmem_sym_sensor.language import _decompress_frames
            frames = _decompress_frames(bytes(unit["exemplar_frames"]))
            voice_seq = frames.get("voice")  # (T, 512)
            sub_feat_seq = {
                k.replace("voice_", "").replace("_features", ""): v
                for k, v in frames.items() if k.startswith("voice_")
            }
            waveform = self.decode_sequence(voice_seq, sub_feat_seq or None)
        else:
            # Fallback: expand centroid (static, sounds like buzzing)
            sub_features = {}
            for key in ("pitch", "harmonic", "formant", "vad", "spectral"):
                vec = _parse_vec(unit[key])
                if vec is not None:
                    sub_features[key] = vec

            # Check for existing motor program
            program = await self.pool.fetchrow(
                "SELECT production_embedding::text as emb, quality_score FROM motor_programs WHERE sound_unit_id = $1",
                sound_unit_id,
            )
            production_emb = _parse_vec(program["emb"]) if program and program["emb"] else target
            waveform = self.decode(production_emb, sub_features=sub_features or None)

        if waveform is None:
            return None, 0.0

        # Self-monitor: re-encode produced waveform and compare to target
        re_encoded = self.encode(waveform)
        if re_encoded is not None:
            quality = float(np.dot(re_encoded, target))
        else:
            quality = 0.0

        label = unit["stt_label"] or f"unit#{sound_unit_id}"
        log.debug("Produced '%s': quality=%.3f, duration=%.3fs",
                  label, quality, len(waveform) / self.sample_rate)

        return waveform, quality

    async def practice(
        self,
        sound_unit_id: int,
        iterations: int = 50,
        noise_scale: float = 0.05,
    ) -> dict:
        """Self-play refinement loop (babbling).

        Starts from the unit centroid (or existing motor program),
        adds small perturbations, keeps improvements. Like a baby
        refining their pronunciation through trial and error.

        Returns dict with quality, iterations, improved flag.
        """
        # Get target centroid
        unit = await self.pool.fetchrow(
            "SELECT centroid::text as c FROM sound_units WHERE id = $1",
            sound_unit_id,
        )
        if not unit or not unit["c"]:
            return {"quality": 0.0, "iterations": 0, "improved": False}

        target = np.array([float(x) for x in unit["c"].strip("[]").split(",")])

        # Start from motor program or centroid
        program = await self.pool.fetchrow(
            "SELECT production_embedding::text as emb, quality_score FROM motor_programs WHERE sound_unit_id = $1",
            sound_unit_id,
        )

        if program and program["emb"]:
            best_emb = np.array([float(x) for x in program["emb"].strip("[]").split(",")])
            best_quality = float(program["quality_score"] or 0)
        else:
            best_emb = target.copy()
            best_quality = 0.0

        initial_quality = best_quality

        for _ in range(iterations):
            # Perturbation shrinks as quality improves (convergence)
            scale = noise_scale * (1.0 - best_quality * 0.8)
            noise = np.random.randn(len(best_emb)) * scale
            candidate = best_emb + noise
            # L2 normalise
            norm = np.linalg.norm(candidate)
            if norm > 1e-8:
                candidate = candidate / norm

            # Decode → waveform → re-encode → compare
            waveform = self.decode(candidate)
            if waveform is None:
                break

            re_encoded = self.encode(waveform)
            if re_encoded is None:
                break

            quality = float(np.dot(re_encoded, target))

            if quality > best_quality:
                best_emb = candidate
                best_quality = quality

        # Save motor program
        improved = best_quality > initial_quality
        myelinated = best_quality >= 0.8

        await self.pool.execute(
            """
            INSERT INTO motor_programs (sound_unit_id, production_embedding, quality_score, practice_count, myelinated)
            VALUES ($1, $2::vector, $3, $4, $5)
            ON CONFLICT (sound_unit_id)
            DO UPDATE SET
                production_embedding = CASE WHEN $3 > motor_programs.quality_score
                    THEN $2::vector ELSE motor_programs.production_embedding END,
                quality_score = GREATEST(motor_programs.quality_score, $3),
                practice_count = motor_programs.practice_count + $4,
                myelinated = motor_programs.myelinated OR $5,
                updated_at = NOW()
            """,
            sound_unit_id, str(best_emb.tolist()), round(best_quality, 6),
            iterations, myelinated,
        )

        if improved:
            log.info("Practice: unit #%d quality %.3f → %.3f (%s)",
                     sound_unit_id, initial_quality, best_quality,
                     "myelinated!" if myelinated else "improving")

        return {
            "quality": round(best_quality, 4),
            "initial_quality": round(initial_quality, 4),
            "iterations": iterations,
            "improved": improved,
            "myelinated": myelinated,
        }

    async def practice_syllable(
        self,
        syllable_id: int,
    ) -> dict:
        """Practice producing a syllable: decode constituent exemplar frames,
        re-encode, compare. LTP on success with downward reinforcement.

        Steps:
        1. Fetch syllable's unit_ids and concatenated exemplar frames
        2. Decode through TABULA2 → waveform
        3. Re-encode waveform → embedding
        4. Compare against mean of constituent unit centroids
        5. Quality > threshold: reinforce syllable + constituent transitions
        6. Quality < threshold: no active weakening (decay handles it)

        Returns dict with quality, reinforced, myelinated status.
        """
        import math

        from nmem_sym_sensor.cooccurrence import N_MIN_MYELINATE, TAU_0, TAU_MYELINATE
        from nmem_sym_sensor.language import _decompress_frames

        row = await self.pool.fetchrow(
            """SELECT id, unit_ids, exemplar_frames, observation_count,
                      self_play_confirmations, confidence, half_life
               FROM speech_syllables WHERE id = $1""",
            syllable_id,
        )
        if row is None:
            return {"error": "syllable not found"}

        unit_ids = list(row["unit_ids"])
        exemplar_data = row["exemplar_frames"]

        if exemplar_data is None:
            return {"error": "no exemplar frames", "syllable_id": syllable_id}

        # Decode exemplar frames through TABULA2
        frames_dict = _decompress_frames(exemplar_data)
        voice_frames = frames_dict.get("voice")
        if voice_frames is None or len(voice_frames) == 0:
            return {"error": "empty exemplar voice frames"}

        waveform = self.decode_sequence(voice_frames)
        if waveform is None or len(waveform) == 0:
            return {"error": "decode failed"}

        # Re-encode and compare
        re_encoded = self.encode(waveform)
        if re_encoded is None:
            return {"error": "re-encode failed"}

        # Target: mean of the exemplar frames themselves (ground truth for this syllable)
        target = np.mean(voice_frames, axis=0).astype(np.float32)
        norm = np.linalg.norm(target)
        if norm > 1e-8:
            target /= norm

        quality = float(np.dot(re_encoded, target))
        threshold = config.SYLLABLE_PRACTICE_QUALITY_THRESHOLD

        sp = row["self_play_confirmations"]
        obs = row["observation_count"]
        reinforced = False
        myelinated = row.get("myelinated", False)

        if quality >= threshold:
            # LTP: reinforce syllable
            sp += 1
            new_hl = TAU_0 * (1 + math.log(1 + obs)) * (0.5 + quality) * (1 + sp / 10.0)
            myelinated = new_hl > TAU_MYELINATE and obs > N_MIN_MYELINATE

            await self.pool.execute(
                """UPDATE speech_syllables SET
                    self_play_confirmations = $2,
                    confidence = LEAST(1.0, confidence + 0.01),
                    half_life = $3,
                    myelinated = $4,
                    updated_at = NOW()
                   WHERE id = $1""",
                syllable_id, sp, new_hl, myelinated,
            )

            # Downward reinforcement: bump constituent phoneme transitions
            for i in range(len(unit_ids) - 1):
                await self.pool.execute(
                    """UPDATE sound_sequences SET
                        count = count + 1,
                        confidence = LEAST(1.0, (count + 1)::float / 20.0),
                        updated_at = NOW()
                       WHERE from_unit_id = $1 AND to_unit_id = $2""",
                    unit_ids[i], unit_ids[i + 1],
                )
            reinforced = True

        return {
            "syllable_id": syllable_id,
            "unit_ids": unit_ids,
            "quality": round(quality, 4),
            "reinforced": reinforced,
            "self_play_confirmations": sp,
            "myelinated": myelinated,
        }

    async def speak(
        self,
        visual_node_ids: list[int] | None = None,
        sound_unit_ids: list[int] | None = None,
    ) -> tuple[np.ndarray | None, list[dict]]:
        """Speak about a visual concept or produce a sequence of sound units.

        1. If visual_node_ids given: look up bound sound units
        2. Get sound sequences for ordering
        3. Produce each sound in sequence
        4. Concatenate with appropriate gaps

        Returns (full_waveform, production_log).
        """
        # Resolve sound units to produce
        units_to_produce = []

        if sound_unit_ids:
            units_to_produce = sound_unit_ids
        elif visual_node_ids:
            # Find sound units bound to these visual nodes (unified table)
            from nmem_sym_sensor.cooccurrence import cooc_store
            all_sounds: dict[int, int] = {}
            for vid in visual_node_ids:
                coocs = await cooc_store.query(vid, "visual", self.pool,
                                               target_modality="voice", min_count=3, limit=5)
                for c in coocs:
                    all_sounds[c["unit_id"]] = all_sounds.get(c["unit_id"], 0) + c["count"]
            sound_ids = sorted(all_sounds, key=all_sounds.get, reverse=True)[:5]
            rows = await self.pool.fetch(
                "SELECT id, stt_label FROM sound_units WHERE id = ANY($1)", sound_ids,
            )
            rows = [dict(r) | {"binding": all_sounds.get(r["id"], 0)} for r in rows]
            units_to_produce = [r["id"] for r in rows]

        if not units_to_produce:
            return None, []

        # Pre-fetch learned gap timings between consecutive units
        gap_cache = {}
        for i in range(len(units_to_produce) - 1):
            a, b = units_to_produce[i], units_to_produce[i + 1]
            row = await self.pool.fetchrow(
                "SELECT gap_ms_avg FROM sound_sequences "
                "WHERE from_unit_id = $1 AND to_unit_id = $2 AND count >= 2",
                a, b,
            )
            if row and row["gap_ms_avg"]:
                gap_cache[(a, b)] = max(50, min(row["gap_ms_avg"], 1000))

        # Produce each sound with learned inter-unit timing
        waveforms = []
        production_log = []
        default_gap_ms = 150

        for i, uid in enumerate(units_to_produce):
            waveform, quality = await self.produce(uid)
            if waveform is not None:
                waveforms.append(waveform)

                # Use learned gap or default
                if i < len(units_to_produce) - 1:
                    next_uid = units_to_produce[i + 1]
                    gap_ms = gap_cache.get((uid, next_uid), default_gap_ms)
                    gap_samples = int(gap_ms / 1000.0 * self.sample_rate)
                    waveforms.append(np.zeros(gap_samples, dtype=np.float32))

                unit = await self.pool.fetchrow(
                    "SELECT stt_label FROM sound_units WHERE id = $1", uid)
                production_log.append({
                    "sound_unit_id": uid,
                    "label": unit["stt_label"] if unit else None,
                    "quality": quality,
                    "duration_ms": len(waveform) / self.sample_rate * 1000,
                    "gap_ms": gap_cache.get(
                        (uid, units_to_produce[i + 1]) if i < len(units_to_produce) - 1 else None,
                        default_gap_ms,
                    ),
                })

        if not waveforms:
            return None, production_log

        full_waveform = np.concatenate(waveforms)
        return full_waveform, production_log
