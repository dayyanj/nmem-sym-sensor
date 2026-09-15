"""
Auditory dream inspector: hear what the system has learned.

Uses TABULA2's decoder to reconstruct audio from stored sound unit
centroids. The quality is lossy — an auditory "impression", not a
recording. This is the acoustic dimension of the mind's eye.

Usage:
    inspector = AuditoryDreamInspector(pool, tabula2)
    waveform = await inspector.hear(sound_unit_id=42)
    waveform = await inspector.hear_label("triangle")
    report = await inspector.dream_report()
"""
import logging
import wave

import asyncpg
import numpy as np

log = logging.getLogger(__name__)


class AuditoryDreamInspector:
    """Reconstruct what the system 'hears' for learned sound patterns."""

    def __init__(self, pool: asyncpg.Pool, tabula2):
        self.pool = pool
        self.tabula2 = tabula2

    async def hear(self, sound_unit_id: int, n_frames: int = 50) -> np.ndarray | None:
        """Reconstruct the waveform for a sound unit.

        Fetches all stored centroid embeddings (main + sub-features),
        builds the full feature dict, and runs through TABULA2 decoder.
        Returns float32 mono waveform at 16kHz, or None.
        """
        row = await self.pool.fetchrow(
            """SELECT centroid::text as c,
                      pitch_centroid::text as pitch,
                      harmonic_centroid::text as harmonic,
                      formant_centroid::text as formant,
                      vad_centroid::text as vad,
                      spectral_centroid::text as spectral
               FROM sound_units WHERE id = $1""",
            sound_unit_id,
        )
        if not row or not row["c"]:
            return None

        import torch

        def _parse(s, dim):
            if not s:
                return None
            arr = np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)
            return torch.tensor(arr).unsqueeze(0).expand(n_frames, -1).unsqueeze(0)  # (1, T, D)

        # Build full feature dict for decoder
        voice_features = {}
        voice_features["voice_embeddings"] = _parse(row["c"], 512)
        if row["pitch"]:
            voice_features["voice_pitch_features"] = _parse(row["pitch"], 96)
        if row["harmonic"]:
            voice_features["voice_harmonic_features"] = _parse(row["harmonic"], 96)
        if row["formant"]:
            voice_features["voice_formant_features"] = _parse(row["formant"], 80)
        if row["vad"]:
            voice_features["voice_vad_features"] = _parse(row["vad"], 48)
        if row["spectral"]:
            voice_features["voice_spectral_features"] = _parse(row["spectral"], 192)

        if voice_features["voice_embeddings"] is None:
            return None

        # Load decoder if needed
        if self.tabula2._decoder is None:
            self.tabula2._load_decoder()
        if self.tabula2._decoder is None:
            return None

        try:
            with torch.no_grad():
                outputs = self.tabula2._decoder({"voice": voice_features})
            waveform = outputs.get("voice_waveform")
            if waveform is not None:
                return waveform[0].numpy()
        except Exception as e:
            log.debug("Dream decode failed for unit %d: %s", sound_unit_id, e)

        return None

    async def hear_label(self, label: str) -> np.ndarray | None:
        """Reconstruct what the system hears for a word (by STT label).

        Finds the most-observed sound unit with that label.
        """
        unit = await self.pool.fetchrow(
            "SELECT id FROM sound_units WHERE stt_label = $1 "
            "ORDER BY total_observations DESC LIMIT 1",
            label.strip().lower(),
        )
        if unit:
            return await self.hear(unit["id"])
        return None

    async def dream_report(self, limit: int = 10) -> list[dict]:
        """Generate a report of top sound units with reconstructed audio.

        Returns list of dicts with unit info + waveform for each.
        """
        units = await self.pool.fetch(
            """
            SELECT su.id, su.stt_label, su.total_observations, su.member_count,
                   su.modality, su.speaker_variance,
                   (SELECT COUNT(*) FROM sensory_cooccurrences sc
                    WHERE ((sc.unit_a_id = su.id AND sc.modality_a = su.modality AND sc.modality_b = 'visual')
                        OR (sc.unit_b_id = su.id AND sc.modality_b = su.modality AND sc.modality_a = 'visual'))
                      AND sc.count > 0) as visual_bindings
            FROM sound_units su
            ORDER BY su.total_observations DESC
            LIMIT $1
            """,
            limit,
        )

        report = []
        for u in units:
            waveform = await self.hear(u["id"])
            report.append({
                "id": u["id"],
                "label": u["stt_label"] or "(unlabelled)",
                "observations": u["total_observations"],
                "members": u["member_count"],
                "modality": u["modality"],
                "visual_bindings": u["visual_bindings"],
                "has_audio": waveform is not None,
                "waveform": waveform,
            })

        return report


def save_wav(waveform: np.ndarray, path: str, sample_rate: int = 16000):
    """Save a float32 waveform to a WAV file."""
    # Clip and convert to int16
    clipped = np.clip(waveform, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
