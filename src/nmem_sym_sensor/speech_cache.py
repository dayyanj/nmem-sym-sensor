"""
In-memory cache of myelinated speech syllables for fast recognition.

Analogous to ClusterCache for visual clusters (familiarity.py).
Loaded from DB during connect() and refreshed after each consolidation cycle.

When a phoneme is heard, check if it starts a cached (myelinated) syllable.
If so, predict the rest without per-phoneme DB queries.

Recognition is embedding-based, not STT. The system recognises "ball" because
the sound embedding matches a myelinated word chunk, not because Whisper
transcribed it. STT labels are diagnostic annotations only.
"""
import logging

import asyncpg

log = logging.getLogger(__name__)


class SpeechCache:
    """In-memory cache of myelinated syllables for fast recognition.

    Structure:
        _by_first_unit: dict[int, list[dict]]
        Maps first phoneme unit_id -> list of syllables starting with that unit.

    Usage::

        cache = SpeechCache()
        await cache.refresh(pool)
        matches = cache.match_start(current_unit_id)
    """

    def __init__(self):
        self._by_first_unit: dict[int, list[dict]] = {}
        self._syllable_count: int = 0

    async def refresh(self, pool: asyncpg.Pool):
        """Reload myelinated syllables from DB.

        Only loads syllables where myelinated=TRUE.
        Parses unit_ids array for fast first-unit lookup.
        """
        rows = await pool.fetch("""
            SELECT id, unit_ids, stt_label, confidence, observation_count,
                   total_duration_ms
            FROM speech_syllables
            WHERE myelinated = TRUE
        """)
        self._by_first_unit.clear()
        for r in rows:
            unit_ids = list(r["unit_ids"])
            if not unit_ids:
                continue
            first = unit_ids[0]
            entry = {
                "syllable_id": r["id"],
                "unit_ids": unit_ids,
                "stt_label": r["stt_label"],
                "confidence": float(r["confidence"]),
                "observation_count": r["observation_count"],
                "duration_ms": r["total_duration_ms"],
            }
            self._by_first_unit.setdefault(first, []).append(entry)
        self._syllable_count = sum(len(v) for v in self._by_first_unit.values())
        if self._syllable_count > 0:
            log.info("SpeechCache refreshed: %d myelinated syllables across %d first-units",
                     self._syllable_count, len(self._by_first_unit))

    def match_start(self, unit_id: int) -> list[dict]:
        """Find myelinated syllables that start with this phoneme.

        Returns list of syllable dicts sorted by confidence descending.
        """
        matches = self._by_first_unit.get(unit_id, [])
        return sorted(matches, key=lambda x: x["confidence"], reverse=True)

    def match_sequence(self, unit_id_sequence: list[int]) -> dict | None:
        """Check if a sequence of recently heard phonemes matches a cached syllable.

        Compares the sequence against all syllables starting with sequence[0].
        Returns the matching syllable dict, or None.
        """
        if not unit_id_sequence:
            return None
        candidates = self._by_first_unit.get(unit_id_sequence[0], [])
        for c in candidates:
            if c["unit_ids"][:len(unit_id_sequence)] == unit_id_sequence:
                return c
        return None

    @property
    def size(self) -> int:
        return self._syllable_count
