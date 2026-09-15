"""
Speech prediction engine: predict upcoming phonemes and syllables
from learned transition probabilities.

Code-driven (DB queries on sound_sequences and speech_syllables),
NOT a trained model. Language-agnostic — works for any language or
even non-language sounds (music patterns, animal call sequences).

During observation: when a phoneme is heard, predict what comes next.
Compare prediction to what actually arrives. Surprise signal on
mismatch drives attention and memory formation.

Mirrors the visual prediction system (sensory_prediction.py) but
operates on temporal speech sequences rather than spatial frames.
"""
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from nmem_sym_sensor.speech_cache import SpeechCache

log = logging.getLogger(__name__)


@dataclass
class SpeechPrediction:
    """A prediction about the next speech unit."""
    level: str                    # "phoneme" | "syllable"
    predicted_unit_id: int        # sound_unit ID or speech_syllable ID
    confidence: float             # transition confidence from DB
    source_unit_id: int           # the current unit that triggered this prediction
    stt_label: str | None = None  # diagnostic label


@dataclass
class SpeechPredictionResult:
    """Outcome of a speech prediction."""
    prediction: SpeechPrediction
    status: str                   # "confirmed" | "refuted" | "surprising"
    observed_unit_id: int | None = None
    surprise: float = 0.0         # 0 = fully expected, 1 = completely unexpected


class SpeechPredictor:
    """Predict upcoming speech at phoneme and syllable levels.

    Two prediction sources:
    1. Phoneme-level: sound_sequences transition probabilities
    2. Syllable-level: myelinated syllable patterns from SpeechCache

    The predict→evaluate cycle mirrors how humans process speech:
    hearing "ba" primes "na" if "banana" is a known pattern.
    Getting it right costs nothing (just confirmation). Getting it
    wrong generates surprise that strengthens the actual pattern.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        speech_cache: "SpeechCache | None" = None,
    ):
        self.pool = pool
        self._cache = speech_cache
        self._pending: list[SpeechPrediction] = []
        # Stats
        self.total_predictions = 0
        self.total_confirmed = 0
        self.total_refuted = 0

    async def predict_next(
        self,
        current_unit_id: int,
        top_k: int = 3,
    ) -> list[SpeechPrediction]:
        """Generate predictions for what comes next after current_unit_id.

        Queries sound_sequences for top-k transitions by confidence,
        then checks if current_unit_id starts any known syllable.

        Predictions are stored internally as pending for later evaluation.
        Returns the predictions generated.
        """
        predictions = []

        # 1. Phoneme-level: top-k transitions from sound_sequences
        rows = await self.pool.fetch(
            """
            SELECT s.to_unit_id, s.confidence, u.stt_label
            FROM sound_sequences s
            LEFT JOIN sound_units u ON u.id = s.to_unit_id
            WHERE s.from_unit_id = $1 AND s.confidence > 0.1
            ORDER BY s.confidence DESC
            LIMIT $2
            """,
            current_unit_id, top_k,
        )
        for r in rows:
            predictions.append(SpeechPrediction(
                level="phoneme",
                predicted_unit_id=r["to_unit_id"],
                confidence=float(r["confidence"]),
                source_unit_id=current_unit_id,
                stt_label=r["stt_label"],
            ))

        # 2. Syllable-level: check if current_unit starts a known syllable
        if self._cache is not None:
            syllable_matches = self._cache.match_start(current_unit_id)
            for s in syllable_matches[:top_k]:
                # Predict the second phoneme of the syllable
                if len(s["unit_ids"]) >= 2:
                    predictions.append(SpeechPrediction(
                        level="syllable",
                        predicted_unit_id=s["unit_ids"][1],
                        confidence=s["confidence"],
                        source_unit_id=current_unit_id,
                        stt_label=s["stt_label"],
                    ))

        self._pending = predictions
        self.total_predictions += len(predictions)
        return predictions

    async def evaluate(
        self,
        observed_unit_id: int,
    ) -> list[SpeechPredictionResult]:
        """Compare pending predictions against what was actually heard.

        For each pending prediction:
        - predicted_unit_id == observed: confirmed (surprise=0)
        - predicted_unit_id != observed: refuted (surprise scales with confidence)

        On confirmation: bump the transition count (LTP).
        On refutation: no active LTD (Ebbinghaus decay handles weakening).

        Clears pending predictions after evaluation.
        Returns results with surprise scores.
        """
        if not self._pending:
            return []

        results = []

        for pred in self._pending:
            if pred.predicted_unit_id == observed_unit_id:
                result = SpeechPredictionResult(
                    prediction=pred,
                    status="confirmed",
                    observed_unit_id=observed_unit_id,
                    surprise=0.0,
                )
                self.total_confirmed += 1

                # LTP: reinforce the transition that was correctly predicted
                await self.pool.execute(
                    """
                    UPDATE sound_sequences
                    SET count = count + 1,
                        confidence = LEAST(1.0, (count + 1)::float / 20.0),
                        updated_at = NOW()
                    WHERE from_unit_id = $1 AND to_unit_id = $2
                    """,
                    pred.source_unit_id, pred.predicted_unit_id,
                )
            else:
                # Surprise proportional to how confident the wrong prediction was
                result = SpeechPredictionResult(
                    prediction=pred,
                    status="refuted",
                    observed_unit_id=observed_unit_id,
                    surprise=pred.confidence,
                )
                self.total_refuted += 1

            results.append(result)

        self._pending = []
        return results

    @property
    def accuracy(self) -> float:
        """Running prediction accuracy."""
        total = self.total_confirmed + self.total_refuted
        if total == 0:
            return 0.0
        return self.total_confirmed / total
