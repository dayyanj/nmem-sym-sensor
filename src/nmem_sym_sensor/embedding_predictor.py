"""
Embedding-space predictor — JEPA-inspired online learning.

Instead of predicting "which node appears next" (discrete), this predicts
the actual embedding vector of the next frame (continuous). Trained online
as frames arrive — no separate training phase needed.

Architecture: EMA baseline + learned residual.
- The EMA tracks what the scene "looks like" (strong baseline)
- The MLP learns to predict the residual (delta) between consecutive frames
- Prediction = EMA + learned_residual
- Surprise = cosine distance between predicted and actual

This captures temporal patterns the discrete system can't:
- Smooth transitions (camera pan → gradual embedding drift)
- Scene-level patterns (flashcard: color swatch always followed by shape)
- Anomaly detection (sudden embedding jump = scene change)

No GPU needed. Pure numpy.
"""
import logging
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


class EmbeddingPredictor:
    """Online embedding-space predictor with EMA baseline + learned residual.

    The prediction is: next_emb ≈ ema + W @ recent_deltas

    The EMA adapts in 5-10 frames (no learning needed).
    The residual MLP learns temporal dynamics (transitions, patterns).
    Surprise is the cosine distance between predicted and actual.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        context_len: int = 3,
        hidden_dim: int = 128,
        lr: float = 0.003,
        ema_alpha: float = 0.3,
    ):
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.ema_alpha = ema_alpha

        # EMA baseline — tracks the "current scene" embedding
        self._ema: np.ndarray | None = None

        # Recent embeddings and deltas for the residual predictor
        self._recent: deque[np.ndarray] = deque(maxlen=context_len + 1)

        # Residual MLP: recent deltas → predicted delta
        # Input: context_len deltas concatenated (context_len × embed_dim)
        input_dim = embed_dim * context_len
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, embed_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(embed_dim, dtype=np.float32)

        # Stats
        self.total_predictions = 0
        self.total_updates = 0
        self.running_error = 0.0  # EMA of prediction error
        self._error_ema_alpha = 0.05

    def has_context(self) -> bool:
        """Whether we have enough history to make a prediction."""
        return self._ema is not None and len(self._recent) > self.context_len

    def predict(self) -> np.ndarray | None:
        """Predict the next embedding."""
        if not self.has_context():
            return None

        base = self._ema.copy()
        deltas = self._get_deltas()
        if deltas is not None:
            residual = self._forward(deltas)
            base = base + residual

        # L2 normalize
        norm = np.linalg.norm(base)
        if norm > 0:
            base = base / norm
        return base

    def observe(self, embedding: np.ndarray) -> dict:
        """Observe a new embedding. Learn from prediction error.

        1. If we have context, predict next embedding
        2. Compare prediction to actual (cosine distance = surprise)
        3. Backprop to update residual predictor
        4. Update EMA and context

        Returns dict with surprise metrics.
        """
        embedding = embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        result = {
            "predicted": False,
            "error": 0.0,
            "cosine_surprise": 0.0,
            "ema_surprise": 0.0,
            "running_error": self.running_error,
        }

        if self.has_context():
            # Predict
            predicted = self.predict()

            # Cosine surprise
            cosine_sim = self._cosine_sim(predicted, embedding)
            cosine_surprise = max(0.0, 1.0 - cosine_sim)

            # Also compute EMA-only surprise (baseline without residual)
            ema_sim = self._cosine_sim(self._ema, embedding)
            ema_surprise = max(0.0, 1.0 - ema_sim)

            # Learn: train residual predictor
            deltas = self._get_deltas()
            if deltas is not None:
                # Target residual = actual - ema (what the MLP should have predicted)
                target_residual = embedding - self._ema
                predicted_residual = self._forward(deltas)
                self._backward(deltas, predicted_residual, target_residual)

            self.total_updates += 1
            self.total_predictions += 1

            # Update running error
            self.running_error = (
                self._error_ema_alpha * cosine_surprise
                + (1 - self._error_ema_alpha) * self.running_error
            )

            result.update({
                "predicted": True,
                "error": round(float(np.mean((predicted - embedding) ** 2)), 6),
                "cosine_surprise": round(cosine_surprise, 4),
                "ema_surprise": round(ema_surprise, 4),
                "running_error": round(self.running_error, 4),
            })

        # Update EMA
        if self._ema is None:
            self._ema = embedding.copy()
        else:
            self._ema = self.ema_alpha * embedding + (1 - self.ema_alpha) * self._ema
            # Re-normalize EMA
            ema_norm = np.linalg.norm(self._ema)
            if ema_norm > 0:
                self._ema = self._ema / ema_norm

        # Add to context
        self._recent.append(embedding.copy())
        return result

    def reset_context(self):
        """Reset context on scene change."""
        self._ema = None
        self._recent.clear()

    def get_stats(self) -> dict:
        return {
            "predictions": self.total_predictions,
            "updates": self.total_updates,
            "running_error": round(self.running_error, 4),
            "context_len": len(self._recent),
            "params": self.W1.size + self.b1.size + self.W2.size + self.b2.size,
        }

    # ── Internal ─────────────────────────────────────────

    def _get_deltas(self) -> np.ndarray | None:
        """Get recent frame-to-frame deltas as input for the residual MLP."""
        recent = list(self._recent)
        if len(recent) < self.context_len + 1:
            return None
        deltas = []
        for i in range(1, self.context_len + 1):
            deltas.append(recent[-i] - recent[-(i + 1)])
        return np.concatenate(deltas)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: deltas → hidden (ReLU) → residual."""
        self._h_pre = x @ self.W1 + self.b1
        self._h = np.maximum(0, self._h_pre)
        return self._h @ self.W2 + self.b2

    def _backward(self, x: np.ndarray, predicted: np.ndarray, target: np.ndarray):
        """Backprop MSE loss on the residual prediction."""
        d_out = 2.0 * (predicted - target) / self.embed_dim

        d_W2 = np.outer(self._h, d_out)
        d_b2 = d_out

        d_h = d_out @ self.W2.T
        d_h[self._h_pre <= 0] = 0

        d_W1 = np.outer(x, d_h)
        d_b1 = d_h

        # Gradient clipping
        for grad in [d_W1, d_b1, d_W2, d_b2]:
            norm = np.linalg.norm(grad)
            if norm > 1.0:
                grad *= 1.0 / norm

        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── Persistence ──────────────────────────────────────

    def save(self, path: str):
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            running_error=self.running_error,
            total_predictions=self.total_predictions,
            total_updates=self.total_updates,
        )
        log.debug("Saved embedding predictor to %s", path)

    def load(self, path: str):
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.running_error = float(data["running_error"])
        self.total_predictions = int(data["total_predictions"])
        self.total_updates = int(data["total_updates"])
        log.debug("Loaded embedding predictor from %s (%d prior predictions)",
                  path, self.total_predictions)
