"""Familiarity-gated processing — spend less compute on familiar content.

Like the brain: familiar surroundings run on a cached model with minimal
verification. Novel elements get full processing. Processing budget flows
from the familiar toward the novel.

Two signals combined:
1. Coarse visual similarity (32x32 grayscale → 1024-dim, trivially cheap)
2. EmbeddingPredictor.running_error (prediction accuracy — already computed)
"""
import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

COARSE_SIZE = 8  # 8x8 color → 192-dim (optimal: same separation as 32x32 at 1/16 cost)


class ClusterCache:
    """In-memory cache of grounded cluster centroids for fast object matching.

    Loaded at connect() time and refreshed after each consolidation cycle.
    Avoids per-frame DB queries — typically < 100 entries, trivial memory.

    Usage: check if a visual embedding matches a known/grounded cluster.
    If it does, the object is "familiar" and gets reduced foveal effort.
    """

    def __init__(self):
        self._centroids: dict[int, tuple[np.ndarray, str, float]] = {}
        # {cluster_id: (centroid_embedding, label, confidence)}

    async def refresh(self, pool):
        """Reload grounded cluster centroids from DB."""
        rows = await pool.fetch("""
            SELECT id, visual_centroid::text as vc, grounded_label,
                   grounding_confidence
            FROM sensory_clusters
            WHERE grounded_label IS NOT NULL
              AND grounding_confidence > 0.5
              AND visual_centroid IS NOT NULL
        """)
        self._centroids.clear()
        for r in rows:
            try:
                vec = np.array(
                    [float(x) for x in r["vc"].strip("[]").split(",")],
                    dtype=np.float32,
                )
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                self._centroids[r["id"]] = (
                    vec, r["grounded_label"],
                    float(r["grounding_confidence"]),
                )
            except (ValueError, AttributeError):
                continue
        log.info("ClusterCache refreshed: %d grounded clusters", len(self._centroids))

    def match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.80,
    ) -> tuple[int | None, str | None, float]:
        """Find best matching grounded cluster for an embedding.

        Returns (cluster_id, label, similarity) or (None, None, 0.0).
        """
        if not self._centroids:
            return None, None, 0.0

        best_id, best_label, best_sim = None, None, 0.0
        emb = embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        for cid, (centroid, label, conf) in self._centroids.items():
            sim = float(np.dot(emb, centroid))
            if sim > best_sim and sim >= threshold:
                best_id, best_label, best_sim = cid, label, sim

        return best_id, best_label, best_sim

    @property
    def size(self) -> int:
        return len(self._centroids)


@dataclass
class AttentionBudget:
    """Compute allocation for this frame based on familiarity."""
    max_saccades: int = 8
    edge_follow_steps: int = 6
    run_intelligence: bool = True
    frame_familiarity: float = 0.0   # 0=novel, 1=identical
    scene_stability: float = 0.0    # rolling frame-to-frame similarity
    cluster_cache: "ClusterCache | None" = None  # for object-level matching


class SceneMemory:
    """Rolling buffer of coarse frame signatures for familiarity detection.

    Maintains a short history of downscaled frame embeddings. Comparing
    a new frame against this buffer tells us if we've seen this scene
    before — much cheaper than full analysis.
    """

    def __init__(
        self,
        buffer_size: int = 30,
        skip_threshold: float = 0.95,
        reduce_threshold: float = 0.80,
        min_saccades: int = 4,
        reduced_saccades: int = 6,
        full_saccades: int = 8,
        min_edge_steps: int = 2,
        reduced_edge_steps: int = 4,
        full_edge_steps: int = 6,
    ):
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._stability_window: deque[float] = deque(maxlen=10)

        self.skip_threshold = skip_threshold
        self.reduce_threshold = reduce_threshold
        self.min_saccades = min_saccades
        self.reduced_saccades = reduced_saccades
        self.full_saccades = full_saccades
        self.min_edge_steps = min_edge_steps
        self.reduced_edge_steps = reduced_edge_steps
        self.full_edge_steps = full_edge_steps

        self._last_coarse: np.ndarray | None = None  # last computed coarse embedding

        # Stats
        self.frames_seen = 0
        self.frames_skipped = 0
        self.frames_reduced = 0
        self.frames_full = 0

    @staticmethod
    def _coarse_embedding(image: np.ndarray) -> np.ndarray:
        """Downscale to 16x16 and flatten. Uses color if available.

        Grayscale loses too much — a red scene and blue scene look
        identical at 32x32 gray. Color preserves the main differentiator.
        """
        small = cv2.resize(image, (COARSE_SIZE, COARSE_SIZE)).astype(np.float32)
        flat = small.flatten()
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat /= norm
        return flat

    def compute_familiarity(
        self,
        gray: np.ndarray,
        emb_running_error: float = 0.5,
    ) -> AttentionBudget:
        """Compute familiarity score and derive attention budget.

        Combines two signals:
        1. Coarse frame similarity to recent buffer (cheap visual match)
        2. EmbeddingPredictor.running_error (how well the system
           predicted this frame — already computed for free)
        """
        self.frames_seen += 1
        coarse = self._coarse_embedding(gray)

        # Compare against buffer
        if self._buffer:
            sims = [float(np.dot(coarse, prev)) for prev in self._buffer]
            max_sim = max(sims)
        else:
            max_sim = 0.0

        # Track frame-to-frame stability
        if self._buffer:
            frame_sim = float(np.dot(coarse, self._buffer[-1]))
            self._stability_window.append(frame_sim)

        stability = self.scene_stability

        # Add to buffer
        self._buffer.append(coarse)
        self._last_coarse = coarse  # expose for chronoception

        # Combine signals:
        # - coarse visual similarity (have I seen this frame?)
        # - prediction accuracy (did I expect this frame?)
        # Weight prediction accuracy higher — it's a richer signal
        prediction_familiarity = max(0.0, 1.0 - emb_running_error)
        frame_familiarity = 0.4 * max_sim + 0.6 * prediction_familiarity

        # Derive budget
        budget = AttentionBudget(
            frame_familiarity=frame_familiarity,
            scene_stability=stability,
        )

        if frame_familiarity > self.skip_threshold:
            budget.max_saccades = self.min_saccades
            budget.edge_follow_steps = self.min_edge_steps
            budget.run_intelligence = False  # monitor phase — scan only
            self.frames_skipped += 1
        elif frame_familiarity > self.reduce_threshold:
            budget.max_saccades = self.reduced_saccades
            budget.edge_follow_steps = self.reduced_edge_steps
            budget.run_intelligence = False  # reduced — save compute
            self.frames_reduced += 1
        else:
            budget.max_saccades = self.full_saccades
            budget.edge_follow_steps = self.full_edge_steps
            budget.run_intelligence = True  # novel/unfamiliar — full processing
            self.frames_full += 1

        return budget

    @property
    def scene_stability(self) -> float:
        if not self._stability_window:
            return 0.0
        return sum(self._stability_window) / len(self._stability_window)

    def reset(self):
        """Reset on scene change."""
        self._buffer.clear()
        self._stability_window.clear()

    def stats(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "frames_skipped": self.frames_skipped,
            "frames_reduced": self.frames_reduced,
            "frames_full": self.frames_full,
            "skip_pct": round(self.frames_skipped / max(1, self.frames_seen) * 100, 1),
            "reduce_pct": round(self.frames_reduced / max(1, self.frames_seen) * 100, 1),
        }
