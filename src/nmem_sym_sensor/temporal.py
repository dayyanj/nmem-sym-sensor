"""
Temporal continuity: object persistence across frames.

The cup in frame 10 is the same cup in frame 11 — not a new observation.
Objects are tracked by embedding similarity + spatial proximity. Matched
objects update in place rather than creating new iconic buffer entries.

Temporal predictions: an object at (200,300) moving right at 5px/frame
should be at (205,300) next frame. If found there → no surprise. If not
found or found somewhere unexpected → surprise signal.

This is object permanence — the understanding that objects continue to
exist when they move, and that their motion has continuity.
"""
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    id: int                              # unique tracking ID
    embedding: np.ndarray                # MoE embedding (512-dim)
    label: str                           # current best label
    position: tuple[float, float]        # center (x, y) in frame coords
    velocity: tuple[float, float] = (0.0, 0.0)  # pixels per frame
    size: tuple[float, float] = (0.0, 0.0)      # (width, height)
    age_frames: int = 0                  # how many frames since first seen
    last_seen_frame: int = 0             # frame number when last matched
    node_id: int | None = None           # sensory graph node ID
    confidence: float = 1.0              # tracking confidence (decays when missing)
    features: dict = field(default_factory=dict)  # latest part features


@dataclass
class TrackingResult:
    """Result of matching observations against tracked objects."""
    matched: list[tuple[TrackedObject, dict]]   # (tracked, observation) pairs
    new_objects: list[dict]                      # observations with no match (novel)
    disappeared: list[TrackedObject]             # tracked objects not seen this frame
    predictions: list[dict]                      # position predictions for next frame


class ObjectTracker:
    """Tracks objects across frames using embedding similarity.

    The tracker maintains a set of active objects. Each frame, new
    observations are matched against active objects. Matched observations
    update the object's state. Unmatched observations become new objects.
    Objects not seen for several frames are marked as disappeared.

    This provides:
    1. Object identity across frames (same cup, not new observation)
    2. Motion estimation (velocity from position changes)
    3. Temporal predictions (where should this object be next frame?)
    4. Disappearance detection (object left view or was occluded)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        spatial_weight: float = 0.3,
        max_missing_frames: int = 10,
        max_tracked: int = 50,
    ):
        self.similarity_threshold = similarity_threshold
        self.spatial_weight = spatial_weight
        self.max_missing_frames = max_missing_frames
        self.max_tracked = max_tracked

        self.tracked: dict[int, TrackedObject] = {}
        self._next_id = 0
        self._frame_number = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def update(
        self,
        observations: list[dict],
        frame_number: int | None = None,
    ) -> TrackingResult:
        """Match new observations against tracked objects.

        Args:
            observations: List of dicts with keys:
                - embedding: np.ndarray (512-dim)
                - label: str
                - position: (x, y) center
                - size: (w, h)
                - features: dict
                - node_id: int | None
            frame_number: Current frame number (auto-increments if None).

        Returns:
            TrackingResult with matched, new, disappeared, and predictions.
        """
        self._frame_number = frame_number if frame_number is not None else self._frame_number + 1

        if not observations and not self.tracked:
            return TrackingResult([], [], [], [])

        # Build cost matrix: tracked × observations
        tracked_list = list(self.tracked.values())
        matched_pairs = []
        used_obs = set()
        used_tracked = set()

        if tracked_list and observations:
            scores = np.zeros((len(tracked_list), len(observations)))

            for i, obj in enumerate(tracked_list):
                for j, obs in enumerate(observations):
                    # Embedding similarity (primary)
                    obs_emb = np.array(obs["embedding"])
                    emb_sim = float(np.dot(obj.embedding, obs_emb))

                    # Spatial proximity (secondary)
                    # Predict where the object should be based on velocity
                    pred_x = obj.position[0] + obj.velocity[0]
                    pred_y = obj.position[1] + obj.velocity[1]
                    obs_x, obs_y = obs["position"]

                    # Normalise spatial distance by frame diagonal
                    spatial_dist = np.sqrt((pred_x - obs_x)**2 + (pred_y - obs_y)**2)
                    max_dist = np.sqrt(obs.get("frame_w", 640)**2 + obs.get("frame_h", 480)**2)
                    spatial_sim = max(0, 1.0 - spatial_dist / (max_dist * 0.3))

                    # Combined score
                    scores[i, j] = (1 - self.spatial_weight) * emb_sim + self.spatial_weight * spatial_sim

            # Greedy matching: best score first
            while True:
                if scores.size == 0:
                    break
                best_idx = np.unravel_index(np.argmax(scores), scores.shape)
                best_score = scores[best_idx]

                if best_score < self.similarity_threshold:
                    break

                i, j = best_idx
                if i in used_tracked or j in used_obs:
                    scores[i, j] = -1
                    continue

                matched_pairs.append((tracked_list[i], observations[j]))
                used_tracked.add(i)
                used_obs.add(j)
                scores[i, :] = -1
                scores[:, j] = -1

        # Update matched objects
        matched_results = []
        for obj, obs in matched_pairs:
            old_pos = obj.position
            new_pos = obs["position"]

            # Update velocity with EMA smoothing
            raw_vx = new_pos[0] - old_pos[0]
            raw_vy = new_pos[1] - old_pos[1]
            alpha = 0.3  # velocity smoothing
            obj.velocity = (
                alpha * raw_vx + (1 - alpha) * obj.velocity[0],
                alpha * raw_vy + (1 - alpha) * obj.velocity[1],
            )

            obj.position = new_pos
            obj.size = obs.get("size", obj.size)
            obj.embedding = np.array(obs["embedding"])
            obj.label = obs.get("label", obj.label)
            obj.features = obs.get("features", obj.features)
            obj.age_frames += 1
            obj.last_seen_frame = self._frame_number
            obj.confidence = min(1.0, obj.confidence + 0.1)
            if obs.get("node_id"):
                obj.node_id = obs["node_id"]

            matched_results.append((obj, obs))

        # New objects (unmatched observations)
        new_objects = []
        for j, obs in enumerate(observations):
            if j not in used_obs:
                obj = TrackedObject(
                    id=self._new_id(),
                    embedding=np.array(obs["embedding"]),
                    label=obs.get("label", "unknown"),
                    position=obs["position"],
                    size=obs.get("size", (0, 0)),
                    age_frames=0,
                    last_seen_frame=self._frame_number,
                    node_id=obs.get("node_id"),
                    features=obs.get("features", {}),
                )
                self.tracked[obj.id] = obj
                new_objects.append(obs)

        # Disappeared objects (tracked but not seen)
        disappeared = []
        to_remove = []
        for i, obj in enumerate(tracked_list):
            if i not in used_tracked:
                frames_missing = self._frame_number - obj.last_seen_frame
                obj.confidence *= 0.8  # decay confidence

                if frames_missing > self.max_missing_frames or obj.confidence < 0.1:
                    disappeared.append(obj)
                    to_remove.append(obj.id)

        for obj_id in to_remove:
            self.tracked.pop(obj_id, None)

        # Cap tracked objects
        if len(self.tracked) > self.max_tracked:
            sorted_objs = sorted(self.tracked.values(), key=lambda o: o.confidence)
            for obj in sorted_objs[:len(self.tracked) - self.max_tracked]:
                self.tracked.pop(obj.id, None)

        # Generate predictions for next frame
        predictions = []
        for obj in self.tracked.values():
            pred_x = obj.position[0] + obj.velocity[0]
            pred_y = obj.position[1] + obj.velocity[1]
            predictions.append({
                "tracked_id": obj.id,
                "label": obj.label,
                "predicted_position": (pred_x, pred_y),
                "predicted_size": obj.size,
                "confidence": obj.confidence,
                "source": "temporal",
            })

        return TrackingResult(
            matched=matched_results,
            new_objects=new_objects,
            disappeared=disappeared,
            predictions=predictions,
        )

    def get_active_objects(self) -> list[TrackedObject]:
        """Return all currently tracked objects."""
        return list(self.tracked.values())

    def reset(self):
        """Clear all tracked objects (e.g., on scene change)."""
        self.tracked.clear()
        self._frame_number = 0

    @property
    def stats(self) -> dict:
        return {
            "tracked": len(self.tracked),
            "frame": self._frame_number,
            "avg_age": (
                sum(o.age_frames for o in self.tracked.values()) / max(len(self.tracked), 1)
            ),
            "avg_confidence": (
                sum(o.confidence for o in self.tracked.values()) / max(len(self.tracked), 1)
            ),
        }
