"""Chronoception — the sensory system's sense of time and temporal context.

The brain doesn't just perceive WHAT is in front of it — it perceives WHEN
things happen, how long they've been present, and what temporal context
they belong to. Chronoception manages:

1. Scene context — "where am I?" (kitchen, bedroom, outdoors)
2. Active set — "what do I expect to see here?" (scene-gated clusters)
3. Temporal decay — "how recently did I see this?" (activation fading)
4. Scene transitions — "the world just changed" (decay bridge)
5. Duration awareness — "I've been here for 5 minutes" (stability tracking)
6. Spatial graph — allocentric object-to-object relations within a scene

Allocentric relations (observer-independent):
  near, above, below, on, contains, attached_to, faces, opposite, between

Egocentric relations (left_of, right_of, behind, in_front_of) are NOT
stored — they depend on observer position/orientation and will be computed
at query time when head tracking (accelerometer/compass) is available.

Integration with existing systems:
- SceneMemory (familiarity.py) provides coarse frame signatures
- ClusterCache (familiarity.py) provides cluster matching
- EmbeddingPredictor provides prediction error as novelty signal
- sensory_dreamstate provides offline scene consolidation
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import asyncpg
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ActiveCluster:
    """A cluster in the active set with temporal activation state."""
    cluster_id: int
    centroid: np.ndarray         # 512-dim embedding for matching
    label: str                   # grounded label (e.g., "triangle")
    activation: float = 1.0     # 0.0 = dormant, 1.0 = fully active
    decaying: bool = False       # True when scene deactivated
    source: str = "scene"        # "scene", "observation", "recall"
    last_reinforced: float = 0.0  # monotonic timestamp
    reinforcement_count: int = 0


# Valid allocentric spatial relations (observer-independent)
ALLOCENTRIC_RELATIONS = frozenset({
    "near",          # proximity (doesn't change with observer position)
    "above",         # gravity-relative (universal)
    "below",         # gravity-relative (universal)
    "on",            # physical contact/support
    "contains",      # containment
    "attached_to",   # physical attachment
    "faces",         # object-to-object orientation
    "opposite",      # across from each other
    "between",       # intermediary position
})


@dataclass
class SpatialEdge:
    """An allocentric spatial relation between two clusters in a scene."""
    cluster_a_id: int
    cluster_b_id: int
    relation: str               # one of ALLOCENTRIC_RELATIONS
    confidence: float = 1.0
    observation_count: int = 1


@dataclass
class SceneContext:
    """A recognised scene — a retrieval cue that primes cluster loading.

    Walking into the kitchen → kitchen clusters activate.
    A scene is not just a place — it's an expectation of what you'll find,
    and how things are arranged relative to each other.

    The spatial_graph stores allocentric (observer-independent) relations:
    "the TV is near the wall", "the fan is above the table". Egocentric
    relations (left/right/behind) are computed at query time from observer
    position + orientation when sensors are available.
    """
    scene_id: int
    scene_embedding: np.ndarray          # coarse 1024-dim signature
    associated_clusters: set[int] = field(default_factory=set)
    spatial_graph: list[SpatialEdge] = field(default_factory=list)
    visit_count: int = 1
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    myelinated: bool = False             # well-known scene, instant load


class ActiveSet:
    """Tier 1: Scene-gated clusters with temporal decay.

    The sensory equivalent of working memory. Contains only the clusters
    relevant to the current scene — typically 50-200 out of potentially
    millions in deep storage.

    Like a transformer's attention window: finite, relevant, dynamically
    loaded based on context.
    """

    def __init__(
        self,
        decay_rate: float = 0.92,
        deactivation_threshold: float = 0.05,
        max_clusters: int = 200,
    ):
        self._clusters: dict[int, ActiveCluster] = {}
        self.decay_rate = decay_rate
        self.deactivation_threshold = deactivation_threshold
        self.max_clusters = max_clusters

        # Tracking
        self.scene_transitions = 0
        self.total_reinforcements = 0

    def activate_scene(self, scene: SceneContext, centroids: dict[int, tuple[np.ndarray, str]]):
        """Load a scene's clusters into the active set.

        Args:
            scene: The scene context to activate.
            centroids: {cluster_id: (centroid_embedding, label)} for clusters
                       in this scene. Loaded from DB by the caller.
        """
        now = time.monotonic()
        for cid in scene.associated_clusters:
            if cid in centroids:
                centroid, label = centroids[cid]
                if cid in self._clusters:
                    # Already active — boost and cancel decay
                    self._clusters[cid].activation = 1.0
                    self._clusters[cid].decaying = False
                    self._clusters[cid].last_reinforced = now
                else:
                    self._clusters[cid] = ActiveCluster(
                        cluster_id=cid,
                        centroid=centroid,
                        label=label,
                        activation=1.0,
                        source="scene",
                        last_reinforced=now,
                    )
        self.scene_transitions += 1
        log.info("Scene %d activated: %d clusters loaded (%d total active)",
                 scene.scene_id, len(scene.associated_clusters), len(self._clusters))

    def deactivate_scene(self, scene: SceneContext, keep_hot: set[int] | None = None):
        """Begin decay for a scene's clusters.

        Don't purge immediately — fade them. Clusters shared with the new
        scene (or recently reinforced) stay active.
        """
        keep = keep_hot or set()
        for cid in scene.associated_clusters:
            if cid in self._clusters and cid not in keep:
                self._clusters[cid].decaying = True

    def tick(self):
        """Per-frame update: decay flagged clusters, remove dead ones.

        Called once per frame. Clusters below threshold get removed from
        the active set (demoted to Tier 2).
        """
        to_remove = []
        for cid, ac in self._clusters.items():
            if ac.decaying:
                ac.activation *= self.decay_rate
                if ac.activation < self.deactivation_threshold:
                    to_remove.append(cid)
        for cid in to_remove:
            del self._clusters[cid]

    def reinforce(self, cluster_id: int):
        """Object seen in frame → boost activation, cancel decay.

        This is the key mechanism: if an object from the old scene appears
        in the new scene (you carried it with you), it stays active.
        """
        if cluster_id in self._clusters:
            self._clusters[cluster_id].activation = 1.0
            self._clusters[cluster_id].decaying = False
            self._clusters[cluster_id].last_reinforced = time.monotonic()
            self._clusters[cluster_id].reinforcement_count += 1
            self.total_reinforcements += 1

    def activate_observed(self, cluster_id: int, centroid: np.ndarray, label: str):
        """A new cluster observed in the current scene — add to active set.

        This handles objects not in the scene's pre-loaded set (novel objects
        discovered during exploration of a new scene).
        """
        now = time.monotonic()
        if cluster_id in self._clusters:
            self.reinforce(cluster_id)
        else:
            # Evict lowest-activation cluster if at capacity
            if len(self._clusters) >= self.max_clusters:
                weakest = min(self._clusters, key=lambda k: self._clusters[k].activation)
                del self._clusters[weakest]

            self._clusters[cluster_id] = ActiveCluster(
                cluster_id=cluster_id,
                centroid=centroid,
                label=label,
                activation=1.0,
                source="observation",
                last_reinforced=now,
            )

    def match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.80,
    ) -> tuple[int | None, str | None, float]:
        """Match an embedding against the active set.

        Only searches Tier 1 (active clusters). If no match, the caller
        should cascade to Tier 2 (pgvector).

        Returns (cluster_id, label, similarity) or (None, None, 0.0).
        """
        if not self._clusters:
            return None, None, 0.0

        emb = embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        best_id, best_label, best_sim = None, None, 0.0
        for cid, ac in self._clusters.items():
            # Only match clusters above minimum activation
            if ac.activation < 0.1:
                continue
            sim = float(np.dot(emb, ac.centroid))
            if sim > best_sim and sim >= threshold:
                best_id, best_label, best_sim = cid, ac.label, sim

        return best_id, best_label, best_sim

    @property
    def size(self) -> int:
        return len(self._clusters)

    @property
    def active_ids(self) -> set[int]:
        return set(self._clusters.keys())

    def stats(self) -> dict:
        if not self._clusters:
            return {"active": 0, "decaying": 0, "mean_activation": 0}
        activations = [ac.activation for ac in self._clusters.values()]
        return {
            "active": len(self._clusters),
            "decaying": sum(1 for ac in self._clusters.values() if ac.decaying),
            "mean_activation": round(sum(activations) / len(activations), 3),
            "min_activation": round(min(activations), 3),
            "scene_transitions": self.scene_transitions,
        }


class Chronoception:
    """The sensory system's sense of time and temporal context.

    Manages scene recognition, active set loading, and temporal decay.
    Sits between the raw perception pipeline and the memory system.
    """

    def __init__(
        self,
        scene_match_threshold: float = 0.85,
        scene_buffer_size: int = 5,
    ):
        self.active_set = ActiveSet()
        self.current_scene: SceneContext | None = None
        self._scene_match_threshold = scene_match_threshold

        # Recent scene embeddings for transition detection
        self._recent_scene_embs: deque[np.ndarray] = deque(maxlen=scene_buffer_size)

        # Timing
        self._scene_entered_at: float = 0.0
        self._frame_count_in_scene: int = 0

    async def on_frame(
        self,
        scene_embedding: np.ndarray,
        pool: asyncpg.Pool,
    ) -> dict:
        """Called each frame. Detects scene changes and manages the active set.

        Args:
            scene_embedding: Coarse 1024-dim frame signature (from SceneMemory).
            pool: DB pool for loading scene snapshots.

        Returns:
            Dict with scene status info.
        """
        self._frame_count_in_scene += 1
        self.active_set.tick()

        # Detect scene change
        if self._recent_scene_embs:
            prev = self._recent_scene_embs[-1]
            sim = float(np.dot(scene_embedding, prev))
            scene_changed = sim < self._scene_match_threshold
        else:
            scene_changed = True  # first frame

        self._recent_scene_embs.append(scene_embedding)

        result = {
            "scene_changed": scene_changed,
            "scene_duration_frames": self._frame_count_in_scene,
            "active_set": self.active_set.stats(),
        }

        if scene_changed:
            await self._handle_scene_change(scene_embedding, pool)
            result["new_scene_id"] = self.current_scene.scene_id if self.current_scene else None

        return result

    async def _handle_scene_change(
        self,
        scene_embedding: np.ndarray,
        pool: asyncpg.Pool,
    ):
        """Handle a scene transition: recognise or create, load clusters."""
        old_scene = self.current_scene

        # Try to recognise this scene from saved snapshots
        recognised = await self._recognise_scene(scene_embedding, pool)

        if recognised:
            self.current_scene = recognised
            log.info("Scene recognised: id=%d (visited %d times)",
                     recognised.scene_id, recognised.visit_count)

            # Update visit count
            await pool.execute(
                """UPDATE scene_snapshots
                   SET visit_count = visit_count + 1,
                       last_seen = NOW()
                   WHERE id = $1""",
                recognised.scene_id,
            )
        else:
            # New scene — create snapshot
            self.current_scene = await self._create_scene(scene_embedding, pool)
            log.info("New scene created: id=%d", self.current_scene.scene_id)

        # Decay old scene's clusters, load new scene's
        if old_scene:
            # Keep clusters that are in BOTH scenes (shared objects)
            shared = old_scene.associated_clusters & self.current_scene.associated_clusters
            self.active_set.deactivate_scene(old_scene, keep_hot=shared)

        # Load new scene's clusters into active set
        if self.current_scene.associated_clusters:
            centroids = await self._load_cluster_centroids(
                self.current_scene.associated_clusters, pool,
            )
            self.active_set.activate_scene(self.current_scene, centroids)

        self._scene_entered_at = time.monotonic()
        self._frame_count_in_scene = 0

    async def _recognise_scene(
        self,
        embedding: np.ndarray,
        pool: asyncpg.Pool,
    ) -> SceneContext | None:
        """Match against saved scene snapshots using pgvector."""
        emb_str = str(embedding.tolist())
        row = await pool.fetchrow(
            """SELECT id, embedding::text as emb,
                      1 - (embedding <=> $1::vector) as similarity,
                      visit_count, first_seen, last_seen
               FROM scene_snapshots
               ORDER BY embedding <=> $1::vector
               LIMIT 1""",
            emb_str,
        )

        if not row or row["similarity"] < self._scene_match_threshold:
            return None

        # Load cluster members
        member_rows = await pool.fetch(
            "SELECT cluster_id FROM scene_members WHERE scene_id = $1",
            row["id"],
        )

        # Load spatial graph
        edge_rows = await pool.fetch(
            """SELECT cluster_a_id, cluster_b_id, relation, confidence, observation_count
               FROM scene_spatial_edges WHERE scene_id = $1""",
            row["id"],
        )
        spatial_graph = [
            SpatialEdge(
                cluster_a_id=e["cluster_a_id"],
                cluster_b_id=e["cluster_b_id"],
                relation=e["relation"],
                confidence=e["confidence"],
                observation_count=e["observation_count"],
            )
            for e in edge_rows
        ]

        return SceneContext(
            scene_id=row["id"],
            scene_embedding=embedding,
            associated_clusters={r["cluster_id"] for r in member_rows},
            spatial_graph=spatial_graph,
            visit_count=row["visit_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )

    async def _create_scene(
        self,
        embedding: np.ndarray,
        pool: asyncpg.Pool,
    ) -> SceneContext:
        """Create a new scene snapshot."""
        emb_str = str(embedding.tolist())
        scene_id = await pool.fetchval(
            """INSERT INTO scene_snapshots (embedding)
               VALUES ($1::vector)
               RETURNING id""",
            emb_str,
        )
        return SceneContext(
            scene_id=scene_id,
            scene_embedding=embedding,
            associated_clusters=set(),
            visit_count=1,
            first_seen=datetime.now(UTC),
            last_seen=datetime.now(UTC),
        )

    async def observe_cluster_in_scene(
        self,
        cluster_id: int,
        pool: asyncpg.Pool,
    ):
        """Record that a cluster was observed in the current scene.

        Called when a visual primitive matches a grounded cluster during
        frame processing. Builds scene membership over time.
        """
        if self.current_scene is None:
            return

        self.current_scene.associated_clusters.add(cluster_id)

        await pool.execute(
            """INSERT INTO scene_members (scene_id, cluster_id)
               VALUES ($1, $2)
               ON CONFLICT (scene_id, cluster_id)
               DO UPDATE SET observation_count = scene_members.observation_count + 1,
                             last_seen_at = NOW()""",
            self.current_scene.scene_id,
            cluster_id,
        )

    async def observe_spatial_relation(
        self,
        cluster_a_id: int,
        cluster_b_id: int,
        relation: str,
        pool: asyncpg.Pool,
    ):
        """Record an allocentric spatial relation between clusters in this scene.

        Only accepts allocentric relations (near, above, below, on, etc).
        Egocentric relations (left_of, right_of) are rejected — they depend
        on observer position and should be computed at query time.

        Called during frame analysis when spatial relations between
        recognised objects are detected.
        """
        if self.current_scene is None:
            return

        if relation not in ALLOCENTRIC_RELATIONS:
            log.debug("Rejected egocentric relation '%s' — only allocentric stored", relation)
            return

        # Canonical ordering: smaller cluster_id first (undirected for
        # symmetric relations like 'near', directed for 'above'/'below')
        directed = relation in ("above", "below", "on", "contains")
        if not directed and cluster_a_id > cluster_b_id:
            cluster_a_id, cluster_b_id = cluster_b_id, cluster_a_id

        await pool.execute(
            """INSERT INTO scene_spatial_edges
                   (scene_id, cluster_a_id, cluster_b_id, relation)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (scene_id, cluster_a_id, cluster_b_id, relation)
               DO UPDATE SET observation_count = scene_spatial_edges.observation_count + 1,
                             confidence = LEAST(1.0, scene_spatial_edges.confidence + 0.01),
                             updated_at = NOW()""",
            self.current_scene.scene_id,
            cluster_a_id,
            cluster_b_id,
            relation,
        )

    async def query_spatial_context(
        self,
        cluster_id: int,
        pool: asyncpg.Pool,
    ) -> list[dict]:
        """Query: what's near/around this cluster in the current scene?

        Returns allocentric neighbours. When head tracking is available,
        the caller can layer egocentric directions on top.
        """
        if self.current_scene is None:
            return []

        rows = await pool.fetch(
            """SELECT cluster_a_id, cluster_b_id, relation, confidence
               FROM scene_spatial_edges
               WHERE scene_id = $1
                 AND (cluster_a_id = $2 OR cluster_b_id = $2)
               ORDER BY confidence DESC""",
            self.current_scene.scene_id,
            cluster_id,
        )

        result = []
        for r in rows:
            other = r["cluster_b_id"] if r["cluster_a_id"] == cluster_id else r["cluster_a_id"]
            result.append({
                "cluster_id": other,
                "relation": r["relation"],
                "confidence": r["confidence"],
            })
        return result

    async def _load_cluster_centroids(
        self,
        cluster_ids: set[int],
        pool: asyncpg.Pool,
    ) -> dict[int, tuple[np.ndarray, str]]:
        """Load centroid embeddings for a set of cluster IDs."""
        if not cluster_ids:
            return {}

        rows = await pool.fetch(
            """SELECT id, visual_centroid::text as vc, grounded_label
               FROM sensory_clusters
               WHERE id = ANY($1)
                 AND visual_centroid IS NOT NULL
                 AND grounded_label IS NOT NULL""",
            list(cluster_ids),
        )

        result = {}
        for r in rows:
            try:
                vec = np.array(
                    [float(x) for x in r["vc"].strip("[]").split(",")],
                    dtype=np.float32,
                )
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                result[r["id"]] = (vec, r["grounded_label"])
            except (ValueError, AttributeError):
                continue

        return result

    @property
    def scene_duration_s(self) -> float:
        """How long we've been in the current scene."""
        if self._scene_entered_at == 0:
            return 0.0
        return time.monotonic() - self._scene_entered_at

    @property
    def scene_duration_frames(self) -> int:
        return self._frame_count_in_scene
