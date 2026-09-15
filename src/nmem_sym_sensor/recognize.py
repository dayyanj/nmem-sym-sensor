"""
Real-time recognition: the "inner voice" that interprets sensory input.

Runs alongside perception, not after it. When incoming observations
activate a known cluster, recognition is instant (cache hit, no LLM).
When they activate an unknown cluster, the LLM is invoked to interpret.

Cognitive load model:
  - Known (grounded, confident): instant recognition, no LLM call
  - Familiar (grounded, low confidence): background re-evaluation queued
  - Novel (stable cluster, ungrounded): LLM interpretation ("what is this?")
  - Raw (iconic/short-term): too early to interpret, skip

This mirrors how the brain spends decreasing processing time on
familiar stimuli. The first time you see a cup, you study it.
The thousandth time, you glance and know.
"""
import logging
import time
from dataclasses import dataclass, field

import asyncpg

log = logging.getLogger(__name__)


@dataclass
class Recognition:
    """Result of recognizing a set of sensory observations."""
    # What was recognized
    labels: list[str] = field(default_factory=list)         # known labels for matched clusters
    novel_descriptions: list[str] = field(default_factory=list)  # LLM descriptions for unknown clusters
    novel_proposals: list[dict] = field(default_factory=list)     # {cluster_id, proposed_label, confidence}

    # Timing
    recognition_ms: float = 0.0         # total recognition time
    cache_hits: int = 0                 # clusters recognized from cache
    llm_calls: int = 0                  # clusters sent to LLM
    skipped: int = 0                    # too early / too raw to interpret

    # What was seen
    activated_clusters: list[int] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)


class RecognitionEngine:
    """Real-time recognition engine.

    Maintains a cache of known cluster → label mappings. When new
    observations come in, matches them against known clusters and
    returns instant recognition for familiar patterns. Unknown
    patterns are sent to the LLM for interpretation.

    Usage::

        engine = RecognitionEngine(pool, llm_callable)

        # During video ingestion, after each frame:
        recognition = await engine.recognize(promoted_node_ids)
        # recognition.labels = ["red", "circle"]  (instant, cached)
        # recognition.novel_proposals = [{"cluster_id": 42, "proposed_label": "cup"}]
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        llm_callable=None,
        min_observations_to_interpret: int = 10,
        cache_ttl_s: float = 300.0,
    ):
        self.pool = pool
        self.llm = llm_callable
        self.min_obs = min_observations_to_interpret
        self.cache_ttl = cache_ttl_s

        # Recognition cache: cluster_id → (label, confidence, timestamp)
        self._cache: dict[int, tuple[str, float, float]] = {}

        # Cooldown: don't re-query the LLM for the same cluster too quickly
        self._cooldown: dict[int, float] = {}  # cluster_id → last_query_timestamp
        self._cooldown_s = 30.0  # seconds between LLM queries for same cluster

        # Stats
        self.total_cache_hits = 0
        self.total_llm_calls = 0
        self.total_recognitions = 0

    async def warm_cache(self):
        """Pre-load all grounded clusters into the recognition cache."""
        rows = await self.pool.fetch("""
            SELECT id, grounded_label, grounding_confidence
            FROM sensory_clusters
            WHERE grounded_label IS NOT NULL
              AND grounding_status IN ('confident', 'speculative')
        """)
        for r in rows:
            self._cache[r["id"]] = (
                r["grounded_label"],
                r["grounding_confidence"] or 0.0,
                time.time(),
            )
        log.info("Recognition cache warmed: %d grounded clusters", len(rows))

    async def recognize(
        self,
        node_ids: list[int],
        timestamp: float | None = None,
    ) -> Recognition:
        """Recognize a set of currently-active sensory nodes.

        This is the main entry point, called after each frame/audio window
        is processed and nodes are promoted from the iconic buffer.

        Args:
            node_ids: Sensory node IDs that are currently active.
            timestamp: Video timestamp (for logging).

        Returns:
            Recognition result with known labels and novel proposals.
        """
        t0 = time.time()
        result = Recognition(node_ids=list(node_ids))

        if not node_ids:
            return result

        # Find which clusters these nodes belong to
        cluster_matches = await self._find_cluster_matches(node_ids)

        for cluster_id, match_info in cluster_matches.items():
            result.activated_clusters.append(cluster_id)

            # Check cache first (instant recognition)
            cached = self._cache.get(cluster_id)
            if cached:
                label, confidence, cache_time = cached
                if time.time() - cache_time < self.cache_ttl:
                    result.labels.append(label)
                    result.cache_hits += 1
                    self.total_cache_hits += 1
                    continue

            # Check if cluster is grounded in DB
            cluster = await self.pool.fetchrow("""
                SELECT grounded_label, grounding_confidence, grounding_status,
                       cluster_type, total_observations
                FROM sensory_clusters WHERE id = $1
            """, cluster_id)

            if not cluster:
                continue

            # Known (grounded + confident): cache and return
            if (cluster["grounded_label"]
                    and cluster["grounding_status"] == "confident"):
                label = cluster["grounded_label"]
                conf = cluster["grounding_confidence"] or 0.0
                self._cache[cluster_id] = (label, conf, time.time())
                result.labels.append(label)
                result.cache_hits += 1
                self.total_cache_hits += 1
                continue

            # Familiar (grounded but low confidence): return cached, queue re-eval
            if cluster["grounded_label"]:
                label = cluster["grounded_label"]
                result.labels.append(f"{label}?")  # ? = uncertain
                result.cache_hits += 1
                continue

            # Too raw: not enough observations
            if cluster["total_observations"] < self.min_obs:
                result.skipped += 1
                continue

            # Not stable enough
            if cluster["cluster_type"] not in ("stable", "grounded"):
                result.skipped += 1
                continue

            # Novel: needs LLM interpretation
            if self.llm:
                proposal = await self._interpret_cluster(cluster_id, timestamp)
                if proposal:
                    result.novel_proposals.append(proposal)
                    result.llm_calls += 1
                    self.total_llm_calls += 1

        result.recognition_ms = (time.time() - t0) * 1000
        self.total_recognitions += 1

        if result.labels or result.novel_proposals:
            log.debug(
                "Recognition [%.1fs]: known=%s, novel=%s (%.1fms, %d cache, %d LLM)",
                timestamp or 0,
                result.labels or "none",
                [p["proposed_label"] for p in result.novel_proposals] or "none",
                result.recognition_ms,
                result.cache_hits,
                result.llm_calls,
            )

        return result

    async def _find_cluster_matches(self, node_ids: list[int]) -> dict[int, dict]:
        """Find which clusters the active nodes belong to."""
        rows = await self.pool.fetch("""
            SELECT DISTINCT cm.cluster_id, c.cluster_type, c.total_observations
            FROM sensory_cluster_members cm
            JOIN sensory_clusters c ON c.id = cm.cluster_id
            WHERE cm.node_id = ANY($1)
              AND c.cluster_type IN ('stable', 'grounded')
            ORDER BY c.total_observations DESC
        """, node_ids)

        return {
            r["cluster_id"]: {
                "type": r["cluster_type"],
                "observations": r["total_observations"],
            }
            for r in rows
        }

    async def _interpret_cluster(
        self,
        cluster_id: int,
        timestamp: float | None = None,
    ) -> dict | None:
        """Ask the LLM to interpret an unknown cluster.

        Respects cooldown to avoid hammering the LLM with the same
        question repeatedly within a short window.
        """
        # Cooldown check
        last_query = self._cooldown.get(cluster_id, 0)
        if time.time() - last_query < self._cooldown_s:
            return None

        self._cooldown[cluster_id] = time.time()

        # Generate description
        from nmem_sym_sensor.describe import describe_cluster
        desc = await describe_cluster(self.pool, cluster_id)
        if not desc.query:
            return None

        try:
            response = await self.llm(desc.query)
            response = response.strip().strip('"').strip("'")

            if not response or len(response) > 50:
                return None
            if response.lower() in ("unknown", "i don't know", "unsure", "ungrounded"):
                return None

            proposal = {
                "cluster_id": cluster_id,
                "proposed_label": response,
                "confidence": desc.confidence,
                "observation_count": desc.observation_count,
                "timestamp": timestamp,
            }

            log.info("Novel recognition at %.1fs: cluster #%d → \"%s\"",
                    timestamp or 0, cluster_id, response)

            return proposal

        except Exception as e:
            log.warning("LLM interpretation failed for cluster #%d: %s",
                       cluster_id, e)
            return None

    def stats(self) -> dict:
        """Return recognition engine statistics."""
        return {
            "cache_size": len(self._cache),
            "total_recognitions": self.total_recognitions,
            "total_cache_hits": self.total_cache_hits,
            "total_llm_calls": self.total_llm_calls,
            "cache_hit_rate": (
                self.total_cache_hits / max(self.total_cache_hits + self.total_llm_calls, 1)
            ),
        }


# ── Activation matching ──────────────────────────────────

async def find_matching_clusters(
    pool: asyncpg.Pool,
    node_ids: list[int],
    min_overlap: int = 2,
) -> list[dict]:
    """Find clusters that significantly overlap with active nodes.

    Instead of exact membership, this uses a softer matching:
    if N of a cluster's members are currently active, the cluster
    is considered "activated" and eligible for recognition.

    This handles partial observations — seeing just a handle and
    brown color can activate the "cup" cluster even if the full
    cylinder isn't visible.

    Args:
        pool: Database connection pool.
        node_ids: Currently active sensory node IDs.
        min_overlap: Minimum number of matching members to consider activated.

    Returns:
        List of dicts with cluster_id, overlap_count, total_members.
    """
    rows = await pool.fetch("""
        SELECT cm.cluster_id,
               COUNT(*) as overlap_count,
               c.member_count as total_members,
               c.grounded_label,
               c.grounding_confidence,
               c.total_observations
        FROM sensory_cluster_members cm
        JOIN sensory_clusters c ON c.id = cm.cluster_id
        WHERE cm.node_id = ANY($1)
          AND c.cluster_type IN ('stable', 'grounded')
        GROUP BY cm.cluster_id, c.member_count, c.grounded_label,
                 c.grounding_confidence, c.total_observations
        HAVING COUNT(*) >= $2
        ORDER BY COUNT(*) DESC
    """, node_ids, min_overlap)

    return [
        {
            "cluster_id": r["cluster_id"],
            "overlap": r["overlap_count"],
            "total_members": r["total_members"],
            "overlap_ratio": r["overlap_count"] / max(r["total_members"], 1),
            "grounded_label": r["grounded_label"],
            "confidence": r["grounding_confidence"],
            "observations": r["total_observations"],
        }
        for r in rows
    ]
