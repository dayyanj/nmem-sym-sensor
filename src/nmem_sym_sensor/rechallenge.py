"""
Grounding rechallenge: living hypotheses, not permanent stamps.

A label assigned to a sensory cluster is a hypothesis. As the cluster
evolves — gaining new members, shifting its centroid, binding to new
cross-modal observations — the label may no longer fit.

The rechallenge system:
  1. Detects when a grounded cluster has changed significantly
  2. Decays confidence as new observations diverge from the grounding snapshot
  3. Triggers re-description and LLM re-query when thresholds are crossed
  4. Promotes, demotes, or revokes labels based on LLM responses

Grounding status lifecycle:
    ungrounded → speculative → confident → disputed → speculative/ungrounded
                     ↑              ↑          ↓
                     └──────────────┘     (rechallenge)

A "cup" that starts holding soil and plants should become "pot".
A "ball" that turns out to be many different round objects should
be revoked entirely (the cluster was too broad for a single label).
"""
import logging
from dataclasses import dataclass

import asyncpg
import numpy as np

from nmem_sym_sensor import config

log = logging.getLogger(__name__)


@dataclass
class RechallengeResult:
    """Result of a single rechallenge evaluation."""
    cluster_id: int
    previous_label: str
    action: str                     # confirmed | revised | demoted | revoked
    new_label: str | None = None    # set if revised
    reason: str = ""                # why the rechallenge was triggered
    old_confidence: float = 0.0
    new_confidence: float = 0.0
    drift: float = 0.0             # centroid drift since grounding
    observations_since_grounding: int = 0


# ── Drift detection ──────────────────────────────────────

async def compute_centroid_drift(
    pool: asyncpg.Pool,
    cluster_id: int,
) -> float:
    """Measure how much a cluster's centroid has drifted since grounding.

    Compares the current centroid (recomputed from members) against the
    snapshot taken at grounding time. Returns cosine distance [0, 1].
    0 = identical, 1 = orthogonal.
    """
    cluster = await pool.fetchrow(
        """
        SELECT modality, grounding_centroid_snapshot
        FROM sensory_clusters WHERE id = $1
        """,
        cluster_id,
    )

    if not cluster or not cluster["grounding_centroid_snapshot"]:
        return 0.0

    # Parse stored snapshot
    snapshot_str = cluster["grounding_centroid_snapshot"].strip("[]")
    if not snapshot_str:
        return 0.0
    snapshot = np.array([float(x) for x in snapshot_str.split(",")])

    # Compute current centroid from members
    modality = cluster["modality"]
    emb_col = "visual_embedding" if modality == "visual" else "audio_embedding"

    embeddings = await pool.fetch(
        f"""
        SELECT n.{emb_col}::text as emb
        FROM sensory_cluster_members cm
        JOIN sensory_nodes n ON n.id = cm.node_id
        WHERE cm.cluster_id = $1 AND n.{emb_col} IS NOT NULL
        """,
        cluster_id,
    )

    if not embeddings:
        return 0.0

    vecs = []
    for row in embeddings:
        emb_str = row["emb"].strip("[]")
        if emb_str:
            vecs.append([float(x) for x in emb_str.split(",")])

    if not vecs:
        return 0.0

    current_centroid = np.mean(vecs, axis=0)

    # Cosine distance
    norm_a = np.linalg.norm(snapshot)
    norm_b = np.linalg.norm(current_centroid)
    if norm_a == 0 or norm_b == 0:
        return 1.0

    cosine_sim = np.dot(snapshot, current_centroid) / (norm_a * norm_b)
    return float(1.0 - max(min(cosine_sim, 1.0), -1.0))


# ── Confidence decay ─────────────────────────────────────

async def decay_grounding_confidence(pool: asyncpg.Pool) -> list[dict]:
    """Apply confidence decay to all grounded clusters.

    Confidence decays proportionally to how many new members have been
    added since grounding. The idea: the original label was based on
    the members present at grounding time. New members dilute certainty.

    Returns list of clusters whose confidence dropped below demotion threshold.
    """
    # Fetch grounded clusters with enough observations for decay
    clusters = await pool.fetch(
        """
        SELECT id, grounded_label, grounding_confidence,
               observations_at_grounding, total_observations,
               member_count, grounding_status
        FROM sensory_clusters
        WHERE grounded_label IS NOT NULL
          AND grounding_status IN ('speculative', 'confident')
          AND total_observations > observations_at_grounding
        """,
    )

    demoted = []
    for row in clusters:
        new_obs = row["total_observations"] - row["observations_at_grounding"]
        decay = new_obs * config.RECHALLENGE_DECAY_RATE
        new_confidence = max(0.0, row["grounding_confidence"] - decay)

        await pool.execute(
            """
            UPDATE sensory_clusters
            SET grounding_confidence = $1, updated_at = NOW()
            WHERE id = $2
            """,
            round(new_confidence, 4),
            row["id"],
        )

        if new_confidence < config.RECHALLENGE_DEMOTION_THRESHOLD:
            if row["grounding_status"] == "confident":
                await pool.execute(
                    """
                    UPDATE sensory_clusters
                    SET grounding_status = 'disputed', updated_at = NOW()
                    WHERE id = $1
                    """,
                    row["id"],
                )
                demoted.append({
                    "cluster_id": row["id"],
                    "label": row["grounded_label"],
                    "old_confidence": row["grounding_confidence"],
                    "new_confidence": new_confidence,
                    "new_observations": new_obs,
                })
                log.info("Demoted cluster #%d ('%s') to disputed: confidence %.3f → %.3f",
                        row["id"], row["grounded_label"],
                        row["grounding_confidence"], new_confidence)

    return demoted


# ── Rechallenge candidates ───────────────────────────────

async def find_rechallenge_candidates(pool: asyncpg.Pool) -> list[dict]:
    """Find grounded clusters that need rechallenging.

    A cluster is a candidate if any of:
      1. Enough new observations since last challenge (RECHALLENGE_OBSERVATION_INTERVAL)
      2. Centroid has drifted beyond RECHALLENGE_DRIFT_THRESHOLD
      3. Confidence has decayed below RECHALLENGE_DEMOTION_THRESHOLD
      4. Status is 'disputed'

    Returns list of candidate dicts with reason.
    """
    candidates = []

    # Fetch grounded clusters that are mature enough
    clusters = await pool.fetch(
        """
        SELECT id, grounded_label, grounding_confidence, grounding_status,
               total_observations, observations_at_grounding,
               observations_at_last_challenge, challenge_count,
               challenge_failures, modality
        FROM sensory_clusters
        WHERE grounded_label IS NOT NULL
          AND total_observations >= $1
        ORDER BY total_observations DESC
        """,
        config.RECHALLENGE_MIN_MATURITY,
    )

    for row in clusters:
        reasons = []
        obs_since_challenge = row["total_observations"] - row["observations_at_last_challenge"]
        obs_since_grounding = row["total_observations"] - row["observations_at_grounding"]

        # 1. Observation interval
        if obs_since_challenge >= config.RECHALLENGE_OBSERVATION_INTERVAL:
            reasons.append("observation_interval")

        # 2. Disputed status (from confidence decay)
        if row["grounding_status"] == "disputed":
            reasons.append("confidence_decay")

        # 3. Drift check (only if we have an interval trigger or disputed status)
        if reasons:
            drift = await compute_centroid_drift(pool, row["id"])
            if drift >= config.RECHALLENGE_DRIFT_THRESHOLD:
                reasons.append("drift")
        else:
            # Even without interval trigger, check drift — it's the most
            # important signal that the cluster's nature has changed
            drift = await compute_centroid_drift(pool, row["id"])
            if drift >= config.RECHALLENGE_DRIFT_THRESHOLD:
                reasons.append("drift")

        if not reasons:
            continue

        candidates.append({
            "cluster_id": row["id"],
            "label": row["grounded_label"],
            "confidence": row["grounding_confidence"],
            "status": row["grounding_status"],
            "total_observations": row["total_observations"],
            "observations_since_grounding": obs_since_grounding,
            "observations_since_challenge": obs_since_challenge,
            "challenge_count": row["challenge_count"],
            "challenge_failures": row["challenge_failures"],
            "drift": drift if "drift" in reasons else 0.0,
            "reasons": reasons,
        })

    log.info("Found %d rechallenge candidates", len(candidates))
    return candidates


# ── Rechallenge execution ────────────────────────────────

async def rechallenge_cluster(
    pool: asyncpg.Pool,
    cluster_id: int,
    llm_callable,
    reason: str = "manual",
) -> RechallengeResult:
    """Rechallenge a single cluster's grounding label.

    Generates a description of the cluster's current state, asks
    the LLM what it is, and compares to the existing label.

    Possible outcomes:
      - confirmed: LLM agrees with current label → boost confidence
      - revised: LLM proposes a different label → update grounding
      - demoted: LLM is unsure → demote to speculative
      - revoked: too many failures → remove grounding entirely

    Args:
        pool: Sensory DB connection pool.
        cluster_id: Cluster to rechallenge.
        llm_callable: Async function(prompt) → response string.
        reason: Why this rechallenge was triggered.

    Returns:
        RechallengeResult with the outcome.
    """
    from nmem_sym_sensor.describe import describe_cluster

    # Fetch current state
    cluster = await pool.fetchrow(
        """
        SELECT id, grounded_label, grounding_confidence, grounding_status,
               total_observations, observations_at_grounding,
               challenge_count, challenge_failures
        FROM sensory_clusters WHERE id = $1
        """,
        cluster_id,
    )

    if not cluster or not cluster["grounded_label"]:
        return RechallengeResult(
            cluster_id=cluster_id,
            previous_label="",
            action="skipped",
            reason="not grounded",
        )

    old_label = cluster["grounded_label"]
    old_confidence = cluster["grounding_confidence"] or 0.0
    obs_since = cluster["total_observations"] - cluster["observations_at_grounding"]
    drift = await compute_centroid_drift(pool, cluster_id)

    # Generate rechallenge description
    desc = await describe_cluster(pool, cluster_id)

    # Build rechallenge prompt — different from initial naming.
    # We tell the LLM what we currently call it and ask if that's still right.
    prompt = (
        f"I previously identified a recurring sensory pattern as \"{old_label}\" "
        f"based on {cluster['observations_at_grounding']} observations. "
        f"Since then, {obs_since} new observations have been added "
        f"and the pattern may have evolved.\n\n"
        f"Current sensory description:\n{desc.structural}\n\n"
        f"Based on the current features, is \"{old_label}\" still the best label? "
        f"Respond with one of:\n"
        f"  CONFIRMED - if \"{old_label}\" is still correct\n"
        f"  REVISED: <new label> - if a different name fits better (e.g., REVISED: flower pot)\n"
        f"  UNSURE - if the pattern is too ambiguous to name confidently"
    )

    try:
        response = await llm_callable(prompt)
        response = response.strip()
    except Exception as e:
        log.warning("LLM rechallenge failed for cluster #%d: %s", cluster_id, e)
        return RechallengeResult(
            cluster_id=cluster_id,
            previous_label=old_label,
            action="error",
            reason=str(e),
            old_confidence=old_confidence,
            drift=drift,
            observations_since_grounding=obs_since,
        )

    # Parse response
    response_upper = response.upper().strip()

    if response_upper.startswith("CONFIRMED"):
        return await _handle_confirmed(pool, cluster, old_label, old_confidence, drift, obs_since, reason)

    elif response_upper.startswith("REVISED:"):
        new_label = response[response.upper().index("REVISED:") + 8:].strip().strip('"').strip("'")
        if not new_label or len(new_label) > 50:
            return await _handle_unsure(pool, cluster, old_label, old_confidence, drift, obs_since, reason)
        return await _handle_revised(pool, cluster, old_label, new_label, old_confidence, drift, obs_since, reason)

    elif response_upper.startswith("UNSURE"):
        return await _handle_unsure(pool, cluster, old_label, old_confidence, drift, obs_since, reason)

    else:
        # Try to interpret freeform response
        if old_label.lower() in response.lower():
            return await _handle_confirmed(pool, cluster, old_label, old_confidence, drift, obs_since, reason)
        elif len(response.split()) <= 4:
            # Short response might be a revised label
            return await _handle_revised(pool, cluster, old_label, response.strip(), old_confidence, drift, obs_since, reason)
        else:
            return await _handle_unsure(pool, cluster, old_label, old_confidence, drift, obs_since, reason)


async def _handle_confirmed(pool, cluster, old_label, old_confidence, drift, obs_since, reason):
    """Label confirmed — boost confidence, snapshot current centroid."""
    new_confidence = min(1.0, old_confidence + 0.1)
    new_status = "confident" if new_confidence >= config.GROUNDING_MIN_CONFIDENCE else "speculative"

    # Snapshot current centroid (reset drift baseline)
    centroid_snapshot = await _get_current_centroid(pool, cluster["id"], cluster.get("modality"))

    await pool.execute(
        """
        UPDATE sensory_clusters
        SET grounding_confidence = $1,
            grounding_status = $2,
            observations_at_last_challenge = total_observations,
            challenge_count = challenge_count + 1,
            grounding_centroid_snapshot = $3,
            updated_at = NOW()
        WHERE id = $4
        """,
        round(new_confidence, 4),
        new_status,
        centroid_snapshot,
        cluster["id"],
    )

    await _log_rechallenge(pool, cluster["id"], old_label, new_confidence,
                           "rechallenge_confirmed", None, reason)

    log.info("Rechallenge CONFIRMED cluster #%d as '%s' (confidence %.3f → %.3f)",
            cluster["id"], old_label, old_confidence, new_confidence)

    return RechallengeResult(
        cluster_id=cluster["id"],
        previous_label=old_label,
        action="confirmed",
        reason=reason,
        old_confidence=old_confidence,
        new_confidence=new_confidence,
        drift=drift,
        observations_since_grounding=obs_since,
    )


async def _handle_revised(pool, cluster, old_label, new_label, old_confidence, drift, obs_since, reason):
    """Label revised — update to new label, record failure for old label."""
    failures = cluster["challenge_failures"] + 1

    # Should we revoke entirely or accept the new label?
    if failures >= config.RECHALLENGE_REVOKE_AFTER_FAILURES:
        # Too many revisions — the cluster is too unstable for a single label
        return await _handle_revoke(pool, cluster, old_label, new_label, old_confidence, drift, obs_since, reason, failures)

    centroid_snapshot = await _get_current_centroid(pool, cluster["id"], cluster.get("modality"))

    await pool.execute(
        """
        UPDATE sensory_clusters
        SET grounded_label = $1,
            grounding_confidence = $2,
            grounding_status = 'speculative',
            observations_at_grounding = total_observations,
            observations_at_last_challenge = total_observations,
            challenge_count = challenge_count + 1,
            challenge_failures = $3,
            grounding_centroid_snapshot = $4,
            updated_at = NOW()
        WHERE id = $5
        """,
        new_label,
        round(config.GROUNDING_MIN_CONFIDENCE, 4),  # reset to minimum
        failures,
        centroid_snapshot,
        cluster["id"],
    )

    await _log_rechallenge(pool, cluster["id"], new_label, config.GROUNDING_MIN_CONFIDENCE,
                           "rechallenge_revised", old_label, reason)

    log.info("Rechallenge REVISED cluster #%d: '%s' → '%s' (failure #%d)",
            cluster["id"], old_label, new_label, failures)

    return RechallengeResult(
        cluster_id=cluster["id"],
        previous_label=old_label,
        action="revised",
        new_label=new_label,
        reason=reason,
        old_confidence=old_confidence,
        new_confidence=config.GROUNDING_MIN_CONFIDENCE,
        drift=drift,
        observations_since_grounding=obs_since,
    )


async def _handle_unsure(pool, cluster, old_label, old_confidence, drift, obs_since, reason):
    """LLM is unsure — demote to speculative, decay confidence."""
    new_confidence = max(0.0, old_confidence - 0.15)

    await pool.execute(
        """
        UPDATE sensory_clusters
        SET grounding_confidence = $1,
            grounding_status = CASE WHEN $1 < $2 THEN 'disputed' ELSE 'speculative' END,
            observations_at_last_challenge = total_observations,
            challenge_count = challenge_count + 1,
            updated_at = NOW()
        WHERE id = $3
        """,
        round(new_confidence, 4),
        config.RECHALLENGE_DEMOTION_THRESHOLD,
        cluster["id"],
    )

    log.info("Rechallenge UNSURE for cluster #%d ('%s'), confidence %.3f → %.3f",
            cluster["id"], old_label, old_confidence, new_confidence)

    return RechallengeResult(
        cluster_id=cluster["id"],
        previous_label=old_label,
        action="demoted",
        reason=reason,
        old_confidence=old_confidence,
        new_confidence=new_confidence,
        drift=drift,
        observations_since_grounding=obs_since,
    )


async def _handle_revoke(pool, cluster, old_label, proposed_label, old_confidence, drift, obs_since, reason, failures):
    """Too many failures — revoke grounding entirely.

    The cluster keeps its members but loses its label. It becomes
    eligible for fresh naming via auto_name() again.
    """
    await pool.execute(
        """
        UPDATE sensory_clusters
        SET grounded_label = NULL,
            grounded_symbol_id = NULL,
            grounding_confidence = NULL,
            grounding_status = 'ungrounded',
            cluster_type = 'stable',
            challenge_failures = $1,
            observations_at_last_challenge = total_observations,
            challenge_count = challenge_count + 1,
            grounding_centroid_snapshot = NULL,
            updated_at = NOW()
        WHERE id = $2
        """,
        failures,
        cluster["id"],
    )

    await _log_rechallenge(pool, cluster["id"], old_label, 0.0,
                           "rechallenge_revoked", old_label, reason)

    log.info("Rechallenge REVOKED cluster #%d: '%s' removed after %d failures (proposed: '%s')",
            cluster["id"], old_label, failures, proposed_label)

    return RechallengeResult(
        cluster_id=cluster["id"],
        previous_label=old_label,
        action="revoked",
        new_label=None,
        reason=f"revoked after {failures} failures (last proposed: {proposed_label})",
        old_confidence=old_confidence,
        new_confidence=0.0,
        drift=drift,
        observations_since_grounding=obs_since,
    )


# ── Helpers ──────────────────────────────────────────────

async def _get_current_centroid(pool, cluster_id, modality=None) -> str | None:
    """Compute and return the current centroid as a string for storage."""
    if not modality:
        modality = await pool.fetchval(
            "SELECT modality FROM sensory_clusters WHERE id = $1", cluster_id
        )

    emb_col = "visual_embedding" if modality == "visual" else "audio_embedding"

    embeddings = await pool.fetch(
        f"""
        SELECT n.{emb_col}::text as emb
        FROM sensory_cluster_members cm
        JOIN sensory_nodes n ON n.id = cm.node_id
        WHERE cm.cluster_id = $1 AND n.{emb_col} IS NOT NULL
        """,
        cluster_id,
    )

    if not embeddings:
        return None

    vecs = []
    for row in embeddings:
        emb_str = row["emb"].strip("[]")
        if emb_str:
            vecs.append([float(x) for x in emb_str.split(",")])

    if not vecs:
        return None

    centroid = np.mean(vecs, axis=0).tolist()
    return str(centroid)


async def _log_rechallenge(pool, cluster_id, label, confidence, grounding_type, previous_label, reason):
    """Record a rechallenge event in the grounding log."""
    await pool.execute(
        """
        INSERT INTO sensory_grounding_log
            (cluster_id, symbol_node_id, symbol_label, confidence,
             grounding_type, previous_label, rechallenge_reason)
        VALUES ($1, 0, $2, $3, $4, $5, $6)
        """,
        cluster_id, label, round(confidence, 4),
        grounding_type, previous_label, reason,
    )


# ── Full rechallenge cycle ───────────────────────────────

async def run_rechallenge_cycle(
    pool: asyncpg.Pool,
    llm_callable,
    max_challenges: int = 5,
) -> list[RechallengeResult]:
    """Run a complete rechallenge cycle.

    1. Decay confidence on all grounded clusters
    2. Find candidates that need rechallenging
    3. Rechallenge each (up to max_challenges)
    4. Return results

    Args:
        pool: Sensory DB connection pool.
        llm_callable: Async function(prompt) → response string.
        max_challenges: Maximum clusters to rechallenge per cycle.

    Returns:
        List of RechallengeResults.
    """
    # 1. Decay
    demoted = await decay_grounding_confidence(pool)
    if demoted:
        log.info("Confidence decay demoted %d clusters", len(demoted))

    # 2. Find candidates
    candidates = await find_rechallenge_candidates(pool)

    # 3. Rechallenge (prioritize: disputed > drift > interval)
    priority_order = {"confidence_decay": 0, "drift": 1, "observation_interval": 2}
    candidates.sort(key=lambda c: min(
        priority_order.get(r, 99) for r in c["reasons"]
    ))

    results = []
    for candidate in candidates[:max_challenges]:
        primary_reason = candidate["reasons"][0]
        result = await rechallenge_cluster(
            pool, candidate["cluster_id"],
            llm_callable, reason=primary_reason,
        )
        results.append(result)

    log.info("Rechallenge cycle complete: %d challenged, outcomes: %s",
            len(results),
            {r.action: sum(1 for x in results if x.action == r.action) for r in results} if results else "none")

    return results
