"""
Bidirectional Symbol-Sensory Bridge.

Direction 1 — Symbol → Sensory (top-down predictions):
  When nmem-sym activates a concept, generate sensory expectations.
  "cup" activates → predict cylinder + handle visual cluster.

Direction 2 — Sensory → Symbol (bottom-up events):
  Novel observations, surprise signals, prediction outcomes feed
  back to nmem-sym as events that modify drive pressures and
  trigger LTP/LTD on symbolic edges.

This replaces the one-way grounding in bridge.py with a full
cognitive loop where symbolic knowledge guides perception and
perception updates symbolic knowledge.
"""
import logging
from dataclasses import dataclass
from typing import Any

import asyncpg

log = logging.getLogger(__name__)


@dataclass
class SymbolEvent:
    """An event from the sensory system to feed to nmem-sym."""
    event_type: str     # 'prediction.confirmed' | 'prediction.refuted' | 'anomaly.detected' | etc.
    data: dict          # event-specific payload
    pressure: dict      # drive_name → pressure_delta


# ── Symbol → Sensory (top-down) ──────────────────────────

async def symbol_to_sensory_expectations(
    pool: asyncpg.Pool,
    sym_pool: asyncpg.Pool,
    activated_symbol_ids: list[int],
) -> list[dict]:
    """Generate sensory expectations from activated symbolic concepts.

    When nmem-sym activates concepts (via query or dreamstate), this
    function returns what the sensory system should look for.

    Args:
        pool: Sensory DB pool.
        sym_pool: nmem-sym DB pool.
        activated_symbol_ids: Symbol node IDs that were activated.

    Returns:
        List of sensory expectations (label, modality, cluster_id, etc.)
    """
    expectations = []

    for sym_id in activated_symbol_ids:
        # Find sensory clusters grounded to this symbol
        clusters = await pool.fetch(
            """
            SELECT id, modality, member_count, grounded_label, grounding_confidence
            FROM sensory_clusters
            WHERE grounded_symbol_id = $1
              AND grounding_status IN ('confident', 'speculative')
            ORDER BY grounding_confidence DESC
            """,
            sym_id,
        )

        for cluster in clusters:
            # Get cluster member descriptions
            members = await pool.fetch(
                """
                SELECT n.label, n.node_type, n.features
                FROM sensory_cluster_members cm
                JOIN sensory_nodes n ON n.id = cm.node_id
                WHERE cm.cluster_id = $1
                ORDER BY n.groundedness DESC LIMIT 5
                """,
                cluster["id"],
            )

            expectations.append({
                "symbol_id": sym_id,
                "cluster_id": cluster["id"],
                "label": cluster["grounded_label"],
                "modality": cluster["modality"],
                "confidence": float(cluster["grounding_confidence"] or 0),
                "members": [
                    {"label": m["label"], "type": m["node_type"]}
                    for m in members
                ],
            })

    return expectations


# ── Sensory → Symbol (bottom-up) ─────────────────────────

def create_sensory_events(
    surprise_score: float,
    prediction_stats: dict,
    novel_count: int = 0,
    grounding_changes: list[dict] | None = None,
) -> list[SymbolEvent]:
    """Convert sensory observations into events for nmem-sym's drive system.

    Args:
        surprise_score: Frame surprise (0-1).
        prediction_stats: Dict with confirmed/refuted/partial counts.
        novel_count: Number of novel (untracked) objects.
        grounding_changes: List of grounding events (new, revised, revoked).

    Returns:
        List of SymbolEvents to feed to nmem-sym.
    """
    events = []

    # Prediction outcomes
    confirmed = prediction_stats.get("confirmed", 0)
    refuted = prediction_stats.get("refuted", 0)

    if confirmed > 0:
        events.append(SymbolEvent(
            event_type="prediction.confirmed",
            data={"count": confirmed, "source": "sensory"},
            pressure={
                "uncertainty": -0.03 * confirmed,
                "coherence": -0.02 * confirmed,
            },
        ))

    if refuted > 0:
        events.append(SymbolEvent(
            event_type="prediction.refuted",
            data={"count": refuted, "source": "sensory"},
            pressure={
                "uncertainty": 0.05 * refuted,
                "coherence": 0.03 * refuted,
            },
        ))

    # High surprise → anomaly event
    if surprise_score > 0.7:
        events.append(SymbolEvent(
            event_type="anomaly.detected",
            data={"surprise": surprise_score, "source": "sensory"},
            pressure={
                "uncertainty": 0.15,
                "novelty": -0.05,  # surprise satisfies novelty drive
            },
        ))

    # Novel objects → novelty events
    if novel_count > 0:
        events.append(SymbolEvent(
            event_type="sensory.novel_objects",
            data={"count": novel_count},
            pressure={
                "novelty": -0.02 * novel_count,  # satisfies novelty
                "integration": 0.03 * novel_count,  # needs grounding
            },
        ))

    # Grounding changes
    if grounding_changes:
        for change in grounding_changes:
            events.append(SymbolEvent(
                event_type=f"grounding.{change.get('action', 'update')}",
                data=change,
                pressure={
                    "integration": -0.05,  # grounding satisfies integration
                },
            ))

    return events


# ── Hypothesis Bridge (recombination ↔ nmem-sym) ─────────


async def emit_hypothesis_candidate(
    sym_pool: asyncpg.Pool | None,
    bridge: Any | None,
    candidate: dict,
) -> bool:
    """Pass a novel activation path from recombination to nmem-sym.

    The sensory system discovers novel paths during dreamstate
    recombination. It does NOT store or manage hypotheses — that's
    nmem-sym's job. This function hands off the discovery.

    Args:
        sym_pool: nmem-sym DB pool (for direct insert fallback).
        bridge: nmem-sym SymbolBridge instance (preferred path).
        candidate: Dict with keys:
            source, recombination_id, unit_a, unit_b,
            modality_a, modality_b, plausibility, context.

    Returns:
        True if successfully handed off, False otherwise.
    """
    # Preferred: use bridge's event system
    if bridge is not None:
        try:
            bridge.feed_event({
                "type": "hypothesis.candidate",
                "source": "dreamstate_recombination",
                **candidate,
            })
            log.info(
                "Hypothesis candidate emitted via bridge: %s↔%s (plausibility=%.3f)",
                candidate.get("unit_a"), candidate.get("unit_b"),
                candidate.get("plausibility", 0),
            )
            return True
        except Exception as e:
            log.debug("Bridge hypothesis emit failed: %s", e)

    # Fallback: store as a drive event that nmem-sym picks up
    if sym_pool is not None:
        try:
            await sym_pool.execute(
                """
                INSERT INTO symbol_drive_events
                    (drive_name, pressure_delta, source, created_at)
                VALUES ('integration', $1, $2, NOW())
                ON CONFLICT DO NOTHING
                """,
                candidate.get("plausibility", 0.5) * 0.1,
                "hypothesis.candidate",
            )
            return True
        except Exception:
            pass

    return False


async def emit_confirmed_hypothesis(
    pool: asyncpg.Pool,
    hypothesis: dict,
) -> None:
    """Callback from nmem-sym when a hypothesis is confirmed.

    Creates a real co-occurrence in the sensory system based on the
    confirmed prediction. Dream-confirmed = high quality encoding.

    Args:
        pool: Sensory DB pool.
        hypothesis: Dict with keys:
            unit_a, modality_a, unit_b, modality_b, source.
    """
    from nmem_sym_sensor.cooccurrence import cooc_store

    await cooc_store.observe(
        hypothesis["unit_a"], hypothesis["modality_a"],
        hypothesis["unit_b"], hypothesis["modality_b"],
        pool,
        attention=0.8,    # dream-confirmed = high quality encoding
        surprise=0.3,     # expected (we predicted it)
        proximity=0.5,    # moderate (inferred, not directly co-observed)
    )
    log.info(
        "Confirmed hypothesis -> co-occurrence: %s:%s <-> %s:%s",
        hypothesis["modality_a"], hypothesis["unit_a"],
        hypothesis["modality_b"], hypothesis["unit_b"],
    )


async def feed_events_to_drives(
    sym_pool: asyncpg.Pool | None,
    events: list[SymbolEvent],
    bridge: Any | None = None,
) -> int:
    """Feed sensory events to nmem-sym's drive accumulator.

    If a SymbolBridge instance is available, uses its event system.
    Otherwise, feeds events directly to the drive accumulator.

    Args:
        sym_pool: nmem-sym DB pool (for direct drive table updates).
        events: List of SymbolEvents.
        bridge: nmem-sym SymbolBridge instance (optional).

    Returns:
        Number of events fed.
    """
    fed = 0

    for event in events:
        # Try bridge first (preferred — goes through full event system)
        if bridge is not None:
            try:
                bridge.feed_event({
                    "type": event.event_type,
                    **event.data,
                })
                fed += 1
                continue
            except Exception as e:
                log.debug("Bridge feed failed: %s", e)

        # Direct pressure injection if no bridge
        if sym_pool is not None:
            for drive_name, delta in event.pressure.items():
                try:
                    # Store as a pressure event that the drive system picks up
                    await sym_pool.execute(
                        """
                        INSERT INTO symbol_drive_events (drive_name, pressure_delta, source, created_at)
                        VALUES ($1, $2, $3, NOW())
                        ON CONFLICT DO NOTHING
                        """,
                        drive_name, delta, event.event_type,
                    )
                    fed += 1
                except Exception:
                    # Table might not exist if drives aren't enabled
                    pass

    if fed > 0:
        log.debug("Fed %d sensory events to drives", fed)

    return fed
