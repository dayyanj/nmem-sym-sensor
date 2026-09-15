"""
Sensory Prediction Engine: the predict → observe → compare → update loop.

Three prediction sources:
1. Structural: composition.py says "eye" → expect "nose" and "mouth" nearby
2. Temporal: object tracker says cup was at (200,300) → predict at (205,300)
3. Symbolic: nmem-sym activates "cup" → predict cylinder + handle cluster

Each prediction is stored with status 'pending'. When a matching observation
arrives, it's 'confirmed' (→ LTP on causal path). When contradicted, it's
'refuted' (→ LTD). When no observation within timeout, it's 'expired'.

This is the core learning mechanism. Without it, the system records but
doesn't learn. With it, confirmed predictions strengthen knowledge and
refuted predictions drive discovery.
"""
import logging
from dataclasses import dataclass

import asyncpg
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SensoryPrediction:
    """A prediction about what should be observed."""
    prediction_type: str                  # 'structural' | 'temporal' | 'symbolic'
    source_description: str               # human-readable: "cup should have handle"
    expected_label: str | None = None     # expected node label
    expected_node_type: str | None = None # expected node type
    expected_embedding: np.ndarray | None = None  # expected embedding
    expected_position: tuple[float, float] | None = None  # expected (x, y)
    source_cluster_id: int | None = None  # sensory cluster that generated this
    source_symbol_id: int | None = None   # nmem-sym symbol that generated this
    confidence: float = 0.5              # prediction confidence
    frame_id: str | None = None


@dataclass
class VerificationResult:
    """Result of comparing a prediction to observations."""
    prediction: SensoryPrediction
    status: str                          # 'confirmed' | 'refuted' | 'partial'
    matched_node_id: int | None = None   # what actually matched
    matched_label: str | None = None
    similarity: float = 0.0             # embedding similarity to prediction
    position_error: float | None = None  # pixels off from predicted position
    surprise: float = 0.0               # 0 = fully expected, 1 = completely unexpected


# ── Prediction generation ────────────────────────────────

async def generate_structural_predictions(
    pool: asyncpg.Pool,
    active_node_ids: list[int],
) -> list[SensoryPrediction]:
    """Generate predictions from structural expectations.

    "I see an eye → I predict a nose and mouth should be nearby."
    Uses the structural_expectations table built by composition.py.
    """
    if not active_node_ids:
        return []

    predictions = []

    for node_id in active_node_ids:
        # What wholes could this part belong to?
        from nmem_sym_sensor.composition import infer_whole_from_part
        inferences = await infer_whole_from_part(pool, node_id, active_node_ids)

        for inf in inferences:
            if inf["confidence"] < 0.3:
                continue

            # Predict missing siblings
            for sibling in inf.get("siblings_missing", []):
                predictions.append(SensoryPrediction(
                    prediction_type="structural",
                    source_description=f"'{sibling}' expected as sibling of node {node_id} in '{inf['whole']}'",
                    expected_label=sibling,
                    expected_node_type="shape",
                    confidence=inf["confidence"] * 0.7,
                    source_cluster_id=None,
                ))

    return predictions


def generate_temporal_predictions(
    tracking_predictions: list[dict],
) -> list[SensoryPrediction]:
    """Convert temporal tracker predictions to SensoryPredictions.

    "The cup was at (200,300) moving right → predict at (205,300)."
    """
    predictions = []
    for tp in tracking_predictions:
        predictions.append(SensoryPrediction(
            prediction_type="temporal",
            source_description=f"'{tp['label']}' predicted at {tp['predicted_position']}",
            expected_label=tp["label"],
            expected_position=tp["predicted_position"],
            confidence=tp["confidence"],
        ))
    return predictions


async def generate_symbolic_predictions(
    pool: asyncpg.Pool,
    sym_pool: asyncpg.Pool | None,
    active_node_ids: list[int],
) -> list[SensoryPrediction]:
    """Generate predictions from nmem-sym symbolic knowledge.

    "nmem-sym activates 'cup' → predict cylinder + handle visual cluster."
    Requires a connection to the nmem-sym database.
    """
    if sym_pool is None or not active_node_ids:
        return []

    predictions = []

    # Find grounded clusters for active nodes
    grounded = await pool.fetch(
        """
        SELECT c.grounded_label, c.grounded_symbol_id, c.id as cluster_id
        FROM sensory_cluster_members cm
        JOIN sensory_clusters c ON c.id = cm.cluster_id
        WHERE cm.node_id = ANY($1)
          AND c.grounded_symbol_id IS NOT NULL
          AND c.grounding_status IN ('confident', 'speculative')
        """,
        active_node_ids,
    )

    for g in grounded:
        symbol_id = g["grounded_symbol_id"]

        # Query nmem-sym for related concepts via causal edges
        try:
            related = await sym_pool.fetch(
                """
                SELECT t.label, t.node_type, e.edge_type
                FROM symbol_edges e
                JOIN symbol_nodes t ON t.id = e.target_id
                WHERE e.source_id = $1
                  AND e.edge_type IN ('causes', 'triggers', 'part_of', 'co_occurs_with')
                  AND NOT t.archived
                ORDER BY e.weight DESC
                LIMIT 5
                """,
                symbol_id,
            )
        except Exception:
            continue

        for rel in related:
            predictions.append(SensoryPrediction(
                prediction_type="symbolic",
                source_description=f"'{g['grounded_label']}' {rel['edge_type']} '{rel['label']}'",
                expected_label=rel["label"],
                source_symbol_id=symbol_id,
                source_cluster_id=g["cluster_id"],
                confidence=0.5,
            ))

    return predictions


async def generate_all_predictions(
    pool: asyncpg.Pool,
    active_node_ids: list[int],
    tracking_predictions: list[dict] | None = None,
    sym_pool: asyncpg.Pool | None = None,
) -> list[SensoryPrediction]:
    """Generate predictions from all three sources."""
    all_preds = []

    # Structural (from composition expectations)
    structural = await generate_structural_predictions(pool, active_node_ids)
    all_preds.extend(structural)

    # Temporal (from object tracker)
    if tracking_predictions:
        temporal = generate_temporal_predictions(tracking_predictions)
        all_preds.extend(temporal)

    # Symbolic (from nmem-sym knowledge)
    if sym_pool:
        symbolic = await generate_symbolic_predictions(pool, sym_pool, active_node_ids)
        all_preds.extend(symbolic)

    return all_preds


# ── Prediction verification ──────────────────────────────

async def verify_predictions(
    pool: asyncpg.Pool,
    predictions: list[SensoryPrediction],
    observed_nodes: list[dict],
) -> list[VerificationResult]:
    """Compare predictions against actual observations.

    For each prediction, find the best matching observation.
    Compute surprise as 1 - match_quality.

    Args:
        predictions: What we expected to see.
        observed_nodes: What we actually saw. Each dict has:
            label, node_type, embedding, position, node_id.

    Returns:
        List of VerificationResults.
    """
    results = []

    for pred in predictions:
        best_match = None
        best_score = 0.0

        for obs in observed_nodes:
            score = 0.0
            match_count = 0

            # Label match
            if pred.expected_label and obs.get("label"):
                if pred.expected_label.lower() in obs["label"].lower():
                    score += 0.4
                match_count += 1

            # Embedding similarity
            if pred.expected_embedding is not None and obs.get("embedding") is not None:
                emb_sim = float(np.dot(
                    np.array(pred.expected_embedding),
                    np.array(obs["embedding"]),
                ))
                score += 0.4 * max(0, emb_sim)
                match_count += 1

            # Position proximity (for temporal predictions)
            if pred.expected_position and obs.get("position"):
                px, py = pred.expected_position
                ox, oy = obs["position"]
                dist = np.sqrt((px - ox)**2 + (py - oy)**2)
                pos_score = max(0, 1.0 - dist / 200.0)  # within 200px = partial match
                score += 0.2 * pos_score
                match_count += 1

            if match_count > 0:
                score = score / (match_count * 0.4)  # normalise

            if score > best_score:
                best_score = score
                best_match = obs

        # Determine status
        if best_score >= 0.6:
            status = "confirmed"
        elif best_score >= 0.3:
            status = "partial"
        else:
            status = "refuted"

        surprise = 1.0 - best_score

        pos_error = None
        if pred.expected_position and best_match and best_match.get("position"):
            px, py = pred.expected_position
            ox, oy = best_match["position"]
            pos_error = float(np.sqrt((px - ox)**2 + (py - oy)**2))

        results.append(VerificationResult(
            prediction=pred,
            status=status,
            matched_node_id=best_match.get("node_id") if best_match else None,
            matched_label=best_match.get("label") if best_match else None,
            similarity=best_score,
            position_error=pos_error,
            surprise=surprise,
        ))

    return results


# ── Prediction feedback (LTP/LTD) ────────────────────────

async def apply_prediction_feedback(
    pool: asyncpg.Pool,
    results: list[VerificationResult],
    sym_pool: asyncpg.Pool | None = None,
) -> dict:
    """Apply learning feedback from prediction verification.

    Confirmed predictions → strengthen paths (LTP).
    Refuted predictions → weaken paths (LTD).
    This is how the system learns — correct predictions reinforce
    the knowledge that generated them, wrong predictions weaken it.

    Returns stats dict.
    """
    stats = {"confirmed": 0, "refuted": 0, "partial": 0, "ltp_applied": 0, "ltd_applied": 0}

    for result in results:
        stats[result.status] = stats.get(result.status, 0) + 1
        pred = result.prediction

        # Store prediction outcome
        try:
            await pool.execute(
                """
                INSERT INTO sensory_predictions
                    (prediction_type, source_description, expected_node_type,
                     expected_label, expected_position_x, expected_position_y,
                     source_symbol_id, source_cluster_id, status,
                     matched_node_id, surprise_score, resolved_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                """,
                pred.prediction_type,
                pred.source_description,
                pred.expected_node_type,
                pred.expected_label,
                pred.expected_position[0] if pred.expected_position else None,
                pred.expected_position[1] if pred.expected_position else None,
                pred.source_symbol_id,
                pred.source_cluster_id,
                result.status,
                result.matched_node_id,
                result.surprise,
            )
        except Exception as e:
            log.debug("Failed to store prediction: %s", e)

        # Apply LTP/LTD to structural expectations
        if pred.prediction_type == "structural" and pred.source_cluster_id:
            if result.status == "confirmed":
                # Strengthen the structural expectation
                await pool.execute(
                    """
                    UPDATE structural_expectations
                    SET confidence = LEAST(0.99, confidence + 0.03),
                        observation_count = observation_count + 1,
                        updated_at = NOW()
                    WHERE part_label = $1
                    """,
                    pred.expected_label,
                )
                stats["ltp_applied"] += 1

            elif result.status == "refuted":
                # Weaken the structural expectation
                await pool.execute(
                    """
                    UPDATE structural_expectations
                    SET confidence = GREATEST(0.05, confidence - 0.02),
                        updated_at = NOW()
                    WHERE part_label = $1
                    """,
                    pred.expected_label,
                )
                stats["ltd_applied"] += 1

        # Apply LTP/LTD to nmem-sym edges (if symbolic prediction)
        if pred.prediction_type == "symbolic" and sym_pool and pred.source_symbol_id:
            if result.status == "confirmed":
                try:
                    await sym_pool.execute(
                        """
                        UPDATE symbol_edges
                        SET ltp_score = LEAST(1.0, ltp_score + 0.05),
                            weight = LEAST(1.0, weight + 0.02),
                            myelinated = CASE WHEN ltp_score + 0.05 >= 0.7 THEN TRUE ELSE myelinated END
                        WHERE source_id = $1
                        """,
                        pred.source_symbol_id,
                    )
                    stats["ltp_applied"] += 1
                except Exception:
                    pass

            elif result.status == "refuted":
                try:
                    await sym_pool.execute(
                        """
                        UPDATE symbol_edges
                        SET ltd_score = LEAST(1.0, ltd_score + 0.05),
                            weight = GREATEST(0.1, weight - 0.02)
                        WHERE source_id = $1
                        """,
                        pred.source_symbol_id,
                    )
                    stats["ltd_applied"] += 1
                except Exception:
                    pass

    if stats["confirmed"] + stats["refuted"] > 0:
        log.info("Prediction feedback: %d confirmed, %d refuted, %d partial, "
                "%d LTP, %d LTD",
                stats["confirmed"], stats["refuted"], stats["partial"],
                stats["ltp_applied"], stats["ltd_applied"])

    return stats
