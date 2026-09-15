"""
Surprise signal: compare predictions to reality.

Surprise = the degree to which observations differ from predictions.
High surprise → stronger memory encoding, deeper decomposition, attention shift.
Low surprise → skip quickly, standard processing.

This is the information-theoretic core of learning: you only learn from
the unexpected. A familiar cup teaches nothing. A cup with two handles
teaches a lot because it violates expectations.

Surprise modulates:
- Attention: foveal dwell time, acuity depth
- Memory: iconic buffer promotion threshold
- Decomposition: triggers deeper analysis of surprising segments
- Drives: feeds uncertainty and coherence pressure to nmem-sym
"""
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class SurpriseSignal:
    """Frame-level surprise computed from prediction verification."""
    score: float                        # 0.0 = fully predicted, 1.0 = completely unexpected
    total_predictions: int              # how many predictions were active
    confirmed: int                      # predictions that matched
    refuted: int                        # predictions that failed
    partial: int                        # partial matches
    novel_objects: int                  # objects with no prediction at all
    strongest_surprise: str | None      # description of most surprising observation
    strongest_score: float = 0.0        # surprise score of most surprising


def compute_frame_surprise(
    verification_results: list,
    novel_count: int = 0,
) -> SurpriseSignal:
    """Compute frame-level surprise from prediction verification results.

    Args:
        verification_results: List of VerificationResult from sensory_prediction.
        novel_count: Number of new objects with no predictions (from tracker).

    Returns:
        SurpriseSignal with overall score and breakdown.
    """
    if not verification_results and novel_count == 0:
        return SurpriseSignal(
            score=0.5,  # no predictions = moderate uncertainty (not surprised, not certain)
            total_predictions=0, confirmed=0, refuted=0, partial=0,
            novel_objects=novel_count, strongest_surprise=None,
        )

    confirmed = sum(1 for r in verification_results if r.status == "confirmed")
    refuted = sum(1 for r in verification_results if r.status == "refuted")
    partial = sum(1 for r in verification_results if r.status == "partial")
    total = len(verification_results)

    # Overall surprise: weighted combination
    # Each refuted prediction contributes full surprise
    # Each novel object contributes moderate surprise (no expectation to violate)
    # Partial matches contribute proportional surprise
    if total + novel_count > 0:
        surprise_sum = (
            refuted * 1.0 +
            partial * 0.5 +
            confirmed * 0.0 +
            novel_count * 0.7  # novel is surprising but less than a violated prediction
        )
        score = surprise_sum / (total + novel_count)
    else:
        score = 0.5

    # Find the most surprising individual result
    strongest = None
    strongest_score = 0.0
    for r in verification_results:
        if r.surprise > strongest_score:
            strongest_score = r.surprise
            strongest = r.prediction.source_description

    return SurpriseSignal(
        score=round(score, 3),
        total_predictions=total,
        confirmed=confirmed,
        refuted=refuted,
        partial=partial,
        novel_objects=novel_count,
        strongest_surprise=strongest,
        strongest_score=round(strongest_score, 3),
    )


async def apply_surprise_effects(
    surprise: SurpriseSignal,
    foveal_state: dict | None = None,
    drive_events: list | None = None,
) -> dict:
    """Apply surprise-modulated effects to attention, memory, and drives.

    This doesn't modify the database directly — it returns adjustment
    parameters that the caller applies to each system.

    Args:
        surprise: The frame's surprise signal.
        foveal_state: Current foveal attention state (if available).
        drive_events: List to append drive events to (if available).

    Returns:
        Dict of adjustments:
        - iconic_threshold_scale: multiplier on ICONIC_PROMOTION_THRESHOLD
        - attention_dwell_scale: multiplier on foveal dwell frames
        - decompose_trigger: whether to trigger deeper decomposition
        - drive_events: list of (drive_name, pressure_delta) tuples
    """
    adjustments = {
        "iconic_threshold_scale": 1.0,
        "attention_dwell_scale": 1.0,
        "decompose_trigger": False,
        "drive_events": [],
    }

    s = surprise.score

    if s > 0.7:
        # HIGH SURPRISE: something unexpected happened
        # Lower promotion threshold (remember surprising things more easily)
        adjustments["iconic_threshold_scale"] = 0.5
        # Increase attention dwell (look longer at the unexpected)
        adjustments["attention_dwell_scale"] = 2.0
        # Trigger decomposition (need finer detail to understand)
        adjustments["decompose_trigger"] = True
        # Drive pressure: uncertainty up, coherence up
        adjustments["drive_events"].extend([
            ("uncertainty", 0.15),
            ("coherence", 0.10),
        ])
        log.info("HIGH SURPRISE (%.2f): %s", s, surprise.strongest_surprise)

    elif s > 0.4:
        # MODERATE SURPRISE: partially unexpected
        adjustments["iconic_threshold_scale"] = 0.8
        adjustments["attention_dwell_scale"] = 1.3
        adjustments["drive_events"].extend([
            ("uncertainty", 0.05),
        ])

    elif s < 0.2:
        # LOW SURPRISE: familiar scene, predictions mostly confirmed
        adjustments["iconic_threshold_scale"] = 1.2  # higher threshold (don't waste memory)
        adjustments["attention_dwell_scale"] = 0.7   # move on quickly
        # Reduce uncertainty pressure (world is predictable)
        adjustments["drive_events"].extend([
            ("uncertainty", -0.05),
        ])

    # Feed drive events if collector provided
    if drive_events is not None:
        drive_events.extend(adjustments["drive_events"])

    return adjustments
