"""
Curriculum system: progressive learning through staged video content.

Mirrors how children learn — start with simple, isolated concepts
(colors, shapes) and progressively introduce complexity (objects,
scenes, actions). The curriculum tracks what the system has learned
and decides when it's ready for the next phase.

Phases:
  1. Primitives: colors, basic shapes (children's flashcard videos)
  2. Objects: simple named objects against clean backgrounds
  3. Composition: objects with parts, spatial relationships
  4. Scenes: multiple objects, natural backgrounds
  5. Actions: temporal sequences, verb concepts
  6. Open: real-world content, no constraints

Advancement is based on grounding coverage, not time elapsed.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import asyncpg

log = logging.getLogger(__name__)


@dataclass
class Phase:
    """A single curriculum phase."""
    id: int
    name: str
    description: str
    # Advancement criteria
    min_grounded_clusters: int         # must have at least this many grounded concepts
    min_grounding_confidence: float    # average confidence must exceed this
    min_confirmed_labels: int          # labels confirmed via rechallenge
    # Content guidance
    video_tags: list[str]              # tags to search for in video sources
    example_content: str               # what kind of videos to use


# The curriculum
PHASES = [
    Phase(
        id=1,
        name="primitives",
        description="Colors, basic shapes, simple sounds",
        min_grounded_clusters=15,
        min_grounding_confidence=0.7,
        min_confirmed_labels=10,
        video_tags=["colors for kids", "shapes for kids", "learn colors",
                    "basic shapes", "color song", "shape song"],
        example_content=(
            "Flashcard-style videos: single colored shape on white background, "
            "narrator says 'this is red', 'this is a circle'. Repetitive, slow."
        ),
    ),
    Phase(
        id=2,
        name="objects",
        description="Common objects: cup, ball, car, dog, cat, tree, house",
        min_grounded_clusters=50,
        min_grounding_confidence=0.65,
        min_confirmed_labels=30,
        video_tags=["first words for babies", "learn objects", "naming things",
                    "vocabulary for toddlers", "picture dictionary kids"],
        example_content=(
            "Object naming videos: single object shown, narrator says its name. "
            "Multiple angles of each object. Clean backgrounds preferred."
        ),
    ),
    Phase(
        id=3,
        name="composition",
        description="Objects with parts, spatial relationships: cup ON table, ball IN box",
        min_grounded_clusters=100,
        min_grounding_confidence=0.6,
        min_confirmed_labels=60,
        video_tags=["prepositions for kids", "parts of objects", "where is it",
                    "on in under", "spatial words"],
        example_content=(
            "Spatial relationship videos: objects placed relative to each other. "
            "Narrator describes 'the cup is on the table', 'the ball is in the box'. "
            "Multiple examples of each relationship."
        ),
    ),
    Phase(
        id=4,
        name="scenes",
        description="Multiple objects in natural settings: kitchen, playground, garden",
        min_grounded_clusters=200,
        min_grounding_confidence=0.55,
        min_confirmed_labels=100,
        video_tags=["rooms of the house", "around the house", "kitchen vocabulary",
                    "playground", "garden vocabulary", "scene description kids"],
        example_content=(
            "Scene tours: camera moves through environments, narrator describes "
            "what's visible. Multiple objects per frame, natural backgrounds."
        ),
    ),
    Phase(
        id=5,
        name="actions",
        description="Temporal sequences, verbs: pouring, stacking, running, eating",
        min_grounded_clusters=300,
        min_grounding_confidence=0.5,
        min_confirmed_labels=150,
        video_tags=["action words for kids", "verbs for children", "what are they doing",
                    "daily routines kids", "cooking for kids"],
        example_content=(
            "Action demonstration videos: person performs actions, narrator describes "
            "with verbs. 'She is pouring the water', 'He is stacking the blocks'."
        ),
    ),
    Phase(
        id=6,
        name="open",
        description="Real-world content — documentaries, vlogs, instructional videos",
        min_grounded_clusters=0,       # no advancement from here
        min_grounding_confidence=0.0,
        min_confirmed_labels=0,
        video_tags=[],                 # anything goes
        example_content=(
            "Any real-world video content. The system has enough foundational "
            "concepts to learn from natural, unstructured content."
        ),
    ),
]


@dataclass
class CurriculumState:
    """Current state of the curriculum progression."""
    current_phase: int = 1
    phase_name: str = "primitives"
    grounded_clusters: int = 0
    avg_confidence: float = 0.0
    confirmed_labels: int = 0
    total_videos_ingested: int = 0
    total_frames_processed: int = 0
    ready_to_advance: bool = False
    advancement_blockers: list[str] = field(default_factory=list)
    phase_history: list[dict] = field(default_factory=list)


# ── State management ─────────────────────────────────────

_STATE_FILE = "curriculum_state.json"


def _get_state_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / _STATE_FILE


def load_state(base_dir: str | Path) -> CurriculumState:
    """Load curriculum state from disk."""
    path = _get_state_path(base_dir)
    if not path.exists():
        return CurriculumState()

    data = json.loads(path.read_text())
    state = CurriculumState()
    for k, v in data.items():
        if hasattr(state, k):
            setattr(state, k, v)
    return state


def save_state(state: CurriculumState, base_dir: str | Path):
    """Save curriculum state to disk."""
    path = _get_state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "current_phase": state.current_phase,
        "phase_name": state.phase_name,
        "total_videos_ingested": state.total_videos_ingested,
        "total_frames_processed": state.total_frames_processed,
        "phase_history": state.phase_history,
    }
    path.write_text(json.dumps(data, indent=2))


# ── Phase evaluation ─────────────────────────────────────

async def evaluate_phase(
    pool: asyncpg.Pool,
    current_phase: int,
) -> CurriculumState:
    """Evaluate whether the system is ready to advance to the next phase.

    Queries the sensory graph for grounding statistics and compares
    against the current phase's advancement criteria.

    Args:
        pool: Sensory DB connection pool.
        current_phase: Current phase number (1-6).

    Returns:
        CurriculumState with current metrics and readiness assessment.
    """
    phase = PHASES[min(current_phase - 1, len(PHASES) - 1)]

    # Count grounded clusters
    grounded = await pool.fetchval(
        "SELECT COUNT(*) FROM sensory_clusters WHERE grounded_label IS NOT NULL"
    )

    # Average confidence of grounded clusters
    avg_conf = await pool.fetchval(
        """
        SELECT COALESCE(AVG(grounding_confidence), 0)
        FROM sensory_clusters
        WHERE grounded_label IS NOT NULL AND grounding_confidence IS NOT NULL
        """
    )

    # Confirmed labels (challenged and confirmed)
    confirmed = await pool.fetchval(
        """
        SELECT COUNT(*) FROM sensory_clusters
        WHERE grounding_status = 'confident'
        """
    )

    state = CurriculumState(
        current_phase=current_phase,
        phase_name=phase.name,
        grounded_clusters=grounded,
        avg_confidence=round(float(avg_conf), 3),
        confirmed_labels=confirmed,
    )

    # Check advancement criteria
    blockers = []
    if current_phase < len(PHASES):
        if grounded < phase.min_grounded_clusters:
            blockers.append(
                f"Need {phase.min_grounded_clusters} grounded clusters, have {grounded}"
            )
        if float(avg_conf) < phase.min_grounding_confidence:
            blockers.append(
                f"Need avg confidence >= {phase.min_grounding_confidence}, have {float(avg_conf):.3f}"
            )
        if confirmed < phase.min_confirmed_labels:
            blockers.append(
                f"Need {phase.min_confirmed_labels} confirmed labels, have {confirmed}"
            )

        state.ready_to_advance = len(blockers) == 0
        state.advancement_blockers = blockers

    return state


async def advance_phase(
    pool: asyncpg.Pool,
    state: CurriculumState,
    base_dir: str | Path,
) -> CurriculumState:
    """Advance to the next curriculum phase if ready.

    Args:
        pool: Sensory DB connection pool.
        state: Current curriculum state.
        base_dir: Directory for state persistence.

    Returns:
        Updated CurriculumState.
    """
    if not state.ready_to_advance:
        log.info("Not ready to advance from phase %d (%s): %s",
                state.current_phase, state.phase_name,
                "; ".join(state.advancement_blockers))
        return state

    if state.current_phase >= len(PHASES):
        log.info("Already at final phase (open)")
        return state

    # Record phase completion
    state.phase_history.append({
        "phase": state.current_phase,
        "name": state.phase_name,
        "completed_at": datetime.now(UTC).isoformat(),
        "grounded_clusters": state.grounded_clusters,
        "avg_confidence": state.avg_confidence,
        "confirmed_labels": state.confirmed_labels,
    })

    # Advance
    state.current_phase += 1
    new_phase = PHASES[state.current_phase - 1]
    state.phase_name = new_phase.name
    state.ready_to_advance = False
    state.advancement_blockers = []

    save_state(state, base_dir)

    log.info("Advanced to phase %d: %s — %s",
            state.current_phase, new_phase.name, new_phase.description)

    return state


def get_current_phase(phase_num: int) -> Phase:
    """Get the Phase definition for a phase number."""
    return PHASES[min(phase_num - 1, len(PHASES) - 1)]


def format_curriculum_status(state: CurriculumState) -> str:
    """Format curriculum status for display."""
    phase = get_current_phase(state.current_phase)
    lines = [
        f"=== Curriculum: Phase {state.current_phase} — {phase.name} ===",
        f"  {phase.description}",
        "",
        f"  Grounded clusters: {state.grounded_clusters} / {phase.min_grounded_clusters}",
        f"  Avg confidence:    {state.avg_confidence:.3f} / {phase.min_grounding_confidence}",
        f"  Confirmed labels:  {state.confirmed_labels} / {phase.min_confirmed_labels}",
        f"  Videos ingested:   {state.total_videos_ingested}",
        f"  Frames processed:  {state.total_frames_processed}",
        "",
    ]

    if state.ready_to_advance:
        lines.append("  STATUS: Ready to advance!")
    elif state.advancement_blockers:
        lines.append("  Blockers:")
        for b in state.advancement_blockers:
            lines.append(f"    - {b}")

    if phase.video_tags:
        lines.append("")
        lines.append("  Suggested video search terms:")
        for tag in phase.video_tags[:4]:
            lines.append(f"    - \"{tag}\"")

    lines.append("")
    lines.append(f"  Content guidance: {phase.example_content}")

    return "\n".join(lines)
