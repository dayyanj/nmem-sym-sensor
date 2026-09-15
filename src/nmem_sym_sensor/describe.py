"""
Compositional description: sensory graph → natural language.

Translates sensory graph structure into structured natural language
descriptions that an LLM can interpret — without requiring grounded
names. The system doesn't need to know "cup" to describe what it sees:

    "A cylindrical brown smooth shape containing a dark oval liquid
     surface, with a rectangular brown protrusion attached to its
     right side. Associated sound: warm-timbred clink (12 co-occurrences)."

Three output modes:
  1. Structural: graph-traversal notation for precise LLM context injection
  2. Narrative: flowing natural language description
  3. Query: "what is this?" prompt for LLM naming proposals

The LLM's existing language knowledge bridges from description to concept.
The sensory system sees structure; the LLM maps it to meaning.
"""
import json
import logging
from dataclasses import dataclass, field

import asyncpg

log = logging.getLogger(__name__)


def _safe_features(features) -> dict:
    """Ensure features is a dict, parsing JSON string if needed."""
    if isinstance(features, dict):
        return features
    if isinstance(features, str):
        try:
            return json.loads(features)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


# ── Data structures ──────────────────────────────────────

@dataclass
class SensoryDescription:
    """A complete natural language description of a sensory observation."""
    # Identity
    cluster_id: int | None = None       # if describing a cluster
    node_id: int | None = None          # if describing a single node
    grounded_label: str | None = None   # if already grounded

    # Description components
    structural: str = ""                # graph-traversal notation
    narrative: str = ""                 # natural language description
    query: str = ""                     # "what is this?" prompt for LLM

    # Metadata
    confidence: float = 0.0            # how rich/certain the description is
    observation_count: int = 0
    modalities: list[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0


@dataclass
class _NodeInfo:
    """Internal: enriched node for description building."""
    id: int
    label: str
    node_type: str
    modality: str
    features: dict
    groundedness: int
    role: str = "member"               # member | prototype


@dataclass
class _EdgeInfo:
    """Internal: enriched edge for description building."""
    source_id: int
    target_id: int
    edge_type: str
    weight: float
    groundedness: int


# ── Natural language fragments ───────────────────────────

# How to phrase each node type in natural language
_NODE_TYPE_PHRASES = {
    "shape": "shape",
    "color": "coloring",
    "texture": "surface",
    "spatial_rel": "spatial arrangement",
    "frequency": "frequency component",
    "rhythm": "rhythmic pattern",
    "timbre": "tonal quality",
    "audio_event": "sound event",
}

# How to phrase each edge type as a relationship
_EDGE_PHRASES = {
    # Spatial
    "adjacent_to": "next to",
    "contains": "containing",
    "contained_by": "inside",
    "above": "above",
    "below": "below",
    "left_of": "to the left of",
    "right_of": "to the right of",
    "overlaps": "overlapping with",
    # Compositional
    "part_of": "attached to",
    "co_occurs_with": "appearing alongside",
    "member_of": "part of the same group as",
    # Temporal
    "simultaneous": "occurring at the same time as",
    "follows": "following after",
    "precedes": "occurring before",
    # Cross-modal
    "bound_to": "associated with",
}

# How to describe feature values in natural language
def _describe_shape_features(features: dict) -> str:
    """Generate a natural language fragment from shape features."""
    parts = []
    shape = features.get("shape", "")
    if shape:
        parts.append(shape)

    area = features.get("area_fraction", 0)
    if area > 0.3:
        parts.append("large")
    elif area < 0.05:
        parts.append("small")

    aspect = features.get("aspect_ratio", 1.0)
    if aspect > 2.0:
        parts.append("elongated")
    elif aspect > 1.5:
        parts.append("oblong")

    circ = features.get("circularity", 0)
    if circ > 0.9:
        parts.append("perfectly round")

    return " ".join(parts) if parts else "shape"


def _describe_color_features(features: dict) -> str:
    """Generate a natural language fragment from color features."""
    s = features.get("saturation", 0)
    v = features.get("value", 0)

    modifiers = []
    if s < 0.2:
        modifiers.append("muted")
    elif s > 0.8:
        modifiers.append("vivid")

    if v < 0.3:
        modifiers.append("dark")
    elif v > 0.8:
        modifiers.append("bright")

    return " ".join(modifiers) if modifiers else ""


def _describe_texture_features(features: dict) -> str:
    """Generate a natural language fragment from texture features."""
    variance = features.get("variance", 0)
    edge_density = features.get("edge_density", 0)

    if variance < 100 and edge_density < 0.05:
        return "smooth"
    elif edge_density > 0.4:
        return "highly textured"
    elif edge_density > 0.2:
        return "moderately textured"
    elif variance > 2000:
        return "noisy"
    else:
        return "matte"


def _describe_timbre_features(features: dict) -> str:
    """Generate a natural language fragment from timbre features."""
    brightness = features.get("brightness", "neutral")
    centroid = features.get("spectral_centroid", 0)

    parts = [brightness]
    if centroid > 6000:
        parts.append("high-pitched")
    elif centroid < 300:
        parts.append("deep")

    return " ".join(parts)


def _describe_rhythm_features(features: dict) -> str:
    """Generate a natural language fragment from rhythm features."""
    tempo = features.get("tempo_class", "moderate")
    regularity = features.get("regularity", 0)

    parts = [tempo]
    if regularity > 0.8:
        parts.append("steady")
    elif regularity < 0.3:
        parts.append("irregular")

    count = features.get("onset_count", 0)
    if count == 1:
        parts.append("single")
    elif count > 20:
        parts.append("rapid")

    return " ".join(parts)


_FEATURE_DESCRIBERS = {
    "shape": _describe_shape_features,
    "color": _describe_color_features,
    "texture": _describe_texture_features,
    "timbre": _describe_timbre_features,
    "rhythm": _describe_rhythm_features,
}


def _describe_node(node: _NodeInfo) -> str:
    """Produce a single natural language phrase for one node."""
    describer = _FEATURE_DESCRIBERS.get(node.node_type)
    feature_desc = describer(node.features) if describer else ""

    # Combine: "elongated oval" (shape), "vivid red" (color), "smooth" (texture)
    if feature_desc:
        return f"{feature_desc} {node.label}".strip()
    return node.label


# ── Graph traversal ──────────────────────────────────────

async def _fetch_cluster_graph(
    pool: asyncpg.Pool,
    cluster_id: int,
) -> tuple[list[_NodeInfo], list[_EdgeInfo]]:
    """Fetch all nodes and edges for a cluster, including cross-modal bindings."""

    # Get member nodes
    member_rows = await pool.fetch(
        """
        SELECT n.id, n.label, n.node_type, n.modality, n.features,
               n.groundedness, cm.role
        FROM sensory_cluster_members cm
        JOIN sensory_nodes n ON n.id = cm.node_id
        WHERE cm.cluster_id = $1
          AND NOT n.archived
        ORDER BY cm.role DESC, n.groundedness DESC
        """,
        cluster_id,
    )

    nodes = [
        _NodeInfo(
            id=r["id"], label=r["label"], node_type=r["node_type"],
            modality=r["modality"], features=_safe_features(r["features"]),
            groundedness=r["groundedness"], role=r["role"],
        )
        for r in member_rows
    ]

    if not nodes:
        return [], []

    node_ids = [n.id for n in nodes]

    # Get edges between members
    edge_rows = await pool.fetch(
        """
        SELECT source_id, target_id, edge_type, weight, groundedness
        FROM sensory_edges
        WHERE source_id = ANY($1) AND target_id = ANY($1)
        ORDER BY weight DESC
        """,
        node_ids,
    )

    edges = [
        _EdgeInfo(
            source_id=r["source_id"], target_id=r["target_id"],
            edge_type=r["edge_type"], weight=r["weight"],
            groundedness=r["groundedness"],
        )
        for r in edge_rows
    ]

    # Also fetch cross-modal bindings (edges to nodes outside the cluster)
    bound_rows = await pool.fetch(
        """
        SELECT e.source_id, e.target_id, e.edge_type, e.weight, e.groundedness,
               n.id as bound_id, n.label as bound_label, n.node_type as bound_type,
               n.modality as bound_modality, n.features as bound_features,
               n.groundedness as bound_groundedness
        FROM sensory_edges e
        JOIN sensory_nodes n ON (
            CASE WHEN e.source_id = ANY($1) THEN e.target_id ELSE e.source_id END
        ) = n.id
        WHERE (e.source_id = ANY($1) OR e.target_id = ANY($1))
          AND NOT (e.source_id = ANY($1) AND e.target_id = ANY($1))
          AND e.edge_type = 'bound_to'
          AND NOT n.archived
        ORDER BY e.weight DESC
        LIMIT 10
        """,
        node_ids,
    )

    for r in bound_rows:
        # Add the bound node
        nodes.append(_NodeInfo(
            id=r["bound_id"], label=r["bound_label"],
            node_type=r["bound_type"], modality=r["bound_modality"],
            features=_safe_features(r["bound_features"]),
            groundedness=r["bound_groundedness"], role="bound",
        ))
        edges.append(_EdgeInfo(
            source_id=r["source_id"], target_id=r["target_id"],
            edge_type=r["edge_type"], weight=r["weight"],
            groundedness=r["groundedness"],
        ))

    return nodes, edges


async def _fetch_node_neighborhood(
    pool: asyncpg.Pool,
    node_id: int,
    max_hops: int = 2,
    max_nodes: int = 20,
) -> tuple[list[_NodeInfo], list[_EdgeInfo]]:
    """Fetch a node and its local neighborhood for description."""

    # Seed node
    seed = await pool.fetchrow(
        """
        SELECT id, label, node_type, modality, features, groundedness
        FROM sensory_nodes WHERE id = $1 AND NOT archived
        """,
        node_id,
    )
    if not seed:
        return [], []

    visited = {seed["id"]}
    nodes = [_NodeInfo(
        id=seed["id"], label=seed["label"], node_type=seed["node_type"],
        modality=seed["modality"], features=seed["features"] or {},
        groundedness=seed["groundedness"], role="focus",
    )]
    all_edges = []
    frontier = {seed["id"]}

    for hop in range(max_hops):
        if not frontier or len(nodes) >= max_nodes:
            break

        neighbor_rows = await pool.fetch(
            """
            SELECT e.source_id, e.target_id, e.edge_type, e.weight, e.groundedness,
                   n.id as n_id, n.label, n.node_type, n.modality, n.features, n.groundedness as n_ground
            FROM sensory_edges e
            JOIN sensory_nodes n ON n.id = (
                CASE WHEN e.source_id = ANY($1) THEN e.target_id ELSE e.source_id END
            )
            WHERE (e.source_id = ANY($1) OR e.target_id = ANY($1))
              AND NOT n.archived
            ORDER BY e.weight DESC
            LIMIT $2
            """,
            list(frontier),
            max_nodes - len(nodes),
        )

        next_frontier = set()
        for r in neighbor_rows:
            all_edges.append(_EdgeInfo(
                source_id=r["source_id"], target_id=r["target_id"],
                edge_type=r["edge_type"], weight=r["weight"],
                groundedness=r["groundedness"],
            ))
            if r["n_id"] not in visited:
                visited.add(r["n_id"])
                nodes.append(_NodeInfo(
                    id=r["n_id"], label=r["label"], node_type=r["node_type"],
                    modality=r["modality"], features=_safe_features(r["features"]),
                    groundedness=r["n_ground"],
                ))
                next_frontier.add(r["n_id"])

        frontier = next_frontier

    return nodes, all_edges


# ── Description builders ─────────────────────────────────

def _build_structural(
    nodes: list[_NodeInfo],
    edges: list[_EdgeInfo],
    grounded_label: str | None = None,
) -> str:
    """Build a structural (graph-notation) description.

    Output format designed for precise LLM context injection:
        [SENSORY OBSERVATION]
        Grounded: "cup" (or Ungrounded)
        Visual components:
          (prototype) shape: elongated oval-brown-smooth [47 obs]
            ├─ contains: oval-black-smooth (color: dark liquid surface)
            ├─ part_of: rectangle-brown-smooth (shape: handle)
            └─ adjacent_to: circle-white-smooth (shape: saucer)
        Audio components:
          bound_to: warm-timbre (tonal quality: warm, 12 obs)
        Spatial layout:
          handle is right_of body
          body is above saucer
    """
    if not nodes:
        return "[SENSORY OBSERVATION: empty]"

    node_map = {n.id: n for n in nodes}
    lines = ["[SENSORY OBSERVATION]"]

    if grounded_label:
        lines.append(f"Grounded: \"{grounded_label}\"")
    else:
        lines.append("Ungrounded (no known label)")

    # Separate by modality
    visual_nodes = [n for n in nodes if n.modality == "visual"]
    audio_nodes = [n for n in nodes if n.modality == "audio"]

    # Find the prototype or focus node
    focus = next((n for n in nodes if n.role in ("prototype", "focus")), nodes[0])

    # Visual components
    if visual_nodes:
        lines.append("Visual components:")

        # Prototype / focus first
        focus_desc = _describe_node(focus)
        obs_str = f"[{focus.groundedness} obs]" if focus.groundedness > 1 else ""
        role_str = f"({focus.role}) " if focus.role in ("prototype", "focus") else ""
        lines.append(f"  {role_str}{focus.node_type}: {focus_desc} {obs_str}".rstrip())

        # Edges from focus → other visual nodes
        focus_edges = [e for e in edges if e.source_id == focus.id or e.target_id == focus.id]
        spatial_edges = []

        for edge in sorted(focus_edges, key=lambda e: e.weight, reverse=True):
            other_id = edge.target_id if edge.source_id == focus.id else edge.source_id
            other = node_map.get(other_id)
            if not other or other.modality != "visual":
                continue

            if edge.edge_type in ("above", "below", "left_of", "right_of",
                                   "adjacent_to", "overlaps"):
                spatial_edges.append(edge)
                continue

            phrase = _EDGE_PHRASES.get(edge.edge_type, edge.edge_type)
            other_desc = _describe_node(other)
            obs_str = f" [{other.groundedness} obs]" if other.groundedness > 1 else ""
            lines.append(f"    ├─ {phrase}: {other_desc}{obs_str}")

        # Remaining visual nodes not connected to focus
        connected_ids = {focus.id} | {
            e.target_id if e.source_id == focus.id else e.source_id
            for e in focus_edges
        }
        for node in visual_nodes:
            if node.id not in connected_ids and node.id != focus.id:
                desc = _describe_node(node)
                lines.append(f"    └─ {node.node_type}: {desc}")

        # Spatial layout
        all_spatial = [e for e in edges if e.edge_type in (
            "above", "below", "left_of", "right_of",
            "adjacent_to", "overlaps", "contains", "contained_by",
        )]
        if all_spatial:
            lines.append("Spatial layout:")
            for edge in all_spatial:
                src = node_map.get(edge.source_id)
                tgt = node_map.get(edge.target_id)
                if src and tgt:
                    phrase = _EDGE_PHRASES.get(edge.edge_type, edge.edge_type)
                    lines.append(f"  {src.label} {phrase} {tgt.label}")

    # Audio components
    if audio_nodes:
        lines.append("Audio components:")
        for node in audio_nodes:
            desc = _describe_node(node)
            bound = node.role == "bound"
            prefix = "bound_to: " if bound else f"{node.node_type}: "
            obs_str = f" [{node.groundedness} obs]" if node.groundedness > 1 else ""
            lines.append(f"  {prefix}{desc}{obs_str}")

        # Audio temporal relations
        audio_edges = [e for e in edges
                      if e.edge_type in ("simultaneous", "follows", "precedes")
                      and node_map.get(e.source_id, _NodeInfo(0, "", "", "visual", {}, 0)).modality == "audio"]
        for edge in audio_edges:
            src = node_map.get(edge.source_id)
            tgt = node_map.get(edge.target_id)
            if src and tgt:
                phrase = _EDGE_PHRASES.get(edge.edge_type, edge.edge_type)
                lines.append(f"  {src.label} {phrase} {tgt.label}")

    return "\n".join(lines)


def _build_narrative(
    nodes: list[_NodeInfo],
    edges: list[_EdgeInfo],
    grounded_label: str | None = None,
) -> str:
    """Build a flowing natural language narrative description.

    Reads like a person describing what they see:
        "I observe a large, elongated brown shape with a smooth surface.
         It contains a dark oval area that appears to be liquid. A smaller
         rectangular protrusion extends from its right side. The object
         rests on a white circular surface below it. When interacted with,
         it produces a warm-toned clinking sound."
    """
    if not nodes:
        return "No sensory data available."

    node_map = {n.id: n for n in nodes}
    sentences = []

    # Opening: grounded or not
    if grounded_label:
        sentences.append(f"This has been identified as \"{grounded_label}\".")
    else:
        sentences.append("This has not been identified yet.")


    # Describe the primary visual element
    visual_nodes = [n for n in nodes if n.modality == "visual"]
    audio_nodes = [n for n in nodes if n.modality == "audio"]

    if visual_nodes:
        # Primary shape
        shapes = [n for n in visual_nodes if n.node_type == "shape"]
        colors = [n for n in visual_nodes if n.node_type == "color"]
        textures = [n for n in visual_nodes if n.node_type == "texture"]

        if shapes:
            primary = shapes[0]
            shape_desc = _describe_shape_features(primary.features)

            # Find associated color and texture
            color_desc = ""
            if colors:
                color_mod = _describe_color_features(colors[0].features)
                color_desc = f"{color_mod} {colors[0].label}".strip()

            texture_desc = ""
            if textures:
                texture_desc = _describe_texture_features(textures[0].features)

            # Compose primary description
            parts = []
            if primary.features.get("area_fraction", 0) > 0.3:
                parts.append("a large")
            elif primary.features.get("area_fraction", 0) < 0.05:
                parts.append("a small")
            else:
                parts.append("a")

            if color_desc:
                parts.append(color_desc)
            if texture_desc:
                parts.append(texture_desc)
            parts.append(f"{shape_desc} shape")

            sentence = f"I observe {' '.join(parts)}."
            sentences.append(sentence)

        # Describe spatial relationships
        spatial_edges = [e for e in edges if e.edge_type in (
            "contains", "contained_by", "above", "below",
            "left_of", "right_of", "adjacent_to",
        )]

        for edge in spatial_edges[:5]:  # cap to avoid overly long descriptions
            src = node_map.get(edge.source_id)
            tgt = node_map.get(edge.target_id)
            if not src or not tgt:
                continue

            src_desc = _describe_node(src)
            tgt_desc = _describe_node(tgt)

            if edge.edge_type == "contains":
                sentences.append(f"It contains {_indef(tgt_desc)}.")
            elif edge.edge_type == "contained_by":
                sentences.append(f"It sits inside {_indef(tgt_desc)}.")
            elif edge.edge_type == "above":
                sentences.append(f"The {src_desc} is positioned above {_indef(tgt_desc)}.")
            elif edge.edge_type == "below":
                sentences.append(f"Below it is {_indef(tgt_desc)}.")
            elif edge.edge_type == "left_of":
                sentences.append(f"To its left is {_indef(tgt_desc)}.")
            elif edge.edge_type == "right_of":
                sentences.append(f"To its right is {_indef(tgt_desc)}.")
            elif edge.edge_type == "adjacent_to":
                sentences.append(f"Adjacent to it is {_indef(tgt_desc)}.")

        # Part-of relationships
        part_edges = [e for e in edges if e.edge_type == "part_of"]
        for edge in part_edges[:3]:
            src = node_map.get(edge.source_id)
            tgt = node_map.get(edge.target_id)
            if src and tgt:
                sentences.append(
                    f"A {_describe_node(src)} protrusion is attached to the {_describe_node(tgt)}."
                )

    # Audio description
    if audio_nodes:
        audio_parts = []
        for node in audio_nodes:
            if node.node_type == "timbre":
                desc = _describe_timbre_features(node.features)
                audio_parts.append(f"a {desc} sound")
            elif node.node_type == "rhythm":
                desc = _describe_rhythm_features(node.features)
                audio_parts.append(f"a {desc} rhythmic pattern")
            elif node.node_type == "audio_event":
                freq_class = node.features.get("freq_class", "mid")
                audio_parts.append(f"a {freq_class}-frequency sound event")
            elif node.node_type == "frequency":
                band = node.features.get("band", "")
                if band:
                    audio_parts.append(f"prominent {band.replace('_', ' ')} frequencies")

        if audio_parts:
            # Check if cross-modal binding exists
            bound_edges = [e for e in edges if e.edge_type == "bound_to"]
            if bound_edges:
                sentences.append(
                    f"When interacted with, it produces {', '.join(audio_parts[:3])}."
                )
            else:
                sentences.append(
                    f"Simultaneously observed: {', '.join(audio_parts[:3])}."
                )

    # Observation confidence
    total_obs = sum(n.groundedness for n in nodes)
    if total_obs > 50:
        sentences.append(f"This pattern has been observed {total_obs} times with high consistency.")
    elif total_obs > 10:
        sentences.append(f"This pattern has been observed {total_obs} times.")
    else:
        sentences.append(f"This is a relatively new observation ({total_obs} sightings).")

    return " ".join(sentences)


def _build_query(
    nodes: list[_NodeInfo],
    edges: list[_EdgeInfo],
    grounded_label: str | None = None,
) -> str:
    """Build a naming query for the LLM.

    Presents the structural description and asks the LLM to propose
    a label. The response can be fed back into bridge.ground_label()
    to create a speculative grounding.

    Format:
        I have observed a recurring visual pattern that I cannot yet name.
        [structural description]
        Based on these sensory features, what is this object most likely called?
        Respond with just the name (one or two words).
    """
    if grounded_label:
        return (
            f"I previously identified this as \"{grounded_label}\". "
            f"Based on continued observations, is this label still accurate?\n\n"
            f"{_build_structural(nodes, edges, grounded_label)}\n\n"
            f"If the label is wrong, what should it be called instead? "
            f"Respond with just the name (one or two words), or 'correct' if the label is right."
        )

    structural = _build_structural(nodes, edges)
    total_obs = sum(n.groundedness for n in nodes)

    return (
        f"I have observed a recurring sensory pattern ({total_obs} observations) "
        f"that I cannot yet name.\n\n"
        f"{structural}\n\n"
        f"Based on these sensory features and spatial relationships, "
        f"what is this object or scene element most likely called? "
        f"Respond with just the name (one or two words)."
    )


def _indef(noun_phrase: str) -> str:
    """Add indefinite article to a noun phrase."""
    if not noun_phrase:
        return "something"
    first_char = noun_phrase.lstrip()[0].lower() if noun_phrase.strip() else ""
    article = "an" if first_char in "aeiou" else "a"
    return f"{article} {noun_phrase}"


# ── Public API ───────────────────────────────────────────

async def describe_cluster(
    pool: asyncpg.Pool,
    cluster_id: int,
) -> SensoryDescription:
    """Generate a complete description of a sensory cluster.

    Works for grounded and ungrounded clusters. Traverses the cluster's
    member graph and cross-modal bindings to produce structural,
    narrative, and query descriptions.

    Args:
        pool: Sensory DB connection pool.
        cluster_id: The cluster to describe.

    Returns:
        SensoryDescription with all three output modes populated.
    """
    # Fetch cluster metadata
    cluster = await pool.fetchrow(
        """
        SELECT id, modality, grounded_label, grounding_confidence,
               member_count, total_observations, coherence, cluster_type
        FROM sensory_clusters WHERE id = $1
        """,
        cluster_id,
    )
    if not cluster:
        return SensoryDescription(cluster_id=cluster_id)

    nodes, edges = await _fetch_cluster_graph(pool, cluster_id)
    if not nodes:
        return SensoryDescription(cluster_id=cluster_id)

    grounded = cluster["grounded_label"]
    modalities = list(set(n.modality for n in nodes))

    return SensoryDescription(
        cluster_id=cluster_id,
        grounded_label=grounded,
        structural=_build_structural(nodes, edges, grounded),
        narrative=_build_narrative(nodes, edges, grounded),
        query=_build_query(nodes, edges, grounded),
        confidence=cluster["grounding_confidence"] or cluster["coherence"] or 0.0,
        observation_count=cluster["total_observations"],
        modalities=modalities,
        node_count=len(nodes),
        edge_count=len(edges),
    )


async def describe_node(
    pool: asyncpg.Pool,
    node_id: int,
    max_hops: int = 2,
) -> SensoryDescription:
    """Generate a description centered on a specific sensory node.

    Traverses the node's local neighborhood (up to max_hops) and
    produces descriptions of what surrounds it.

    Args:
        pool: Sensory DB connection pool.
        node_id: The node to describe.
        max_hops: How far to traverse from the focus node.

    Returns:
        SensoryDescription with all three output modes.
    """
    nodes, edges = await _fetch_node_neighborhood(pool, node_id, max_hops)
    if not nodes:
        return SensoryDescription(node_id=node_id)

    # Check if this node belongs to a grounded cluster
    cluster_info = await pool.fetchrow(
        """
        SELECT c.id, c.grounded_label, c.grounding_confidence
        FROM sensory_cluster_members cm
        JOIN sensory_clusters c ON c.id = cm.cluster_id
        WHERE cm.node_id = $1 AND c.cluster_type = 'grounded'
        LIMIT 1
        """,
        node_id,
    )

    grounded = cluster_info["grounded_label"] if cluster_info else None
    modalities = list(set(n.modality for n in nodes))

    return SensoryDescription(
        node_id=node_id,
        cluster_id=cluster_info["id"] if cluster_info else None,
        grounded_label=grounded,
        structural=_build_structural(nodes, edges, grounded),
        narrative=_build_narrative(nodes, edges, grounded),
        query=_build_query(nodes, edges, grounded),
        confidence=cluster_info["grounding_confidence"] if cluster_info else 0.0,
        observation_count=sum(n.groundedness for n in nodes),
        modalities=modalities,
        node_count=len(nodes),
        edge_count=len(edges),
    )


async def describe_active(
    pool: asyncpg.Pool,
    node_ids: list[int],
) -> SensoryDescription:
    """Describe a set of currently-active sensory nodes.

    Used during real-time observation: the caller provides the set of
    node IDs that are currently activated (from a frame or audio window),
    and this function produces a description of the scene.

    Args:
        pool: Sensory DB connection pool.
        node_ids: Currently activated node IDs.

    Returns:
        SensoryDescription of the active scene.
    """
    if not node_ids:
        return SensoryDescription()

    # Fetch all active nodes
    rows = await pool.fetch(
        """
        SELECT id, label, node_type, modality, features, groundedness
        FROM sensory_nodes
        WHERE id = ANY($1) AND NOT archived
        """,
        node_ids,
    )

    nodes = [
        _NodeInfo(
            id=r["id"], label=r["label"], node_type=r["node_type"],
            modality=r["modality"], features=_safe_features(r["features"]),
            groundedness=r["groundedness"],
        )
        for r in rows
    ]

    # Fetch edges between active nodes
    edge_rows = await pool.fetch(
        """
        SELECT source_id, target_id, edge_type, weight, groundedness
        FROM sensory_edges
        WHERE source_id = ANY($1) AND target_id = ANY($1)
        ORDER BY weight DESC
        """,
        node_ids,
    )

    edges = [
        _EdgeInfo(
            source_id=r["source_id"], target_id=r["target_id"],
            edge_type=r["edge_type"], weight=r["weight"],
            groundedness=r["groundedness"],
        )
        for r in edge_rows
    ]

    # Check if any active nodes belong to grounded clusters
    grounded_info = await pool.fetchrow(
        """
        SELECT c.grounded_label, c.grounding_confidence
        FROM sensory_cluster_members cm
        JOIN sensory_clusters c ON c.id = cm.cluster_id
        WHERE cm.node_id = ANY($1) AND c.cluster_type = 'grounded'
        ORDER BY c.grounding_confidence DESC
        LIMIT 1
        """,
        node_ids,
    )

    grounded = grounded_info["grounded_label"] if grounded_info else None
    modalities = list(set(n.modality for n in nodes))

    return SensoryDescription(
        grounded_label=grounded,
        structural=_build_structural(nodes, edges, grounded),
        narrative=_build_narrative(nodes, edges, grounded),
        query=_build_query(nodes, edges, grounded),
        confidence=grounded_info["grounding_confidence"] if grounded_info else 0.0,
        observation_count=sum(n.groundedness for n in nodes),
        modalities=modalities,
        node_count=len(nodes),
        edge_count=len(edges),
    )


async def describe_all_ungrounded(
    pool: asyncpg.Pool,
    limit: int = 10,
) -> list[SensoryDescription]:
    """Describe all stable but ungrounded clusters.

    Returns descriptions sorted by observation count (most-observed
    first). Useful for batch naming: feed each query to the LLM
    and ground the results.

    Args:
        pool: Sensory DB connection pool.
        limit: Maximum clusters to describe.

    Returns:
        List of SensoryDescriptions, each with a query for the LLM.
    """
    clusters = await pool.fetch(
        """
        SELECT id FROM sensory_clusters
        WHERE cluster_type = 'stable'
          AND grounded_symbol_id IS NULL
        ORDER BY total_observations DESC
        LIMIT $1
        """,
        limit,
    )

    results = []
    for row in clusters:
        desc = await describe_cluster(pool, row["id"])
        if desc.node_count > 0:
            results.append(desc)

    return results
