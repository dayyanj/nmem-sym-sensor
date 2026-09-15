"""
Compositional structure: recursive part-whole decomposition.

Objects are graphs of parts. Parts are graphs of sub-parts. Each level
can be zoomed into for finer detail, driven by attention and need.

    Level 0: Primitives (red, circle, smooth)
    Level 1: Parts (head-with-ears, elongated-body, thin-leg)
    Level 2: Objects (cat, cup, face)
    Level 3+: Fine detail (iris, pupil, vessel-branch)

The depth isn't fixed — it's recursive. Any node can be decomposed
further when the system needs to discriminate at finer granularity.

Key capabilities:
1. Decompose: break a node into sub-parts (driven by foveal attention + acuity)
2. Compose: discover recurring sub-graph patterns → promote to higher level
3. Infer: see a part → predict the whole ("this eye is part of a face")
4. Context-check: verify inference against observed siblings
"""
import logging

import asyncpg

log = logging.getLogger(__name__)


# ── Structural expectation learning ──────────────────────

async def record_part_whole(
    pool: asyncpg.Pool,
    part_node_id: int,
    whole_node_id: int,
    sibling_node_ids: list[int] | None = None,
) -> None:
    """Record that a part was observed as belonging to a whole.

    Called when part_of edges are created during frame ingestion.
    Over time, builds up expectations: "eyes are usually part of faces."

    Args:
        pool: DB pool.
        part_node_id: The part node (e.g., "eye").
        whole_node_id: The whole node (e.g., "face").
        sibling_node_ids: Other parts observed in the same whole.
    """
    # Get labels
    part = await pool.fetchrow(
        "SELECT label FROM sensory_nodes WHERE id = $1",
        part_node_id,
    )
    whole = await pool.fetchrow(
        "SELECT label FROM sensory_nodes WHERE id = $1",
        whole_node_id,
    )
    if not part or not whole:
        return

    part_label = part["label"]
    whole_label = whole["label"]

    # Get sibling labels
    siblings = []
    if sibling_node_ids:
        rows = await pool.fetch(
            "SELECT label as lbl FROM sensory_nodes WHERE id = ANY($1)",
            sibling_node_ids,
        )
        siblings = [r["lbl"] for r in rows]

    # Upsert expectation
    # Use a simple text[] column approach via JSONB to avoid double-encoding
    import json as _json

    existing = await pool.fetchrow(
        """SELECT expected_siblings, observation_count, confidence
           FROM structural_expectations
           WHERE part_label = $1 AND whole_label = $2 AND relation = 'part_of'""",
        part_label, whole_label,
    )

    if existing:
        # Parse existing siblings — handle the JSONB string/list ambiguity
        existing_sibs = set()
        raw = existing["expected_siblings"]
        if isinstance(raw, str):
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, list):
                    existing_sibs.update(s for s in parsed if isinstance(s, str))
            except (ValueError, TypeError):
                pass
        elif isinstance(raw, list):
            existing_sibs.update(s for s in raw if isinstance(s, str))

        existing_sibs.update(siblings)
        sibs_json = _json.dumps(sorted(existing_sibs))

        new_conf = min(0.99, existing["confidence"] + (1.0 - existing["confidence"]) * 0.05)

        await pool.execute(
            """UPDATE structural_expectations SET
                observation_count = observation_count + 1,
                confidence = $1,
                expected_siblings = $2::jsonb,
                updated_at = NOW()
               WHERE part_label = $3 AND whole_label = $4 AND relation = 'part_of'""",
            new_conf,
            sibs_json,
            part_label,
            whole_label,
        )
    else:
        await pool.execute(
            """INSERT INTO structural_expectations
                (part_label, whole_label, observation_count, confidence, expected_siblings)
               VALUES ($1, $2, 1, 0.5, $3::jsonb)""",
            part_label,
            whole_label,
            _json.dumps(siblings),
        )


async def infer_whole_from_part(
    pool: asyncpg.Pool,
    part_node_id: int,
    active_node_ids: list[int] | None = None,
) -> list[dict]:
    """Given a visible part, infer what whole it might belong to.

    Returns ranked list of possible wholes with confidence and
    sibling verification results.

    Example:
        Input: node "eye" is visible
        Output: [
            {"whole": "face", "confidence": 0.95, "siblings_present": ["nose", "mouth"],
             "siblings_missing": ["ear"], "context_match": 0.8},
            {"whole": "doll", "confidence": 0.3, "siblings_present": [],
             "siblings_missing": ["body", "hair"], "context_match": 0.1},
        ]

    The caller can use this to:
    - Annotate the current scene ("I think I'm looking at a face close-up")
    - Guide attention ("I should look for a nose nearby to confirm")
    - Distinguish part-of-whole from isolated object
    """
    part = await pool.fetchrow(
        "SELECT label FROM sensory_nodes WHERE id = $1",
        part_node_id,
    )
    if not part:
        return []

    part_label = part["label"]

    # Find all known wholes for this part
    expectations = await pool.fetch(
        """
        SELECT whole_label, confidence, observation_count, expected_siblings
        FROM structural_expectations
        WHERE part_label = $1
        ORDER BY confidence DESC
        """,
        part_label,
    )

    if not expectations:
        return []

    # Get labels of currently active nodes (for sibling verification)
    active_labels = set()
    if active_node_ids:
        rows = await pool.fetch(
            "SELECT label as lbl FROM sensory_nodes WHERE id = ANY($1)",
            active_node_ids,
        )
        active_labels = {r["lbl"] for r in rows}

    results = []
    for exp in expectations:
        whole_label = exp["whole_label"]
        base_confidence = float(exp["confidence"])

        # Check which expected siblings are present
        expected_sibs = set()
        if exp["expected_siblings"]:
            import json as _json
            raw = exp["expected_siblings"]
            # asyncpg may return JSONB as string or parsed object
            if isinstance(raw, str):
                try:
                    parsed = _json.loads(raw)
                    if isinstance(parsed, list):
                        expected_sibs.update(s for s in parsed if isinstance(s, str))
                    elif isinstance(parsed, str):
                        expected_sibs.add(parsed)
                except (ValueError, TypeError):
                    pass
            elif isinstance(raw, list):
                expected_sibs.update(s for s in raw if isinstance(s, str))

        # Deduplicate and remove self
        expected_sibs.discard(part_label)

        siblings_present = list(active_labels & expected_sibs) if expected_sibs else []
        siblings_missing = list(expected_sibs - active_labels) if expected_sibs else []

        # Context match: what fraction of expected siblings are present?
        if expected_sibs:
            context_match = len(siblings_present) / len(expected_sibs)
        else:
            context_match = 0.5  # no expectation data yet

        # Adjusted confidence: base confidence * context match
        # High context_match = "yes, this looks like part of a face"
        # Low context_match = "the siblings are missing, might be isolated"
        adjusted_confidence = base_confidence * (0.3 + 0.7 * context_match)

        results.append({
            "whole": whole_label,
            "confidence": round(adjusted_confidence, 3),
            "base_confidence": round(base_confidence, 3),
            "context_match": round(context_match, 3),
            "observation_count": exp["observation_count"],
            "siblings_present": siblings_present[:5],
            "siblings_missing": siblings_missing[:5],
        })

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


# ── Compositional pattern discovery ──────────────────────

async def discover_parts(
    pool: asyncpg.Pool,
    min_cooccurrences: int = 5,
    min_confidence: float = 0.6,
) -> list[dict]:
    """Discover recurring part compositions from part_of edges.

    When certain parts consistently appear together within the same
    parent, that's a compositional pattern worth promoting.

    Example: "eye", "nose", "mouth" consistently part_of the same
    parent → that parent pattern is a "face-like" composition.

    Returns list of discovered compositions.
    """
    # Find part_of edges with high groundedness (frequently observed)
    compositions = await pool.fetch(
        """
        SELECT
            e.target_id as parent_id,
            n_parent.label as parent_label,
            array_agg(n_child.label ORDER BY n_child.label) as part_labels,
            array_agg(n_child.id ORDER BY n_child.label) as part_ids,
            COUNT(*) as part_count,
            AVG(e.confidence) as avg_confidence
        FROM sensory_edges e
        JOIN sensory_nodes n_child ON n_child.id = e.source_id
        JOIN sensory_nodes n_parent ON n_parent.id = e.target_id
        WHERE e.edge_type = 'part_of'
          AND e.groundedness >= $1
          AND NOT n_child.archived
          AND NOT n_parent.archived
        GROUP BY e.target_id, n_parent.label
        HAVING COUNT(*) >= 2 AND AVG(e.confidence) >= $2
        ORDER BY COUNT(*) DESC
        LIMIT 50
        """,
        min_cooccurrences,
        min_confidence,
    )

    results = []
    for comp in compositions:
        results.append({
            "parent_id": comp["parent_id"],
            "parent_label": comp["parent_label"],
            "parts": comp["part_labels"],
            "part_ids": comp["part_ids"],
            "part_count": comp["part_count"],
            "avg_confidence": round(float(comp["avg_confidence"]), 3),
        })

        # Record structural expectations for each part
        for part_id in comp["part_ids"]:
            sibling_ids = [pid for pid in comp["part_ids"] if pid != part_id]
            await record_part_whole(
                pool, part_id, comp["parent_id"], sibling_ids,
            )

    if results:
        log.info("Discovered %d compositional patterns", len(results))

    return results


# ── Decomposition request ────────────────────────────────

async def should_decompose(
    pool: asyncpg.Pool,
    node_id: int,
) -> dict:
    """Determine whether a node should be decomposed further.

    Returns a recommendation based on:
    - Has the node been decomposed before? (check children)
    - Is the node's embedding ambiguous? (similar to multiple clusters)
    - Is the system trying to discriminate this node from similar ones?

    Returns dict with 'should_decompose', 'reason', 'current_depth'.
    """
    node = await pool.fetchrow(
        """
        SELECT id, label, level, decomposition_depth, observation_count,
               (SELECT COUNT(*) FROM sensory_nodes WHERE parent_node_id = $1) as child_count
        FROM sensory_nodes WHERE id = $1
        """,
        node_id,
    )

    if not node:
        return {"should_decompose": False, "reason": "node not found"}

    # Already decomposed?
    if node["child_count"] > 0:
        return {
            "should_decompose": False,
            "reason": "already decomposed",
            "current_depth": node["decomposition_depth"],
            "children": node["child_count"],
        }

    # Not enough observations to warrant decomposition
    if node["observation_count"] < 10:
        return {
            "should_decompose": False,
            "reason": "insufficient observations",
            "current_depth": node["decomposition_depth"],
        }

    # Check if this node is confused with other nodes (similar embeddings, different labels)
    confusion = await pool.fetchval(
        """
        SELECT COUNT(*) FROM sensory_nodes other
        WHERE other.id != $1
          AND other.node_type = (SELECT node_type FROM sensory_nodes WHERE id = $1)
          AND other.modality = 'visual'
          AND NOT other.archived
          AND other.visual_embedding IS NOT NULL
          AND (SELECT visual_embedding FROM sensory_nodes WHERE id = $1) IS NOT NULL
          AND 1 - (other.visual_embedding <=> (SELECT visual_embedding FROM sensory_nodes WHERE id = $1)) > 0.8
        """,
        node_id,
    )

    if confusion and confusion > 2:
        return {
            "should_decompose": True,
            "reason": f"confused with {confusion} similar nodes — finer detail needed",
            "current_depth": node["decomposition_depth"],
        }

    return {
        "should_decompose": False,
        "reason": "no discrimination needed",
        "current_depth": node["decomposition_depth"],
    }


# ── Run compositional analysis ───────────────────────────

async def run_compositional_analysis(pool: asyncpg.Pool) -> dict:
    """Run a full compositional analysis cycle.

    1. Discover recurring part compositions
    2. Build structural expectations
    3. Flag nodes that need deeper decomposition

    Called periodically during consolidation.
    """
    stats = {}

    # Discover compositional patterns
    compositions = await discover_parts(pool)
    stats["compositions_found"] = len(compositions)

    # Count structural expectations
    exp_count = await pool.fetchval(
        "SELECT COUNT(*) FROM structural_expectations WHERE confidence >= 0.5"
    )
    stats["expectations"] = exp_count

    if compositions:
        log.info("Compositional analysis: %d patterns, %d expectations",
                len(compositions), exp_count)

    return stats
