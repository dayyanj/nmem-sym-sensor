"""
Consolidation: sensory primitive clustering and concept formation.

This is where learning happens. Raw sensory primitives that consistently
co-occur are grouped into clusters, and clusters that reach sufficient
stability become concepts eligible for grounding to nmem-sym symbols.

The process mirrors human perceptual learning:
1. Iconic buffer → short-term nodes (frequent observations survive)
2. Short-term → long-term (repeated reinforcement promotes)
3. Long-term nodes cluster by embedding similarity
4. Stable clusters become concepts
5. Concepts ground to symbolic labels via nmem-sym bridge

No labels are assigned during consolidation — the system only discovers
structure. Labeling happens when concepts are grounded to the text-based
symbol graph.
"""
import logging
from typing import TYPE_CHECKING

import asyncpg
import numpy as np

from nmem_sym_sensor import config
from nmem_sym_sensor.graph import update_firing_thresholds

if TYPE_CHECKING:
    from nmem_sym_sensor.vocal_tract import VocalTract

log = logging.getLogger(__name__)


# ── Tier promotion ───────────────────────────────────────

async def promote_short_to_long(pool: asyncpg.Pool) -> list[int]:
    """Promote short-term nodes to long-term when they have enough observations.

    Returns list of promoted node IDs.
    """
    rows = await pool.fetch(
        """
        UPDATE sensory_nodes
        SET memory_tier = 'long_term',
            tier_promoted_at = NOW(),
            updated_at = NOW()
        WHERE memory_tier = 'short_term'
          AND observation_count >= $1
          AND NOT archived
        RETURNING id, label
        """,
        config.SHORT_TERM_PROMOTION_THRESHOLD,
    )

    for r in rows:
        log.debug("Promoted '%s' (id=%d) to long_term", r["label"], r["id"])

    return [r["id"] for r in rows]


async def decay_short_term(pool: asyncpg.Pool) -> int:
    """Archive short-term nodes that have decayed past their retention period.

    Returns count of archived nodes.
    """
    result = await pool.execute(
        f"""
        UPDATE sensory_nodes
        SET archived = TRUE, updated_at = NOW()
        WHERE memory_tier = 'short_term'
          AND NOT archived
          AND updated_at < NOW() - INTERVAL '{config.SHORT_TERM_DECAY_HOURS} hours'
          AND observation_count < $1
        """,
        config.SHORT_TERM_PROMOTION_THRESHOLD,
    )
    count = int(result.split()[-1])
    if count > 0:
        log.info("Archived %d decayed short-term nodes", count)
    return count


# ── Clustering ───────────────────────────────────────────

async def run_visual_clustering(
    pool: asyncpg.Pool,
    dry_run: bool = False,
) -> dict:
    """Cluster visual primitives by embedding similarity, PER NODE TYPE.

    Colors cluster with colors, shapes with shapes, textures with textures.
    This prevents "red" and "oval" from ending up in the same cluster.

    Each node type uses only its relevant embedding dimensions for
    similarity computation, so color differences aren't drowned out
    by 250 zero dimensions.

    Returns combined stats dict.
    """
    # Cluster each visual node type separately
    visual_types = ["color", "shape", "texture"]
    combined = {"modality": "visual", "nodes": 0, "pairs_found": 0, "clusters_formed": 0}

    for node_type in visual_types:
        result = await _cluster_modality(
            pool, "visual", "visual_embedding",
            config.VISUAL_MERGE_THRESHOLD, config.VISUAL_FEATURE_DIM,
            dry_run=dry_run,
            node_type_filter=node_type,
        )
        combined["nodes"] += result.get("nodes", 0)
        combined["pairs_found"] += result.get("pairs_found", 0)
        combined["clusters_formed"] += result.get("clusters_formed", 0)

    return combined


async def run_audio_clustering(
    pool: asyncpg.Pool,
    dry_run: bool = False,
) -> dict:
    """Cluster audio primitives by embedding similarity."""
    return await _cluster_modality(
        pool, "audio", "audio_embedding",
        config.AUDIO_MERGE_THRESHOLD, config.AUDIO_FEATURE_DIM,
        dry_run=dry_run,
    )


async def _cluster_modality(
    pool: asyncpg.Pool,
    modality: str,
    embedding_col: str,
    merge_threshold: float,
    embed_dim: int,
    dry_run: bool = False,
    node_type_filter: str | None = None,
) -> dict:
    """Generic clustering for a single modality (optionally filtered by node type).

    When node_type_filter is set (e.g. "color"), only clusters nodes of that
    type and uses type-specific embedding dimensions for similarity. This
    prevents color signals from being drowned out by 250 zero dimensions.

    Type-specific dimension slices for visual embeddings:
      - shape: dims 0-15 (one-hot shape encoding)
      - color: dims 16-21 (HSV circular encoding)
      - texture: dims 32-39 (variance + edge density)
      - geometry: dims 40-47 (area, aspect ratio)

    For audio, uses the full embedding (all dims are meaningful).
    """
    # Fetch long-term, unarchived nodes with embeddings
    type_clause = f"AND node_type = '{node_type_filter}'" if node_type_filter else ""
    nodes = await pool.fetch(
        f"""
        SELECT id, label, node_type, {embedding_col}::text as embedding,
               groundedness, observation_count, embedder_id
        FROM sensory_nodes
        WHERE modality = $1
          AND memory_tier = 'long_term'
          AND NOT archived
          AND {embedding_col} IS NOT NULL
          {type_clause}
        ORDER BY groundedness DESC
        """,
        modality,
    )

    if len(nodes) < 2:
        return {"modality": modality, "nodes": len(nodes), "clusters_formed": 0,
                "node_type": node_type_filter}

    # node_id → vector-space tag, so a formed cluster inherits its members' embedder_id (its
    # centroid is only comparable within that space). NB: clustering here does not yet PARTITION
    # by embedder_id — safe while one encoder is in use; a mixed-space DB would need that filter.
    node_embedder = {n["id"]: n["embedder_id"] for n in nodes}

    # Parse embeddings and extract type-specific dimensions
    node_ids = []
    embeddings = []
    for node in nodes:
        emb_str = node["embedding"].strip("[]")
        if not emb_str:
            continue
        vec = [float(x) for x in emb_str.split(",")]
        node_ids.append(node["id"])
        embeddings.append(vec)

    if len(node_ids) < 2:
        return {"modality": modality, "nodes": 0, "clusters_formed": 0,
                "node_type": node_type_filter}

    emb_matrix = np.array(embeddings)

    # Extract type-specific dims for visual nodes
    # This concentrates the signal: color similarity computed on 6 color dims,
    # not diluted by 250 zero dims
    if modality == "visual" and node_type_filter:
        dim_slices = {
            "color": list(range(16, 22)),    # HSV circular encoding
            "shape": list(range(0, 16)),     # one-hot shape + geometry
            "texture": list(range(32, 40)),  # variance + edge density
        }
        relevant_dims = dim_slices.get(node_type_filter)
        if relevant_dims:
            emb_matrix = emb_matrix[:, relevant_dims]

    # Compute pairwise cosine similarity in Python (not pgvector)
    # This uses the type-specific dims, not the full vector
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = emb_matrix / norms
    sim_matrix = normalized @ normalized.T

    # Find pairs above threshold — but ONLY union nodes in the SAME vector space. Two 512-vecs from
    # different embedder generations (JEPA on/off, model swap) are not comparable, so a cross-space
    # cosine "match" is meaningless; keeping the guard here means every formed cluster is homogeneous
    # by construction (so its centroid + inherited embedder_id tag are truthful). Python equality
    # gives the right semantics: NULL(None)==NULL true, tag==same true, tag==other / tag==NULL false.
    pairs = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if (sim_matrix[i, j] >= merge_threshold
                    and node_embedder.get(node_ids[i]) == node_embedder.get(node_ids[j])):
                pairs.append((node_ids[i], node_ids[j], float(sim_matrix[i, j])))

    if not pairs:
        return {"modality": modality, "nodes": len(nodes), "clusters_formed": 0}

    # Union-find for transitive closure
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, _ in pairs:
        union(a, b)

    # Group nodes by cluster root
    clusters: dict[int, list[int]] = {}
    for node in nodes:
        root = find(node["id"])
        clusters.setdefault(root, []).append(node["id"])

    # Filter: only clusters with enough members
    valid_clusters = {
        root: members
        for root, members in clusters.items()
        if len(members) >= config.CONCEPT_FORMATION_MIN_MEMBERS
    }

    if dry_run:
        log.info("[DRY RUN] Would form %d %s clusters from %d nodes",
                len(valid_clusters), modality, len(nodes))
        return {
            "modality": modality,
            "nodes": len(nodes),
            "pairs_found": len(pairs),
            "clusters_formed": len(valid_clusters),
            "dry_run": True,
        }

    # Create or update cluster records
    clusters_formed = 0
    for root, member_ids in valid_clusters.items():
        # Compute total observations
        total_obs = await pool.fetchval(
            """
            SELECT COALESCE(SUM(observation_count), 0)
            FROM sensory_nodes WHERE id = ANY($1)
            """,
            member_ids,
        )

        # Compute centroid embedding
        embeddings = await pool.fetch(
            f"""
            SELECT {embedding_col}::text as emb
            FROM sensory_nodes WHERE id = ANY($1) AND {embedding_col} IS NOT NULL
            """,
            member_ids,
        )

        centroid = None
        if embeddings:
            vecs = []
            for row in embeddings:
                # Parse pgvector text format: [0.1,0.2,...]
                emb_str = row["emb"].strip("[]")
                if emb_str:
                    vecs.append([float(x) for x in emb_str.split(",")])
            if vecs:
                centroid_vec = np.mean(vecs, axis=0).tolist()
                # Compute coherence (mean pairwise similarity)
                if len(vecs) > 1:
                    from itertools import combinations
                    sims = []
                    for va, vb in combinations(vecs, 2):
                        va, vb = np.array(va), np.array(vb)
                        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                        if na > 0 and nb > 0:
                            sims.append(float(np.dot(va, vb) / (na * nb)))
                    coherence = np.mean(sims) if sims else 0.0
                else:
                    coherence = 1.0

                centroid = centroid_vec

        centroid_col = "visual_centroid" if modality == "visual" else "audio_centroid"

        # Cluster inherits its members' vector-space tag (centroid lives in that space).
        cluster_embedder_id = next(
            (node_embedder.get(m) for m in member_ids if node_embedder.get(m)), None)

        # Upsert cluster
        cluster_id = await pool.fetchval(
            f"""
            INSERT INTO sensory_clusters
                (modality, cluster_type, {centroid_col}, member_count,
                 total_observations, coherence, embedder_id)
            VALUES ($1, 'proto', $2, $3, $4, $5, $6)
            RETURNING id
            """,
            modality,
            str(centroid) if centroid else None,
            len(member_ids),
            total_obs,
            round(coherence, 4) if centroid else 0.0,
            cluster_embedder_id,
        )

        # Add members
        for mid in member_ids:
            await pool.execute(
                """
                INSERT INTO sensory_cluster_members (cluster_id, node_id, role)
                VALUES ($1, $2, 'member')
                ON CONFLICT (cluster_id, node_id) DO NOTHING
                """,
                cluster_id, mid,
            )

        # Set prototype (highest groundedness member)
        prototype_id = await pool.fetchval(
            """
            SELECT id FROM sensory_nodes
            WHERE id = ANY($1)
            ORDER BY groundedness DESC, observation_count DESC
            LIMIT 1
            """,
            member_ids,
        )
        if prototype_id:
            await pool.execute(
                """
                UPDATE sensory_cluster_members SET role = 'prototype'
                WHERE cluster_id = $1 AND node_id = $2
                """,
                cluster_id, prototype_id,
            )

        # Create instance_of edges from members to cluster node
        for mid in member_ids:
            if mid != prototype_id:
                await pool.execute(
                    """
                    INSERT INTO sensory_edges (source_id, target_id, edge_type, weight, confidence)
                    VALUES ($1, $2, 'member_of', 0.8, 0.8)
                    ON CONFLICT (source_id, target_id, edge_type) DO NOTHING
                    """,
                    mid, prototype_id,
                )

        clusters_formed += 1
        log.info("Formed %s cluster #%d: %d members, %d total observations, coherence=%.3f",
                modality, cluster_id, len(member_ids), total_obs,
                coherence if centroid else 0.0)

    # Update firing thresholds after structural changes
    await update_firing_thresholds(pool)

    return {
        "modality": modality,
        "nodes": len(nodes),
        "pairs_found": len(pairs),
        "clusters_formed": clusters_formed,
    }


# ── Concept promotion ───────────────────────────────────

async def promote_clusters_to_stable(pool: asyncpg.Pool) -> list[int]:
    """Promote proto clusters to stable when they reach sufficient size and groundedness.

    Stable clusters are eligible for grounding to nmem-sym symbols.
    Returns list of promoted cluster IDs.
    """
    rows = await pool.fetch(
        """
        UPDATE sensory_clusters
        SET cluster_type = 'stable', updated_at = NOW()
        WHERE cluster_type = 'proto'
          AND member_count >= $1
          AND total_observations >= $2
          AND coherence >= 0.5
        RETURNING id, modality, member_count, total_observations
        """,
        config.CONCEPT_FORMATION_MIN_MEMBERS,
        config.CONCEPT_FORMATION_MIN_GROUNDEDNESS,
    )

    for r in rows:
        log.info("Cluster #%d (%s) promoted to stable: %d members, %d observations",
                r["id"], r["modality"], r["member_count"], r["total_observations"])

    return [r["id"] for r in rows]


# ── Cluster splitting (cognitive maturation) ─────────────

async def split_overloaded_clusters(pool: asyncpg.Pool) -> list[dict]:
    """Split clusters that have become too diverse.

    As a cluster accumulates members, its coherence may drop below
    SPLIT_COHERENCE_THRESHOLD. This means the members are no longer
    similar enough to represent a single concept.

    When this happens, the cluster is re-clustered with a tighter
    similarity threshold. The original becomes the parent category
    and the subclusters become children.

    Example: "blue" cluster (coherence 0.55) splits into:
      - "light blue" subcluster (coherence 0.92)
      - "dark blue" subcluster (coherence 0.88)
      - "blue" remains as parent category

    Returns list of split results.
    """
    # Find clusters eligible for splitting
    candidates = await pool.fetch(
        """
        SELECT id, modality, member_count, total_observations,
               coherence, coherence_at_formation, grounded_label, generation
        FROM sensory_clusters
        WHERE cluster_type IN ('stable', 'grounded')
          AND member_count >= $1
          AND total_observations >= $2
          AND coherence < $3
          AND generation < $4
        ORDER BY coherence ASC
        LIMIT 10
        """,
        config.SPLIT_MIN_MEMBERS,
        config.SPLIT_MIN_OBSERVATIONS,
        config.SPLIT_COHERENCE_THRESHOLD,
        config.SPLIT_MAX_GENERATION,
    )

    if not candidates:
        return []

    results = []
    for cluster in candidates:
        result = await _split_cluster(pool, cluster)
        if result:
            results.append(result)

    return results


async def _split_cluster(pool: asyncpg.Pool, cluster) -> dict | None:
    """Split a single cluster into tighter subclusters.

    Uses KNN within the cluster's members to find natural subgroups.
    """
    cluster_id = cluster["id"]
    modality = cluster["modality"]
    emb_col = "visual_embedding" if modality == "visual" else "audio_embedding"

    # Fetch member embeddings
    members = await pool.fetch(
        f"""
        SELECT n.id, n.label, n.{emb_col}::text as emb, n.groundedness, n.embedder_id
        FROM sensory_cluster_members cm
        JOIN sensory_nodes n ON n.id = cm.node_id
        WHERE cm.cluster_id = $1 AND n.{emb_col} IS NOT NULL
        """,
        cluster_id,
    )

    if len(members) < config.SPLIT_MIN_MEMBERS:
        return None

    # Parse embeddings
    node_ids = []
    embeddings = []
    for m in members:
        emb_str = m["emb"].strip("[]")
        if not emb_str:
            continue
        node_ids.append(m["id"])
        embeddings.append([float(x) for x in emb_str.split(",")])

    if len(node_ids) < config.SPLIT_MIN_MEMBERS:
        return None

    emb_matrix = np.array(embeddings)

    # Compute pairwise cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = emb_matrix / norms
    sim_matrix = normalized @ normalized.T

    # Tighter threshold for subclusters
    base_threshold = (config.VISUAL_MERGE_THRESHOLD if modality == "visual"
                      else config.AUDIO_MERGE_THRESHOLD)
    tight_threshold = min(0.98, base_threshold * config.SPLIT_TIGHTER_FACTOR)

    # Union-find with tighter threshold
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if sim_matrix[i, j] >= tight_threshold:
                union(i, j)

    # Group by subcluster
    subgroups: dict[int, list[int]] = {}
    for idx in range(len(node_ids)):
        root = find(idx)
        subgroups.setdefault(root, []).append(idx)

    # Filter: need at least 2 subclusters with minimum size
    valid_subs = [
        indices for indices in subgroups.values()
        if len(indices) >= config.SPLIT_MIN_SUBCLUSTER_SIZE
    ]

    if len(valid_subs) < 2:
        # Can't meaningfully split — cluster is diverse but not separable
        # Update coherence_at_formation to avoid re-checking
        await pool.execute(
            """
            UPDATE sensory_clusters
            SET coherence_at_formation = coherence, updated_at = NOW()
            WHERE id = $1
            """,
            cluster_id,
        )
        return None

    # Create subclusters
    parent_label = cluster["grounded_label"]
    parent_gen = cluster["generation"]
    centroid_col = "visual_centroid" if modality == "visual" else "audio_centroid"
    # Subclusters inherit their members' vector-space tag (same space as the parent).
    sub_embedder_id = next((m["embedder_id"] for m in members if m["embedder_id"]), None)

    created_subclusters = []
    for sub_indices in valid_subs:
        sub_node_ids = [node_ids[i] for i in sub_indices]
        sub_embeddings = emb_matrix[sub_indices]

        # Compute subcluster centroid
        sub_centroid = np.mean(sub_embeddings, axis=0).tolist()

        # Compute subcluster coherence
        if len(sub_indices) > 1:
            sub_norm = sub_embeddings / np.linalg.norm(sub_embeddings, axis=1, keepdims=True).clip(1e-8)
            sub_sim = sub_norm @ sub_norm.T
            # Mean of upper triangle
            n = len(sub_indices)
            upper = [sub_sim[i, j] for i in range(n) for j in range(i + 1, n)]
            sub_coherence = float(np.mean(upper)) if upper else 1.0
        else:
            sub_coherence = 1.0

        # Total observations for subcluster
        sub_obs = await pool.fetchval(
            "SELECT COALESCE(SUM(observation_count), 0) FROM sensory_nodes WHERE id = ANY($1)",
            sub_node_ids,
        )

        # Create the subcluster
        sub_id = await pool.fetchval(
            f"""
            INSERT INTO sensory_clusters
                (modality, cluster_type, parent_cluster_id, generation,
                 {centroid_col}, member_count, total_observations,
                 coherence, coherence_at_formation, embedder_id)
            VALUES ($1, 'stable', $2, $3, $4, $5, $6, $7, $7, $8)
            RETURNING id
            """,
            modality,
            cluster_id,
            parent_gen + 1,
            str(sub_centroid),
            len(sub_node_ids),
            sub_obs,
            round(sub_coherence, 4),
            sub_embedder_id,
        )

        # Move members from parent to subcluster
        for nid in sub_node_ids:
            await pool.execute(
                """
                DELETE FROM sensory_cluster_members
                WHERE cluster_id = $1 AND node_id = $2
                """,
                cluster_id, nid,
            )
            await pool.execute(
                """
                INSERT INTO sensory_cluster_members (cluster_id, node_id, role)
                VALUES ($1, $2, 'member')
                ON CONFLICT (cluster_id, node_id) DO NOTHING
                """,
                sub_id, nid,
            )

        # Set prototype (highest groundedness)
        proto_id = await pool.fetchval(
            "SELECT id FROM sensory_nodes WHERE id = ANY($1) ORDER BY groundedness DESC LIMIT 1",
            sub_node_ids,
        )
        if proto_id:
            await pool.execute(
                "UPDATE sensory_cluster_members SET role = 'prototype' WHERE cluster_id = $1 AND node_id = $2",
                sub_id, proto_id,
            )

        # Get representative member labels for logging
        sub_labels = await pool.fetch(
            "SELECT label FROM sensory_nodes WHERE id = ANY($1) ORDER BY groundedness DESC LIMIT 4",
            sub_node_ids,
        )
        label_str = ", ".join(r["label"] for r in sub_labels)

        created_subclusters.append({
            "subcluster_id": sub_id,
            "members": len(sub_node_ids),
            "coherence": round(sub_coherence, 4),
            "labels": label_str,
        })

        log.info("  Subcluster #%d: %d members, coherence=%.3f [%s]",
                sub_id, len(sub_node_ids), sub_coherence, label_str)

    # Update parent: mark as category, keep label, update member count
    remaining_members = await pool.fetchval(
        "SELECT COUNT(*) FROM sensory_cluster_members WHERE cluster_id = $1",
        cluster_id,
    )
    await pool.execute(
        """
        UPDATE sensory_clusters
        SET member_count = $1,
            coherence_at_formation = coherence,
            updated_at = NOW()
        WHERE id = $2
        """,
        remaining_members,
        cluster_id,
    )

    # Delete parent if all members were moved to subclusters
    if remaining_members == 0:
        await pool.execute("DELETE FROM sensory_clusters WHERE id = $1", cluster_id)
        log.info("  Parent cluster #%d emptied after split — deleted", cluster_id)

    result = {
        "parent_cluster_id": cluster_id,
        "parent_label": parent_label,
        "parent_coherence": cluster["coherence"],
        "subclusters": created_subclusters,
        "generation": parent_gen + 1,
    }

    log.info("SPLIT cluster #%d ('%s', coherence=%.3f) → %d subclusters (gen %d)",
            cluster_id, parent_label or "ungrounded",
            cluster["coherence"], len(created_subclusters), parent_gen + 1)

    return result


# ── Coherence refresh ────────────────────────────────────

async def refresh_cluster_coherence(pool: asyncpg.Pool) -> int:
    """Recompute coherence for all clusters.

    Called before split detection to ensure coherence reflects
    the current state (new members may have been added since
    the cluster was formed).

    Returns number of clusters updated.
    """
    # Only refresh clusters that are large enough to matter and might split.
    # Small clusters and deep subclusters don't need frequent coherence updates.
    clusters = await pool.fetch("""
        SELECT id, modality FROM sensory_clusters
        WHERE cluster_type IN ('stable', 'grounded')
          AND member_count >= $1
          AND generation < $2
        ORDER BY updated_at ASC
        LIMIT 50
    """, config.SPLIT_MIN_MEMBERS, config.SPLIT_MAX_GENERATION)

    updated = 0
    for c in clusters:
        emb_col = "visual_embedding" if c["modality"] == "visual" else "audio_embedding"

        embeddings = await pool.fetch(
            f"""
            SELECT n.{emb_col}::text as emb
            FROM sensory_cluster_members cm
            JOIN sensory_nodes n ON n.id = cm.node_id
            WHERE cm.cluster_id = $1 AND n.{emb_col} IS NOT NULL
            """,
            c["id"],
        )

        if len(embeddings) < 2:
            continue

        vecs = []
        for row in embeddings:
            emb_str = row["emb"].strip("[]")
            if emb_str:
                vecs.append([float(x) for x in emb_str.split(",")])

        if len(vecs) < 2:
            continue

        arr = np.array(vecs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(1e-8)
        normalized = arr / norms
        sim = normalized @ normalized.T
        n = len(vecs)
        upper = [sim[i, j] for i in range(n) for j in range(i + 1, n)]
        coherence = float(np.mean(upper))

        await pool.execute(
            """
            UPDATE sensory_clusters SET coherence = $1, updated_at = NOW()
            WHERE id = $2
            """,
            round(coherence, 4),
            c["id"],
        )
        updated += 1

    return updated


# ── Full consolidation cycle ─────────────────────────────

async def run_consolidation(
    pool: asyncpg.Pool,
    dry_run: bool = False,
    vocal_tract: "VocalTract | None" = None,
) -> dict:
    """Run a complete consolidation cycle.

    1. Flush expired iconic buffer entries
    2. Promote iconic → short-term (handled by graph.promote_from_iconic)
    3. Promote short-term → long-term
    4. Decay stale short-term nodes
    5. Cluster visual primitives
    6. Cluster audio primitives
    7. Promote proto clusters → stable

    Returns combined stats.
    """
    from nmem_sym_sensor.graph import flush_expired_iconic, promote_from_iconic

    stats = {}

    # 1. Flush expired iconic
    stats["iconic_flushed"] = await flush_expired_iconic(pool)

    # 2. Promote from iconic buffer
    promoted_iconic = await promote_from_iconic(pool)
    stats["iconic_promoted"] = len(promoted_iconic)

    # 3. Promote short-term → long-term
    promoted_lt = await promote_short_to_long(pool)
    stats["short_to_long"] = len(promoted_lt)

    # 4. Decay stale short-term
    stats["short_term_decayed"] = await decay_short_term(pool)

    # 5-6. Cluster by modality
    if not dry_run:
        stats["visual_clustering"] = await run_visual_clustering(pool)
        stats["audio_clustering"] = await run_audio_clustering(pool)
    else:
        stats["visual_clustering"] = await run_visual_clustering(pool, dry_run=True)
        stats["audio_clustering"] = await run_audio_clustering(pool, dry_run=True)

    # 7. Promote stable clusters
    stable = await promote_clusters_to_stable(pool)
    stats["clusters_promoted_stable"] = len(stable)

    # 8. Refresh coherence and split overloaded clusters
    if not dry_run:
        stats["coherence_refreshed"] = await refresh_cluster_coherence(pool)
        splits = await split_overloaded_clusters(pool)
        stats["clusters_split"] = len(splits)
        if splits:
            for s in splits:
                log.info("Split: #%d ('%s') → %d subclusters",
                        s["parent_cluster_id"], s.get("parent_label", "?"),
                        len(s["subclusters"]))

    # 9. Darwinian selection pressure — competitive displacement
    # Strong associations actively suppress weak ones. Overcrowded
    # memberships get pruned. Contested labels resolve toward the fittest.
    if not dry_run:
        from nmem_sym_sensor.selection import run_selection_pressure
        stats["selection"] = await run_selection_pressure(pool)

    # 10. Compositional analysis — discover part-whole patterns
    # and build structural expectations ("eye is usually part of face")
    if not dry_run:
        try:
            from nmem_sym_sensor.composition import run_compositional_analysis
            stats["composition"] = await run_compositional_analysis(pool)
        except Exception as e:  # noqa: BLE001 — one step must not abort the cycle (matches 11/12)
            log.debug("Compositional analysis skipped: %s", e)

    # 11. Sensory dreamstate — light offline consolidation
    # Full dreamstate runs after grounding batches; this is a quick pass
    if not dry_run:
        try:
            from nmem_sym_sensor.sensory_dreamstate import run_sensory_dreamstate
            dream_stats = await run_sensory_dreamstate(pool, max_duration_s=10.0, vocal_tract=vocal_tract)
            stats["dreamstate"] = dream_stats
        except Exception as e:
            log.debug("Dreamstate skipped: %s", e)

    # 12. Sound language consolidation — merge converged units, discover phrases
    if not dry_run:
        try:
            from nmem_sym_sensor import config as _cfg
            if _cfg.LANGUAGE_LEARNING_ENABLED:
                from nmem_sym_sensor.language import SoundLanguage
                lang = SoundLanguage(pool, similarity_threshold=_cfg.SOUND_SIMILARITY_THRESHOLD)
                lang_stats = await lang.consolidate()
                stats["language"] = lang_stats
        except Exception as e:
            log.debug("Language consolidation skipped: %s", e)

    # Final pass: sync cluster member counts and prune empties
    if not dry_run:
        from nmem_sym_sensor.graph import sync_cluster_counts
        stats["empty_clusters_pruned"] = await sync_cluster_counts(pool)

    log.info("Consolidation cycle complete: %s", stats)
    return stats
