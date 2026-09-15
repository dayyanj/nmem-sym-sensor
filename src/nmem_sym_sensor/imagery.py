"""
Mental imagery: cross-modal activation between sound and vision.

When you hear "elephant", you see one in your mind's eye.
When you see an elephant, the word comes to mind.

This is bidirectional cross-activation — the same mechanism that
lets you visualise a friend's face when you hear their name, or
"hear" a song when you see its album cover.

Uses the sound-visual co-occurrence data built by the language
module. No new learning happens here — imagery is purely
activating existing associations.
"""
import logging

import asyncpg

log = logging.getLogger(__name__)


class MentalImagery:
    """Cross-modal activation between sound units and visual nodes.

    Usage::

        imagery = MentalImagery(pool)

        # Hear "elephant" → see elephant
        visuals = await imagery.sound_to_visual(sound_unit_id=42)

        # See elephant → hear "elephant"
        sounds = await imagery.visual_to_sound(visual_node_ids=[101, 102])

        # Both directions at once
        result = await imagery.cross_activate(sound_unit_id=42, visual_node_ids=[101])
    """

    def __init__(self, pool: asyncpg.Pool, min_binding_count: int = 3):
        self.pool = pool
        self.min_binding_count = min_binding_count

    async def sound_to_visual(self, sound_unit_id: int) -> list[dict]:
        """Activate visual representations from a sound.

        "I hear this sound → what does it look like?"
        Returns visual nodes/clusters bound to this sound, ranked by
        co-occurrence strength.
        """
        from nmem_sym_sensor.cooccurrence import cooc_store
        coocs = await cooc_store.query(
            sound_unit_id, "voice", self.pool,
            target_modality="visual", min_count=self.min_binding_count, limit=10,
        )
        if not coocs:
            return []
        visual_ids = [c["unit_id"] for c in coocs]
        rows = await self.pool.fetch(
            """
            SELECT id as visual_node_id, label as visual_label, node_type,
                   visual_embedding IS NOT NULL as has_embedding
            FROM sensory_nodes
            WHERE id = ANY($1) AND NOT archived
            """,
            visual_ids,
        )
        # Merge count from coocs into rows
        count_map = {c["unit_id"]: c["count"] for c in coocs}
        for r in rows:
            r = dict(r)
            r["count"] = count_map.get(r["visual_node_id"], 0)
        rows = sorted([dict(r) | {"count": count_map.get(r["id"], 0)} for r in rows],
                       key=lambda x: x["count"], reverse=True)

        results = []
        for r in rows:
            # Check if this visual node belongs to a grounded cluster
            cluster = await self.pool.fetchrow(
                """
                SELECT c.grounded_label, c.grounding_confidence
                FROM sensory_cluster_members cm
                JOIN sensory_clusters c ON c.id = cm.cluster_id
                WHERE cm.node_id = $1
                  AND c.grounded_label IS NOT NULL
                ORDER BY c.grounding_confidence DESC
                LIMIT 1
                """,
                r["visual_node_id"],
            )

            results.append({
                "visual_node_id": r["visual_node_id"],
                "visual_label": r["visual_label"],
                "node_type": r["node_type"],
                "binding_strength": r["count"],
                "grounded_label": cluster["grounded_label"] if cluster else None,
                "grounding_confidence": float(cluster["grounding_confidence"]) if cluster else None,
            })

        if results:
            labels = [r["grounded_label"] or r["visual_label"] for r in results[:3]]
            log.debug("Sound→Visual imagery: unit #%d → %s", sound_unit_id, labels)

        return results

    async def visual_to_sound(self, visual_node_ids: list[int]) -> list[dict]:
        """Activate sound representations from visual input.

        "I see this → what does it sound like? What's it called?"
        Returns sound units bound to these visual nodes, ranked by
        co-occurrence strength. The STT label (if available) is the
        system's "inner voice" — the word that comes to mind.
        """
        if not visual_node_ids:
            return []

        from nmem_sym_sensor.cooccurrence import cooc_store
        # Collect voice units bound to any of these visual nodes
        all_sound: dict[int, int] = {}  # sound_id → total count
        for vid in visual_node_ids:
            coocs = await cooc_store.query(
                vid, "visual", self.pool,
                target_modality="voice", min_count=self.min_binding_count, limit=10,
            )
            for c in coocs:
                all_sound[c["unit_id"]] = all_sound.get(c["unit_id"], 0) + c["count"]

        if not all_sound:
            return []

        sound_ids = sorted(all_sound, key=all_sound.get, reverse=True)[:10]
        rows = await self.pool.fetch(
            """
            SELECT id as sound_unit_id, stt_label, stt_confidence,
                   total_observations, speaker_variance
            FROM sound_units WHERE id = ANY($1)
            """,
            sound_ids,
        )
        # Merge binding counts
        rows_with_count = []
        for r in rows:
            d = dict(r)
            d["total_binding"] = all_sound.get(d["sound_unit_id"], 0)
            rows_with_count.append(d)
        rows_with_count.sort(key=lambda x: x["total_binding"], reverse=True)
        rows = rows_with_count

        results = []
        for r in rows:
            results.append({
                "sound_unit_id": r["sound_unit_id"],
                "stt_label": r["stt_label"],  # the "word" — inner voice
                "stt_confidence": float(r["stt_confidence"] or 0),
                "binding_strength": r["total_binding"],
                "observations": r["total_observations"],
                "speaker_variance": float(r["speaker_variance"] or 0),
            })

        if results:
            words = [r["stt_label"] or f"unit#{r['sound_unit_id']}" for r in results[:3]]
            log.debug("Visual→Sound imagery: %d nodes → %s", len(visual_node_ids), words)

        return results

    async def cross_activate(
        self,
        sound_unit_id: int | None = None,
        visual_node_ids: list[int] | None = None,
    ) -> dict:
        """Bidirectional cross-activation.

        Given either or both modalities, activate the other direction.
        Returns a dict with 'visual_imagery' and 'sound_imagery' lists.
        """
        result = {
            "visual_imagery": [],  # what we "see" from sound
            "sound_imagery": [],   # what we "hear" from vision
        }

        if sound_unit_id is not None:
            result["visual_imagery"] = await self.sound_to_visual(sound_unit_id)

        if visual_node_ids:
            result["sound_imagery"] = await self.visual_to_sound(visual_node_ids)

        return result

    async def describe_mental_image(
        self,
        sound_unit_id: int | None = None,
        visual_node_ids: list[int] | None = None,
    ) -> str:
        """Human-readable description of what the system is imagining.

        Useful for debugging and understanding the system's inner state.
        """
        activation = await self.cross_activate(sound_unit_id, visual_node_ids)
        parts = []

        if sound_unit_id is not None:
            # Get the sound unit's label
            unit = await self.pool.fetchrow(
                "SELECT stt_label FROM sound_units WHERE id = $1", sound_unit_id,
            )
            sound_name = unit["stt_label"] if unit and unit["stt_label"] else f"sound#{sound_unit_id}"
            parts.append(f"Hearing '{sound_name}'")

            if activation["visual_imagery"]:
                visuals = [v["grounded_label"] or v["visual_label"]
                           for v in activation["visual_imagery"][:3]]
                parts.append(f"  → seeing: {', '.join(visuals)}")
            else:
                parts.append("  → no visual associations")

        if visual_node_ids:
            parts.append(f"Seeing {len(visual_node_ids)} visual features")

            if activation["sound_imagery"]:
                sounds = [s["stt_label"] or f"sound#{s['sound_unit_id']}"
                          for s in activation["sound_imagery"][:3]]
                parts.append(f"  → hearing: {', '.join(sounds)}")
            else:
                parts.append("  → no sound associations")

        return "\n".join(parts) if parts else "Nothing active"
