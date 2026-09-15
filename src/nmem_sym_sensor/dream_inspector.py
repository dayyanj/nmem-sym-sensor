"""
Dream Inspector: visualise the system's mental imagery.

Say a word → see what the system imagines. Shows crude reconstructions
of the visual node embeddings that fire strongest for that word,
overlaid with opacity proportional to binding strength.

This is the system's mind's eye made visible. The images will be
blurry (MSE bottleneck limitation) but that's authentic — a mental
image IS a blurry impression, not a photograph.

Usage::

    inspector = DreamInspector(pool, decoder_path)

    # What does "elephant" look like in the system's mind?
    img = await inspector.imagine("elephant")
    cv2.imwrite("elephant_dream.png", img)

    # What does the system see when it hears sound unit #42?
    img = await inspector.imagine_sound(sound_unit_id=42)

    # What does the system associate with these visual nodes?
    img = await inspector.reconstruct_nodes([21561, 21546])
"""
import logging
from pathlib import Path

import asyncpg
import cv2
import numpy as np

log = logging.getLogger(__name__)


class DreamInspector:
    """Reconstruct and visualise the system's mental imagery."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        decoder_path: str | Path = "models/visual_decoder_v5.onnx",
        output_size: tuple[int, int] = (512, 512),
    ):
        self.pool = pool
        self.output_size = output_size
        self._decoder = self._load_decoder(decoder_path)

    def _load_decoder(self, path: str | Path):
        """Load visual decoder ONNX model."""
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            log.info("Dream inspector: loaded visual decoder from %s", path)
            return sess
        except Exception as e:
            log.warning("Dream inspector: decoder not available: %s", e)
            return None

    def decode_embedding(self, embedding: np.ndarray) -> np.ndarray | None:
        """Decode a 512-dim visual embedding to a 128x128 RGB image.

        Returns HWC uint8 BGR image or None.
        """
        if self._decoder is None:
            return None

        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        input_name = self._decoder.get_inputs()[0].name
        result = self._decoder.run(None, {input_name: emb})
        img = result[0][0]  # remove batch dim → (3, H, W)

        # Convert CHW float [0,1] → HWC uint8 BGR
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    async def imagine(
        self,
        word: str,
        max_nodes: int = 6,
        background: tuple[int, int, int] = (32, 32, 32),
    ) -> np.ndarray:
        """Visualise what the system imagines when it hears a word.

        Finds visual nodes most strongly co-occurring with the word,
        reconstructs each one, and composites them together with
        opacity proportional to binding strength.

        Returns BGR image at output_size.
        """
        word = word.lower().strip()

        # Find visual nodes bound to this word
        rows = await self.pool.fetch(
            """
            SELECT wv.node_id, wv.count,
                   n.label, n.visual_embedding::text as emb,
                   n.features::text as features
            FROM word_visual_cooccurrences wv
            JOIN sensory_nodes n ON n.id = wv.node_id
            WHERE wv.word = $1
              AND n.visual_embedding IS NOT NULL
              AND NOT n.archived
            ORDER BY wv.count DESC
            LIMIT $2
            """,
            word, max_nodes,
        )

        if not rows:
            # Try sound units instead
            return await self._imagine_via_sound(word, max_nodes, background)

        return await self._composite(rows, background)

    async def imagine_sound(
        self,
        sound_unit_id: int,
        max_nodes: int = 6,
        background: tuple[int, int, int] = (32, 32, 32),
    ) -> np.ndarray:
        """Visualise what the system imagines for a sound unit."""
        from nmem_sym_sensor.cooccurrence import cooc_store
        coocs = await cooc_store.query(
            sound_unit_id, "voice", self.pool,
            target_modality="visual", min_count=1, limit=max_nodes,
        )
        if not coocs:
            return await self._composite([], background)
        visual_ids = [c["unit_id"] for c in coocs]
        count_map = {c["unit_id"]: c["count"] for c in coocs}
        rows = await self.pool.fetch(
            """
            SELECT id as node_id, label, visual_embedding::text as emb,
                   features::text as features
            FROM sensory_nodes
            WHERE id = ANY($1) AND visual_embedding IS NOT NULL AND NOT archived
            """,
            visual_ids,
        )
        # Add count from coocs
        rows = [dict(r) | {"count": count_map.get(r["id"], 0)} for r in rows]
        rows.sort(key=lambda x: x["count"], reverse=True)

        return await self._composite(rows, background)

    async def reconstruct_nodes(
        self,
        node_ids: list[int],
        background: tuple[int, int, int] = (32, 32, 32),
    ) -> np.ndarray:
        """Reconstruct specific visual nodes."""
        rows = await self.pool.fetch(
            """
            SELECT id as node_id, 1 as count,
                   label, visual_embedding::text as emb,
                   features::text as features
            FROM sensory_nodes
            WHERE id = ANY($1) AND visual_embedding IS NOT NULL
            """,
            node_ids,
        )

        return await self._composite(rows, background)

    async def _imagine_via_sound(
        self,
        word: str,
        max_nodes: int,
        background: tuple[int, int, int],
    ) -> np.ndarray:
        """Fall back to sound unit → visual binding."""
        unit = await self.pool.fetchrow(
            "SELECT id FROM sound_units WHERE stt_label = $1 ORDER BY total_observations DESC LIMIT 1",
            word,
        )
        if unit:
            return await self.imagine_sound(unit["id"], max_nodes, background)

        # Nothing found — return blank with text
        canvas = np.full((*self.output_size, 3), background, dtype=np.uint8)
        cv2.putText(canvas, f'"{word}" — no associations', (20, self.output_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 1)
        return canvas

    async def _composite(
        self,
        rows,
        background: tuple[int, int, int],
    ) -> np.ndarray:
        """Lay out node reconstructions as a grid of tiles.

        Each tile shows one visual association, labelled with its name
        and binding strength. Tiles are sorted strongest-first,
        sized to fill the output canvas.
        """
        w, h = self.output_size
        canvas = np.full((h, w, 3), background, dtype=np.uint8)

        if not rows:
            return canvas

        # Decode all nodes
        patches = []
        max_count = max(r["count"] for r in rows)
        for r in rows:
            emb_str = r["emb"].strip("[]")
            if not emb_str:
                continue
            emb = np.array([float(x) for x in emb_str.split(",")])
            patch = self.decode_embedding(emb)
            if patch is not None:
                patches.append((patch, r["label"], r["count"]))

        if not patches:
            return canvas

        # Grid layout — pick columns/rows to fit
        n = len(patches)
        cols = min(n, 3)
        grid_rows = (n + cols - 1) // cols
        tile_w = w // cols
        tile_h = h // grid_rows
        pad = 4  # px between tiles

        for idx, (patch, label, count) in enumerate(patches):
            row_i = idx // cols
            col_i = idx % cols
            x0 = col_i * tile_w + pad
            y0 = row_i * tile_h + pad
            inner_w = tile_w - 2 * pad
            inner_h = tile_h - 2 * pad

            tile = cv2.resize(patch, (inner_w, inner_h), interpolation=cv2.INTER_LINEAR)

            # Brightness proportional to binding strength
            strength = count / max(max_count, 1)
            brightness = 0.4 + 0.6 * strength
            tile = np.clip(tile.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            canvas[y0:y0 + inner_h, x0:x0 + inner_w] = tile

            # Label under the tile
            pct = int(strength * 100)
            text = f"{label} ({pct}%)"
            cv2.putText(canvas, text, (x0 + 2, y0 + inner_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return canvas

    async def dream_report(
        self,
        words: list[str],
        output_dir: str = "/tmp/dreams",
    ) -> list[str]:
        """Generate dream images for a list of words.

        Returns list of saved file paths.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        paths = []
        for word in words:
            img = await self.imagine(word)
            path = os.path.join(output_dir, f"dream_{word}.png")
            cv2.imwrite(path, img)
            paths.append(path)
            log.info("Dream: '%s' → %s", word, path)

        return paths
