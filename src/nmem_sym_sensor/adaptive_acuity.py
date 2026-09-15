"""
Adaptive Acuity: Two-pass visual processing with dynamic resolution.

Pass 1 (Structure): Fast full-resolution edge/contour extraction that
identifies WHAT structures exist and WHERE. Classifies each region by
complexity (empty → simple → interesting → critical).

Pass 2 (Detail): Per-region encoding at resolution matched to complexity.
Empty regions are skipped. Critical regions get overlapping high-res tiles
with full MoE Complex path processing.

This mirrors biological foveated vision: the fovea provides ultra-high
resolution at the point of attention while peripheral vision provides
coarse spatial awareness. The key difference is that acuity is allocated
by structural complexity, not just gaze direction.

Use cases:
  - Sensory memory: efficient video frame processing, skip backgrounds
  - Medical imaging: vessel tracing in CT scans, high-acuity on vessels
  - General: any task where detail matters in some regions but not others
"""
import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

log = logging.getLogger(__name__)


class Acuity(IntEnum):
    """Processing detail level for a region."""
    EMPTY = 0        # uniform/blank — skip entirely
    SIMPLE = 1       # low complexity — 32x32 tile, fast path
    INTERESTING = 2  # moderate complexity — 64x64 tile, moderate path
    CRITICAL = 3     # high complexity — 128x128 tile with overlap, deep path


@dataclass
class ContourChain:
    """A continuous contour spanning multiple tiles.

    Represents a structure (vessel, edge, outline) that runs across
    the image. Built by stitching contour fragments from adjacent tiles.
    """
    chain_id: int
    points: list[tuple[int, int]]         # (x, y) points along the contour
    tiles: list[tuple[int, int]]          # (col, row) tiles this chain passes through
    entry_exit: list[dict]                # per-tile entry/exit points and angles
    length_px: float = 0.0
    mean_curvature: float = 0.0
    mean_thickness: float = 0.0           # estimated from edge distance
    is_branching: bool = False
    branch_points: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class StructureMap:
    """Output of Pass 1: global structural analysis."""
    edges: np.ndarray                     # full-res edge map (HxW, uint8)
    contours: list                        # raw contour lists from cv2
    chains: list[ContourChain]            # stitched contour chains
    acuity_grid: np.ndarray              # (grid_rows, grid_cols) Acuity values
    grid_size: int                        # tile size used for the grid
    image_shape: tuple[int, int]          # (H, W)


@dataclass
class TileSpec:
    """Specification for a single tile to encode in Pass 2."""
    col: int                              # grid column
    row: int                              # grid row
    x: int                                # pixel x of top-left
    y: int                                # pixel y of top-left
    size: int                             # tile size in pixels
    acuity: Acuity                        # processing level
    overlap: float                        # overlap fraction with neighbors
    contour_fragments: list | None = None # contour points within this tile
    entry_exits: list | None = None       # where contours cross tile boundary


# ── Pass 1: Structure extraction ─────────────────────────

def extract_structure(
    image: np.ndarray,
    grid_size: int = 64,
    edge_low: int = 30,
    edge_high: int = 100,
) -> StructureMap:
    """Pass 1: Extract global structure from full-resolution image.

    Runs edge detection on the full image, finds contours, and
    classifies each grid cell by structural complexity.

    Args:
        image: HWC uint8 image (BGR or RGB).
        grid_size: Base tile size for the acuity grid.
        edge_low: Canny low threshold.
        edge_high: Canny high threshold.

    Returns:
        StructureMap with edges, contours, chains, and acuity grid.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required for structure extraction")

    h, w = image.shape[:2]

    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Edge detection at full resolution
    edges = cv2.Canny(gray, edge_low, edge_high)

    # Find contours
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Build acuity grid
    grid_rows = math.ceil(h / grid_size)
    grid_cols = math.ceil(w / grid_size)
    acuity_grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * grid_size
            x1 = c * grid_size
            y2 = min(y1 + grid_size, h)
            x2 = min(x1 + grid_size, w)

            tile_edges = edges[y1:y2, x1:x2]
            edge_density = tile_edges.sum() / (255.0 * (y2 - y1) * (x2 - x1))

            # Also check local contrast (std dev of intensity)
            tile_gray = gray[y1:y2, x1:x2]
            contrast = float(np.std(tile_gray)) / 128.0  # normalize to 0-1ish

            # Classify acuity
            if edge_density < 0.005 and contrast < 0.05:
                acuity_grid[r, c] = Acuity.EMPTY
            elif edge_density < 0.03 and contrast < 0.15:
                acuity_grid[r, c] = Acuity.SIMPLE
            elif edge_density < 0.10:
                acuity_grid[r, c] = Acuity.INTERESTING
            else:
                acuity_grid[r, c] = Acuity.CRITICAL

    # Stitch contours into chains
    chains = _stitch_contour_chains(contours, w, h, grid_size)

    # Upgrade acuity for tiles containing chain segments
    # (chains represent continuous structures worth detailed processing)
    for chain in chains:
        for col, row in chain.tiles:
            if 0 <= row < grid_rows and 0 <= col < grid_cols:
                current = acuity_grid[row, col]
                if chain.length_px > grid_size * 2:
                    # Long chains (likely vessels/edges) get critical
                    acuity_grid[row, col] = max(current, Acuity.CRITICAL)
                else:
                    acuity_grid[row, col] = max(current, Acuity.INTERESTING)

    return StructureMap(
        edges=edges,
        contours=contours,
        chains=chains,
        acuity_grid=acuity_grid,
        grid_size=grid_size,
        image_shape=(h, w),
    )


def _stitch_contour_chains(
    contours: list,
    img_w: int,
    img_h: int,
    grid_size: int,
) -> list[ContourChain]:
    """Stitch raw contours into cross-tile chains.

    Each contour from cv2 is already a connected component. We convert
    them to ChainChain objects with tile membership and entry/exit info.
    """
    chains = []

    for i, contour in enumerate(contours):
        if len(contour) < 5:
            continue

        points_raw = contour.reshape(-1, 2)  # (N, 2) x,y
        points = [(int(p[0]), int(p[1])) for p in points_raw]

        # Which tiles does this contour pass through?
        tile_set = set()
        for x, y in points:
            col = x // grid_size
            row = y // grid_size
            tile_set.add((col, row))

        # Compute length
        length = 0.0
        for j in range(1, len(points)):
            dx = points[j][0] - points[j - 1][0]
            dy = points[j][1] - points[j - 1][1]
            length += math.sqrt(dx * dx + dy * dy)

        # Entry/exit points per tile
        entry_exit = []
        sorted_tiles = sorted(tile_set)
        for col, row in sorted_tiles:
            x1 = col * grid_size
            y1 = row * grid_size
            x2 = x1 + grid_size
            y2 = y1 + grid_size

            # Points in this tile
            tile_pts = [(x, y) for x, y in points
                       if x1 <= x < x2 and y1 <= y < y2]
            if not tile_pts:
                continue

            # Entry: first point near tile border
            # Exit: last point near tile border
            border_margin = 3
            entries = []
            for px, py in tile_pts:
                near_border = (px - x1 < border_margin or x2 - px < border_margin or
                              py - y1 < border_margin or y2 - py < border_margin)
                if near_border:
                    entries.append((px, py))

            entry_exit.append({
                "tile": (col, row),
                "n_points": len(tile_pts),
                "border_points": entries,
            })

        # Curvature estimate (mean angle change)
        curvatures = []
        for j in range(1, len(points) - 1):
            v1 = (points[j][0] - points[j-1][0], points[j][1] - points[j-1][1])
            v2 = (points[j+1][0] - points[j][0], points[j+1][1] - points[j][1])
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2) or 1
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2) or 1
            cos_angle = max(-1, min(1,
                (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)))
            curvatures.append(abs(math.acos(cos_angle)))

        chain = ContourChain(
            chain_id=i,
            points=points,
            tiles=sorted_tiles,
            entry_exit=entry_exit,
            length_px=length,
            mean_curvature=float(np.mean(curvatures)) if curvatures else 0.0,
        )
        chains.append(chain)

    # Sort by length (longest first — most likely to be significant structures)
    chains.sort(key=lambda c: c.length_px, reverse=True)

    return chains


# ── Pass 2: Adaptive tile generation ─────────────────────

def generate_tiles(
    image: np.ndarray,
    structure: StructureMap,
    min_acuity: Acuity = Acuity.SIMPLE,
) -> list[TileSpec]:
    """Generate tiles for Pass 2 encoding based on acuity map.

    Each tile is sized according to its acuity level:
      EMPTY:       skipped (not in output)
      SIMPLE:      32x32, no overlap
      INTERESTING: 64x64, 25% overlap with neighbors
      CRITICAL:    128x128, 50% overlap with neighbors

    Args:
        image: Original image.
        structure: StructureMap from Pass 1.
        min_acuity: Minimum acuity to include (default: skip EMPTY only).

    Returns:
        List of TileSpec objects for encoding.
    """
    h, w = image.shape[:2]
    grid = structure.acuity_grid
    base_size = structure.grid_size
    tiles = []

    # Size and overlap per acuity level
    acuity_config = {
        Acuity.SIMPLE: {"size": 32, "overlap": 0.0},
        Acuity.INTERESTING: {"size": 64, "overlap": 0.25},
        Acuity.CRITICAL: {"size": 128, "overlap": 0.50},
    }

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            acuity = Acuity(grid[r, c])
            if acuity < min_acuity:
                continue

            cfg = acuity_config[acuity]
            tile_size = cfg["size"]
            overlap = cfg["overlap"]

            # Center the tile on the grid cell
            cell_cx = c * base_size + base_size // 2
            cell_cy = r * base_size + base_size // 2
            x = max(0, min(cell_cx - tile_size // 2, w - tile_size))
            y = max(0, min(cell_cy - tile_size // 2, h - tile_size))

            # Find contour fragments in this tile
            contour_frags = None
            entry_exits = None
            for chain in structure.chains:
                for ee in chain.entry_exit:
                    if ee["tile"] == (c, r):
                        if contour_frags is None:
                            contour_frags = []
                            entry_exits = []
                        contour_frags.append({
                            "chain_id": chain.chain_id,
                            "n_points": ee["n_points"],
                            "length": chain.length_px,
                        })
                        entry_exits.extend(ee["border_points"])

            tiles.append(TileSpec(
                col=c, row=r,
                x=x, y=y,
                size=tile_size,
                acuity=acuity,
                overlap=overlap,
                contour_fragments=contour_frags,
                entry_exits=entry_exits,
            ))

    log.debug("Generated %d tiles: %d simple, %d interesting, %d critical",
             len(tiles),
             sum(1 for t in tiles if t.acuity == Acuity.SIMPLE),
             sum(1 for t in tiles if t.acuity == Acuity.INTERESTING),
             sum(1 for t in tiles if t.acuity == Acuity.CRITICAL))

    return tiles


def extract_tile_crop(
    image: np.ndarray,
    tile: TileSpec,
    target_size: int = 128,
) -> np.ndarray:
    """Extract and resize a tile crop from the image.

    All tiles are resized to target_size for the encoder, but CRITICAL
    tiles start at 128x128 (native resolution), while SIMPLE tiles
    start at 32x32 (upscaled 4x — less detail but the encoder handles
    it through the Simple MoE path).

    Args:
        image: Full image.
        tile: TileSpec from generate_tiles.
        target_size: Encoder input size.

    Returns:
        (target_size, target_size, C) uint8 array.
    """
    crop = image[tile.y:tile.y + tile.size, tile.x:tile.x + tile.size]

    if crop.shape[0] != target_size or crop.shape[1] != target_size:
        try:
            import cv2
            crop = cv2.resize(crop, (target_size, target_size),
                            interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Nearest neighbor fallback
            h, w = crop.shape[:2]
            row_idx = (np.arange(target_size) * h // target_size).astype(int)
            col_idx = (np.arange(target_size) * w // target_size).astype(int)
            crop = crop[np.ix_(row_idx, col_idx)]

    return crop


# ── Full two-pass pipeline ───────────────────────────────

@dataclass
class AcuityResult:
    """Complete result of adaptive acuity processing."""
    structure: StructureMap
    tiles: list[TileSpec]
    embeddings: list[tuple[TileSpec, np.ndarray]]  # (tile, embedding) pairs
    chains: list[ContourChain]
    stats: dict


def process_image(
    image: np.ndarray,
    encoder_fn=None,
    grid_size: int = 64,
    min_acuity: Acuity = Acuity.SIMPLE,
    encoder_input_size: int = 128,
) -> AcuityResult:
    """Full two-pass adaptive acuity processing.

    Pass 1: Structure extraction (fast, CPU, full resolution)
    Pass 2: Tile encoding (MoE encoder, per-tile at adaptive resolution)

    Args:
        image: HWC uint8 image.
        encoder_fn: Callable(batch_of_crops) → batch_of_embeddings.
                   If None, skips encoding (structure-only mode).
        grid_size: Base grid size for acuity classification.
        min_acuity: Minimum acuity to process.
        encoder_input_size: Input size for the encoder (tiles get resized).

    Returns:
        AcuityResult with structure, tiles, embeddings, and chains.
    """
    import time

    t0 = time.time()

    # Pass 1: Structure
    structure = extract_structure(image, grid_size)
    t_structure = time.time() - t0

    # Generate tiles
    tiles = generate_tiles(image, structure, min_acuity)

    # Pass 2: Encode tiles
    embeddings = []
    if encoder_fn is not None and tiles:
        # Batch encode for efficiency
        crops = []
        for tile in tiles:
            crop = extract_tile_crop(image, tile, encoder_input_size)
            crops.append(crop)

        if crops:
            import torch

            # Stack into batch tensor
            batch = np.stack(crops)
            batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0

            # Encode (the MoE router handles acuity-appropriate routing)
            with torch.no_grad():
                batch_embeddings = encoder_fn(batch_tensor)

            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.numpy()

            for tile, emb in zip(tiles, batch_embeddings):
                embeddings.append((tile, emb))

    t_total = time.time() - t0

    # Stats
    acuity_counts = {}
    for a in Acuity:
        acuity_counts[a.name] = int((structure.acuity_grid == a.value).sum())

    stats = {
        "image_shape": image.shape[:2],
        "grid_size": grid_size,
        "total_cells": structure.acuity_grid.size,
        "acuity_distribution": acuity_counts,
        "tiles_generated": len(tiles),
        "tiles_encoded": len(embeddings),
        "chains_found": len(structure.chains),
        "longest_chain_px": structure.chains[0].length_px if structure.chains else 0,
        "time_structure_ms": round(t_structure * 1000, 1),
        "time_total_ms": round(t_total * 1000, 1),
    }

    return AcuityResult(
        structure=structure,
        tiles=tiles,
        embeddings=embeddings,
        chains=structure.chains,
        stats=stats,
    )
