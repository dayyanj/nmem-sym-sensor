"""
Visual encoder: image → sensory primitives.

Decomposes images into constituent segments and extracts unlabeled feature
descriptors (shape, color, texture, spatial relationships). No classification
— the graph learns what things are through repeated observation.

Uses Canny edge detection + contour analysis. No external models — the system
learns to segment through observation, not pre-trained segmentation.
"""
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from nmem_sym_sensor import config

log = logging.getLogger(__name__)


@dataclass
class VisualPrimitive:
    """A single visual feature extracted from an image segment."""
    label: str                          # auto-generated: "oval-red-smooth"
    node_type: str                      # shape | color | texture
    features: dict                      # modality-specific descriptors
    embedding: list[float] | None = None  # visual feature vector
    bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    mask: np.ndarray | None = field(default=None, repr=False)  # binary mask
    area_fraction: float = 0.0          # fraction of total image area


@dataclass
class SpatialRelation:
    """A spatial relationship between two segments."""
    source_label: str
    target_label: str
    relation: str                       # above | below | left_of | right_of | contains | adjacent_to | overlaps
    confidence: float = 1.0


@dataclass
class FrameAnalysis:
    """Complete analysis of a single image frame."""
    frame_id: str
    primitives: list[VisualPrimitive]
    relations: list[SpatialRelation]
    width: int
    height: int
    fixations: list[tuple[float, float, float]] | None = None  # (x, y, saliency) temporal order


# ── Shape descriptors ────────────────────────────────────

def _classify_shape(contour_points: np.ndarray, area: float, perimeter: float) -> str:
    """Classify a contour into a basic geometric shape descriptor.

    Returns a shape name — these are descriptors, not object labels.
    The graph learns that "circle" shapes with "brown" color and
    "smooth" texture tend to form "cup-rim" clusters.
    """
    if perimeter == 0:
        return "point"

    circularity = 4 * math.pi * area / (perimeter * perimeter)
    n_vertices = len(contour_points)

    if circularity > 0.85:
        return "circle"
    elif circularity > 0.7:
        return "oval"

    # Approximate polygon
    try:
        import cv2
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour_points, epsilon, True)
        n_sides = len(approx)
    except ImportError:
        # Fallback: estimate from vertex count
        n_sides = max(3, min(n_vertices // 4, 12))

    if n_sides == 3:
        return "triangle"
    elif n_sides == 4:
        # Check aspect ratio for rectangle vs square
        # Squeeze contour_points to (N, 2) — cv2.findContours returns (N, 1, 2)
        pts = contour_points.reshape(-1, 2) if contour_points.ndim == 3 else contour_points
        if pts.size >= 4:
            x_range = pts[:, 0].max() - pts[:, 0].min()
            y_range = pts[:, 1].max() - pts[:, 1].min()
            if x_range > 0 and y_range > 0:
                aspect = max(x_range, y_range) / min(x_range, y_range)
                return "square" if aspect < 1.3 else "rectangle"
        return "rectangle"
    elif n_sides == 5:
        return "pentagon"
    elif n_sides == 6:
        return "hexagon"
    else:
        return "irregular"


def _quantize_color(color_bgr: np.ndarray) -> tuple[str, dict]:
    """Quantize a BGR color to a named color and feature dict.

    Uses HSV space for perceptually meaningful bucketing.
    Returns (color_name, {hue, saturation, value}).
    """
    # Convert BGR → HSV
    r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c

    # Value (brightness)
    v = max_c / 255.0

    # Saturation
    s = (diff / max_c) if max_c > 0 else 0.0

    # Hue
    if diff == 0:
        h = 0.0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    # Quantize to color name
    if v < 0.15:
        name = "black"
    elif s < 0.15 and v > 0.85:
        name = "white"
    elif s < 0.15:
        name = "gray"
    elif h < 15 or h >= 345:
        name = "red"
    elif h < 45:
        name = "orange"
    elif h < 75:
        name = "yellow"
    elif h < 165:
        name = "green"
    elif h < 195:
        name = "cyan"
    elif h < 255:
        name = "blue"
    elif h < 285:
        name = "purple"
    else:
        name = "pink"

    return name, {"hue": round(h, 1), "saturation": round(s, 3), "value": round(v, 3)}


def _compute_texture(region: np.ndarray) -> tuple[str, dict]:
    """Compute a basic texture descriptor from a grayscale region.

    Returns (texture_name, {variance, edge_density}).
    """
    if region.size == 0:
        return "flat", {"variance": 0.0, "edge_density": 0.0}

    # Convert to grayscale if needed
    if region.ndim == 3:
        gray = np.mean(region, axis=2)
    else:
        gray = region.astype(float)

    variance = float(np.var(gray))

    # Simple edge density via gradient magnitude
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        gy = np.diff(gray, axis=0)
        gx = np.diff(gray, axis=1)
        # Crop to matching dimensions
        min_h = min(gy.shape[0], gx.shape[0])
        min_w = min(gy.shape[1], gx.shape[1])
        grad_mag = np.sqrt(gy[:min_h, :min_w] ** 2 + gx[:min_h, :min_w] ** 2)
        edge_density = float(np.mean(grad_mag > 30))
    else:
        edge_density = 0.0

    # Classify
    if variance < 100 and edge_density < 0.05:
        name = "smooth"
    elif edge_density > 0.3:
        name = "textured"
    elif variance > 1000:
        name = "noisy"
    else:
        name = "matte"

    return name, {"variance": round(variance, 2), "edge_density": round(edge_density, 4)}


# ── Spatial relation extraction ──────────────────────────

def _compute_part_features(
    primitive: "VisualPrimitive",
    parent: "VisualPrimitive | None" = None,
    image_h: int = 1,
    image_w: int = 1,
) -> dict:
    """Compute richer part descriptors for a segment.

    These capture structural properties that are more view-invariant
    than directional relationships. A leg is always thin and vertical,
    a handle is always thin and curved, regardless of viewing angle.

    Returns dict to merge into the primitive's features.
    """
    if primitive.bbox is None:
        return {}

    bx, by, bw, bh = primitive.bbox
    features = {}

    # Orientation: categorical (vertical/horizontal/square) + numeric angle
    if bh > 0 and bw > 0:
        ar = bh / bw
        if ar > 2.0:
            features["orientation"] = "vertical"
        elif ar < 0.5:
            features["orientation"] = "horizontal"
        else:
            features["orientation"] = "square"
        features["elongation"] = round(max(ar, 1/ar), 2)

        # Numeric orientation angle from mask moments (Shepard & Metzler 1971).
        # Principal axis via eigenvalues of the inertia tensor. 0°=horizontal, 90°=vertical.
        # For near-circular shapes (eccentricity < 0.3) angle is noise — skip.
        if primitive.mask is not None:
            try:
                import cv2
                M = cv2.moments(primitive.mask.astype(np.uint8))
                mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
                if mu20 + mu02 > 0:
                    angle_rad = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
                    angle_deg = round(math.degrees(angle_rad) % 180, 1)
                    # Eccentricity from eigenvalues of the covariance matrix
                    # lambda = 0.5 * ((mu20+mu02) ± sqrt((mu20-mu02)^2 + 4*mu11^2))
                    diff_sq = (mu20 - mu02) ** 2
                    cross_sq = 4 * mu11 ** 2
                    discriminant = math.sqrt(diff_sq + cross_sq)
                    lam1 = 0.5 * ((mu20 + mu02) + discriminant)
                    lam2 = 0.5 * ((mu20 + mu02) - discriminant)
                    ecc = math.sqrt(1 - lam2 / max(lam1, 1e-8)) if lam1 > 0 else 0
                    if ecc > 0.3:
                        features["orientation_angle"] = angle_deg
            except Exception:
                pass

    # Position in frame (normalised 0-1)
    features["pos_x"] = round((bx + bw/2) / max(image_w, 1), 3)
    features["pos_y"] = round((by + bh/2) / max(image_h, 1), 3)

    # Relative size and position within parent (if known)
    if parent is not None and parent.bbox is not None:
        px, py, pw, ph = parent.bbox
        if pw > 0 and ph > 0:
            features["relative_size"] = round(primitive.area_fraction / max(parent.area_fraction, 0.001), 3)
            # Where is this part within the parent? (0=top/left, 1=bottom/right)
            features["rel_pos_x"] = round(((bx + bw/2) - px) / pw, 3)
            features["rel_pos_y"] = round(((by + bh/2) - py) / ph, 3)
            # Which region of the parent? (top, bottom, left, right, center)
            rx = features["rel_pos_x"]
            ry = features["rel_pos_y"]
            if rx < 0.33:
                h_region = "left"
            elif rx > 0.67:
                h_region = "right"
            else:
                h_region = "center"
            if ry < 0.33:
                v_region = "top"
            elif ry > 0.67:
                v_region = "bottom"
            else:
                v_region = "middle"
            features["parent_region"] = f"{v_region}-{h_region}"

    # Contour curvature (if mask available)
    if primitive.mask is not None:
        try:
            import cv2
            mask_uint8 = primitive.mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                # Solidity: how much of the convex hull is filled (1.0 = convex, <0.8 = concave/complex)
                features["solidity"] = round(contour_area / max(hull_area, 1), 3)
                # Convexity defects indicate protrusions (handles, legs, branches)
                if len(contour) > 5:
                    defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
                    if defects is not None:
                        # A defect row is [start, end, farthest, depth]; depth is the 4th value.
                        # opencv 4.x returns shape (N,1,4) (d[0][3]); opencv 5.x returns (N,4)
                        # (d[3]) — ravel handles BOTH so the depth read is version-agnostic.
                        significant_defects = sum(1 for d in defects if np.ravel(d)[3] > 1000)
                        features["protrusions"] = significant_defects
        except (ImportError, cv2.error):
            pass

    return features


def _extract_spatial_relations(
    primitives: list[VisualPrimitive],
) -> list[SpatialRelation]:
    """Extract pairwise spatial relationships between segments.

    Prioritises structural relationships (part_of, contains, adjacent_to)
    over directional ones (above, below, left_of, right_of).

    Structural relationships are more view-invariant: a handle is always
    part_of a cup regardless of viewing angle. Directional relationships
    change with perspective and are stored as secondary information.

    Uses mask overlap when masks are available, falls back to
    bounding box analysis for edge-detected contours.
    """
    shapes = [p for p in primitives if p.node_type == "shape" and p.bbox is not None]
    relations = []

    for i, a in enumerate(shapes):
        ax, ay, aw, ah = a.bbox
        a_cx, a_cy = ax + aw / 2, ay + ah / 2

        for b in shapes[i + 1:]:
            bx, by, bw, bh = b.bbox
            b_cx, b_cy = bx + bw / 2, by + bh / 2

            # ── Containment / part_of (highest priority) ─────
            # Use mask overlap if available (more accurate than bbox)
            if a.mask is not None and b.mask is not None:
                overlap = (a.mask & b.mask).sum()
                a_pixels = a.mask.sum()
                b_pixels = b.mask.sum()

                if b_pixels > 0 and overlap / b_pixels > 0.7:
                    # b is mostly inside a → b is part_of a
                    relations.append(SpatialRelation(a.label, b.label, "contains", confidence=round(overlap / b_pixels, 2)))
                    relations.append(SpatialRelation(b.label, a.label, "part_of", confidence=round(overlap / b_pixels, 2)))
                    continue
                elif a_pixels > 0 and overlap / a_pixels > 0.7:
                    relations.append(SpatialRelation(b.label, a.label, "contains", confidence=round(overlap / a_pixels, 2)))
                    relations.append(SpatialRelation(a.label, b.label, "part_of", confidence=round(overlap / a_pixels, 2)))
                    continue
                elif overlap > 0 and overlap / min(a_pixels, b_pixels) > 0.3:
                    relations.append(SpatialRelation(a.label, b.label, "overlaps"))
                    continue
            else:
                # Bbox fallback
                if (ax <= bx and ay <= by and ax + aw >= bx + bw and ay + ah >= by + bh):
                    relations.append(SpatialRelation(a.label, b.label, "contains"))
                    relations.append(SpatialRelation(b.label, a.label, "part_of"))
                    continue
                if (bx <= ax and by <= ay and bx + bw >= ax + aw and by + bh >= ay + ah):
                    relations.append(SpatialRelation(b.label, a.label, "contains"))
                    relations.append(SpatialRelation(a.label, b.label, "part_of"))
                    continue

            # ── Adjacency (second priority) ──────────────────
            gap_x = max(0, max(ax, bx) - min(ax + aw, bx + bw))
            gap_y = max(0, max(ay, by) - min(ay + ah, by + bh))
            gap = math.sqrt(gap_x ** 2 + gap_y ** 2)
            avg_size = (aw + ah + bw + bh) / 4

            if gap < avg_size * 0.3:
                relations.append(SpatialRelation(a.label, b.label, "adjacent_to"))

            # ── Directional (lowest priority, still useful) ──
            dx = b_cx - a_cx
            dy = b_cy - a_cy
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < avg_size * 0.3:
                if gap == 0:
                    relations.append(SpatialRelation(a.label, b.label, "overlaps"))
            elif abs(dx) > abs(dy):
                relations.append(SpatialRelation(
                    a.label, b.label, "left_of" if dx > 0 else "right_of"))
            else:
                relations.append(SpatialRelation(
                    a.label, b.label, "above" if dy > 0 else "below"))

    return relations


# ── Dual-stream encoder (singleton) ──────────────────────

_dual_stream_encoder = None


def _get_dual_stream_encoder():
    """Lazy-load the dual-stream visual encoder."""
    global _dual_stream_encoder
    if _dual_stream_encoder is not None:
        return _dual_stream_encoder

    from nmem_sym_sensor.dual_stream_encoder import DualStreamEncoder
    _dual_stream_encoder = DualStreamEncoder(
        jepa_onnx_path=getattr(config, 'JEPA_V2_ONNX', None),
        jepa_torch_path=getattr(config, 'JEPA_V2_CHECKPOINT', None),
    )
    log.info(
        "Dual-stream encoder ready (JEPA: %s)",
        "active" if _dual_stream_encoder.has_jepa else "geometric-only",
    )
    return _dual_stream_encoder


def current_visual_embedder_id() -> str | None:
    """The vector-space tag (``embedder_id``) of the dual-stream VISUAL encoder currently in use.

    Every stored visual vector is tagged with this so similarity is ONLY ever compared WITHIN one
    space — a JEPA-on run and a geometric-only fallback both emit 512-vecs that are NOT comparable
    (the tag differs: ``dualstream_geo320_jepa192v2`` vs ``dualstream_geo320_nojepa``). Best-effort:
    returns None if the encoder can't be resolved, in which case the row is left untagged (legacy)."""
    try:
        return _get_dual_stream_encoder().embedder_id
    except Exception:  # noqa: BLE001
        return None


# ── Feature vector construction ──────────────────────────

def _build_visual_embedding(
    shape: str,
    color_features: dict,
    texture_features: dict,
    area_fraction: float,
    aspect_ratio: float,
    image_crop: np.ndarray | None = None,
) -> list[float]:
    """Build a 512-dim visual embedding using the dual-stream encoder.

    Uses the DualStreamEncoder (geometric 320-dim + JEPA 192-dim).
    If no image crop is available, falls back to a sparse hand-crafted vector.

    Args:
        shape: Shape name from contour analysis.
        color_features: HSV color dict.
        texture_features: Variance/edge density dict.
        area_fraction: Segment area as fraction of image.
        aspect_ratio: Bounding box aspect ratio.
        image_crop: Raw image crop of the segment (HWC uint8 BGR).

    Returns:
        List of floats (512,).
    """
    if image_crop is not None:
        encoder = _get_dual_stream_encoder()
        embedding = encoder.encode(image_crop)
        return embedding.tolist()

    # Sparse fallback when no image crop is available
    dim = config.VISUAL_FEATURE_DIM
    vec = np.zeros(dim, dtype=np.float32)

    shape_map = {
        "circle": 0, "oval": 1, "triangle": 2, "square": 3,
        "rectangle": 4, "pentagon": 5, "hexagon": 6, "irregular": 7,
        "point": 8,
    }
    idx = shape_map.get(shape, 8)
    vec[idx] = 1.0

    h = color_features.get("hue", 0) / 360.0
    s = color_features.get("saturation", 0)
    v = color_features.get("value", 0)
    vec[16] = math.cos(2 * math.pi * h)
    vec[17] = math.sin(2 * math.pi * h)
    vec[18] = s
    vec[19] = v
    vec[20] = math.cos(4 * math.pi * h)
    vec[21] = math.sin(4 * math.pi * h)

    vec[32] = min(texture_features.get("variance", 0) / 5000.0, 1.0)
    vec[33] = texture_features.get("edge_density", 0)

    vec[40] = min(area_fraction, 1.0)
    vec[41] = min(aspect_ratio / 5.0, 1.0)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


# ── Edge backend ─────────────────────────────────────────

def analyze_frame_edge(
    image: np.ndarray,
    frame_id: str = "frame_0",
) -> FrameAnalysis:
    """Analyze an image using Canny edge detection + contour analysis.

    This is the CPU-only backend. No deep learning models required.
    Produces shape, color, and texture primitives from detected contours.

    Args:
        image: HWC uint8 BGR image (OpenCV format) or RGB.
        frame_id: Identifier for this frame.

    Returns:
        FrameAnalysis with extracted primitives and spatial relations.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) required for edge backend. "
            "Install with: pip install opencv-python-headless"
        )

    h, w = image.shape[:2]
    total_area = h * w
    # Raise minimum area — small fragments are noise, not objects
    min_area = total_area * max(config.VISUAL_MIN_SEGMENT_AREA, 0.02)

    # Frame center for attention weighting
    cx, cy = w / 2, h / 2

    # Convert to grayscale for edge detection
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Dilate to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── Saliency-based attention filtering ───────────────
    # Score each contour by: size, centrality, color saturation.
    # Only process the most salient segments. This mimics foveal
    # attention — the visual system focuses on what matters and
    # ignores background noise.
    scored_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)

        # Size score: larger segments are more likely to be objects (0-1)
        size_score = min(area / total_area * 10, 1.0)

        # Centrality score: how close to frame center (0-1)
        # Objects of interest are usually centered in educational videos
        seg_cx = bx + bw / 2
        seg_cy = by + bh / 2
        dist_from_center = math.sqrt((seg_cx - cx) ** 2 + (seg_cy - cy) ** 2)
        max_dist = math.sqrt(cx ** 2 + cy ** 2)
        centrality_score = 1.0 - (dist_from_center / max_dist)

        # Color saturation score: vivid colors are more interesting than gray
        mask_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_temp, [contour], 0, 255, -1)
        mean_color = cv2.mean(image, mask=mask_temp)[:3]
        _, color_feats = _quantize_color(np.array(mean_color))
        saturation = color_feats.get("saturation", 0)
        color_score = saturation  # 0 = gray, 1 = vivid

        # Combined saliency: weighted sum
        saliency = (size_score * 0.4) + (centrality_score * 0.35) + (color_score * 0.25)

        scored_contours.append((contour, saliency, mean_color))

    # Sort by saliency, take top segments
    scored_contours.sort(key=lambda x: x[1], reverse=True)
    # Limit to top 8 most salient (reduces noise dramatically)
    scored_contours = scored_contours[:min(config.VISUAL_MAX_SEGMENTS, 8)]

    primitives: list[VisualPrimitive] = []

    for contour, saliency, mean_color in scored_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        area_frac = area / total_area

        # Bounding box
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

        # Shape classification
        shape_name = _classify_shape(contour, area, perimeter)

        # Color (already computed during saliency scoring)
        color_name, color_features = _quantize_color(np.array(mean_color))

        # Texture: from masked region
        roi = image[by:by + bh, bx:bx + bw]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        roi_mask = mask[by:by + bh, bx:bx + bw]
        masked_roi = roi.copy()
        masked_roi[roi_mask == 0] = 0
        texture_name, texture_features = _compute_texture(masked_roi)

        # Extract segment crop for learned encoder
        segment_crop = image[by:by + bh, bx:bx + bw].copy()

        # Build visual embedding
        embedding = _build_visual_embedding(
            shape_name, color_features, texture_features,
            area_frac, aspect_ratio,
            image_crop=segment_crop,
        )

        # Auto-generate descriptive label
        label = f"{shape_name}-{color_name}-{texture_name}"

        # Shape primitive
        mask_bool = mask.astype(bool)
        shape_prim = VisualPrimitive(
            label=label,
            node_type="shape",
            features={
                "shape": shape_name,
                "area_fraction": round(area_frac, 4),
                "aspect_ratio": round(aspect_ratio, 3),
                "perimeter": round(perimeter, 1),
                "circularity": round(4 * math.pi * area / max(perimeter ** 2, 1), 3),
                "saliency": round(saliency, 3),
            },
            embedding=embedding,
            bbox=(bx, by, bw, bh),
            mask=mask_bool,
            area_fraction=area_frac,
        )
        # Add orientation, elongation, position features
        part_features = _compute_part_features(shape_prim, None, h, w)
        shape_prim.features.update(part_features)
        primitives.append(shape_prim)

        # Color primitive — only emit for segments with meaningful color
        # Gray/white/black with low saturation in small segments = background
        if not (color_name in ("gray", "white", "black") and
                color_features.get("saturation", 0) < 0.1 and
                area_frac < 0.05):
            primitives.append(VisualPrimitive(
                label=color_name,
                node_type="color",
                features=color_features,
                embedding=embedding,
                bbox=(bx, by, bw, bh),
                area_fraction=area_frac,
            ))

        # Texture primitive — skip non-discriminative textures
        # "noisy" and "matte" are too generic to be useful (they appear everywhere)
        if texture_name not in ("smooth", "noisy", "matte"):
            primitives.append(VisualPrimitive(
                label=texture_name,
                node_type="texture",
                features=texture_features,
                embedding=embedding,
                bbox=(bx, by, bw, bh),
                area_fraction=area_frac,
            ))

    # ── Color-region fallback ─────────────────────────────
    # When edge detection finds nothing (solid color swatches, low-contrast
    # scenes), fall back to dominant-color analysis. A solid red field IS
    # visual information — the edge analyzer just can't see it.
    if not primitives:
        # K-means on pixel colors to find dominant regions
        pixels = image.reshape(-1, 3).astype(np.float32)
        # Simple dominant color: mean of the whole frame
        mean_bgr = pixels.mean(axis=0)
        color_name, color_features = _quantize_color(mean_bgr)

        # Texture from full frame
        texture_name, texture_features = _compute_texture(image)

        # Build embedding for the color field
        embedding = _build_visual_embedding(
            "field", color_features, texture_features,
            area_fraction=1.0, aspect_ratio=1.0,
            image_crop=image,
        )

        label = f"field-{color_name}-{texture_name}"
        primitives.append(VisualPrimitive(
            label=label,
            node_type="color",
            features={
                **color_features,
                "area_fraction": 1.0,
                "fallback": "color_region",
            },
            embedding=embedding,
            bbox=(0, 0, w, h),
            area_fraction=1.0,
        ))

        # Also emit as a standalone color primitive
        primitives.append(VisualPrimitive(
            label=color_name,
            node_type="color",
            features=color_features,
            embedding=embedding,
            bbox=(0, 0, w, h),
            area_fraction=1.0,
        ))

    # Extract spatial relationships
    relations = _extract_spatial_relations(primitives)

    return FrameAnalysis(
        frame_id=frame_id,
        primitives=primitives,
        relations=relations,
        width=w,
        height=h,
    )




# ── Public interface ─────────────────────────────────────

def analyze_frame(
    image: np.ndarray,
    frame_id: str = "frame_0",
) -> FrameAnalysis:
    """Analyze an image frame and extract visual primitives.

    Uses Canny edge detection + contour analysis. No external models.

    Args:
        image: HWC uint8 BGR image (OpenCV format).
        frame_id: Identifier for this frame.
        backend: Reserved for future use. Only "edge" is supported.

    Returns:
        FrameAnalysis with primitives and spatial relations.
    """
    return analyze_frame_edge(image, frame_id)
