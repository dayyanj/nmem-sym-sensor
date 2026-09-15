#!/usr/bin/env python3
"""Generate diverse geometric primitives for diffusion decoder training.

Produces 64×64 images with high variety:
- Shapes: triangle, circle, rectangle, pentagon, hexagon, star, cross,
  semicircle, arc, ring, ellipse, parallelogram, trapezoid, arrow,
  crescent, wave, spiral, bezier curves, polylines
- Variations: solid/outline, partial/complete, thick/thin strokes,
  rotated, skewed, scaled, translated
- Compositions: single shapes, overlapping, nested, scattered
- Colors: random fills, gradients, multiple colors per image
- Backgrounds: solid, gradient, noisy

Output: directory of PNG images suitable for diffusion decoder training.

Usage:
    python scripts/generate_training_primitives.py \
        --output-dir training_data/primitives \
        --count 5000
"""
import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

SIZE = 128  # Output image size (matches MoE encoder input)


def random_color():
    """Random RGB color, biased toward saturated colors."""
    if random.random() < 0.3:
        # Pure-ish colors
        channels = [random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)]
        channels[random.randint(0, 2)] = random.randint(180, 255)
        random.shuffle(channels)
        return tuple(channels)
    return tuple(random.randint(20, 255) for _ in range(3))


def random_bg():
    """Random background — solid, gradient, or noisy."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    choice = random.random()

    if choice < 0.5:
        # Solid color
        color = random_color()
        img[:] = color
    elif choice < 0.75:
        # Vertical gradient
        c1 = np.array(random_color(), dtype=np.float32)
        c2 = np.array(random_color(), dtype=np.float32)
        for y in range(SIZE):
            t = y / SIZE
            img[y, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    else:
        # Noisy solid
        base = random_color()
        img[:] = base
        noise = np.random.randint(-20, 20, (SIZE, SIZE, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def random_center_and_size(margin=0.15):
    """Random center and radius within the image."""
    cx = random.uniform(margin, 1 - margin) * SIZE
    cy = random.uniform(margin, 1 - margin) * SIZE
    max_r = min(cx, cy, SIZE - cx, SIZE - cy) * 0.9
    r = random.uniform(max_r * 0.3, max_r)
    return int(cx), int(cy), int(r)


def rotate_points(pts, cx, cy, angle_deg):
    """Rotate points around center."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    result = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        nx = cx + dx * cos_a - dy * sin_a
        ny = cy + dx * sin_a + dy * cos_a
        result.append((int(nx), int(ny)))
    return result


def skew_points(pts, cx, cy, skew_x=0, skew_y=0):
    """Apply shear transform."""
    result = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        nx = cx + dx + skew_x * dy
        ny = cy + dy + skew_y * dx
        result.append((int(nx), int(ny)))
    return result


def polygon_points(cx, cy, r, n_sides, start_angle=0):
    """Generate regular polygon vertices."""
    pts = []
    for i in range(n_sides):
        angle = start_angle + 2 * math.pi * i / n_sides - math.pi / 2
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        pts.append((int(px), int(py)))
    return pts


def star_points(cx, cy, r_outer, r_inner, n_points):
    """Generate star vertices."""
    pts = []
    for i in range(n_points * 2):
        angle = math.pi * i / n_points - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        pts.append((int(px), int(py)))
    return pts


def draw_shape(img, shape_type, filled, color, thickness):
    """Draw a single shape on the image."""
    cx, cy, r = random_center_and_size()
    angle = random.uniform(0, 360)
    skew_x = random.uniform(-0.3, 0.3) if random.random() < 0.3 else 0
    skew_y = random.uniform(-0.3, 0.3) if random.random() < 0.3 else 0

    if shape_type == "circle":
        # Vary eccentricity slightly
        rx = int(r * random.uniform(0.8, 1.2))
        ry = int(r * random.uniform(0.8, 1.2))
        if filled:
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, color, -1)
        else:
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, color, thickness)

    elif shape_type == "arc":
        rx = int(r * random.uniform(0.8, 1.2))
        ry = int(r * random.uniform(0.8, 1.2))
        start = random.uniform(0, 270)
        span = random.uniform(45, 270)
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, start, start + span, color, thickness)

    elif shape_type == "semicircle":
        rx = int(r)
        cv2.ellipse(img, (cx, cy), (rx, rx), angle, 0, 180, color, -1 if filled else thickness)

    elif shape_type == "ring":
        r_outer = int(r)
        r_inner = int(r * random.uniform(0.4, 0.7))
        cv2.circle(img, (cx, cy), r_outer, color, -1 if filled else thickness)
        if filled:
            # Cut out center
            bg_at_center = img[cy, cx].tolist() if 0 <= cy < SIZE and 0 <= cx < SIZE else [0, 0, 0]
            cv2.circle(img, (cx, cy), r_inner, tuple(bg_at_center), -1)

    elif shape_type == "crescent":
        r1 = int(r)
        r2 = int(r * 0.8)
        offset = int(r * 0.3)
        cv2.circle(img, (cx, cy), r1, color, -1)
        # Subtract offset circle in background color
        mask_color = img[max(0, cy-1), max(0, cx-1)].tolist()
        cv2.circle(img, (cx + offset, cy), r2, tuple(mask_color), -1)

    elif shape_type in ("triangle", "pentagon", "hexagon", "heptagon", "octagon"):
        sides = {"triangle": 3, "pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}[shape_type]
        pts = polygon_points(cx, cy, r, sides, math.radians(angle))
        pts = skew_points(pts, cx, cy, skew_x, skew_y)
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts_np], color)
        else:
            cv2.polylines(img, [pts_np], True, color, thickness)

    elif shape_type == "rectangle":
        w = int(r * random.uniform(0.6, 1.8))
        h = int(r * random.uniform(0.6, 1.8))
        pts = [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
        pts = rotate_points(pts, cx, cy, angle)
        pts = skew_points(pts, cx, cy, skew_x, skew_y)
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts_np], color)
        else:
            cv2.polylines(img, [pts_np], True, color, thickness)

    elif shape_type == "parallelogram":
        w = int(r * 1.3)
        h = int(r * 0.7)
        skew = int(r * 0.4)
        pts = [(cx - w + skew, cy - h), (cx + w + skew, cy - h),
               (cx + w - skew, cy + h), (cx - w - skew, cy + h)]
        pts = rotate_points(pts, cx, cy, angle)
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts_np], color)
        else:
            cv2.polylines(img, [pts_np], True, color, thickness)

    elif shape_type == "trapezoid":
        w_top = int(r * 0.6)
        w_bot = int(r * 1.2)
        h = int(r * 0.8)
        pts = [(cx - w_top, cy - h), (cx + w_top, cy - h),
               (cx + w_bot, cy + h), (cx - w_bot, cy + h)]
        pts = rotate_points(pts, cx, cy, angle)
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts_np], color)
        else:
            cv2.polylines(img, [pts_np], True, color, thickness)

    elif shape_type == "star":
        n = random.choice([4, 5, 6, 8])
        r_inner = r * random.uniform(0.3, 0.5)
        pts = star_points(cx, cy, r, r_inner, n)
        pts = rotate_points(pts, cx, cy, angle)
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts_np], color)
        else:
            cv2.polylines(img, [pts_np], True, color, thickness)

    elif shape_type == "cross":
        arm_w = int(r * random.uniform(0.2, 0.4))
        pts = [
            (cx - arm_w, cy - r), (cx + arm_w, cy - r),
            (cx + arm_w, cy - arm_w), (cx + r, cy - arm_w),
            (cx + r, cy + arm_w), (cx + arm_w, cy + arm_w),
            (cx + arm_w, cy + r), (cx - arm_w, cy + r),
            (cx - arm_w, cy + arm_w), (cx - r, cy + arm_w),
            (cx - r, cy - arm_w), (cx - arm_w, cy - arm_w),
        ]
        pts = rotate_points(pts, cx, cy, angle)
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts_np], color)
        else:
            cv2.polylines(img, [pts_np], True, color, thickness)

    elif shape_type == "arrow":
        head_w = int(r * 0.6)
        shaft_w = int(r * 0.2)
        pts = [
            (cx, cy - r),  # tip
            (cx + head_w, cy), (cx + shaft_w, cy),
            (cx + shaft_w, cy + r), (cx - shaft_w, cy + r),
            (cx - shaft_w, cy), (cx - head_w, cy),
        ]
        pts = rotate_points(pts, cx, cy, angle)
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts_np], color)
        else:
            cv2.polylines(img, [pts_np], True, color, thickness)

    elif shape_type == "bezier":
        # Random bezier curve
        n_pts = random.randint(3, 6)
        ctrl_pts = [(random.randint(5, SIZE - 5), random.randint(5, SIZE - 5)) for _ in range(n_pts)]
        # Draw as polyline approximation
        t_vals = np.linspace(0, 1, 50)
        curve_pts = []
        for t in t_vals:
            pts_arr = np.array(ctrl_pts, dtype=np.float64)
            while len(pts_arr) > 1:
                pts_arr = (1 - t) * pts_arr[:-1] + t * pts_arr[1:]
            curve_pts.append(tuple(pts_arr[0].astype(int)))
        curve_np = np.array(curve_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [curve_np], False, color, thickness)

    elif shape_type == "wave":
        # Sine wave
        amplitude = random.uniform(5, 20)
        freq = random.uniform(1, 4)
        y_center = cy
        pts = []
        for x in range(5, SIZE - 5):
            y = int(y_center + amplitude * math.sin(freq * (x - 5) / SIZE * 2 * math.pi))
            pts.append((x, y))
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_np], False, color, thickness)

    elif shape_type == "spiral":
        pts = []
        turns = random.uniform(1.5, 4)
        for i in range(200):
            t = i / 200 * turns * 2 * math.pi
            sr = (r * i / 200)
            px = int(cx + sr * math.cos(t))
            py = int(cy + sr * math.sin(t))
            pts.append((px, py))
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_np], False, color, thickness)

    elif shape_type == "line_segment":
        angle_rad = math.radians(angle)
        x1 = int(cx - r * math.cos(angle_rad))
        y1 = int(cy - r * math.sin(angle_rad))
        x2 = int(cx + r * math.cos(angle_rad))
        y2 = int(cy + r * math.sin(angle_rad))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    elif shape_type == "polyline":
        n_pts = random.randint(3, 8)
        pts = [(random.randint(5, SIZE - 5), random.randint(5, SIZE - 5)) for _ in range(n_pts)]
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_np], False, color, thickness)

    elif shape_type == "dot_cluster":
        n_dots = random.randint(3, 15)
        dot_r = random.randint(1, 4)
        for _ in range(n_dots):
            dx = int(cx + random.gauss(0, r * 0.5))
            dy = int(cy + random.gauss(0, r * 0.5))
            cv2.circle(img, (dx, dy), dot_r, color, -1)

    elif shape_type == "point":
        pr = random.randint(1, 3)
        cv2.circle(img, (cx, cy), pr, color, -1)

    elif shape_type == "ellipse":
        rx = int(r * random.uniform(0.4, 1.0))
        ry = int(r * random.uniform(0.4, 1.0))
        if filled:
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, color, -1)
        else:
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, color, thickness)

    elif shape_type == "quad":
        # Irregular quadrilateral
        pts = [(cx + random.randint(-r, r), cy + random.randint(-r, r)) for _ in range(4)]
        # Sort by angle to make convex-ish
        pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts_np], color)
        else:
            cv2.polylines(img, [pts_np], True, color, thickness)

    # ── 3D primitive projections (2D views of 3D shapes) ──

    elif shape_type == "cube_projection":
        # Isometric cube projection
        s = int(r * 0.7)
        dx = int(s * 0.5)
        dy = int(s * 0.3)
        # Front face
        front = [(cx - s, cy), (cx, cy - dy), (cx + s, cy), (cx, cy + dy)]
        # Top face
        top = [(cx - s, cy), (cx, cy - dy), (cx, cy - dy - s), (cx - s, cy - s)]
        # Right face
        right = [(cx, cy - dy), (cx + s, cy), (cx + s, cy - s), (cx, cy - dy - s)]
        front = rotate_points(front, cx, cy, angle)
        top = rotate_points(top, cx, cy, angle)
        right = rotate_points(right, cx, cy, angle)
        c2_color = tuple(max(0, c - 40) for c in color)
        c3_color = tuple(max(0, c - 80) for c in color)
        cv2.fillPoly(img, [np.array(front, dtype=np.int32)], color)
        cv2.fillPoly(img, [np.array(top, dtype=np.int32)], c2_color)
        cv2.fillPoly(img, [np.array(right, dtype=np.int32)], c3_color)
        if not filled:
            for face in [front, top, right]:
                cv2.polylines(img, [np.array(face, dtype=np.int32)], True, color, 1)

    elif shape_type == "sphere_projection":
        # Sphere = circle with shading gradient
        cv2.circle(img, (cx, cy), int(r), color, -1)
        # Highlight
        hx, hy = cx - int(r * 0.3), cy - int(r * 0.3)
        highlight = tuple(min(255, c + 80) for c in color)
        cv2.circle(img, (hx, hy), int(r * 0.3), highlight, -1)
        # Edge darkening
        for ring in range(3):
            ring_r = int(r * (0.85 + ring * 0.05))
            dark = tuple(max(0, c - 20 * (ring + 1)) for c in color)
            cv2.circle(img, (cx, cy), ring_r, dark, 1)

    elif shape_type == "cylinder_projection":
        # Cylinder = rectangle with ellipse caps
        w = int(r * 0.6)
        h = int(r)
        cap_h = int(r * 0.2)
        pts_body = [(cx - w, cy - h + cap_h), (cx + w, cy - h + cap_h),
                     (cx + w, cy + h - cap_h), (cx - w, cy + h - cap_h)]
        pts_body = rotate_points(pts_body, cx, cy, angle)
        cv2.fillPoly(img, [np.array(pts_body, dtype=np.int32)], color)
        # Top ellipse
        cv2.ellipse(img, (cx, cy - h + cap_h), (w, cap_h), 0, 0, 360,
                    tuple(min(255, c + 30) for c in color), -1)
        # Bottom ellipse (half visible)
        cv2.ellipse(img, (cx, cy + h - cap_h), (w, cap_h), 0, 0, 180,
                    tuple(max(0, c - 30) for c in color), -1)

    elif shape_type == "cone_projection":
        # Cone = triangle with ellipse base
        tip_y = cy - int(r)
        base_y = cy + int(r * 0.7)
        w = int(r * 0.6)
        cap_h = int(r * 0.15)
        pts = [(cx, tip_y), (cx + w, base_y), (cx - w, base_y)]
        pts = rotate_points(pts, cx, cy, angle)
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)
        cv2.ellipse(img, (cx, base_y), (w, cap_h), 0, 0, 360,
                    tuple(max(0, c - 30) for c in color), -1)

    elif shape_type == "torus_projection":
        # Torus = ring with shading
        r_outer = int(r)
        r_inner = int(r * 0.5)
        cv2.circle(img, (cx, cy), r_outer, color, -1)
        bg_sample = img[max(0, cy - r_outer - 2), max(0, cx)].tolist()
        cv2.circle(img, (cx, cy), r_inner, tuple(bg_sample), -1)
        # Inner shadow
        cv2.ellipse(img, (cx, cy), (r_inner + 2, r_inner + 2), 0, 200, 340,
                    tuple(max(0, c - 50) for c in color), 2)

    elif shape_type == "pyramid_projection":
        # Pyramid from slightly above
        s = int(r * 0.8)
        tip = (cx, cy - int(r))
        bl = (cx - s, cy + int(r * 0.4))
        br = (cx + s, cy + int(r * 0.4))
        bm = (cx + int(s * 0.3), cy + int(r * 0.6))
        # Front face
        cv2.fillPoly(img, [np.array([tip, bl, br], dtype=np.int32)], color)
        # Side face (darker)
        darker = tuple(max(0, c - 60) for c in color)
        cv2.fillPoly(img, [np.array([tip, br, bm], dtype=np.int32)], darker)
        # Edges
        for p1, p2 in [(tip, bl), (tip, br), (tip, bm), (bl, br), (br, bm)]:
            cv2.line(img, p1, p2, (0, 0, 0), 1)


ALL_SHAPES = [
    # 2D primitives
    "point", "line_segment", "polyline",
    "circle", "ellipse", "arc", "semicircle", "ring", "crescent",
    "triangle", "quad", "rectangle", "parallelogram", "trapezoid",
    "pentagon", "hexagon", "heptagon", "octagon",
    "star", "cross", "arrow",
    "bezier", "wave", "spiral", "dot_cluster",
    # 3D primitive projections (2D views of 3D objects)
    "cube_projection", "sphere_projection", "cylinder_projection",
    "cone_projection", "torus_projection", "pyramid_projection",
]


def generate_single(shape_type=None):
    """Generate a single shape on a background."""
    img = random_bg()
    shape = shape_type or random.choice(ALL_SHAPES)
    filled = random.random() < 0.6
    color = random_color()
    thickness = random.choice([1, 1, 2, 2, 3])
    draw_shape(img, shape, filled, color, thickness)
    return img


def generate_composition():
    """Generate 2-4 overlapping shapes."""
    img = random_bg()
    n_shapes = random.randint(2, 4)
    for _ in range(n_shapes):
        shape = random.choice(ALL_SHAPES)
        filled = random.random() < 0.5
        color = random_color()
        thickness = random.choice([1, 2, 2, 3])
        draw_shape(img, shape, filled, color, thickness)
    return img


def generate_partial():
    """Generate a shape that's partially cropped by the frame edge."""
    img = random_bg()
    shape = random.choice(ALL_SHAPES[:14])  # stick to closed shapes
    filled = random.random() < 0.6
    color = random_color()
    thickness = random.choice([1, 2, 3])

    # Draw on a larger canvas, then crop
    big = np.zeros((SIZE * 3, SIZE * 3, 3), dtype=np.uint8)
    big[:] = img[0, 0]  # same background

    # Offset so shape is partially visible
    ox = random.randint(-SIZE, SIZE * 2)
    oy = random.randint(-SIZE, SIZE * 2)
    cx, cy = SIZE + ox, SIZE + oy
    r = random.randint(SIZE // 3, SIZE)

    # Draw on big canvas at the offset position
    temp_shape = random.choice(ALL_SHAPES[:14])
    if temp_shape in ("circle", "arc", "semicircle"):
        if filled:
            cv2.circle(big, (cx, cy), r, color, -1)
        else:
            cv2.circle(big, (cx, cy), r, color, thickness)
    else:
        sides = {"triangle": 3, "rectangle": 4, "pentagon": 5, "hexagon": 6,
                 "heptagon": 7, "octagon": 8, "star": 5}.get(temp_shape, 4)
        pts = polygon_points(cx, cy, r, sides)
        pts_np = np.array(pts, dtype=np.int32)
        if filled:
            cv2.fillPoly(big, [pts_np], color)
        else:
            cv2.polylines(big, [pts_np], True, color, thickness)

    # Crop center to get partial view
    img = big[SIZE:SIZE * 2, SIZE:SIZE * 2]
    return img


def generate_nested():
    """Generate concentric/nested shapes."""
    img = random_bg()
    cx, cy = SIZE // 2 + random.randint(-10, 10), SIZE // 2 + random.randint(-10, 10)
    n_layers = random.randint(2, 4)
    max_r = random.randint(15, 28)

    for i in range(n_layers):
        r = max_r - i * (max_r // (n_layers + 1))
        if r < 3:
            break
        shape = random.choice(["circle", "triangle", "rectangle", "pentagon", "hexagon"])
        color = random_color()
        filled = (i == n_layers - 1)  # only innermost filled
        thickness = random.choice([1, 2])

        if shape == "circle":
            if filled:
                cv2.circle(img, (cx, cy), r, color, -1)
            else:
                cv2.circle(img, (cx, cy), r, color, thickness)
        else:
            sides = {"triangle": 3, "rectangle": 4, "pentagon": 5, "hexagon": 6}[shape]
            angle = random.uniform(0, 360)
            pts = polygon_points(cx, cy, r, sides, math.radians(angle))
            pts_np = np.array(pts, dtype=np.int32)
            if filled:
                cv2.fillPoly(img, [pts_np], color)
            else:
                cv2.polylines(img, [pts_np], True, color, thickness)

    return img


def generate_solid_color():
    """Pure solid color fill — like our color training videos."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[:] = random_color()
    return img


def generate_gradient():
    """Color gradient fill."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    c1 = np.array(random_color(), dtype=np.float32)
    c2 = np.array(random_color(), dtype=np.float32)
    angle = random.uniform(0, math.pi)

    for y in range(SIZE):
        for x in range(SIZE):
            t = (x * math.cos(angle) + y * math.sin(angle)) / (SIZE * 1.4) + 0.5
            t = max(0, min(1, t))
            img[y, x] = c1 * (1 - t) + c2 * t

    return img.astype(np.uint8)


def generate_texture():
    """Random texture patterns — stripes, checkerboard, dots."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    c1 = random_color()
    c2 = random_color()

    pattern = random.choice(["stripes", "checkerboard", "dots", "crosshatch"])

    if pattern == "stripes":
        freq = random.randint(3, 12)
        angle = random.uniform(0, math.pi)
        for y in range(SIZE):
            for x in range(SIZE):
                v = (x * math.cos(angle) + y * math.sin(angle)) / SIZE * freq
                img[y, x] = c1 if int(v) % 2 == 0 else c2

    elif pattern == "checkerboard":
        sq = random.randint(4, 16)
        for y in range(SIZE):
            for x in range(SIZE):
                img[y, x] = c1 if ((x // sq) + (y // sq)) % 2 == 0 else c2

    elif pattern == "dots":
        img[:] = c1
        spacing = random.randint(6, 14)
        dot_r = random.randint(1, 3)
        for y in range(spacing // 2, SIZE, spacing):
            for x in range(spacing // 2, SIZE, spacing):
                cv2.circle(img, (x, y), dot_r, c2, -1)

    elif pattern == "crosshatch":
        img[:] = c1
        spacing = random.randint(4, 10)
        for i in range(-SIZE, SIZE * 2, spacing):
            cv2.line(img, (i, 0), (i + SIZE, SIZE), c2, 1)
            cv2.line(img, (i, SIZE), (i + SIZE, 0), c2, 1)

    return img


def generate_joined_pair():
    """Two shapes touching/sharing an edge — the bridge to objects."""
    img = random_bg()
    pair_type = random.choice([
        "edge_shared", "tangent", "stacked", "side_by_side", "t_junction",
    ])

    c1 = random_color()
    c2 = random_color()
    filled1 = random.random() < 0.7
    filled2 = random.random() < 0.7
    t1 = -1 if filled1 else random.randint(1, 3)
    t2 = -1 if filled2 else random.randint(1, 3)

    cx, cy = SIZE // 2, SIZE // 2
    s = random.randint(20, 45)

    if pair_type == "edge_shared":
        # Two triangles sharing an edge → diamond/bowtie
        if random.random() < 0.5:
            # Diamond: triangles pointing up and down
            pts1 = np.array([[cx, cy - s], [cx - s, cy], [cx + s, cy]])
            pts2 = np.array([[cx, cy + s], [cx - s, cy], [cx + s, cy]])
        else:
            # Bowtie: triangles pointing left and right
            pts1 = np.array([[cx - s, cy - s], [cx - s, cy + s], [cx, cy]])
            pts2 = np.array([[cx + s, cy - s], [cx + s, cy + s], [cx, cy]])
        if filled1:
            cv2.fillPoly(img, [pts1], c1)
        else:
            cv2.polylines(img, [pts1], True, c1, t1)
        if filled2:
            cv2.fillPoly(img, [pts2], c2)
        else:
            cv2.polylines(img, [pts2], True, c2, t2)

    elif pair_type == "tangent":
        # Circle tangent to rectangle
        rect_w = random.randint(30, 60)
        rect_h = random.randint(20, 40)
        rx1 = cx - rect_w // 2
        ry1 = cy
        cv2.rectangle(img, (rx1, ry1), (rx1 + rect_w, ry1 + rect_h), c1, t1)
        # Circle sits on top
        r = random.randint(12, 25)
        cv2.circle(img, (cx, ry1 - r), r, c2, t2)

    elif pair_type == "stacked":
        # House: rectangle with triangle roof
        w = random.randint(30, 55)
        h = random.randint(25, 45)
        rx1 = cx - w // 2
        ry1 = cy
        cv2.rectangle(img, (rx1, ry1), (rx1 + w, ry1 + h), c1, t1)
        # Triangle roof touching the top edge
        roof_h = random.randint(15, 30)
        pts = np.array([[cx, ry1 - roof_h], [rx1, ry1], [rx1 + w, ry1]])
        if filled2:
            cv2.fillPoly(img, [pts], c2)
        else:
            cv2.polylines(img, [pts], True, c2, t2)

    elif pair_type == "side_by_side":
        # Two shapes adjacent, touching at one side
        gap = random.randint(0, 3)  # 0 = touching, 1-3 = small gap
        s1 = random.choice(["rect", "circle", "triangle"])
        s2 = random.choice(["rect", "circle", "triangle"])
        half = SIZE // 2

        # Left shape
        if s1 == "rect":
            w1 = random.randint(20, half - gap)
            h1 = random.randint(30, 60)
            cv2.rectangle(img, (half - w1 - gap, cy - h1//2), (half - gap, cy + h1//2), c1, t1)
        elif s1 == "circle":
            r1 = random.randint(15, half - gap - 5)
            cv2.circle(img, (half - gap - r1, cy), r1, c1, t1)
        else:
            s_sz = random.randint(15, half - gap - 5)
            pts = np.array([[half - gap - s_sz, cy + s_sz], [half - gap, cy + s_sz], [half - gap - s_sz//2, cy - s_sz]])
            if filled1: cv2.fillPoly(img, [pts], c1)
            else: cv2.polylines(img, [pts], True, c1, t1)

        # Right shape
        if s2 == "rect":
            w2 = random.randint(20, half - gap)
            h2 = random.randint(30, 60)
            cv2.rectangle(img, (half + gap, cy - h2//2), (half + gap + w2, cy + h2//2), c2, t2)
        elif s2 == "circle":
            r2 = random.randint(15, half - gap - 5)
            cv2.circle(img, (half + gap + r2, cy), r2, c2, t2)
        else:
            s_sz = random.randint(15, half - gap - 5)
            pts = np.array([[half + gap, cy + s_sz], [half + gap + s_sz, cy + s_sz], [half + gap + s_sz//2, cy - s_sz]])
            if filled2: cv2.fillPoly(img, [pts], c2)
            else: cv2.polylines(img, [pts], True, c2, t2)

    elif pair_type == "t_junction":
        # Horizontal bar with vertical bar meeting it (T or L shape)
        bar_w = random.randint(50, 90)
        bar_h = random.randint(8, 16)
        vert_w = random.randint(8, 16)
        vert_h = random.randint(30, 55)
        # Horizontal
        cv2.rectangle(img, (cx - bar_w//2, cy - bar_h//2), (cx + bar_w//2, cy + bar_h//2), c1, t1)
        # Vertical meeting at random point along horizontal
        vx = cx + random.randint(-bar_w//3, bar_w//3)
        if random.random() < 0.5:
            # T shape: vertical goes down
            cv2.rectangle(img, (vx - vert_w//2, cy + bar_h//2), (vx + vert_w//2, cy + bar_h//2 + vert_h), c2, t2)
        else:
            # Inverted T: vertical goes up
            cv2.rectangle(img, (vx - vert_w//2, cy - bar_h//2 - vert_h), (vx + vert_w//2, cy - bar_h//2), c2, t2)

    # Random rotation of entire image
    if random.random() < 0.4:
        angle = random.uniform(-45, 45)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)

    return img


def generate_composite_object():
    """Recognisable composite objects from joined primitives."""
    img = random_bg()
    obj_type = random.choice([
        "house", "arrow", "ice_cream", "tree", "person_stick",
        "car_side", "flower", "key", "flag", "hourglass",
    ])

    c1 = random_color()
    c2 = random_color()
    c3 = random_color()
    cx, cy = SIZE // 2, SIZE // 2

    if obj_type == "house":
        w = random.randint(35, 60)
        h = random.randint(30, 50)
        # Body
        cv2.rectangle(img, (cx-w//2, cy), (cx+w//2, cy+h), c1, -1)
        # Roof
        roof_h = random.randint(20, 35)
        pts = np.array([[cx, cy-roof_h], [cx-w//2-5, cy], [cx+w//2+5, cy]])
        cv2.fillPoly(img, [pts], c2)
        # Door
        dw, dh = w//4, h//2
        cv2.rectangle(img, (cx-dw//2, cy+h-dh), (cx+dw//2, cy+h), c3, -1)

    elif obj_type == "arrow":
        # Shaft + head
        shaft_w = random.randint(6, 14)
        shaft_l = random.randint(30, 60)
        head_w = random.randint(20, 35)
        head_l = random.randint(15, 25)
        # Shaft
        cv2.rectangle(img, (cx-shaft_w//2, cy), (cx+shaft_w//2, cy+shaft_l), c1, -1)
        # Arrowhead
        pts = np.array([[cx, cy-head_l], [cx-head_w//2, cy], [cx+head_w//2, cy]])
        cv2.fillPoly(img, [pts], c1)

    elif obj_type == "ice_cream":
        # Cone (triangle) + scoop (circle)
        cone_w = random.randint(25, 40)
        cone_h = random.randint(35, 55)
        r = cone_w // 2 + random.randint(2, 8)
        # Cone
        pts = np.array([[cx, cy+cone_h], [cx-cone_w//2, cy], [cx+cone_w//2, cy]])
        cv2.fillPoly(img, [pts], c1)
        # Scoop
        cv2.circle(img, (cx, cy-r+5), r, c2, -1)

    elif obj_type == "tree":
        # Trunk (rect) + canopy (circle or triangle)
        tw = random.randint(8, 16)
        th = random.randint(25, 40)
        cv2.rectangle(img, (cx-tw//2, cy), (cx+tw//2, cy+th), (80, 50, 20), -1)
        if random.random() < 0.5:
            # Round canopy
            r = random.randint(20, 35)
            cv2.circle(img, (cx, cy-r+5), r, c2, -1)
        else:
            # Triangular canopy
            w = random.randint(30, 50)
            h = random.randint(30, 45)
            pts = np.array([[cx, cy-h], [cx-w//2, cy+5], [cx+w//2, cy+5]])
            cv2.fillPoly(img, [pts], c2)

    elif obj_type == "person_stick":
        # Head (circle) + body (line) + arms (lines) + legs (lines)
        head_r = random.randint(8, 14)
        body_l = random.randint(25, 40)
        cv2.circle(img, (cx, cy-head_r-body_l//2), head_r, c1, -1)
        # Body
        neck = cy - body_l//2
        cv2.line(img, (cx, neck), (cx, neck+body_l), c1, 2)
        # Arms
        arm_y = neck + body_l//3
        arm_l = random.randint(15, 25)
        cv2.line(img, (cx, arm_y), (cx-arm_l, arm_y+arm_l//2), c1, 2)
        cv2.line(img, (cx, arm_y), (cx+arm_l, arm_y+arm_l//2), c1, 2)
        # Legs
        foot = neck + body_l
        leg_l = random.randint(15, 25)
        cv2.line(img, (cx, foot), (cx-leg_l//2, foot+leg_l), c1, 2)
        cv2.line(img, (cx, foot), (cx+leg_l//2, foot+leg_l), c1, 2)

    elif obj_type == "car_side":
        # Body rect + roof rect + 2 wheels
        bw = random.randint(50, 75)
        bh = random.randint(18, 28)
        by = cy + 5
        cv2.rectangle(img, (cx-bw//2, by), (cx+bw//2, by+bh), c1, -1)
        # Roof
        rw = bw * 2 // 3
        rh = random.randint(14, 22)
        cv2.rectangle(img, (cx-rw//2, by-rh), (cx+rw//2, by), c2, -1)
        # Wheels
        wr = random.randint(7, 12)
        cv2.circle(img, (cx-bw//3, by+bh), wr, (40, 40, 40), -1)
        cv2.circle(img, (cx+bw//3, by+bh), wr, (40, 40, 40), -1)

    elif obj_type == "flower":
        # Stem + petals around center
        stem_h = random.randint(25, 45)
        cv2.line(img, (cx, cy+5), (cx, cy+5+stem_h), (50, 150, 50), 2)
        # Petals
        r_petal = random.randint(8, 14)
        r_center = random.randint(5, 8)
        n_petals = random.randint(5, 8)
        for i in range(n_petals):
            angle = 2 * math.pi * i / n_petals
            px = int(cx + (r_petal + r_center) * math.cos(angle))
            py = int(cy - (r_petal + r_center) * math.sin(angle))
            cv2.circle(img, (px, py), r_petal, c1, -1)
        cv2.circle(img, (cx, cy), r_center, c2, -1)

    elif obj_type == "key":
        # Circle head + rectangular shaft + teeth
        head_r = random.randint(12, 18)
        shaft_l = random.randint(30, 50)
        shaft_w = random.randint(4, 8)
        cv2.circle(img, (cx - shaft_l//2, cy), head_r, c1, -1)
        cv2.circle(img, (cx - shaft_l//2, cy), head_r//2, random_color(), -1)
        cv2.rectangle(img, (cx - shaft_l//2 + head_r, cy-shaft_w//2),
                      (cx + shaft_l//2, cy+shaft_w//2), c1, -1)
        # Teeth
        n_teeth = random.randint(2, 4)
        tooth_w = 4
        for t in range(n_teeth):
            tx = cx + shaft_l//2 - t * (tooth_w + 3) - 5
            cv2.rectangle(img, (tx, cy+shaft_w//2), (tx+tooth_w, cy+shaft_w//2+8), c1, -1)

    elif obj_type == "flag":
        # Pole + rectangular flag
        pole_h = random.randint(50, 80)
        cv2.line(img, (cx-20, cy+pole_h//2), (cx-20, cy-pole_h//2), (100, 100, 100), 2)
        # Flag
        fw = random.randint(30, 50)
        fh = random.randint(20, 35)
        cv2.rectangle(img, (cx-19, cy-pole_h//2), (cx-19+fw, cy-pole_h//2+fh), c1, -1)

    elif obj_type == "hourglass":
        # Two triangles point-to-point
        s = random.randint(20, 40)
        pts1 = np.array([[cx-s, cy-s], [cx+s, cy-s], [cx, cy]])
        pts2 = np.array([[cx-s, cy+s], [cx+s, cy+s], [cx, cy]])
        cv2.fillPoly(img, [pts1], c1)
        cv2.fillPoly(img, [pts2], c2)

    # Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)

    return img


def generate_chain():
    """Chain of connected shapes — line of circles, zigzag triangles, etc."""
    img = random_bg()
    chain_type = random.choice(["circles", "squares", "triangles_zigzag", "alternating"])
    c1 = random_color()
    c2 = random_color()
    n = random.randint(3, 6)

    if chain_type == "circles":
        r = random.randint(8, 18)
        y = SIZE // 2
        spacing = r * 2 + random.randint(-2, 3)  # touching or slight overlap
        start_x = (SIZE - spacing * (n - 1)) // 2
        for i in range(n):
            cv2.circle(img, (start_x + i * spacing, y), r,
                       c1 if i % 2 == 0 else c2,
                       -1 if random.random() < 0.6 else 2)

    elif chain_type == "squares":
        s = random.randint(12, 22)
        y = SIZE // 2 - s // 2
        spacing = s + random.randint(-3, 2)
        start_x = (SIZE - spacing * (n - 1)) // 2 - s // 2
        for i in range(n):
            x = start_x + i * spacing
            cv2.rectangle(img, (x, y), (x + s, y + s),
                          c1 if i % 2 == 0 else c2, -1)

    elif chain_type == "triangles_zigzag":
        s = random.randint(15, 25)
        spacing = s + random.randint(0, 5)
        start_x = (SIZE - spacing * (n - 1)) // 2
        for i in range(n):
            x = start_x + i * spacing
            if i % 2 == 0:
                pts = np.array([[x, SIZE//2 - s], [x - s//2, SIZE//2 + s//2], [x + s//2, SIZE//2 + s//2]])
            else:
                pts = np.array([[x, SIZE//2 + s], [x - s//2, SIZE//2 - s//2], [x + s//2, SIZE//2 - s//2]])
            cv2.fillPoly(img, [pts], c1 if i % 2 == 0 else c2)

    elif chain_type == "alternating":
        s = random.randint(10, 18)
        spacing = s * 2 + random.randint(0, 4)
        start_x = (SIZE - spacing * (n - 1)) // 2
        for i in range(n):
            x = start_x + i * spacing
            if i % 2 == 0:
                cv2.circle(img, (x, SIZE//2), s, c1, -1)
            else:
                cv2.rectangle(img, (x - s, SIZE//2 - s), (x + s, SIZE//2 + s), c2, -1)

    return img


def generate_containment():
    """One shape inside another — circle in square, triangle in circle, etc."""
    img = random_bg()
    c_outer = random_color()
    c_inner = random_color()
    cx, cy = SIZE // 2 + random.randint(-10, 10), SIZE // 2 + random.randint(-10, 10)

    outer = random.choice(["circle", "square", "triangle"])
    inner = random.choice(["circle", "square", "triangle"])

    outer_s = random.randint(30, 50)
    inner_s = random.randint(10, outer_s - 8)

    # Draw outer
    if outer == "circle":
        cv2.circle(img, (cx, cy), outer_s, c_outer, -1)
    elif outer == "square":
        cv2.rectangle(img, (cx-outer_s, cy-outer_s), (cx+outer_s, cy+outer_s), c_outer, -1)
    else:
        pts = np.array([[cx, cy-outer_s], [cx-outer_s, cy+outer_s], [cx+outer_s, cy+outer_s]])
        cv2.fillPoly(img, [pts], c_outer)

    # Draw inner
    if inner == "circle":
        cv2.circle(img, (cx, cy), inner_s, c_inner, -1)
    elif inner == "square":
        cv2.rectangle(img, (cx-inner_s, cy-inner_s), (cx+inner_s, cy+inner_s), c_inner, -1)
    else:
        pts = np.array([[cx, cy-inner_s], [cx-inner_s, cy+inner_s], [cx+inner_s, cy+inner_s]])
        cv2.fillPoly(img, [pts], c_inner)

    return img


GENERATORS = [
    (generate_single, 0.20),            # 20% single shapes
    (generate_composition, 0.10),       # 10% overlapping shapes
    (generate_partial, 0.08),           # 8% partially cropped
    (generate_nested, 0.05),            # 5% concentric/nested
    (generate_solid_color, 0.05),       # 5% solid colors
    (generate_gradient, 0.04),          # 4% gradients
    (generate_texture, 0.03),           # 3% textures
    (generate_joined_pair, 0.15),       # 15% touching/shared-edge pairs
    (generate_composite_object, 0.15),  # 15% recognisable composite objects
    (generate_chain, 0.08),             # 8% chains of connected shapes
    (generate_containment, 0.07),       # 7% shape inside shape
]
# Remaining ~10% = per-shape-type singles (ensure all shapes represented)


def generate_image():
    """Generate one training image using weighted random selection."""
    # 10% chance of dedicated shape type (ensures coverage)
    if random.random() < 0.10:
        shape = random.choice(ALL_SHAPES)
        return generate_single(shape_type=shape)

    # Weighted selection
    r = random.random()
    cumulative = 0
    for gen, weight in GENERATORS:
        cumulative += weight
        if r < cumulative:
            return gen()

    return generate_single()


def main():
    parser = argparse.ArgumentParser(description="Generate training primitives")
    parser.add_argument("--output-dir", default="training_data/primitives_diverse",
                        help="Output directory")
    parser.add_argument("--count", type=int, default=5000, help="Number of images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} primitives to {out_dir}/")

    for i in range(args.count):
        img = generate_image()
        filename = out_dir / f"prim_{i:06d}.png"
        cv2.imwrite(str(filename), img)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.count}")

    print(f"Done. {args.count} images in {out_dir}/")

    # Print distribution stats
    print("\nShape types in ALL_SHAPES:")
    for s in ALL_SHAPES:
        print(f"  {s}")
    print(f"\nTotal shape types: {len(ALL_SHAPES)}")
    print(f"Generator weights: {[(g.__name__, w) for g, w in GENERATORS]}")


if __name__ == "__main__":
    main()
