"""Validate JEPA encoder quality — does it learn useful visual features?

Tests:
1. Shape discrimination — triangle vs circle vs rectangle embeddings should be far apart
2. Color discrimination — same shape, different colors should be separable
3. Background invariance — same shape on different backgrounds should be similar
4. Size invariance — same shape at different sizes should be recognisable
5. Rotation sensitivity — rotated shapes should be similar but not identical
6. Curve vs line — must distinguish (our old encoder failed this)
7. Composite objects — house (rect+triangle) should differ from arrow (rect+triangle)
8. Masked prediction quality — how well does it predict hidden patches?
9. Nearest neighbour retrieval — does it find visually similar images?
10. Patch-level probing — what do individual patches encode?

Usage:
    python validate_jepa.py --checkpoint checkpoints/jepa_best.pt --data-dir /path/to/images
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Also check local directory (for running on training server)
sys.path.insert(0, str(Path(__file__).parent))

SIZE = 128


# ── Synthetic test image generators ──────────────────────────

def make_img(draw_fn, bg_color=(255, 255, 255)):
    img = np.full((SIZE, SIZE, 3), bg_color, dtype=np.uint8)
    draw_fn(img)
    return img


def make_triangle(color=(0, 0, 255), bg=(255, 255, 255), size=40, cx=64, cy=64, angle=0):
    def draw(img):
        pts = np.array([
            [cx, cy - size],
            [cx - size, cy + size],
            [cx + size, cy + size],
        ], dtype=np.float32)
        if angle != 0:
            cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            pts_centered = pts - [cx, cy]
            rotated = np.column_stack([
                pts_centered[:, 0] * cos_a - pts_centered[:, 1] * sin_a,
                pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a,
            ]) + [cx, cy]
            pts = rotated
        cv2.fillPoly(img, [pts.astype(np.int32)], color)
    return make_img(draw, bg)


def make_circle(color=(255, 0, 0), bg=(255, 255, 255), radius=35, cx=64, cy=64):
    return make_img(lambda i: cv2.circle(i, (cx, cy), radius, color, -1), bg)


def make_rectangle(color=(0, 255, 0), bg=(255, 255, 255), w=50, h=35, cx=64, cy=64):
    return make_img(lambda i: cv2.rectangle(i, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), color, -1), bg)


def make_pentagon(color=(255, 165, 0), bg=(255, 255, 255), size=35, cx=64, cy=64):
    def draw(img):
        pts = []
        for k in range(5):
            angle = 2 * math.pi * k / 5 - math.pi / 2
            pts.append([int(cx + size * math.cos(angle)), int(cy + size * math.sin(angle))])
        cv2.fillPoly(img, [np.array(pts)], color)
    return make_img(draw, bg)


def make_star(color=(255, 255, 0), bg=(255, 255, 255), size=35, cx=64, cy=64):
    def draw(img):
        pts = []
        for k in range(10):
            angle = 2 * math.pi * k / 10 - math.pi / 2
            r = size if k % 2 == 0 else size * 0.4
            pts.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
        cv2.fillPoly(img, [np.array(pts)], color)
    return make_img(draw, bg)


def make_curve(color=(0, 0, 0), bg=(255, 255, 255)):
    return make_img(lambda i: cv2.ellipse(i, (64, 64), (40, 40), 0, 0, 180, color, 3), bg)


def make_line(color=(0, 0, 0), bg=(255, 255, 255)):
    return make_img(lambda i: cv2.line(i, (24, 64), (104, 64), color, 3), bg)


def make_house(c1=(0, 0, 200), c2=(0, 200, 0), bg=(255, 255, 255)):
    def draw(img):
        cv2.rectangle(img, (39, 64), (89, 104), c1, -1)
        pts = np.array([[64, 34], [34, 64], [94, 64]])
        cv2.fillPoly(img, [pts], c2)
    return make_img(draw, bg)


def make_arrow(color=(0, 0, 200), bg=(255, 255, 255)):
    def draw(img):
        cv2.rectangle(img, (54, 64), (74, 104), color, -1)
        pts = np.array([[64, 34], [39, 64], [89, 64]])
        cv2.fillPoly(img, [pts], color)
    return make_img(draw, bg)


# ── Encoder wrapper ──────────────────────────────────────────

class JEPAEncoder:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from patch_jepa import PatchJEPA
        self.device = torch.device(device)
        self.model = PatchJEPA(
            img_size=128, patch_size=16, embed_dim=256, output_dim=512,
            encoder_depth=6, predictor_depth=3,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        log.info("Loaded JEPA checkpoint: %s (epoch %d, loss %.4f)",
                 checkpoint_path, ckpt.get("epoch", -1), ckpt.get("loss", -1))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def encode(self, img_np: np.ndarray) -> np.ndarray:
        """Encode a 128×128 BGR numpy image to 512-dim embedding."""
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode(tensor)
        return emb.cpu().numpy()[0]

    def encode_patches(self, img_np: np.ndarray) -> np.ndarray:
        """Encode to patch-level grid: (64, 256)."""
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            patches = self.model.patch_embed(tensor)
            patches = patches + self.model.pos_embed
            for block in self.model.encoder:
                patches = block(patches)
            patches = self.model.encoder_norm(patches)
        return patches.cpu().numpy()[0]  # (64, 256)

    def masked_prediction_loss(self, img_np: np.ndarray) -> float:
        """Run masked prediction and return loss."""
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        return out["pred_loss"].item()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Test suite ───────────────────────────────────────────────

def test_shape_discrimination(enc: JEPAEncoder) -> dict:
    """Different shapes should have low similarity."""
    shapes = {
        "triangle": make_triangle(),
        "circle": make_circle(),
        "rectangle": make_rectangle(),
        "pentagon": make_pentagon(),
        "star": make_star(),
    }
    embs = {k: enc.encode(v) for k, v in shapes.items()}

    results = {}
    names = list(shapes.keys())
    sims = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                s = cosine(embs[n1], embs[n2])
                results[f"{n1}_vs_{n2}"] = round(s, 3)
                sims.append(s)

    results["mean_cross_shape"] = round(np.mean(sims), 3)
    results["max_cross_shape"] = round(np.max(sims), 3)
    results["PASS"] = np.max(sims) < 0.7  # shapes should be separable
    return results


def test_color_discrimination(enc: JEPAEncoder) -> dict:
    """Same shape, different colors should be distinguishable but related."""
    colors = {
        "red_tri": make_triangle(color=(0, 0, 255)),
        "blue_tri": make_triangle(color=(255, 0, 0)),
        "green_tri": make_triangle(color=(0, 255, 0)),
        "red_circle": make_circle(color=(0, 0, 255)),
        "blue_circle": make_circle(color=(255, 0, 0)),
    }
    embs = {k: enc.encode(v) for k, v in colors.items()}

    results = {}
    # Same shape diff color
    results["red_vs_blue_tri"] = round(cosine(embs["red_tri"], embs["blue_tri"]), 3)
    results["red_vs_green_tri"] = round(cosine(embs["red_tri"], embs["green_tri"]), 3)
    # Cross shape same color
    results["red_tri_vs_red_circle"] = round(cosine(embs["red_tri"], embs["red_circle"]), 3)
    # Ideal: same-shape-diff-color > cross-shape-same-color
    results["PASS"] = results["red_vs_blue_tri"] > results["red_tri_vs_red_circle"]
    return results


def test_background_invariance(enc: JEPAEncoder) -> dict:
    """Same shape on different backgrounds should be similar."""
    bgs = [
        ("white", (255, 255, 255)),
        ("gray", (128, 128, 128)),
        ("blue", (200, 100, 50)),
        ("green", (50, 200, 50)),
        ("noisy", None),
    ]

    results = {}
    embs = {}
    for name, bg in bgs:
        if bg is None:
            img = np.random.randint(100, 200, (SIZE, SIZE, 3), dtype=np.uint8)
            cv2.fillPoly(img, [np.array([[64, 24], [24, 104], [104, 104]])], (0, 0, 255))
            embs[name] = enc.encode(img)
        else:
            embs[name] = enc.encode(make_triangle(bg=bg))

    sims = []
    names = list(embs.keys())
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                s = cosine(embs[n1], embs[n2])
                results[f"{n1}_vs_{n2}"] = round(s, 3)
                sims.append(s)

    results["mean_bg_similarity"] = round(np.mean(sims), 3)
    results["min_bg_similarity"] = round(np.min(sims), 3)
    results["PASS"] = np.min(sims) > 0.5  # same shape should be > 0.5 regardless of bg
    return results


def test_size_invariance(enc: JEPAEncoder) -> dict:
    """Same shape at different sizes should be recognisable."""
    sizes = {
        "tiny": make_triangle(size=15),
        "small": make_triangle(size=25),
        "medium": make_triangle(size=40),
        "large": make_triangle(size=55),
    }
    embs = {k: enc.encode(v) for k, v in sizes.items()}

    results = {}
    results["tiny_vs_large"] = round(cosine(embs["tiny"], embs["large"]), 3)
    results["small_vs_medium"] = round(cosine(embs["small"], embs["medium"]), 3)
    results["tiny_vs_medium"] = round(cosine(embs["tiny"], embs["medium"]), 3)
    results["PASS"] = results["tiny_vs_large"] > 0.4
    return results


def test_rotation(enc: JEPAEncoder) -> dict:
    """Rotated shapes should be similar but not identical."""
    rotations = {f"rot_{a}": make_triangle(angle=a) for a in [0, 30, 60, 90, 120, 180]}
    embs = {k: enc.encode(v) for k, v in rotations.items()}

    results = {}
    results["0_vs_30"] = round(cosine(embs["rot_0"], embs["rot_30"]), 3)
    results["0_vs_90"] = round(cosine(embs["rot_0"], embs["rot_90"]), 3)
    results["0_vs_180"] = round(cosine(embs["rot_0"], embs["rot_180"]), 3)
    # Should be similar (same shape) but not 1.0 (different orientation)
    results["PASS"] = results["0_vs_90"] > 0.3 and results["0_vs_90"] < 0.95
    return results


def test_curve_vs_line(enc: JEPAEncoder) -> dict:
    """CRITICAL: old encoder scored 0.976 (couldn't tell them apart)."""
    curve = enc.encode(make_curve())
    line = enc.encode(make_line())
    sim = cosine(curve, line)
    results = {
        "curve_vs_line": round(sim, 3),
        "old_encoder_score": 0.976,
        "PASS": sim < 0.7,  # must be clearly different
    }
    return results


def test_composite_discrimination(enc: JEPAEncoder) -> dict:
    """Composites with same primitives in different arrangements should differ."""
    house = enc.encode(make_house())
    arrow = enc.encode(make_arrow())
    plain_tri = enc.encode(make_triangle())
    plain_rect = enc.encode(make_rectangle())

    results = {
        "house_vs_arrow": round(cosine(house, arrow), 3),
        "house_vs_triangle": round(cosine(house, plain_tri), 3),
        "house_vs_rectangle": round(cosine(house, plain_rect), 3),
        "arrow_vs_triangle": round(cosine(arrow, plain_tri), 3),
    }
    results["PASS"] = results["house_vs_arrow"] < 0.8  # same parts, different arrangement
    return results


def test_masked_prediction(enc: JEPAEncoder) -> dict:
    """How well does the model predict masked patches?"""
    images = {
        "triangle": make_triangle(),
        "circle": make_circle(),
        "house": make_house(),
        "solid_red": np.full((SIZE, SIZE, 3), (0, 0, 255), dtype=np.uint8),
        "noise": np.random.randint(0, 255, (SIZE, SIZE, 3), dtype=np.uint8),
    }
    results = {}
    for name, img in images.items():
        loss = enc.masked_prediction_loss(img)
        results[f"{name}_loss"] = round(loss, 4)

    # Structured images should have lower prediction loss than noise
    structured_avg = np.mean([results[f"{k}_loss"] for k in ["triangle", "circle", "house"]])
    results["structured_avg"] = round(structured_avg, 4)
    results["noise_loss"] = results["noise_loss"]
    results["PASS"] = structured_avg < results["noise_loss"]
    return results


def test_patch_diversity(enc: JEPAEncoder) -> dict:
    """Patch embeddings should be diverse (not all the same)."""
    images = {
        "triangle": make_triangle(),
        "solid": np.full((SIZE, SIZE, 3), (0, 0, 255), dtype=np.uint8),
    }
    results = {}
    for name, img in images.items():
        patches = enc.encode_patches(img)  # (64, 256)
        # Pairwise similarity between patches
        norms = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
        sims = norms @ norms.T
        np.fill_diagonal(sims, 0)
        results[f"{name}_mean_patch_sim"] = round(sims.mean(), 3)
        results[f"{name}_std_patch_sim"] = round(sims.std(), 3)

    # Triangle should have diverse patches (edges vs background)
    # Solid should have uniform patches
    results["PASS"] = results["triangle_mean_patch_sim"] < results["solid_mean_patch_sim"]
    return results


def run_all(enc: JEPAEncoder) -> dict:
    tests = [
        ("shape_discrimination", test_shape_discrimination),
        ("color_discrimination", test_color_discrimination),
        ("background_invariance", test_background_invariance),
        ("size_invariance", test_size_invariance),
        ("rotation", test_rotation),
        ("curve_vs_line", test_curve_vs_line),
        ("composite_discrimination", test_composite_discrimination),
        ("masked_prediction", test_masked_prediction),
        ("patch_diversity", test_patch_diversity),
    ]

    all_results = {}
    passed = 0
    total = 0

    for name, test_fn in tests:
        log.info("Running: %s", name)
        try:
            result = test_fn(enc)
            all_results[name] = result
            total += 1
            if result.get("PASS"):
                passed += 1
                log.info("  PASS: %s", {k: v for k, v in result.items() if k != "PASS"})
            else:
                log.info("  FAIL: %s", {k: v for k, v in result.items() if k != "PASS"})
        except Exception as e:
            log.error("  ERROR: %s", e)
            all_results[name] = {"error": str(e)}
            total += 1

    all_results["summary"] = {
        "passed": passed,
        "total": total,
        "accuracy": f"{passed}/{total} ({passed/total*100:.0f}%)" if total > 0 else "N/A",
    }
    log.info("Results: %s", all_results["summary"]["accuracy"])

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate JEPA encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to JEPA checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    enc = JEPAEncoder(args.checkpoint, args.device)
    results = run_all(enc)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info("Saved results to %s", args.output)
