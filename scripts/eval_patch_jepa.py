"""Evaluate PatchJEPA: patch-level vs mean-pool discrimination.

Tests whether the 64x256 patch grid retains discriminative information
that mean-pooling destroys. Generates synthetic test images and compares
similarity scores at both levels.

Usage:
  python eval_patch_jepa.py --checkpoint /path/to/jepa_final.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from nmem_sym_sensor.patch_jepa import PatchJEPA

# ── Synthetic test image generators ──────────────────────


def make_image(size=128):
    """Create a blank white image."""
    return Image.new("RGB", (size, size), "white")


def draw_triangle(img, color="red", bg="white", cx=64, cy=64, r=40):
    """Draw a triangle centered at (cx, cy)."""
    img = Image.new("RGB", img.size, bg)
    d = ImageDraw.Draw(img)
    pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
    d.polygon(pts, fill=color)
    return img


def draw_circle(img, color="red", bg="white", cx=64, cy=64, r=40):
    img = Image.new("RGB", img.size, bg)
    d = ImageDraw.Draw(img)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return img


def draw_rectangle(img, color="red", bg="white", cx=64, cy=64, w=60, h=40):
    img = Image.new("RGB", img.size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], fill=color)
    return img


def draw_star(img, color="red", bg="white", cx=64, cy=64, r=40):
    import math
    img = Image.new("RGB", img.size, bg)
    d = ImageDraw.Draw(img)
    pts = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.4
        pts.append((cx + rad * math.cos(angle), cy - rad * math.sin(angle)))
    d.polygon(pts, fill=color)
    return img


def draw_pentagon(img, color="red", bg="white", cx=64, cy=64, r=40):
    import math
    img = Image.new("RGB", img.size, bg)
    d = ImageDraw.Draw(img)
    pts = [(cx + r * math.cos(math.pi/2 + i * 2*math.pi/5),
            cy - r * math.sin(math.pi/2 + i * 2*math.pi/5)) for i in range(5)]
    d.polygon(pts, fill=color)
    return img


def rotate_image(img, angle):
    """Rotate image by angle degrees, white fill."""
    return img.rotate(angle, fillcolor="white", expand=False)


def resize_shape(draw_fn, size_factor, **kwargs):
    """Draw shape at different sizes."""
    r = int(40 * size_factor)
    return draw_fn(make_image(), r=max(r, 5), **kwargs)


# ── Encoding helpers ─────────────────────────────────────


def img_to_tensor(img, device):
    """PIL Image -> normalized tensor."""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return t(img).unsqueeze(0).to(device)


def encode_global(model, tensor):
    """Mean-pooled 512-dim embedding."""
    with torch.no_grad():
        return model.encode(tensor)  # (1, 512)


def encode_patches(model, tensor):
    """Full 64x256 patch grid (no mean-pool)."""
    with torch.no_grad():
        patches = model.patch_embed(tensor)
        patches = patches + model.pos_embed
        for block in model.encoder:
            patches = block(patches)
        patches = model.encoder_norm(patches)
        return patches  # (1, 64, 256)


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(a.flatten().unsqueeze(0),
                                b.flatten().unsqueeze(0)).item()


def patch_sim(patches_a, patches_b):
    """Patch-level similarity: average of per-patch cosine similarities.

    Compares each patch position in A with the same position in B.
    This preserves spatial structure — patches that differ (shape region
    vs background) will lower the score differently than mean-pool.
    """
    # (1, 64, 256) -> (64, 256)
    a = F.normalize(patches_a.squeeze(0), dim=1)
    b = F.normalize(patches_b.squeeze(0), dim=1)
    per_patch = (a * b).sum(dim=1)  # (64,)
    return per_patch.mean().item(), per_patch.min().item(), per_patch.std().item()


def patch_set_sim(patches_a, patches_b):
    """Patch-set similarity: best-match between patch sets (order-independent).

    For each patch in A, find the most similar patch in B.
    Average of best matches. This is robust to spatial shifts.
    """
    a = F.normalize(patches_a.squeeze(0), dim=1)  # (64, 256)
    b = F.normalize(patches_b.squeeze(0), dim=1)  # (64, 256)
    sims = a @ b.T  # (64, 64)
    best_a_to_b = sims.max(dim=1).values.mean().item()
    best_b_to_a = sims.max(dim=0).values.mean().item()
    return (best_a_to_b + best_b_to_a) / 2


# ── Test suites ──────────────────────────────────────────


def test_shape_discrimination(model, device):
    """Can patches discriminate triangle from circle from rectangle?"""
    shapes = {
        "triangle": draw_triangle(make_image(), color="red"),
        "circle": draw_circle(make_image(), color="red"),
        "rectangle": draw_rectangle(make_image(), color="red"),
        "pentagon": draw_pentagon(make_image(), color="red"),
        "star": draw_star(make_image(), color="red"),
    }
    tensors = {k: img_to_tensor(v, device) for k, v in shapes.items()}

    # Encode at both levels
    globals_ = {k: encode_global(model, v) for k, v in tensors.items()}
    patches_ = {k: encode_patches(model, v) for k, v in tensors.items()}

    results = {}
    names = list(shapes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = f"{a}_vs_{b}"
            g_sim = cosine_sim(globals_[a], globals_[b])
            p_mean, p_min, p_std = patch_sim(patches_[a], patches_[b])
            p_set = patch_set_sim(patches_[a], patches_[b])
            results[key] = {
                "global": round(g_sim, 4),
                "patch_positional": round(p_mean, 4),
                "patch_min": round(p_min, 4),
                "patch_std": round(p_std, 4),
                "patch_set": round(p_set, 4),
            }

    # Summary
    g_vals = [v["global"] for v in results.values()]
    p_vals = [v["patch_positional"] for v in results.values()]
    ps_vals = [v["patch_set"] for v in results.values()]
    results["_summary"] = {
        "global_mean": round(np.mean(g_vals), 4),
        "patch_positional_mean": round(np.mean(p_vals), 4),
        "patch_set_mean": round(np.mean(ps_vals), 4),
        "discrimination_gain": round(np.mean(g_vals) - np.mean(p_vals), 4),
    }
    return results


def test_color_discrimination(model, device):
    """Can patches discriminate colors on same shape?"""
    colors = {
        "red_tri": draw_triangle(make_image(), color="red"),
        "blue_tri": draw_triangle(make_image(), color="blue"),
        "green_tri": draw_triangle(make_image(), color="green"),
        "red_circle": draw_circle(make_image(), color="red"),
    }
    tensors = {k: img_to_tensor(v, device) for k, v in colors.items()}
    globals_ = {k: encode_global(model, v) for k, v in tensors.items()}
    patches_ = {k: encode_patches(model, v) for k, v in tensors.items()}

    pairs = [
        ("red_tri", "blue_tri"),
        ("red_tri", "green_tri"),
        ("red_tri", "red_circle"),
    ]
    results = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        g_sim = cosine_sim(globals_[a], globals_[b])
        p_mean, p_min, p_std = patch_sim(patches_[a], patches_[b])
        p_set = patch_set_sim(patches_[a], patches_[b])
        results[key] = {
            "global": round(g_sim, 4),
            "patch_positional": round(p_mean, 4),
            "patch_set": round(p_set, 4),
        }
    return results


def test_background_invariance(model, device):
    """Same shape on different backgrounds — should be similar."""
    bgs = {
        "white": draw_triangle(make_image(), color="red", bg="white"),
        "gray": draw_triangle(make_image(), color="red", bg="gray"),
        "blue_bg": draw_triangle(make_image(), color="red", bg="lightblue"),
        "green_bg": draw_triangle(make_image(), color="red", bg="lightgreen"),
    }
    tensors = {k: img_to_tensor(v, device) for k, v in bgs.items()}
    globals_ = {k: encode_global(model, v) for k, v in tensors.items()}
    patches_ = {k: encode_patches(model, v) for k, v in tensors.items()}

    results = {}
    names = list(bgs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = f"{a}_vs_{b}"
            g_sim = cosine_sim(globals_[a], globals_[b])
            p_mean, _, _ = patch_sim(patches_[a], patches_[b])
            p_set = patch_set_sim(patches_[a], patches_[b])
            results[key] = {
                "global": round(g_sim, 4),
                "patch_positional": round(p_mean, 4),
                "patch_set": round(p_set, 4),
            }

    g_vals = [v["global"] for v in results.values()]
    p_vals = [v["patch_positional"] for v in results.values()]
    ps_vals = [v["patch_set"] for v in results.values()]
    results["_summary"] = {
        "global_mean": round(np.mean(g_vals), 4),
        "patch_positional_mean": round(np.mean(p_vals), 4),
        "patch_set_mean": round(np.mean(ps_vals), 4),
    }
    return results


def test_size_invariance(model, device):
    """Same shape at different sizes."""
    sizes = {
        "tiny": draw_triangle(make_image(), color="red", r=10),
        "small": draw_triangle(make_image(), color="red", r=20),
        "medium": draw_triangle(make_image(), color="red", r=40),
        "large": draw_triangle(make_image(), color="red", r=55),
    }
    tensors = {k: img_to_tensor(v, device) for k, v in sizes.items()}
    globals_ = {k: encode_global(model, v) for k, v in tensors.items()}
    patches_ = {k: encode_patches(model, v) for k, v in tensors.items()}

    pairs = [("tiny", "large"), ("small", "medium"), ("tiny", "medium")]
    results = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        g_sim = cosine_sim(globals_[a], globals_[b])
        p_mean, _, _ = patch_sim(patches_[a], patches_[b])
        p_set = patch_set_sim(patches_[a], patches_[b])
        results[key] = {
            "global": round(g_sim, 4),
            "patch_positional": round(p_mean, 4),
            "patch_set": round(p_set, 4),
        }
    return results


def test_rotation(model, device):
    """Same shape at different rotations."""
    base = draw_triangle(make_image(), color="red")
    angles = {"0": base, "30": rotate_image(base, 30),
              "90": rotate_image(base, 90), "180": rotate_image(base, 180)}
    tensors = {k: img_to_tensor(v, device) for k, v in angles.items()}
    globals_ = {k: encode_global(model, v) for k, v in tensors.items()}
    patches_ = {k: encode_patches(model, v) for k, v in tensors.items()}

    pairs = [("0", "30"), ("0", "90"), ("0", "180")]
    results = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        g_sim = cosine_sim(globals_[a], globals_[b])
        p_mean, _, _ = patch_sim(patches_[a], patches_[b])
        p_set = patch_set_sim(patches_[a], patches_[b])
        results[key] = {
            "global": round(g_sim, 4),
            "patch_positional": round(p_mean, 4),
            "patch_set": round(p_set, 4),
        }
    return results


# ── Main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info("Loading checkpoint: %s", args.checkpoint)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})

    model = PatchJEPA(
        img_size=cfg.get("img_size", 128),
        patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 256),
        output_dim=cfg.get("output_dim", 512),
        encoder_depth=cfg.get("encoder_depth", 6),
        predictor_depth=cfg.get("predictor_depth", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Model loaded (epoch %d, loss %.4f)", ckpt.get("epoch", 0), ckpt.get("loss", 0))

    results = {}

    log.info("=== Shape Discrimination ===")
    results["shape"] = test_shape_discrimination(model, device)
    for k, v in results["shape"].items():
        if k.startswith("_"):
            log.info("  SUMMARY: %s", v)
        else:
            log.info("  %-30s global=%.3f  patch_pos=%.3f  patch_set=%.3f",
                     k, v["global"], v["patch_positional"], v["patch_set"])

    log.info("=== Color Discrimination ===")
    results["color"] = test_color_discrimination(model, device)
    for k, v in results["color"].items():
        log.info("  %-30s global=%.3f  patch_pos=%.3f  patch_set=%.3f",
                 k, v["global"], v["patch_positional"], v["patch_set"])

    log.info("=== Background Invariance ===")
    results["background"] = test_background_invariance(model, device)
    for k, v in results["background"].items():
        if k.startswith("_"):
            log.info("  SUMMARY: %s", v)
        else:
            log.info("  %-30s global=%.3f  patch_pos=%.3f  patch_set=%.3f",
                     k, v["global"], v["patch_positional"], v["patch_set"])

    log.info("=== Size Invariance ===")
    results["size"] = test_size_invariance(model, device)
    for k, v in results["size"].items():
        log.info("  %-30s global=%.3f  patch_pos=%.3f  patch_set=%.3f",
                 k, v["global"], v["patch_positional"], v["patch_set"])

    log.info("=== Rotation ===")
    results["rotation"] = test_rotation(model, device)
    for k, v in results["rotation"].items():
        log.info("  %-30s global=%.3f  patch_pos=%.3f  patch_set=%.3f",
                 k, v["global"], v["patch_positional"], v["patch_set"])

    # Overall verdict
    shape_summary = results["shape"]["_summary"]
    log.info("")
    log.info("=== VERDICT ===")
    log.info("Shape discrimination:")
    log.info("  Global mean sim:   %.3f (want < 0.7)", shape_summary["global_mean"])
    log.info("  Patch-pos mean:    %.3f (want < 0.7)", shape_summary["patch_positional_mean"])
    log.info("  Patch-set mean:    %.3f (want < 0.8)", shape_summary["patch_set_mean"])
    log.info("  Gain (global-patch): %.3f (positive = patches better)",
             shape_summary["discrimination_gain"])

    # Save results
    out_path = Path(args.checkpoint).parent / "patch_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
