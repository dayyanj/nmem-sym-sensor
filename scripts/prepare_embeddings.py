#!/usr/bin/env python3
"""Pre-encode images into embeddings for diffusion decoder training.

Saves two files:
    embeddings.npy  — (N, 512) float32, MoE visual embeddings
    images.npy      — (N, 3, 128, 128) float32, normalized images [-1, 1]

Usage:
    python scripts/prepare_embeddings.py \
        --data-dir /path/to/images \
        --encoder /path/to/visual_moe_v4.onnx \
        --output-dir /path/to/prepared \
        --device cuda
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_onnx_encoder(path: str, device: str = "cuda"):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
    session = ort.InferenceSession(path, providers=providers)
    input_name = session.get_inputs()[0].name
    log.info("ONNX encoder loaded: %s, providers=%s", path, session.get_providers())

    def encode(batch_np):
        return session.run(None, {input_name: batch_np})[0]

    return encode


def main():
    parser = argparse.ArgumentParser(description="Pre-encode images for diffusion training")
    parser.add_argument("--data-dir", required=True, help="Directory of training images")
    parser.add_argument("--encoder", required=True, help="Path to ONNX encoder")
    parser.add_argument("--output-dir", required=True, help="Output directory for .npy files")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0, help="Limit (0=all)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(root, f))
    paths.sort()
    if args.max_images > 0:
        paths = paths[:args.max_images]
    log.info("Found %d images", len(paths))

    # Load encoder
    encode = load_onnx_encoder(args.encoder, args.device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Pre-allocate arrays
    all_embeddings = []
    all_images = []
    skipped = 0

    for i in tqdm(range(0, len(paths), args.batch_size), desc="Encoding"):
        batch_paths = paths[i:i + args.batch_size]
        batch_imgs = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = transform(img)
                batch_imgs.append(tensor)
            except Exception:
                skipped += 1
                continue

        if not batch_imgs:
            continue

        imgs_np = torch.stack(batch_imgs).numpy()
        embs = encode(imgs_np)

        all_embeddings.append(embs)
        all_images.append(imgs_np)

    # Concatenate and save
    embeddings = np.concatenate(all_embeddings, axis=0)
    images = np.concatenate(all_images, axis=0)

    emb_path = out_dir / "embeddings.npy"
    img_path = out_dir / "images.npy"

    np.save(emb_path, embeddings)
    np.save(img_path, images)

    log.info("Saved: %s (%s, %.1f MB)", emb_path, embeddings.shape,
             embeddings.nbytes / 1e6)
    log.info("Saved: %s (%s, %.1f MB)", img_path, images.shape,
             images.nbytes / 1e6)
    log.info("Skipped %d corrupt/unreadable images", skipped)

    # Save metadata
    meta = {
        "n_images": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "image_shape": list(images.shape[1:]),
        "encoder": args.encoder,
        "data_dir": args.data_dir,
    }
    import json
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata: %s", meta)


if __name__ == "__main__":
    main()
