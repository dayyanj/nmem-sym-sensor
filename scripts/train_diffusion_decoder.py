#!/usr/bin/env python3
"""Train the visual diffusion decoder on diverse images.

Encodes images with the MoE visual encoder → 512-dim embeddings,
then trains a diffusion UNet to reconstruct 64×64 images from those embeddings.

Usage:
    python scripts/train_diffusion_decoder.py \
        --data-dir /path/to/training/images \
        --encoder-ckpt checkpoints_v4_medical/visual_moe/moe_visual_encoder_final.pt \
        --epochs 100 \
        --batch-size 32 \
        --device cuda

Training data should be a directory of images (jpg/png). The script will:
1. Load each image, resize to 64×64
2. Encode with MoE encoder → 512-dim embedding
3. Train diffusion decoder to reconstruct from embedding

Expected training time: ~8-12 hours on RTX 4090 with 37K images.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nmem_sym_sensor.visual_diffusion import (
    DiffusionSchedule,
    VisualDiffusionUNet,
    count_parameters,
    ddim_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class ImageEmbeddingDataset(Dataset):
    """Load pre-encoded embeddings and images from .npy files.

    If .npy files exist in data_dir (from prepare_embeddings.py), loads directly.
    Otherwise falls back to live encoding (slower).
    """

    def __init__(self, image_dir: str, encoder=None, device: str = "cuda", max_images: int = 0):
        self.images = []
        self.embeddings = []

        # Try loading pre-prepared .npy files first
        emb_path = os.path.join(image_dir, "embeddings.npy")
        img_path = os.path.join(image_dir, "images.npy")

        if os.path.exists(emb_path) and os.path.exists(img_path):
            log.info("Loading pre-encoded data from %s", image_dir)
            embs = np.load(emb_path)
            imgs = np.load(img_path)
            if max_images > 0:
                embs = embs[:max_images]
                imgs = imgs[:max_images]
            self.embeddings = [torch.from_numpy(embs[i]) for i in range(len(embs))]
            self.images = [torch.from_numpy(imgs[i]) for i in range(len(imgs))]
            log.info("Loaded %d pre-encoded samples", len(self.images))
            return

        # Fallback: live encoding
        if encoder is None:
            raise ValueError(f"No .npy files in {image_dir} and no encoder provided. "
                             "Run prepare_embeddings.py first.")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    paths.append(os.path.join(root, f))
        paths.sort()
        if max_images > 0:
            paths = paths[:max_images]

        log.info("Found %d images in %s (live encoding)", len(paths), image_dir)
        encoder.eval()
        batch_size = 64

        for i in tqdm(range(0, len(paths), batch_size), desc="Encoding"):
            batch_paths = paths[i:i + batch_size]
            batch_imgs = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensor = self.transform(img)
                    batch_imgs.append(tensor)
                except Exception as e:
                    log.debug("Skip %s: %s", p, e)

            if not batch_imgs:
                continue

            imgs_batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                embs = encoder(imgs_batch)

            for idx, img_tensor in enumerate(batch_imgs):
                self.images.append(img_tensor)
                self.embeddings.append(embs[idx].cpu())

        log.info("Dataset: %d images encoded", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.embeddings[idx]


def load_moe_encoder(checkpoint_path: str, device: str = "cuda"):
    """Load the MoE visual encoder for embedding generation."""
    # Try ONNX first (faster, no arch dependency)
    onnx_path = checkpoint_path if checkpoint_path.endswith(".onnx") else str(Path(checkpoint_path).parent.parent.parent / "models" / "visual_moe_v4.onnx")
    if os.path.exists(onnx_path):
        import onnxruntime as ort
        log.info("Loading MoE encoder from ONNX: %s", onnx_path)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_path, providers=providers)

        class ONNXEncoder:
            def __init__(self, sess):
                self.sess = sess
                self.input_name = sess.get_inputs()[0].name

            def eval(self):
                return self

            def __call__(self, x):
                # x is (B, 3, 64, 64) tensor
                if isinstance(x, torch.Tensor):
                    x_np = x.cpu().numpy()
                else:
                    x_np = x
                results = self.sess.run(None, {self.input_name: x_np})
                return torch.from_numpy(results[0]).to(x.device if isinstance(x, torch.Tensor) else device)

        return ONNXEncoder(session)

    # Fall back to PyTorch checkpoint
    log.info("Loading MoE encoder from checkpoint: %s", checkpoint_path)
    from nmem_sym_sensor.visual_encoder import build_moe_encoder
    encoder = build_moe_encoder()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "encoder_state_dict" in state:
        encoder.load_state_dict(state["encoder_state_dict"])
    else:
        encoder.load_state_dict(state)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def train(args):
    device = args.device
    log.info("Device: %s", device)

    # Check if pre-encoded data exists (skip encoder loading)
    emb_path = os.path.join(args.data_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        log.info("Pre-encoded data found — skipping encoder load")
        encoder = None
    else:
        encoder = load_moe_encoder(args.encoder_ckpt, device)

    # Build dataset
    dataset = ImageEmbeddingDataset(
        args.data_dir, encoder, device,
        max_images=args.max_images,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    # Build model
    model = VisualDiffusionUNet(cond_dim=512).to(device)
    schedule = DiffusionSchedule(timesteps=1000, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader),
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("loss", float("inf"))
        log.info("Resumed at epoch %d, loss=%.4f", start_epoch, best_loss)
        # Reset LR scheduler for warm restart
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(dataloader),
        )

    log.info("Model parameters: %s", f"{count_parameters(model):,}")
    log.info("Dataset size: %d", len(dataset))
    log.info("Batch size: %d, Epochs: %d (starting at %d)", args.batch_size, args.epochs, start_epoch)

    # Checkpoint directory
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    if not args.resume:
        best_loss = float("inf")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, embeddings in pbar:
            images = images.to(device)       # (B, 3, 64, 64) in [-1, 1]
            embeddings = embeddings.to(device)  # (B, 512)

            # Sample random timesteps
            t = torch.randint(0, schedule.timesteps, (images.shape[0],), device=device)

            # Add noise
            noise = torch.randn_like(images)
            noisy = schedule.q_sample(images, t, noise)

            # Predict noise
            pred_noise = model(noisy, t, embeddings)

            # Loss
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info("Epoch %d: loss=%.4f, lr=%.6f", epoch, avg_loss, scheduler.get_last_lr()[0])

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"diffusion_decoder_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
            }, ckpt_dir / "diffusion_decoder_best.pt")

        # Generate samples every 20 epochs — diverse selection
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Pick 8 diverse samples: spread evenly across the dataset
                n = len(dataset)
                indices = [int(n * i / 8) for i in range(8)]
                sample_embs = torch.stack([dataset.embeddings[i] for i in indices]).to(device)
                samples = ddim_sample(
                    model, schedule, sample_embs,
                    steps=20, shape=(sample_embs.shape[0], 3, 128, 128), device=device,
                )
                from torchvision.utils import save_image
                samples = (samples + 1) / 2
                originals = torch.stack([dataset.images[i] for i in indices]).to(device)
                originals = (originals + 1) / 2
                grid = torch.cat([originals, samples], dim=0)
                save_image(grid, ckpt_dir / f"samples_epoch_{epoch:04d}.png", nrow=8)
                log.info("Saved samples: %s", ckpt_dir / f"samples_epoch_{epoch:04d}.png")

    log.info("Training complete. Best loss: %.4f", best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train visual diffusion decoder")
    parser.add_argument("--data-dir", required=True, help="Directory of training images")
    parser.add_argument("--encoder-ckpt", default="checkpoints_v4_medical/visual_moe/moe_visual_encoder_final.pt")
    parser.add_argument("--output-dir", default="checkpoints/diffusion_decoder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-images", type=int, default=0, help="Limit dataset size (0=all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()
    train(args)
