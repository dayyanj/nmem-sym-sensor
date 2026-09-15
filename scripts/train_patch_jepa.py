"""Train PatchJEPA visual encoder.

Usage:
  # Prepare (encode is not needed — JEPA trains end-to-end):
  python train_patch_jepa.py --data-dir /path/to/images --epochs 100 --device cuda

  # Resume:
  python train_patch_jepa.py --data-dir /path/to/images --resume checkpoints/best.pt --epochs 100

Training data: any directory of images (jpg/png). Recursively scanned.
No labels needed — self-supervised via masked patch prediction.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Add src to path for patch_jepa import
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nmem_sym_sensor.patch_jepa import PatchJEPA, count_parameters


class ImageFolderDataset(Dataset):
    """Simple image dataset — recursively loads all images from a directory."""

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    def __init__(self, root: str, img_size: int = 128, augment: bool = True):
        self.img_size = img_size
        self.augment = augment

        self.paths = []
        root_path = Path(root)
        for ext in self.EXTENSIONS:
            self.paths.extend(root_path.rglob(f"*{ext}"))
            self.paths.extend(root_path.rglob(f"*{ext.upper()}"))
        self.paths = sorted(set(self.paths))

        # Transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        log.info("Found %d images in %s", len(self.paths), root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img)
        except Exception:
            # Return random noise for corrupt images
            return torch.randn(3, self.img_size, self.img_size)


def train(args):
    device = torch.device(args.device)
    log.info("Device: %s", device)

    # Dataset
    dataset = ImageFolderDataset(args.data_dir, img_size=128, augment=True)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.data_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = PatchJEPA(
        img_size=128,
        patch_size=16,
        embed_dim=256,
        output_dim=512,
        encoder_depth=6,
        encoder_heads=4,
        predictor_depth=3,
        predictor_heads=4,
        mask_ratio=args.mask_ratio,
        sigreg_weight=args.sigreg_weight,
    ).to(device)

    log.info("Parameters: %s", f"{count_parameters(model):,}")
    log.info("Dataset: %d images", len(dataset))
    log.info("Batch size: %d, Epochs: %d", args.batch_size, args.epochs)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine LR with warmup
    warmup_steps = len(dataloader) * args.warmup_epochs
    total_steps = len(dataloader) * args.epochs

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Resume
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('loss', float('inf'))
        log.info("Resumed at epoch %d, loss=%.4f", start_epoch, best_loss)

    # Output directory
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    log.info("Starting training: epochs %d-%d", start_epoch, start_epoch + args.epochs - 1)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_sig = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{start_epoch + args.epochs - 1}")
        for batch in pbar:
            batch = batch.to(device)

            out = model(batch)
            loss = out['total_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_pred += out['pred_loss'].item()
            epoch_sig += out['sigreg_loss'].item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                pred=f"{out['pred_loss'].item():.4f}",
                sig=f"{out['sigreg_loss'].item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}",
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_pred = epoch_pred / max(n_batches, 1)
        avg_sig = epoch_sig / max(n_batches, 1)

        log.info("Epoch %d: loss=%.4f (pred=%.4f, sig=%.4f), lr=%.6f",
                 epoch, avg_loss, avg_pred, avg_sig, scheduler.get_last_lr()[0])

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = ckpt_dir / f"jepa_epoch_{epoch:04d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, ckpt_dir / "jepa_best.pt")

        # Embedding quality check every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Grab a batch and compute embeddings
                sample_batch = next(iter(dataloader))[:8].to(device)
                embs = model.encode(sample_batch)

                # Check embedding statistics
                emb_mean = embs.mean().item()
                emb_std = embs.std().item()
                emb_norm = embs.norm(dim=1).mean().item()

                # Pairwise similarities (should be spread, not all the same)
                sims = F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), dim=2)
                sim_mean = sims.mean().item()
                sim_min = sims.min().item()
                sim_max = sims[~torch.eye(8, dtype=torch.bool, device=device)].max().item()

                log.info("Embedding stats: mean=%.3f, std=%.3f, norm=%.3f",
                         emb_mean, emb_std, emb_norm)
                log.info("Pairwise similarity: mean=%.3f, min=%.3f, max=%.3f (off-diag)",
                         sim_mean, sim_min, sim_max)

                # Save sample embeddings for inspection
                np.save(ckpt_dir / f"sample_embs_epoch_{epoch:04d}.npy",
                        embs.cpu().numpy())

            model.train()

    log.info("Training complete. Best loss: %.4f", best_loss)

    # Export final encoder for inference
    log.info("Exporting encoder for inference...")
    model.eval()
    dummy = torch.randn(1, 3, 128, 128).to(device)
    emb = model.encode(dummy)
    log.info("Final encoder output: %s", emb.shape)

    # Save full model state
    torch.save({
        'epoch': start_epoch + args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'loss': best_loss,
        'config': {
            'img_size': 128,
            'patch_size': 16,
            'embed_dim': 256,
            'output_dim': 512,
            'encoder_depth': 6,
            'predictor_depth': 3,
        },
    }, ckpt_dir / "jepa_final.pt")
    log.info("Saved final model: %s", ckpt_dir / "jepa_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PatchJEPA visual encoder")
    parser.add_argument("--data-dir", required=True, help="Directory of training images (recursive)")
    parser.add_argument("--output-dir", default="checkpoints/jepa", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--sigreg-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)
