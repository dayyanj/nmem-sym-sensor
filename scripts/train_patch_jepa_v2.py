"""Train PatchJEPA v2 — local-to-global crops, Gram anchoring, saliency masking.

Usage:
  python train_patch_jepa_v2.py --data-dir /path/to/images --epochs 100 --device cuda
"""
from __future__ import annotations

import argparse
import logging
import sys
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from patch_jepa_v2 import PatchJEPAv2, count_parameters


class LocalGlobalDataset(Dataset):
    """Dataset that provides both a local crop and the full image."""

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    def __init__(self, root: str, img_size: int = 128, crop_scale=(0.4, 0.75)):
        self.img_size = img_size
        self.crop_scale = crop_scale

        self.paths = []
        for ext in self.EXTENSIONS:
            self.paths.extend(Path(root).rglob(f"*{ext}"))
            self.paths.extend(Path(root).rglob(f"*{ext.upper()}"))
        self.paths = sorted(set(self.paths))

        # Teacher transform: full image with mild augmentation
        self.teacher_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        # Student transform: random crop (local view)
        self.student_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=crop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        log.info("Found %d images in %s", len(self.paths), root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            teacher = self.teacher_transform(img)
            student = self.student_transform(img)
            return student, teacher
        except Exception:
            t = torch.randn(3, self.img_size, self.img_size)
            return t, t


def train(args):
    device = torch.device(args.device)
    log.info("Device: %s", device)

    dataset = LocalGlobalDataset(args.data_dir, img_size=128, crop_scale=(0.4, 0.75))
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

    model = PatchJEPAv2(
        img_size=128,
        patch_size=16,
        embed_dim=256,
        output_dim=256,
        encoder_depth=6,
        predictor_depth=3,
        mask_ratio=0.5,
        sigreg_weight=args.sigreg_weight,
        gram_weight=args.gram_weight,
        ema_momentum=0.996,
    ).to(device)

    log.info("Parameters: %s", f"{count_parameters(model):,}")
    log.info("Dataset: %d images", len(dataset))

    # Fixed LR (DINOv3 lesson — no warmup, no decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05,
    )

    # Resume
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('loss', float('inf'))
        log.info("Resumed at epoch %d, loss=%.4f", start_epoch, best_loss)

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info("Batch size: %d, Epochs: %d, LR: %.6f (fixed)", args.batch_size, args.epochs, args.lr)
    log.info("Starting training: epochs %d-%d", start_epoch, start_epoch + args.epochs - 1)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_sig = 0
        epoch_gram = 0
        n_batches = 0

        # Gram anchor: save at epoch 10, activate at epoch 15
        if epoch == start_epoch + 9:
            model.save_gram_teacher()
        if epoch == start_epoch + 14 and not model.gram_active:
            if model.gram_teacher is not None:
                model.gram_active = True
                log.info("Gram anchoring activated")

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{start_epoch + args.epochs - 1}")
        for student_batch, teacher_batch in pbar:
            student_batch = student_batch.to(device)
            teacher_batch = teacher_batch.to(device)

            out = model(student_batch, teacher_batch)
            loss = out['total_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pred += out['pred_loss'].item()
            epoch_sig += out['sigreg_loss'].item()
            epoch_gram += out['gram_loss'].item() if isinstance(out['gram_loss'], torch.Tensor) else out['gram_loss']
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                pred=f"{out['pred_loss'].item():.4f}",
                sig=f"{out['sigreg_loss'].item():.4f}",
                gram=f"{out['gram_loss'].item() if isinstance(out['gram_loss'], torch.Tensor) else 0:.4f}",
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_pred = epoch_pred / max(n_batches, 1)
        avg_sig = epoch_sig / max(n_batches, 1)
        avg_gram = epoch_gram / max(n_batches, 1)

        log.info("Epoch %d: loss=%.4f (pred=%.4f, sig=%.4f, gram=%.4f)",
                 epoch, avg_loss, avg_pred, avg_sig, avg_gram)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_dir / f"jepa_v2_epoch_{epoch:04d}.pt")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, ckpt_dir / "jepa_v2_best.pt")

        # Validation every 20 epochs
        if epoch % 20 == 0 or epoch == start_epoch:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(dataloader))
                student_imgs = sample_batch[0][:8].to(device)
                embs = model.encode(student_imgs)

                emb_mean = embs.mean().item()
                emb_std = embs.std().item()
                emb_norm = embs.norm(dim=1).mean().item()

                sims = F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), dim=2)
                mask = ~torch.eye(8, dtype=torch.bool, device=device)
                sim_mean = sims[mask].mean().item()
                sim_min = sims[mask].min().item()
                sim_max = sims[mask].max().item()

                log.info("Embedding: mean=%.3f std=%.3f norm=%.3f | Sim: mean=%.3f min=%.3f max=%.3f",
                         emb_mean, emb_std, emb_norm, sim_mean, sim_min, sim_max)

                np.save(ckpt_dir / f"sample_embs_v2_epoch_{epoch:04d}.npy", embs.cpu().numpy())
            model.train()

    log.info("Training complete. Best loss: %.4f", best_loss)

    torch.save({
        'epoch': start_epoch + args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'loss': best_loss,
        'config': {
            'img_size': 128, 'patch_size': 16, 'embed_dim': 256, 'output_dim': 256,
            'encoder_depth': 6, 'predictor_depth': 3,
        },
    }, ckpt_dir / "jepa_v2_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="checkpoints/jepa_v2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigreg-weight", type=float, default=1.0)
    parser.add_argument("--gram-weight", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args)
