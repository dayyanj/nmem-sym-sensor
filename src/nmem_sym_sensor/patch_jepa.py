"""
PatchJEPA — Joint Embedding Predictive Architecture for visual primitives.

Adapted from SliceJEPA (medical vision) for 2D image understanding.
Learns by predicting masked patch embeddings from visible context.

Architecture:
  Patch embed: 128×128 image → 64 patches of 16×16 → 256-dim each
  Context encoder: 6-layer ViT over visible patches
  Predictor: 3-layer transformer, predicts target patch embeddings
  SIGReg: prevents representation collapse without EMA
  Output: mean-pool all patch embeddings → 512-dim image embedding

Training:
  1. Split image into patches, randomly mask 40-60%
  2. Context encoder processes visible patches
  3. Predictor predicts masked patch embeddings from context
  4. Loss = MSE(predicted, target) + λ * SIGReg(all_embeddings)
  5. No labels, no negatives, no augmentation tricks needed

Progression:
  Phase 1: primitives (edges, curves, colors)
  Phase 2: compound objects (overlapping shapes, varied backgrounds)
  Phase 3: real scenes (COCO, faces, street views)
  Same architecture throughout — curriculum learning, not architecture change.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SIGReg — from SliceJEPA (medical vision)
# ---------------------------------------------------------------------------
class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.

    Prevents representation collapse by enforcing the embedding distribution
    is isotropic Gaussian. Uses moment matching (kurtosis + covariance).
    """

    def __init__(self, embed_dim: int, num_projections: int = 512):
        super().__init__()
        self.num_projections = num_projections
        directions = torch.randn(embed_dim, num_projections)
        directions = F.normalize(directions, dim=0)
        self.register_buffer('directions', directions)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=z.device)

        projections = z @ self.directions
        mu = projections.mean(dim=0, keepdim=True)
        std = projections.std(dim=0, keepdim=True).clamp(min=1e-6)
        standardized = (projections - mu) / std

        m2 = (standardized ** 2).mean(dim=0)
        m4 = (standardized ** 4).mean(dim=0)
        loss_m2 = ((m2 - 1.0) ** 2).mean()
        loss_m4 = ((m4 - 3.0) ** 2).mean()

        n_sub = min(64, self.num_projections)
        idx = torch.randperm(self.num_projections, device=z.device)[:n_sub]
        x_sub = standardized[:, idx]
        cov = (x_sub.T @ x_sub) / (B - 1)
        off_diag = cov - torch.eye(n_sub, device=z.device)
        loss_cov = (off_diag ** 2).mean()

        return loss_m2 + loss_m4 + loss_cov


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and embed each.

    128×128 → 8×8 grid of 16×16 patches → 64 tokens of patch_dim.
    Uses a CNN stem (not just linear projection) for better low-level features.
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # CNN stem — better than linear projection for capturing edges/textures
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),   # 16→8
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),           # 8→4
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),    # 4→2
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                               # 2→1
            nn.Flatten(),                                          # (B, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, N_patches, embed_dim)."""
        B, C, H, W = x.shape
        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)
        # patches: (B, C, grid_h, grid_w, patch_h, patch_w)
        patches = patches.contiguous().view(
            B, C, self.n_patches, self.patch_size, self.patch_size,
        )
        # patches: (B, C, N, pH, pW) → process each patch
        patch_embs = []
        for i in range(self.n_patches):
            patch = patches[:, :, i]  # (B, C, pH, pW)
            emb = self.stem(patch)    # (B, embed_dim)
            patch_embs.append(emb)

        return torch.stack(patch_embs, dim=1)  # (B, N, embed_dim)


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# PatchJEPA
# ---------------------------------------------------------------------------
class PatchJEPA(nn.Module):
    """Joint Embedding Predictive Architecture for images.

    Given a partially masked image:
    1. Encode visible patches with context encoder
    2. Predict masked patch embeddings from context
    3. Compare predictions to target embeddings
    4. SIGReg prevents collapse

    The context encoder learns rich patch representations.
    Mean-pooling all patches gives a 512-dim image embedding.
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 256,
        output_dim: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 4,
        predictor_depth: int = 3,
        predictor_heads: int = 4,
        mask_ratio: float = 0.5,
        sigreg_weight: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.mask_ratio = mask_ratio
        self.sigreg_weight = sigreg_weight

        n_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)

        # Learned position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

        # Mask token (replaces masked patches in predictor input)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Context encoder — processes visible patches
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads, dropout=dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Predictor — predicts masked from visible context
        self.predictor = nn.ModuleList([
            TransformerBlock(embed_dim, predictor_heads, dropout=dropout)
            for _ in range(predictor_depth)
        ])
        self.predictor_norm = nn.LayerNorm(embed_dim)
        self.predictor_proj = nn.Linear(embed_dim, embed_dim)

        # Target projector (asymmetry — prevents trivial solution)
        self.target_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # Output projection: embed_dim → output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # SIGReg
        self.sigreg = SIGReg(embed_dim)

        self.n_patches = n_patches

    def generate_mask(self, B: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate random mask indices.

        Returns (visible_indices, masked_indices) for each sample in batch.
        Uses the same mask across the batch for simplicity.
        """
        N = self.n_patches
        n_masked = int(N * self.mask_ratio)
        n_visible = N - n_masked

        perm = torch.randperm(N, device=device)
        visible = perm[:n_visible].sort().values
        masked = perm[n_visible:].sort().values

        return visible, masked

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoding (no masking) for inference.

        x: (B, 3, H, W) → (B, output_dim)
        """
        patches = self.patch_embed(x)  # (B, N, embed_dim)
        patches = patches + self.pos_embed

        for block in self.encoder:
            patches = block(patches)
        patches = self.encoder_norm(patches)

        # Mean pool → output projection
        pooled = patches.mean(dim=1)  # (B, embed_dim)
        return self.output_proj(pooled)  # (B, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training forward pass with masking.

        x: (B, 3, H, W)

        Returns dict with pred_loss, sigreg_loss, total_loss.
        """
        B = x.shape[0]

        # 1. Embed all patches
        all_patches = self.patch_embed(x)  # (B, N, embed_dim)
        all_patches = all_patches + self.pos_embed

        # 2. Generate mask
        visible_idx, masked_idx = self.generate_mask(B, x.device)

        # 3. Encode visible patches only
        visible_patches = all_patches[:, visible_idx]  # (B, n_vis, D)
        for block in self.encoder:
            visible_patches = block(visible_patches)
        visible_patches = self.encoder_norm(visible_patches)

        # 4. Prepare predictor input: visible embeddings + mask tokens at masked positions
        n_vis = visible_idx.shape[0]
        n_mask = masked_idx.shape[0]

        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        # Add position info to mask tokens
        mask_tokens = mask_tokens + self.pos_embed[:, masked_idx]

        # Concatenate: [visible_encoded, mask_tokens]
        pred_input = torch.cat([visible_patches, mask_tokens], dim=1)

        # 5. Predict
        for block in self.predictor:
            pred_input = block(pred_input)
        pred_input = self.predictor_norm(pred_input)

        # Extract predictions for masked positions (last n_mask tokens)
        predictions = self.predictor_proj(pred_input[:, n_vis:])  # (B, n_mask, D)

        # 6. Target: encode masked patches (detached — no gradient to target)
        with torch.no_grad():
            target_patches = all_patches[:, masked_idx]  # (B, n_mask, D)
            # Re-encode targets through full encoder for better targets
            for block in self.encoder:
                target_patches = block(target_patches)
            target_patches = self.encoder_norm(target_patches)

        # Apply target projector (per-patch)
        targets_flat = target_patches.reshape(B * n_mask, self.embed_dim)
        targets_proj = self.target_proj(targets_flat)
        targets = targets_proj.reshape(B, n_mask, self.embed_dim)

        # 7. Prediction loss
        pred_loss = F.mse_loss(predictions, targets.detach())

        # 8. SIGReg on full encoder output (all patches, no masking)
        with torch.no_grad():
            full_patches = all_patches.clone()
        for block in self.encoder:
            full_patches = block(full_patches)
        full_patches = self.encoder_norm(full_patches)
        # Pool to image-level for SIGReg
        image_embs = full_patches.mean(dim=1)  # (B, embed_dim)
        sigreg_loss = self.sigreg(image_embs)

        total_loss = pred_loss + self.sigreg_weight * sigreg_loss

        return {
            'pred_loss': pred_loss,
            'sigreg_loss': sigreg_loss,
            'total_loss': total_loss,
            'n_visible': n_vis,
            'n_masked': n_mask,
        }

    def get_embedding_dim(self) -> int:
        return self.output_dim


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
