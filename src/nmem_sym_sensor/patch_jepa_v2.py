"""
PatchJEPA v2 — Improved visual encoder with DINOv3-inspired techniques.

Changes from v1:
  1. CLS token (replaces mean-pool for 256-dim output)
  2. EMA teacher encoder (student/teacher self-distillation)
  3. Local-to-global crops (student sees crop, teacher sees full image)
  4. Saliency-weighted masking (predict structure, not background)
  5. Gram anchoring (preserve local patch features during extended training)
  6. SIGReg on CLS token (prevents collapse)

Architecture:
  Patch embed: CNN stem, 16×16 patches → 256-dim each
  CLS token: learned, attends to all patches → 256-dim output
  Context encoder: 6-layer ViT
  Predictor: 3-layer ViT
  Teacher: EMA copy of encoder (momentum 0.996)
  Output: 256-dim (combines with 256-dim geometric encoder → 512 total)
"""
from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SIGReg
# ---------------------------------------------------------------------------
class SIGReg(nn.Module):
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
# Patch Embedding (CNN stem)
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, self.n_patches, self.patch_size, self.patch_size)
        patch_embs = []
        for i in range(self.n_patches):
            emb = self.stem(patches[:, :, i])
            patch_embs.append(emb)
        return torch.stack(patch_embs, dim=1)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.1):
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

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# PatchJEPA v2
# ---------------------------------------------------------------------------
class PatchJEPAv2(nn.Module):

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 256,
        output_dim: int = 256,
        encoder_depth: int = 6,
        encoder_heads: int = 4,
        predictor_depth: int = 3,
        predictor_heads: int = 4,
        mask_ratio: float = 0.5,
        sigreg_weight: float = 1.0,
        gram_weight: float = 0.1,
        dropout: float = 0.1,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.mask_ratio = mask_ratio
        self.sigreg_weight = sigreg_weight
        self.gram_weight = gram_weight
        self.ema_momentum = ema_momentum

        n_patches = (img_size // patch_size) ** 2
        self.n_patches = n_patches

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

        # CLS token — learned aggregation token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Context encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads, dropout=dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Predictor
        self.predictor = nn.ModuleList([
            TransformerBlock(embed_dim, predictor_heads, dropout=dropout)
            for _ in range(predictor_depth)
        ])
        self.predictor_norm = nn.LayerNorm(embed_dim)
        self.predictor_proj = nn.Linear(embed_dim, embed_dim)

        # Target projector (asymmetry)
        self.target_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # Output projection: CLS token → output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # SIGReg on CLS token
        self.sigreg = SIGReg(embed_dim)

        # EMA teacher — deep copy, no gradient
        self.teacher_encoder = None  # initialised on first forward
        self.teacher_patch_embed = None

        # Gram teacher — saved at epoch N, frozen
        self.gram_teacher = None
        self.gram_active = False

    def _init_teacher(self):
        """Initialise EMA teacher as a copy of the student."""
        self.teacher_encoder = nn.ModuleList([
            copy.deepcopy(block) for block in self.encoder
        ])
        self.teacher_patch_embed = copy.deepcopy(self.patch_embed)
        self.teacher_norm = copy.deepcopy(self.encoder_norm)
        # Freeze teacher
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_patch_embed.parameters():
            p.requires_grad = False
        for p in self.teacher_norm.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_teacher(self):
        """EMA update: teacher = momentum * teacher + (1-momentum) * student."""
        m = self.ema_momentum
        for t_block, s_block in zip(self.teacher_encoder, self.encoder):
            for t_p, s_p in zip(t_block.parameters(), s_block.parameters()):
                t_p.data.mul_(m).add_(s_p.data, alpha=1.0 - m)
        for t_p, s_p in zip(self.teacher_patch_embed.parameters(), self.patch_embed.parameters()):
            t_p.data.mul_(m).add_(s_p.data, alpha=1.0 - m)
        for t_p, s_p in zip(self.teacher_norm.parameters(), self.encoder_norm.parameters()):
            t_p.data.mul_(m).add_(s_p.data, alpha=1.0 - m)

    def save_gram_teacher(self):
        """Save current encoder state as Gram anchor."""
        self.gram_teacher = nn.ModuleList([
            copy.deepcopy(block) for block in self.encoder
        ])
        self.gram_teacher_norm = copy.deepcopy(self.encoder_norm)
        for p in self.gram_teacher.parameters():
            p.requires_grad = False
        for p in self.gram_teacher_norm.parameters():
            p.requires_grad = False
        self.gram_active = True
        log.info("Gram teacher saved (anchoring local features)")

    def _saliency_weighted_mask(
        self, patches: torch.Tensor, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask patches weighted by gradient energy (prefer masking structure)."""
        N = self.n_patches
        n_masked = int(N * self.mask_ratio)

        # Compute per-patch energy (variance as proxy for structure)
        with torch.no_grad():
            # patches: (B, N, D) — use variance of embedding dims as saliency
            energy = patches.var(dim=2).mean(dim=0)  # (N,)
            # Add small uniform noise so identical patches don't all get the same rank
            energy = energy + torch.rand_like(energy) * 0.01
            # Higher energy = more interesting = more likely to be masked
            probs = F.softmax(energy * 5.0, dim=0)

        # Sample mask indices weighted by saliency
        masked = torch.multinomial(probs, n_masked, replacement=False).sort().values
        all_idx = torch.arange(N, device=device)
        visible_mask = torch.ones(N, dtype=torch.bool, device=device)
        visible_mask[masked] = False
        visible = all_idx[visible_mask]

        return visible, masked

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoding for inference — returns 256-dim CLS token output."""
        B = x.shape[0]
        patches = self.patch_embed(x) + self.pos_embed

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # (B, 1+N, D)

        for block in self.encoder:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)

        # CLS token is the first token
        cls_out = tokens[:, 0]  # (B, embed_dim)
        return self.output_proj(cls_out)  # (B, output_dim)

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Full patch-level encoding — returns (B, N, embed_dim)."""
        patches = self.patch_embed(x) + self.pos_embed
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        for block in self.encoder:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)
        return tokens[:, 1:]  # skip CLS, return patch tokens

    def forward(
        self, x_student: torch.Tensor, x_teacher: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            x_student: (B, 3, H, W) — student input (crop or full image)
            x_teacher: (B, 3, H, W) — teacher input (full image). If None, same as student.
        """
        if x_teacher is None:
            x_teacher = x_student

        B = x_student.shape[0]

        # Initialise teacher on first call
        if self.teacher_encoder is None:
            self._init_teacher()

        # 1. Student: embed patches + mask + encode visible
        student_patches = self.patch_embed(x_student) + self.pos_embed
        visible_idx, masked_idx = self._saliency_weighted_mask(student_patches, x_student.device)

        visible_patches = student_patches[:, visible_idx]
        # Add CLS token to visible set
        cls = self.cls_token.expand(B, -1, -1)
        visible_with_cls = torch.cat([cls, visible_patches], dim=1)

        for block in self.encoder:
            visible_with_cls = block(visible_with_cls)
        visible_with_cls = self.encoder_norm(visible_with_cls)

        student_cls = visible_with_cls[:, 0]  # CLS output
        visible_encoded = visible_with_cls[:, 1:]  # patch outputs

        # 2. Predictor: predict masked patches from visible context
        n_vis = visible_idx.shape[0]
        n_mask = masked_idx.shape[0]
        mask_tokens = self.mask_token.expand(B, n_mask, -1) + self.pos_embed[:, masked_idx]
        pred_input = torch.cat([visible_encoded, mask_tokens], dim=1)

        for block in self.predictor:
            pred_input = block(pred_input)
        pred_input = self.predictor_norm(pred_input)
        predictions = self.predictor_proj(pred_input[:, n_vis:])  # (B, n_mask, D)

        # 3. Teacher: encode FULL teacher image (no masking, no gradient)
        with torch.no_grad():
            teacher_patches = self.teacher_patch_embed(x_teacher) + self.pos_embed
            teacher_cls = self.cls_token.expand(B, -1, -1)
            teacher_tokens = torch.cat([teacher_cls, teacher_patches], dim=1)
            for block in self.teacher_encoder:
                teacher_tokens = block(teacher_tokens)
            teacher_tokens = self.teacher_norm(teacher_tokens)
            teacher_patch_out = teacher_tokens[:, 1:]  # (B, N, D)

        # Target: teacher's output for masked positions
        target_patches = teacher_patch_out[:, masked_idx]  # (B, n_mask, D)
        targets_flat = target_patches.reshape(B * n_mask, self.embed_dim)
        targets_proj = self.target_proj(targets_flat)
        targets = targets_proj.reshape(B, n_mask, self.embed_dim)

        # 4. Prediction loss
        pred_loss = F.mse_loss(predictions, targets.detach())

        # 5. SIGReg on CLS tokens
        sigreg_loss = self.sigreg(student_cls)

        # 6. Gram anchoring loss (if active)
        gram_loss = torch.tensor(0.0, device=x_student.device)
        if self.gram_active and self.gram_teacher is not None:
            with torch.no_grad():
                gram_patches = self.patch_embed(x_student) + self.pos_embed
                gram_cls = self.cls_token.expand(B, -1, -1)
                gram_tokens = torch.cat([gram_cls, gram_patches], dim=1)
                for block in self.gram_teacher:
                    gram_tokens = block(gram_tokens)
                gram_tokens = self.gram_teacher_norm(gram_tokens)
                gram_patch_features = gram_tokens[:, 1:]  # (B, N, D)

            # Current student's full patch features (no masking)
            current_patches = self.patch_embed(x_student) + self.pos_embed
            current_cls = self.cls_token.expand(B, -1, -1)
            current_tokens = torch.cat([current_cls, current_patches], dim=1)
            for block in self.encoder:
                current_tokens = block(current_tokens)
            current_tokens = self.encoder_norm(current_tokens)
            current_patch_features = current_tokens[:, 1:]

            gram_loss = F.mse_loss(current_patch_features, gram_patch_features.detach())

        # Total loss
        total_loss = (
            pred_loss
            + self.sigreg_weight * sigreg_loss
            + self.gram_weight * gram_loss
        )

        # Update EMA teacher
        self._update_teacher()

        return {
            'pred_loss': pred_loss,
            'sigreg_loss': sigreg_loss,
            'gram_loss': gram_loss,
            'total_loss': total_loss,
            'n_visible': n_vis,
            'n_masked': n_mask,
        }

    def get_embedding_dim(self) -> int:
        return self.output_dim


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
