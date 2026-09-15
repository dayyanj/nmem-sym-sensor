"""Lightweight diffusion decoder for visual mind's eye.

Reconstructs 64×64 RGB images from 512-dim MoE visual embeddings.
Used for dream inspection, imagination, and debugging what the system "sees."

Architecture:
    - Conditioning: 512-dim embedding → AdaGroupNorm into each UNet block
    - UNet: 4 levels (64→32→16→8), ~8M params
    - Scheduler: DDPM training, DDIM inference (20 steps default, 4 for fast preview)
    - Output: 64×64×3 RGB

Training:
    - Encode diverse images with MoE visual encoder → 512-dim
    - Train decoder to reconstruct from embeddings
    - Loss: MSE on predicted noise (standard diffusion objective)
    - Perceptual bonus: VGG feature matching at denoised intermediate

Based on DDPM (Ho et al. 2020) with DDIM sampling (Song et al. 2020).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Noise schedule ──────────────────────────────────────────

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed in 'Improved DDPM'."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Precomputed noise schedule tensors."""

    def __init__(self, timesteps: int = 1000, device: str = "cpu"):
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        ).to(device)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to x0 at timestep t."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        return self


# ── UNet building blocks ────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Timestep embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaGN(nn.Module):
    """Adaptive Group Norm — modulates features with embedding conditioning."""
    def __init__(self, channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale, shift = self.proj(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class ResBlock(nn.Module):
    """Residual block with AdaGN conditioning."""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = AdaGN(in_ch, cond_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGN(out_ch, cond_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x, cond)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h, cond)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial features."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ── UNet ────────────────────────────────────────────────────

class VisualDiffusionUNet(nn.Module):
    """Lightweight UNet for 64×64 image generation conditioned on 512-dim embedding.

    Architecture:
        Down: 64→32→16→8 (channels: 128→256→256→512)
        Mid: 512 with self-attention
        Up: 8→16→32→64 (channels: 512→256→256→128)
        Self-attention at 16×16 and 8×8 resolutions

    ~8M parameters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        cond_dim: int = 512,
        base_channels: int = 64,
        channel_mult: tuple = (1, 1, 2, 2, 4),
        attn_resolutions: tuple = (16, 8),
        image_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.cond_dim = cond_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Visual embedding projection (512 → time_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_channels
        current_res = image_size

        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch, out_ch, time_dim, dropout),
                ResBlock(out_ch, out_ch, time_dim, dropout),
                SelfAttention(out_ch) if current_res in attn_resolutions else nn.Identity(),
            ]))
            if i < len(channel_mult) - 1:
                self.down_samples.append(Downsample(out_ch))
                current_res //= 2
            else:
                self.down_samples.append(nn.Identity())
            ch = out_ch

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult
            # Skip connection doubles the channels
            skip_ch = ch + out_ch if i > 0 else ch + ch
            self.up_blocks.append(nn.ModuleList([
                ResBlock(skip_ch, out_ch, time_dim, dropout),
                ResBlock(out_ch, out_ch, time_dim, dropout),
                SelfAttention(out_ch) if current_res in attn_resolutions else nn.Identity(),
            ]))
            if i < len(channel_mult) - 1:
                self.up_samples.append(Upsample(out_ch))
                current_res *= 2
            else:
                self.up_samples.append(nn.Identity())
            ch = out_ch

        # Output conv
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise given noisy image x, timestep t, and conditioning embedding."""
        # Combine time and visual conditioning
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        emb = t_emb + c_emb  # additive conditioning

        # Input
        h = self.conv_in(x)

        # Downsampling with skip connections
        skips = [h]
        for (res1, res2, attn), down in zip(self.down_blocks, self.down_samples):
            h = res1(h, emb)
            h = res2(h, emb)
            h = attn(h) if not isinstance(attn, nn.Identity) else h
            skips.append(h)
            h = down(h)

        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Upsampling with skip connections
        for (res1, res2, attn), up in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = res1(h, emb)
            h = res2(h, emb)
            h = attn(h) if not isinstance(attn, nn.Identity) else h
            h = up(h)

        return self.conv_out(h)


# ── Sampling ────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    model: VisualDiffusionUNet,
    schedule: DiffusionSchedule,
    cond: torch.Tensor,
    steps: int = 20,
    shape: tuple = (1, 3, 128, 128),
    eta: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    """DDIM sampling for fast inference.

    Args:
        model: trained UNet
        schedule: noise schedule
        cond: (B, 512) conditioning embedding
        steps: number of denoising steps (20 for quality, 4 for preview)
        shape: output image shape
        eta: 0 = deterministic DDIM, 1 = stochastic DDPM
    """
    model.eval()
    b = shape[0]

    # Subsequence of timesteps
    step_size = schedule.timesteps // steps
    timesteps = list(range(0, schedule.timesteps, step_size))[::-1]

    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch, cond)

        alpha_t = schedule.alphas_cumprod[t]
        alpha_prev = schedule.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

        # Predicted x0
        x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev - eta**2 * (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) * pred_noise

        # Random noise
        noise = torch.randn_like(x) if eta > 0 and i + 1 < len(timesteps) else 0

        x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
        if eta > 0 and i + 1 < len(timesteps):
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            x = x + sigma * noise

    return x.clamp(-1, 1)


# ── Convenience ─────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cond_dim: int = 512, device: str = "cuda") -> tuple:
    """Build model + schedule, return (model, schedule)."""
    model = VisualDiffusionUNet(cond_dim=cond_dim).to(device)
    schedule = DiffusionSchedule(timesteps=1000, device=device)
    return model, schedule
