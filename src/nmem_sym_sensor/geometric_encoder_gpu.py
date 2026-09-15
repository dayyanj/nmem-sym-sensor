"""GPU-accelerated geometric encoder — 320 dimensions, zero learnable parameters.

Reimplements the hand-crafted GeometricEncoder as a pure torch.nn.Module with
fixed-weight convolution kernels. All operations are dense tensor ops that run
on GPU via ONNX Runtime or PyTorch CUDA.

Key reformulations from the CPU encoder:
  - CPDA curvature → Dense Hessian eigenvalue curvature field (conv2d)
  - findContours → Gradient magnitude thresholding (no sequential flood fill)
  - Border ownership per-point → CCL-inspired region asymmetry via directional blurs
  - Per-quadrant HoGC → Quadrant masks applied to dense curvature, then soft histograms

All convolution kernels (Sobel, Hessian, Gaussian) are registered as buffers —
no learnable parameters. The module exports cleanly to ONNX.

Output is designed to be comparable (not identical) to the CPU GeometricEncoder.
The dense Hessian curvature correlates with CPDA but is computed differently,
so cosine similarities between GPU and CPU embeddings will be high but not 1.0.

Usage:
    encoder = GeometricEncoderGPU()
    embedding = encoder(crop_tensor)  # (B, 3, H, W) → (B, 320)

    # Or via the convenience wrapper:
    encoder = GeometricEncoderGPU()
    embedding = encoder.encode_numpy(crop_bgr)  # (H, W, 3) uint8 → (320,) float32
"""
from __future__ import annotations

import math

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """1D Gaussian kernel."""
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    return g / g.sum()


def _make_2d_gaussian(size: int, sigma: float) -> torch.Tensor:
    """2D Gaussian kernel (size x size)."""
    g1d = _make_gaussian_kernel(size, sigma)
    return g1d.unsqueeze(1) * g1d.unsqueeze(0)


class GeometricEncoderGPU(nn.Module):
    """Dense tensor geometric encoder — zero learnable parameters.

    Sections (same layout as CPU GeometricEncoder):
      1. Quadrant HoGC       (40 dims)  weight 6.0
      2. Global HoGC + stats (22 dims)  weight 4.0
      3. Border ownership     (14 dims)  weight 3.0
      4. Quadrant colour      (16 dims)  weight 3.0
      5. Global colour        (24 dims)  weight 2.0
      6. Shading              (14 dims)  weight 1.5
      7. Border context       (14 dims)  weight 1.0
      → 144 raw dims, padded to 320.
    """

    EMBEDDING_DIM = 320
    N_HOGC_BINS = 10

    def __init__(self):
        super().__init__()

        # ── Fixed convolution kernels (registered as buffers) ──

        # Sobel 3x3
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # Sobel 5x5 (for shading — matches CPU encoder's ksize=5)
        sobel5_x = torch.tensor([
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1],
        ], dtype=torch.float32)
        sobel5_y = sobel5_x.t()
        self.register_buffer('sobel5_x', sobel5_x.view(1, 1, 5, 5))
        self.register_buffer('sobel5_y', sobel5_y.view(1, 1, 5, 5))

        # Second-order derivative kernels (Hessian components)
        # Ixx: d²I/dx²
        d2x = torch.tensor([[1, -2, 1]], dtype=torch.float32).view(1, 1, 1, 3)
        # Iyy: d²I/dy²
        d2y = torch.tensor([[1], [-2], [1]], dtype=torch.float32).view(1, 1, 3, 1)
        # Ixy: d²I/dxdy (Sobel-based cross derivative)
        d2xy = torch.tensor([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) * 0.25
        self.register_buffer('d2x', d2x)
        self.register_buffer('d2y', d2y)
        self.register_buffer('d2xy', d2xy)

        # Gaussian blur for structure tensor smoothing
        gauss = _make_2d_gaussian(7, 1.5)
        self.register_buffer('gauss_kernel', gauss.view(1, 1, 7, 7))

        # Directional blur kernels for border ownership (asymmetric Gaussians)
        # Left/right side sampling — 1D horizontal Gaussian, offset
        side_kernel_size = 9
        half = side_kernel_size // 2
        x = torch.arange(side_kernel_size, dtype=torch.float32) - half
        sigma = 2.0
        g = torch.exp(-x**2 / (2 * sigma**2))
        # Side A: positive direction only
        side_a = g.clone()
        side_a[:half] = 0
        side_a = side_a / (side_a.sum() + 1e-8)
        # Side B: negative direction only
        side_b = g.clone()
        side_b[half + 1:] = 0
        side_b = side_b / (side_b.sum() + 1e-8)
        # Horizontal
        self.register_buffer('side_a_h', side_a.view(1, 1, 1, side_kernel_size))
        self.register_buffer('side_b_h', side_b.view(1, 1, 1, side_kernel_size))
        # Vertical
        self.register_buffer('side_a_v', side_a.view(1, 1, side_kernel_size, 1))
        self.register_buffer('side_b_v', side_b.view(1, 1, side_kernel_size, 1))

        # HoGC bin edges
        self.register_buffer('hogc_bins', torch.tensor(
            [0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 2.0],
            dtype=torch.float32,
        ))

        # Hue histogram bin edges (16 bins over 0-180)
        self.register_buffer('hue_bins', torch.linspace(0, 180, 17))

        # Section weights
        self.register_buffer('section_weights', torch.tensor(
            [6.0, 4.0, 3.0, 3.0, 2.0, 1.5, 1.0], dtype=torch.float32,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode BGR image tensor to 320-dim embedding.

        Args:
            x: (B, 3, H, W) float32, values in [0, 1]. Channel order: BGR.

        Returns:
            (B, 320) float32, L2-normalised.
        """
        B, C, H, W = x.shape
        mid_h, mid_w = H // 2, W // 2

        # Convert to grayscale and HSV-like channels
        # BGR → gray: 0.114*B + 0.587*G + 0.299*R
        gray = x[:, 0:1] * 0.114 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.299  # (B, 1, H, W)

        # BGR → approximate HSV
        hue, sat, val = self._bgr_to_hsv(x)  # each (B, 1, H, W)

        # ── Dense curvature field (Hessian eigenvalues) ──
        curvature, edge_mask = self._dense_curvature(gray, H, W)  # (B, 1, H, W) each

        # ── Quadrant masks ──
        q_masks = self._quadrant_masks(B, H, W, mid_h, mid_w, x.device)  # (4, B, 1, H, W)

        # Build sections
        sec1 = self._quadrant_hogc(curvature, edge_mask, q_masks)     # (B, 40)
        sec2 = self._global_hogc_stats(curvature, edge_mask)           # (B, 22)
        sec3 = self._border_ownership(gray, edge_mask, H, W)           # (B, 14)
        sec4 = self._quadrant_colour(hue, sat, val, q_masks)           # (B, 16)
        sec5 = self._global_colour(hue, sat)                            # (B, 24)
        sec6 = self._shading(gray, hue, sat, val, H)                   # (B, 14)
        sec7 = self._border_context(gray, H, W)                        # (B, 14)

        # Section-normalise and weight
        sections = [sec1, sec2, sec3, sec4, sec5, sec6, sec7]
        normed = []
        for i, sec in enumerate(sections):
            norm = sec.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normed.append((sec / norm) * self.section_weights[i])

        raw = torch.cat(normed, dim=1)  # (B, 144)

        # Pad to 320
        embedding = F.pad(raw, (0, self.EMBEDDING_DIM - raw.shape[1]))

        # Final L2 normalise
        norm = embedding.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return embedding / norm

    # ── BGR to HSV (differentiable approximation) ──

    def _bgr_to_hsv(self, x: torch.Tensor):
        """BGR [0,1] → (hue [0,180], sat [0,255], val [0,255]) as float tensors."""
        b, g, r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        val_raw = torch.max(torch.max(r, g), b)  # [0,1]
        min_raw = torch.min(torch.min(r, g), b)
        delta = (val_raw - min_raw).clamp(min=1e-8)

        # Saturation
        sat_raw = torch.where(val_raw > 1e-8, delta / val_raw, torch.zeros_like(val_raw))

        # Hue (0-180 OpenCV scale)
        hue = torch.zeros_like(val_raw)
        # R is max
        r_mask = (val_raw == r)
        hue = torch.where(r_mask, 30.0 * ((g - b) / delta), hue)
        # G is max
        g_mask = (val_raw == g) & ~r_mask
        hue = torch.where(g_mask, 60.0 + 30.0 * ((b - r) / delta), hue)
        # B is max
        b_mask = ~r_mask & ~g_mask
        hue = torch.where(b_mask, 120.0 + 30.0 * ((r - g) / delta), hue)
        hue = hue % 180.0

        return hue, sat_raw * 255.0, val_raw * 255.0

    # ── Dense curvature via gradient orientation change ──

    def _dense_curvature(self, gray: torch.Tensor, H: int, W: int):
        """Compute dense curvature via local gradient orientation variance.

        Straight lines have locally parallel gradients -> low orientation variance.
        Curves have fanning gradients -> high orientation variance.
        Corners have abrupt direction changes -> very high variance.

        Uses the structure tensor in a local window: curvature = 1 - coherence.
        Coherence high for straight edges, low for curved edges.

        Multi-scale: average curvature at 3 Gaussian smoothing scales to match
        the CPU encoder's multi-chord-length CPDA approach.

        Returns:
            curvature: (B, 1, H, W) curvature [0, ~2]
            edge_mask: (B, 1, H, W) binary edge mask
        """
        # First derivatives
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2)

        # Edge mask (adaptive threshold)
        gm_flat = grad_mag.reshape(grad_mag.shape[0], -1)
        threshold = gm_flat.mean(dim=1) + gm_flat.std(dim=1)
        threshold = threshold.view(-1, 1, 1, 1)
        edge_mask = (grad_mag > threshold).float()

        # Gradient outer products (shared across scales)
        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy

        # Multi-scale curvature from structure tensor
        # Small sigma catches corners, large sigma catches broad curves
        curvature_sum = torch.zeros_like(gray)
        for pad in [3, 5, 9]:
            ks = pad * 2 + 1
            sigma = pad / 2.0
            g = _make_2d_gaussian(ks, sigma).to(gray.device).view(1, 1, ks, ks)

            Sxx = F.conv2d(gxx, g, padding=pad)
            Syy = F.conv2d(gyy, g, padding=pad)
            Sxy = F.conv2d(gxy, g, padding=pad)

            trace = Sxx + Syy
            det = Sxx * Syy - Sxy * Sxy
            disc = torch.sqrt((trace**2 - 4 * det).clamp(min=0))
            l1 = (trace + disc) / 2
            l2 = (trace - disc) / 2

            # Coherence: 1 for straight edges, 0 for isotropic
            denom = (l1 + l2).clamp(min=1e-8)
            coherence = ((l1 - l2) / denom) ** 2

            # Curvature = 1 - coherence (straight=0, curved=high)
            curvature_sum = curvature_sum + (1.0 - coherence)

        # Average across scales, clamp to [0, 2]
        curvature = (curvature_sum / 3.0).clamp(0, 2.0)

        return curvature, edge_mask

    # ── Quadrant masks ──

    def _quadrant_masks(self, B, H, W, mid_h, mid_w, device):
        """Create 4 quadrant masks: TL, TR, BL, BR. Returns (4, B, 1, H, W)."""
        mask = torch.zeros(4, 1, 1, H, W, device=device)
        mask[0, :, :, :mid_h, :mid_w] = 1.0  # TL
        mask[1, :, :, :mid_h, mid_w:] = 1.0   # TR
        mask[2, :, :, mid_h:, :mid_w] = 1.0   # BL
        mask[3, :, :, mid_h:, mid_w:] = 1.0   # BR
        return mask.expand(-1, B, -1, -1, -1)  # (4, B, 1, H, W)

    # ── Soft histogram ──

    def _soft_histogram(self, values: torch.Tensor, mask: torch.Tensor,
                        bin_edges: torch.Tensor) -> torch.Tensor:
        """Differentiable histogram using soft binning.

        Args:
            values: (B, 1, H, W) values to histogram
            mask: (B, 1, H, W) weight mask (edge pixels)
            bin_edges: (N+1,) bin edges

        Returns:
            (B, N) normalised histogram
        """
        n_bins = len(bin_edges) - 1
        # Flatten spatial dims
        v = (values * mask).reshape(values.shape[0], -1)   # (B, HW)
        m = mask.reshape(mask.shape[0], -1)                  # (B, HW)

        hists = []
        for i in range(n_bins):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            # Soft bin: sigmoid ramps at edges (temperature=50 for near-hard binning)
            in_bin = torch.sigmoid((v - low) * 50) * torch.sigmoid((high - v) * 50)
            hists.append((in_bin * m).sum(dim=1, keepdim=True))

        hist = torch.cat(hists, dim=1)  # (B, N)
        total = hist.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return hist / total

    # ── Section 1: Quadrant HoGC (40 dims) ──

    def _quadrant_hogc(self, curvature, edge_mask, q_masks):
        """HoGC histogram per quadrant: 4 × 10 = 40 dims."""
        parts = []
        for qi in range(4):
            qm = q_masks[qi]  # (B, 1, H, W)
            combined_mask = edge_mask * qm
            hist = self._soft_histogram(curvature, combined_mask, self.hogc_bins)  # (B, 10)
            parts.append(hist)
        return torch.cat(parts, dim=1)  # (B, 40)

    # ── Section 2: Global HoGC + stats (22 dims) ──

    def _global_hogc_stats(self, curvature, edge_mask):
        """Global HoGC histogram (10) + curvature summary stats (12) = 22 dims."""
        # Global HoGC
        hist = self._soft_histogram(curvature, edge_mask, self.hogc_bins)  # (B, 10)

        # Masked curvature values for stats
        c_masked = curvature * edge_mask  # zero where no edge
        n_edge = edge_mask.reshape(edge_mask.shape[0], -1).sum(dim=1).clamp(min=1)  # (B,)

        c_flat = c_masked.reshape(c_masked.shape[0], -1)  # (B, HW)
        m_flat = edge_mask.reshape(edge_mask.shape[0], -1)

        # Weighted stats (only over edge pixels)
        c_sum = c_flat.sum(dim=1)
        c_mean = c_sum / n_edge  # (B,)
        c_var = ((c_flat - c_mean.unsqueeze(1))**2 * m_flat).sum(dim=1) / n_edge
        c_std = c_var.sqrt()
        c_max = c_flat.max(dim=1).values

        # Fraction stats
        frac_straight = ((c_flat < 0.02) * m_flat).sum(dim=1) / n_edge
        frac_curved = ((c_flat > 0.1) * m_flat).sum(dim=1) / n_edge
        frac_corner = ((c_flat > 0.5) * m_flat).sum(dim=1) / n_edge

        # Curvature continuity — approximate via spatial neighbours
        # Difference between adjacent edge pixels (horizontal)
        c_diff_h = (c_masked[:, :, :, 1:] - c_masked[:, :, :, :-1]).abs()
        e_both_h = edge_mask[:, :, :, 1:] * edge_mask[:, :, :, :-1]
        n_pairs = e_both_h.reshape(e_both_h.shape[0], -1).sum(dim=1).clamp(min=1)
        diff_mean = (c_diff_h * e_both_h).reshape(c_diff_h.shape[0], -1).sum(dim=1) / n_pairs
        diff_max = (c_diff_h * e_both_h).reshape(c_diff_h.shape[0], -1).max(dim=1).values

        # Stack: 10 (hist) + 12 (stats)
        stats = torch.stack([
            c_mean, c_std, c_mean,  # median ≈ mean for dense field
            c_max,
            frac_straight, frac_curved, frac_corner,
            # Signed curvature: Hessian gives unsigned, so approximate
            torch.zeros_like(c_mean),  # frac convex (placeholder)
            torch.zeros_like(c_mean),  # frac concave (placeholder)
            torch.zeros_like(c_mean),  # net direction (placeholder)
            diff_mean, diff_max,
        ], dim=1)  # (B, 12)

        return torch.cat([hist, stats], dim=1)  # (B, 22)

    # ── Section 3: Border ownership (14 dims) ──

    def _border_ownership(self, gray, edge_mask, H, W):
        """Dense border ownership via directional asymmetry at edges."""
        features = torch.zeros(gray.shape[0], 14, device=gray.device)

        # Side-A and Side-B mean brightness (horizontal asymmetry)
        side_a_h = F.conv2d(gray, self.side_a_h, padding=(0, 4))
        side_b_h = F.conv2d(gray, self.side_b_h, padding=(0, 4))
        # Side-A and Side-B (vertical asymmetry)
        side_a_v = F.conv2d(gray, self.side_a_v, padding=(4, 0))
        side_b_v = F.conv2d(gray, self.side_b_v, padding=(4, 0))

        # Combined asymmetry at edge pixels
        bright_diff_h = (side_a_h - side_b_h) * edge_mask
        bright_diff_v = (side_a_v - side_b_v) * edge_mask
        bright_diff = (bright_diff_h + bright_diff_v) / 2.0

        n_edge = edge_mask.reshape(edge_mask.shape[0], -1).sum(dim=1).clamp(min=1)
        bd_flat = bright_diff.reshape(bright_diff.shape[0], -1)
        em_flat = edge_mask.reshape(edge_mask.shape[0], -1)

        # Brightness asymmetry stats
        bd_mean = (bd_flat * em_flat).sum(dim=1) / n_edge
        features[:, 0] = bd_mean
        features[:, 1] = bd_mean.abs()
        features[:, 2] = ((bd_flat > 0.02) * em_flat).sum(dim=1) / n_edge  # frac A brighter
        features[:, 3] = ((bd_flat < -0.02) * em_flat).sum(dim=1) / n_edge  # frac B brighter

        # Texture asymmetry (using local variance as proxy)
        gray_sq = gray ** 2
        side_a_var_h = F.conv2d(gray_sq, self.side_a_h, padding=(0, 4)) - side_a_h**2
        side_b_var_h = F.conv2d(gray_sq, self.side_b_h, padding=(0, 4)) - side_b_h**2
        tex_diff = (side_a_var_h - side_b_var_h) * edge_mask
        td_flat = tex_diff.reshape(tex_diff.shape[0], -1)
        td_mean = (td_flat * em_flat).sum(dim=1) / n_edge
        features[:, 4] = td_mean
        features[:, 5] = td_mean.abs()
        features[:, 6] = ((td_flat > 0) * em_flat).sum(dim=1) / n_edge  # frac A more textured

        # Owner side mean brightness and texture
        owner_is_a = (td_mean > 0).float()  # (B,)
        mean_a = (side_a_h * edge_mask).reshape(side_a_h.shape[0], -1).sum(dim=1) / n_edge
        mean_b = (side_b_h * edge_mask).reshape(side_b_h.shape[0], -1).sum(dim=1) / n_edge
        var_a = (side_a_var_h.clamp(min=0) * edge_mask).reshape(side_a_var_h.shape[0], -1).sum(dim=1) / n_edge
        var_b = (side_b_var_h.clamp(min=0) * edge_mask).reshape(side_b_var_h.shape[0], -1).sum(dim=1) / n_edge

        features[:, 9] = owner_is_a * mean_a + (1 - owner_is_a) * mean_b
        features[:, 10] = owner_is_a * var_a.sqrt() + (1 - owner_is_a) * var_b.sqrt()
        features[:, 11] = (1 - owner_is_a) * mean_a + owner_is_a * mean_b
        features[:, 12] = (1 - owner_is_a) * var_a.sqrt() + owner_is_a * var_b.sqrt()
        features[:, 13] = (features[:, 9] - features[:, 11]).abs()

        return features

    # ── Section 4: Quadrant colour (16 dims) ──

    def _quadrant_colour(self, hue, sat, val, q_masks):
        """Per-quadrant [hue_cos, hue_sin, mean_sat, mean_val] = 4×4 = 16 dims."""
        parts = []
        for qi in range(4):
            qm = q_masks[qi]  # (B, 1, H, W)
            n_px = qm.reshape(qm.shape[0], -1).sum(dim=1).clamp(min=1)

            # Chromatic mask (sat > 30/255 ≈ 0.118)
            chromatic = (sat > 30).float() * qm

            hue_rad = hue * (math.pi / 90.0)  # 0-180 → 0-2π
            n_chrom = chromatic.reshape(chromatic.shape[0], -1).sum(dim=1).clamp(min=1)
            hue_cos = (torch.cos(hue_rad) * chromatic).reshape(hue_rad.shape[0], -1).sum(dim=1) / n_chrom
            hue_sin = (torch.sin(hue_rad) * chromatic).reshape(hue_rad.shape[0], -1).sum(dim=1) / n_chrom
            mean_sat = (sat * qm).reshape(sat.shape[0], -1).sum(dim=1) / (n_px * 255.0)
            mean_val = (val * qm).reshape(val.shape[0], -1).sum(dim=1) / (n_px * 255.0)

            parts.append(torch.stack([hue_cos, hue_sin, mean_sat, mean_val], dim=1))

        return torch.cat(parts, dim=1)  # (B, 16)

    # ── Section 5: Global colour (24 dims) ──

    def _global_colour(self, hue, sat):
        """16-bin hue histogram + 8 saturation stats = 24 dims."""
        chromatic = (sat > 30).float()
        hist = self._soft_histogram(hue, chromatic, self.hue_bins)  # (B, 16)

        sat_norm = sat / 255.0
        s_flat = sat_norm.reshape(sat_norm.shape[0], -1)
        c_flat = chromatic.reshape(chromatic.shape[0], -1)

        s_mean = s_flat.mean(dim=1)
        s_std = s_flat.std(dim=1)
        s_median = s_flat.median(dim=1).values
        frac_vivid = (s_flat > 0.5).float().mean(dim=1)
        frac_gray = (s_flat < 0.1).float().mean(dim=1)
        s_p90 = s_flat.quantile(0.9, dim=1)
        s_p10 = s_flat.quantile(0.1, dim=1)
        frac_chrom = c_flat.mean(dim=1)

        stats = torch.stack([
            s_mean, s_std, s_median, frac_vivid,
            frac_gray, s_p90, s_p10, frac_chrom,
        ], dim=1)  # (B, 8)

        return torch.cat([hist, stats], dim=1)  # (B, 24)

    # ── Section 6: Shading (14 dims) ──

    def _shading(self, gray, hue, sat, val, H):
        """Brightness gradient + specular/shadow + coherence = 14 dims."""
        gx = F.conv2d(gray, self.sobel5_x, padding=2)
        gy = F.conv2d(gray, self.sobel5_y, padding=2)
        grad_mag = torch.sqrt(gx**2 + gy**2)

        g_flat = gray.reshape(gray.shape[0], -1)

        features = torch.zeros(gray.shape[0], 14, device=gray.device)
        features[:, 0] = gx.reshape(gx.shape[0], -1).mean(dim=1)
        features[:, 1] = gy.reshape(gy.shape[0], -1).mean(dim=1)
        features[:, 2] = grad_mag.reshape(grad_mag.shape[0], -1).mean(dim=1)
        features[:, 3] = grad_mag.reshape(grad_mag.shape[0], -1).std(dim=1)

        features[:, 4] = g_flat.mean(dim=1)
        features[:, 5] = g_flat.std(dim=1)
        features[:, 6] = g_flat.quantile(0.05, dim=1)
        features[:, 7] = g_flat.quantile(0.95, dim=1)
        features[:, 8] = features[:, 7] - features[:, 6]

        # Specular + shadow
        val_norm = val / 255.0
        sat_norm = sat / 255.0
        features[:, 9] = ((val_norm > 0.9 * 255) & (sat_norm < 0.2 * 255)).float().reshape(gray.shape[0], -1).mean(dim=1)
        features[:, 10] = ((val_norm < 0.2 * 255) & (sat_norm < 0.3 * 255)).float().reshape(gray.shape[0], -1).mean(dim=1)

        # Gradient coherence (structure tensor)
        Sxx = F.conv2d(gx * gx, self.gauss_kernel, padding=3)
        Syy = F.conv2d(gy * gy, self.gauss_kernel, padding=3)
        Sxy = F.conv2d(gx * gy, self.gauss_kernel, padding=3)
        trace = Sxx + Syy
        det = Sxx * Syy - Sxy * Sxy
        disc = torch.sqrt((trace**2 - 4 * det).clamp(min=0))
        l1 = (trace + disc) / 2
        l2 = (trace - disc) / 2
        denom = (l1 + l2)**2
        coherence = torch.where(denom > 1e-10, (l1 - l2)**2 / denom, torch.zeros_like(denom))

        c_flat = coherence.reshape(coherence.shape[0], -1)
        features[:, 11] = c_flat.mean(dim=1)
        features[:, 12] = c_flat.std(dim=1)
        features[:, 13] = (c_flat > 0.5).float().mean(dim=1)

        return features

    # ── Section 7: Border context (14 dims) ──

    def _border_context(self, gray, H, W):
        """Edge strength at borders + symmetry + center vs periphery = 14 dims."""
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2)

        bw = max(H // 8, 2)
        features = torch.zeros(gray.shape[0], 14, device=gray.device)

        # Border edge strength (4 dims)
        features[:, 0] = magnitude[:, :, :bw, :].reshape(magnitude.shape[0], -1).mean(dim=1)
        features[:, 1] = magnitude[:, :, -bw:, :].reshape(magnitude.shape[0], -1).mean(dim=1)
        features[:, 2] = magnitude[:, :, :, :bw].reshape(magnitude.shape[0], -1).mean(dim=1)
        features[:, 3] = magnitude[:, :, :, -bw:].reshape(magnitude.shape[0], -1).mean(dim=1)

        # Cross-border continuity (2 dims)
        features[:, 4] = torch.min(features[:, 0], features[:, 1])
        features[:, 5] = torch.min(features[:, 2], features[:, 3])

        # Horizontal symmetry
        left = gray[:, :, :, :W // 2]
        right = gray[:, :, :, W // 2:].flip(dims=[3])
        min_w = min(left.shape[3], right.shape[3])
        if min_w > 0:
            features[:, 6] = 1.0 - (left[:, :, :, :min_w] - right[:, :, :, :min_w]).abs().reshape(gray.shape[0], -1).mean(dim=1)

        # Vertical symmetry
        top = gray[:, :, :H // 2, :]
        bottom = gray[:, :, H // 2:, :].flip(dims=[2])
        min_h = min(top.shape[2], bottom.shape[2])
        if min_h > 0:
            features[:, 7] = 1.0 - (top[:, :, :min_h, :] - bottom[:, :, :min_h, :]).abs().reshape(gray.shape[0], -1).mean(dim=1)

        # Center vs periphery
        border_mask = torch.ones_like(gray)
        border_mask[:, :, bw:-bw, bw:-bw] = 0  # 1 at border, 0 at center
        center_mask = 1.0 - border_mask

        n_center = center_mask.reshape(gray.shape[0], -1).sum(dim=1).clamp(min=1)
        n_border = border_mask.reshape(gray.shape[0], -1).sum(dim=1).clamp(min=1)

        center_bright = (gray * center_mask).reshape(gray.shape[0], -1).sum(dim=1) / n_center
        border_bright = (gray * border_mask).reshape(gray.shape[0], -1).sum(dim=1) / n_border
        center_edge = (magnitude * center_mask).reshape(magnitude.shape[0], -1).sum(dim=1) / n_center
        border_edge = (magnitude * border_mask).reshape(magnitude.shape[0], -1).sum(dim=1) / n_border

        features[:, 8] = center_bright
        features[:, 9] = border_bright
        features[:, 10] = center_bright - border_bright
        features[:, 11] = center_edge
        features[:, 12] = border_edge
        features[:, 13] = center_edge - border_edge

        return features

    # ── Convenience: numpy interface ──

    def encode_numpy(self, crop_bgr: numpy.ndarray) -> numpy.ndarray:
        """Encode a single BGR uint8 crop. Returns (320,) float32 numpy array."""
        import numpy as np
        tensor = torch.from_numpy(
            crop_bgr.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0)
        if next(self.buffers()).is_cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            emb = self(tensor)
        return emb[0].cpu().numpy()
