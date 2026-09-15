"""Hand-crafted geometric embedding for foveal crops — 320 dimensions.

Extracts interpretable visual features from small image patches using
quadrant-aware spatial encoding. Each quadrant gets its own HoGC histogram
and colour summary, so spatial layout within the crop is preserved.

Sections (all L2-normalised independently, then weighted):
  1. Quadrant HoGC       (4 quadrants x 10 bins = 40 dims) weight 6.0
  2. Global HoGC + stats (10 + 12 = 22 dims)               weight 4.0
  3. Border ownership     (14 dims)                         weight 3.0
  4. Quadrant colour      (4 quadrants x 4 features = 16)   weight 3.0
  5. Global colour        (24 dims)                         weight 2.0
  6. Shading              (14 dims)                         weight 1.5
  7. Border context       (14 dims)                         weight 1.0
  -> Raw total: 144 dims, zero-padded to 320 for future expansion.

No training required. Designed for foveal crops (32-128px) where the
system sees one primitive at a time.

References:
    CPDA curvature estimation:
        Chetverikov, D. & Szabo, Z. (2003). "A Simple and Efficient Algorithm
        for Detection of High Curvature Points in Planar Curves." Uses
        chord-to-point distance accumulation — robust to noise, no derivatives.
        See: doi.org/10.1016/S0167-8655(01)00063-0

    HoGC (Histogram of Gradient Curvature):
        Ommer, B. et al. "Beyond Straight Lines — Object Detection Using
        Curvature." Extends HOG with curvature histograms that capture shape
        information HOG misses. Curvature is orthogonal to gradient orientation.
        See: ommer-lab.com/wp-content/uploads/2021/10/Beyond-Straight-Lines

    Border ownership (V2 neurons):
        Zhou, H., Friedman, H.S. & von der Heydt, R. (2000). "Coding of
        Border Ownership in Monkey Visual Cortex." Neurons in V2 signal which
        side of an edge is figure vs ground using local cues: convexity,
        T-junctions, surroundedness, colour continuity. Computed within 60-100ms.
        See: Journal of Neuroscience, 20(17), 6594-6611

    Section normalisation approach:
        Ensures each feature category contributes independently to similarity,
        preventing high-magnitude sections from drowning subtle features.
        Similar principle to block normalisation in HOG (Dalal & Triggs, 2005).

    Quadrant-aware encoding:
        Computing HoGC per spatial quadrant captures where curvature appears
        within the crop. A handle (curve on one side) differs from a full
        circle (curves everywhere). Without spatial decomposition, global
        stats wash out these layout differences.

Usage:
    encoder = GeometricEncoder()
    embedding = encoder.encode(crop_bgr)  # np.ndarray (320,)
"""
from __future__ import annotations

import cv2
import numpy as np

# HoGC bin edges: from straight (0) to sharp corner (2.0)
_HOGC_BINS = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 2.0])
_N_HOGC_BINS = 10


def _hogc_histogram(curvatures: np.ndarray) -> np.ndarray:
    """Compute a 10-bin HoGC histogram from curvature values."""
    hist = np.zeros(_N_HOGC_BINS, dtype=np.float32)
    if len(curvatures) == 0:
        return hist
    for i in range(_N_HOGC_BINS):
        hist[i] = ((curvatures >= _HOGC_BINS[i]) & (curvatures < _HOGC_BINS[i + 1])).sum()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _section_normalise(vec: np.ndarray, weight: float) -> np.ndarray:
    """L2-normalise a section and apply weight."""
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        return (vec / norm) * weight
    return vec


class GeometricEncoder:
    """Extract geometric + photometric features from a foveal crop.

    Output: 320-dim L2-normalised embedding with quadrant-aware spatial encoding.
    """

    EMBEDDING_DIM = 320

    def __init__(self):
        self._chord_lengths = [5, 10, 20]
        self._sample_dist = 8

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Encode a BGR crop to a 320-dim feature vector.

        Args:
            crop_bgr: (H, W, 3) uint8 BGR image, any size.

        Returns:
            (320,) float32 L2-normalised embedding.
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, w = gray.shape
        mid_h, mid_w = h // 2, w // 2

        # Compute CPDA curvatures + border ownership (shared computation)
        curvatures, curvature_signs, border_ownership, quadrant_curvatures = (
            self._compute_curvature_and_ownership(gray, h, w, mid_h, mid_w)
        )

        # Build sections
        sections = []
        weights = []

        # 1. Quadrant HoGC (4 x 10 = 40 dims)
        q_hogc = np.zeros(40, dtype=np.float32)
        for qi in range(4):
            q_hogc[qi * 10:(qi + 1) * 10] = _hogc_histogram(quadrant_curvatures[qi])
        sections.append(q_hogc)
        weights.append(6.0)

        # 2. Global HoGC + curvature stats (22 dims)
        global_hogc_stats = self._global_hogc_stats(curvatures, curvature_signs)
        sections.append(global_hogc_stats)
        weights.append(4.0)

        # 3. Border ownership (14 dims)
        bo_features = self._border_ownership_features(border_ownership)
        sections.append(bo_features)
        weights.append(3.0)

        # 4. Quadrant colour (4 x 4 = 16 dims)
        q_colour = self._quadrant_colour(hsv, mid_h, mid_w)
        sections.append(q_colour)
        weights.append(3.0)

        # 5. Global colour (24 dims)
        g_colour = self._global_colour(hsv)
        sections.append(g_colour)
        weights.append(2.0)

        # 6. Shading (14 dims)
        shading = self._shading(gray, hsv, h)
        sections.append(shading)
        weights.append(1.5)

        # 7. Border context (14 dims)
        border_ctx = self._border_context(gray, h, w)
        sections.append(border_ctx)
        weights.append(1.0)

        # Normalise each section independently, apply weights
        normed = []
        for sec, wt in zip(sections, weights):
            normed.append(_section_normalise(sec, wt))

        raw = np.concatenate(normed).astype(np.float32)

        # Pad to 320 dims
        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        embedding[:len(raw)] = raw

        # Final L2 normalise
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm

        return embedding

    # ── Core: CPDA curvature + border ownership ───────────

    def _compute_curvature_and_ownership(
        self, gray: np.ndarray, h: int, w: int, mid_h: int, mid_w: int,
    ) -> tuple[np.ndarray, np.ndarray, list[dict], list[np.ndarray]]:
        """Compute per-point curvature and border ownership along all contours.

        Returns:
            curvatures: all curvature values
            curvature_signs: signed curvature (concave vs convex)
            border_ownership: list of per-point ownership dicts
            quadrant_curvatures: [q0, q1, q2, q3] arrays of curvatures per quadrant
        """
        edges = cv2.Canny(gray.astype(np.uint8), 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        curvatures = []
        curvature_signs = []
        border_ownership = []
        # Quadrants: TL=0, TR=1, BL=2, BR=3
        quadrant_curvatures = [[] for _ in range(4)]

        max_chord = max(self._chord_lengths)

        for cnt in contours:
            pts = cnt.squeeze()
            if pts.ndim < 2 or len(pts) < max_chord * 2 + 1:
                continue
            pts_f = pts.astype(np.float64)

            for j in range(max_chord, len(pts_f) - max_chord):
                # CPDA curvature
                cpda_sum = 0.0
                sign_sum = 0.0
                n_chords = 0
                for L in self._chord_lengths:
                    p_before = pts_f[j - L]
                    p_after = pts_f[j + L]
                    p_curr = pts_f[j]
                    chord = p_after - p_before
                    chord_len = np.linalg.norm(chord)
                    if chord_len < 1e-6:
                        continue
                    v = p_curr - p_before
                    cross = chord[0] * v[1] - chord[1] * v[0]
                    dist = abs(cross) / chord_len
                    kappa = dist / (chord_len * 0.5)
                    cpda_sum += kappa
                    sign_sum += np.sign(cross) * kappa
                    n_chords += 1

                if n_chords == 0:
                    continue

                kappa_avg = cpda_sum / n_chords
                sign_avg = sign_sum / n_chords
                curvatures.append(kappa_avg)
                curvature_signs.append(sign_avg)

                # Assign to quadrant based on point position
                px, py = int(pts_f[j][0]), int(pts_f[j][1])
                qi = (0 if py < mid_h else 2) + (0 if px < mid_w else 1)
                quadrant_curvatures[qi].append(kappa_avg)

                # Border ownership
                if 1 <= j < len(pts_f) - 1:
                    bo = self._compute_border_ownership_point(
                        pts_f, j, gray, h, w, sign_avg,
                    )
                    if bo is not None:
                        border_ownership.append(bo)

        curvatures = np.array(curvatures, dtype=np.float32) if curvatures else np.array([], dtype=np.float32)
        curvature_signs = np.array(curvature_signs, dtype=np.float32) if curvature_signs else np.array([], dtype=np.float32)
        quadrant_curvatures = [
            np.array(qc, dtype=np.float32) if qc else np.array([], dtype=np.float32)
            for qc in quadrant_curvatures
        ]

        return curvatures, curvature_signs, border_ownership, quadrant_curvatures

    def _compute_border_ownership_point(
        self, pts: np.ndarray, j: int, gray: np.ndarray,
        h: int, w: int, sign_avg: float,
    ) -> dict | None:
        """Compute V2-inspired border ownership for a single contour point."""
        tangent = pts[j + 1] - pts[j - 1]
        tang_len = np.linalg.norm(tangent)
        if tang_len < 1e-6:
            return None
        tangent /= tang_len
        normal = np.array([-tangent[1], tangent[0]])

        px, py = int(pts[j][0]), int(pts[j][1])
        side_a_vals = []
        side_b_vals = []
        for d in range(2, self._sample_dist + 1):
            ax, ay = int(px + normal[0] * d), int(py + normal[1] * d)
            bx, by = int(px - normal[0] * d), int(py - normal[1] * d)
            if 0 <= ax < w and 0 <= ay < h:
                side_a_vals.append(float(gray[ay, ax]))
            if 0 <= bx < w and 0 <= by < h:
                side_b_vals.append(float(gray[by, bx]))

        if not side_a_vals or not side_b_vals:
            return None

        mean_a = np.mean(side_a_vals)
        mean_b = np.mean(side_b_vals)
        std_a = np.std(side_a_vals) if len(side_a_vals) > 1 else 0
        std_b = np.std(side_b_vals) if len(side_b_vals) > 1 else 0

        return {
            'brightness_diff': (mean_a - mean_b) / 255.0,
            'texture_diff': (std_a - std_b) / max(std_a + std_b, 1.0),
            'convex_sign': np.sign(sign_avg) if abs(sign_avg) > 1e-6 else 0,
            'mean_a': mean_a / 255.0,
            'mean_b': mean_b / 255.0,
            'std_a': std_a / 255.0,
            'std_b': std_b / 255.0,
        }

    # ── Section builders ──────────────────────────────────

    def _global_hogc_stats(
        self, curvatures: np.ndarray, curvature_signs: np.ndarray,
    ) -> np.ndarray:
        """Global HoGC histogram (10) + curvature summary stats (12) = 22 dims."""
        features = np.zeros(22, dtype=np.float32)
        if len(curvatures) == 0:
            return features

        # Global HoGC
        features[:10] = _hogc_histogram(curvatures)

        # Summary stats
        features[10] = curvatures.mean()
        features[11] = curvatures.std()
        features[12] = np.median(curvatures)
        features[13] = curvatures.max()
        features[14] = (curvatures < 0.02).mean()    # fraction straight
        features[15] = (curvatures > 0.1).mean()     # fraction curved
        features[16] = (curvatures > 0.5).mean()     # fraction corner-like

        # Signed curvature
        features[17] = (curvature_signs > 0.02).mean()    # fraction convex
        features[18] = (curvature_signs < -0.02).mean()   # fraction concave
        features[19] = curvature_signs.mean()

        # Curvature continuity
        if len(curvatures) > 2:
            diff = np.abs(np.diff(curvatures))
            features[20] = diff.mean()
            features[21] = diff.max()

        return features

    def _border_ownership_features(self, border_ownership: list[dict]) -> np.ndarray:
        """V2-inspired border ownership (14 dims)."""
        features = np.zeros(14, dtype=np.float32)
        if not border_ownership:
            return features

        bo = border_ownership
        bright = np.array([b['brightness_diff'] for b in bo])
        tex = np.array([b['texture_diff'] for b in bo])
        convex = np.array([b['convex_sign'] for b in bo])
        mean_as = np.array([b['mean_a'] for b in bo])
        mean_bs = np.array([b['mean_b'] for b in bo])
        std_as = np.array([b['std_a'] for b in bo])
        std_bs = np.array([b['std_b'] for b in bo])

        features[0] = bright.mean()
        features[1] = abs(bright.mean())
        features[2] = (bright > 0.05).mean()
        features[3] = (bright < -0.05).mean()
        features[4] = tex.mean()
        features[5] = abs(tex.mean())
        features[6] = (std_as > std_bs).mean()
        features[7] = convex.mean()
        features[8] = abs(convex.mean())

        # Owner side properties
        owner_is_a = (tex.mean() > 0) if abs(tex.mean()) > 0.05 else (convex.mean() > 0)
        if owner_is_a:
            features[9] = mean_as.mean()
            features[10] = std_as.mean()
            features[11] = mean_bs.mean()
            features[12] = std_bs.mean()
        else:
            features[9] = mean_bs.mean()
            features[10] = std_bs.mean()
            features[11] = mean_as.mean()
            features[12] = std_as.mean()

        features[13] = abs(features[9] - features[11])  # figure-ground contrast

        return features

    def _quadrant_colour(self, hsv: np.ndarray, mid_h: int, mid_w: int) -> np.ndarray:
        """Per-quadrant colour summary (4 x 4 = 16 dims).

        Each quadrant: [dominant_hue_cos, dominant_hue_sin, mean_sat, mean_val]
        """
        features = np.zeros(16, dtype=np.float32)
        quadrants = [
            hsv[:mid_h, :mid_w],    # TL
            hsv[:mid_h, mid_w:],    # TR
            hsv[mid_h:, :mid_w],    # BL
            hsv[mid_h:, mid_w:],    # BR
        ]

        for qi, q in enumerate(quadrants):
            if q.size == 0:
                continue
            hue = q[:, :, 0]   # 0-180 in OpenCV
            sat = q[:, :, 1]
            val = q[:, :, 2]

            # Chromatic pixels only
            chromatic = sat > 30
            if chromatic.any():
                h_rad = hue[chromatic] * (np.pi / 90.0)  # 0-180 -> 0-2pi
                features[qi * 4] = np.cos(h_rad).mean()
                features[qi * 4 + 1] = np.sin(h_rad).mean()
            features[qi * 4 + 2] = sat.mean() / 255.0
            features[qi * 4 + 3] = val.mean() / 255.0

        return features

    def _global_colour(self, hsv: np.ndarray) -> np.ndarray:
        """Global colour features (24 dims): 16-bin hue hist + 8 saturation stats."""
        features = np.zeros(24, dtype=np.float32)

        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]

        # Hue histogram (16 bins, chromatic pixels only)
        chromatic = sat > 30
        if chromatic.any():
            hist, _ = np.histogram(hue[chromatic], bins=16, range=(0, 180))
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist /= hist.sum()
            features[:16] = hist

        # Saturation profile
        sat_norm = sat / 255.0
        features[16] = sat_norm.mean()
        features[17] = sat_norm.std()
        features[18] = np.median(sat_norm)
        features[19] = (sat_norm > 0.5).mean()
        features[20] = (sat_norm < 0.1).mean()
        features[21] = np.percentile(sat_norm, 90)
        features[22] = np.percentile(sat_norm, 10)
        features[23] = chromatic.mean()

        return features

    def _shading(self, gray: np.ndarray, hsv: np.ndarray, h: int) -> np.ndarray:
        """Brightness gradient + specular/shadow cues (14 dims)."""
        features = np.zeros(14, dtype=np.float32)
        gray_norm = gray / 255.0

        gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=5)
        grad_mag = np.sqrt(gx**2 + gy**2)

        features[0] = gx.mean()
        features[1] = gy.mean()
        features[2] = grad_mag.mean()
        features[3] = grad_mag.std()

        # Brightness stats
        features[4] = gray_norm.mean()
        features[5] = gray_norm.std()
        features[6] = np.percentile(gray_norm, 5)
        features[7] = np.percentile(gray_norm, 95)
        features[8] = features[7] - features[6]  # dynamic range

        # Specular + shadow
        val = hsv[:, :, 2] / 255.0
        sat = hsv[:, :, 1] / 255.0
        features[9] = ((val > 0.9) & (sat < 0.2)).mean()   # specular fraction
        features[10] = ((val < 0.2) & (sat < 0.3)).mean()  # shadow fraction

        # Gradient coherence (structure tensor)
        Ixx = gx * gx
        Iyy = gy * gy
        Ixy = gx * gy
        k = max(h // 8, 3) | 1
        Sxx = cv2.GaussianBlur(Ixx, (k, k), 0)
        Syy = cv2.GaussianBlur(Iyy, (k, k), 0)
        Sxy = cv2.GaussianBlur(Ixy, (k, k), 0)
        trace = Sxx + Syy
        det = Sxx * Syy - Sxy * Sxy
        disc = np.sqrt(np.maximum(trace**2 - 4 * det, 0))
        l1 = (trace + disc) / 2
        l2 = (trace - disc) / 2
        denom = (l1 + l2)**2
        with np.errstate(invalid='ignore'):
            coherence = np.where(denom > 1e-10, (l1 - l2)**2 / denom, 0)
        features[11] = coherence.mean()
        features[12] = coherence.std()
        features[13] = (coherence > 0.5).mean()

        return features

    def _border_context(self, gray: np.ndarray, h: int, w: int) -> np.ndarray:
        """Edge directions at crop boundaries + symmetry (14 dims)."""
        features = np.zeros(14, dtype=np.float32)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)

        bw = max(h // 8, 2)

        # Edge strength at each border (4 dims)
        features[0] = magnitude[:bw, :].mean()      # top
        features[1] = magnitude[-bw:, :].mean()      # bottom
        features[2] = magnitude[:, :bw].mean()        # left
        features[3] = magnitude[:, -bw:].mean()       # right

        # Cross-border continuity (2 dims)
        features[4] = min(features[0], features[1])   # top-bottom
        features[5] = min(features[2], features[3])   # left-right

        # Symmetry (4 dims)
        left_half = gray[:, :w // 2]
        right_half = cv2.flip(gray[:, w // 2:], 1)
        min_w = min(left_half.shape[1], right_half.shape[1])
        if min_w > 0:
            features[6] = 1.0 - np.abs(
                left_half[:, :min_w].astype(float) - right_half[:, :min_w].astype(float)
            ).mean() / 255.0

        top_half = gray[:h // 2, :]
        bottom_half = cv2.flip(gray[h // 2:, :], 0)
        min_h = min(top_half.shape[0], bottom_half.shape[0])
        if min_h > 0:
            features[7] = 1.0 - np.abs(
                top_half[:min_h, :].astype(float) - bottom_half[:min_h, :].astype(float)
            ).mean() / 255.0

        # Center vs periphery (4 dims)
        border_ring = np.zeros_like(gray, dtype=bool)
        border_ring[:bw, :] = True
        border_ring[-bw:, :] = True
        border_ring[:, :bw] = True
        border_ring[:, -bw:] = True
        center = ~border_ring

        if center.any() and border_ring.any():
            features[8] = gray[center].mean() / 255.0
            features[9] = gray[border_ring].mean() / 255.0
            features[10] = features[8] - features[9]
            features[11] = magnitude[center].mean()
            features[12] = magnitude[border_ring].mean()
            features[13] = features[11] - features[12]

        return features


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def structural_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity using only curvature + border ownership sections.

    After section normalisation, the first 76 dims are:
      quadrant HoGC (40) + global HoGC+stats (22) + border ownership (14).
    """
    a_struct = a[:76]
    b_struct = b[:76]
    na = np.linalg.norm(a_struct)
    nb = np.linalg.norm(b_struct)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a_struct, b_struct) / (na * nb))
