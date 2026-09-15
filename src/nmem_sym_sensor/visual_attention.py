"""
Foveal attention system for video frame sequences.

Mimics biological foveated vision: a high-resolution central region
concentrates processing on salient areas, with progressively lower
attention outward. The system decides WHERE to look using motion
detection, edge saliency, and temporal persistence, then delegates
detailed visual analysis to the existing stateless encoder.

Based on prior research: https://dj-ai.ai/?page=fovea-vision-for-computer-vision

Key mechanisms:
  1. Optical flow motion detection (DIS, FAST preset)
  2. Edge saliency (Canny + GaussianBlur → smooth heatmap)
  3. Composite saliency with temporal EMA smoothing
  4. Attention stickiness with switch margin (hysteresis)
  5. Kalman-filtered focal point tracking
  6. Resolution rings: full detail at fovea, reduced at periphery
  7. Optional face prior for educational videos

The existing analyze_frame_edge() is called on the cropped foveal
region at full resolution. It stays stateless and untouched.
"""
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from nmem_sym_sensor import config
from nmem_sym_sensor.visual import (
    FrameAnalysis,
    VisualPrimitive,
    _extract_spatial_relations,
    analyze_frame_edge,
)

log = logging.getLogger(__name__)


@dataclass
class FovealConfig:
    """Configuration for the foveal attention system."""
    # Motion detection
    motion_weight: float = 0.8
    edge_weight: float = 0.2

    # Temporal smoothing
    ema_alpha: float = 0.8          # higher = more weight on current frame

    # Attention stickiness
    switch_margin: float = 0.10     # 10% more salient to trigger switch
    min_dwell_frames: int = 4       # hold focus for at least N frames

    # Kalman filter
    kalman_blend: float = 0.4       # 40% prediction, 60% measurement

    # Resolution rings
    fovea_radius_min: int = 96
    fovea_radius_max: int = 256
    periphery_scale: float = 0.25   # downscale factor for peripheral analysis
    periphery_enabled: bool = True  # set False to skip periphery entirely

    # Face prior
    face_prior_enabled: bool = False
    face_prior_weight: float = 0.3

    # Multi-saccade
    max_saccades: int = 8           # max fixation points per frame
    saccade_suppression: float = 0.3  # suppress saliency within this fraction of fovea radius after each peak
    overlap_iou_threshold: float = 0.3  # merge crops with IoU above this

    # Performance
    flow_downscale: float = 0.5     # process optical flow at this fraction

    @classmethod
    def from_config(cls) -> "FovealConfig":
        """Build from environment-driven config.py settings."""
        return cls(
            motion_weight=getattr(config, "FOVEAL_MOTION_WEIGHT", 0.8),
            edge_weight=getattr(config, "FOVEAL_EDGE_WEIGHT", 0.2),
            ema_alpha=getattr(config, "FOVEAL_EMA_ALPHA", 0.8),
            switch_margin=getattr(config, "FOVEAL_SWITCH_MARGIN", 0.10),
            min_dwell_frames=getattr(config, "FOVEAL_MIN_DWELL", 4),
            kalman_blend=getattr(config, "FOVEAL_KALMAN_BLEND", 0.4),
            fovea_radius_min=getattr(config, "FOVEAL_RADIUS_MIN", 96),
            fovea_radius_max=getattr(config, "FOVEAL_RADIUS_MAX", 256),
            periphery_scale=getattr(config, "FOVEAL_PERIPHERY_SCALE", 0.25),
            periphery_enabled=getattr(config, "FOVEAL_PERIPHERY_SCALE", 0.25) > 0,
            face_prior_enabled=getattr(config, "FOVEAL_FACE_PRIOR", False),
            face_prior_weight=getattr(config, "FOVEAL_FACE_WEIGHT", 0.3),
            flow_downscale=getattr(config, "FOVEAL_FLOW_DOWNSCALE", 0.5),
        )


@dataclass
class FocalState:
    """Snapshot of current attention state for debugging/logging."""
    focal_x: float = 0.0
    focal_y: float = 0.0
    focal_radius: float = 128.0
    dwell_count: int = 0
    peak_saliency: float = 0.0
    motion_detected: bool = False
    frame_count: int = 0
    n_candidates: int = 0
    focused_candidate_id: int = -1


@dataclass
class AttentionCandidate:
    """A region of interest being tracked by the attention system.

    Lifecycle: detected → investigated → focused → habituated → decayed
    """
    id: int
    cx: float  # center x
    cy: float  # center y
    radius: float  # approximate size
    saliency: float  # current saliency score
    first_seen: int  # frame number
    last_seen: int  # frame number
    dwell_frames: int = 0  # how long we've focused on this
    habituated: bool = False  # tuned out (persists without change)
    velocity_x: float = 0.0  # estimated motion
    velocity_y: float = 0.0
    familiar: bool = False  # matched a known grounded cluster
    cluster_match_id: int | None = None
    cluster_label: str | None = None


class CandidateTracker:
    """Manages attention candidates — the regions competing for focus.

    Inspired by TABULA2's AttentionController. Replaces the broken
    Kalman filter + stickiness system with biologically plausible
    candidate management.

    Key behaviours:
    - Multiple candidates tracked simultaneously
    - New salient regions create candidates
    - Candidates that persist without change habituate (fade out)
    - Candidates that move are tracked (velocity estimation)
    - Focus goes to highest-priority non-habituated candidate
    - Habituated candidates can be re-activated by change
    - Edge-following saccades: when focused, the eye traces along
      the nearest contour rather than jumping to unrelated peaks
    """

    def __init__(
        self,
        max_candidates: int = 12,
        match_radius: float = 60.0,
        habituation_frames: int = 15,
        decay_frames: int = 30,
    ):
        self.max_candidates = max_candidates
        self.match_radius = match_radius
        self.habituation_frames = habituation_frames
        self.decay_frames = decay_frames

        self.candidates: list[AttentionCandidate] = []
        self.focused: AttentionCandidate | None = None
        self._next_id: int = 0
        self._frame: int = 0

        # Edge-following state
        self._edge_map: np.ndarray | None = None
        self._edge_follow_steps: int = 0  # how many follow steps since last peak-jump
        self._max_edge_follow: int = 6    # max contour-following steps before reverting to peaks

        # Coverage tracking — what fraction of the scene has the fovea visited?
        self._visited_cells: set[tuple[int, int]] = set()  # (grid_x, grid_y)
        self._total_salient_cells: int = 0
        self._coverage_grid_size: int = 8  # divide frame into 8x8 grid
        self._new_candidates_this_frame: int = 0

    def set_edge_map(self, edges: np.ndarray):
        """Provide the current frame's edge map (Canny output) for contour following."""
        self._edge_map = edges

    def _follow_edge(self, cx: float, cy: float, frame_w: int, frame_h: int) -> tuple[float, float] | None:
        """From current focus (cx, cy), find the next point along the nearest contour.

        Uses the Canny edge map to trace contours. Finds the strongest
        edge cluster in an annular ring around the current focus — close
        enough to be the same object, far enough to be a new fixation.

        Returns (next_x, next_y) or None if no edge to follow.
        """
        if self._edge_map is None:
            return None

        h, w = self._edge_map.shape[:2]
        ix, iy = int(cx), int(cy)

        # Search in an annular ring: inner=30px (skip where we already are),
        # outer=120px (don't jump too far — stay on same object)
        inner_r = 30
        outer_r = 120

        # Extract the ring region
        x1 = max(0, ix - outer_r)
        y1 = max(0, iy - outer_r)
        x2 = min(w, ix + outer_r)
        y2 = min(h, iy + outer_r)

        ring = self._edge_map[y1:y2, x1:x2].copy()
        if ring.size == 0:
            return None

        # Mask out the inner circle (where we already are)
        ring_h, ring_w = ring.shape
        cy_local = iy - y1
        cx_local = ix - x1
        yy, xx = np.ogrid[:ring_h, :ring_w]
        dist_sq = (xx - cx_local) ** 2 + (yy - cy_local) ** 2
        ring[dist_sq < inner_r ** 2] = 0

        # Blur to find edge clusters (not isolated pixels)
        ring_blurred = cv2.GaussianBlur(ring.astype(np.float32), (11, 11), 0)
        max_val = ring_blurred.max()
        if max_val < 10:  # no significant edges nearby
            return None

        # Find the strongest edge cluster in the ring
        peak_loc = np.unravel_index(ring_blurred.argmax(), ring_blurred.shape)
        next_y = peak_loc[0] + y1
        next_x = peak_loc[1] + x1

        return float(next_x), float(next_y)

    def update(
        self,
        peaks: list[tuple[float, float, float]],
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        """Update candidates from saliency peaks. Returns focal point (fx, fy).

        Args:
            peaks: List of (x, y, saliency) from multi-saccade peak finding.
            frame_w, frame_h: Frame dimensions.

        Returns:
            (focal_x, focal_y) — where the system is looking.
        """
        self._frame += 1

        # ── Edge-following: if currently focused, try to trace the contour ──
        edge_follow_point = None
        if (self.focused is not None
                and not self.focused.habituated
                and self._edge_follow_steps < self._max_edge_follow):
            edge_follow_point = self._follow_edge(
                self.focused.cx, self.focused.cy, frame_w, frame_h,
            )
            if edge_follow_point:
                efx, efy = edge_follow_point
                # Inject the edge-follow point as a high-priority peak
                # so the tracker treats it as a saliency detection
                peaks = list(peaks)  # copy
                peaks.insert(0, (efx, efy, self.focused.saliency * 0.9))
                self._edge_follow_steps += 1

        if edge_follow_point is None:
            # No edge to follow — reset counter
            self._edge_follow_steps = 0

        # ── Match peaks to existing candidates ──
        matched_ids: set[int] = set()
        unmatched_peaks: list[tuple[float, float, float]] = []

        for px, py, sal in peaks:
            best = None
            best_dist = self.match_radius
            for c in self.candidates:
                # Predict candidate's current position
                pred_x = c.cx + c.velocity_x
                pred_y = c.cy + c.velocity_y
                dist = math.sqrt((px - pred_x) ** 2 + (py - pred_y) ** 2)
                if dist < best_dist and c.id not in matched_ids:
                    best = c
                    best_dist = dist

            if best is not None:
                # Update existing candidate
                old_x, old_y = best.cx, best.cy
                alpha = 0.5  # smoothing
                best.cx = (1 - alpha) * best.cx + alpha * px
                best.cy = (1 - alpha) * best.cy + alpha * py
                best.velocity_x = best.cx - old_x
                best.velocity_y = best.cy - old_y
                best.saliency = max(best.saliency * 0.7, sal)  # decay old, take max
                best.last_seen = self._frame

                # Re-activate habituated candidate only if it MOVED
                # (not just because it's still bright — that's the definition of habituated)
                speed = math.sqrt(best.velocity_x ** 2 + best.velocity_y ** 2)
                if best.habituated and speed > 5.0:
                    best.habituated = False
                    best.dwell_frames = 0

                matched_ids.add(best.id)
            else:
                unmatched_peaks.append((px, py, sal))

        # ── Create new candidates from unmatched peaks ──
        self._new_candidates_this_frame = 0
        for px, py, sal in unmatched_peaks:
            if len(self.candidates) < self.max_candidates:
                c = AttentionCandidate(
                    id=self._next_id,
                    cx=px, cy=py,
                    radius=50.0,
                    saliency=sal,
                    first_seen=self._frame,
                    last_seen=self._frame,
                )
                self.candidates.append(c)
                self._next_id += 1
                self._new_candidates_this_frame += 1

        # ── Track foveal coverage (grid-based) ──
        for px, py, sal in peaks:
            gx = int(px / max(frame_w, 1) * self._coverage_grid_size)
            gy = int(py / max(frame_h, 1) * self._coverage_grid_size)
            gx = min(gx, self._coverage_grid_size - 1)
            gy = min(gy, self._coverage_grid_size - 1)
            self._visited_cells.add((gx, gy))

        # ── Habituate and decay ──
        surviving: list[AttentionCandidate] = []
        for c in self.candidates:
            age = self._frame - c.last_seen

            if age > self.decay_frames:
                continue  # Remove — not seen for too long

            # Familiar candidates habituate 3x faster — "I know this, move on"
            hab_threshold = self.habituation_frames // 3 if c.familiar else self.habituation_frames

            if c.id not in matched_ids:
                # Not matched this frame — it's static or gone
                c.dwell_frames += 1
                c.saliency *= 0.95  # gentle decay
                if c.dwell_frames >= hab_threshold:
                    c.habituated = True
            else:
                # Was matched — check if it moved
                speed = math.sqrt(c.velocity_x ** 2 + c.velocity_y ** 2)
                if speed < 2.0:
                    c.dwell_frames += 1
                    if c.dwell_frames >= hab_threshold:
                        c.habituated = True
                else:
                    # Moving — reset habituation
                    c.dwell_frames = 0
                    c.habituated = False

            surviving.append(c)

        self.candidates = surviving

        # ── Select focus: highest saliency non-habituated candidate ──
        active = [c for c in self.candidates if not c.habituated]

        if active:
            self.focused = max(active, key=lambda c: c.saliency)
            return self.focused.cx, self.focused.cy
        elif self.candidates:
            # All habituated — stay on most recent
            self.focused = max(self.candidates, key=lambda c: c.last_seen)
            return self.focused.cx, self.focused.cy
        else:
            # No candidates at all — center of frame
            self.focused = None
            return frame_w / 2, frame_h / 2

    @property
    def focused_id(self) -> int:
        return self.focused.id if self.focused else -1

    @property
    def candidate_coverage(self) -> float:
        """Fraction of the scene grid cells visited by the fovea (0-1)."""
        total = self._coverage_grid_size * self._coverage_grid_size
        return len(self._visited_cells) / total

    @property
    def new_candidates_this_frame(self) -> int:
        return self._new_candidates_this_frame

    def update_salient_cells(self, saliency_map: np.ndarray):
        """Update total salient cell count from current saliency map.

        Called by process_frame to establish the denominator for coverage.
        """
        h, w = saliency_map.shape
        gs = self._coverage_grid_size
        cell_h, cell_w = h // gs, w // gs
        count = 0
        for gy in range(gs):
            for gx in range(gs):
                cell = saliency_map[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
                if cell.max() > 0.05:
                    count += 1
        self._total_salient_cells = max(count, 1)

    def reset_coverage(self):
        """Reset coverage tracking on scene change."""
        self._visited_cells.clear()
        self._total_salient_cells = 0
        self._new_candidates_this_frame = 0


class FovealAttention:
    """Stateful foveal attention system for video frame sequences.

    Maintains inter-frame state: previous frame grayscale, EMA saliency
    map, Kalman filter state, focal point, dwell counter.

    Usage::

        attention = FovealAttention()
        for frame in video_frames:
            analysis = attention.process_frame(frame.image, frame_id)
            # analysis contains primitives from the attended region
    """

    def __init__(self, cfg: FovealConfig | None = None):
        self.cfg = cfg or FovealConfig.from_config()

        # Inter-frame state
        self._prev_gray: np.ndarray | None = None
        self._ema_saliency: np.ndarray | None = None
        self._focal_x: float = 0.0
        self._focal_y: float = 0.0
        self._current_peak: float = 0.0
        self._dwell_count: int = 0
        self._frame_count: int = 0
        self._fovea_radius: float = float(self.cfg.fovea_radius_max)

        # Candidate-based attention tracker (replaces Kalman filter)
        self._tracker = CandidateTracker(
            max_candidates=12,
            match_radius=60.0,
            habituation_frames=15,
            decay_frames=30,
        )

        # DIS optical flow (lazy init)
        self._flow_algo: cv2.DISOpticalFlow | None = None

        # Face cascade (lazy init)
        self._face_cascade: cv2.CascadeClassifier | None = None

    def reset(self):
        """Reset all inter-frame state. Call on scene changes."""
        self._prev_gray = None
        self._ema_saliency = None
        self._focal_x = 0.0
        self._focal_y = 0.0
        self._current_peak = 0.0
        self._dwell_count = 0
        self._tracker = CandidateTracker(
            max_candidates=12,
            match_radius=60.0,
            habituation_frames=15,
            decay_frames=30,
        )

    @property
    def focal_state(self) -> FocalState:
        return FocalState(
            focal_x=self._focal_x,
            focal_y=self._focal_y,
            focal_radius=self._fovea_radius,
            dwell_count=self._dwell_count,
            peak_saliency=self._current_peak,
            n_candidates=len(self._tracker.candidates),
            focused_candidate_id=self._tracker.focused_id,
            motion_detected=self._prev_gray is not None,
            frame_count=self._frame_count,
        )

    def process_frame(
        self,
        image: np.ndarray,
        frame_id: str = "frame_0",
        scene_change: bool = False,
        budget=None,
    ) -> FrameAnalysis:
        """Process a frame with foveal attention.

        1. Compute motion saliency (optical flow vs previous frame)
        2. Compute edge saliency (Canny + GaussianBlur)
        3. Composite + EMA temporal smoothing
        4. Find peak, apply stickiness + Kalman filter
        5. Crop foveal region, run analyze_frame_edge on it
        6. Optionally analyze periphery at reduced resolution
        7. Merge results with full-frame coordinates
        """
        if scene_change:
            self.reset()

        # Apply attention budget: override saccade/edge limits for this frame
        if budget is not None:
            frame_max_saccades = budget.max_saccades
            self._tracker._max_edge_follow = budget.edge_follow_steps
        else:
            frame_max_saccades = self.cfg.max_saccades
            self._tracker._max_edge_follow = 6  # default

        self._frame_count += 1
        h, w = image.shape[:2]

        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Initialize focal point to frame center on first frame
        if self._focal_x == 0.0 and self._focal_y == 0.0:
            self._focal_x = w / 2
            self._focal_y = h / 2

        # ── Step 1: Motion saliency ──────────────────────
        motion_sal = self._compute_motion_saliency(gray)

        # ── Step 2: Edge saliency ────────────────────────
        edge_sal = self._compute_edge_saliency(gray)

        # ── Step 3: Composite + EMA ──────────────────────
        # Adaptive rebalancing: when motion is negligible (static scene,
        # low fps), amplify edge weight so edges drive the fovea
        motion_magnitude = float(motion_sal.max())
        if motion_magnitude < 0.05:
            effective_motion_w = 0.0
            effective_edge_w = 1.0
        else:
            effective_motion_w = self.cfg.motion_weight
            effective_edge_w = self.cfg.edge_weight

        composite = (
            effective_motion_w * motion_sal
            + effective_edge_w * edge_sal
        )

        # Face prior
        if self.cfg.face_prior_enabled:
            face_sal = self._compute_face_prior(gray)
            composite = (
                composite * (1 - self.cfg.face_prior_weight)
                + face_sal * self.cfg.face_prior_weight
            )

        # Normalize to [0, 1] with specular highlight clamping
        # Clamp top 1% to prevent bright reflections from dominating
        p99 = np.percentile(composite, 99)
        if p99 > 0:
            composite = np.clip(composite, 0, p99) / p99

        # EMA temporal smoothing
        composite = self._ema_update(composite)

        # ── Step 4: Find saliency peaks → feed candidate tracker ──
        # Extract multiple peaks from the saliency map for the tracker.
        # The tracker manages candidates (detect → track → habituate → decay)
        # and returns the focal point of the highest-priority candidate.
        #
        # Provide the edge map so the tracker can follow contours
        # (edge-following saccades trace object outlines).
        edges_raw = cv2.Canny(gray, 50, 150)
        self._tracker.set_edge_map(edges_raw)

        peaks = self._find_saliency_peaks(composite, n=frame_max_saccades)
        fx, fy = self._tracker.update(peaks, w, h)
        self._focal_x = fx
        self._focal_y = fy
        self._current_peak = peaks[0][2] if peaks else 0.0

        # Update fovea radius based on saliency concentration
        self._fovea_radius = self._compute_fovea_radius(composite, fx, fy)

        # Store gray for next frame
        self._prev_gray = gray

        # ── Step 5: Multi-saccade analysis ─────────────────
        # Find top N saliency peaks and crop each one. Crops that
        # overlap spatially are merged into composite embeddings,
        # giving a holistic view of clustered features (e.g. a whole
        # triangle from its vertices) while keeping distant objects
        # as separate primitives.
        saccade_primitives, saccade_fixations = self._multi_saccade_analysis(
            image, composite, fx, fy, frame_id,
            max_saccades=frame_max_saccades,
        )

        # ── Step 6: Still-frame fallback ──────────────────
        # If multi-saccade found nothing (e.g. static image with no
        # motion), fall back to full-frame analysis. Many training
        # videos use still frames (color swatches, static flashcards).
        if not saccade_primitives:
            saccade_primitives = analyze_frame_edge(image, f"{frame_id}_full").primitives

        # ── Step 7: Peripheral analysis (optional) ───────
        peripheral_primitives = []
        if self.cfg.periphery_enabled and self.cfg.periphery_scale > 0:
            peripheral_primitives = self._analyze_periphery(
                image, fx, fy, frame_id,
            )

        # ── Step 8: Merge ────────────────────────────────
        all_primitives = saccade_primitives + peripheral_primitives
        relations = _extract_spatial_relations(all_primitives)

        # ── Step 9: Object-level familiarity tagging ───
        # Check each primitive's embedding against cached grounded clusters.
        # Familiar primitives are tagged — downstream processing can use
        # this to reduce effort (skip full edge analysis on re-watch, etc.)
        cluster_cache = budget.cluster_cache if budget else None
        if cluster_cache and cluster_cache.size > 0:
            for p in all_primitives:
                if p.embedding is not None:
                    cid, clabel, csim = cluster_cache.match(
                        np.array(p.embedding, dtype=np.float32),
                    )
                    if cid is not None:
                        if p.features is None:
                            p.features = {}
                        p.features["familiar"] = True
                        p.features["cluster_match_id"] = cid
                        p.features["cluster_label"] = clabel
                        p.features["cluster_similarity"] = round(csim, 3)

        return FrameAnalysis(
            frame_id=frame_id,
            primitives=all_primitives,
            relations=relations,
            width=w,
            height=h,
            fixations=saccade_fixations or None,
        )

    # ── Motion saliency ──────────────────────────────────

    def _compute_motion_saliency(self, gray: np.ndarray) -> np.ndarray:
        """Frame differencing for motion detection.

        Uses cv2.absdiff instead of optical flow — works at any frame rate
        including low fps (2fps) where DIS optical flow fails due to large
        temporal gaps between frames.
        """
        h, w = gray.shape[:2]

        if self._prev_gray is None:
            return np.zeros((h, w), dtype=np.float32)

        # Frame differencing (works at any fps)
        diff = cv2.absdiff(gray, self._prev_gray).astype(np.float32)

        # Smooth to create a saliency map (not raw pixel noise)
        ksize = max(15, h // 16) | 1
        saliency = cv2.GaussianBlur(diff, (ksize, ksize), 0)

        # Normalize to [0, 1]
        p99 = np.percentile(saliency, 99)
        if p99 > 0:
            saliency = np.clip(saliency / p99, 0, 1)

        return saliency

    # ── Edge saliency ────────────────────────────────────

    def _compute_edge_saliency(self, gray: np.ndarray) -> np.ndarray:
        """Edge density as a smooth saliency map."""
        edges = cv2.Canny(gray, 50, 150)

        # Gaussian blur to create smooth hot spots from edge clusters
        # Large kernel = edges contribute to a wide attention field
        ksize = max(15, gray.shape[0] // 16) | 1  # smaller kernel preserves peak sharpness
        saliency = cv2.GaussianBlur(
            edges.astype(np.float32), (ksize, ksize), 0,
        )

        # Normalize
        max_val = saliency.max()
        if max_val > 0:
            saliency = saliency / max_val

        return saliency

    # ── Face prior ───────────────────────────────────────

    def _compute_face_prior(self, gray: np.ndarray) -> np.ndarray:
        """Detect faces and create Gaussian bumps at their locations."""
        h, w = gray.shape[:2]
        saliency = np.zeros((h, w), dtype=np.float32)

        if self._face_cascade is None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                return saliency

        # Detect at quarter resolution for speed
        scale = 0.25
        small = cv2.resize(gray, (int(w * scale), int(h * scale)))
        faces = self._face_cascade.detectMultiScale(
            small, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20),
        )

        for (fx, fy, fw, fh) in faces:
            # Scale back to full resolution
            cx = int((fx + fw / 2) / scale)
            cy = int((fy + fh / 2) / scale)
            sigma = int(max(fw, fh) / scale * 0.8)

            # Gaussian bump
            y_grid, x_grid = np.ogrid[:h, :w]
            bump = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
            saliency = np.maximum(saliency, bump.astype(np.float32))

        return saliency

    # ── Temporal smoothing ───────────────────────────────

    def _ema_update(self, saliency: np.ndarray) -> np.ndarray:
        """Exponential moving average on saliency map."""
        alpha = self.cfg.ema_alpha

        if self._ema_saliency is None or self._ema_saliency.shape != saliency.shape:
            self._ema_saliency = saliency.copy()
            return saliency

        self._ema_saliency = alpha * saliency + (1 - alpha) * self._ema_saliency
        return self._ema_saliency

    # ── Peak finding ─────────────────────────────────────

    def _find_saliency_peaks(
        self, saliency: np.ndarray, n: int = 4,
    ) -> list[tuple[float, float, float]]:
        """Find top N saliency peaks using non-max suppression.

        Returns list of (x, y, saliency) sorted by saliency descending.
        These feed the CandidateTracker which decides where to focus.
        """
        blurred = cv2.GaussianBlur(saliency, (15, 15), 0)
        h, w = blurred.shape
        suppress_r = max(20, min(h, w) // 10)

        peaks = []
        sal_copy = blurred.copy()

        for _ in range(n):
            peak_val = float(sal_copy.max())
            if peak_val < 0.02:  # absolute floor
                break
            loc = np.unravel_index(sal_copy.argmax(), sal_copy.shape)
            py, px = int(loc[0]), int(loc[1])
            peaks.append((float(px), float(py), peak_val))

            # Suppress around this peak
            y1, y2 = max(0, py - suppress_r), min(h, py + suppress_r)
            x1, x2 = max(0, px - suppress_r), min(w, px + suppress_r)
            sal_copy[y1:y2, x1:x2] = 0.0

        return peaks

    def _find_peak(
        self, saliency: np.ndarray,
    ) -> tuple[float, float, float]:
        """Find the single strongest peak. Used by legacy code paths."""
        peaks = self._find_saliency_peaks(saliency, n=1)
        if peaks:
            return peaks[0]
        h, w = saliency.shape
        return float(w / 2), float(h / 2), 0.0

    # ── Attention stickiness + Kalman ────────────────────

    def _update_focal_point(
        self,
        target_x: float,
        target_y: float,
        peak_saliency: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        """Apply stickiness check, then Kalman filter.

        Only shifts focus if new peak exceeds current by switch_margin
        AND we've dwelled for at least min_dwell_frames.
        """
        # Stickiness check
        should_switch = False
        if self._current_peak <= 0:
            should_switch = True  # first frame
        elif peak_saliency > self._current_peak * (1 + self.cfg.switch_margin):
            if self._dwell_count >= self.cfg.min_dwell_frames:
                should_switch = True

        if should_switch:
            self._dwell_count = 0
            self._current_peak = peak_saliency
            # Kalman update with new target
            fx, fy = self._kalman_predict_update(target_x, target_y)
        else:
            self._dwell_count += 1
            # Kalman predict only (maintain current trajectory)
            fx, fy = self._kalman_predict_only()

        # Clamp to frame bounds
        self._focal_x = max(0, min(fx, frame_w - 1))
        self._focal_y = max(0, min(fy, frame_h - 1))

        return self._focal_x, self._focal_y

    def _kalman_predict_update(
        self, mx: float, my: float,
    ) -> tuple[float, float]:
        """Kalman filter predict + update with measurement."""
        # State: [x, y, vx, vy]
        F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        Q = np.diag([1.0, 1.0, 0.5, 0.5])
        R = np.diag([10.0, 10.0])

        if self._kalman_x is None:
            self._kalman_x = np.array([mx, my, 0, 0], dtype=np.float64)
            self._kalman_P = np.diag([100.0, 100.0, 10.0, 10.0])
            return mx, my

        # Predict
        x_pred = F @ self._kalman_x
        P_pred = F @ self._kalman_P @ F.T + Q

        # Update
        z = np.array([mx, my])
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self._kalman_x = x_pred + K @ y
        self._kalman_P = (np.eye(4) - K @ H) @ P_pred

        # Blend: (1 - blend) * measurement + blend * prediction
        blend = self.cfg.kalman_blend
        fx = (1 - blend) * mx + blend * x_pred[0]
        fy = (1 - blend) * my + blend * x_pred[1]

        return float(fx), float(fy)

    def _kalman_predict_only(self) -> tuple[float, float]:
        """Kalman predict without measurement (maintain trajectory during dwell)."""
        if self._kalman_x is None:
            return self._focal_x, self._focal_y

        F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        Q = np.diag([1.0, 1.0, 0.5, 0.5])

        self._kalman_x = F @ self._kalman_x
        self._kalman_P = F @ self._kalman_P @ F.T + Q

        return float(self._kalman_x[0]), float(self._kalman_x[1])

    # ── Fovea radius ─────────────────────────────────────

    def _compute_fovea_radius(
        self, saliency: np.ndarray, fx: float, fy: float,
    ) -> float:
        """Adaptive fovea radius based on saliency concentration.

        Tight radius when saliency is concentrated (clear target).
        Wide radius when saliency is spread (uncertain scene).
        """
        h, w = saliency.shape
        peak = saliency.max()
        if peak <= 0:
            return float(self.cfg.fovea_radius_max)

        # Measure how concentrated the saliency is around the focal point
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - fx) ** 2 + (y_grid - fy) ** 2)

        # Fraction of total saliency within a moderate radius
        test_radius = (self.cfg.fovea_radius_min + self.cfg.fovea_radius_max) / 2
        inner_mask = dist <= test_radius
        inner_sum = (saliency * inner_mask).sum()
        total_sum = saliency.sum()
        concentration = inner_sum / max(total_sum, 1e-6)

        # High concentration → small radius, low → large
        t = max(0, min(concentration, 1))
        radius = self.cfg.fovea_radius_max - t * (
            self.cfg.fovea_radius_max - self.cfg.fovea_radius_min
        )

        return radius

    # ── Foveal region analysis ───────────────────────────

    def _multi_saccade_analysis(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
        primary_fx: float,
        primary_fy: float,
        frame_id: str,
        max_saccades: int | None = None,
    ) -> tuple[list[VisualPrimitive], list[tuple[float, float, float]]]:
        """Perform multiple saccades across a frame and merge overlapping crops.

        1. Find top N saliency peaks (with non-max suppression)
        2. Crop and analyze each fixation region
        3. Group crops by spatial overlap (IoU)
        4. Merge overlapping groups: average their MoE embeddings into
           a composite vector that represents the combined visual content

        For simple primitives (triangle on white), the 3-4 fixation points
        on vertices/edges overlap heavily → merged into one composite
        embedding that captures the whole shape.

        For complex scenes, distant objects stay as separate primitives.
        """
        h, w = image.shape[:2]
        r = int(self._fovea_radius)

        # ── Find saccade targets (non-max suppression) ──
        fixations = []
        sal_copy = saliency.copy()
        suppress_r = max(16, int(r * self.cfg.saccade_suppression))

        n_saccades = max_saccades if max_saccades is not None else self.cfg.max_saccades
        for _ in range(n_saccades):
            peak_val = sal_copy.max()
            if peak_val < 0.02:
                break
            peak_loc = np.unravel_index(sal_copy.argmax(), sal_copy.shape)
            py, px = int(peak_loc[0]), int(peak_loc[1])
            fixations.append((px, py, peak_val))

            # Suppress around this peak
            y1s = max(0, py - suppress_r)
            y2s = min(h, py + suppress_r)
            x1s = max(0, px - suppress_r)
            x2s = min(w, px + suppress_r)
            sal_copy[y1s:y2s, x1s:x2s] = 0.0

        # Always include the primary focal point if not already covered
        if fixations:
            covered = any(
                abs(fx - primary_fx) < r and abs(fy - primary_fy) < r
                for fx, fy, _ in fixations
            )
            if not covered:
                fixations.insert(0, (int(primary_fx), int(primary_fy), 1.0))
        else:
            fixations = [(int(primary_fx), int(primary_fy), 1.0)]

        # ── Crop and analyze each fixation ──
        crops = []  # (x1, y1, x2, y2, primitives)
        for fx, fy, sal_val in fixations:
            x1 = max(0, fx - r)
            y1 = max(0, fy - r)
            x2 = min(w, fx + r)
            y2 = min(h, fy + r)

            if x2 - x1 < 32 or y2 - y1 < 32:
                continue

            crop = image[y1:y2, x1:x2]
            analysis = analyze_frame_edge(crop, f"{frame_id}_sac{len(crops)}")

            # Remap bbox coords to full-frame
            for p in analysis.primitives:
                if p.bbox:
                    bx, by, bw, bh = p.bbox
                    p.bbox = (bx + x1, by + y1, bw, bh)
                if p.features and isinstance(p.features, dict):
                    p.features["attention_region"] = "foveal"
                    p.features["saccade_saliency"] = round(sal_val, 3)

            crops.append((x1, y1, x2, y2, analysis.primitives))

        if not crops:
            return [], fixations

        # ── Collect primitives from individual crops ──
        individual_prims = []
        for c in crops:
            individual_prims.extend(c[4])

        # ── Group overlapping crops by IoU ──
        n = len(crops)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                iou = self._crop_iou(crops[i][:4], crops[j][:4])
                if iou >= self.cfg.overlap_iou_threshold:
                    union(i, j)

        # ── Merge overlapping crops ──
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result_primitives = []
        for group_indices in groups.values():
            if len(group_indices) == 1:
                prims = crops[group_indices[0]][4]
                if prims:
                    result_primitives.extend(prims)
                    continue
                # Single crop found nothing — try with generous padding
                # (the shape may extend beyond the crop radius)
                idx = group_indices[0]
                pad = r
                ux1 = max(0, crops[idx][0] - pad)
                uy1 = max(0, crops[idx][1] - pad)
                ux2 = min(w, crops[idx][2] + pad)
                uy2 = min(h, crops[idx][3] + pad)
            else:
                # Multiple overlapping crops — compute bounding box union
                # with generous padding so the whole object fits
                pad = r
                ux1 = max(0, min(crops[i][0] for i in group_indices) - pad)
                uy1 = max(0, min(crops[i][1] for i in group_indices) - pad)
                ux2 = min(w, max(crops[i][2] for i in group_indices) + pad)
                uy2 = min(h, max(crops[i][3] for i in group_indices) + pad)

            if ux2 - ux1 < 32 or uy2 - uy1 < 32:
                continue

            union_crop = image[uy1:uy2, ux1:ux2]
            analysis = analyze_frame_edge(
                union_crop, f"{frame_id}_union{len(result_primitives)}"
            )

            for p in analysis.primitives:
                if p.bbox:
                    bx, by, bw, bh = p.bbox
                    p.bbox = (bx + ux1, by + uy1, bw, bh)
                if p.features and isinstance(p.features, dict):
                    p.features["attention_region"] = "composite"
                    p.features["composite_saccades"] = len(group_indices)

            result_primitives.extend(analysis.primitives)

        return result_primitives, fixations

    @staticmethod
    def _crop_iou(a: tuple, b: tuple) -> float:
        """Compute IoU between two (x1, y1, x2, y2) rectangles."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def _analyze_foveal_region(
        self,
        image: np.ndarray,
        fx: float,
        fy: float,
        frame_id: str,
    ) -> list[VisualPrimitive]:
        """Analyze the foveal region using edge detection on a cropped region."""
        h, w = image.shape[:2]
        r = int(self._fovea_radius)
        x1 = max(0, int(fx - r))
        y1 = max(0, int(fy - r))
        x2 = min(w, int(fx + r))
        y2 = min(h, int(fy + r))

        if x2 - x1 < 32 or y2 - y1 < 32:
            analysis = analyze_frame_edge(image, frame_id)
            for p in analysis.primitives:
                if p.features and isinstance(p.features, dict):
                    p.features["attention_region"] = "foveal"
            return analysis.primitives

        crop = image[y1:y2, x1:x2]
        analysis = analyze_frame_edge(crop, f"{frame_id}_foveal")

        for p in analysis.primitives:
            if p.bbox:
                bx, by, bw, bh = p.bbox
                p.bbox = (bx + x1, by + y1, bw, bh)
            if p.features and isinstance(p.features, dict):
                p.features["attention_region"] = "foveal"

        return analysis.primitives

    def _analyze_periphery(
        self,
        image: np.ndarray,
        fx: float,
        fy: float,
        frame_id: str,
    ) -> list[VisualPrimitive]:
        """Downscale periphery and run lightweight analysis."""
        scale = self.cfg.periphery_scale
        if scale <= 0:
            return []

        h, w = image.shape[:2]
        small_w, small_h = max(32, int(w * scale)), max(32, int(h * scale))
        small = cv2.resize(image, (small_w, small_h))

        analysis = analyze_frame_edge(small, f"{frame_id}_periph")

        # Scale bounding boxes back to full-frame coordinates
        inv_scale_x = w / small_w
        inv_scale_y = h / small_h
        for p in analysis.primitives:
            if p.bbox:
                bx, by, bw, bh = p.bbox
                p.bbox = (
                    int(bx * inv_scale_x),
                    int(by * inv_scale_y),
                    int(bw * inv_scale_x),
                    int(bh * inv_scale_y),
                )
            p.area_fraction *= (scale * scale)  # adjust for downscaling
            if p.features and isinstance(p.features, dict):
                p.features["attention_region"] = "peripheral"

        # Filter: only keep peripheral primitives NOT overlapping with fovea
        r = int(self._fovea_radius)
        peripheral = []
        for p in analysis.primitives:
            if p.bbox:
                px = p.bbox[0] + p.bbox[2] / 2
                py = p.bbox[1] + p.bbox[3] / 2
                dist = math.sqrt((px - fx) ** 2 + (py - fy) ** 2)
                if dist > r * 0.8:  # outside foveal region
                    peripheral.append(p)
            else:
                peripheral.append(p)

        return peripheral
