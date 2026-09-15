# Familiarity-Gated Processing — Implementation & Testing Plan

## Overview

Build incrementally: each phase is independently testable and delivers value.
Test after each phase before moving to the next.

---

## Phase 1: Frame-Level Gate

### What it does
Before deep analysis, compute a cheap familiarity score for the incoming frame.
High familiarity → reduce processing. Novel frame → full processing.

### New file: `familiarity.py`

```python
"""Familiarity-gated processing — spend less compute on familiar content."""

import logging
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

COARSE_SIZE = 32  # 32x32 grayscale → 1024-dim vector


@dataclass
class AttentionBudget:
    """Compute allocation for this frame based on familiarity."""
    max_saccades: int = 8
    edge_follow_steps: int = 6
    run_intelligence: bool = True
    frame_familiarity: float = 0.0   # 0=novel, 1=identical


class SceneMemory:
    """Rolling buffer of coarse frame signatures for familiarity detection.

    Maintains a short history of downscaled frame embeddings. Comparing
    a new frame against this buffer tells us if we've seen this scene
    before — much cheaper than full analysis.
    """

    def __init__(self, buffer_size: int = 30):
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._stability_window: deque[float] = deque(maxlen=10)

    def _coarse_embedding(self, gray: np.ndarray) -> np.ndarray:
        """Downscale to 32x32 and flatten. Trivially cheap."""
        small = cv2.resize(gray, (COARSE_SIZE, COARSE_SIZE)).astype(np.float32)
        flat = small.flatten()
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat /= norm
        return flat

    def compute_familiarity(
        self,
        gray: np.ndarray,
        emb_running_error: float = 0.5,
    ) -> AttentionBudget:
        """Compute familiarity score and derive attention budget.

        Combines two signals:
        1. Coarse frame similarity to recent buffer (cheap visual match)
        2. EmbeddingPredictor.running_error (how well the system
           predicted this frame — already computed for free)

        Returns AttentionBudget with scaled processing parameters.
        """
        coarse = self._coarse_embedding(gray)

        # Compare against buffer
        if self._buffer:
            sims = [float(np.dot(coarse, prev)) for prev in self._buffer]
            max_sim = max(sims)
            mean_sim = sum(sims) / len(sims)
        else:
            max_sim = 0.0
            mean_sim = 0.0

        # Track frame-to-frame stability
        if self._buffer:
            frame_sim = float(np.dot(coarse, self._buffer[-1]))
            self._stability_window.append(frame_sim)

        scene_stability = (
            sum(self._stability_window) / len(self._stability_window)
            if self._stability_window else 0.0
        )

        # Add to buffer
        self._buffer.append(coarse)

        # Combine signals:
        # - coarse visual similarity (have I seen this frame?)
        # - prediction accuracy (did I expect this frame?)
        # Weight prediction accuracy higher — it's a richer signal
        prediction_familiarity = max(0.0, 1.0 - emb_running_error)
        frame_familiarity = 0.4 * max_sim + 0.6 * prediction_familiarity

        # Derive budget
        budget = AttentionBudget(frame_familiarity=frame_familiarity)

        if frame_familiarity > 0.95:
            # Highly familiar — minimal processing
            budget.max_saccades = 2
            budget.edge_follow_steps = 0
            budget.run_intelligence = False
        elif frame_familiarity > 0.80:
            # Moderately familiar — reduced processing
            budget.max_saccades = 4
            budget.edge_follow_steps = 2
            budget.run_intelligence = True  # periodic only
        else:
            # Novel — full processing
            budget.max_saccades = 8
            budget.edge_follow_steps = 6
            budget.run_intelligence = True

        return budget

    @property
    def scene_stability(self) -> float:
        if not self._stability_window:
            return 0.0
        return sum(self._stability_window) / len(self._stability_window)

    def reset(self):
        """Reset on scene change."""
        self._buffer.clear()
        self._stability_window.clear()
```

### Changes to `api.py`

```python
# In SensorGraph.__init__:
self._scene_memory = None

# In SensorGraph.connect():
from nmem_sym_sensor.familiarity import SceneMemory
self._scene_memory = SceneMemory(buffer_size=30)

# In SensorGraph.ingest_frame() — before foveal processing:
budget = None
if self._scene_memory and self._emb_predictor:
    budget = self._scene_memory.compute_familiarity(
        gray,  # already computed for fovea
        emb_running_error=self._emb_predictor.running_error,
    )

# Pass budget to foveal attention:
analysis = self._foveal.process_frame(image, frame_id, scene_change, budget=budget)
```

### Changes to `visual_attention.py`

```python
# FovealAttention.process_frame() signature:
def process_frame(self, image, frame_id, scene_change=False, budget=None):
    # If budget provided, override config values for this frame
    max_saccades = budget.max_saccades if budget else self.cfg.max_saccades
    edge_steps = budget.edge_follow_steps if budget else self._max_edge_follow

    # Use max_saccades in _find_saliency_peaks and _multi_saccade_analysis
    # Use edge_steps in edge-following logic
```

### Changes to `ingest/video.py`

```python
# In the frame loop, gate intelligence cycle:
if budget and not budget.run_intelligence:
    # Skip intelligence on highly familiar frames
    # But run every 5th frame to catch drift
    if i % 5 != 0:
        pass  # skip intelligence
    else:
        # periodic check
        intel_stats = await sensor_graph.run_intelligence_cycle(...)

# Dynamic frame subsampling:
if scene_memory.scene_stability > 0.9 and consecutive_familiar > 5:
    if i % 2 != 0:
        continue  # skip visual, still process audio
```

### Testing Phase 1

**Test 1 — Familiarity detection accuracy**:
```python
# Process red_shade1_us.mp4 frame by frame
# Frame 1: familiarity should be ~0 (novel)
# Frame 10: familiarity should be >0.8 (same solid red scene)
# First frame of NEXT video: familiarity drops (new scene)
```

**Test 2 — Processing reduction measurement**:
```bash
# Run learner on 10 color videos, 3x watches
# Measure: frames processed per video on watch 1 vs watch 3
# Expected: watch 3 processes 30-50% fewer frames
# Verify: grounded concepts unchanged (no quality loss)
```

**Test 3 — Audio continuity**:
```python
# Verify that sound-visual binding still works when frames are skipped
# Audio processes every frame; visual only on non-skipped frames
# Binding should use most recent visual entries regardless of skips
```

---

## Phase 2: Object-Level Gate

### What it does
Per-candidate familiarity check against known cluster centroids.
Known objects → fast recognition, reduced foveal effort, shift to novelty inspection.

### Changes to `familiarity.py` — add ClusterCache

```python
class ClusterCache:
    """In-memory cache of grounded cluster centroids for fast matching."""

    def __init__(self):
        self._centroids: dict[int, tuple[np.ndarray, str, float]] = {}
        # {cluster_id: (centroid_embedding, label, confidence)}

    async def refresh(self, pool):
        """Reload from DB. Call at connect() and after each consolidation."""
        rows = await pool.fetch("""
            SELECT id, visual_centroid::text as vc, grounded_label, grounding_confidence
            FROM sensory_clusters
            WHERE grounded_label IS NOT NULL
              AND grounding_confidence > 0.5
              AND visual_centroid IS NOT NULL
        """)
        self._centroids.clear()
        for r in rows:
            vec = np.array([float(x) for x in r["vc"].strip("[]").split(",")])
            self._centroids[r["id"]] = (vec, r["grounded_label"], r["grounding_confidence"])
        log.info("ClusterCache refreshed: %d grounded clusters", len(self._centroids))

    def match(self, embedding: np.ndarray, threshold: float = 0.80) -> tuple[int | None, str | None, float]:
        """Find best matching cluster. Returns (cluster_id, label, similarity) or (None, None, 0)."""
        best_id, best_label, best_sim = None, None, 0.0
        for cid, (centroid, label, conf) in self._centroids.items():
            sim = float(np.dot(embedding, centroid))
            if sim > best_sim and sim >= threshold:
                best_id, best_label, best_sim = cid, label, sim
        return best_id, best_label, best_sim
```

### Changes to `visual_attention.py` — per-candidate familiarity

```python
# AttentionCandidate gets new fields:
@dataclass
class AttentionCandidate:
    # ... existing fields ...
    familiar: bool = False
    cluster_match_id: int | None = None
    cluster_label: str | None = None

# In CandidateTracker, after matching/creating candidates:
# If cluster_cache available, check each candidate's crop embedding
# against cached centroids. If match:
#   candidate.familiar = True
#   candidate.habituation_frames reduced (habituate 3x faster)
#   candidate uses fewer edge_follow steps
```

### Testing Phase 2

**Test 1 — Cluster cache accuracy**:
```python
# Show the system a red circle (known from training)
# ClusterCache.match() should return the correct cluster
# Show an unknown shape → match returns None
```

**Test 2 — Processing reduction for known objects**:
```bash
# Process a video with a known shape (triangle, already grounded)
# Count saccade points used: should be 2 (familiar) vs 8 (novel)
# Count edge-follow steps: should be 2 vs 6
```

**Test 3 — Novelty within familiar objects**:
```python
# Show a known shape in a new color
# Cluster match → familiar shape, but color embedding differs
# System should spend less time on shape, more on color detail
# This is the Phase 2 fovea: "I know this shape, what's different?"
```

---

## Phase 3: Intelligence Loop Gating

### What it does
Skip the predict→observe→compare cycle when the scene is stable and familiar.
Still run periodically to catch drift.

### Changes to `api.py`

```python
# In run_intelligence_cycle():
# Early exit for familiar frames
if (budget and budget.frame_familiarity > 0.85
        and frame_index % 5 != 0):
    # Still observe embedding to keep EMA current
    if self._emb_predictor:
        self._emb_predictor.observe(frame_emb)
    return {"skipped": True, "reason": "familiar"}
```

### Testing Phase 3

**Test 1 — Prediction skip count**:
```bash
# Run 10 videos 3x, count intelligence cycles per video
# Watch 1: ~all frames get intelligence
# Watch 3: ~80% of familiar frames skip intelligence
# Total processing time should drop 20-30%
```

---

## Integration Test: Full Pipeline

After all phases implemented, run the comprehensive test:

```bash
# 1. Fresh DB, run primitives 6x with familiarity gating
# 2. Measure per-watch stats:
#    - Frames processed per video (should decrease per watch)
#    - Saccade points per frame (should decrease for familiar objects)
#    - Intelligence cycles per video (should decrease per watch)
#    - Total wall time per watch (should decrease)
# 3. Measure quality:
#    - Grounded concept count (should NOT decrease)
#    - Sound-visual binding accuracy (should NOT decrease)
#    - Concept links formed (should still work)
# 4. The key metric: same recognition quality at lower compute cost
```

## Implementation Order

1. **`familiarity.py`** — SceneMemory + AttentionBudget (standalone, no deps)
2. **Wire into `api.py`** — init SceneMemory, compute budget before fovea
3. **Wire into `visual_attention.py`** — process_frame accepts budget, gates saccades
4. **Wire into `video.py`** — intelligence gating + frame subsampling
5. **Test Phase 1** — verify familiarity detection + processing reduction
6. **Add ClusterCache to `familiarity.py`** — grounded cluster matching
7. **Wire into `visual_attention.py`** — per-candidate familiarity
8. **Test Phase 2** — verify known object fast-path
9. **Intelligence gating in `api.py`** — skip predict cycle on familiar frames
10. **Test Phase 3** — verify intelligence skip + quality maintained
11. **Full integration test** — 6x primitives with all gating active
