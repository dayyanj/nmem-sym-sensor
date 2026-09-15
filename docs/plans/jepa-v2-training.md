# JEPA v2 Training Plan — Improved Visual Encoder

## Architecture: Dual-Stream 512-dim

```
Foveal crop (any size)
  ├── Geometric encoder (256-dim, hand-crafted, no training)
  │     Edge geometry, local shading, surface/material, border context
  │     Per-section normalised. Instant. Interpretable.
  │
  └── JEPA encoder (256-dim, learned, structural)
        Masked patch prediction with local-to-global crops.
        Captures patterns the hand-crafted features miss.

  → Concatenate → 512-dim embedding (system-compatible)
```

## What Failed in JEPA v1

1. **Background dominated predictions** — the model learned "predict background colour"
   as the easiest strategy. 80% of patches are background, so this achieves low loss.

2. **Mean-pooling destroyed discrimination** — 64 patch embeddings pooled to 512-dim
   lost all spatial information. Pentagon and star became identical.

3. **Cosine similarity misleading** — SIGReg normalised embeddings to isotropic Gaussian,
   making cosine similarity uninformative. L2 showed more discrimination but still
   not enough for practical use.

4. **COCO drowned primitives** — 123K complex scenes vs 10K primitives. The model
   optimised for the majority data (predicting scene textures) not the minority
   (predicting shape structure).

5. **Warmup LR too aggressive** — loss went UP during warmup epochs 8-30 before
   recovering. DINOv3 showed fixed LR works better.

## JEPA v2 Changes

### 1. Gram Anchoring (from DINOv3)

**Problem**: Extended training degrades local patch features.
**Fix**: Save checkpoint at epoch 10 as "Gram teacher." During subsequent training,
add a loss term that constrains patch-level features to remain similar to the
Gram teacher's output.

```python
# Gram anchoring loss
gram_loss = F.mse_loss(
    current_patch_features,      # what the model produces now
    gram_teacher_features,       # what it produced at epoch 10
)
total_loss = pred_loss + sigreg_weight * sigreg_loss + gram_weight * gram_loss
```

Gram weight: 0.1 (small — don't prevent learning, just prevent forgetting).
Start gram anchoring at epoch 15 (let the model learn freely first).

### 2. Local-to-Global Crops (from DINO/DINOv3)

**Problem**: Student and teacher both see the same masked full image.
**Fix**: Teacher sees the full image (global context). Student sees a random crop
(local view, simulating foveal crop). Student must predict teacher's patch
embeddings from partial context.

```python
# Training forward pass:
# Teacher: full 128×128 image → all 64 patches → encode
teacher_patches = teacher_encoder(full_image)  # (64, 256), no gradient

# Student: random crop (64×64 to 96×96 of the 128×128) → encode
crop = random_crop(full_image, scale=(0.4, 0.75))
crop_resized = resize(crop, 128)  # resize to encoder input size
student_patches = student_encoder(crop_resized)  # masked prediction as before

# Loss: student predictions should match teacher's FULL image patches
# (student predicts what it can't see from what it CAN see)
pred_loss = F.mse_loss(student_predictions, teacher_target_patches)
```

Teacher is EMA of student (momentum 0.996, updated each step).

### 3. Saliency-Weighted Masking

**Problem**: Random masking means 80% of masked patches are boring background.
**Fix**: Preferentially mask patches that contain edges/structure.

```python
def saliency_weighted_mask(patches, mask_ratio=0.5):
    """Mask more interesting patches more often."""
    # Compute saliency per patch (gradient magnitude)
    saliency = [patch_gradient_energy(p) for p in patches]
    # Higher saliency → higher probability of being masked
    probs = softmax(saliency * temperature)
    # Sample mask indices weighted by saliency
    masked_indices = np.random.choice(n_patches, n_mask, p=probs, replace=False)
    return masked_indices
```

This forces the model to predict structural patches from context,
not just interpolate background colour.

### 4. Fixed Learning Rate (from DINOv3)

**Problem**: Cosine LR with warmup caused loss to increase during warmup.
**Fix**: Fixed LR throughout training. No warmup, no decay.

```python
optimizer = AdamW(params, lr=1e-4, weight_decay=0.05)
# No scheduler at all — fixed LR
```

Simpler, more stable, DINOv3 proved this works at scale.

### 5. Balanced Dataset

**Problem**: 123K COCO images drowned 10K primitives.
**Fix**: Balanced sampling — each batch contains equal proportions from each source.

```python
# Weighted sampler: each source gets equal representation per batch
weights = {
    'primitives': 1.0 / n_primitives,    # 10K → high weight per image
    'coco': 1.0 / n_coco,                # 123K → low weight per image
    'faces': 1.0 / n_faces,              # 16K → medium weight
    ...
}
# Or simpler: cap COCO at 20K random subset, keep all primitives
```

Alternative: train in phases:
- Phase 1 (epochs 1-30): primitives only → learn basic structure
- Phase 2 (epochs 31-100): mix all data → generalise

### 6. Output: 256-dim (not 512)

The JEPA encoder produces 256-dim. Combined with geometric encoder's 256-dim,
total is 512-dim. No mean-pool over patches — use a CLS token or attention-weighted
pool that preserves discrimination.

```python
# CLS token approach (standard ViT)
cls_token = nn.Parameter(torch.randn(1, 1, 256))  # learned
# Prepend to patch sequence, encoder attends over all patches + CLS
# CLS output = 256-dim embedding (attends to all patches weighted by relevance)
```

### 7. RoPE Positional Embeddings (from DINOv3)

**Problem**: Learned position embeddings fix the resolution to 128×128.
**Fix**: Rotary position embeddings work at any resolution.

This is a nice-to-have for variable foveal crop sizes, not critical for v2.

## Training Setup

**Hardware**: RTX 4090 (24GB)
**Dataset**:
  - 10K primitives (singles + composites)
  - 20K COCO subset (random)
  - 16.5K faces
  - 3.4K KITTI
  - 3K Sintel
  - Total: ~53K (manageable, balanced)

**Hyperparameters**:
  - Batch size: 256-512 (fill the GPU)
  - LR: 1e-4 fixed
  - Weight decay: 0.05
  - Epochs: 100
  - Mask ratio: 50% (saliency-weighted)
  - Gram anchor: save at epoch 10, apply from epoch 15 (weight 0.1)
  - Teacher EMA momentum: 0.996
  - Local crop scale: (0.4, 0.75) of full image

**Monitoring**:
  - Sample images every 10 epochs (8 diverse)
  - Validation suite (our 9 tests) every 20 epochs
  - Log patch-level diversity metrics
  - Compare geometric vs JEPA vs combined at each checkpoint

## Validation Criteria (from validate_jepa.py)

Must pass:
1. Shape discrimination (triangle ≠ circle) — cosine < 0.7 OR L2 > 8
2. Background invariance (same shape, different bg) — cosine > 0.5
3. Curve vs line — cosine < 0.7 (was 0.976 in MoE, 0.999 in JEPA v1)
4. Composite discrimination (house ≠ arrow) — cosine < 0.8

Nice to have:
5. Color discrimination (shape > color difference)
6. Size invariance
7. Rotation moderate similarity

## Implementation Order

1. Add CLS token to PatchJEPA (replace mean-pool)
2. Add Gram anchoring (save early checkpoint, constrain later)
3. Add local-to-global crop strategy (student=crop, teacher=full)
4. Add EMA teacher encoder
5. Add saliency-weighted masking
6. Change to fixed LR
7. Balance dataset (cap COCO at 20K)
8. Train 100 epochs
9. Run validation suite at epoch 20, 40, 60, 80, 100
10. If validation passes: export 256-dim encoder
11. Combine with geometric encoder → test full 512-dim pipeline
12. A/B test against MoE encoder on foveal crops
