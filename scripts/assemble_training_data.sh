#!/bin/bash
# Assemble diverse training dataset for diffusion decoder.
# Creates symlinks from local datasets + generated primitives into one flat directory.
# Run where your datasets are mounted.

set -e

OUT_DIR="${1:-/home/dayyan/diffusion_training_data}"
NAS="/mnt/nas_datasets"
PRIMS="/mnt/nas_projects/apps/nmem-sym-sensor/training_data/primitives_128"

mkdir -p "$OUT_DIR"

echo "=== Assembling training data ==="

# 1. Generated primitives (5K)
echo "Linking primitives..."
count=0
for f in "$PRIMS"/*.png; do
    ln -sf "$f" "$OUT_DIR/prim_$(printf '%06d' $count).png"
    count=$((count + 1))
done
echo "  Primitives: $count"

# 2. FFHQ faces (subsample 10K from 70K)
echo "Linking FFHQ faces (10K subsample)..."
count=0
for f in $(ls "$NAS/faces/ffhq/"*.png | shuf -n 10000 --random-source=/dev/urandom); do
    ln -sf "$f" "$OUT_DIR/ffhq_$(printf '%06d' $count).png"
    count=$((count + 1))
done
echo "  FFHQ: $count"

# 3. WFLW faces (all ~6.5K)
echo "Linking WFLW faces..."
count=0
for f in $(find "$NAS/faces/wflw" -name '*.jpg' -type f); do
    ln -sf "$f" "$OUT_DIR/wflw_$(printf '%06d' $count).jpg"
    count=$((count + 1))
done
echo "  WFLW: $count"

# 4. Salicon scenes (all 10K)
echo "Linking Salicon scenes..."
count=0
for f in $(find "$NAS/salicon/train" -name '*.jpg' -type f); do
    ln -sf "$f" "$OUT_DIR/salicon_$(printf '%06d' $count).jpg"
    count=$((count + 1))
done
echo "  Salicon: $count"

# 5. KITTI street scenes (all ~7.7K)
echo "Linking KITTI..."
count=0
for f in $(find "$NAS/kitti" -name '*.png' -type f); do
    ln -sf "$f" "$OUT_DIR/kitti_$(printf '%06d' $count).png"
    count=$((count + 1))
done
echo "  KITTI: $count"

# 6. Sintel animated (all ~6.3K)
echo "Linking Sintel..."
count=0
for f in $(find "$NAS/MPI-Sintel-complete/training" -name '*.png' -type f); do
    ln -sf "$f" "$OUT_DIR/sintel_$(printf '%06d' $count).png"
    count=$((count + 1))
done
echo "  Sintel: $count"

total=$(ls "$OUT_DIR" | wc -l)
echo ""
echo "=== Total: $total images in $OUT_DIR ==="
