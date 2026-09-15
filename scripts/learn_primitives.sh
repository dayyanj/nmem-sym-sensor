#!/usr/bin/env bash
# Learn from primitive videos (shapes, colors, outlines, compounds).
# Run this FIRST on a clean DB to establish foundational visual concepts.
#
# FPS is randomised 2.0-5.0 per video by the learner.
# Each video is watched up to 3 times (different FPS each time).
# Grounding derived every 5 videos.
#
# Usage:
#   ./scripts/learn_primitives.sh              # all 210 primitives
#   ./scripts/learn_primitives.sh --max-videos 20  # quick test

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

: "${NMEM_SENSOR_DB_DSN:?set NMEM_SENSOR_DB_DSN to the sensory DB DSN before running (no hardcoded credential)}"
export NMEM_SENSOR_JEPA_ONNX="models/jepa_v2.onnx"
export NMEM_SENSOR_FOVEAL_ENABLED=1
export NMEM_SENSOR_TABULA2_ENABLED=1
export NMEM_SENSOR_OCULOMOTOR=1
export NMEM_SENSOR_ACUITY_ENABLED=1
export NMEM_SENSOR_FOVEAL_FACE_PRIOR=1
export NMEM_SENSOR_CREATE_SYMBOLS=1

python -m nmem_sym_sensor.ingest run videos/primitives/ \
  --stt-model small \
  --max-watches 3 \
  --ground-every 5 \
  "$@"
