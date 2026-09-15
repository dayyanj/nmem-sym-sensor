#!/usr/bin/env bash
# Learn from Phase 1 videos (color flashcards, basic educational content).
# Run AFTER learn_primitives.sh — builds on foundational concepts.
#
# Usage:
#   ./scripts/learn_phase1.sh
#   ./scripts/learn_phase1.sh --max-videos 5

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

python -m nmem_sym_sensor.ingest run videos/phase_1/ \
  --stt-model small \
  --max-watches 3 \
  --ground-every 5 \
  "$@"
