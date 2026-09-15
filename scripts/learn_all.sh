#!/usr/bin/env bash
# Learn from ALL videos in a pedagogically logical order:
#   1. Primitives (foundation: shapes, colours, simple named objects)
#   2. Simple flashcards (single objects with names)
#   3. Vocabulary building (themed word groups)
#   4. Complex YouTube content (narration, multiple objects, real scenes)
#
# FPS is randomised 2.0-5.0 per video by the learner.
# Each video is watched up to 3 times (different FPS each time).
# Grounding derived every 5 videos.
# Watch counts are respected — already-watched videos are skipped.
#
# Usage:
#   ./scripts/learn_all.sh              # all 619 videos
#   ./scripts/learn_all.sh --max-videos 50  # quick test

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

# Curriculum order: simple → complex
DIRS=(
  videos/primitives         # 1. Foundation: shapes, colours, simple named objects
  videos/phase_1            # 2. Early test videos
  videos/totcards           # 3. Simple flashcards
  videos/colors_extra       # 4. Colour-focused YouTube
  videos/flashcards_extra   # 5. Flashcard compilations
  videos/elf_vocab          # 6. Vocabulary building
  videos/elf_animals        # 7. Animal names
  videos/elf_clothing       # 8. Clothing vocabulary
  videos/elf_verbs          # 9. Action words
  videos/playlist           # 10. General YouTube educational
  videos/other_playlist     # 11. Miscellaneous
  videos/flyingthings3d     # 12. Motion / 3D scenes
)

COMMON_ARGS="--stt-model small --max-watches 3 --ground-every 5"

for dir in "${DIRS[@]}"; do
  [ -d "$dir" ] || continue
  count=$(find "$dir" -name "*.mp4" 2>/dev/null | wc -l)
  [ "$count" -gt 0 ] || continue
  echo "=== Learning: $dir ($count videos) ==="
  python -m nmem_sym_sensor.ingest run "$dir" \
    $COMMON_ARGS \
    "$@"
done
