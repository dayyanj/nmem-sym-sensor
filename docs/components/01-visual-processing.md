# Visual Processing

Extracts shape, color, and texture primitives from image frames. The system's "eyes."

## Modules

| Module | Purpose |
|--------|---------|
| `visual.py` | Image analysis: two backends (edge detection / SAM2 segmentation) produce VisualPrimitive objects with shape, color, texture labels and MoE embeddings (512-dim) |
| `visual_attention.py` | Foveal attention: motion + edge saliency determine where to look. Kalman-filtered focal point, adaptive radius, minimum dwell time. Biologically inspired |
| `adaptive_acuity.py` | Two-pass processing: Pass 1 extracts structural skeleton at full resolution, Pass 2 encodes tiles at resolution matched to complexity (32/64/128px) |

## Data Flow

```
Raw image frame
    |
    v
[visual_attention.py] ── foveal focus point ──> crop region
    |                                               |
    v                                               v
[visual.py] ── edge/SAM2 backend ──> segments ──> VisualPrimitive[]
    |                                               |
    |   shape: "irregular-red-noisy"                |
    |   color: "red"                                |
    |   texture: "noisy"                            |
    |   embedding: [512-dim MoE vector]             |
    v                                               v
[graph.py] ── buffer_observation() ──> iconic buffer
```

## Links to Other Components

- **graph.py** (Sensory Graph): primitives enter the iconic buffer via `buffer_observation()`
- **temporal.py** (Temporal Tracking): visual embeddings are matched across frames for object persistence
- **composition.py** (Composition): spatial relations (part_of, above, contains) between primitives are persisted as edges
- **sensory_prediction.py** (Intelligence Loops): visual observations are compared against structural/temporal predictions
- **binding.py** (Cross-Modal Binding): visual nodes bind to simultaneous audio nodes
- **language.py** (Sound Language): visual nodes from `recent_visual_entries` are passed to sound-visual bindings
- **text_bridge.py** (Text Bridge): visual node IDs are passed alongside STT words for word-visual co-occurrence

## ONNX Models

- `models/visual_moe_v4.onnx` (151KB + 37MB data) — 3-expert MoE: Simple (0.29M), Moderate (1.09M), Complex (7.41M). Router classifies input complexity. 512-dim output embedding.

## Key Config

```
VISUAL_BACKEND = "edge" | "sam2"
VISUAL_FEATURE_DIM = 512
FOVEAL_ATTENTION_ENABLED = 0|1
ADAPTIVE_ACUITY_ENABLED = 0|1
VISUAL_MIN_SEGMENT_AREA = 0.005
VISUAL_MAX_SEGMENTS = 64
```
