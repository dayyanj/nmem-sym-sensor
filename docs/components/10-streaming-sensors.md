# Streaming & Live Sensors

Real-time perception from camera and microphone. The system's live sensory interface (vs batch video ingestion).

## Modules

| Module | Purpose |
|--------|---------|
| `sensor_bus.py` | Unified sensor abstraction: Sensor base class, Observation dataclass, SensorBus multiplexer for managing multiple concurrent sensors |
| `streaming.py` | Real-time processing loop: sensor bus -> analysis -> buffer -> promote -> bind -> consolidate. Background threads for STT and periodic consolidation |
| `sensors/camera.py` | Live video: webcam or RTSP stream at configurable FPS with scene change detection |
| `sensors/microphone.py` | Live audio: system microphone in rolling windows via sounddevice |
| `sensors/file.py` | File replay: video file -> temporal stream of visual + audio observations |

## Architecture

```
[Camera Sensor]  ──┐
                    ├──> [SensorBus] ──> [streaming.py loop]
[Microphone Sensor]──┘         |              |
                               |         analysis + binding
[File Sensor] ─────────────────┘         consolidation
                                         recognition
                                         (same pipeline as batch)
```

## Links to Other Components

- **api.py** (Public API): SensorGraph used for all graph operations
- **visual.py / audio.py** (Perception): same analysis functions as batch mode
- **graph.py** (Sensory Graph): same buffer/promote/bind path
- **config.py** (Config): streaming-specific config (STREAM_CAMERA_FPS, STREAM_AUDIO_WINDOW_S, etc.)

## Key Config

```
STREAM_CAMERA_FPS = 2.0
STREAM_AUDIO_WINDOW_S = 0.5
STREAM_AUDIO_HOP_S = 0.25
STREAM_CONSOLIDATION_S = 30.0
STREAM_GROUNDING_S = 120.0
STREAM_STT_BUFFER_S = 5.0
```

## Note

The streaming pipeline is functional but less exercised than the batch video pipeline. Most teaching currently happens through batch video ingestion via the learner.
