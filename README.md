# nmem-sym-sensor

**Sensory Memory — Visual and Auditory Cognition for AI Agents**

Layer 0 companion to [nmem-sym](https://github.com/dayyanj/nmem-sym). Decomposes visual and audio inputs into unlabeled primitives, clusters them through repeated observation, and grounds the resulting concepts to the symbolic graph.

No classification — the system learns what things are through experience.

## How It Works

```
Image/Audio → Feature Extraction → Iconic Buffer → Short-Term → Long-Term → Clusters → Concepts → Grounding
                  (no labels)         (5s decay)     (1h decay)   (stable)    (grouped)  (formed)   (→ nmem-sym)
```

1. **Visual encoder** segments images into shapes, colors, textures, and spatial relationships
2. **Audio encoder** decomposes waveforms into frequency bands, rhythm, timbre, and onset events
3. **Iconic buffer** holds raw observations for ~5 seconds — only repeated patterns survive
4. **Consolidation** promotes, clusters, and forms concepts from stable primitives
5. **Cross-modal binding** links visual and audio primitives that co-occur temporally
6. **Grounding** connects sensory clusters to nmem-sym symbol nodes via text similarity or temporal co-occurrence

## Installation

```bash
# Core (CPU-only, numpy + sentence-transformers)
pip install -e .

# With visual backend (OpenCV for edge detection)
pip install -e . opencv-python-headless

# With GPU segmentation (SAM2)
pip install -e ".[visual]"

# With audio processing
pip install -e ".[audio]"

# Everything
pip install -e ".[all]"
```

## Quick Start

```python
import asyncio
import numpy as np
from nmem_sym_sensor import SensorGraph

async def main():
    async with SensorGraph(db_dsn="postgresql://user:pass@localhost/sensory") as sg:
        # Ingest an image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        visual_ids = await sg.ingest_frame(image)

        # Ingest audio
        waveform = np.random.randn(16000)  # 1 second at 16kHz
        audio_ids = await sg.ingest_audio(waveform)

        # Bind cross-modal (they happened at the same time)
        await sg.bind(visual_ids, audio_ids)

        # Run consolidation cycle
        stats = await sg.consolidate()
        print(stats)

        # Ground to nmem-sym (requires separate DB)
        # results = await sg.ground(sym_dsn="postgresql://user:pass@localhost/nmem_sym")

asyncio.run(main())
```

## Database Setup

Requires PostgreSQL with pgvector extension:

```bash
createdb sensory_memory
psql sensory_memory -c "CREATE EXTENSION vector;"
psql sensory_memory -f src/nmem_sym_sensor/schema.sql
```

## Configuration

All settings via environment variables (see `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NMEM_SENSOR_DB_DSN` | — | PostgreSQL connection string |
| `NMEM_SENSOR_VISUAL_BACKEND` | `edge` | `edge` (CPU) or `sam2` (GPU) |
| `NMEM_SENSOR_VISUAL_DIM` | `256` | Visual embedding dimensions |
| `NMEM_SENSOR_AUDIO_DIM` | `128` | Audio embedding dimensions |

## Architecture

See [docs/sensory-memory.md](docs/sensory-memory.md) for the full design document.

## The nmem suite

nmem-sym-sensor is one library in a family of composable, framework-agnostic cognitive layers for AI agents. Each is standalone — mix in only the ones you need.

| Repo | Layer | What it does |
|------|-------|--------------|
| [nmem](https://github.com/dayyanj/nmem) | Memory | Hierarchical, self-refining cognitive memory — 6 tiers + a consolidation engine |
| [nmem-sym](https://github.com/dayyanj/nmem-sym) | Reasoning | Symbolic cognition — typed knowledge graph, spreading activation, drives, prediction |
| [nmem-sym-sensor](https://github.com/dayyanj/nmem-sym-sensor) | Perception | Sensory memory — unlabeled visual/audio primitives clustered into grounded concepts |
| [nmem-identity](https://github.com/dayyanj/nmem-identity) | Perception | Self-supervised person identity (voice + face) — learns who it's talking to, no manual tagging |
| [nmem-act](https://github.com/dayyanj/nmem-act) | Action | Typed action/outcome contract + tiered autonomy gate — act to learn, safely |
| [nmem-sandbox](https://github.com/dayyanj/nmem-sandbox) | Action | LLM-agnostic computer-use sandbox — a headless desktop a vision model drives |
| [nmem-exchange](https://github.com/dayyanj/nmem-exchange) | Comms | End-to-end-secured message bus — agents talk without sharing memory |
| [nmem-immune](https://github.com/dayyanj/nmem-immune) | Integrity | Memory immune system — poisoning detection, drift, quarantine |
| [nmem-viz](https://github.com/dayyanj/nmem-viz) | Tooling | Real-time 3D "brain" visualization of any nmem agent |

**Just want to run one?** [**nmem-studio**](https://hub.docker.com/r/dayyanj/nmem-studio) is the pull-and-run appliance — a single Docker image that stands up one fully-configured agent from a web wizard (memory + reasoning, plus optional identity and embodied perception), no config files by hand.

## License

BSL 1.1 — see [LICENSE](LICENSE).
