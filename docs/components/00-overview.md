# nmem-sym-sensor: System Overview

Sensory cognition layer for the nmem cognitive stack. Learns object concepts, language, and speech through observation — not classification, not labelled datasets, not pre-trained models.

## System Diagram

```
                         ┌──────────────────────────────────────────┐
                         │            nmem-sym                      │
                         │       (symbolic cognition)               │
                         │                                          │
                         │  Drives: coherence, novelty, uncertainty,│
                         │          competence, integration,        │
                         │          communication (optional)        │
                         │                                          │
                         │  Procedures: compiled motor programs     │
                         │  Schemas: recurring edge patterns        │
                         │  Prediction: causal forward activation   │
                         └────────────┬──────────┬─────────────────┘
                                      │          │
                         symbol→sensory    sensory→symbol
                         expectations      events + LTP/LTD
                                      │          │
┌─────────────────────────────────────┴──────────┴──────────────────────────┐
│                        nmem-sym-sensor                                    │
│                                                                           │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────────┐   │
│  │   VISION    │   │    AUDIO     │   │     LANGUAGE ACQUISITION     │   │
│  │             │   │              │   │                              │   │
│  │ visual.py   │   │ audio.py     │   │ language.py (sound units)    │   │
│  │ attention   │   │ MoE encoder  │   │ text_bridge.py (word-visual) │   │
│  │ acuity      │   │ 4 expert     │   │                              │   │
│  │             │   │ paths        │   │ Sound units → sequences →    │   │
│  │ 512-dim     │   │ 128-dim      │   │ visual bindings = meaning    │   │
│  └──────┬──────┘   └──────┬───────┘   └──────────────┬───────────────┘   │
│         │                 │                           │                    │
│         ▼                 ▼                           │                    │
│  ┌─────────────────────────────────┐                  │                    │
│  │       SENSORY GRAPH             │                  │                    │
│  │                                 │                  │                    │
│  │  Iconic buffer (5s)             │                  │                    │
│  │    → Short-term (1h)            │                  │                    │
│  │      → Long-term (persistent)   │◄─────────────────┘                    │
│  │        → Clusters → Concepts    │                                       │
│  │                                 │                                       │
│  │  graph.py, consolidation.py,    │                                       │
│  │  binding.py, composition.py,    │                                       │
│  │  selection.py                   │                                       │
│  └──────────────┬──────────────────┘                                       │
│                 │                                                          │
│                 ▼                                                          │
│  ┌─────────────────────────────────┐   ┌──────────────────────────────┐   │
│  │    INTELLIGENCE LOOPS           │   │    SPEECH PRODUCTION         │   │
│  │                                 │   │                              │   │
│  │  temporal.py (object tracking)  │   │  vocal_tract.py (decoder +   │   │
│  │  sensory_prediction.py          │   │    vocoder + self-monitor)   │   │
│  │  surprise.py                    │   │  vocoder.py (Griffin-Lim)    │   │
│  │  sensory_dreamstate.py          │   │  imagery.py (mind's eye)    │   │
│  │                                 │   │                              │   │
│  │  predict → observe → compare    │   │  sound unit → decoder →     │   │
│  │  → update (LTP/LTD)            │   │  mel → waveform → re-encode │   │
│  └─────────────────────────────────┘   │  → compare → refine         │   │
│                                        └──────────────────────────────┘   │
│                                                                           │
│  ┌─────────────────────────────────┐   ┌──────────────────────────────┐   │
│  │    RECOGNITION                  │   │    INGESTION PIPELINE        │   │
│  │                                 │   │                              │   │
│  │  recognize.py (cache + LLM)     │   │  video.py (frame loop)      │   │
│  │  rechallenge.py (hypothesis)    │   │  learner.py (queue)          │   │
│  │  describe.py (natural language) │   │  text_bridge.py (STT)        │   │
│  └─────────────────────────────────┘   │  curriculum.py (phases)      │   │
│                                        │  fetch.py (YouTube)          │   │
│                                        └──────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Documents

1. [Visual Processing](01-visual-processing.md) — shape, color, texture extraction; foveal attention; adaptive acuity
2. [Audio Processing](02-audio-processing.md) — frequency, timbre, rhythm; MoE encoder with speech expert
3. [Sensory Graph & Memory](03-sensory-graph.md) — three-tier memory, clustering, composition, Darwinian selection
4. [Intelligence Loops](04-intelligence-loops.md) — predict-observe-compare-update cycle; temporal tracking; surprise
5. [Language Acquisition](05-language-acquisition.md) — sound units, sequences, sound-visual bindings; STT text bridge
6. [Speech Production](06-speech-production.md) — vocal tract, vocoder, motor programs, babbling practice
7. [nmem-sym Integration](07-nmem-sym-integration.md) — bidirectional bridge, drive system, grounding pipeline
8. [Ingestion Pipeline](08-ingestion-pipeline.md) — video processing, learning queue, curriculum, temporal binding
9. [Recognition](09-recognition.md) — real-time recognition, rechallenge, description
10. [Streaming & Sensors](10-streaming-sensors.md) — live camera/microphone, sensor bus

## Key Principles

- **No classification** — the system never labels images during training. Labels emerge from co-occurrence with spoken words
- **Three-tier memory** — iconic (5s) → short-term (1h) → long-term, mirroring human sensory memory
- **Darwinian selection** — strong associations displace weak ones; overcrowded concepts are pruned
- **Bidirectional grounding** — perception enriches symbols, symbols guide perception
- **Language from sound, not text** — STT is a crutch; the system can learn language purely from audio-visual co-occurrence
- **Learning through observation** — no curriculum beyond progressive video complexity; structure emerges from statistics
