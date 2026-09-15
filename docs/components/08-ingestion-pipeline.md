# Ingestion Pipeline

Video ingestion, continuous learning queue, curriculum management, and YouTube fetching. The system's "classroom."

## Modules

| Module | Purpose |
|--------|---------|
| `ingest/video.py` | Video -> temporally-aligned frames + audio windows + STT transcripts. The main per-frame processing loop |
| `ingest/learner.py` | Continuous learning: queue management, priority ordering (unwatched first), statistical grounding every N videos, dreamstate after grounding |
| `ingest/text_bridge.py` | STT text -> sensory grounding bridge. SpeechAccumulator buffers words into phrases, extract_labels produces concept events, word-visual co-occurrence recording |
| `ingest/curriculum.py` | Progressive learning phases: primitives -> objects -> composition -> scenes -> actions -> open |
| `ingest/fetch.py` | YouTube downloader via yt-dlp with duration/resolution/audio filters |
| `ingest/cli.py` | CLI: learn, learn-dir, fetch-and-learn, run (continuous), queue, live |

## Per-Frame Processing Loop (video.py)

```
For each frame at timestamp t:
  4a. Visual analysis -> iconic buffer entries
  4b. Audio analysis -> iconic buffer entries
  4b'. Sound language -> observe sound, record sequence
  4c. Promote iconic -> sensory nodes
      Separate visual / audio
      Cross-modal binding (visual + audio)
      Intra-modal binding (visual + visual)
      Update recent_visual_entries (5s window)
  4d. Intelligence loop -> predict/observe/compare/update
  4e. Recognition -> known clusters / novel proposals
  4f. STT alignment (bidirectional):
      Backward: words at time t bind to visuals from t-5s..t
      Forward: pending words from last 3s bind to NEW visuals
  4g. Periodic consolidation (every 100 frames)
```

## Temporal Binding Windows

```
Timeline: ──────────────────────────────────────────►

BACKWARD (5s): image appears, then word spoken
  visual at 10s ←───── word "elephant" at 15s
  recent_visual_entries keeps nodes for 5 seconds

FORWARD (3s): word spoken, then image appears
  word "tiger" at 20s ─────► visual at 22s
  pending_words buffer keeps words for 3 seconds
```

## Links to Other Components

- **api.py** (Public API): SensorGraph.ingest_frame/ingest_audio called from the loop
- **visual.py / audio.py** (Perception): produce primitives for each frame/window
- **graph.py** (Sensory Graph): buffer_observation, promote_from_iconic
- **language.py** (Sound Language): audio embeddings fed to SoundLanguage.observe_sound()
- **sensory_prediction.py** (Intelligence): run_intelligence_cycle() called per frame
- **consolidation.py** (Consolidation): periodic consolidation + dreamstate after grounding
- **text_bridge.py** (Text Bridge): STT words -> co-occurrence recording

## Learning Queue (learner.py)

```
Priority order:
  1. Unwatched videos (watch_count = 0), oldest first
  2. Under-watched (watch_count < max_watches), least-watched first
  3. Nothing left -> wait for new content

After every ground_every videos:
  -> ground_from_statistics() derives new groundings
  -> run_sensory_dreamstate() consolidates offline
```

## Database Tables

| Table | Purpose |
|-------|---------|
| `learning_queue` | Video paths, watch counts, status, frame/speech stats |

## Key Config

```
fps = 1-5 (frame extraction rate)
stt_model = tiny | base | small | small.en | medium
consolidate_every = 100 (frames between consolidation)
ground_every = 5 (videos between grounding derivation)
max_watches = 3 (re-watch limit per video)
NMEM_SENSOR_STT_MIN_CONFIDENCE = 0.5
```
