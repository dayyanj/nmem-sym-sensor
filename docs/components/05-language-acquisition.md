# Language Acquisition

Emergent language learning from sensory observation. No grammar rules, no labelled data. Language structure emerges from statistical patterns in sound and vision.

## Modules

| Module | Purpose |
|--------|---------|
| `language.py` | Sound units (speaker-fuzzy audio clusters), sound sequences (emergent grammar), sound-visual bindings (grounded meaning). STT labels are optional bootstrapping |
| `ingest/text_bridge.py` | STT text -> word-visual co-occurrence recording. Speech accumulation, phrase boundary detection, IDF-weighted statistical grounding |

## Three-Layer Architecture

```
Layer 1: SOUND UNITS
  Audio embeddings (128-dim) -> fuzzy clustering (0.65 threshold)
  Same word from man/woman/child -> same sound unit
  Centroid averaging naturally moves toward speaker-invariant representation

Layer 2: SOUND SEQUENCES
  Track which sound units follow which -> bigram co-occurrence
  "sound_17 -> sound_42" appears 50 times = recurring pattern
  Gap timing distinguishes within-word (<200ms) from between-word (>500ms)

Layer 3: SOUND-VISUAL BINDING
  Sound sequences that co-occur with visual clusters = grounded language
  "sound_17->42" always appears with elephant visual cluster
  = system has learned the sound of "elephant" without STT
```

## Two Parallel Text Systems

```
SOUND-LEVEL (language.py)              TEXT-LEVEL (text_bridge.py)
  Audio embedding -> sound unit          Whisper STT -> text words
  Speaker-fuzzy clustering               Word-visual co-occurrence
  Sound sequences (bigrams)              IDF-weighted grounding
  Sound-visual bindings                  Statistical label assignment
  NO STT REQUIRED                        REQUIRES STT

Both run simultaneously. STT labels bootstrap sound units.
Sound units exist independently of text.
```

## Links to Other Components

- **audio.py** (Audio Processing): audio embeddings feed into sound unit clustering
- **visual.py / graph.py** (Sensory Graph): visual node IDs used for sound-visual bindings and word-visual co-occurrences
- **imagery.py** (Mental Imagery): sound units activate visual representations and vice versa
- **vocal_tract.py** (Speech Production): sound units are the targets for speech production
- **api.py** (Public API): SoundLanguage initialised on SensorGraph
- **ingest/video.py** (Video Pipeline): audio embeddings and STT words fed to language system per frame
- **consolidation.py** (Consolidation): language consolidation (unit merging, phrase discovery) runs as a consolidation step
- **sensory_bridge.py** (Bridge): grounding.new events build communication drive pressure

## Database Tables

| Table | Purpose |
|-------|---------|
| `sound_units` | Discovered sound patterns with centroid, STT label, speaker variance |
| `sound_sequences` | Bigram co-occurrences: which sounds follow which, with timing |
| `sound_visual_cooccurrences` | Sound-visual bindings: which sounds co-occur with which visuals |
| `word_visual_cooccurrences` | Text-level word-node co-occurrence statistics |

## Key Config

```
LANGUAGE_LEARNING_ENABLED = 0|1
SOUND_SIMILARITY_THRESHOLD = 0.65
```
