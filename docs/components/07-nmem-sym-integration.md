# nmem-sym Integration

Bidirectional connection between sensory perception and symbolic cognition. Sensory observations ground abstract symbols; symbolic knowledge guides perception.

## Modules

| Module | Purpose |
|--------|---------|
| `bridge.py` | One-way grounding: sensory clusters -> nmem-sym symbol nodes via label matching and temporal co-occurrence |
| `sensory_bridge.py` | Bidirectional bridge: symbol -> sensory expectations (top-down) + sensory -> symbol events with drive pressure (bottom-up) |
| `nmem_adapter.py` | Duck-typed adapter for writing STT text to nmem LTM. Zero-import coupling |

## Bidirectional Flow

```
                nmem-sym (symbolic cognition)
                    |              ^
                    |              |
        symbol_to_sensory    create_sensory_events
        _expectations()      feed_events_to_drives()
                    |              |
                    v              |
                nmem-sym-sensor (perception)


TOP-DOWN (symbol -> sensory):
  nmem-sym activates "cup" concept
    -> generate sensory expectations: cylinder + handle clusters
    -> sensory system looks for matching visual patterns
    -> if found: prediction confirmed -> LTP
    -> if not found: prediction refuted -> LTD

BOTTOM-UP (sensory -> symbol):
  Sensory system detects surprise / novel objects / prediction outcomes
    -> creates SymbolEvent objects
    -> feeds drive pressure deltas to nmem-sym
    -> drives respond: uncertainty, coherence, novelty, integration, communication
```

## Drive Pressure Events

| Sensory Event | Drive Effects |
|---------------|--------------|
| prediction.confirmed | uncertainty -0.03, coherence -0.02 |
| prediction.refuted | uncertainty +0.05, coherence +0.03 |
| anomaly.detected (surprise > 0.7) | uncertainty +0.15, novelty -0.05 |
| sensory.novel_objects | novelty -0.02, integration +0.03 |
| grounding.new | integration -0.05, communication +0.05 |
| recognition.known_object | communication +0.1 |
| communication.success | communication -0.3 |

## Communication Drive (modular)

Lives in nmem-sym but only activates with `NMEM_SYM_COMMUNICATION_DRIVE=1`. The drive:
- Builds pressure when the system sees something it knows the name of
- Fires intent to speak when pressure exceeds threshold (0.8)
- Satisfied by successful speech production
- Does not affect nmem-sym when sensor is not present

## Links to Other Components

- **api.py** (Public API): `_sym_pool` connects to nmem-sym database
- **sensory_prediction.py** (Intelligence Loops): symbolic predictions come through the bridge
- **surprise.py** (Surprise): high surprise -> anomaly events -> drive pressure
- **language.py** (Language): new groundings -> communication drive pressure
- **vocal_tract.py** (Speech): communication drive fires -> speech production
- **sensory_dreamstate.py** (Dreamstate): registered as post-dreamstate hook in nmem-sym

## nmem-sym Drive System

```
nmem-sym drives:
  coherence    -> dreamstate (resolve contradictions)
  novelty      -> explore (seek new information)
  uncertainty  -> verify (ground speculative hypotheses)
  competence   -> extract (more triples from LTM)
  integration  -> ground_sensory (link clusters to symbols)
  communication -> communicate (produce speech) [OPTIONAL, sensor-only]
```

## Grounding Pipeline

```
word_visual_cooccurrences (accumulated over many videos)
    |
    v
ground_from_statistics() -- IDF-weighted scoring
    |
    ├── dominance = count / total_count_for_node
    ├── IDF = log(total_nodes / word_spread)
    └── score = dominance * IDF
    |
    v
Cluster grounding:
    score > threshold -> grounded_label assigned
    confidence = min(0.85, score)
    status = 'confident' | 'speculative'
    |
    v
Rechallenge cycle:
    confidence decays over time
    drift detection (centroid moved?)
    LLM re-query if label may no longer fit
```

## Key Config

```
SYM_DB_DSN = (nmem-sym database connection)
GROUNDING_MIN_CONFIDENCE = 0.6
GROUNDING_LABEL_SIMILARITY = 0.75
GROUNDING_CREATE_SYMBOLS = 0|1
NMEM_SYM_COMMUNICATION_DRIVE = 0|1
```
