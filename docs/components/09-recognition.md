# Recognition & Interpretation

Real-time recognition of known concepts and interpretation of novel observations. The system's "inner voice" that names what it sees.

## Modules

| Module | Purpose |
|--------|---------|
| `recognize.py` | Recognition engine: cache-first for known clusters (instant), LLM for novel clusters. Cognitive load model mirrors human processing |
| `rechallenge.py` | Label hypothesis testing: grounded labels are hypotheses, not facts. Confidence decay, drift detection, periodic LLM re-evaluation |
| `describe.py` | Sensory -> natural language: structural (graph notation for LLM), narrative (flowing text), query (LLM naming prompt) |

## Recognition Flow

```
Active sensory nodes
    |
    v
[recognize.py] ── find_cluster_matches()
    |
    ├── KNOWN (grounded, confident): instant cache hit -> return label
    |     "This is 'elephant' (confidence 0.92)"
    |
    ├── FAMILIAR (grounded, low confidence): return label?, queue re-eval
    |     "This might be 'elephant'?"
    |
    ├── NOVEL (stable cluster, ungrounded): LLM interpretation
    |     describe_cluster() -> LLM("what is this?") -> proposed label
    |
    └── RAW (iconic/short-term): too early, skip
          Not enough observations to interpret
```

## Rechallenge Cycle

```
Grounded label "elephant" (confidence 0.85)
    |
    v
New members added to cluster
    |
    ├── confidence decays by 0.02 per new member
    ├── centroid drift checked against grounding embedding
    └── after 25 new observations -> rechallenge triggered
    |
    v
LLM re-queries: "Is this still an elephant?"
    |
    ├── Same answer -> confidence restored
    ├── Different answer -> old label demoted, new label proposed
    └── "unknown" -> label revoked after 2 failures
```

## Links to Other Components

- **api.py** (Public API): `create_recognition_engine()` factory method
- **graph.py** (Sensory Graph): cluster membership lookup for active nodes
- **describe.py** (Description): generates LLM prompts for unknown clusters
- **ingest/video.py** (Pipeline): recognition runs after promotion in per-frame loop
- **imagery.py** (Mental Imagery): recognition result can trigger cross-modal imagery
- **sensory_bridge.py** (Bridge): recognition.known_object events -> communication drive pressure

## Key Config

```
RECHALLENGE_OBSERVATION_INTERVAL = 25
RECHALLENGE_DECAY_RATE = 0.02
RECHALLENGE_DEMOTION_THRESHOLD = 0.4
RECHALLENGE_DRIFT_THRESHOLD = 0.3
RECHALLENGE_REVOKE_AFTER_FAILURES = 2
```
