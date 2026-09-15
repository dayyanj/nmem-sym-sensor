# Sensory Graph & Memory

Three-tier sensory memory with graph-based knowledge storage. The system's "brain" at the sensory level.

## Modules

| Module | Purpose |
|--------|---------|
| `graph.py` | Core graph operations: iconic buffer insert, promotion, node/edge upsert with embedding-based dedup, threshold updates |
| `consolidation.py` | Clustering and concept formation: tier promotion, per-type visual clustering, audio clustering, cluster splitting (cognitive maturation), Darwinian selection, compositional analysis, dreamstate |
| `binding.py` | Cross-modal temporal binding: visual-audio co-occurrence tracking, edge creation when threshold met |
| `composition.py` | Part-whole relationships: record part_of edges, infer wholes from parts, structural expectations ("eye is usually part_of face") |
| `selection.py` | Darwinian competitive displacement: word co-occurrence competition, cluster membership pruning, label competition |

## Three-Tier Memory

```
ICONIC BUFFER (5 second decay)
  Raw observations. Everything enters here first.
  Promotion threshold: 3 co-occurrences of same label
      |
      | promote_from_iconic()
      v
SHORT-TERM NODES (1 hour decay)
  Promoted primitives that persist across frames.
  Promotion threshold: 10 observations
      |
      | promote_short_to_long()
      v
LONG-TERM NODES (persistent)
  Stable primitives eligible for clustering.
  Archived only by explicit decay or dreamstate.
      |
      | run_visual/audio_clustering()
      v
CLUSTERS (proto -> stable -> grounded)
  Groups of similar long-term nodes.
  Become "concepts" when grounded to words/symbols.
```

## Embedding-Based Shape Dedup

Shape nodes use MoE embedding similarity (0.85 threshold) not just label matching. A cat and dog both labelled "irregular-orange-noisy" get SEPARATE nodes because their embeddings differ.

## Cluster Maturation & Splitting

Clusters that grow too diverse split into subclusters — cognitive development. "Blue" cluster splits into "sky blue" + "navy" when coherence drops below threshold. Max 3 generations deep.

## Links to Other Components

- **visual.py / audio.py** (Perception): primitives flow in via `buffer_observation()`
- **api.py** (Public API): SensorGraph wraps all graph operations
- **sensory_prediction.py** (Intelligence Loops): active nodes generate structural predictions
- **sensory_dreamstate.py** (Dreamstate): offline consolidation prunes, merges, strengthens
- **bridge.py** (Grounding): stable clusters are grounded to nmem-sym symbol nodes
- **text_bridge.py** (Text Bridge): word-visual co-occurrences recorded against nodes
- **language.py** (Sound Language): sound-visual bindings reference sensory node IDs
- **recognize.py** (Recognition): matches active nodes against known clusters

## Database Tables

| Table | Purpose |
|-------|---------|
| `sensory_iconic_buffer` | Raw observations with 5s TTL |
| `sensory_nodes` | Promoted primitives (short-term + long-term) |
| `sensory_edges` | Relationships: part_of, co_occurs_with, bound_to, etc. |
| `sensory_clusters` | Grouped nodes forming concepts |
| `sensory_cluster_members` | Node-to-cluster membership |
| `sensory_cooccurrences` | Temporal co-occurrence counts |
| `structural_expectations` | Learned part-whole patterns |
| `word_visual_cooccurrences` | Word-node co-occurrence statistics |

## Key Config

```
ICONIC_DECAY_SECONDS = 5.0
ICONIC_PROMOTION_THRESHOLD = 3
SHORT_TERM_DECAY_HOURS = 1.0
SHORT_TERM_PROMOTION_THRESHOLD = 10
VISUAL_MERGE_THRESHOLD = 0.85
AUDIO_MERGE_THRESHOLD = 0.80
SPLIT_COHERENCE_THRESHOLD = 0.5
SPLIT_MAX_GENERATION = 3
```
