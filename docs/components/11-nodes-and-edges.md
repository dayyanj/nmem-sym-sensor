# Nodes and Edges

A plain-language guide to what nodes and edges represent in the nmem-sym-sensor sensory graph.

## What is the sensory graph?

The sensory graph is how nmem-sym-sensor stores everything it has perceived. Raw sensory input (pixels, audio samples) is compressed into discrete **nodes** — small, reusable building blocks of perception. Relationships between those building blocks are stored as **edges**. Together, nodes and edges form a graph that the system queries, predicts against, and consolidates into concepts over time.

## Nodes

A node is a single sensory primitive — the smallest unit of perception the system tracks. Every node has three defining properties:

| Property | What it means |
|----------|---------------|
| **modality** | Which sense produced it: `visual` or `audio` |
| **node_type** | What kind of primitive it is (see tables below) |
| **label** | A human-readable descriptor generated during encoding (e.g. `"circle"`, `"red"`, `"rhythmic-120bpm"`) |

Nodes also carry a dense **embedding vector** (512-dim for visual, 128-dim for audio) that captures the full perceptual fingerprint. Two nodes can share the same crude label but be distinguished by their embeddings — a cat and a dog might both be labelled `"irregular-orange-noisy"`, but their embeddings will differ, so they get separate nodes.

### Visual node types

| Type | What it captures | Example labels |
|------|-----------------|----------------|
| `shape` | Geometric contour of a segment | `"circle"`, `"rectangle"`, `"irregular-orange-noisy"` |
| `color` | Quantized color of a region | `"red"`, `"sky-blue"`, `"dark-brown"` |
| `texture` | Surface pattern | `"smooth"`, `"striped"`, `"noisy"` |
| `spatial_rel` | Spatial arrangement between two segments | `"above"`, `"inside"`, `"adjacent"` |
| `visual_cluster` | A consolidated group of co-occurring visual primitives | (system-generated) |
| `visual_object` | A promoted cluster with stable identity | (system-generated) |

### Audio node types

| Type | What it captures | Example labels |
|------|-----------------|----------------|
| `frequency` | Spectral band or pitch cluster | `"440hz-band"`, `"low-rumble"` |
| `rhythm` | Temporal onset pattern | `"rhythmic-120bpm"`, `"irregular-onsets"` |
| `timbre` | Spectral envelope shape | `"bright-harmonic"`, `"dull-noise"` |
| `audio_event` | A discrete sound onset/offset | `"click"`, `"burst"` |
| `audio_cluster` | A consolidated group of co-occurring audio primitives | (system-generated) |
| `audio_object` | A promoted cluster with stable identity | (system-generated) |

### Node lifecycle (memory tiers)

Nodes are not permanent from the moment they appear. They move through tiers that mirror human sensory memory:

```
iconic buffer (5s)  -->  short-term (1h)  -->  long-term (persistent)
```

1. **Iconic buffer** — every raw observation lands here first. Entries expire after 5 seconds. If the same label is seen at least 3 times before expiry, it gets promoted.
2. **Short-term** — promoted primitives that persist across frames. If a short-term node accumulates 10+ observations, it gets promoted again. Otherwise it decays after 1 hour.
3. **Long-term** — stable primitives that persist indefinitely. Eligible for clustering into concepts. Can be archived during dreamstate consolidation but never truly deleted — archived nodes can be reactivated if a similar observation appears later.

## Edges

An edge is a directional relationship between two nodes. Every edge has a **source**, a **target**, an **edge_type**, and numerical **weight**, **confidence**, and **groundedness** scores that strengthen with repeated observation.

### Spatial edges (visual)

These describe how visual segments are arranged relative to each other in a single frame.

| Edge type | Meaning | Example |
|-----------|---------|---------|
| `adjacent_to` | Segments are next to each other | eye `adjacent_to` nose |
| `contains` | A spatially contains B | face `contains` eye |
| `contained_by` | B is inside A (inverse of `contains`) | eye `contained_by` face |
| `above` | A is above B | sky `above` ground |
| `below` | A is below B | ground `below` sky |
| `left_of` | A is to the left of B | left-ear `left_of` nose |
| `right_of` | A is to the right of B | nose `right_of` left-ear |
| `overlaps` | Partial spatial overlap | shadow `overlaps` wall |

### Compositional edges

These capture part-whole structure and co-occurrence patterns.

| Edge type | Meaning | Example |
|-----------|---------|---------|
| `part_of` | A is a structural part of B | wheel `part_of` car |
| `co_occurs_with` | A and B regularly appear together (same modality) | red `co_occurs_with` circle |
| `member_of` | A primitive belongs to a cluster | node-42 `member_of` cluster-7 |

### Temporal edges (audio + video)

These capture the ordering of events in time.

| Edge type | Meaning | Example |
|-----------|---------|---------|
| `simultaneous` | Events occurred at the same time | clap-sound `simultaneous` hand-motion |
| `follows` | A occurs after B | splash `follows` drop |
| `precedes` | A occurs before B | drop `precedes` splash |

### Cross-modal edges

These are the most important edges in the system — they link vision and hearing.

| Edge type | Meaning | Example |
|-----------|---------|---------|
| `bound_to` | Visual and audio primitives are temporally bound | glass-shape `bound_to` clink-sound |

A `bound_to` edge is not created immediately. The system first tracks co-occurrences: every time a visual node and an audio node appear within the same temporal window, a counter increments. Only when that counter reaches the binding threshold does a `bound_to` edge get created. This is how the system learns that the sound of clinking goes with the sight of glasses — through repeated experience, not labels.

### Cluster and concept edges

These connect individual primitives to higher-level groupings.

| Edge type | Meaning | Example |
|-----------|---------|---------|
| `instance_of` | A primitive is an instance of a cluster/concept | this-red `instance_of` red-cluster |
| `prototype_of` | A primitive is the best exemplar of a cluster | node-17 `prototype_of` cat-cluster |
| `grounded_in` | An nmem-sym symbol is grounded in a sensory cluster | sym:"cat" `grounded_in` cat-cluster |

### How edges strengthen

Edges are not binary — they have three numerical properties that evolve:

- **weight** — increases when the relationship is observed again; only ratchets upward
- **confidence** — increases with repeated observation; reflects certainty
- **groundedness** — a simple counter of how many times this edge has been reinforced

When the same relationship is observed again, the existing edge is updated (not duplicated). This means frequently co-occurring patterns develop strong edges, while spurious one-off correlations remain weak and are eventually pruned during consolidation.

## How nodes and edges work together

A single video frame might produce a subgraph like this:

```
[red] --co_occurs_with--> [circle] --part_of--> [face-shape]
                                                      |
                                                  contains
                                                      |
                                                      v
                                                   [eye-shape]
```

Over many frames, if a "meow" audio node consistently appears alongside this visual subgraph, a cross-modal binding forms:

```
[face-shape] --bound_to--> [meow-timbre]
```

Eventually, the visual primitives cluster together, the cluster stabilizes, and nmem-sym grounds it to a symbol:

```
[cat-cluster] <--grounded_in-- sym:"cat"
```

At that point, the system has learned what a cat looks and sounds like — from observation alone.
