# Sensory Memory: Visual and Auditory Cognition for AI Agents

A design document for grounded perception — the missing layer between raw sensory input and symbolic reasoning.

## The Problem

nmem-sym gives AI agents associative reasoning over a typed knowledge graph. Concepts like "cup", "music", "red car" exist as symbolic nodes grounded in text — extracted from LLM conversations and memory entries. But these symbols are **ungrounded in the sensory world**. The system knows *about* cups from language; it has never *seen* one.

This creates three limitations:

1. **No perceptual learning.** The system can't discover new categories from visual or audio input. It knows what a cup is because GPT/Qwen told it, not because it observed handles attached to cylinders containing liquid.

2. **No sensory recall.** When the symbol "cup" activates in the graph, there's no sensory context — no visual memory of what cups look like, no auditory memory of what they sound like when set down.

3. **No cross-modal binding.** The system can't learn that the sound of clinking glass co-occurs with transparent cylindrical shapes, because it has no mechanism to bind visual and auditory observations.

## The Approach

Rather than training classification models on labeled datasets, we extend the nmem-sym architecture with **sensory memory graphs** that learn compositionally through experience:

- **Lightweight feature extractors** (not classifiers) decompose images into geometric primitives (shape, color, texture, spatial relationships) and audio into spectral primitives (frequency bands, rhythm patterns, timbre, onset events).

- **The graph's existing mechanisms do the learning.** Plasticity (LTP/LTD) strengthens associations between co-occurring primitives. Clustering merges similar primitives into proto-concepts. Dreamstate discovers structural patterns. No labeled training data required.

- **Grounding connects perception to language.** When a sensory cluster co-occurs with a text label enough times, it becomes grounded to the corresponding nmem-sym symbol node. The system learns "that visual cluster is called a cup" through temporal binding, not supervised training.

## Architecture

```
Layer 0: Sensory Encoders (feature extraction only)
  ├── Visual encoder: image → segments → {shape, color, texture, spatial_rel}
  │   ├── "edge" backend: Canny + contour analysis (CPU-only)
  │   └── "sam2" backend: Segment Anything 2 (GPU, higher quality)
  └── Audio encoder: waveform → {frequency_band, rhythm, timbre, onset_event}
      └── numpy STFT (CPU-only, no torchaudio required)

Layer 0.5: Iconic Buffer (ultra-short sensory memory)
  └── Raw observations expire in ~5 seconds
  └── Only repeatedly-seen primitives survive to short-term

Layer 1: Sensory Symbol Graphs (per-modality)
  ├── Visual graph: shape/color/texture nodes, spatial edges
  └── Auditory graph: frequency/rhythm/timbre nodes, temporal edges

Layer 2: Cross-Modal Binding
  └── Temporal co-occurrence → bound_to edges between visual & audio nodes

Layer 3: Consolidation (concept formation)
  ├── Clustering: similar long-term primitives → clusters
  ├── Proto → stable: clusters reaching sufficient observations
  └── Stable → grounded: clusters linked to nmem-sym symbols

Layer 4: nmem-sym (symbolic reasoning)
  └── grounded_in edges from concepts to sensory clusters

Layer 5: nmem (episodic memory)
Layer 6: LLM (reasoning)
```

## Three-Tier Sensory Memory

Mirrors human sensory memory architecture:

### Iconic Memory (Buffer)
- **Duration:** ~5 seconds (human: ~250ms; ours is slower due to lower input rate)
- **Capacity:** 256 entries max
- **Purpose:** Raw sensory observations before any processing
- **Promotion:** Primitives seen ≥3 times within the buffer window are promoted to short-term
- **Implementation:** `sensory_iconic_buffer` table with TTL-based expiry

### Short-Term Sensory Memory
- **Duration:** ~1 hour
- **Purpose:** Primitives that survived iconic decay — frequent enough to be meaningful
- **Promotion:** Primitives with ≥10 total observations are promoted to long-term
- **Decay:** Nodes not reinforced within the decay window are archived

### Long-Term Sensory Memory
- **Duration:** Indefinite (subject to salience decay)
- **Purpose:** Stable sensory primitives eligible for clustering and concept formation
- **Clustering:** Similar long-term nodes are grouped by embedding similarity
- **Grounding:** Stable clusters can be linked to nmem-sym symbol nodes

## Visual Encoding

The visual encoder is deliberately **not a classifier**. It decomposes images into constituent geometric and appearance properties:

### Primitives Extracted
- **Shape:** circle, oval, triangle, square, rectangle, pentagon, hexagon, irregular
- **Color:** Quantized HSV → red, orange, yellow, green, cyan, blue, purple, pink, black, white, gray
- **Texture:** smooth, matte, textured, noisy (from gradient magnitude variance)
- **Spatial relations:** above, below, left_of, right_of, contains, contained_by, adjacent_to, overlaps

### Feature Vector (256-dim)
Hand-crafted embedding encoding:
- Shape one-hot (dims 0-15)
- HSV color with circular hue encoding (dims 16-31)
- Texture descriptors (dims 32-39)
- Geometry: area fraction, aspect ratio (dims 40-47)
- L2-normalized for cosine similarity

### Why Not CLIP?
CLIP provides a shared vision-language embedding space, but it already knows what cups are from its training data. Using CLIP would defeat the purpose: we want the system to **discover** that certain shape+color+texture combinations form meaningful categories, not inherit that knowledge from a pre-trained model.

The hand-crafted embeddings are intentionally low-level. They encode geometry and appearance without semantic meaning. Meaning emerges from the graph.

## Audio Encoding

### Primitives Extracted
- **Frequency bands:** sub_bass, bass, low_mid, mid, upper_mid, presence, brilliance — with peak frequency and relative energy
- **Timbre:** spectral centroid, spread, rolloff → brightness classification (dark, warm, neutral, bright)
- **Rhythm:** inter-onset intervals → tempo classification (very_slow to very_fast), regularity score
- **Onset events:** discrete sound onsets with peak frequency and energy

### Feature Vector (128-dim)
- Band energies with normalized peak frequencies (dims 0-13)
- Timbre descriptors (dims 16-23)
- Rhythm features (dims 24-31)
- L2-normalized

## Cross-Modal Binding

When visual and audio observations occur within a 2-second window, they become **binding candidates**. The system tracks co-occurrence counts via `sensory_cooccurrences`.

When a pair exceeds the binding threshold (default: 3 co-occurrences), a `bound_to` edge is created between the visual and audio nodes. Confidence scales with continued co-occurrence.

This is how the system learns that:
- Clinking sounds go with glass shapes
- Pouring sounds go with liquid in cylindrical containers
- Crunching sounds go with irregular textured shapes

No labeled data. Just temporal correlation.

## Concept Formation

The consolidation pipeline runs periodically:

1. **Flush** expired iconic buffer entries
2. **Promote** frequently-seen iconic entries to short-term nodes
3. **Promote** reinforced short-term nodes to long-term
4. **Decay** stale short-term nodes (archive)
5. **Cluster** long-term visual nodes by embedding similarity (threshold: 0.85)
6. **Cluster** long-term audio nodes by embedding similarity (threshold: 0.80)
7. **Promote** proto-clusters with sufficient members (≥5) and observations (≥10) to stable

Stable clusters represent **discovered concepts** — recurring patterns of co-occurring primitives. A cluster containing {cylinder, brown, smooth, handle-protrusion} represents something the system has seen repeatedly but hasn't named yet.

## Grounding to nmem-sym

Two mechanisms connect sensory clusters to the symbolic graph:

### Label Matching
For each stable ungrounded cluster, build a text description from member labels ("cylinder, brown colored, smooth surface, oval"), embed it, and search the symbol graph for similar nodes. If "cup" has embedding similarity ≥0.75, create a grounding edge.

### Temporal Co-occurrence
When the application layer detects that a text label ("that's a cup") occurs while a sensory cluster is active, create a grounding edge. This is the runtime path — learning by being told what you're looking at.

Once grounded, the sensory cluster enriches the symbol node with perceptual context. When nmem-sym activates "cup", the bridge provides: "Sensory context: shape=cylinder, color=brown, texture=smooth, 47 observations, coherence=0.83".

## Graph Flooding Protection

A single image can produce dozens of primitives. Video at 1fps would flood the graph. Several mechanisms prevent this:

1. **Iconic buffer:** Only primitives seen ≥3 times survive to short-term
2. **Short-term decay:** Unreinforced nodes archive after 1 hour
3. **Max segments:** Visual encoder caps at 64 segments per frame
4. **Max audio events:** Audio encoder caps at 32 events per window
5. **Buffer size limit:** Iconic buffer caps at 256 entries, oldest flushed first
6. **Salience gating:** Long-term node salience decays without activation

## Open Questions

### Embedding Space Alignment
Currently using separate embeddings per modality (256-dim visual, 128-dim audio, 384-dim text). Cross-modal binding relies on temporal co-occurrence rather than embedding similarity. A future version could explore learned projection into a shared space, but this risks importing pre-trained semantics.

### Streaming Video/Audio
The current API processes discrete frames and audio windows. Continuous streaming requires:
- Frame differencing to avoid re-extracting static scenes
- Audio windowing with overlap for continuous processing
- A scheduler for periodic consolidation during streaming

### Scale
Untested beyond toy inputs. The iconic buffer provides a natural choke point, but long-term node counts could grow unbounded with diverse visual input. May need more aggressive archival policies.

### Texture and Shape Refinement
The current shape classifier is basic (contour-based). Future versions could use:
- Fourier descriptors for shape signatures
- Gabor filters for texture analysis
- Moment invariants for rotation-independent shape matching

These remain deliberately simple for v0.1. The graph's clustering handles the complexity that the encoder misses.
