# Embedding-Driven Architecture — Remove Labels from Learning Path

## Principle

The system learns what things are through observation and co-occurrence, never
through labels. Labels (edge analyzer, STT) exist only for human debugging.
The learning path is purely embedding-driven:

```
Sensor → Fovea → Crop → Encode → Embedding → Buffer → Node → Co-occurrence → Cluster → Concept
```

No labels at any point in this chain. The concept "triangle" emerges because:
1. Similar embeddings cluster together (all triangle crops produce similar embeddings)
2. A sound pattern co-occurs with that cluster (someone says "triangle" while it's visible)
3. The cluster gets associated with the sound, not labelled by us

## Current Architecture (label-dependent)

These are the places where labels currently influence learning:

### 1. Iconic Buffer Dedup
**Current**: `normalized_label` match — if two buffer entries have the same label,
they're considered the same thing.
**Problem**: Relies on edge analyzer producing correct, consistent labels.
"triangle-blue-noisy" and "triangle-red-smooth" are different labels for the
same shape.

### 2. Node Promotion
**Current**: Promote when 3+ buffer entries share the same `normalized_label`.
**Problem**: If the label is wrong or inconsistent, promotion fails.
A triangle seen 100 times with slightly different labels never promotes.

### 3. Node Dedup / Recognition
**Current**: Two paths:
- Label match: same `normalized_label` → same node
- Embedding match: cosine similarity > threshold → same node
**Problem**: Label match overrides embedding match. Two visually different things
with the same label merge incorrectly.

### 4. Cluster Grounding
**Current**: Word-visual co-occurrences use STT text labels. The "word" in
`word_visual_cooccurrences` comes from Whisper STT.
**Problem**: STT is diagnostic. The real grounding should come from sound unit
co-occurrences (which are embedding-based, not text-based).

### 5. Dashboard / Reporting
**Current**: Labels shown everywhere for human readability.
**This is fine** — labels for display are diagnostic, not learning.

## Target Architecture (embedding-only)

### 1. Iconic Buffer Dedup → Embedding Similarity
```python
# OLD: label match
if new_entry.normalized_label == existing.normalized_label:
    merge(existing, new_entry)

# NEW: embedding similarity
similarity = cosine(new_entry.embedding, existing.embedding)
if similarity > BUFFER_DEDUP_THRESHOLD:  # e.g., 0.85
    merge(existing, new_entry)
```

The label is still stored for debugging but never used for matching.

### 2. Node Promotion → Embedding Clustering in Buffer
```python
# OLD: 3+ entries with same label → promote
if count_by_label[label] >= 3:
    promote(label)

# NEW: 3+ entries with similar embeddings → promote
# Group buffer entries by embedding similarity (simple agglomerative)
clusters = cluster_buffer_entries(threshold=0.80)
for cluster in clusters:
    if len(cluster) >= 3:
        promote(cluster.centroid_embedding)
```

This naturally handles the case where the same shape seen 3 times produces
slightly different embeddings — they cluster together and promote.

### 3. Node Recognition → Embedding Search
```python
# OLD: try label match first, then embedding
node = find_by_label(normalized_label)
if not node:
    node = find_by_embedding(embedding, threshold=0.85)

# NEW: embedding search only
node = find_by_embedding(embedding, threshold=0.85)
# Label stored on node for debugging but not used for matching
```

This uses pgvector HNSW index for efficient nearest-neighbour search.
Already works for shape nodes (embedding-based dedup was added earlier).
Needs to be extended to ALL node types.

### 4. Cluster Grounding → Sound Unit Co-occurrence
```python
# OLD: word_visual_cooccurrences table (STT text → visual node)
# "triangle" (text) co-occurs with node 42

# NEW: unified co-occurrence table (sound unit → visual node)
# Sound unit 78 (embedding of the syllables "tri-an-gle") co-occurs with node 42
# The text label "triangle" is only attached to the sound unit for debugging
```

This is already partially done — the unified co-occurrence table uses
sound unit IDs, not text. But the grounding display still shows STT text.

### 5. Edge Analyzer Role → Debug Overlay Only
```python
# OLD: edge analyzer produces primitives that enter the learning path
primitives = analyze_frame_edge(crop)
for p in primitives:
    buffer_entry = observe(label=p.label, embedding=encode(p.crop))

# NEW: encoder produces embedding directly from the foveal crop
embedding = encode(foveal_crop)  # 512-dim, no label involved
buffer_entry = observe(embedding=embedding)
# Edge analyzer runs SEPARATELY for debug visualization only
if DEBUG:
    debug_primitives = analyze_frame_edge(crop)
    log_debug(debug_primitives)
```

## Foveal Crop Pipeline (revised)

```
Frame arrives
    │
    ├── Scene recognition (coarse embedding, chronoception)
    │
    ├── Foveal attention (saliency → fixations → edge-following)
    │   └── Produces: list of fixation points + convex hull
    │
    ├── Hull crop extraction
    │   └── Bounding box of convex hull → resize to 128×128
    │       This is THE crop that gets encoded
    │
    ├── Encoding (THE learning path)
    │   ├── Geometric encoder (240-dim) — curvature, shading, material, border
    │   └── JEPA encoder (272-dim) — learned structural features
    │       └── Concatenate → 512-dim embedding
    │
    ├── Iconic buffer
    │   └── Store embedding + position metadata
    │   └── Dedup by EMBEDDING similarity (not label)
    │   └── Promote when 3+ similar embeddings accumulate
    │
    ├── Sensory node
    │   └── Recognised by EMBEDDING search (pgvector HNSW)
    │   └── Features: center_x, center_y from hull centroid
    │   └── Label: diagnostic only (from edge analyzer if DEBUG)
    │
    ├── Co-occurrence binding
    │   └── Sound active during this crop → visual↔voice co-occurrence
    │   └── Motor trajectory → visual↔motor co-occurrence
    │   └── Spatial offset from previous crop → allocentric edge
    │   └── Quality: attention phase, surprise, temporal proximity
    │
    ├── [DEBUG ONLY] Edge analyzer
    │   └── Canny edges, contour classification
    │   └── Label generation ("triangle-blue-noisy")
    │   └── Displayed in dashboard, never enters learning path
    │
    └── [DEBUG ONLY] STT
        └── Whisper transcription
        └── Text displayed in dashboard alongside sound units
        └── Never enters learning path
```

## What the Encoder Must Capture

For the co-occurrence graph to build meaningful concepts, the encoder must produce
embeddings where:

### Discrimination (different things → different embeddings)
- Curved edge ≠ straight edge
- Handle curve ≠ rim curve (different curvature)
- Brown surface ≠ white surface
- Textured surface ≠ smooth surface
- Edge region ≠ flat region

### Consistency (same thing → similar embeddings)
- Triangle crop from video A ≈ triangle crop from video B
- Cup handle from cup 1 ≈ cup handle from cup 2
- Red surface in morning light ≈ red surface in evening light

### Invariance (irrelevant variation → ignored)
- Background colour shouldn't dominate
- Small position shifts in the crop shouldn't matter
- Moderate rotation should produce similar (not identical) embeddings

### Sensitivity (relevant variation → captured)
- Size differences should be detectable (small vs large triangle)
- Colour differences should be clear (red vs blue)
- Curvature differences should be captured (gentle vs tight curve)

## Encoder Validation Criteria

Before wiring any encoder into the pipeline, it must pass:

1. **Curve vs straight** < 0.5 similarity — the fundamental structural test
2. **Background invariance** > 0.7 — same shape on different backgrounds
3. **Colour discrimination** — red ≠ blue on same shape
4. **Structure vs surface** < 0.3 — edge crop ≠ flat surface crop
5. **Cross-object consistency** — cup handle ≈ mug handle (same primitive type)
6. **Real image crops** — passes tests on actual photographs, not just synthetics

## Implementation Order

### Phase 1: Decouple labels from learning (code changes)
1. Buffer dedup: switch from label match to embedding similarity
2. Node promotion: switch from label count to embedding cluster count
3. Node recognition: remove label-first path, embedding-only
4. Verify: run learning cycle, confirm nodes form without labels
5. Keep edge analyzer running in parallel for debug display

### Phase 2: Encoder integration
6. Wire geometric encoder into foveal crop path
7. Wire JEPA v2 encoder (when validated) alongside geometric
8. Concatenate → 512-dim (adjust dims as needed)
9. Rebuild pgvector HNSW indexes for new embedding dimensions
10. Run validation suite on real images

### Phase 3: Hull crop extraction
11. Change crop extraction from per-primitive to per-hull
12. Single crop per foveal fixation sequence (not per edge contour)
13. The hull crop IS the input to the encoder
14. Edge analyzer only runs on the hull crop for debug overlay

### Phase 4: Validation
15. Full learning cycle with primitives
16. Recall test — does the system learn shape-word associations?
17. Compare accuracy vs label-dependent pipeline
18. Real image test — cup crops, mixed objects

## Risk: What if the encoder isn't good enough?

If the encoder can't discriminate the primitives the system needs, the
co-occurrence graph will be noisy — wrong things cluster together, correct
associations get diluted.

Mitigation: the dual-stream encoder (geometric + JEPA) provides a fallback.
If JEPA fails on curvature, the geometric encoder catches it. If the geometric
encoder misses a learned pattern, JEPA catches it.

If BOTH fail on a specific discrimination, we add a targeted hand-crafted
feature (like we did with CPDA for curvature). The geometric encoder is
extensible — we can add dimensions for specific problems without retraining.

## Risk: What if embedding-only dedup is too aggressive/too loose?

Threshold too high (0.95): different things merge → noisy concepts
Threshold too low (0.70): same thing creates many nodes → fragmentation

Mitigation: the threshold should be adaptive. Start conservative (0.85),
let the dreamstate consolidation merge similar nodes over time. Better to
have too many nodes (which can be merged) than too few (which can't be split).
