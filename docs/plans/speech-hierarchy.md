# Hierarchical Speech Learning & Familiarity-Gated Processing

## The Human Model

A child learning to speak goes through clear stages:

1. **Babbling** (0-6 months) — individual sounds, no meaning. Motor experimentation.
2. **Syllables** (6-12 months) — "ba", "da", "ma" — reliable sound chunks. Repetitive practice.
3. **Words** (12-18 months) — "mama", "ball" — syllable sequences bound to meaning.
4. **Phrases** (18-36 months) — "want ball", "mama go" — word sequences expressing intent.

At each stage, the lower levels become automatic. A 3-year-old doesn't think about
how to form "ba" — it's myelinated. Their processing budget is free to focus on
sentence construction. The familiar runs on autopilot.

This mirrors the visual attention hierarchy we already have: familiar scenes get
minimal processing (MONITOR phase), freeing budget for novel elements. Speech
should work the same way.

## Current State

We have the building blocks:

| Layer | Storage | Status |
|-------|---------|--------|
| Sound units (phonemes) | `sound_units` — 3,634 units, 2,184 with exemplar frames | Learned |
| Phoneme transitions | `sound_sequences` — 12,668 chains with confidence | Learned |
| Word labels | `sound_units.stt_label` — 1,849 labelled | From STT |
| Visual bindings | `sensory_cooccurrences` — 255K+ | Learned |
| Speech production | `vocal_tract.py` — exemplar frames through TABULA2 decoder | Working |

What's missing: chunking sequences into higher-level units, practising those chunks,
and reducing processing cost as chunks myelinate.

## Architecture: Four-Level Speech Hierarchy

```
Level 3: Phrases      "want ball"         [word_id, word_id]
              |
Level 2: Words         "ball"              [syllable_id, syllable_id]
              |
Level 1: Syllables     "ba"                [unit_id, unit_id, unit_id]
              |
Level 0: Phonemes      sound units         exemplar frames (TABULA2 512-dim)
```

Each level has the same structure:
- **Members**: ordered list of child-level IDs
- **Exemplar**: best observed production (frame sequence or concatenated child exemplars)
- **Confidence**: how reliable is this chunk (from practice round-trips)
- **Half-life + myelination**: same Ebbinghaus system as co-occurrences
- **Semantic bindings**: co-occurrences with visual clusters at this level

### Level 0: Phonemes (DONE)

`sound_units` table. Each unit has a centroid embedding and exemplar frames.
Produced by the TABULA2 voice encoder during observation.

### Level 1: Syllables (NEW)

A syllable is a high-confidence subsequence of phoneme transitions.

**Formation criteria:**
- A sequence of 2-4 sound units where each transition has `confidence > 0.5`
- The sequence appears together at least 5 times
- Total duration 50-500ms (typical syllable range)

**Discovery:** Run during consolidation. Scan `sound_sequences` for chains where
consecutive pairs all exceed the confidence threshold. Group into syllable candidates.

**Table: `speech_syllables`**
```sql
CREATE TABLE speech_syllables (
    id SERIAL PRIMARY KEY,
    unit_ids INTEGER[] NOT NULL,        -- ordered phoneme unit IDs
    exemplar_frames BYTEA,              -- concatenated exemplar from constituent units
    exemplar_duration_ms FLOAT,
    confidence FLOAT DEFAULT 0.0,       -- from practice round-trips
    half_life FLOAT DEFAULT 300.0,
    self_play_confirmations INTEGER DEFAULT 0,
    myelinated BOOLEAN DEFAULT FALSE,
    stt_label TEXT,                      -- STT word fragment if matched
    observation_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Level 2: Words (NEW)

A word is a sequence of syllables that co-occurs with a visual concept.

**Formation criteria:**
- 1-4 syllables in consistent order
- Co-occurs with a specific visual cluster or grounded concept
- OR matches an STT word label

The semantic binding is the key differentiator from syllables. A syllable "ba"
is just motor memory. A word "ball" is motor memory + meaning (visual cluster
of round things).

**Table: `speech_words`**
```sql
CREATE TABLE speech_words (
    id SERIAL PRIMARY KEY,
    syllable_ids INTEGER[] NOT NULL,    -- ordered syllable IDs
    visual_cluster_ids INTEGER[],       -- semantic bindings (what this word means)
    exemplar_frames BYTEA,
    exemplar_duration_ms FLOAT,
    confidence FLOAT DEFAULT 0.0,
    half_life FLOAT DEFAULT 300.0,
    self_play_confirmations INTEGER DEFAULT 0,
    myelinated BOOLEAN DEFAULT FALSE,
    stt_label TEXT,                      -- matched STT word
    observation_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Level 3: Phrases (FUTURE)

Word sequences expressing relationships. This is where grammar emerges —
not from rules, but from observed word-order co-occurrence patterns.

"red ball" = [word:red, word:ball] — colour modifier before object.
"want ball" = [word:want, word:ball] — verb before object.

This level connects to nmem-sym for propositional meaning (triples).
Deferred until words are reliable.

## Practice Loop

Each level has its own practice cycle, run during dreamstate:

```
1. Select a chunk to practice (syllable or word)
2. Produce it: concatenate child exemplar frames, decode through TABULA2
3. Re-encode the output: feed waveform back through TABULA2 voice encoder
4. Compare: cosine similarity between target embedding and re-encoded output
5. If similarity > threshold:
   - self_play_confirmations += 1
   - Recompute half_life with self-play multiplier
   - If child units not yet myelinated, reinforce them too (downward pressure)
6. If similarity < threshold:
   - Weaken the chunk
   - Identify which child transition failed (where did the round-trip diverge?)
   - Practice that specific child transition
```

**Downward reinforcement** is critical: practising "banana" reinforces "ba", "na", "na"
underneath. Higher-level practice accelerates lower-level myelination.

**Upward gating**: a syllable can only be considered for word membership once its
confidence exceeds 0.5. A word can only join a phrase once myelinated. You can't
build on shaky foundations.

## Familiarity-Gated Processing

As chunks myelinate, speech processing gets cheaper — exactly like visual scenes.

### Observation Phase (hearing speech)

| What's heard | Processing | Analogy to vision |
|-------------|-----------|-------------------|
| Novel phoneme | Full exemplar encoding, high attention | EXPLORE: trace every contour |
| Familiar syllable (myelinated) | Recognise chunk, skip per-phoneme encoding | INSPECT: recognise familiar object |
| Familiar word (myelinated) | Recognise word, bind to meaning, skip syllable decomposition | MONITOR: scene runs on cached model |
| Novel word (known syllables) | Decompose into syllables, focus on new combination | Familiar scene, one new object |

### Implementation: Speech Attention Budget

```python
def process_speech_segment(segment, speech_cache):
    """Budget-aware speech processing."""

    # Try word-level recognition first (cheapest if myelinated)
    word_match = speech_cache.match_word(segment.embedding)
    if word_match and word_match.myelinated:
        # Familiar word — just record co-occurrence with visual context
        return WordRecognition(word_match, cost=LOW)

    # Try syllable-level recognition
    syllable_matches = speech_cache.match_syllables(segment)
    if all(s.myelinated for s in syllable_matches):
        # Known syllables in new combination — focus on the combination
        return SyllableSequence(syllable_matches, cost=MEDIUM)

    # Fall back to full phoneme-level processing
    return FullPhonemeAnalysis(segment, cost=HIGH)
```

### Speech Cache (analogous to ClusterCache for vision)

An in-memory cache of myelinated speech units at each level:
- Myelinated syllables: embedding centroids for fast matching
- Myelinated words: embedding centroids + semantic bindings
- Refreshed from DB during consolidation (like `ClusterCache.refresh()`)

Recognition is embedding similarity, not STT. The system recognises "ball"
because the sound embedding matches a myelinated word chunk, not because
Whisper transcribed it. STT labels are diagnostic annotations, not the
recognition mechanism.

## Predictive Processing

The visual system already predicts the next frame's embeddings and learns from
prediction errors (`sensory_prediction.py`). Speech should do the same at every
level of the hierarchy.

### The Human Model

When someone says "I kicked the...", you've already predicted "ball" before they
say it. This prediction is:
- **Context-driven**: the verb "kicked" + visual context (a ball is present) prime the prediction
- **Multi-level**: you predict the word AND its phoneme sequence AND approximate timing
- **Error-driven learning**: unexpected completions ("I kicked the habit") generate surprise,
  which strengthens the novel association and weakens the failed prediction

Children do this constantly. They hear "ready, set..." and predict "go!" —
getting it right reinforces the sequence, getting it wrong teaches a new pattern.

### Prediction at Each Level

```
Level 0: Phoneme prediction
  Given phoneme A, predict phoneme B
  Source: sound_sequences (already exists, confidence scores)
  Error signal: expected "ba" after "ma", heard "da" instead

Level 1: Syllable prediction
  Given syllable "ba", predict next syllable "na" (within word "banana")
  Source: syllable transition table (new)
  Error signal: expected "na" after "ba", heard "by" instead → different word

Level 2: Word prediction
  Given word "red", predict next word "ball" (from co-occurrence context)
  Source: word sequences + visual context
  Error signal: expected "ball" (ball is visible), heard "car" → update model

Level 3: Semantic prediction (future)
  Given visual scene + first words, predict the proposition
  Source: nmem-sym symbolic reasoning
  Error signal: predicted "the cat is on the mat", heard "the cat is under the mat"
```

### Implementation: Speech Prediction Engine

Extends the existing `sensory_prediction.py` framework:

```python
class SpeechPredictor:
    """Predict upcoming speech at multiple hierarchy levels."""

    def predict_next(self, current_unit, level, visual_context=None):
        """Generate predictions for what comes next.

        At phoneme level: use sound_sequences transition probabilities.
        At syllable level: use syllable transition table.
        At word level: use word sequences + visual context priming.

        Visual context priming: if a ball is visible and we just heard
        "red", predict "ball" with high confidence. This is cross-modal
        prediction — vision primes auditory expectations.
        """

    def evaluate(self, predicted, observed):
        """Compare prediction to what was actually heard.

        Returns:
          - confirmed: prediction matched → LTP on the sequence
          - refuted: prediction wrong → LTD on predicted, LTP on observed
          - partial: right level but wrong specific unit
          - surprising: nothing predicted, novel input → high encoding weight
        """
```

### Visual-Auditory Priming

The most powerful predictions are cross-modal. When the system sees a ball
and hears "the red...", it should predict "ball" with high confidence because:

1. Visual context: ball cluster is active in the scene
2. Word co-occurrence: "red" frequently precedes object names
3. Semantic binding: word "ball" is bound to the active visual cluster

This is the same mechanism as visual prediction ("I see a cup handle, predict
the cup body is nearby") but across modalities. The `sensory_cooccurrences`
table already stores cross-modal associations — prediction just reads them
in the forward direction.

### Surprise and Attention

Speech prediction errors drive the same surprise signal as visual errors:

- **Low surprise** (correct prediction): minimal processing, confirm and move on.
  Like hearing the expected word in a familiar phrase — "once upon a..." → "time".
- **High surprise** (wrong prediction): full processing, strong encoding.
  Like hearing an unexpected word — "once upon a... cheese" → surprise spike,
  strong memory formation.
- **Habituation**: highly predictable speech (counting "1, 2, 3, 4...") generates
  decreasing surprise. The system habituates, just like visual habituation on
  static scenes. Processing budget drops.

The surprise signal feeds back into the attention controller. Novel speech
in a familiar visual scene could trigger a phase shift from MONITOR back to
INSPECT — the system heard something unexpected and needs to pay attention.

### Prediction Confidence and Myelination

Predictions that are consistently confirmed accumulate self-play-like evidence:

```
"ba" → "na" predicted 500 times, confirmed 480 times (96%)
  → high prediction confidence
  → contributes to sequence half-life growth
  → accelerates myelination of the "ba-na" transition
```

Predictions that are frequently wrong get weakened:

```
"ba" → "ll" predicted 50 times, confirmed 5 times (10%)
  → low prediction confidence
  → sequence strength decays faster
  → eventually pruned
```

This is natural selection on speech patterns — only reliable predictions survive.

## Consolidation Cycle Integration

Speech consolidation runs as part of the existing `language.py` consolidation:

```
Existing:
  1. Sound unit merging (similar centroids)
  2. Sequence discovery (phoneme transitions)
  3. Word-visual binding counts

New additions:
  4. Syllable chunking (high-confidence phoneme chains → syllable candidates)
  5. Word formation (syllable chains + semantic binding → word candidates)
  6. Speech prediction evaluation (confirm/refute predictions, update confidence)
  7. Speech practice (dreamstate: produce → re-encode → compare → LTP/LTD)
  8. Myelination check (same threshold as co-occurrences)
  9. Speech cache refresh (push myelinated items to in-memory cache)
```

## Myelination Cascade

The self-play multiplier on half-life applies at every level:

```
Phoneme:  tau = tau_0 * (1 + ln(1+n)) * (0.5 + q) * (1 + sp/10)
Syllable: tau = tau_0 * (1 + ln(1+n)) * (0.5 + q) * (1 + sp/10)
Word:     tau = tau_0 * (1 + ln(1+n)) * (0.5 + q) * (1 + sp/10)
```

But the `n` (observation count) at higher levels includes both:
- Direct observations (hearing the word spoken)
- Indirect reinforcement (parent-level practice reinforcing children)

This creates a **myelination cascade**: practising words reinforces syllables,
which reinforces phonemes. The most-used phonemes myelinate first, then syllables
containing them, then words. Natural frequency drives consolidation order.

## Connection to nmem-sym

Once words myelinate, they become symbols that can be grounded in nmem-sym:

```
sensory_memory                          nmem-sym
-------------                          --------
speech_word "ball"                      Symbol: "ball"
  visual_cluster_ids: [42, 87, 103]     Grounded to: concept_node "ball"
  confidence: 0.95                       Relations: ball → is_a → toy
  myelinated: true                                  ball → has_property → round
```

The grounding bridge already exists (`sensory_bridge.py`). Words are just a
richer grounding signal than individual sound units — instead of grounding
phoneme "ba" to visual cluster 42, we ground word "ball" to the concept.

## Implementation Order

1. **Syllable chunking** in `language.py` consolidation — discover high-confidence chains
2. **`speech_syllables` table** — store chunks with exemplar frames
3. **Syllable prediction** — predict next phoneme/syllable during observation, evaluate on arrival
4. **Syllable practice** in dreamstate — produce → re-encode → compare
5. **Word formation** — syllable sequences with semantic bindings
6. **`speech_words` table** — store with visual cluster references
7. **Word prediction** — visual context priming, cross-modal prediction
8. **Speech attention budget** — familiarity-gated processing during observation
9. **Speech cache** — in-memory myelinated items for fast recognition
10. **Phrase sequences** — deferred until words are reliable

Steps 1-4 are the immediate next work. Prediction (step 3) slots in naturally
because `sound_sequences` already has the transition probabilities — we just
need to read them forward as predictions. Steps 5-7 follow once syllables
are myelinating. Steps 8-9 are optimisation. Step 10 is a future milestone.
