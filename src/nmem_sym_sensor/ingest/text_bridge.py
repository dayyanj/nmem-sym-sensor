"""
Text bridge: STT speech → nmem LTM → sensory grounding.

Connects the speech-to-text output to the nmem memory system so that
spoken words create LTM entries that can later ground sensory clusters.

The bridge works in two directions:
  1. Forward: STT word → nmem LTM write (with timestamp metadata)
  2. Reverse: nmem concept activation → sensory cluster lookup

The forward path is used during video ingestion. The reverse path is
used during runtime when the LLM encounters a concept and wants to
know what it looks/sounds like.

nmem integration is via duck-typing (no import dependency). The caller
provides callback functions that implement the nmem interface.
"""
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# Words to skip — function words that don't ground to sensory concepts
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "or", "but", "nor", "not", "no", "if", "then", "than",
    "so", "as", "at", "by", "for", "from", "in", "of", "on", "to",
    "with", "up", "down", "out", "off", "over", "under", "about",
    "very", "just", "also", "now", "here", "there", "some", "any",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "such", "only", "own", "same", "too", "how", "when", "where",
    "why", "again", "once", "well", "really", "like", "okay", "oh",
    "um", "uh", "hmm", "yeah", "yes", "no", "right", "let", "lets",
    "look", "see", "say", "go", "get", "got", "put", "take", "make",
    "come", "give", "tell", "know", "think", "want", "good", "great",
})

# Patterns that indicate a naming/labeling moment in children's videos
_NAMING_PATTERNS = [
    re.compile(r"this is (?:a |an )?(.+)", re.IGNORECASE),
    re.compile(r"that is (?:a |an )?(.+)", re.IGNORECASE),
    re.compile(r"it'?s (?:a |an )?(.+)", re.IGNORECASE),
    re.compile(r"here'?s (?:a |an )?(.+)", re.IGNORECASE),
    re.compile(r"look,? (?:a |an )?(.+)", re.IGNORECASE),
    re.compile(r"see the (.+)", re.IGNORECASE),
    re.compile(r"called (?:a |an )?(.+)", re.IGNORECASE),
    re.compile(r"what (?:color|shape|sound) is (?:this|that|it)\??\s*(.+)", re.IGNORECASE),
    re.compile(r"the (.+?)(?:\s+is|\s+goes|\s+says|\!|\.)", re.IGNORECASE),
]


@dataclass
class TextEvent:
    """A processed text event ready for nmem and sensory grounding."""
    text: str                          # original spoken text
    label: str                         # extracted concept label (may differ from text)
    timestamp: float                   # seconds from video start
    event_type: str                    # naming | mention | description
    confidence: float = 1.0
    active_visual_ids: list[int] = field(default_factory=list)  # what was being seen


def extract_labels(
    text: str,
    timestamp: float,
    stt_confidence: float = 1.0,
    active_visual_ids: list[int] | None = None,
) -> list[TextEvent]:
    """Extract groundable concept labels from spoken text.

    Distinguishes between:
      - Naming events: "this is a cup" → high-confidence label "cup"
      - Mentions: "the red ball" → label "red ball"
      - Single nouns: "cup" → label "cup"

    Children's learning videos are heavy on naming patterns, which
    is exactly what we want for grounding.

    STT confidence is factored into the event confidence. Clean speech
    (confidence ~0.95) produces strong binding candidates. Speech over
    music (confidence ~0.4) produces weak candidates that need more
    repetitions to promote — this naturally handles noisy children's
    song videos without filtering them out entirely.

    Args:
        text: Spoken text (word or phrase from STT).
        timestamp: When this was spoken.
        stt_confidence: Whisper's confidence for this word/segment (0-1).
            Low confidence = speech over music/noise. High = clean speech.

    Returns:
        List of TextEvents. May be empty (for stop words / noise).
    """
    text = text.strip()
    if not text:
        return []

    # Skip very low confidence segments (pure noise / hallucination)
    if stt_confidence < 0.3:
        return []

    events = []

    # Check for naming patterns first (highest value)
    for pattern in _NAMING_PATTERNS:
        match = pattern.match(text)
        if match:
            label = match.group(1).strip().rstrip("!.,?")
            label = _clean_label(label)
            if label and label.lower() not in _STOP_WORDS:
                # Naming confidence = base 0.9 × STT confidence
                events.append(TextEvent(
                    text=text,
                    label=label,
                    timestamp=timestamp,
                    event_type="naming",
                    confidence=round(0.9 * stt_confidence, 3),
                    active_visual_ids=active_visual_ids or [],
                ))
                return events  # naming pattern is sufficient

    # Split into words and extract content words
    words = text.lower().split()
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 1]

    if not content_words:
        return []

    vids = active_visual_ids or []

    # Multi-word phrases (adjective + noun patterns)
    if len(content_words) >= 2:
        # "red ball", "big cup", "blue circle"
        phrase = " ".join(content_words[:3])  # cap at 3 words
        events.append(TextEvent(
            text=text,
            label=phrase,
            timestamp=timestamp,
            event_type="mention",
            confidence=round(0.7 * stt_confidence, 3),
            active_visual_ids=vids,
        ))

    # Individual content words
    for word in content_words:
        word = _clean_label(word)
        if word and len(word) > 1:
            events.append(TextEvent(
                text=text,
                label=word,
                timestamp=timestamp,
                event_type="mention",
                confidence=round(0.5 * stt_confidence, 3),
                active_visual_ids=vids,
            ))

    return events


def _clean_label(label: str) -> str:
    """Clean a label: lowercase, strip punctuation, normalize whitespace."""
    label = re.sub(r"[^\w\s-]", "", label.lower())
    label = re.sub(r"\s+", " ", label).strip()
    return label


# ── Accumulator for phrase building ──────────────────────

class SpeechAccumulator:
    """Accumulates individual STT words into phrases before processing.

    Whisper gives us word-level timestamps, but individual words like
    "this" "is" "a" "cup" need to be assembled into "this is a cup"
    before we can detect naming patterns.

    Flushes accumulated words when:
      - A sentence-ending word is detected (period, question mark)
      - A pause gap exceeds the threshold
      - The buffer reaches max length
    """

    def __init__(self, pause_threshold_s: float = 1.0, max_words: int = 15):
        self.pause_threshold = pause_threshold_s
        self.max_words = max_words
        # (word, timestamp, stt_confidence, active_visual_ids_at_time_of_speech)
        self._words: list[tuple[str, float, float, list[int]]] = []

    def add_word(
        self,
        word: str,
        timestamp: float,
        stt_confidence: float = 1.0,
        active_visual_ids: list[int] | None = None,
    ) -> list[TextEvent]:
        """Add a word and return any completed phrases as TextEvents.

        Args:
            word: The spoken word.
            timestamp: When it was spoken (seconds from video start).
            stt_confidence: Whisper's confidence for this word (0-1).
                Low = speech over music/noise. High = clean speech.

        Returns events only when a phrase boundary is detected.
        """
        events = []

        # Check for pause gap
        if self._words:
            last_ts = self._words[-1][1]
            if timestamp - last_ts > self.pause_threshold:
                events.extend(self._flush())

        self._words.append((word.strip(), timestamp, stt_confidence, active_visual_ids or []))

        # Check for sentence end
        if word.rstrip().endswith((".", "!", "?")):
            events.extend(self._flush())

        # Check buffer length
        if len(self._words) >= self.max_words:
            events.extend(self._flush())

        return events

    def flush(self) -> list[TextEvent]:
        """Force flush any remaining words."""
        return self._flush()

    def _flush(self) -> list[TextEvent]:
        if not self._words:
            return []

        phrase = " ".join(w for w, _, _, _ in self._words)
        timestamp = self._words[0][1]
        avg_confidence = sum(c for _, _, c, _ in self._words) / len(self._words)

        # Collect all visual IDs that were active across the buffered words
        # This is the visual context at the time of speech
        all_visual_ids: list[int] = []
        for _, _, _, vids in self._words:
            all_visual_ids.extend(vids)
        # Deduplicate preserving order
        seen = set()
        visual_ids = []
        for vid in all_visual_ids:
            if vid not in seen:
                seen.add(vid)
                visual_ids.append(vid)

        # Isolated word detection: a single content word spoken with
        # high confidence after a pause is likely a label (flashcard style).
        is_isolated_label = (
            len(self._words) == 1
            and avg_confidence >= 0.5
            and self._words[0][0].strip().lower() not in _STOP_WORDS
            and len(self._words[0][0].strip()) > 1
        )

        self._words.clear()

        if is_isolated_label:
            label = _clean_label(phrase)
            if label:
                return [TextEvent(
                    text=phrase,
                    label=label,
                    timestamp=timestamp,
                    event_type="naming",
                    confidence=round(0.85 * avg_confidence, 3),
                    active_visual_ids=visual_ids,
                )]

        return extract_labels(phrase, timestamp, stt_confidence=avg_confidence,
                              active_visual_ids=visual_ids)


# ── nmem bridge callbacks ────────────────────────────────

def create_text_callback(
    nmem_write_fn=None,
    sensor_graph=None,
    accumulator: SpeechAccumulator | None = None,
):
    """Create a text_callback function for video ingestion.

    This is the glue between the video pipeline and the nmem system.
    Returns an async callback that:
      1. Accumulates words into phrases
      2. Extracts concept labels
      3. Writes to nmem LTM (via nmem_write_fn)
      4. Optionally triggers sensory grounding attempts

    Args:
        nmem_write_fn: Async function to write to nmem LTM. Signature::

            async def write(content: str, metadata: dict) -> None

            If None, text events are logged but not persisted to nmem.

        sensor_graph: Optional SensorGraph instance. If provided, naming
            events trigger immediate grounding attempts against active
            sensory clusters.

        accumulator: SpeechAccumulator instance. Created if not provided.

    Returns:
        Async callback function(text: str, timestamp: float).
    """
    acc = accumulator or SpeechAccumulator()

    async def callback(
        text: str,
        timestamp: float,
        stt_confidence: float = 1.0,
        active_visual_ids: list[int] | None = None,
    ):
        # Accumulate words into phrases, carrying visual context per word
        events = acc.add_word(text, timestamp, stt_confidence,
                              active_visual_ids=active_visual_ids)

        for event in events:
            log.debug("Text event: [%.1fs] %s '%s' (%.2f)",
                     event.timestamp, event.event_type, event.label, event.confidence)

            # Write to nmem LTM
            if nmem_write_fn:
                try:
                    await nmem_write_fn(
                        content=event.label,
                        metadata={
                            "source": "stt",
                            "original_text": event.text,
                            "timestamp": event.timestamp,
                            "event_type": event.event_type,
                            "confidence": event.confidence,
                        },
                    )
                except Exception as e:
                    log.warning("nmem write failed for '%s': %s", event.label, e)

            # Record co-occurrence: every content word alongside every
            # active visual node. No immediate grounding — just accumulate
            # statistics. The grounding emerges over time from the word-node
            # pairs with the highest co-occurrence counts.
            if sensor_graph and event.active_visual_ids and event.confidence >= 0.3:
                try:
                    await _record_word_visual_cooccurrence(
                        sensor_graph.pool, event.label, event.active_visual_ids,
                    )
                except Exception as e:
                    log.debug("Co-occurrence recording failed for '%s': %s", event.label, e)

    return callback


async def _record_word_visual_cooccurrence(
    pool,
    label: str,
    active_visual_ids: list[int],
):
    """Record that a word was heard while visual nodes were active.

    No grounding happens here. We just count how many times each word
    appeared alongside each visual node. Over hundreds of exposures,
    "red" will have high co-occurrence with the red color node and
    low co-occurrence with blue. The grounding emerges from these
    statistics during periodic consolidation.

    A sentence like "the red ball is on the blue table" records
    co-occurrences between ALL content words and ALL active visual
    nodes. The statistical signal sorts itself out over time.
    """
    words = label.lower().split()
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 1]

    if not content_words:
        return

    for word in content_words:
        for node_id in active_visual_ids:
            await pool.execute(
                """
                INSERT INTO word_visual_cooccurrences (word, node_id, count)
                VALUES ($1, $2, 1)
                ON CONFLICT (word, node_id)
                DO UPDATE SET
                    count = word_visual_cooccurrences.count + 1,
                    last_seen = NOW()
                """,
                word, node_id,
            )

            # Darwinian displacement: this word strengthened, competitors weaken
            from nmem_sym_sensor.selection import displace_word_competitors
            await displace_word_competitors(pool, word, node_id, gain=1)

    if len(content_words) > 0 and len(active_visual_ids) > 0:
        log.debug("Recorded %d word × %d node co-occurrences",
                 len(content_words), len(active_visual_ids))


async def _eureka_reactivate(
    pool,
    min_cooccurrences: int = 15,
) -> int:
    """Reactivate archived nodes that have word co-occurrences with grounded words.

    The eureka moment: a word the system kept hearing alongside certain visuals
    is now confirmed as meaningful (it grounds to active nodes). Archived nodes
    that also had co-occurrences with that word are probably the same concept
    seen earlier but lost to short-term decay. Reactivate them so their
    accumulated evidence isn't wasted.

    Like a child who heard "kirin" many times without understanding, then
    finally sees a giraffe at the zoo and everything clicks — all those
    earlier weak traces suddenly become useful.

    Returns count of reactivated nodes.
    """
    # Find archived nodes that have co-occurrences with words that also
    # strongly co-occur with active nodes. The archived node saw the same
    # word as an active node — they're likely the same concept.
    reactivated = await pool.fetch(
        """
        WITH strong_words AS (
            -- Words that have strong co-occurrences with active nodes
            SELECT DISTINCT wvc.word
            FROM word_visual_cooccurrences wvc
            JOIN sensory_nodes n ON n.id = wvc.node_id
            WHERE wvc.count >= $1
              AND NOT n.archived
        )
        SELECT DISTINCT wvc.node_id, wvc.word, wvc.count, n.label
        FROM word_visual_cooccurrences wvc
        JOIN sensory_nodes n ON n.id = wvc.node_id
        JOIN strong_words sw ON sw.word = wvc.word
        WHERE n.archived = TRUE
          AND wvc.count >= 3
        """,
        min_cooccurrences,
    )

    if not reactivated:
        return 0

    node_ids = list({r["node_id"] for r in reactivated})

    await pool.execute(
        """
        UPDATE sensory_nodes
        SET archived = FALSE,
            memory_tier = 'short_term',
            updated_at = NOW()
        WHERE id = ANY($1)
        """,
        node_ids,
    )

    for r in reactivated:
        log.info("EUREKA: reactivated archived node %d ('%s') — "
                 "word '%s' now grounds strongly (co-occurred %d times)",
                 r["node_id"], r["label"], r["word"], r["count"])

    return len(node_ids)


async def ground_from_statistics(
    pool,
    min_cooccurrences: int = 15,
    min_dominance: float = 0.5,
) -> list[dict]:
    """Derive groundings from accumulated word-visual co-occurrence statistics.

    For each visual node, find the word that co-occurs with it most often.
    If that word dominates (accounts for > min_dominance of all co-occurrences
    for that node), ground the node with that word.

    This is the periodic consolidation step. Call after ingesting videos.
    No single observation creates a grounding — only accumulated evidence.

    Args:
        pool: Database connection pool.
        min_cooccurrences: Minimum times a word-node pair must co-occur.
        min_dominance: Minimum fraction of a node's total co-occurrences
            that the winning word must account for (0.5 = majority).

    Returns:
        List of grounding results.
    """
    # Find nodes where one word dominates the co-occurrence statistics
    candidates = await pool.fetch(
        """
        WITH
        -- Total visual nodes in the system (for IDF denominator)
        total_nodes AS (
            SELECT COUNT(*)::float as n
            FROM sensory_nodes
            WHERE modality = 'visual' AND memory_tier = 'long_term' AND NOT archived
        ),
        -- How many different nodes each word touches (word spread)
        word_spread AS (
            SELECT word, COUNT(DISTINCT node_id) as node_count
            FROM word_visual_cooccurrences
            GROUP BY word
        ),
        -- Per-node totals for dominance calculation
        node_totals AS (
            SELECT node_id, SUM(count) as total_count
            FROM word_visual_cooccurrences
            GROUP BY node_id
        ),
        -- IDF-weighted ranking: words that touch fewer nodes get higher scores
        -- IDF = log(total_nodes / word_spread) — high for specific words, low for ubiquitous ones
        word_ranks AS (
            SELECT wvc.word, wvc.node_id, wvc.count,
                   nt.total_count,
                   wvc.count::float / GREATEST(nt.total_count, 1) as dominance,
                   ws.node_count as word_spread,
                   LN(GREATEST(tn.n, 1) / GREATEST(ws.node_count, 1)) as idf,
                   -- IDF-weighted score: dominance * IDF
                   -- "red" on 3 nodes with 0.6 dominance scores higher than
                   -- "click" on 42 nodes with 0.6 dominance
                   (wvc.count::float / GREATEST(nt.total_count, 1))
                     * LN(GREATEST(tn.n, 1) / GREATEST(ws.node_count, 1))
                     as idf_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY wvc.node_id
                       ORDER BY (wvc.count::float / GREATEST(nt.total_count, 1))
                                * LN(GREATEST(tn.n, 1) / GREATEST(ws.node_count, 1)) DESC
                   ) as rank
            FROM word_visual_cooccurrences wvc
            JOIN node_totals nt ON nt.node_id = wvc.node_id
            JOIN word_spread ws ON ws.word = wvc.word
            CROSS JOIN total_nodes tn
            WHERE wvc.count >= $1
        )
        SELECT wr.word, wr.node_id, wr.count, wr.total_count,
               wr.dominance, wr.idf, wr.idf_score, wr.word_spread,
               n.label as node_label, n.node_type, n.modality
        FROM word_ranks wr
        JOIN sensory_nodes n ON n.id = wr.node_id
        WHERE wr.rank = 1
          AND wr.idf_score >= $2
          AND NOT n.archived
        ORDER BY wr.idf_score DESC
        """,
        min_cooccurrences,
        min_dominance,
    )

    # Eureka trigger: reactivate archived nodes that have co-occurrences
    # with words that are now strongly grounding to active nodes.
    # This is the "ohhh!" moment — a word the system kept hearing finally
    # clicks because the same word is now grounding to similar active nodes.
    await _eureka_reactivate(pool, min_cooccurrences)

    results = []
    for row in candidates:
        word = row["word"]
        node_id = row["node_id"]
        count = row["count"]
        dominance = row["dominance"]
        idf_score = row["idf_score"]
        word_spread = row["word_spread"]
        node_label = row["node_label"]
        node_type = row["node_type"]

        # Find or create a cluster for this node
        existing_cluster = await pool.fetchrow(
            """
            SELECT c.id, c.grounded_label, c.grounding_confidence
            FROM sensory_cluster_members cm
            JOIN sensory_clusters c ON c.id = cm.cluster_id
            WHERE cm.node_id = $1 AND c.modality = 'visual'
            ORDER BY c.grounding_confidence DESC NULLS LAST
            LIMIT 1
            """,
            node_id,
        )

        if existing_cluster:
            cluster_id = existing_cluster["id"]
            existing_label = existing_cluster["grounded_label"]
            existing_conf = existing_cluster["grounding_confidence"] or 0.0

            if existing_label and existing_label == word:
                # Re-confirmation — boost (weighted by IDF so specific words boost more)
                new_conf = min(1.0, existing_conf + idf_score * 0.1)
                await pool.execute(
                    """
                    UPDATE sensory_clusters
                    SET grounding_confidence = $1::float,
                        grounding_status = CASE WHEN $1::float >= 0.7 THEN 'confident' ELSE grounding_status END,
                        updated_at = NOW()
                    WHERE id = $2
                    """,
                    float(new_conf), int(cluster_id),
                )
                results.append({
                    "action": "confirmed",
                    "word": word, "node": node_label,
                    "count": count, "dominance": round(dominance, 3),
                    "idf_score": round(idf_score, 3),
                    "word_spread": word_spread,
                    "confidence": round(new_conf, 4),
                })
                continue

            if existing_label and existing_conf >= idf_score:
                continue  # existing is stronger

        else:
            # Create primitive cluster
            obs = await pool.fetchval(
                "SELECT observation_count FROM sensory_nodes WHERE id = $1", node_id,
            )
            cluster_id = await pool.fetchval(
                """
                INSERT INTO sensory_clusters
                    (modality, cluster_type, member_count, total_observations,
                     coherence, coherence_at_formation)
                VALUES ('visual', 'stable', 1, $1, 1.0, 1.0)
                RETURNING id
                """,
                obs or 0,
            )
            await pool.execute(
                """
                INSERT INTO sensory_cluster_members (cluster_id, node_id, role)
                VALUES ($1, $2, 'prototype')
                """,
                cluster_id, node_id,
            )

        # Ground it — confidence based on IDF-weighted score, not raw dominance
        grounding_conf = round(min(0.85, idf_score), 4)
        await pool.execute(
            """
            UPDATE sensory_clusters
            SET grounded_label = $1,
                grounding_confidence = $2,
                grounding_status = 'speculative',
                cluster_type = 'grounded',
                observations_at_grounding = total_observations,
                updated_at = NOW()
            WHERE id = $3
            """,
            word, grounding_conf, cluster_id,
        )

        await pool.execute(
            """
            INSERT INTO sensory_grounding_log
                (cluster_id, symbol_node_id, symbol_label, confidence,
                 grounding_type, previous_label, rechallenge_reason)
            VALUES ($1, 0, $2, $3, 'statistical', $4, NULL)
            """,
            cluster_id, word, grounding_conf,
            existing_cluster["grounded_label"] if existing_cluster else None,
        )

        results.append({
            "action": "grounded",
            "word": word, "node": node_label, "node_type": node_type,
            "count": count, "dominance": round(dominance, 3),
            "idf_score": round(idf_score, 3),
            "word_spread": word_spread,
            "cluster_id": cluster_id,
        })

        log.info("STATISTICAL GROUNDING: '%s' → %s:%s (count=%d, dominance=%.0f%%, idf=%.2f, spread=%d nodes)",
                word, node_type, node_label, count, dominance * 100, idf_score, word_spread)

    return results


async def ground_from_sound_statistics(
    pool,
    min_cooccurrences: int = 10,
    min_dominance: float = 0.5,
) -> list[dict]:
    """Derive groundings from sound-visual co-occurrence statistics.

    Same IDF-weighted logic as ground_from_statistics() but reads from
    the unified sensory_cooccurrences table (voice/environmental ↔ visual).

    Uses sound_unit.stt_label as the human-readable name when available.
    If no STT label exists, uses "sound_N" as a placeholder — the concept
    is still grounded, just not yet labelled in human terms.

    This is the PRIMARY grounding path in the unified symbol system.
    """
    candidates = await pool.fetch(
        """
        WITH
        -- Flatten unified table to (sound_unit_id, visual_node_id, count)
        sv AS (
            SELECT unit_b_id as sound_unit_id, unit_a_id as visual_node_id, count
            FROM sensory_cooccurrences
            WHERE modality_a = 'visual' AND modality_b IN ('voice', 'environmental') AND strength > 0
            UNION ALL
            SELECT unit_a_id as sound_unit_id, unit_b_id as visual_node_id, count
            FROM sensory_cooccurrences
            WHERE modality_b = 'visual' AND modality_a IN ('voice', 'environmental') AND strength > 0
        ),
        total_nodes AS (
            SELECT COUNT(*)::float as n
            FROM sensory_nodes
            WHERE modality = 'visual' AND memory_tier = 'long_term' AND NOT archived
        ),
        unit_spread AS (
            SELECT sound_unit_id, COUNT(DISTINCT visual_node_id) as node_count
            FROM sv
            GROUP BY sound_unit_id
        ),
        node_totals AS (
            SELECT visual_node_id, SUM(count) as total_count
            FROM sv
            GROUP BY visual_node_id
        ),
        unit_ranks AS (
            SELECT svc.sound_unit_id, svc.visual_node_id, svc.count,
                   nt.total_count,
                   svc.count::float / GREATEST(nt.total_count, 1) as dominance,
                   us.node_count as unit_spread,
                   LN(GREATEST(tn.n, 1) / GREATEST(us.node_count, 1)) as idf,
                   (svc.count::float / GREATEST(nt.total_count, 1))
                     * LN(GREATEST(tn.n, 1) / GREATEST(us.node_count, 1))
                     as idf_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY svc.visual_node_id
                       ORDER BY (svc.count::float / GREATEST(nt.total_count, 1))
                                * LN(GREATEST(tn.n, 1) / GREATEST(us.node_count, 1)) DESC
                   ) as rank
            FROM sv svc
            JOIN node_totals nt ON nt.visual_node_id = svc.visual_node_id
            JOIN unit_spread us ON us.sound_unit_id = svc.sound_unit_id
            CROSS JOIN total_nodes tn
            WHERE svc.count >= $1
        )
        SELECT ur.sound_unit_id, ur.visual_node_id, ur.count, ur.total_count,
               ur.dominance, ur.idf, ur.idf_score, ur.unit_spread,
               su.stt_label,
               n.label as node_label, n.node_type, n.modality
        FROM unit_ranks ur
        JOIN sensory_nodes n ON n.id = ur.visual_node_id
        JOIN sound_units su ON su.id = ur.sound_unit_id
        WHERE ur.rank = 1
          AND ur.idf_score >= $2
          AND NOT n.archived
        ORDER BY ur.idf_score DESC
        """,
        min_cooccurrences,
        min_dominance,
    )

    results = []
    for row in candidates:
        sound_unit_id = row["sound_unit_id"]
        node_id = row["visual_node_id"]
        count = row["count"]
        dominance = row["dominance"]
        idf_score = row["idf_score"]
        node_label = row["node_label"]
        node_type = row["node_type"]

        # Use STT label if available, otherwise placeholder
        label = row["stt_label"] or f"sound_{sound_unit_id}"

        # Find or create cluster
        existing_cluster = await pool.fetchrow(
            """
            SELECT c.id, c.grounded_label, c.grounding_confidence
            FROM sensory_cluster_members cm
            JOIN sensory_clusters c ON c.id = cm.cluster_id
            WHERE cm.node_id = $1 AND c.modality = 'visual'
            ORDER BY c.grounding_confidence DESC NULLS LAST
            LIMIT 1
            """,
            node_id,
        )

        if existing_cluster:
            cluster_id = existing_cluster["id"]
            existing_label = existing_cluster["grounded_label"]
            existing_conf = existing_cluster["grounding_confidence"] or 0.0

            if existing_label and existing_label == label:
                new_conf = min(1.0, existing_conf + idf_score * 0.1)
                await pool.execute(
                    """
                    UPDATE sensory_clusters
                    SET grounding_confidence = $1::float,
                        grounding_status = CASE WHEN $1::float >= 0.7 THEN 'confident' ELSE grounding_status END,
                        updated_at = NOW()
                    WHERE id = $2
                    """,
                    float(new_conf), int(cluster_id),
                )
                results.append({
                    "action": "confirmed", "source": "sound_statistical",
                    "word": label, "sound_unit_id": sound_unit_id,
                    "node": node_label, "count": count,
                    "dominance": round(dominance, 3),
                    "idf_score": round(idf_score, 3),
                    "confidence": round(new_conf, 4),
                })
                continue

            if existing_label and existing_conf >= idf_score:
                continue
        else:
            obs = await pool.fetchval(
                "SELECT observation_count FROM sensory_nodes WHERE id = $1", node_id,
            )
            cluster_id = await pool.fetchval(
                """
                INSERT INTO sensory_clusters
                    (modality, cluster_type, member_count, total_observations,
                     coherence, coherence_at_formation)
                VALUES ('visual', 'stable', 1, $1, 1.0, 1.0)
                RETURNING id
                """,
                obs or 0,
            )
            await pool.execute(
                """
                INSERT INTO sensory_cluster_members (cluster_id, node_id, role)
                VALUES ($1, $2, 'prototype')
                """,
                cluster_id, node_id,
            )

        grounding_conf = round(min(0.85, idf_score), 4)
        await pool.execute(
            """
            UPDATE sensory_clusters
            SET grounded_label = $1,
                grounding_confidence = $2,
                grounding_status = 'speculative',
                cluster_type = 'grounded',
                observations_at_grounding = total_observations,
                updated_at = NOW()
            WHERE id = $3
            """,
            label, grounding_conf, cluster_id,
        )

        await pool.execute(
            """
            INSERT INTO sensory_grounding_log
                (cluster_id, symbol_node_id, symbol_label, confidence,
                 grounding_type, previous_label, rechallenge_reason)
            VALUES ($1, 0, $2, $3, 'sound_statistical', $4, NULL)
            """,
            cluster_id, label, grounding_conf,
            existing_cluster["grounded_label"] if existing_cluster else None,
        )

        results.append({
            "action": "grounded", "source": "sound_statistical",
            "word": label, "sound_unit_id": sound_unit_id,
            "node": node_label, "node_type": node_type,
            "count": count, "dominance": round(dominance, 3),
            "idf_score": round(idf_score, 3),
            "cluster_id": cluster_id,
        })

        log.info("SOUND GROUNDING: '%s' (unit #%d) → %s:%s (count=%d, dominance=%.0f%%, idf=%.2f)",
                label, sound_unit_id, node_type, node_label, count, dominance * 100, idf_score)

    return results


async def ground_unified(pool, min_cooccurrences: int = 10) -> list[dict]:
    """Run both grounding paths: sound-based (primary) + text-based (fallback).

    Sound-based grounding is the correct path — symbol ↔ symbol co-occurrence.
    Text-based is kept as fallback for backward compatibility and when
    sound units haven't clustered enough yet.
    """
    # Primary: sound symbol path
    sound_results = await ground_from_sound_statistics(pool, min_cooccurrences)

    # Fallback: text path (diagnostic, catches concepts that sound path missed)
    text_results = await ground_from_statistics(pool, min_cooccurrences)
    for r in text_results:
        r["source"] = "text_statistical"

    # Merge: sound results take priority, text fills gaps
    grounded_nodes = {r.get("node") for r in sound_results if r["action"] == "grounded"}
    text_additions = [r for r in text_results
                      if r["action"] == "grounded" and r.get("node") not in grounded_nodes]

    return sound_results + text_additions


# ── Legacy grounding (kept for direct API use) ───────────

async def _ground_speech_label(
    pool,
    label: str,
    timestamp: float,
    confidence: float,
    active_visual_ids: list[int] | None = None,
):
    """Legacy direct grounding. Kept for API compatibility.
    The primary grounding path is now statistical (ground_from_statistics).
    """
    label_lower = label.lower().strip()

    # First, check if any active visual node IS the spoken concept directly.
    # For primitives (colors, basic shapes), the node itself is the concept.
    # No cluster needed — we create one on the spot.
    if active_visual_ids:
        direct_match = await pool.fetchrow(
            """
            SELECT id, label, node_type, observation_count
            FROM sensory_nodes
            WHERE id = ANY($1)
              AND LOWER(label) = $2
              AND node_type IN ('color', 'shape', 'texture')
              AND NOT archived
            """,
            active_visual_ids,
            label_lower,
        )

        if direct_match:
            # Found a node that IS the spoken concept. Ensure it has a cluster.
            existing_cluster = await pool.fetchrow(
                """
                SELECT c.id, c.grounded_label, c.grounding_confidence
                FROM sensory_cluster_members cm
                JOIN sensory_clusters c ON c.id = cm.cluster_id
                WHERE cm.node_id = $1
                  AND c.modality = 'visual'
                ORDER BY c.grounding_confidence DESC NULLS LAST
                LIMIT 1
                """,
                direct_match["id"],
            )

            if existing_cluster:
                cluster_id = existing_cluster["id"]
                existing_label = existing_cluster["grounded_label"]
                existing_conf = existing_cluster["grounding_confidence"] or 0.0
            else:
                # Create a single-node cluster for this primitive concept
                cluster_id = await pool.fetchval(
                    """
                    INSERT INTO sensory_clusters
                        (modality, cluster_type, member_count, total_observations,
                         coherence, coherence_at_formation)
                    VALUES ('visual', 'stable', 1, $1, 1.0, 1.0)
                    RETURNING id
                    """,
                    direct_match["observation_count"],
                )
                await pool.execute(
                    """
                    INSERT INTO sensory_cluster_members (cluster_id, node_id, role)
                    VALUES ($1, $2, 'prototype')
                    """,
                    cluster_id, direct_match["id"],
                )
                existing_label = None
                existing_conf = 0.0
                log.info("Created primitive cluster #%d for node '%s' (%s)",
                        cluster_id, direct_match["label"], direct_match["node_type"])

            # Now ground this cluster (skip to grounding logic below)
            matching_clusters = [{"cluster_id": cluster_id, "grounded_label": existing_label,
                                  "grounding_confidence": existing_conf, "active_overlap": 1,
                                  "member_count": 1}]
        else:
            # No direct match — look for clusters containing active nodes
            matching_clusters = await pool.fetch(
                """
                SELECT DISTINCT cm.cluster_id, c.grounded_label, c.grounding_confidence,
                       c.grounding_status, c.total_observations, c.cluster_type,
                       c.member_count,
                       COUNT(cm.node_id) FILTER (WHERE cm.node_id = ANY($1)) as active_overlap
                FROM sensory_cluster_members cm
                JOIN sensory_clusters c ON c.id = cm.cluster_id
                WHERE c.cluster_type IN ('stable', 'grounded')
                  AND cm.node_id = ANY($1)
                GROUP BY cm.cluster_id, c.grounded_label, c.grounding_confidence,
                         c.grounding_status, c.total_observations, c.cluster_type,
                         c.member_count
                ORDER BY COUNT(cm.node_id) FILTER (WHERE cm.node_id = ANY($1)) DESC,
                         c.total_observations DESC
                LIMIT 3
                """,
                active_visual_ids,
            )
    else:
        matching_clusters = []

    if not matching_clusters:
        # COMPOSITIONAL GROUNDING: unknown word + visual context = new concept.
        # "Pumpkin" while seeing {orange, circle, smooth} → create a compound
        # cluster containing all active visual nodes and ground it as "pumpkin".
        # This is how objects are learned: the word names the bundle of
        # sensory features being perceived at that moment.
        if active_visual_ids and confidence >= 0.4:
            compound = await _create_compound_concept(
                pool, label_lower, active_visual_ids, confidence,
            )
            if compound:
                return
        log.debug("No match for speech label '%s' (active_ids=%d)",
                 label, len(active_visual_ids or []))
        return

    best = matching_clusters[0]
    cluster_id = best["cluster_id"]
    existing_label = best.get("grounded_label")
    existing_conf = best.get("grounding_confidence") or 0.0

    speech_confidence = min(0.85, confidence)

    # Record this label as a candidate (always — even for re-confirmations)
    await pool.execute(
        """
        INSERT INTO sensory_label_candidates (cluster_id, label, tally, total_confidence)
        VALUES ($1, $2, 1, $3)
        ON CONFLICT (cluster_id, label)
        DO UPDATE SET
            tally = sensory_label_candidates.tally + 1,
            total_confidence = sensory_label_candidates.total_confidence + $3,
            last_seen = NOW()
        """,
        cluster_id, label_lower, round(speech_confidence, 4),
    )

    if not existing_label:
        # Ungrounded: first label wins immediately
        await _apply_grounding(pool, cluster_id, label_lower, speech_confidence, None)
        log.info("SPEECH GROUNDING: '%s' → cluster #%d (first label, confidence=%.3f)",
                label, cluster_id, speech_confidence)
        return

    if existing_label.lower() == label_lower:
        # Re-confirmation: strengthen confidence
        new_conf = min(1.0, existing_conf + 0.05)
        await pool.execute(
            """
            UPDATE sensory_clusters
            SET grounding_confidence = $1,
                grounding_status = CASE WHEN $1 >= 0.7 THEN 'confident' ELSE grounding_status END,
                updated_at = NOW()
            WHERE id = $2
            """,
            round(new_conf, 4), cluster_id,
        )
        log.info("Speech re-confirmed '%s' for cluster #%d (%.3f → %.3f)",
                label, cluster_id, existing_conf, new_conf)
        return

    # Conflicting label: don't overwrite immediately.
    # Decay the existing label slightly (it was challenged).
    decay = 0.03
    new_existing_conf = max(0.0, existing_conf - decay)
    await pool.execute(
        """
        UPDATE sensory_clusters
        SET grounding_confidence = $1, updated_at = NOW()
        WHERE id = $2
        """,
        round(new_existing_conf, 4), cluster_id,
    )

    # Check if the challenger has accumulated enough evidence to overtake.
    # The challenger needs more total tallies than the incumbent.
    candidates = await pool.fetch(
        """
        SELECT label, tally, total_confidence
        FROM sensory_label_candidates
        WHERE cluster_id = $1
        ORDER BY tally DESC, total_confidence DESC
        """,
        cluster_id,
    )

    if candidates:
        winner = candidates[0]
        if (winner["label"] != existing_label.lower()
                and winner["tally"] >= 3
                and winner["tally"] > sum(c["tally"] for c in candidates if c["label"] == existing_label.lower())):
            # Challenger has accumulated enough evidence — overtake
            avg_conf = winner["total_confidence"] / winner["tally"]
            await _apply_grounding(pool, cluster_id, winner["label"], avg_conf, existing_label)
            log.info("LABEL OVERTAKE: cluster #%d '%s' → '%s' (tally=%d, avg_conf=%.3f)",
                    cluster_id, existing_label, winner["label"],
                    winner["tally"], avg_conf)
        else:
            log.debug("Challenger '%s' recorded for cluster #%d (tally=%d vs incumbent '%s'), decayed to %.3f",
                     label, cluster_id,
                     next((c["tally"] for c in candidates if c["label"] == label_lower), 0),
                     existing_label, new_existing_conf)


async def _apply_grounding(pool, cluster_id, label, confidence, previous_label):
    """Apply a grounding label to a cluster."""
    await pool.execute(
        """
        UPDATE sensory_clusters
        SET grounded_label = $1,
            grounding_confidence = $2,
            grounding_status = 'speculative',
            cluster_type = 'grounded',
            observations_at_grounding = total_observations,
            observations_at_last_challenge = total_observations,
            updated_at = NOW()
        WHERE id = $3
        """,
        label, round(confidence, 4), cluster_id,
    )
    await pool.execute(
        """
        INSERT INTO sensory_grounding_log
            (cluster_id, symbol_node_id, symbol_label, confidence,
             grounding_type, previous_label, rechallenge_reason)
        VALUES ($1, 0, $2, $3, 'speech_cooccurrence', $4, NULL)
        """,
        cluster_id, label, round(confidence, 4), previous_label,
    )


async def _create_compound_concept(
    pool,
    label: str,
    active_visual_ids: list[int],
    confidence: float,
) -> bool:
    """Create a compound concept from the current visual context.

    When an unknown word is heard while seeing a combination of visual
    primitives, this creates a new cluster containing those primitives
    and grounds it with the spoken word.

    "Pumpkin" while seeing {color:orange, shape:circle-orange-smooth}
    → cluster "pumpkin" = {orange, circle-orange-smooth}

    The concept IS the bundle of sensory features perceived at the
    moment of naming. This is how children learn object words.

    Returns True if a concept was created.
    """
    # Check if a compound concept with this label already exists
    existing = await pool.fetchrow(
        """
        SELECT id, grounding_confidence, member_count
        FROM sensory_clusters
        WHERE grounded_label = $1 AND modality = 'visual'
        """,
        label,
    )

    if existing:
        # Concept already exists — add any new active nodes to it
        # (seeing "pumpkin" again might reveal new features)
        existing_members = await pool.fetch(
            "SELECT node_id FROM sensory_cluster_members WHERE cluster_id = $1",
            existing["id"],
        )
        existing_ids = {r["node_id"] for r in existing_members}

        new_ids = [vid for vid in active_visual_ids if vid not in existing_ids]
        for nid in new_ids:
            await pool.execute(
                """
                INSERT INTO sensory_cluster_members (cluster_id, node_id, role)
                VALUES ($1, $2, 'member')
                ON CONFLICT (cluster_id, node_id) DO NOTHING
                """,
                existing["id"], nid,
            )

        if new_ids:
            await pool.execute(
                """
                UPDATE sensory_clusters
                SET member_count = member_count + $1,
                    grounding_confidence = LEAST(1.0, grounding_confidence + 0.03),
                    updated_at = NOW()
                WHERE id = $2
                """,
                len(new_ids), existing["id"],
            )
            log.info("COMPOUND ENRICHED: '%s' (cluster #%d) gained %d new features (now %d members)",
                    label, existing["id"], len(new_ids),
                    existing["member_count"] + len(new_ids))
        else:
            # Pure re-confirmation
            await pool.execute(
                """
                UPDATE sensory_clusters
                SET grounding_confidence = LEAST(1.0, grounding_confidence + 0.05),
                    grounding_status = CASE WHEN grounding_confidence + 0.05 >= 0.7
                                            THEN 'confident' ELSE grounding_status END,
                    updated_at = NOW()
                WHERE id = $1
                """,
                existing["id"],
            )
            log.info("COMPOUND re-confirmed: '%s' (cluster #%d, conf→%.3f)",
                    label, existing["id"],
                    min(1.0, (existing["grounding_confidence"] or 0) + 0.05))

        return True

    # Get info about the active visual nodes
    nodes = await pool.fetch(
        """
        SELECT id, label, node_type, observation_count
        FROM sensory_nodes
        WHERE id = ANY($1) AND NOT archived AND modality = 'visual'
        ORDER BY observation_count DESC
        """,
        active_visual_ids,
    )

    if not nodes:
        return False

    # Create the compound cluster
    total_obs = sum(n["observation_count"] for n in nodes)
    cluster_id = await pool.fetchval(
        """
        INSERT INTO sensory_clusters
            (modality, cluster_type, member_count, total_observations,
             coherence, coherence_at_formation, grounded_label,
             grounding_confidence, grounding_status,
             observations_at_grounding, observations_at_last_challenge)
        VALUES ('visual', 'grounded', $1, $2, 1.0, 1.0, $3, $4, 'speculative', $2, $2)
        RETURNING id
        """,
        len(nodes),
        total_obs,
        label,
        round(min(0.7, confidence), 4),
    )

    # Add all active visual nodes as members
    for i, node in enumerate(nodes):
        role = "prototype" if i == 0 else "member"
        await pool.execute(
            """
            INSERT INTO sensory_cluster_members (cluster_id, node_id, role)
            VALUES ($1, $2, $3)
            ON CONFLICT (cluster_id, node_id) DO NOTHING
            """,
            cluster_id, node["id"], role,
        )

    # Log it
    await pool.execute(
        """
        INSERT INTO sensory_grounding_log
            (cluster_id, symbol_node_id, symbol_label, confidence,
             grounding_type, previous_label, rechallenge_reason)
        VALUES ($1, 0, $2, $3, 'compositional', NULL, NULL)
        """,
        cluster_id, label, round(min(0.7, confidence), 4),
    )

    # Log the component features for visibility
    feature_str = ", ".join(f"{n['node_type']}:{n['label']}" for n in nodes[:6])
    if len(nodes) > 6:
        feature_str += f" +{len(nodes)-6} more"

    log.info("COMPOUND CONCEPT: '%s' → cluster #%d (%d features: %s)",
            label, cluster_id, len(nodes), feature_str)

    return True


def create_flush_callback(accumulator: SpeechAccumulator, nmem_write_fn=None):
    """Create a callback to flush the accumulator at end of video.

    Call this after video ingestion completes to process any remaining
    buffered words.
    """
    async def flush():
        events = accumulator.flush()
        for event in events:
            if nmem_write_fn:
                try:
                    await nmem_write_fn(
                        content=event.label,
                        metadata={
                            "source": "stt",
                            "original_text": event.text,
                            "timestamp": event.timestamp,
                            "event_type": event.event_type,
                            "confidence": event.confidence,
                        },
                    )
                except Exception as e:
                    log.warning("nmem write failed for '%s': %s", event.label, e)
        return events

    return flush
