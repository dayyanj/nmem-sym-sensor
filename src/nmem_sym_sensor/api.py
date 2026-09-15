"""
Public API for nmem-sym-sensor.

SensorGraph is the single entry point for library consumers.
"""
import logging
from datetime import datetime

import asyncpg
from sentence_transformers import SentenceTransformer

from nmem_sym_sensor import config
from nmem_sym_sensor.audio import AudioAnalysis, analyze_audio
from nmem_sym_sensor.binding import (
    bind_frame_audio,
    bind_within_modality,
    get_binding_candidates,
)
from nmem_sym_sensor.consolidation import run_consolidation
from nmem_sym_sensor.graph import (
    buffer_observation,
    flush_expired_iconic,
    graph_stats,
    promote_from_iconic,
    upsert_edge,
)
from nmem_sym_sensor.visual import FrameAnalysis, analyze_frame

log = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required: pip install numpy")


class SensorGraph:
    """Sensory memory graph with visual and audio cognition.

    Usage::

        async with SensorGraph(db_dsn="postgresql://...") as sg:
            # Ingest a visual frame
            node_ids, _ = await sg.ingest_frame(image)

            # Ingest audio
            audio_ids, _ = await sg.ingest_audio(waveform)

            # Bind cross-modal
            await sg.bind(node_ids, audio_ids)

            # Consolidate (cluster, promote, form concepts)
            await sg.consolidate()

            # Ground to nmem-sym
            await sg.ground(sym_dsn="postgresql://...")
    """

    def __init__(
        self,
        db_dsn: str | None = None,
        embed_model: str | None = None,
        visual_backend: str | None = None,
    ):
        self.db_dsn = db_dsn or config.DB_DSN
        if not self.db_dsn:
            raise ValueError("db_dsn required — set NMEM_SENSOR_DB_DSN or pass explicitly")

        self.embed_model_name = embed_model or config.EMBED_MODEL
        self.visual_backend = visual_backend or config.VISUAL_BACKEND

        self._pool: asyncpg.Pool | None = None
        self._embedder: SentenceTransformer | None = None
        self._foveal = None  # FovealAttention instance (if enabled)
        self._tracker = None  # ObjectTracker for temporal continuity
        self._sym_pool: asyncpg.Pool | None = None  # nmem-sym pool (optional)
        self._emb_predictor = None   # Embedding-space predictor (JEPA-inspired)
        self._sound_language = None  # SoundLanguage for emergent language learning
        self._imagery = None         # MentalImagery for cross-modal activation
        self._vocal_tract = None     # VocalTract for speech production
        self._saccade_memory = None  # SaccadeMemory for eye movement patterns
        self._tabula2 = None         # TABULA2 auditory cortex (disentangler)
        self._attention = None       # AttentionController (three-phase state machine)
        self._cluster_cache = None   # ClusterCache for object-level familiarity
        self._chronoception = None   # Chronoception for scene-gated activation
        self._pressure = None        # DreamstatePressure (demand-driven consolidation)

    async def connect(self):
        """Initialize connection pool and embedding model."""
        self._pool = await asyncpg.create_pool(self.db_dsn)
        await self._run_migrations()
        _device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self._embedder = SentenceTransformer(self.embed_model_name, device=_device)

        # Initialize attention controller + scene awareness
        if config.FAMILIARITY_ENABLED:
            from nmem_sym_sensor.attention_controller import AttentionController
            from nmem_sym_sensor.chronoception import Chronoception
            from nmem_sym_sensor.dreamstate_pressure import DreamstatePressure
            from nmem_sym_sensor.familiarity import ClusterCache
            self._attention = AttentionController()
            self._cluster_cache = ClusterCache()
            await self._cluster_cache.refresh(self._pool)
            self._chronoception = Chronoception()
            self._pressure = DreamstatePressure()
            log.info("Attention controller + pressure system enabled (cluster cache: %d entries)",
                     self._cluster_cache.size)

        # Initialize foveal attention if enabled
        if config.FOVEAL_ATTENTION_ENABLED:
            from nmem_sym_sensor.visual_attention import FovealAttention, FovealConfig
            self._foveal = FovealAttention(FovealConfig.from_config())
            log.info("Foveal attention enabled")

        # Initialize object tracker for temporal continuity
        if config.INTELLIGENCE_LOOPS_ENABLED:
            from nmem_sym_sensor.temporal import ObjectTracker
            self._tracker = ObjectTracker()

            # Initialize embedding-space predictor
            from nmem_sym_sensor.embedding_predictor import EmbeddingPredictor
            self._emb_predictor = EmbeddingPredictor(
                embed_dim=512, context_len=3, hidden_dim=256, lr=0.001,
            )
            # Load saved weights if available
            import os
            predictor_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "checkpoints", "emb_predictor.npz"
            )
            if os.path.exists(predictor_path):
                self._emb_predictor.load(predictor_path)
                log.info("Loaded embedding predictor (%d prior predictions)",
                         self._emb_predictor.total_predictions)
            else:
                log.info("Embedding predictor initialized (fresh)")

            # Connect to nmem-sym if DSN available
            if config.SYM_DB_DSN:
                try:
                    self._sym_pool = await asyncpg.create_pool(config.SYM_DB_DSN)
                    log.info("Connected to nmem-sym for intelligence loops")
                except Exception as e:
                    log.warning("nmem-sym connection failed (loops run without symbolic predictions): %s", e)

        # Initialize sound-level language learning (independent of intelligence loops)
        if config.LANGUAGE_LEARNING_ENABLED:
            from nmem_sym_sensor.language import SoundLanguage
            self._sound_language = SoundLanguage(
                self._pool, similarity_threshold=config.SOUND_SIMILARITY_THRESHOLD,
            )
            log.info("Sound language learning enabled (threshold=%.2f)", config.SOUND_SIMILARITY_THRESHOLD)

        # Initialize mental imagery (cross-modal activation)
        if config.LANGUAGE_LEARNING_ENABLED:
            from nmem_sym_sensor.imagery import MentalImagery
            self._imagery = MentalImagery(self._pool)

        # Initialize oculomotor saccade memory (eye movement patterns)
        if config.OCULOMOTOR_ENABLED:
            from nmem_sym_sensor.oculomotor import SaccadeMemory
            self._saccade_memory = SaccadeMemory(
                self._pool, similarity_threshold=config.SACCADE_SIMILARITY_THRESHOLD,
            )
            log.info("Oculomotor saccade memory enabled (threshold=%.2f)", config.SACCADE_SIMILARITY_THRESHOLD)

        # Initialize TABULA2 auditory cortex (replaces formant embedding)
        if config.TABULA2_ENABLED:
            try:
                from nmem_sym_sensor.tabula2 import TABULA2Disentangler
                self._tabula2 = TABULA2Disentangler(
                    checkpoint_path=config.TABULA2_CHECKPOINT,
                    decoder_path=config.TABULA2_DECODER_CHECKPOINT,
                    vad_threshold=config.TABULA2_VAD_THRESHOLD,
                )
                log.info("TABULA2 auditory cortex enabled (voice=%d-dim)", config.TABULA2_VOICE_DIM)
            except Exception as e:
                log.warning("TABULA2 load failed, falling back to formant: %s", e)

        # Initialize vocal tract (speech production) via TABULA2 decoder
        if self._tabula2 is not None:
            from nmem_sym_sensor.vocal_tract import VocalTract
            self._vocal_tract = VocalTract(self._pool, tabula2=self._tabula2)
            await self._vocal_tract.init_db()
            log.info("Vocal tract enabled (TABULA2 voice decoder)")

        # Initialize speech hierarchy (syllable tables, cache, predictor)
        if config.LANGUAGE_LEARNING_ENABLED:
            from nmem_sym_sensor.language import init_speech_hierarchy
            await init_speech_hierarchy(self._pool)
            from nmem_sym_sensor.speech_cache import SpeechCache
            self._speech_cache = SpeechCache()
            await self._speech_cache.refresh(self._pool)
            from nmem_sym_sensor.speech_prediction import SpeechPredictor
            self._speech_predictor = SpeechPredictor(self._pool, speech_cache=self._speech_cache)
            log.info("Speech hierarchy enabled (cache: %d myelinated syllables)", self._speech_cache.size)

    async def _run_migrations(self):
        """Apply pending sensory-schema migrations via the shared nmem migration runner.

        Uses the SAME generic runner as nmem-sym (`nmem.migrate.MigrationRunner`), pointed at this
        lib's migrations dir + pool. Bootstrap first marks any hand-applied schema as already-applied
        (so an agent whose sensory_* tables predate migrations reconciles without a rebuild). No-op
        when nmem.migrate isn't importable (standalone sensor usage)."""
        from pathlib import Path
        try:
            from nmem.migrate import MigrationRunner
        except ImportError:
            return  # standalone usage without nmem core
        migrations_dir = Path(__file__).parent / "migrations"
        if not migrations_dir.is_dir():
            return
        runner = MigrationRunner(
            project="nmem-sym-sensor",
            migrations_dir=migrations_dir,
            pool=self._pool,
        )
        from nmem_sym_sensor.migrate_bootstrap import bootstrap_existing
        await bootstrap_existing(runner)
        await runner.run()

    async def close(self):
        """Close the connection pool and save predictor state."""
        if self._emb_predictor and self._emb_predictor.total_updates > 0:
            import os
            predictor_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "checkpoints", "emb_predictor.npz"
            )
            os.makedirs(os.path.dirname(predictor_path), exist_ok=True)
            self._emb_predictor.save(predictor_path)
            log.info("Saved embedding predictor (%d predictions, running_error=%.4f)",
                     self._emb_predictor.total_predictions,
                     self._emb_predictor.running_error)
        if self._sym_pool:
            await self._sym_pool.close()
            self._sym_pool = None
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def last_scene_id(self) -> int | None:
        """The chronoception scene the most recent ``ingest_frame`` resolved to (recognized OR newly
        created), or None when scene tracking is off / no frame seen yet. Lets a caller associate a
        whole-screen "place" with the pursuit it occurred in (scene-level episodic memory)."""
        scene = getattr(self._chronoception, "current_scene", None) if self._chronoception else None
        return getattr(scene, "scene_id", None) if scene else None

    @property
    def pool(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("Not connected — call connect() or use as async context manager")
        return self._pool

    @property
    def embedder(self) -> SentenceTransformer:
        if not self._embedder:
            raise RuntimeError("Not connected — call connect() or use as async context manager")
        return self._embedder

    # ── Visual ingestion ─────────────────────────────────

    async def ingest_frame(
        self,
        image: np.ndarray,
        frame_id: str | None = None,
        backend: str | None = None,
        scene_change: bool = False,
        depth: str = "full",
    ) -> tuple[list[int], "FrameAnalysis"]:
        """Analyze an image frame and ingest primitives into the sensory graph.

        Primitives first enter the iconic buffer. After enough observations
        they are promoted to short-term sensory nodes.

        If foveal attention is enabled, the attention system decides where
        to focus and runs detailed analysis on the attended region.

        Args:
            image: HWC uint8 image array.
            frame_id: Optional frame identifier.
            backend: Visual backend ("edge").
            scene_change: If True, resets foveal attention state.

        Returns:
            Tuple of (buffer entry IDs, FrameAnalysis with fixation data).
        """
        frame_id = frame_id or f"frame_{datetime.now().timestamp():.0f}"

        # Attention controller: compute budget from knowledge signals
        self._last_budget = None
        if self._attention is not None:
            from nmem_sym_sensor.attention_controller import AttentionSignals
            from nmem_sym_sensor.familiarity import SceneMemory

            # Gather signals
            pred_acc = max(0.0, 1.0 - (self._emb_predictor.running_error if self._emb_predictor else 0.5))
            coverage = self._foveal._tracker.candidate_coverage if self._foveal else 0.0
            new_cands = self._foveal._tracker.new_candidates_this_frame if self._foveal else 0
            surprise = self._emb_predictor.running_error if self._emb_predictor else 0.5

            # Chronoception: scene tracking
            chrono_scene_changed = False
            chrono_return_visit = False
            if self._chronoception is not None:
                try:
                    # Compute coarse scene embedding (8x8 color, whole frame)
                    coarse_emb = SceneMemory._coarse_embedding(image)
                    chrono_status = await self._chronoception.on_frame(coarse_emb, self._pool)
                    chrono_scene_changed = chrono_status.get("scene_changed", False)
                    if chrono_scene_changed:
                        chrono_return_visit = self._chronoception.current_scene is not None
                        log.info("Scene change detected (scene_id=%s, active=%d clusters)",
                                 chrono_status.get("new_scene_id"),
                                 chrono_status["active_set"]["active"])
                except Exception as e:
                    log.warning("Chronoception frame failed: %s", e, exc_info=True)

            signals = AttentionSignals(
                prediction_accuracy=pred_acc,
                candidate_coverage=coverage,
                new_candidate_count=new_cands,
                surprise=surprise,
                is_return_visit=chrono_return_visit,
                scene_changed=scene_change or chrono_scene_changed,
            )
            old_phase = self._attention.phase.value
            self._last_budget = self._attention.update(signals)
            if self._cluster_cache is not None:
                self._last_budget.cluster_cache = self._cluster_cache

            # Feed pressure system with surprise and phase changes
            if self._pressure is not None:
                import time as _time
                now = _time.monotonic()
                self._pressure.surprise.observe(now, surprise)
                self._pressure.prediction_accuracy.observe(now, pred_acc)
                new_phase = self._attention.phase.value
                if new_phase != old_phase:
                    self._pressure.idle.on_phase_change(new_phase, now)
                elif self._pressure.idle.monitor_entered_at is None and new_phase == "monitor":
                    # First frame: ensure idle tracker knows we're in monitor
                    self._pressure.idle.on_phase_change(new_phase, now)

        if depth == "minimal":
            # Reduced: edge analysis + buffer observation only (no foveal, no intelligence)
            analysis = analyze_frame(image, frame_id)
        elif self._foveal is not None:
            analysis = self._foveal.process_frame(image, frame_id, scene_change, budget=self._last_budget)
        else:
            analysis = analyze_frame(image, frame_id)

        # After analysis: update signals with promoted count + record scene observations
        if self._attention is not None and self._last_budget is not None:
            self._last_budget.promoted_visual_count = len(analysis.primitives)

        if self._chronoception is not None and self._chronoception.current_scene is not None:
            # Build label→cluster_id map from familiar primitives
            label_to_cluster: dict[str, int] = {}
            for p in analysis.primitives:
                feat = p.features or {}
                cid = feat.get("cluster_match_id")
                if cid is not None:
                    try:
                        await self._chronoception.observe_cluster_in_scene(cid, self._pool)
                        self._chronoception.active_set.reinforce(cid)
                    except Exception:
                        pass
                    if p.label:
                        label_to_cluster[p.label] = cid

            # Record allocentric spatial relations between recognised clusters
            RELATION_MAP = {"adjacent_to": "near", "overlaps": "near"}
            for rel in analysis.relations:
                cid_a = label_to_cluster.get(rel.source_label)
                cid_b = label_to_cluster.get(rel.target_label)
                if cid_a is not None and cid_b is not None and cid_a != cid_b:
                    mapped = RELATION_MAP.get(rel.relation, rel.relation)
                    try:
                        await self._chronoception.observe_spatial_relation(
                            cid_a, cid_b, mapped, self._pool,
                        )
                    except Exception:
                        pass

        entry_ids = await self._ingest_visual_analysis(analysis)
        return entry_ids, analysis

    async def _ingest_visual_analysis(self, analysis: FrameAnalysis) -> list[int]:
        """Ingest a visual analysis into the graph.

        Stores primitives in the iconic buffer AND persists spatial
        relations (part_of, contains, adjacent_to) as sensory edges.

        Node lookup for edges uses label→embedding mapping from the
        current frame's primitives, falling back to label-based search.
        """
        entry_ids = []
        # Map primitive labels to their embeddings for node lookup
        label_to_embedding: dict[str, list[float]] = {}

        # Vector-space tag for this frame's VISUAL vectors (which encoder produced them). Resolved
        # once per frame; carried onto the iconic row → promoted node so similarity stays within
        # one space (a JEPA-on run and a geometric-only fallback are NOT comparable). Audio prims
        # keep None (a different embedder; tagging audio is out of scope here).
        from nmem_sym_sensor.visual import current_visual_embedder_id
        _visual_embedder_id = current_visual_embedder_id()

        for prim in analysis.primitives:
            feat = dict(prim.features) if prim.features else {}
            if prim.bbox:
                bx, by, bw, bh = prim.bbox
                feat["bbox_x"] = bx
                feat["bbox_y"] = by
                feat["bbox_w"] = bw
                feat["bbox_h"] = bh
                feat["center_x"] = bx + bw / 2
                feat["center_y"] = by + bh / 2

            entry_id = await buffer_observation(
                self.pool,
                modality="visual",
                node_type=prim.node_type,
                label=prim.label,
                features=feat,
                embedding=prim.embedding,
                frame_id=analysis.frame_id,
                embedder_id=_visual_embedder_id,
            )
            entry_ids.append(entry_id)
            if prim.embedding is not None:
                label_to_embedding[prim.label] = prim.embedding

        # Persist spatial relations as edges between nodes
        if analysis.relations:
            for rel in analysis.relations:
                src_id = await self._find_node_for_label(
                    rel.source_label, label_to_embedding.get(rel.source_label),
                )
                tgt_id = await self._find_node_for_label(
                    rel.target_label, label_to_embedding.get(rel.target_label),
                )
                if src_id and tgt_id:
                    await upsert_edge(
                        self.pool,
                        source_id=src_id,
                        target_id=tgt_id,
                        edge_type=rel.relation,
                        confidence=rel.confidence,
                        source_ref={"frame": analysis.frame_id, "type": "spatial"},
                    )

                    if rel.relation == "part_of":
                        from nmem_sym_sensor.composition import record_part_whole
                        sibling_ids = []
                        for other_rel in analysis.relations:
                            if (other_rel.relation == "part_of"
                                    and other_rel.target_label == rel.target_label
                                    and other_rel.source_label != rel.source_label):
                                sib_id = await self._find_node_for_label(
                                    other_rel.source_label,
                                    label_to_embedding.get(other_rel.source_label),
                                )
                                if sib_id:
                                    sibling_ids.append(sib_id)
                        await record_part_whole(
                            self.pool, src_id, tgt_id, sibling_ids,
                        )

            log.debug("Frame %s: %d primitives, %d relations persisted",
                     analysis.frame_id, len(analysis.primitives), len(analysis.relations))

        return entry_ids

    async def _find_node_for_label(
        self,
        label: str,
        embedding: list[float] | None = None,
    ) -> int | None:
        """Find a node ID by embedding similarity, falling back to label search.

        Primary: embedding cosine similarity >= 0.85.
        Fallback: label ILIKE match (for nodes without embeddings).
        """
        if embedding is not None:
            from nmem_sym_sensor.graph import _dumps
            emb_val = _dumps(embedding)
            row = await self.pool.fetchrow(
                """
                SELECT id, 1 - (visual_embedding <=> $1::vector) as similarity
                FROM sensory_nodes
                WHERE modality = 'visual' AND NOT archived
                  AND visual_embedding IS NOT NULL
                ORDER BY visual_embedding <=> $1::vector
                LIMIT 1
                """,
                emb_val,
            )
            if row and row["similarity"] >= 0.85:
                return row["id"]

        # Fallback: label search (for color nodes etc. without embeddings)
        row = await self.pool.fetchrow(
            """SELECT id FROM sensory_nodes
               WHERE label ILIKE $1 AND modality = 'visual' AND NOT archived
               ORDER BY updated_at DESC LIMIT 1""",
            label,
        )
        return row["id"] if row else None

    # ── Audio ingestion ──────────────────────────────────

    async def ingest_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int | None = None,
        window_id: str | None = None,
    ) -> tuple[list[int], object, bool]:
        """Analyze an audio waveform and ingest primitives into the sensory graph.

        Args:
            waveform: Audio samples (1D or 2D array).
            sample_rate: Sample rate in Hz.
            window_id: Optional window identifier.

        Returns:
            Tuple of (buffer_entry_ids, sound_data, has_speech).
            sound_data is list[SyllableBurst] when TABULA2 active,
            or list[float] (formant embedding) as legacy fallback.
        """
        window_id = window_id or f"window_{datetime.now().timestamp():.0f}"
        sr = sample_rate or config.AUDIO_SAMPLE_RATE

        # Standard audio analysis (timbre, frequency, rhythm) — legacy path.
        # Skip when TABULA2 is active: TABULA2 disentangles voice/noise properly.
        # The legacy analysis creates raw-audio neurons (bass-109hz etc.) that
        # bypass disentanglement and pollute visual co-occurrences.
        entry_ids = []
        if self._tabula2 is None:
            analysis = analyze_audio(waveform, sr, window_id)
            entry_ids = await self._ingest_audio_analysis(analysis)

        if self._tabula2 is not None:
            # TABULA2 path: disentangled voice/noise → syllable bursts
            from nmem_sym_sensor.syllable_segmenter import segment_syllables

            t2_output = self._tabula2.process(waveform, sr)

            # Voice stream: VAD-gated syllable bursts
            bursts = segment_syllables(
                t2_output.voice_embeddings,
                t2_output.vad_mask,
                t2_output.vad_features,
                min_frames=config.TABULA2_MIN_SYLLABLE_FRAMES,
                sub_features=t2_output.sub_features,
            )

            # Noise stream: mean embedding for environmental sounds
            # Only when there's significant noise energy (not silence)
            noise_emb = None
            noise_energy = np.abs(t2_output.noise_embeddings).mean()
            if noise_energy > 0.5:  # absolute threshold like voice VAD
                noise_emb = t2_output.noise_embeddings.mean(axis=0)  # (512,)
                norm = np.linalg.norm(noise_emb)
                if norm > 0:
                    noise_emb = noise_emb / norm

            has_speech = len(bursts) > 0
            # Store noise embedding on the bursts list as metadata
            # so video.py can access it
            return entry_ids, (bursts, noise_emb), has_speech
        else:
            # Legacy: formant embedding
            return entry_ids, analysis.formant_embedding, analysis.has_onset

    async def _ingest_audio_analysis(self, analysis: AudioAnalysis) -> list[int]:
        """Ingest an audio analysis into the graph."""
        entry_ids = []

        for prim in analysis.primitives:
            entry_id = await buffer_observation(
                self.pool,
                modality="audio",
                node_type=prim.node_type,
                label=prim.label,
                features=prim.features,
                embedding=prim.embedding,
                frame_id=analysis.window_id,
            )
            entry_ids.append(entry_id)

        return entry_ids

    # ── Cross-modal binding ──────────────────────────────

    async def bind(
        self,
        visual_node_ids: list[int],
        audio_node_ids: list[int],
        timestamp: datetime | None = None,
    ) -> list[int]:
        """Bind visual and audio observations that occurred simultaneously.

        Args:
            visual_node_ids: Node IDs from visual ingestion.
            audio_node_ids: Node IDs from audio ingestion.
            timestamp: When the binding occurred.

        Returns:
            List of new cross-modal edge IDs.
        """
        return await bind_frame_audio(self.pool, visual_node_ids, audio_node_ids, timestamp)

    async def bind_intra(self, node_ids: list[int]) -> list[int]:
        """Bind nodes within the same modality (same frame/window)."""
        return await bind_within_modality(self.pool, node_ids)

    async def binding_candidates(self, min_count: int | None = None) -> list[dict]:
        """Get current binding candidates (diagnostic)."""
        return await get_binding_candidates(self.pool, min_count)

    # ── Consolidation ────────────────────────────────────

    async def consolidate(self, dry_run: bool = False) -> dict:
        """Run a full consolidation cycle.

        Promotes iconic → short-term → long-term,
        clusters primitives, forms concepts.
        """
        result = await run_consolidation(self.pool, dry_run=dry_run, vocal_tract=self._vocal_tract)

        # Refresh cluster cache after consolidation (new clusters may have formed)
        if self._cluster_cache is not None:
            await self._cluster_cache.refresh(self._pool)

        # Refresh speech cache after consolidation (new syllables may have been discovered)
        if hasattr(self, '_speech_cache') and self._speech_cache is not None:
            await self._speech_cache.refresh(self._pool)

        return result

    async def promote_iconic(self) -> list[int]:
        """Manually promote iconic buffer entries."""
        return await promote_from_iconic(self.pool)

    async def flush_iconic(self) -> int:
        """Manually flush expired iconic buffer entries."""
        return await flush_expired_iconic(self.pool)

    # ── Grounding to nmem-sym ────────────────────────────

    async def ground(
        self,
        sym_dsn: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Ground stable sensory clusters to nmem-sym symbol nodes.

        Requires a separate connection to the nmem-sym database.

        Args:
            sym_dsn: DSN for the nmem-sym database. Defaults to NMEM_SYM_DB_DSN env var.
            limit: Max clusters to process.

        Returns:
            List of grounding results.
        """
        import os

        from nmem_sym_sensor.bridge import ground_clusters_by_label

        dsn = sym_dsn or os.environ.get("NMEM_SYM_DB_DSN")
        if not dsn:
            raise ValueError("sym_dsn required for grounding — set NMEM_SYM_DB_DSN or pass explicitly")

        sym_pool = await asyncpg.create_pool(dsn)
        try:
            return await ground_clusters_by_label(self.pool, sym_pool, self.embedder, limit)
        finally:
            await sym_pool.close()

    async def ground_label(
        self,
        cluster_id: int,
        text_label: str,
        sym_dsn: str | None = None,
    ) -> dict | None:
        """Ground a specific cluster via a text label (temporal co-occurrence)."""
        import os

        from nmem_sym_sensor.bridge import ground_by_temporal_cooccurrence

        dsn = sym_dsn or os.environ.get("NMEM_SYM_DB_DSN")
        if not dsn:
            raise ValueError("sym_dsn required for grounding")

        sym_pool = await asyncpg.create_pool(dsn)
        try:
            return await ground_by_temporal_cooccurrence(
                self.pool, sym_pool, cluster_id, text_label, self.embedder,
            )
        finally:
            await sym_pool.close()

    # ── Description & language ─────────────────────────────

    async def describe_cluster(self, cluster_id: int):
        """Describe a sensory cluster in natural language.

        Returns a SensoryDescription with three modes:
          - structural: graph-notation for precise LLM context injection
          - narrative: flowing natural language description
          - query: "what is this?" prompt for LLM naming
        """
        from nmem_sym_sensor.describe import describe_cluster
        return await describe_cluster(self.pool, cluster_id)

    async def describe_node(self, node_id: int, max_hops: int = 2):
        """Describe a specific sensory node and its neighborhood."""
        from nmem_sym_sensor.describe import describe_node
        return await describe_node(self.pool, node_id, max_hops)

    async def describe_active(self, node_ids: list[int]):
        """Describe a set of currently-active sensory nodes (live scene)."""
        from nmem_sym_sensor.describe import describe_active
        return await describe_active(self.pool, node_ids)

    async def describe_ungrounded(self, limit: int = 10):
        """Describe all stable but unnamed clusters.

        Returns descriptions sorted by observation count. Feed each
        description's .query to an LLM and pass the response to
        ground_label() to create speculative groundings.
        """
        from nmem_sym_sensor.describe import describe_all_ungrounded
        return await describe_all_ungrounded(self.pool, limit)

    async def auto_name(
        self,
        llm_callable,
        sym_dsn: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Automatically name ungrounded clusters using an LLM.

        The full naming loop:
          1. Find stable ungrounded clusters
          2. Generate a query description for each
          3. Send query to LLM (caller provides the callable)
          4. Ground the LLM's response as a speculative label

        Args:
            llm_callable: Async function that takes a prompt string and
                returns a response string. E.g.::

                    async def ask_llm(prompt: str) -> str:
                        resp = await client.chat(messages=[{"role": "user", "content": prompt}])
                        return resp.content.strip()

            sym_dsn: DSN for nmem-sym database (for grounding).
            limit: Max clusters to name per call.

        Returns:
            List of dicts with cluster_id, proposed_label, grounding_result.
        """
        from nmem_sym_sensor.describe import describe_all_ungrounded

        descriptions = await describe_all_ungrounded(self.pool, limit)
        results = []

        for desc in descriptions:
            if not desc.query:
                continue

            try:
                proposed_label = await llm_callable(desc.query)
                proposed_label = proposed_label.strip().strip('"').strip("'")

                # Skip nonsense responses
                if not proposed_label or len(proposed_label) > 50:
                    continue
                if proposed_label.lower() in ("unknown", "i don't know", "unsure"):
                    continue

                # Ground it (speculatively)
                grounding = None
                if sym_dsn or __import__("os").environ.get("NMEM_SYM_DB_DSN"):
                    try:
                        grounding = await self.ground_label(
                            desc.cluster_id, proposed_label, sym_dsn
                        )
                    except Exception as e:
                        log.warning("Grounding failed for '%s': %s", proposed_label, e)

                results.append({
                    "cluster_id": desc.cluster_id,
                    "proposed_label": proposed_label,
                    "observation_count": desc.observation_count,
                    "confidence": desc.confidence,
                    "grounding_result": grounding,
                })
                log.info("Auto-named cluster #%d → '%s'",
                        desc.cluster_id, proposed_label)

            except Exception as e:
                log.warning("LLM naming failed for cluster #%d: %s",
                           desc.cluster_id, e)

        return results

    # ── Rechallenge ──────────────────────────────────────

    async def rechallenge(
        self,
        llm_callable,
        max_challenges: int = 5,
    ) -> list:
        """Run a rechallenge cycle on grounded clusters.

        Decays confidence, detects drift, and re-queries the LLM for
        clusters whose labels may no longer fit. Call this periodically
        (e.g., after every consolidation cycle).

        Args:
            llm_callable: Async function(prompt) → response string.
            max_challenges: Max clusters to rechallenge per cycle.

        Returns:
            List of RechallengeResults.
        """
        from nmem_sym_sensor.rechallenge import run_rechallenge_cycle
        return await run_rechallenge_cycle(self.pool, llm_callable, max_challenges)

    async def rechallenge_cluster(
        self,
        cluster_id: int,
        llm_callable,
        reason: str = "manual",
    ):
        """Rechallenge a specific cluster's label.

        Use when you have reason to believe a label is wrong.
        """
        from nmem_sym_sensor.rechallenge import rechallenge_cluster
        return await rechallenge_cluster(self.pool, cluster_id, llm_callable, reason)

    async def rechallenge_candidates(self) -> list[dict]:
        """List clusters that are candidates for rechallenging.

        Diagnostic: shows which clusters have drifted, decayed, or
        accumulated enough new observations to warrant re-evaluation.
        """
        from nmem_sym_sensor.rechallenge import find_rechallenge_candidates
        return await find_rechallenge_candidates(self.pool)

    # ── Recognition ───────────────────────────────────────

    def create_recognition_engine(self, llm_callable=None, **kwargs):
        """Create a recognition engine for real-time interpretation.

        The engine caches known cluster→label mappings for instant
        recognition. Unknown clusters are sent to the LLM.

        Args:
            llm_callable: Async function(prompt) → response string.
            **kwargs: Additional args for RecognitionEngine.

        Returns:
            RecognitionEngine instance.
        """
        from nmem_sym_sensor.recognize import RecognitionEngine
        return RecognitionEngine(self.pool, llm_callable, **kwargs)

    # ── Compositional inference ─────────────────────────

    async def infer_whole(
        self,
        node_id: int,
        active_node_ids: list[int] | None = None,
    ) -> list[dict]:
        """Given a visible part, infer what whole it belongs to.

        Uses learned structural expectations to predict the unseen whole.
        Checks context (are expected sibling parts present?) to distinguish
        "close-up of an eye in a face" from "isolated eyeball on a tray."

        Args:
            node_id: The visible part node.
            active_node_ids: Other currently visible nodes (for sibling verification).

        Returns:
            Ranked list of possible wholes with confidence and context match.
        """
        from nmem_sym_sensor.composition import infer_whole_from_part
        return await infer_whole_from_part(self.pool, node_id, active_node_ids)

    async def should_decompose(self, node_id: int) -> dict:
        """Check whether a node needs finer decomposition.

        Returns recommendation with reason (e.g., "confused with 5 similar
        nodes — finer detail needed to discriminate").
        """
        from nmem_sym_sensor.composition import should_decompose
        return await should_decompose(self.pool, node_id)

    # ── Intelligence loops ─────────────────────────────────

    async def run_intelligence_cycle(
        self,
        promoted_node_ids: list[int],
        frame_id: str,
        scene_change: bool = False,
        active_context_ids: list[int] | None = None,
    ) -> dict:
        """Run one predict→observe→compare→update cycle for a frame.

        Called after visual analysis + promotion. Returns intelligence
        stats dict (surprise score, prediction counts, etc).

        Args:
            promoted_node_ids: Nodes promoted from iconic buffer this frame.
            frame_id: Frame identifier.
            scene_change: Reset tracker on scene change.
            active_context_ids: All recently active visual node IDs (time-windowed).
                Used for structural prediction verification — "is sibling Y
                visible anywhere in the current scene?" not just "was Y promoted
                this exact frame?" If None, falls back to promoted_node_ids only.
        """
        if not config.INTELLIGENCE_LOOPS_ENABLED or not promoted_node_ids:
            return {}

        from nmem_sym_sensor.sensory_bridge import create_sensory_events, feed_events_to_drives
        from nmem_sym_sensor.sensory_prediction import (
            apply_prediction_feedback,
            generate_all_predictions,
            verify_predictions,
        )
        from nmem_sym_sensor.surprise import apply_surprise_effects, compute_frame_surprise

        if self._tracker is None:
            from nmem_sym_sensor.temporal import ObjectTracker
            self._tracker = ObjectTracker()

        if scene_change:
            self._tracker.reset()
            if self._emb_predictor:
                self._emb_predictor.reset_context()

        # 1. Build observation dicts from promoted nodes (for tracker)
        observations = await self._build_observations(promoted_node_ids)
        if not observations:
            return {}

        # 2. Track objects across frames (temporal continuity)
        tracking_result = self._tracker.update(observations)

        # 3. Build wider context for structural verification
        # Structural predictions ask "is sibling Y visible in the scene?"
        # We need to check all recently active nodes, not just this frame's promotions
        if active_context_ids:
            context_ids = list(set(active_context_ids))
            context_observations = await self._build_observations(context_ids)
        else:
            context_observations = observations

        # 4. Generate predictions (structural + temporal + symbolic)
        active_node_ids = [o["node_id"] for o in observations if o.get("node_id")]
        predictions = await generate_all_predictions(
            self.pool,
            active_node_ids,
            tracking_predictions=tracking_result.predictions,
            sym_pool=self._sym_pool,
        )

        # 5. Verify predictions against the appropriate observation set:
        # - Temporal predictions verify against this frame's observations (position-specific)
        # - Structural predictions verify against the full visual context (scene-wide)
        temporal_preds = [p for p in predictions if p.prediction_type == "temporal"]
        structural_preds = [p for p in predictions if p.prediction_type != "temporal"]

        verification_results = []
        if temporal_preds:
            verification_results.extend(
                await verify_predictions(self.pool, temporal_preds, observations)
            )
        if structural_preds:
            verification_results.extend(
                await verify_predictions(self.pool, structural_preds, context_observations)
            )

        # 5a. Embedding-space prediction (continuous surprise)
        emb_prediction = {}
        if self._emb_predictor:
            # Get the mean visual embedding of promoted nodes for this frame
            emb_rows = await self.pool.fetch(
                """
                SELECT visual_embedding::text as emb
                FROM sensory_nodes
                WHERE id = ANY($1) AND visual_embedding IS NOT NULL
                """,
                promoted_node_ids,
            )
            if emb_rows:
                vecs = []
                for r in emb_rows:
                    s = r["emb"].strip("[]")
                    if s:
                        vecs.append([float(x) for x in s.split(",")])
                if vecs:
                    frame_emb = np.mean(vecs, axis=0).astype(np.float32)
                    emb_prediction = self._emb_predictor.observe(frame_emb)

        # 5b. Compute discrete surprise (from node-level predictions)
        surprise = compute_frame_surprise(verification_results, len(tracking_result.new_objects))

        # 5c. Blend continuous and discrete surprise
        # Continuous surprise modulates the discrete signal — if the embedding
        # predictor saw this coming (low cosine_surprise), dampen overall surprise.
        # If embedding predictor is also surprised, amplify it.
        if emb_prediction.get("predicted"):
            cosine_surprise = emb_prediction["cosine_surprise"]
            # Blend: 60% discrete + 40% continuous
            blended = 0.6 * surprise.score + 0.4 * cosine_surprise
            surprise.score = round(blended, 4)

        # 6. Apply surprise effects (returns adjustment params, doesn't modify DB)
        adjustments = await apply_surprise_effects(surprise)

        # 7. Apply LTP/LTD feedback from prediction outcomes
        feedback_stats = {}
        if verification_results:
            feedback_stats = await apply_prediction_feedback(
                self.pool, verification_results, self._sym_pool,
            )

        # 8. Create sensory events and feed to nmem-sym drives
        events = create_sensory_events(
            surprise.score,
            feedback_stats,
            novel_count=len(tracking_result.new_objects),
        )
        if events:
            await feed_events_to_drives(self._sym_pool, events)

        return {
            "surprise": surprise.score,
            "predictions": len(predictions),
            "confirmed": surprise.confirmed,
            "refuted": surprise.refuted,
            "novel_objects": len(tracking_result.new_objects),
            "tracked_objects": len(self._tracker.tracked),
            "disappeared": len(tracking_result.disappeared),
            "ltp": feedback_stats.get("ltp_applied", 0),
            "ltd": feedback_stats.get("ltd_applied", 0),
            "adjustments": adjustments,
            "emb_surprise": emb_prediction.get("cosine_surprise", 0.0),
            "emb_running_error": emb_prediction.get("running_error", 0.0),
        }

    async def _build_observations(self, node_ids: list[int]) -> list[dict]:
        """Convert promoted node IDs to observation dicts for the tracker."""
        import json as _json

        rows = await self.pool.fetch(
            """
            SELECT id, label, node_type, visual_embedding::text as emb,
                   features::text as features, modality
            FROM sensory_nodes
            WHERE id = ANY($1) AND modality = 'visual' AND NOT archived
            """,
            node_ids,
        )

        observations = []
        for r in rows:
            emb = None
            if r["emb"]:
                emb_str = r["emb"].strip("[]")
                if emb_str:
                    emb = [float(x) for x in emb_str.split(",")]

            if emb is None:
                continue

            # Parse JSONB features (comes as text from explicit cast)
            raw_features = r["features"]
            if isinstance(raw_features, str):
                try:
                    features = _json.loads(raw_features)
                except (ValueError, TypeError):
                    features = {}
            elif isinstance(raw_features, dict):
                features = raw_features
            else:
                features = {}

            pos = (
                features.get("center_x", 0.0),
                features.get("center_y", 0.0),
            )

            observations.append({
                "node_id": r["id"],
                "label": r["label"],
                "node_type": r["node_type"],
                "embedding": emb,
                "position": pos,
                "size": (features.get("width", 0), features.get("height", 0)),
                "features": features,
            })

        return observations

    async def run_dreamstate(self, max_duration_s: float | None = None) -> dict:
        """Run sensory dreamstate consolidation manually (full cycle)."""
        from nmem_sym_sensor.sensory_dreamstate import run_sensory_dreamstate
        duration = max_duration_s or config.DREAMSTATE_MAX_DURATION_S
        return await run_sensory_dreamstate(
            self.pool, self._sym_pool, duration, vocal_tract=self._vocal_tract,
        )

    async def check_and_run_pressure_dreamstate(
        self, max_duration_s: float | None = None,
    ) -> dict | None:
        """Check pressure signals and run selective dreamstate if needed.

        Call this periodically (e.g. between frames or between videos).
        Only runs the specific dreamstate functions that pressure demands.
        Returns stats dict if dreamstate ran, None if no pressure.

        The dreamstate never interrupts active observation — callers
        should only invoke this when the system is idle or between
        processing units.
        """
        if self._pressure is None:
            return None

        # Sync co-occurrence volume from the store's counter
        from nmem_sym_sensor.cooccurrence import cooc_store
        delta = cooc_store.observations_since_reset
        if delta > 0:
            self._pressure.cooc_volume.observe(delta)
            cooc_store.observations_since_reset = 0

        needs = self._pressure.what_needs_running()
        if not needs:
            return None

        from nmem_sym_sensor.sensory_dreamstate import run_sensory_dreamstate
        duration = max_duration_s or config.DREAMSTATE_MAX_DURATION_S

        log.info("Pressure dreamstate triggered: %s", needs)
        stats = await run_sensory_dreamstate(
            self.pool, self._sym_pool, duration,
            only_functions=set(needs),
            vocal_tract=self._vocal_tract,
        )

        # Mark all ran functions so cooldowns apply
        for fn in stats.get("ran_functions", []):
            self._pressure.mark_ran(fn)

        # Reset surprise after consolidation (pressure released)
        if "decay" in needs or "self_play" in needs:
            self._pressure.surprise.reset()

        return stats

    @property
    def pressure_stats(self) -> dict | None:
        """Current pressure state for dashboard/monitoring."""
        if self._pressure is None:
            return None
        return self._pressure.stats()

    # ── Mental imagery ────────────────────────────────────

    async def imagine_from_sound(self, sound_unit_id: int) -> list[dict]:
        """Hear a sound → see it in the mind's eye."""
        if self._imagery is None:
            return []
        return await self._imagery.sound_to_visual(sound_unit_id)

    async def imagine_from_visual(self, visual_node_ids: list[int]) -> list[dict]:
        """See something → hear the associated sound (inner voice)."""
        if self._imagery is None:
            return []
        return await self._imagery.visual_to_sound(visual_node_ids)

    async def describe_imagination(self, **kwargs) -> str:
        """Human-readable description of current mental imagery state."""
        if self._imagery is None:
            return "Imagery not enabled"
        return await self._imagery.describe_mental_image(**kwargs)

    # ── Speech production ─────────────────────────────────

    async def speak(
        self,
        visual_node_ids: list[int] | None = None,
        sound_unit_ids: list[int] | None = None,
        save_path: str | None = None,
    ) -> dict:
        """Produce speech about a visual concept or from sound units.

        Returns dict with quality, duration, production log.
        If save_path given, saves waveform as WAV file.
        """
        if self._vocal_tract is None:
            return {"error": "Vocal tract not available"}

        waveform, log_entries = await self._vocal_tract.speak(
            visual_node_ids=visual_node_ids,
            sound_unit_ids=sound_unit_ids,
        )

        result = {
            "produced": waveform is not None,
            "production_log": log_entries,
        }

        if waveform is not None:
            result["duration_s"] = round(len(waveform) / 16000, 3)
            result["avg_quality"] = round(
                sum(e["quality"] for e in log_entries) / max(len(log_entries), 1), 4
            )
            if save_path:
                from nmem_sym_sensor.vocoder import save_wav
                save_wav(waveform, save_path)
                result["saved_to"] = save_path

        return result

    async def practice_speaking(
        self,
        sound_unit_id: int,
        iterations: int = 50,
    ) -> dict:
        """Self-play practice for a specific sound unit (babbling refinement)."""
        if self._vocal_tract is None:
            return {"error": "Vocal tract not available"}
        return await self._vocal_tract.practice(sound_unit_id, iterations)

    # ── Stats & diagnostics ──────────────────────────────

    async def stats(self) -> dict:
        """Return current sensory graph statistics."""
        return await graph_stats(self.pool)
