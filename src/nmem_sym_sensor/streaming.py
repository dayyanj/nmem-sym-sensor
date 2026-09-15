"""
Real-time streaming loop: connects sensor bus to SensorGraph.

Reads observations from any combination of sensors and feeds them
through the existing analysis → buffer → promote → bind → consolidate
pipeline. Replaces the batch-oriented ingest_video() for live input
while reusing all existing machinery.

Background tasks handle periodic consolidation and statistical grounding.
STT runs in a thread pool to avoid blocking the event loop.
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from nmem_sym_sensor import config
from nmem_sym_sensor.ingest.text_bridge import (
    SpeechAccumulator,
    create_text_callback,
    ground_from_statistics,
)
from nmem_sym_sensor.sensor_bus import Modality, Observation, SensorBus

log = logging.getLogger(__name__)


class SttWorker:
    """Runs Whisper STT in a background thread.

    Accumulates audio windows into a buffer. When the buffer reaches
    the target duration, submits it to Whisper in a ThreadPoolExecutor.
    Results are published back as TEXT observations via a callback.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        buffer_duration_s: float = 5.0,
        sample_rate: int | None = None,
    ):
        self._model_size = model_size
        self._buffer_duration = buffer_duration_s
        self._sr = sample_rate or config.AUDIO_SAMPLE_RATE
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._audio_chunks: list[np.ndarray] = []
        self._accumulated_s = 0.0
        self._on_transcript = None  # async callback for results

    def set_callback(self, callback):
        """Set the async callback for transcript results.

        callback(word: str, start: float, confidence: float)
        """
        self._on_transcript = callback

    def submit_audio(self, waveform: np.ndarray, sample_rate: int):
        """Add audio to the transcription buffer.

        When enough audio has accumulated, submits to Whisper.
        """
        self._audio_chunks.append(waveform)
        self._accumulated_s += len(waveform) / sample_rate

        if self._accumulated_s >= self._buffer_duration:
            full = np.concatenate(self._audio_chunks)
            self._audio_chunks.clear()
            self._accumulated_s = 0.0

            # Submit to thread pool
            loop = asyncio.get_event_loop()
            loop.run_in_executor(self._executor, self._transcribe, full, sample_rate)

    def _transcribe(self, waveform: np.ndarray, sr: int):
        """Runs in background thread. Blocking Whisper call."""
        try:
            import whisper
        except ImportError:
            log.warning("openai-whisper not installed, STT disabled")
            return

        if self._model is None:
            log.info("Loading Whisper %s model (CPU)...", self._model_size)
            self._model = whisper.load_model(self._model_size, device="cpu")

        try:
            result = self._model.transcribe(
                waveform.astype(np.float32),
                language=None,
                word_timestamps=True,
                verbose=False,
            )
        except Exception as e:
            log.warning("Whisper transcription failed: %s", e)
            return

        # Extract words and schedule callback on event loop
        if self._on_transcript:
            loop = asyncio.get_event_loop()
            for seg in result.get("segments", []):
                for word_info in seg.get("words", []):
                    word = word_info.get("word", "").strip()
                    if not word:
                        continue
                    start = word_info.get("start", 0)
                    confidence = word_info.get("probability", 1.0)

                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        self._on_transcript(word, start, confidence),
                    )

    def shutdown(self):
        """Clean up the thread pool."""
        self._executor.shutdown(wait=False)


class StreamingLoop:
    """Real-time perception loop that reads from a SensorBus.

    Connects any combination of sensors to the SensorGraph pipeline.
    Handles visual, audio, and text observations, runs periodic
    consolidation and grounding, and maintains the visual context
    window for speech grounding.

    Usage::

        async with SensorGraph(db_dsn="...") as sg:
            bus = SensorBus()
            bus.register(CameraSensor(source=0))
            bus.register(MicrophoneSensor())

            loop = StreamingLoop(sg, bus, stt_enabled=True)
            await loop.run()  # runs until Ctrl+C or bus closes
    """

    def __init__(
        self,
        sensor_graph,
        bus: SensorBus,
        consolidation_interval_s: float = 30.0,
        grounding_interval_s: float = 120.0,
        stt_enabled: bool = False,
        stt_model: str = "tiny",
        recognition_engine=None,
        nmem_write_fn=None,
    ):
        self._sg = sensor_graph
        self._bus = bus
        self._consolidation_interval = consolidation_interval_s
        self._grounding_interval = grounding_interval_s
        self._stt_enabled = stt_enabled
        self._stt_model = stt_model
        self._recognition_engine = recognition_engine

        # Visual context window (same pattern as ingest_video)
        self._recent_visual: list[tuple[int, float]] = []
        self.VISUAL_CONTEXT_WINDOW_S = 5.0

        # Text bridge
        self._accumulator = SpeechAccumulator()
        self._text_callback = create_text_callback(
            nmem_write_fn=nmem_write_fn,
            sensor_graph=sensor_graph,
            accumulator=self._accumulator,
        )

        # STT worker
        self._stt_worker: SttWorker | None = None

        # Stats
        self._frames_processed = 0
        self._audio_windows_processed = 0
        self._text_events_processed = 0
        self._start_time: float | None = None

    async def run(self):
        """Main loop. Runs until the bus closes or KeyboardInterrupt."""
        self._start_time = time.monotonic()

        # Start background tasks
        tasks = [
            asyncio.create_task(self._consolidation_timer(), name="consolidation"),
            asyncio.create_task(self._grounding_timer(), name="grounding"),
        ]

        # Initialize STT worker
        if self._stt_enabled:
            self._stt_worker = SttWorker(
                model_size=self._stt_model,
                sample_rate=config.AUDIO_SAMPLE_RATE,
            )
            self._stt_worker.set_callback(self._handle_stt_word)

        log.info("Streaming loop started (consolidation=%ds, grounding=%ds, stt=%s)",
                self._consolidation_interval, self._grounding_interval,
                self._stt_model if self._stt_enabled else "off")

        try:
            async for obs in self._bus.stream():
                try:
                    if obs.modality == Modality.VISUAL:
                        await self._handle_visual(obs)
                    elif obs.modality == Modality.AUDIO:
                        await self._handle_audio(obs)
                    elif obs.modality == Modality.TEXT:
                        await self._handle_text(obs)
                except Exception as e:
                    log.warning("Error handling %s observation: %s",
                               obs.modality.value, e)
        except asyncio.CancelledError:
            pass
        finally:
            for task in tasks:
                task.cancel()
            if self._stt_worker:
                self._stt_worker.shutdown()

            elapsed = time.monotonic() - self._start_time
            log.info(
                "Streaming loop stopped after %.1fs: %d frames, %d audio, %d text",
                elapsed, self._frames_processed,
                self._audio_windows_processed, self._text_events_processed,
            )

    async def _handle_visual(self, obs: Observation):
        """Process a visual frame observation."""
        frame_id = obs.metadata.get("frame_id", f"f_{self._frames_processed}")
        scene_change = obs.metadata.get("scene_change", False)

        # Ingest through SensorGraph (uses foveal attention if enabled)
        _, _ = await self._sg.ingest_frame(
            obs.data, frame_id=frame_id, scene_change=scene_change,
        )

        # Promote and bind
        promoted = await self._sg.promote_iconic()
        if promoted:
            rows = await self._sg.pool.fetch(
                "SELECT id, modality FROM sensory_nodes WHERE id = ANY($1)",
                promoted,
            )
            promoted_visual = [r["id"] for r in rows if r["modality"] == "visual"]
            promoted_audio = [r["id"] for r in rows if r["modality"] == "audio"]

            # Update visual context window
            now = time.monotonic()
            for vid in promoted_visual:
                self._recent_visual.append((vid, now))
            self._recent_visual = [
                (vid, ts) for vid, ts in self._recent_visual
                if now - ts <= self.VISUAL_CONTEXT_WINDOW_S
            ]

            # Cross-modal binding
            if promoted_visual and promoted_audio:
                try:
                    await self._sg.bind(promoted_visual, promoted_audio)
                except Exception:
                    pass

            # Intra-modal binding
            if len(promoted_visual) > 1:
                try:
                    await self._sg.bind_intra(promoted_visual)
                except Exception:
                    pass

            # Recognition
            if self._recognition_engine and promoted:
                try:
                    await self._recognition_engine.recognize(promoted)
                except Exception:
                    pass

        self._frames_processed += 1

    async def _handle_audio(self, obs: Observation):
        """Process an audio window observation."""
        sr = obs.metadata.get("sample_rate", config.AUDIO_SAMPLE_RATE)
        window_id = obs.metadata.get("window_id", f"a_{self._audio_windows_processed}")

        _, _, _ = await self._sg.ingest_audio(obs.data, sample_rate=sr, window_id=window_id)

        # Forward to STT worker if enabled
        if self._stt_worker:
            self._stt_worker.submit_audio(obs.data, sr)

        self._audio_windows_processed += 1

    async def _handle_text(self, obs: Observation):
        """Process a text/transcript observation."""
        if self._text_callback:
            current_visual = [vid for vid, _ in self._recent_visual]
            await self._text_callback(
                obs.data,
                obs.metadata.get("start", obs.timestamp),
                obs.metadata.get("confidence", 1.0),
                active_visual_ids=current_visual,
            )
        self._text_events_processed += 1

    async def _handle_stt_word(self, word: str, start: float, confidence: float):
        """Callback from SttWorker when a word is transcribed."""
        current_visual = [vid for vid, _ in self._recent_visual]
        if self._text_callback:
            await self._text_callback(
                word, start, confidence,
                active_visual_ids=current_visual,
            )
        self._text_events_processed += 1

    async def _consolidation_timer(self):
        """Periodic consolidation background task."""
        while True:
            await asyncio.sleep(self._consolidation_interval)
            try:
                stats = await self._sg.consolidate()
                log.info("Periodic consolidation: clusters=%s, split=%s",
                        stats.get("clusters_promoted_stable", 0),
                        stats.get("clusters_split", 0))
            except Exception as e:
                log.warning("Consolidation failed: %s", e)

    async def _grounding_timer(self):
        """Periodic statistical grounding background task."""
        while True:
            await asyncio.sleep(self._grounding_interval)
            try:
                groundings = await ground_from_statistics(self._sg.pool)
                new = [g for g in groundings if g["action"] == "grounded"]
                if new:
                    for g in new:
                        log.info("GROUNDED: '%s' → %s:%s",
                                g["word"], g["node_type"], g["node"])
            except Exception as e:
                log.warning("Grounding failed: %s", e)

    @property
    def stats(self) -> dict:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "elapsed_s": round(elapsed, 1),
            "frames": self._frames_processed,
            "audio_windows": self._audio_windows_processed,
            "text_events": self._text_events_processed,
            "visual_context_size": len(self._recent_visual),
            "fps": round(self._frames_processed / max(elapsed, 1), 2),
        }
