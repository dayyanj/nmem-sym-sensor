"""
Video ingestion: video file → temporally-aligned sensory observations.

Extracts frames, audio windows, and (optionally) speech transcripts from
a video file. Feeds everything into the SensorGraph with correct temporal
alignment so that the binding system connects what's seen with what's
heard and what's said.

Dependencies:
  - cv2 (opencv-python-headless): frame extraction
  - numpy: always required
  - ffmpeg (system binary): audio extraction from video
"""
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nmem_sym_sensor import config

log = logging.getLogger(__name__)

# Module-level Whisper model cache — loaded once, reused across videos
_whisper_model_cache: dict = {}


@dataclass
class VideoConfig:
    """Configuration for video ingestion."""
    # Frame extraction
    fps: float = 2.0                   # frames per second to extract (1-5 is sensible)
    max_frames: int | None = None      # cap total frames (None = all)
    scene_change_threshold: float = 30.0  # scene change detection sensitivity (lower = more sensitive)
    skip_similar_frames: bool = True   # skip frames too similar to previous

    # Audio windows
    audio_window_s: float = 0.5        # audio analysis window in seconds
    audio_hop_s: float = 0.25          # hop between audio windows

    # STT
    stt_model: str = "base"            # whisper model size: tiny | base | small | medium
    stt_language: str | None = None    # force language (None = auto-detect)

    # Processing
    batch_size: int = 10               # frames to process before yielding
    consolidate_every: int = 100       # run consolidation every N frames


@dataclass
class TranscriptSegment:
    """A timestamped word or phrase from STT."""
    text: str
    start: float                       # seconds from video start
    end: float                         # seconds from video start
    confidence: float = 1.0


@dataclass
class VideoFrame:
    """A single extracted video frame with metadata."""
    image: np.ndarray                  # HWC uint8 BGR
    timestamp: float                   # seconds from video start
    frame_idx: int
    is_scene_change: bool = False


@dataclass
class IngestProgress:
    """Progress tracking for video ingestion."""
    video_path: str
    total_frames: int = 0
    processed_frames: int = 0
    total_audio_windows: int = 0
    processed_audio_windows: int = 0
    transcript_segments: int = 0
    visual_nodes_created: int = 0
    audio_nodes_created: int = 0
    bindings_created: int = 0
    groundings_from_speech: int = 0
    intelligence_cycles: int = 0
    total_surprise: float = 0.0
    predictions_confirmed: int = 0
    predictions_refuted: int = 0
    errors: list[str] = field(default_factory=list)


# ── Frame extraction ─────────────────────────────────────

def extract_frames(
    video_path: str | Path,
    cfg: VideoConfig | None = None,
) -> list[VideoFrame]:
    """Extract frames from a video file at the configured FPS.

    Uses OpenCV for frame extraction. Optionally detects scene changes
    and skips near-duplicate frames to reduce noise.

    Args:
        video_path: Path to video file.
        cfg: Video configuration.

    Returns:
        List of VideoFrame objects.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python-headless required: pip install opencv-python-headless")

    cfg = cfg or VideoConfig()
    path = str(video_path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        log.warning("Could not determine FPS, assuming 30")
        native_fps = 30.0

    # Calculate frame interval
    frame_interval = max(1, int(native_fps / cfg.fps))

    frames = []
    prev_gray = None
    frame_idx = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / native_fps

        # Scene change detection
        is_scene_change = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and cfg.skip_similar_frames:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(diff))

            if mean_diff < 5.0:
                # Frame too similar to previous — skip
                frame_idx += 1
                continue

            if mean_diff > cfg.scene_change_threshold:
                is_scene_change = True

        prev_gray = gray

        frames.append(VideoFrame(
            image=frame,
            timestamp=timestamp,
            frame_idx=extracted,
            is_scene_change=is_scene_change,
        ))
        extracted += 1

        if cfg.max_frames and extracted >= cfg.max_frames:
            break

        frame_idx += 1

    cap.release()
    log.info("Extracted %d frames from %s (native: %d frames, %.1f fps)",
            len(frames), path, total_native_frames, native_fps)

    return frames


# ── Audio extraction ─────────────────────────────────────

def extract_audio(
    video_path: str | Path,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Extract audio track from a video file as a numpy array.

    Uses ffmpeg to extract and convert to mono WAV, then loads
    with numpy. Falls back to scipy.io.wavfile if available.

    Args:
        video_path: Path to video file.
        sample_rate: Target sample rate (default: config.AUDIO_SAMPLE_RATE).

    Returns:
        (waveform, sample_rate) tuple. Waveform is float32 mono.
    """
    sr = sample_rate or config.AUDIO_SAMPLE_RATE

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp_path = tmp.name

    # Extract audio with ffmpeg
    # -acodec pcm_s16le forces raw PCM output regardless of input codec
    # (handles Opus in webm, AAC in mp4, etc.)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn",                          # no video
        "-ac", "1",                     # mono
        "-ar", str(sr),                 # target sample rate
        "-acodec", "pcm_s16le",         # force PCM output
        "-f", "wav",                    # WAV container
        "-y",                           # overwrite
        tmp_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: apt install ffmpeg")

    # Load WAV
    try:
        import soundfile as sf
        waveform, actual_sr = sf.read(tmp_path, dtype="float32")
    except ImportError:
        from scipy.io import wavfile
        actual_sr, waveform = wavfile.read(tmp_path)
        if waveform.dtype == np.int16:
            waveform = waveform.astype(np.float32) / 32768.0
        elif waveform.dtype == np.int32:
            waveform = waveform.astype(np.float32) / 2147483648.0

    # Clean up
    Path(tmp_path).unlink(missing_ok=True)

    # Mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    log.info("Extracted audio: %.1fs, %d Hz", len(waveform) / actual_sr, actual_sr)
    return waveform.astype(np.float32), actual_sr


def window_audio(
    waveform: np.ndarray,
    sample_rate: int,
    window_s: float = 0.5,
    hop_s: float = 0.25,
) -> list[tuple[np.ndarray, float, float]]:
    """Split a waveform into overlapping windows.

    Returns list of (window_samples, start_time, end_time).
    """
    window_samples = int(window_s * sample_rate)
    hop_samples = int(hop_s * sample_rate)
    windows = []

    pos = 0
    while pos + window_samples <= len(waveform):
        start_t = pos / sample_rate
        end_t = (pos + window_samples) / sample_rate
        windows.append((waveform[pos:pos + window_samples], start_t, end_t))
        pos += hop_samples

    return windows


# ── Speech-to-text ───────────────────────────────────────

def run_stt(
    video_path: str | Path,
    model_size: str = "base",
    language: str | None = None,
) -> list[TranscriptSegment]:
    """Run speech-to-text on a video's audio track.

    Uses OpenAI Whisper for transcription with word-level timestamps.

    Args:
        video_path: Path to video file.
        model_size: Whisper model size (tiny/base/small/medium).
        language: Force language code (e.g., "en"). None = auto-detect.

    Returns:
        List of TranscriptSegments with timestamps.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper required for STT: pip install openai-whisper"
        )

    # Use cached model if available, keyed by model size
    model = _whisper_model_cache.get(model_size)
    if model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading Whisper %s model on %s...", model_size, device)
        model = whisper.load_model(model_size, device=device)
        _whisper_model_cache[model_size] = model

    log.info("Transcribing %s...", video_path)
    result = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        verbose=False,
        condition_on_previous_text=False,
    )

    min_confidence = float(os.environ.get("NMEM_SENSOR_STT_MIN_CONFIDENCE", "0.5"))
    segments = []
    for seg in result.get("segments", []):
        # Whisper provides word-level timestamps
        for word_info in seg.get("words", []):
            text = word_info.get("word", "").strip()
            if not text:
                continue
            conf = word_info.get("probability", 1.0)
            if conf < min_confidence:
                continue  # drop low-confidence hallucinations
            segments.append(TranscriptSegment(
                text=text,
                start=word_info.get("start", seg["start"]),
                end=word_info.get("end", seg["end"]),
                confidence=conf,
            ))

        # Fallback: if no word-level timestamps, use segment-level
        if not seg.get("words"):
            text = seg.get("text", "").strip()
            if text:
                segments.append(TranscriptSegment(
                    text=text,
                    start=seg["start"],
                    end=seg["end"],
                ))

    log.info("STT produced %d word segments", len(segments))
    return segments


# ── Main ingestion pipeline ──────────────────────────────

async def ingest_video(
    sensor_graph,
    video_path: str | Path,
    cfg: VideoConfig | None = None,
    text_callback=None,
    progress_callback=None,
    recognition_engine=None,
) -> IngestProgress:
    """Ingest a complete video into the sensory graph.

    The full pipeline:
      1. Extract frames at configured FPS
      2. Extract audio and split into windows
      3. Run STT for speech transcription
      4. For each time slice:
         a. Analyze visual frame → buffer observations
         b. Analyze audio window → buffer observations
         c. Bind visual + audio nodes (temporal co-occurrence)
         d. If STT word at this timestamp → call text_callback
      5. Run consolidation periodically

    Args:
        sensor_graph: Connected SensorGraph instance.
        video_path: Path to video file.
        cfg: Video ingestion configuration.
        text_callback: Async function(text, timestamp) called for each
            STT word/phrase. This is where you bridge to nmem — send the
            text to nmem's LTM with the timestamp so that text concepts
            activate when the corresponding sensory observations fire.
            Example::

                async def on_speech(text: str, timestamp: float):
                    await nmem.write_ltm(
                        content=text,
                        metadata={"source": "stt", "timestamp": timestamp},
                    )

        progress_callback: Async function(IngestProgress) called periodically.

    Returns:
        Final IngestProgress with statistics.
    """
    cfg = cfg or VideoConfig()
    path = Path(video_path)
    progress = IngestProgress(video_path=str(path))

    log.info("Starting video ingestion: %s", path)

    # ── Step 1: Extract frames ───────────────────────────
    try:
        frames = extract_frames(path, cfg)
        # Still-image videos: if only 1 unique frame was extracted from a
        # multi-second video, repeat it at the target fps so the iconic buffer
        # gets enough observations to promote. A 7-second still image should
        # create ~14 observations at 2fps, not just 1.
        if len(frames) == 1:
            import subprocess as _sp
            duration_result = _sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True,
            )
            try:
                duration = float(duration_result.stdout.strip())
            except (ValueError, AttributeError):
                duration = 0.0
            expected_frames = int(duration * cfg.fps)
            if expected_frames > 1:
                base = frames[0]
                for f_idx in range(1, expected_frames):
                    frames.append(VideoFrame(
                        image=base.image,
                        timestamp=f_idx / cfg.fps,
                        frame_idx=f_idx,
                        is_scene_change=False,
                    ))
                log.info("Still-image video: replicated 1 frame → %d observations (%.1fs at %.0ffps)",
                         len(frames), duration, cfg.fps)
        progress.total_frames = len(frames)
    except Exception as e:
        progress.errors.append(f"Frame extraction failed: {e}")
        log.error("Frame extraction failed: %s", e)
        return progress

    # ── Step 2: Extract audio ────────────────────────────
    waveform = None
    audio_windows = []
    try:
        waveform, sr = extract_audio(path)
        audio_windows = window_audio(waveform, sr, cfg.audio_window_s, cfg.audio_hop_s)
        progress.total_audio_windows = len(audio_windows)
    except Exception as e:
        progress.errors.append(f"Audio extraction failed: {e}")
        log.warning("Audio extraction failed (continuing visual-only): %s", e)

    # ── Step 3: Run STT ──────────────────────────────────
    transcript: list[TranscriptSegment] = []
    try:
        transcript = run_stt(path, cfg.stt_model, cfg.stt_language)
        progress.transcript_segments = len(transcript)
    except Exception as e:
        progress.errors.append(f"STT failed: {e}")
        log.warning("STT failed (continuing without speech): %s", e)

    # Build a time-indexed transcript lookup
    # For each second, collect all words spoken in that window
    transcript_by_second: dict[int, list[TranscriptSegment]] = {}
    for seg in transcript:
        second = int(seg.start)
        transcript_by_second.setdefault(second, []).append(seg)

    # ── Step 3b: TABULA2 full-audio processing ────────────
    # Process entire audio in one shot through TABULA2 disentangler.
    # Pre-computes syllable bursts indexed by time for matching during
    # frame loop. No windowing — avoids boundary artifacts.
    tabula2_bursts: list = []  # (start_time, SyllableBurst)
    tabula2_noise_events: list = []  # (time_s, embedding_512)
    if sensor_graph._tabula2 is not None and waveform is not None:
        try:
            import numpy as _np

            from nmem_sym_sensor.syllable_segmenter import segment_syllables as _seg_syl

            t2_output = sensor_graph._tabula2.process(waveform, config.AUDIO_SAMPLE_RATE)
            bursts = _seg_syl(
                t2_output.voice_embeddings,
                t2_output.vad_mask,
                t2_output.vad_features,
                min_frames=config.TABULA2_MIN_SYLLABLE_FRAMES,
                tail_pad_frames=10,
                sub_features=t2_output.sub_features,
            )
            # Index bursts by time (seconds)
            for b in bursts:
                t_start = b.start_frame / t2_output.frame_rate
                tabula2_bursts.append((t_start, b))

            # Noise stream: novelty-gated environmental sounds
            # Only observe when the acoustic environment CHANGES.
            # Constant music → 1 observation. Music stops, birds start → new observation.
            # This is auditory habituation — the brain tunes out constant sounds.
            noise_embs = t2_output.noise_embeddings  # (T, 512)
            if len(noise_embs) > 40:
                window = 20  # ~100ms analysis window
                prev_mean = noise_embs[:window].mean(axis=0)
                prev_norm = _np.linalg.norm(prev_mean)
                if prev_norm > 0:
                    prev_mean /= prev_norm
                seg_start = 0
                novelty_threshold = 0.85

                for f in range(window, len(noise_embs) - window, window):
                    curr_mean = noise_embs[f:f + window].mean(axis=0)
                    curr_norm = _np.linalg.norm(curr_mean)
                    if curr_norm > 0:
                        curr_mean /= curr_norm
                    sim = float(_np.dot(prev_mean, curr_mean))

                    if sim < novelty_threshold:
                        # Environment changed — record previous segment
                        seg_emb = noise_embs[seg_start:f].mean(axis=0)
                        seg_norm = _np.linalg.norm(seg_emb)
                        if seg_norm > 0:
                            t_s = seg_start / t2_output.frame_rate
                            tabula2_noise_events.append((t_s, seg_emb / seg_norm))
                        seg_start = f
                    prev_mean = curr_mean

                # Final segment
                seg_emb = noise_embs[seg_start:].mean(axis=0)
                seg_norm = _np.linalg.norm(seg_emb)
                if seg_norm > 0:
                    t_s = seg_start / t2_output.frame_rate
                    tabula2_noise_events.append((t_s, seg_emb / seg_norm))

                if tabula2_noise_events:
                    log.info("TABULA2: %d environmental sound changes", len(tabula2_noise_events))

            log.info("TABULA2: %d syllable bursts from %.1fs audio",
                     len(bursts), len(waveform) / config.AUDIO_SAMPLE_RATE)

            # Free large TABULA2 arrays — bursts already hold their own copies
            del t2_output, noise_embs, bursts
        except Exception as e:
            log.warning("TABULA2 full-audio processing failed: %s", e)

    # Free waveform and audio windows when TABULA2 handles all audio
    if sensor_graph._tabula2 is not None:
        del waveform
        audio_windows = []
        import gc
        gc.collect()

    # ── Step 4: Process time slices ──────────────────────
    # Budget allocator — thalamic gate distributes compute across sensors
    from nmem_sym_sensor.budget_allocator import (
        LEVEL_MINIMAL,
        LEVEL_REDUCED,
        LEVEL_STANDARD,
        BudgetAllocator,
        SensorBid,
    )
    _budget_allocator = BudgetAllocator()
    _speech_surprise = 0.0  # fed from speech predictor each frame
    _frames_suppressed = 0  # visual frames below LEVEL_MINIMAL
    _tabula2_burst_idx = 0  # index pointer for consumed bursts (avoids O(N) scan)
    _tabula2_noise_idx = 0  # index pointer for consumed noise events

    # Map audio windows by timestamp for lookup
    audio_window_idx = 0

    # Time-windowed recently-promoted visual node IDs.
    # Each entry is (node_id, video_timestamp). Entries older than
    # VISUAL_CONTEXT_WINDOW_S are pruned. This ensures that when
    # "blue" is spoken at 24s, only nodes from ~22-24s are active,
    # not orange nodes from 10s ago.
    VISUAL_CONTEXT_WINDOW_S = 5.0  # seconds of video time to keep (backward: image before word)
    WORD_FORWARD_WINDOW_S = 3.0   # seconds to keep pending words (forward: word before image)
    FIXATION_ACCUMULATE_S = 2.0   # seconds of fixations to accumulate for attend hull
    recent_visual_entries: list[tuple[int, float, bool]] = []  # (node_id, timestamp, attended)
    recent_fixations: list[tuple[float, float, float, float]] = []  # (x, y, saliency, timestamp)
    pending_words: list[tuple[str, float, float, list[int]]] = []  # (text, start, confidence, visual_ids_at_time)

    # Sound language tracking (sound unit sequences across frames)
    prev_sound_unit_id: int | None = None
    prev_sound_timestamp: float = 0.0

    for i, frame in enumerate(frames):
        t = frame.timestamp
        frame_id = f"{path.stem}_f{frame.frame_idx}_{t:.2f}s"

        # ── Thalamic gate: allocate compute budget ──
        is_scene_change = frame.is_scene_change or (i == 0)
        if is_scene_change:
            _budget_allocator.reset_visual()

        v_novelty = _budget_allocator.visual_novelty(frame.image)
        a_novelty = _budget_allocator.auditory_novelty(
            has_speech=bool(tabula2_bursts),
            speech_surprise=_speech_surprise,
        )
        i_pressure = _budget_allocator.internal_pressure(
            sensor_graph._pressure if hasattr(sensor_graph, '_pressure') else None,
        )
        frame_budget = _budget_allocator.allocate([
            SensorBid("visual", v_novelty),
            SensorBid("auditory", a_novelty),
            SensorBid("internal", i_pressure),
        ])

        # 4a. Visual analysis — depth controlled by budget
        frame_analysis = None
        visual_ids = []
        if frame_budget.above("visual", LEVEL_MINIMAL):
            # Full processing: edge analysis + foveal saccades + buffer observation
            try:
                visual_ids, frame_analysis = await sensor_graph.ingest_frame(
                    frame.image,
                    frame_id=frame_id,
                    backend=None,
                    scene_change=is_scene_change,
                )
                progress.visual_nodes_created += len(visual_ids)
            except Exception as e:
                progress.errors.append(f"Visual analysis failed at {t:.1f}s: {e}")
        elif frame_budget.above("visual", 5):
            # Reduced: edge analysis + buffer observation only (no foveal, no intelligence)
            # Still discovers new primitives in familiar scenes
            try:
                visual_ids, frame_analysis = await sensor_graph.ingest_frame(
                    frame.image,
                    frame_id=frame_id,
                    backend=None,
                    scene_change=is_scene_change,
                    depth="minimal",
                )
                progress.visual_nodes_created += len(visual_ids)
            except Exception as e:
                progress.errors.append(f"Visual analysis (reduced): {e}")
        else:
            _frames_suppressed += 1
            # True suppression: carry forward previous visual neurons
            recent_visual_entries = [(nid, t, att) for nid, ts, att in recent_visual_entries
                                     if t - ts < VISUAL_CONTEXT_WINDOW_S]

        # 4a-bis. Oculomotor: capture saccade trajectory as motor memory
        # (deferred until after audio analysis so we have sound unit IDs)

        # 4b. Audio analysis (find overlapping audio window)
        audio_ids = []
        formant_emb = None
        audio_has_onset = False
        # Legacy windowed audio analysis — only when TABULA2 is NOT enabled.
        # When TABULA2 is active, voice/noise are processed in Step 3b (full-audio)
        # and handled in Step 4d-bis. Skip windowed analysis to avoid:
        # 1. Duplicate processing (TABULA2 already disentangled voice/noise)
        # 2. Memory bloat (audio_windows holds waveform slices)
        # 3. Legacy timbre/frequency/rhythm nodes that dominate clustering
        if not sensor_graph._tabula2:
            while audio_window_idx < len(audio_windows):
                window_samples, win_start, win_end = audio_windows[audio_window_idx]

                if win_end < t:
                    audio_window_idx += 1
                    continue
                elif win_start > t + (1.0 / cfg.fps):
                    break
                else:
                    window_id = f"{path.stem}_a{audio_window_idx}_{win_start:.2f}s"
                    try:
                        aids, sound_data, audio_has_onset = await sensor_graph.ingest_audio(
                            window_samples,
                            sample_rate=config.AUDIO_SAMPLE_RATE,
                            window_id=window_id,
                        )
                        noise_emb = None
                        if isinstance(sound_data, tuple):
                            formant_emb, noise_emb = sound_data
                        else:
                            formant_emb = sound_data
                        audio_ids.extend(aids)
                        progress.audio_nodes_created += len(aids)
                        progress.processed_audio_windows += 1
                    except Exception as e:
                        progress.errors.append(f"Audio analysis failed at {win_start:.1f}s: {e}")

                    audio_window_idx += 1
                    break

        # 4b-bis. Sound language: DEFERRED to after promotion (step 4d-bis)
        # so that recent_visual_entries is populated with this frame's nodes.

        # 4b-ter. Oculomotor: capture saccade trajectory with sound symbol context
        if (sensor_graph._saccade_memory
                and frame_analysis
                and frame_analysis.fixations):
            try:
                current_visual = list(set(
                    vid for vid, ts, att in recent_visual_entries if att
                ))
                # Sound symbol context: recent sound unit IDs
                current_sound_ids = []
                if prev_sound_unit_id is not None:
                    current_sound_ids.append(prev_sound_unit_id)

                await sensor_graph._saccade_memory.observe_trajectory(
                    fixations=frame_analysis.fixations,
                    timestamp=t,
                    active_visual_ids=current_visual,
                    active_sound_unit_ids=current_sound_ids or None,
                )
            except Exception as e:
                log.debug("Saccade memory failed at %.1fs: %s", t, e)

        # 4c. Promote iconic → nodes, then bind
        # ingest_frame/ingest_audio return iconic buffer IDs, not node IDs.
        # We need to promote to get real node IDs for binding.
        try:
            promoted_node_ids = await sensor_graph.promote_iconic()
        except Exception:
            promoted_node_ids = []

        promoted_visual = []
        promoted_audio = []
        if promoted_node_ids:
            # Separate promoted nodes by modality for binding
            promoted_visual = []
            promoted_audio = []
            try:
                rows = await sensor_graph.pool.fetch(
                    """
                    SELECT id, modality FROM sensory_nodes
                    WHERE id = ANY($1)
                    """,
                    promoted_node_ids,
                )
                for r in rows:
                    if r["modality"] == "visual":
                        promoted_visual.append(r["id"])
                    else:
                        promoted_audio.append(r["id"])
            except Exception:
                pass

            # Record co-activation edges between visual neurons from same frame.
            # Only between different node_types (shape↔color, shape↔texture, etc.)
            # to avoid connecting alternative interpretations of the same region.
            if len(promoted_visual) >= 2:
                try:
                    type_map = await sensor_graph.pool.fetch(
                        "SELECT id, node_type FROM sensory_nodes WHERE id = ANY($1)",
                        promoted_visual,
                    )
                    by_id = {r["id"]: r["node_type"] for r in type_map}
                    from nmem_sym_sensor.graph import upsert_edge
                    for vi in range(len(promoted_visual)):
                        for vj in range(vi + 1, len(promoted_visual)):
                            a, b = promoted_visual[vi], promoted_visual[vj]
                            if by_id.get(a) != by_id.get(b):  # cross-type only
                                await upsert_edge(
                                    sensor_graph.pool, a, b,
                                    "co_activated", confidence=0.5,
                                )
                except Exception:
                    pass

            # Update time-windowed visual context with attention gating.
            # Only visual nodes inside the foveal attend region bind to sound.
            # The attend region is the convex hull of accumulated fixation
            # points across recent frames (not just this frame), giving a
            # dynamic shape that traces whatever the fovea explored.
            if promoted_visual:
                fixations = frame_analysis.fixations if frame_analysis else None

                # Accumulate fixations across frames
                if fixations:
                    for fx_pt, fy_pt, sal_pt in fixations:
                        recent_fixations.append((fx_pt, fy_pt, sal_pt, t))
                # Prune old fixations
                recent_fixations = [
                    (fx_pt, fy_pt, s, ts) for fx_pt, fy_pt, s, ts in recent_fixations
                    if t - ts <= FIXATION_ACCUMULATE_S
                ]

                # Build attend region from accumulated fixation convex hull
                attend_hull = None
                if len(recent_fixations) >= 3:
                    import cv2 as _cv2
                    pts = np.array(
                        [(int(fx_pt), int(fy_pt)) for fx_pt, fy_pt, _, _ in recent_fixations],
                        dtype=np.int32,
                    )
                    try:
                        attend_hull = _cv2.convexHull(pts)
                    except Exception:
                        attend_hull = None

                # Fallback: if < 3 accumulated fixations, use radius from each point
                attend_radius = 120  # fallback radius
                hull_pad = 40  # pixels of padding around hull

                for vid in promoted_visual:
                    attended = True  # default if no fovea
                    if fixations:
                        try:
                            row = await sensor_graph.pool.fetchrow(
                                "SELECT features::text as feat FROM sensory_nodes WHERE id = $1",
                                vid,
                            )
                            if row and row["feat"]:
                                import json as _json
                                feat = _json.loads(row["feat"])
                                nx = feat.get("center_x", feat.get("bbox_x", -1))
                                ny = feat.get("center_y", feat.get("bbox_y", -1))
                                if nx >= 0 and ny >= 0:
                                    if attend_hull is not None:
                                        # Point-in-hull test with padding
                                        import cv2 as _cv2
                                        dist = _cv2.pointPolygonTest(
                                            attend_hull,
                                            (float(nx), float(ny)),
                                            True,  # return signed distance
                                        )
                                        # dist > 0 = inside, dist < 0 = outside
                                        # Allow hull_pad pixels outside
                                        attended = dist >= -hull_pad
                                    else:
                                        # < 3 fixations: fallback to radius check
                                        attended = False
                                        for fx, fy, _ in fixations:
                                            d = ((nx - fx) ** 2 + (ny - fy) ** 2) ** 0.5
                                            if d < attend_radius:
                                                attended = True
                                                break
                        except Exception:
                            attended = True  # on error, include it

                    recent_visual_entries.append((vid, t, attended))

            # Prune entries older than the context window
            recent_visual_entries = [
                (vid, ts, att) for vid, ts, att in recent_visual_entries
                if t - ts <= VISUAL_CONTEXT_WINDOW_S
            ]

            # Cross-modal binding
            if promoted_visual and promoted_audio:
                try:
                    new_edges = await sensor_graph.bind(promoted_visual, promoted_audio)
                    progress.bindings_created += len(new_edges)
                except Exception as e:
                    progress.errors.append(f"Binding failed at {t:.1f}s: {e}")

            # Intra-modal binding
            if len(promoted_visual) > 1:
                try:
                    await sensor_graph.bind_intra(promoted_visual)
                except Exception:
                    pass

        # 4d-bis. Sound language: observe syllable bursts near this frame
        # Runs AFTER promotion so recent_visual_entries has this frame's nodes.
        if sensor_graph._sound_language:
            try:
                # Visual context for sound-visual binding — ATTENTION GATED
                current_visual = list(set(
                    vid for vid, ts, att in recent_visual_entries if att
                ))

                # Find STT word at this timestamp (diagnostic label only)
                stt_word = None
                stt_conf = 0.0
                sec = int(t)
                if sec in transcript_by_second:
                    for seg in transcript_by_second[sec]:
                        if abs(seg.start - t) < 1.0:
                            stt_word = seg.text
                            stt_conf = seg.confidence
                            break

                if tabula2_bursts:
                    # TABULA2 path: find bursts within ±frame_interval of this frame
                    # Use index pointer to avoid O(N) scan — bursts are time-sorted
                    frame_interval = 1.0 / cfg.fps
                    # Advance past consumed bursts
                    while _tabula2_burst_idx < len(tabula2_bursts) and tabula2_bursts[_tabula2_burst_idx][0] < t - frame_interval:
                        _tabula2_burst_idx += 1
                    for _bi in range(_tabula2_burst_idx, len(tabula2_bursts)):
                        burst_time, burst = tabula2_bursts[_bi]
                        if burst_time > t + frame_interval:
                            break
                        if abs(burst_time - t) < frame_interval:
                            burst_sub = {}
                            for attr in ("pitch_embedding", "harmonic_embedding",
                                         "formant_embedding", "vad_embedding",
                                         "spectral_embedding"):
                                val = getattr(burst, attr, None)
                                if val is not None:
                                    burst_sub[attr] = val.tolist()

                            # Quality signals for Ebbinghaus encoding
                            import math as _math
                            _attn = 1.0
                            if hasattr(sensor_graph, '_attention_ctrl') and sensor_graph._attention_ctrl:
                                phase = sensor_graph._attention_ctrl.phase
                                _attn = {"explore": 1.0, "inspect": 0.7, "monitor": 0.3}.get(phase, 0.5)
                            _surprise = 0.5  # default moderate
                            if progress.intelligence_cycles > 0:
                                _surprise = min(1.0, progress.total_surprise / progress.intelligence_cycles)
                            _temporal_prox = _math.exp(-abs(burst_time - t) / 2.0)

                            unit_id = await sensor_graph._sound_language.observe_sound(
                                audio_embedding=burst.embedding.tolist(),
                                timestamp=burst_time,
                                active_visual_ids=current_visual,
                                stt_text=stt_word,
                                stt_confidence=stt_conf,
                                modality="voice",
                                sub_features=burst_sub or None,
                                attention=_attn,
                                surprise=_surprise,
                                proximity=_temporal_prox,
                                voice_frames=burst.voice_frames,
                                sub_feature_frames=burst.sub_feature_frames,
                                burst_energy=burst.energy,
                            )
                            if unit_id and prev_sound_unit_id is not None:
                                gap_ms = (burst_time - prev_sound_timestamp) * 1000
                                await sensor_graph._sound_language.record_sequence(
                                    prev_sound_unit_id, unit_id, gap_ms,
                                )
                            if unit_id:
                                # Speech prediction: evaluate pending and generate new
                                if hasattr(sensor_graph, '_speech_predictor') and sensor_graph._speech_predictor:
                                    pred_results = await sensor_graph._speech_predictor.evaluate(unit_id)
                                    await sensor_graph._speech_predictor.predict_next(unit_id)
                                    # Feed prediction surprise into pressure system
                                    if pred_results:
                                        max_surprise = max((r.surprise for r in pred_results), default=0.0)
                                        # Feed surprise to budget allocator for next frame
                                        _speech_surprise = max_surprise
                                        if max_surprise > 0.3 and sensor_graph._pressure:
                                            import time as _time
                                            sensor_graph._pressure.surprise.observe(
                                                _time.monotonic(), max_surprise,
                                            )

                                prev_sound_unit_id = unit_id
                                prev_sound_timestamp = burst_time

                    # Environmental sounds: observe novelty events near this frame
                    # Use index pointer — noise events are time-sorted
                    while _tabula2_noise_idx < len(tabula2_noise_events) and tabula2_noise_events[_tabula2_noise_idx][0] < t - frame_interval:
                        _tabula2_noise_idx += 1
                    for _ni in range(_tabula2_noise_idx, len(tabula2_noise_events)):
                        evt_time, evt_emb = tabula2_noise_events[_ni]
                        if evt_time > t + frame_interval:
                            break
                        if abs(evt_time - t) < frame_interval:
                            await sensor_graph._sound_language.observe_sound(
                                audio_embedding=evt_emb.tolist(),
                                timestamp=evt_time,
                                active_visual_ids=current_visual,
                                modality="environmental",
                            )

                elif formant_emb and audio_has_onset:
                    # Legacy formant path
                    unit_id = await sensor_graph._sound_language.observe_sound(
                        audio_embedding=formant_emb,
                        timestamp=t,
                        active_visual_ids=current_visual,
                        stt_text=stt_word,
                        stt_confidence=stt_conf,
                    )
                    if unit_id and prev_sound_unit_id is not None:
                        gap = (t - prev_sound_timestamp) * 1000
                        await sensor_graph._sound_language.record_sequence(
                            prev_sound_unit_id, unit_id, gap,
                        )
                    if unit_id:
                        prev_sound_unit_id = unit_id
                        prev_sound_timestamp = t
            except Exception as e:
                log.warning("Sound language failed at %.1fs: %s", t, e, exc_info=True)

        # 4d. Intelligence loop: predict → observe → compare → update
        # Budget-gated: only run when visual budget >= STANDARD (novel frames).
        # This replaces the old familiarity gate with thalamic budget control.
        if config.INTELLIGENCE_LOOPS_ENABLED and promoted_node_ids and frame_budget.above("visual", LEVEL_STANDARD):
            try:
                # Pass full visual context so structural predictions verify
                # against the whole scene, not just this frame's promotions
                context_ids = list(set(vid for vid, ts, att in recent_visual_entries))
                intel_stats = await sensor_graph.run_intelligence_cycle(
                    promoted_node_ids,
                    frame_id=frame_id,
                    scene_change=frame.is_scene_change,
                    active_context_ids=context_ids,
                )
                if intel_stats:
                    progress.intelligence_cycles += 1
                    progress.total_surprise += intel_stats.get("surprise", 0)
                    progress.predictions_confirmed += intel_stats.get("confirmed", 0)
                    progress.predictions_refuted += intel_stats.get("refuted", 0)
                    if intel_stats.get("surprise", 0) > config.SURPRISE_HIGH:
                        log.info("HIGH SURPRISE at %.1fs (%.2f): %d predictions, %d refuted",
                                 t, intel_stats["surprise"],
                                 intel_stats["predictions"], intel_stats["refuted"])
            except Exception as e:
                log.debug("Intelligence cycle failed at %.1fs: %s", t, e)

        # 4e. Real-time recognition
        # If promoted nodes match known clusters, recognize instantly.
        # If they match unknown clusters, ask the LLM "what is this?"
        if recognition_engine and promoted_node_ids:
            try:
                recognition = await recognition_engine.recognize(promoted_node_ids, timestamp=t)
                if recognition.labels:
                    log.info("Recognized at %.1fs: %s", t, recognition.labels)
                for proposal in recognition.novel_proposals:
                    log.info("Novel at %.1fs: cluster #%d → \"%s\"",
                            t, proposal["cluster_id"], proposal["proposed_label"])
            except Exception as e:
                log.debug("Recognition failed at %.1fs: %s", t, e)

        # 4e. STT → text callback (bidirectional temporal binding)
        #
        # Two binding directions, mirroring how humans learn:
        #   Backward: image appears, then narrator says the name (visual leads audio by 0-5s)
        #   Forward:  narrator says the name, then image appears (audio leads visual by 0-3s)
        #
        # Backward binding: words at time t bind to visual nodes from t-5s..t (already in recent_visual_entries)
        # Forward binding: words spoken recently bind to NEW visual nodes appearing now
        next_t = frames[i + 1].timestamp if i + 1 < len(frames) else t + (1.0 / cfg.fps)
        if text_callback:
            # Backward: process words spoken during this frame's time window
            for sec in range(int(t), int(next_t) + 1):
                if sec in transcript_by_second:
                    for seg in transcript_by_second[sec]:
                        try:
                            current_visual = list(set(
                                vid for vid, ts, att in recent_visual_entries if att
                            ))
                            await text_callback(
                                seg.text, seg.start, seg.confidence,
                                active_visual_ids=current_visual,
                            )
                            progress.groundings_from_speech += 1
                            # Store for forward binding — these words may bind
                            # to visual nodes that appear in the next few seconds
                            pending_words.append((seg.text, seg.start, seg.confidence, []))
                        except Exception as e:
                            progress.errors.append(f"Text callback failed for '{seg.text}': {e}")

            # Forward: bind pending words to newly promoted visual nodes
            # Words from the last 3s that were spoken BEFORE these visuals appeared
            if promoted_visual and pending_words:
                # Prune expired pending words
                pending_words = [
                    (text, start, conf, vids)
                    for text, start, conf, vids in pending_words
                    if t - start <= WORD_FORWARD_WINDOW_S
                ]
                for pw_idx, (pw_text, pw_start, pw_conf, pw_vids) in enumerate(pending_words):
                    # Only bind to visual nodes that are NEW (not already bound to this word)
                    new_visual = [v for v in promoted_visual if v not in pw_vids]
                    if new_visual:
                        try:
                            await text_callback(
                                pw_text, pw_start, pw_conf,
                                active_visual_ids=new_visual,
                            )
                            # Track which visuals we've already bound to avoid double-counting
                            pending_words[pw_idx] = (pw_text, pw_start, pw_conf, pw_vids + new_visual)
                        except Exception:
                            pass

        progress.processed_frames = i + 1

        # 4e. Internal processing — use freed compute when sensors are quiet.
        # When visual budget is below REDUCED, the system has spare capacity.
        # Use it for lightweight internal work (decay, self-play).
        if frame_budget.above("internal", LEVEL_REDUCED) and not frame_budget.above("visual", LEVEL_REDUCED):
            try:
                p_stats = await sensor_graph.check_and_run_pressure_dreamstate()
                if p_stats:
                    ran = p_stats.get("ran_functions", [])
                    log.debug("Internal processing at frame %d: %s", i + 1, ran)
            except Exception:
                pass

        # 4f. Periodic consolidation + pressure-driven dreamstate
        if (i + 1) % cfg.consolidate_every == 0:
            try:
                await sensor_graph.consolidate()
                log.info("Consolidation at frame %d/%d", i + 1, len(frames))
            except Exception as e:
                progress.errors.append(f"Consolidation failed: {e}")

            # Check if pressure demands a dreamstate cycle
            try:
                p_stats = await sensor_graph.check_and_run_pressure_dreamstate()
                if p_stats:
                    ran = p_stats.get("ran_functions", [])
                    log.info("Pressure dreamstate at frame %d: %s (%.1fs)",
                             i + 1, ran, p_stats.get("duration_s", 0))
            except Exception as e:
                log.debug("Pressure dreamstate check failed: %s", e)

        # Free frame pixel data after processing — prevents 3-6GB accumulation
        frame.image = None

        # Prune old visual entries every frame (not just on promotion)
        cutoff = t - VISUAL_CONTEXT_WINDOW_S
        recent_visual_entries = [(nid, ts, att) for nid, ts, att in recent_visual_entries if ts >= cutoff]

        # Progress callback
        if progress_callback and (i + 1) % cfg.batch_size == 0:
            try:
                await progress_callback(progress)
            except Exception:
                pass

    # ── Step 5: Final consolidation ──────────────────────
    try:
        await sensor_graph.consolidate()
    except Exception as e:
        progress.errors.append(f"Final consolidation failed: {e}")

    avg_surprise = progress.total_surprise / max(progress.intelligence_cycles, 1)
    budget_stats = _budget_allocator.stats
    log.info(
        "Video ingestion complete: %s — %d frames (%d suppressed), %d visual, %d audio, "
        "%d speech, %d intel cycles, avg surprise %.2f, budget: %s",
        path.name,
        progress.processed_frames,
        _frames_suppressed,
        progress.visual_nodes_created,
        progress.audio_nodes_created,
        progress.transcript_segments,
        progress.intelligence_cycles,
        avg_surprise,
        budget_stats,
    )

    return progress
