"""
File sensor: replay pre-recorded video through the sensor bus.

Wraps the existing extract_frames/extract_audio infrastructure for
backward compatibility. Videos can be processed through the same
StreamingLoop as live camera/mic input.
"""
import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from nmem_sym_sensor import config
from nmem_sym_sensor.sensor_bus import Modality, Observation, Sensor

log = logging.getLogger(__name__)


class FileSensor(Sensor):
    """Replays a video file as a stream of visual + audio observations.

    Extracts frames and audio from a video file and publishes them
    as Observations with correct timestamps. This makes file-based
    ingestion composable with the sensor bus architecture.

    Args:
        video_path: Path to the video file.
        fps: Frame extraction rate.
        include_audio: Whether to also extract and publish audio windows.
        sensor_id: Unique identifier (defaults to filename).
        queue_size: Max observations buffered.
    """

    def __init__(
        self,
        video_path: str | Path,
        fps: float = 2.0,
        include_audio: bool = True,
        audio_window_s: float = 0.5,
        audio_hop_s: float = 0.25,
        sensor_id: str | None = None,
        queue_size: int = 128,
    ):
        path = Path(video_path)
        sid = sensor_id or f"file_{path.stem}"
        super().__init__(sid, Modality.VISUAL, queue_size)
        self._path = str(path)
        self._fps = fps
        self._include_audio = include_audio
        self._audio_window_s = audio_window_s
        self._audio_hop_s = audio_hop_s

    async def _capture_loop(self) -> None:
        from nmem_sym_sensor.ingest.video import (
            VideoConfig,
            extract_audio,
            extract_frames,
            window_audio,
        )

        # Extract all frames
        cfg = VideoConfig(fps=self._fps)
        try:
            frames = extract_frames(self._path, cfg)
        except Exception as e:
            log.error("Frame extraction failed for %s: %s", self._path, e)
            return

        # Extract audio windows
        audio_windows = []
        if self._include_audio:
            try:
                waveform, sr = extract_audio(self._path)
                audio_windows = window_audio(
                    waveform, sr, self._audio_window_s, self._audio_hop_s,
                )
            except Exception as e:
                log.warning("Audio extraction failed: %s", e)

        # Interleave frames and audio by timestamp
        audio_idx = 0
        base_time = time.monotonic()

        for frame in frames:
            # Publish any audio windows that precede this frame
            while audio_idx < len(audio_windows):
                samples, win_start, win_end = audio_windows[audio_idx]
                if win_start > frame.timestamp:
                    break

                audio_obs = Observation(
                    modality=Modality.AUDIO,
                    timestamp=base_time + win_start,
                    wall_time=datetime.now(UTC),
                    sensor_id=self.sensor_id + "_audio",
                    data=samples,
                    metadata={
                        "sample_rate": config.AUDIO_SAMPLE_RATE,
                        "window_id": f"{Path(self._path).stem}_a{audio_idx}",
                    },
                )
                await self._publish(audio_obs)
                audio_idx += 1

            # Publish visual frame
            visual_obs = Observation(
                modality=Modality.VISUAL,
                timestamp=base_time + frame.timestamp,
                wall_time=datetime.now(UTC),
                sensor_id=self.sensor_id,
                data=frame.image,
                metadata={
                    "frame_id": f"{Path(self._path).stem}_f{frame.frame_idx}",
                    "scene_change": frame.is_scene_change,
                    "video_timestamp": frame.timestamp,
                },
            )
            await self._publish(visual_obs)

            # Yield control to allow consumption
            await asyncio.sleep(0)

        # Remaining audio windows
        while audio_idx < len(audio_windows):
            samples, win_start, win_end = audio_windows[audio_idx]
            audio_obs = Observation(
                modality=Modality.AUDIO,
                timestamp=base_time + win_start,
                wall_time=datetime.now(UTC),
                sensor_id=self.sensor_id + "_audio",
                data=samples,
                metadata={
                    "sample_rate": config.AUDIO_SAMPLE_RATE,
                    "window_id": f"{Path(self._path).stem}_a{audio_idx}",
                },
            )
            await self._publish(audio_obs)
            audio_idx += 1

        log.info("FileSensor '%s': %d frames, %d audio windows published",
                self.sensor_id, len(frames), audio_idx)
