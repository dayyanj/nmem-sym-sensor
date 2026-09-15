"""
Microphone sensor: live audio from system microphone.

Captures audio in rolling windows using sounddevice, publishes
audio Observations to the sensor bus.
"""
import asyncio
import logging
import time
from datetime import UTC, datetime

import numpy as np

from nmem_sym_sensor import config
from nmem_sym_sensor.sensor_bus import Modality, Observation, Sensor

log = logging.getLogger(__name__)


class MicrophoneSensor(Sensor):
    """Captures rolling audio windows from a microphone.

    Uses sounddevice for cross-platform audio capture. The audio
    callback runs in a separate thread; data is passed to the
    asyncio event loop via call_soon_threadsafe.

    Args:
        device: Audio device index (None = system default).
        sample_rate: Sample rate in Hz.
        window_s: Analysis window duration in seconds.
        hop_s: Hop between consecutive windows in seconds.
        sensor_id: Unique identifier for this sensor.
        queue_size: Max observations buffered before dropping.
    """

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int | None = None,
        window_s: float = 0.5,
        hop_s: float = 0.25,
        sensor_id: str = "mic_0",
        queue_size: int = 128,
    ):
        super().__init__(sensor_id, Modality.AUDIO, queue_size)
        self._device = device
        self._sr = sample_rate or config.AUDIO_SAMPLE_RATE
        self._window_samples = int(window_s * self._sr)
        self._hop_samples = int(hop_s * self._sr)

    async def _capture_loop(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            log.error("sounddevice not installed: pip install sounddevice")
            return

        loop = asyncio.get_event_loop()

        # Thread-safe queue for audio chunks from the callback
        chunk_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)

        def audio_callback(indata, frames, time_info, status):
            """Called from sounddevice's audio thread."""
            if status:
                log.debug("Audio status: %s", status)
            # Copy data and send to event loop
            loop.call_soon_threadsafe(
                chunk_queue.put_nowait, indata.copy(),
            )

        # Ring buffer for assembling full windows
        ring = np.zeros(self._window_samples, dtype=np.float32)
        ring_pos = 0
        window_idx = 0
        samples_since_emit = 0

        stream = sd.InputStream(
            device=self._device,
            samplerate=self._sr,
            channels=1,
            blocksize=self._hop_samples,
            dtype="float32",
            callback=audio_callback,
        )

        log.info("Microphone opened: device=%s, %d Hz, %.1fs windows",
                self._device or "default", self._sr,
                self._window_samples / self._sr)
        stream.start()

        try:
            while self._running:
                # Wait for next audio chunk from the callback thread
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=2.0)
                except TimeoutError:
                    continue

                # Flatten to mono
                chunk_mono = chunk[:, 0] if chunk.ndim > 1 else chunk.flatten()
                n = len(chunk_mono)

                # Fill ring buffer
                end = ring_pos + n
                if end <= len(ring):
                    ring[ring_pos:end] = chunk_mono
                else:
                    first = len(ring) - ring_pos
                    ring[ring_pos:] = chunk_mono[:first]
                    ring[:n - first] = chunk_mono[first:]
                ring_pos = end % len(ring)
                samples_since_emit += n

                # Emit a window every hop
                if samples_since_emit >= self._hop_samples:
                    # Linearize the ring buffer (roll so position 0 is oldest)
                    window = np.roll(ring, -ring_pos).copy()

                    obs = Observation(
                        modality=Modality.AUDIO,
                        timestamp=time.monotonic(),
                        wall_time=datetime.now(UTC),
                        sensor_id=self.sensor_id,
                        data=window,
                        metadata={
                            "sample_rate": self._sr,
                            "window_id": f"live_a{window_idx}",
                        },
                    )
                    await self._publish(obs)
                    window_idx += 1
                    samples_since_emit = 0

        except asyncio.CancelledError:
            pass
        finally:
            stream.stop()
            stream.close()
            log.info("Microphone closed (%d windows captured)", window_idx)
