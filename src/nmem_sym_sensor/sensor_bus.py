"""
Sensor bus: unified abstraction for all sensory inputs.

Any input source — camera, microphone, file replay, future touch sensor —
implements the Sensor interface and publishes timestamped Observations
to the bus. The StreamingLoop consumes from the bus via async iteration.

Adding a new sensor type:
  1. Create a Sensor subclass with a _capture_loop()
  2. Call self._publish(Observation(...)) to emit data
  3. Register it on the bus: bus.register(MySensor())
  4. The streaming loop handles it automatically via modality dispatch
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

log = logging.getLogger(__name__)


class Modality(Enum):
    """Sensory modality types. Extensible for future sensors."""
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"             # STT transcription output
    # Future modalities:
    # TOUCH = "touch"
    # PROPRIOCEPTION = "proprioception"
    # TEMPERATURE = "temperature"
    # DEPTH = "depth"


@dataclass
class Observation:
    """A single timestamped sensory observation from any sensor.

    This is the universal currency of the sensor bus. Every sensor
    produces Observations regardless of modality. The StreamingLoop
    dispatches by modality to the appropriate handler.

    Attributes:
        modality: What kind of sensory data this is.
        timestamp: Monotonic clock time (time.monotonic()) for synchronization.
        wall_time: UTC wall clock for logging/storage.
        sensor_id: Which sensor produced this (e.g. "camera_0", "mic_0").
        data: The actual data. Type depends on modality:
            VISUAL: np.ndarray (HWC uint8 BGR image)
            AUDIO: np.ndarray (float32 mono waveform)
            TEXT: str (transcribed text)
        metadata: Modality-specific metadata:
            VISUAL: {"frame_id": str, "scene_change": bool}
            AUDIO: {"sample_rate": int, "window_id": str}
            TEXT: {"confidence": float, "start": float, "end": float}
    """
    modality: Modality
    timestamp: float                    # time.monotonic()
    wall_time: datetime
    sensor_id: str
    data: object                        # np.ndarray or str
    metadata: dict = field(default_factory=dict)


class Sensor(ABC):
    """Base class for all sensors.

    Subclasses implement _capture_loop() which runs as an asyncio task
    and calls self._publish() to emit observations. The internal queue
    provides backpressure — if the consumer can't keep up, the oldest
    observations are dropped (biological: vision drops frames constantly).
    """

    def __init__(
        self,
        sensor_id: str,
        modality: Modality,
        queue_size: int = 64,
    ):
        self.sensor_id = sensor_id
        self.modality = modality
        self._queue: asyncio.Queue[Observation | None] = asyncio.Queue(
            maxsize=queue_size,
        )
        self._running = False
        self._task: asyncio.Task | None = None
        self._published_count = 0
        self._dropped_count = 0

    @abstractmethod
    async def _capture_loop(self) -> None:
        """Subclasses implement sensor-specific capture logic.

        Call self._publish(observation) to emit data.
        Check self._running to know when to stop.
        """
        ...

    async def _publish(self, obs: Observation) -> None:
        """Put an observation on the queue.

        If the queue is full, drops the oldest observation (backpressure).
        This is intentional — better to process recent data than to
        accumulate a growing backlog.
        """
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_count += 1
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(obs)
        self._published_count += 1

    async def start(self) -> None:
        """Start the sensor capture loop."""
        self._running = True
        self._task = asyncio.create_task(
            self._capture_loop(),
            name=f"sensor_{self.sensor_id}",
        )
        log.info("Sensor '%s' (%s) started", self.sensor_id, self.modality.value)

    async def stop(self) -> None:
        """Stop the sensor and clean up."""
        self._running = False
        await self._queue.put(None)  # sentinel to unblock reader
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info(
            "Sensor '%s' stopped: %d published, %d dropped",
            self.sensor_id, self._published_count, self._dropped_count,
        )

    async def read(self) -> Observation | None:
        """Read the next observation. Returns None when sensor stops."""
        return await self._queue.get()

    @property
    def stats(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "modality": self.modality.value,
            "published": self._published_count,
            "dropped": self._dropped_count,
            "queue_size": self._queue.qsize(),
            "running": self._running,
        }


class SensorBus:
    """Multiplexes multiple sensors into a single observation stream.

    Usage::

        bus = SensorBus()
        bus.register(CameraSensor(source=0))
        bus.register(MicrophoneSensor())

        async with bus:
            async for obs in bus.stream():
                if obs.modality == Modality.VISUAL:
                    process_frame(obs.data)
                elif obs.modality == Modality.AUDIO:
                    process_audio(obs.data)
    """

    def __init__(self):
        self._sensors: dict[str, Sensor] = {}

    def register(self, sensor: Sensor) -> None:
        """Register a sensor on the bus."""
        if sensor.sensor_id in self._sensors:
            raise ValueError(f"Sensor '{sensor.sensor_id}' already registered")
        self._sensors[sensor.sensor_id] = sensor
        log.info("Registered sensor '%s' (%s)", sensor.sensor_id, sensor.modality.value)

    async def start_all(self) -> None:
        """Start all registered sensors."""
        for sensor in self._sensors.values():
            await sensor.start()

    async def stop_all(self) -> None:
        """Stop all registered sensors."""
        for sensor in self._sensors.values():
            await sensor.stop()

    async def stream(self):
        """Async generator yielding observations from all sensors.

        Observations arrive in order of completion (first-available),
        not strictly timestamp-sorted. For sensors running at different
        rates (camera at 2fps, mic at 64 windows/sec), audio observations
        will naturally interleave with visual ones.
        """
        if not self._sensors:
            return

        # Create one reader task per sensor
        pending: dict[str, asyncio.Task] = {}
        for sid, sensor in self._sensors.items():
            pending[sid] = asyncio.create_task(sensor.read())

        active = set(self._sensors.keys())

        try:
            while active:
                done, _ = await asyncio.wait(
                    [pending[sid] for sid in active],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    # Find which sensor produced this
                    sid = next(s for s, t in pending.items() if t is task)

                    try:
                        obs = task.result()
                    except Exception as e:
                        log.warning("Sensor '%s' read failed: %s", sid, e)
                        active.discard(sid)
                        continue

                    if obs is None:
                        # Sensor stopped
                        active.discard(sid)
                        continue

                    yield obs

                    # Re-arm reader for this sensor
                    if sid in active:
                        pending[sid] = asyncio.create_task(
                            self._sensors[sid].read()
                        )
        finally:
            # Cancel any pending reads
            for sid in active:
                if sid in pending:
                    pending[sid].cancel()

    def sensor_stats(self) -> list[dict]:
        """Return stats for all registered sensors."""
        return [s.stats for s in self._sensors.values()]

    async def __aenter__(self):
        await self.start_all()
        return self

    async def __aexit__(self, *exc):
        await self.stop_all()
