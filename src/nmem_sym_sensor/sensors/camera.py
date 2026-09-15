"""
Camera sensor: live video from webcam or RTSP stream.

Captures frames at a configurable FPS, detects scene changes,
and publishes visual Observations to the sensor bus.
"""
import asyncio
import logging
import time
from datetime import UTC, datetime

import numpy as np

from nmem_sym_sensor.sensor_bus import Modality, Observation, Sensor

log = logging.getLogger(__name__)


class CameraSensor(Sensor):
    """Captures frames from a webcam or RTSP URL.

    Args:
        source: OpenCV device index (0 = default webcam) or RTSP URL string.
        fps: Target frame rate. Camera is throttled to this rate.
        scene_change_threshold: Mean pixel difference to detect scene changes.
        sensor_id: Unique identifier for this sensor.
        queue_size: Max observations buffered before dropping.
    """

    def __init__(
        self,
        source: int | str = 0,
        fps: float = 2.0,
        scene_change_threshold: float = 30.0,
        sensor_id: str = "camera_0",
        queue_size: int = 64,
    ):
        super().__init__(sensor_id, Modality.VISUAL, queue_size)
        self._source = source
        self._fps = fps
        self._scene_threshold = scene_change_threshold
        self._cap = None
        self._prev_gray = None

    async def _capture_loop(self) -> None:
        import cv2

        loop = asyncio.get_event_loop()
        self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            log.error("Cannot open camera: %s", self._source)
            return

        # Log camera info
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera opened: %s (%dx%d, target %.1f fps)",
                self._source, actual_w, actual_h, self._fps)

        interval = 1.0 / self._fps
        frame_idx = 0

        try:
            while self._running:
                t0 = time.monotonic()

                # Read frame in executor (VideoCapture.read can block)
                ret, frame = await loop.run_in_executor(None, self._cap.read)
                if not ret:
                    log.warning("Camera read failed, stopping")
                    break

                # Scene change detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scene_change = False
                if self._prev_gray is not None:
                    diff = cv2.absdiff(gray, self._prev_gray)
                    mean_diff = float(np.mean(diff))
                    if mean_diff > self._scene_threshold:
                        scene_change = True
                self._prev_gray = gray

                obs = Observation(
                    modality=Modality.VISUAL,
                    timestamp=time.monotonic(),
                    wall_time=datetime.now(UTC),
                    sensor_id=self.sensor_id,
                    data=frame,
                    metadata={
                        "frame_id": f"live_{frame_idx}",
                        "scene_change": scene_change,
                    },
                )
                await self._publish(obs)
                frame_idx += 1

                # Throttle to target FPS
                elapsed = time.monotonic() - t0
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

        except asyncio.CancelledError:
            pass
        finally:
            if self._cap:
                self._cap.release()
                log.info("Camera released (%d frames captured)", frame_idx)
