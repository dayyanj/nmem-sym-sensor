"""
Kinect v1 frame capture using libfreenect callback API.

The sync API hangs when requesting depth + RGB in the same process.
This module uses the callback-based runloop to grab both simultaneously.
"""
import threading
import time

import freenect
import numpy as np


class KinectCapture:
    """Captures synchronized RGB + depth frames from Kinect v1.

    Usage::

        kinect = KinectCapture()
        kinect.start()
        rgb, depth = kinect.get_frame()  # blocks until frame ready
        kinect.stop()
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._rgb_ts: float = 0
        self._depth_ts: float = 0
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._ctx = None

    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait for first frame
        self._frame_event.wait(timeout=5.0)

    def stop(self):
        """Stop capture and release device."""
        self._running = False
        if self._ctx is not None:
            try:
                freenect.shutdown(self._ctx)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_frame(self, timeout: float = 2.0) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get the latest RGB + depth frame.

        Returns:
            (rgb_480x640x3_uint8, depth_480x640_uint16) or (None, None)
        """
        self._frame_event.wait(timeout=timeout)
        with self._lock:
            return (
                self._rgb.copy() if self._rgb is not None else None,
                self._depth.copy() if self._depth is not None else None,
            )

    @property
    def has_frame(self) -> bool:
        return self._rgb is not None and self._depth is not None

    def _run(self):
        """Capture loop using freenect callback API."""
        try:
            self._ctx = freenect.init()
            dev = freenect.open_device(self._ctx, self.device_index)
            if dev is None:
                self._running = False
                return

            freenect.set_depth_mode(
                dev, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_11BIT,
            )
            freenect.set_video_mode(
                dev, freenect.RESOLUTION_MEDIUM, freenect.VIDEO_RGB,
            )

            def depth_cb(dev, data, timestamp):
                with self._lock:
                    self._depth = data.copy()
                    self._depth_ts = time.monotonic()
                if self._rgb is not None:
                    self._frame_event.set()

            def video_cb(dev, data, timestamp):
                with self._lock:
                    self._rgb = data.copy()
                    self._rgb_ts = time.monotonic()
                if self._depth is not None:
                    self._frame_event.set()

            freenect.set_depth_callback(dev, depth_cb)
            freenect.set_video_callback(dev, video_cb)
            freenect.start_depth(dev)
            freenect.start_video(dev)

            while self._running:
                # Process USB events (drives callbacks)
                if freenect.process_events(self._ctx) < 0:
                    break

        except Exception as e:
            print(f"Kinect capture error: {e}")
        finally:
            self._running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
