from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2
from ultralytics import YOLO

from traphill.config import AppConfig, CameraConfig
from traphill.engine.tracker import SessionTracker
from traphill.engine.types import VehicleEvent
from traphill.stream.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 5.0


class CameraWorker(threading.Thread):
    """
    Runs in its own thread. Connects to a camera stream, processes frames,
    and calls on_event for each completed VehicleEvent.
    Reconnects automatically on failure.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        app_config: AppConfig,
        frame_buffer: FrameBuffer,
        on_event: Callable[[str, VehicleEvent], None],
    ):
        super().__init__(name=f"camera-{camera_config.id}", daemon=True)
        self._cam = camera_config
        self._app = app_config
        self._buffer = frame_buffer
        self._on_event = on_event
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def camera_id(self) -> str:
        return self._cam.id

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_session()
            except Exception as exc:
                logger.error("[%s] Unexpected error: %s", self._cam.id, exc)
            if not self._stop_event.is_set():
                logger.info("[%s] Reconnecting in %ss…", self._cam.id, RECONNECT_DELAY_S)
                self._stop_event.wait(RECONNECT_DELAY_S)

    def _run_session(self) -> None:
        url = self._cam.url
        logger.info("[%s] Connecting to %s", self._cam.id, url)

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logger.warning("[%s] Cannot open stream: %s", self._cam.id, url)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        model = YOLO(self._app.yolo_model)
        tracker = SessionTracker(
            model=model,
            camera_config=self._cam,
            fps=fps,
            stale_timeout_s=self._app.stale_track_timeout_s,
        )

        logger.info("[%s] Stream opened at %.1f fps", self._cam.id, fps)
        frame_number = 0

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("[%s] Frame read failed at frame %d", self._cam.id, frame_number)
                    break

                annotated, events = tracker.process_frame(frame, frame_number)
                self._buffer.put(annotated)
                frame_number += 1

                for event in events:
                    self._on_event(self._cam.id, event)

        finally:
            for event in tracker.flush():
                self._on_event(self._cam.id, event)
            cap.release()
            logger.info("[%s] Session ended after %d frames", self._cam.id, frame_number)
