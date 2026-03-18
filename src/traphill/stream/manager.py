from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Callable

from traphill.config import AppConfig, CameraConfig
from traphill.engine.types import VehicleEvent
from traphill.stream.camera import CameraWorker
from traphill.stream.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)

STOP_TIMEOUT_S = 10.0


class CameraManager:
    """Starts and stops CameraWorker threads for all configured cameras."""

    def __init__(
        self,
        app_config: AppConfig,
        on_event: Callable[[str, VehicleEvent], None],
        event_loop: asyncio.AbstractEventLoop | None = None,
    ):
        self._app_config = app_config
        self._on_event = on_event
        self._loop = event_loop
        self._workers: dict[str, CameraWorker] = {}
        self._buffers: dict[str, FrameBuffer] = {}
        # camera_id -> set of asyncio.Queue subscribers
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._sub_lock = threading.Lock()

    def start(self) -> None:
        for cam in self._app_config.cameras:
            self._start_camera(cam)

    def _start_camera(self, cam: CameraConfig) -> None:
        if cam.id in self._workers:
            logger.warning("Camera %s already running, skipping", cam.id)
            return
        buf = FrameBuffer()
        worker = CameraWorker(
            camera_config=cam,
            app_config=self._app_config,
            frame_buffer=buf,
            on_event=self._handle_event,
        )
        self._buffers[cam.id] = buf
        self._workers[cam.id] = worker
        self._subscribers.setdefault(cam.id, set())
        worker.start()
        logger.info("Started worker for camera %s", cam.id)

    def add_camera(self, cam: CameraConfig) -> None:
        self._start_camera(cam)

    def update_camera(self, cam: CameraConfig) -> None:
        """Stop the existing worker for cam.id and restart it with the new config."""
        self.remove_camera(cam.id)
        self._start_camera(cam)

    def remove_camera(self, camera_id: str) -> None:
        worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
            worker.join(timeout=STOP_TIMEOUT_S)
        self._buffers.pop(camera_id, None)
        with self._sub_lock:
            self._subscribers.pop(camera_id, None)

    def stop_all(self) -> None:
        for worker in self._workers.values():
            worker.stop()
        for worker in self._workers.values():
            worker.join(timeout=STOP_TIMEOUT_S)
            if worker.is_alive():
                logger.warning("Worker %s did not stop in time", worker.camera_id)
        self._workers.clear()
        self._buffers.clear()
        logger.info("All camera workers stopped")

    def get_frame_buffer(self, camera_id: str) -> FrameBuffer | None:
        return self._buffers.get(camera_id)

    def camera_ids(self) -> list[str]:
        return list(self._workers.keys())

    def is_alive(self, camera_id: str) -> bool:
        worker = self._workers.get(camera_id)
        return worker is not None and worker.is_alive()

    def subscribe(self, camera_id: str, queue: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subscribers.setdefault(camera_id, set()).add(queue)

    def unsubscribe(self, camera_id: str, queue: asyncio.Queue) -> None:
        with self._sub_lock:
            subs = self._subscribers.get(camera_id, set())
            subs.discard(queue)

    def _handle_event(self, camera_id: str, event: VehicleEvent) -> None:
        # Call the primary persistence callback
        self._on_event(camera_id, event)
        # Fan-out to WebSocket subscribers (thread-safe bridge to asyncio)
        if self._loop is None:
            return
        payload = {
            "tracker_id": event.tracker_id,
            "vehicle_class": event.vehicle_class,
            "direction": event.direction,
            "speed_kmh": round(event.speed_kmh, 2) if event.speed_kmh is not None else None,
            "incomplete": event.incomplete,
        }
        with self._sub_lock:
            queues = list(self._subscribers.get(camera_id, set()))
        for q in queues:
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, payload)
            except asyncio.QueueFull:
                pass  # slow consumer; drop the event rather than block the worker
