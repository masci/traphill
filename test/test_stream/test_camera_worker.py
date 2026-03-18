"""
Tests for CameraWorker reconnection and event emission.
Uses mocked cv2.VideoCapture and YOLO model.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from traphill.config import AppConfig, CameraConfig, TripwireConfig
from traphill.stream.camera import CameraWorker
from traphill.stream.frame_buffer import FrameBuffer


def make_app_config(camera_id: str = "cam1", url: str = "test://stream") -> AppConfig:
    return AppConfig(
        cameras=[
            CameraConfig(
                id=camera_id,
                name="Test Camera",
                url=url,
                confidence_threshold=0.6,
                tripwire=TripwireConfig(line_a_x=100, line_b_x=200, physical_distance_m=5.0),
            )
        ],
        yolo_model="yolo11n.mnn",
        stale_track_timeout_s=0.2,
    )


def _make_blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_worker_stops_cleanly():
    """Worker should stop within a short timeout when stop() is called."""
    app_config = make_app_config()
    cam_config = app_config.cameras[0]
    buf = FrameBuffer()
    events = []

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False  # immediately fail to open

    with patch("traphill.stream.camera.cv2.VideoCapture", return_value=mock_cap), \
         patch("traphill.stream.camera.YOLO"):
        worker = CameraWorker(cam_config, app_config, buf, lambda cid, e: events.append(e))
        worker.start()
        time.sleep(0.1)
        worker.stop()
        worker.join(timeout=2.0)
        assert not worker.is_alive()


def test_worker_reconnects_after_failed_open():
    """
    Worker should attempt to open the stream again after a failure.
    First call to isOpened() returns False, second returns True with finite frames.
    """
    app_config = make_app_config()
    cam_config = app_config.cameras[0]
    buf = FrameBuffer()
    events = []
    call_count = {"n": 0}

    frame = _make_blank_frame()

    def make_cap(*args, **kwargs):
        call_count["n"] += 1
        cap = MagicMock()
        if call_count["n"] == 1:
            cap.isOpened.return_value = False
        else:
            cap.isOpened.return_value = True
            cap.get.return_value = 25.0
            # Return 3 frames then signal end
            cap.read.side_effect = [
                (True, frame),
                (True, frame),
                (True, frame),
                (False, None),
            ]
        return cap

    mock_tracker = MagicMock()
    mock_tracker.process_frame.return_value = (frame, [])
    mock_tracker.flush.return_value = []

    with patch("traphill.stream.camera.cv2.VideoCapture", side_effect=make_cap), \
         patch("traphill.stream.camera.YOLO"), \
         patch("traphill.stream.camera.SessionTracker", return_value=mock_tracker), \
         patch("traphill.stream.camera.RECONNECT_DELAY_S", 0.05):

        worker = CameraWorker(cam_config, app_config, buf, lambda cid, e: events.append(e))
        worker.start()
        # Wait long enough for reconnect cycle
        time.sleep(0.5)
        worker.stop()
        worker.join(timeout=3.0)

    assert call_count["n"] >= 2, "VideoCapture should have been called at least twice"
    assert mock_tracker.process_frame.call_count >= 3


def test_worker_puts_frames_in_buffer():
    """Processed frames should appear in the FrameBuffer."""
    app_config = make_app_config()
    cam_config = app_config.cameras[0]
    buf = FrameBuffer()

    frame = _make_blank_frame()
    annotated = frame.copy()
    annotated[:] = 50  # slightly different so we know it came through tracker

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 25.0
    mock_cap.read.side_effect = [(True, frame), (True, frame), (False, None)]

    mock_tracker = MagicMock()
    mock_tracker.process_frame.return_value = (annotated, [])
    mock_tracker.flush.return_value = []

    with patch("traphill.stream.camera.cv2.VideoCapture", return_value=mock_cap), \
         patch("traphill.stream.camera.YOLO"), \
         patch("traphill.stream.camera.SessionTracker", return_value=mock_tracker), \
         patch("traphill.stream.camera.RECONNECT_DELAY_S", 0.05):

        worker = CameraWorker(cam_config, app_config, buf, lambda cid, e: None)
        worker.start()
        time.sleep(0.3)
        worker.stop()
        worker.join(timeout=2.0)

    assert len(buf) > 0
