"""Tests for CameraManager lifecycle."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from traphill.config import AppConfig, CameraConfig, TripwireConfig
from traphill.stream.manager import CameraManager


def make_app_config(n_cameras: int = 3) -> AppConfig:
    cameras = [
        CameraConfig(
            id=f"cam{i}",
            name=f"Camera {i}",
            url=f"test://stream{i}",
            confidence_threshold=0.6,
            tripwire=TripwireConfig(line_a_x=100, line_b_x=200, physical_distance_m=5.0),
        )
        for i in range(n_cameras)
    ]
    return AppConfig(cameras=cameras, yolo_model="yolo11n.mnn", stale_track_timeout_s=0.2)


def _never_opens(*args, **kwargs):
    cap = MagicMock()
    cap.isOpened.return_value = False
    return cap


def test_manager_starts_all_workers():
    app_config = make_app_config(3)
    manager = CameraManager(app_config, lambda cid, e: None)

    with patch("traphill.stream.camera.cv2.VideoCapture", side_effect=_never_opens), \
         patch("traphill.stream.camera.YOLO"):
        manager.start()
        time.sleep(0.1)
        assert set(manager.camera_ids()) == {"cam0", "cam1", "cam2"}
        for cid in manager.camera_ids():
            assert manager.is_alive(cid)
        manager.stop_all()


def test_manager_stop_all_joins_threads():
    app_config = make_app_config(2)
    manager = CameraManager(app_config, lambda cid, e: None)

    with patch("traphill.stream.camera.cv2.VideoCapture", side_effect=_never_opens), \
         patch("traphill.stream.camera.YOLO"):
        manager.start()
        time.sleep(0.1)
        manager.stop_all()
        assert manager.camera_ids() == []


def test_manager_exposes_frame_buffers():
    app_config = make_app_config(2)
    manager = CameraManager(app_config, lambda cid, e: None)

    with patch("traphill.stream.camera.cv2.VideoCapture", side_effect=_never_opens), \
         patch("traphill.stream.camera.YOLO"):
        manager.start()
        assert manager.get_frame_buffer("cam0") is not None
        assert manager.get_frame_buffer("cam1") is not None
        assert manager.get_frame_buffer("cam99") is None
        manager.stop_all()
