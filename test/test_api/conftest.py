from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from traphill.config import AppConfig, CameraConfig, TripwireConfig
from traphill.database import get_engine, init_db
from traphill.storage.events import insert_event
from traphill.engine.types import CrossingEvent, VehicleEvent


def make_vehicle_event(tracker_id=1, speed_kmh=72.0, incomplete=False):
    ca = CrossingEvent(tracker_id=tracker_id, wire="A", timestamp_s=0.0, frame_number=10, centroid=(300, 200))
    cb = CrossingEvent(tracker_id=tracker_id, wire="B", timestamp_s=0.6, frame_number=25, centroid=(600, 200))
    return VehicleEvent(
        tracker_id=tracker_id,
        vehicle_class="car",
        direction="left_to_right",
        speed_kmh=speed_kmh,
        entered_at_s=0.0,
        exited_at_s=1.0,
        incomplete=incomplete,
        crossing_a=ca,
        crossing_b=cb,
    )


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    """Create a FastAPI TestClient with a real in-memory-like SQLite DB and no camera workers."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from contextlib import asynccontextmanager

    db_path = str(tmp_path / "test.db")
    engine = get_engine(db_path)
    init_db(engine)

    app_config = AppConfig(
        cameras=[
            CameraConfig(
                id="cam1",
                name="Test Camera 1",
                url="test://stream1",
                confidence_threshold=0.6,
                tripwire=TripwireConfig(line_a_x=300, line_b_x=700, physical_distance_m=10.0),
            )
        ],
        yolo_model="yolo11n.mnn",
        stale_track_timeout_s=2.0,
    )

    # Patch CameraManager to not actually start threads
    from unittest.mock import MagicMock
    mock_manager = MagicMock()
    mock_manager.camera_ids.return_value = ["cam1"]
    mock_manager.is_alive.return_value = True
    mock_manager.get_frame_buffer.return_value = None

    from traphill.api.routers import cameras as cameras_router, events as events_router, stream as stream_router
    from fastapi import FastAPI

    config_path = str(tmp_path / "config.yaml")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config_path = config_path
        app.state.app_config = app_config
        app.state.engine = engine
        app.state.manager = mock_manager
        yield

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(cameras_router.router)
    test_app.include_router(events_router.router)

    return test_app, engine, app_config, mock_manager
