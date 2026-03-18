"""Tests for the persistence layer."""
from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from traphill.database import get_engine, init_db
from traphill.engine.types import CrossingEvent, VehicleEvent
from traphill.storage.events import insert_event, query_events


@pytest.fixture
def engine(tmp_path):
    db_path = str(tmp_path / "test.db")
    eng = get_engine(db_path)
    init_db(eng)
    return eng


def make_event(
    tracker_id: int = 1,
    speed_kmh: float | None = 60.0,
    incomplete: bool = False,
    direction: str | None = "left_to_right",
) -> VehicleEvent:
    ca = CrossingEvent(tracker_id=tracker_id, wire="A", timestamp_s=0.0, frame_number=10, centroid=(300, 200))
    cb = CrossingEvent(tracker_id=tracker_id, wire="B", timestamp_s=0.6, frame_number=25, centroid=(600, 200))
    return VehicleEvent(
        tracker_id=tracker_id,
        vehicle_class="car",
        direction=direction,
        speed_kmh=speed_kmh,
        entered_at_s=0.0,
        exited_at_s=1.0,
        incomplete=incomplete,
        crossing_a=ca if not incomplete else None,
        crossing_b=cb if not incomplete else None,
    )


def test_insert_and_retrieve(engine):
    event = make_event(tracker_id=1, speed_kmh=72.0)
    row_id = insert_event(engine, "cam1", event)
    assert row_id == 1

    rows = query_events(engine)
    assert len(rows) == 1
    assert rows[0]["speed_kmh"] == pytest.approx(72.0)
    assert rows[0]["camera_id"] == "cam1"
    assert rows[0]["tracker_id"] == 1
    assert rows[0]["incomplete"] == 0


def test_filter_by_min_speed(engine):
    insert_event(engine, "cam1", make_event(tracker_id=1, speed_kmh=50.0))
    insert_event(engine, "cam1", make_event(tracker_id=2, speed_kmh=90.0))
    insert_event(engine, "cam1", make_event(tracker_id=3, speed_kmh=120.0))

    rows = query_events(engine, min_speed_kmh=80.0)
    assert len(rows) == 2
    assert all(r["speed_kmh"] >= 80.0 for r in rows)


def test_filter_by_camera_id(engine):
    insert_event(engine, "cam1", make_event(tracker_id=1))
    insert_event(engine, "cam2", make_event(tracker_id=2))
    insert_event(engine, "cam1", make_event(tracker_id=3))

    rows = query_events(engine, camera_id="cam1")
    assert len(rows) == 2
    assert all(r["camera_id"] == "cam1" for r in rows)


def test_filter_by_speed_range(engine):
    for speed in [30.0, 60.0, 90.0, 120.0]:
        insert_event(engine, "cam1", make_event(speed_kmh=speed))

    rows = query_events(engine, min_speed_kmh=50.0, max_speed_kmh=100.0)
    assert len(rows) == 2
    speeds = {r["speed_kmh"] for r in rows}
    assert speeds == {60.0, 90.0}


def test_incomplete_event_stored(engine):
    event = make_event(incomplete=True, speed_kmh=None, direction=None)
    insert_event(engine, "cam1", event)

    rows = query_events(engine)
    assert len(rows) == 1
    assert rows[0]["incomplete"] == 1
    assert rows[0]["speed_kmh"] is None
    assert rows[0]["direction"] is None
    assert rows[0]["wire_a_frame"] is None
    assert rows[0]["wire_b_frame"] is None


def test_pagination(engine):
    for i in range(10):
        insert_event(engine, "cam1", make_event(tracker_id=i))

    page1 = query_events(engine, limit=4, offset=0)
    page2 = query_events(engine, limit=4, offset=4)
    assert len(page1) == 4
    assert len(page2) == 4
    ids1 = {r["id"] for r in page1}
    ids2 = {r["id"] for r in page2}
    assert ids1.isdisjoint(ids2)


def test_concurrent_writes(engine):
    """Two threads writing 100 events each should produce 200 rows with no errors."""
    errors = []

    def writer(camera_id: str):
        try:
            for i in range(100):
                insert_event(engine, camera_id, make_event(tracker_id=i))
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=writer, args=("cam1",))
    t2 = threading.Thread(target=writer, args=("cam2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    all_rows = query_events(engine, limit=300)
    assert len(all_rows) == 200
