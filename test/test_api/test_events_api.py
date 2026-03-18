from fastapi.testclient import TestClient

from traphill.storage.events import insert_event
from test.test_api.conftest import make_vehicle_event


def test_list_events_empty(app_and_client):
    app, engine, _, _ = app_and_client
    with TestClient(app) as client:
        resp = client.get("/events")
        assert resp.status_code == 200
        assert resp.json()["items"] == []


def test_list_events_returns_inserted(app_and_client):
    app, engine, _, _ = app_and_client
    insert_event(engine, "cam1", make_vehicle_event(tracker_id=1, speed_kmh=85.0))
    insert_event(engine, "cam1", make_vehicle_event(tracker_id=2, speed_kmh=55.0))
    with TestClient(app) as client:
        resp = client.get("/events")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 2


def test_filter_min_speed(app_and_client):
    app, engine, _, _ = app_and_client
    insert_event(engine, "cam1", make_vehicle_event(tracker_id=1, speed_kmh=50.0))
    insert_event(engine, "cam1", make_vehicle_event(tracker_id=2, speed_kmh=100.0))
    insert_event(engine, "cam1", make_vehicle_event(tracker_id=3, speed_kmh=130.0))
    with TestClient(app) as client:
        resp = client.get("/events?min_speed=90")
        items = resp.json()["items"]
        assert len(items) == 2
        assert all(i["speed_kmh"] >= 90 for i in items)


def test_filter_by_camera(app_and_client):
    app, engine, _, _ = app_and_client
    insert_event(engine, "cam1", make_vehicle_event(tracker_id=1))
    insert_event(engine, "cam2", make_vehicle_event(tracker_id=2))
    with TestClient(app) as client:
        resp = client.get("/events?camera_id=cam1")
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["camera_id"] == "cam1"


def test_pagination(app_and_client):
    app, engine, _, _ = app_and_client
    for i in range(10):
        insert_event(engine, "cam1", make_vehicle_event(tracker_id=i))
    with TestClient(app) as client:
        p1 = client.get("/events?limit=4&offset=0").json()["items"]
        p2 = client.get("/events?limit=4&offset=4").json()["items"]
        assert len(p1) == 4
        assert len(p2) == 4
        assert {i["id"] for i in p1}.isdisjoint({i["id"] for i in p2})
