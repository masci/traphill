from fastapi.testclient import TestClient


def test_list_cameras(app_and_client):
    app, engine, app_config, manager = app_and_client
    with TestClient(app) as client:
        resp = client.get("/cameras")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "cam1"
        assert data[0]["alive"] is True


def test_add_camera(app_and_client):
    app, engine, app_config, manager = app_and_client
    with TestClient(app) as client:
        payload = {
            "id": "cam2",
            "name": "Camera 2",
            "url": "rtsp://192.168.1.20:554/stream",
            "confidence_threshold": 0.6,
            "tripwire": {"line_a_x": 200, "line_b_x": 500, "physical_distance_m": 8.0},
        }
        resp = client.post("/cameras", json=payload)
        assert resp.status_code == 201
        assert resp.json()["id"] == "cam2"
        manager.add_camera.assert_called_once()


def test_add_duplicate_camera_returns_409(app_and_client):
    app, engine, app_config, manager = app_and_client
    with TestClient(app) as client:
        payload = {
            "id": "cam1",  # already exists
            "name": "Dup",
            "url": "rtsp://x",
            "tripwire": {"line_a_x": 100, "line_b_x": 200, "physical_distance_m": 5.0},
        }
        resp = client.post("/cameras", json=payload)
        assert resp.status_code == 409


def test_remove_camera(app_and_client):
    app, engine, app_config, manager = app_and_client
    with TestClient(app) as client:
        resp = client.delete("/cameras/cam1")
        assert resp.status_code == 204
        manager.remove_camera.assert_called_once_with("cam1")


def test_remove_nonexistent_camera_returns_404(app_and_client):
    app, engine, app_config, manager = app_and_client
    with TestClient(app) as client:
        resp = client.delete("/cameras/does_not_exist")
        assert resp.status_code == 404
