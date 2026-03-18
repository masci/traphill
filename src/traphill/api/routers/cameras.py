from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from traphill.api.schemas import CameraCreateRequest, CameraResponse, CameraUpdateRequest, TripwireSchema
from traphill.config import CameraConfig, save_config
from traphill.config import TripwireConfig as ConfigTripwire

router = APIRouter(prefix="/cameras", tags=["cameras"])


def _to_response(cam: CameraConfig, alive: bool) -> CameraResponse:
    return CameraResponse(
        id=cam.id,
        name=cam.name,
        url=cam.url,
        confidence_threshold=cam.confidence_threshold,
        tripwire=TripwireSchema(
            line_a_x=cam.tripwire.line_a_x,
            line_b_x=cam.tripwire.line_b_x,
            physical_distance_m=cam.tripwire.physical_distance_m,
        ),
        alive=alive,
    )


@router.get("", response_model=list[CameraResponse])
def list_cameras(request: Request):
    manager = request.app.state.manager
    app_config = request.app.state.app_config
    return [_to_response(cam, manager.is_alive(cam.id)) for cam in app_config.cameras]


@router.post("", response_model=CameraResponse, status_code=201)
def add_camera(request: Request, body: CameraCreateRequest):
    manager = request.app.state.manager
    app_config = request.app.state.app_config

    if any(c.id == body.id for c in app_config.cameras):
        raise HTTPException(status_code=409, detail=f"Camera '{body.id}' already exists")

    cam = CameraConfig(
        id=body.id,
        name=body.name,
        url=body.url,
        confidence_threshold=body.confidence_threshold,
        tripwire=ConfigTripwire(
            line_a_x=body.tripwire.line_a_x,
            line_b_x=body.tripwire.line_b_x,
            physical_distance_m=body.tripwire.physical_distance_m,
        ),
    )
    app_config.cameras.append(cam)
    manager.add_camera(cam)
    save_config(request.app.state.config_path, app_config)

    return _to_response(cam, manager.is_alive(cam.id))


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: str, request: Request, body: CameraUpdateRequest):
    manager = request.app.state.manager
    app_config = request.app.state.app_config

    cam = next((c for c in app_config.cameras if c.id == camera_id), None)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    # Apply partial updates
    updated = cam.model_copy(update={
        k: v for k, v in {
            "name": body.name,
            "url": body.url,
            "confidence_threshold": body.confidence_threshold,
            "tripwire": ConfigTripwire(
                line_a_x=body.tripwire.line_a_x,
                line_b_x=body.tripwire.line_b_x,
                physical_distance_m=body.tripwire.physical_distance_m,
            ) if body.tripwire else None,
        }.items() if v is not None
    })

    idx = next(i for i, c in enumerate(app_config.cameras) if c.id == camera_id)
    app_config.cameras[idx] = updated
    manager.update_camera(updated)
    save_config(request.app.state.config_path, app_config)

    return _to_response(updated, manager.is_alive(camera_id))


@router.delete("/{camera_id}", status_code=204)
def remove_camera(camera_id: str, request: Request):
    manager = request.app.state.manager
    app_config = request.app.state.app_config

    if not any(c.id == camera_id for c in app_config.cameras):
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    manager.remove_camera(camera_id)
    app_config.cameras = [c for c in app_config.cameras if c.id != camera_id]
    save_config(request.app.state.config_path, app_config)
