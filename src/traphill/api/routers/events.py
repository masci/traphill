from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query, Request

from traphill.api.schemas import EventsResponse, VehicleEventResponse
from traphill.storage.events import query_events

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=EventsResponse)
def list_events(
    request: Request,
    camera_id: str | None = Query(default=None),
    min_speed: float | None = Query(default=None),
    max_speed: float | None = Query(default=None),
    since: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    engine = request.app.state.engine
    rows = query_events(
        engine,
        camera_id=camera_id,
        min_speed_kmh=min_speed,
        max_speed_kmh=max_speed,
        since=since,
        limit=limit,
        offset=offset,
    )
    items = [
        VehicleEventResponse(
            id=r["id"],
            camera_id=r["camera_id"],
            tracker_id=r["tracker_id"],
            vehicle_class=r["vehicle_class"],
            direction=r["direction"],
            speed_kmh=r["speed_kmh"],
            entered_at_s=r["entered_at_s"],
            exited_at_s=r["exited_at_s"],
            incomplete=bool(r["incomplete"]),
            recorded_at=r["recorded_at"],
            wire_a_frame=r["wire_a_frame"],
            wire_b_frame=r["wire_b_frame"],
        )
        for r in rows
    ]
    return EventsResponse(items=items, total=len(items), limit=limit, offset=offset)
