from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.engine import Engine

from traphill.engine.types import VehicleEvent
from traphill.models import vehicle_events


def insert_event(engine: Engine, camera_id: str, event: VehicleEvent) -> int:
    """Insert a VehicleEvent and return the new row id."""
    row = {
        "camera_id": camera_id,
        "tracker_id": event.tracker_id,
        "vehicle_class": event.vehicle_class,
        "direction": event.direction,
        "speed_kmh": event.speed_kmh,
        "entered_at_s": event.entered_at_s,
        "exited_at_s": event.exited_at_s,
        "incomplete": 1 if event.incomplete else 0,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wire_a_frame": event.crossing_a.frame_number if event.crossing_a else None,
        "wire_b_frame": event.crossing_b.frame_number if event.crossing_b else None,
    }
    with engine.begin() as conn:
        result = conn.execute(vehicle_events.insert().values(**row))
        return result.inserted_primary_key[0]


def query_events(
    engine: Engine,
    camera_id: str | None = None,
    min_speed_kmh: float | None = None,
    max_speed_kmh: float | None = None,
    since: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Query vehicle events with optional filters. Returns list of dicts."""
    stmt = select(vehicle_events)

    if camera_id is not None:
        stmt = stmt.where(vehicle_events.c.camera_id == camera_id)
    if min_speed_kmh is not None:
        stmt = stmt.where(vehicle_events.c.speed_kmh >= min_speed_kmh)
    if max_speed_kmh is not None:
        stmt = stmt.where(vehicle_events.c.speed_kmh <= max_speed_kmh)
    if since is not None:
        stmt = stmt.where(vehicle_events.c.recorded_at >= since.isoformat())

    stmt = stmt.order_by(vehicle_events.c.id.desc()).limit(limit).offset(offset)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
