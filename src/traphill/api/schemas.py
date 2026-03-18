from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TripwireSchema(BaseModel):
    line_a_x: int
    line_b_x: int
    physical_distance_m: float = Field(gt=0)


class CameraUpdateRequest(BaseModel):
    name: str | None = None
    url: str | None = None
    confidence_threshold: float | None = None
    tripwire: TripwireSchema | None = None


class CameraCreateRequest(BaseModel):
    id: str
    name: str
    url: str
    confidence_threshold: float = 0.6
    tripwire: TripwireSchema


class CameraResponse(BaseModel):
    id: str
    name: str
    url: str
    confidence_threshold: float
    tripwire: TripwireSchema
    alive: bool


class VehicleEventResponse(BaseModel):
    id: int
    camera_id: str
    tracker_id: int
    vehicle_class: str
    direction: str | None
    speed_kmh: float | None
    entered_at_s: float
    exited_at_s: float
    incomplete: bool
    recorded_at: str
    wire_a_frame: int | None
    wire_b_frame: int | None


class EventsResponse(BaseModel):
    items: list[VehicleEventResponse]
    total: int
    limit: int
    offset: int
