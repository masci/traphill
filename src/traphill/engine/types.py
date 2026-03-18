from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Detection:
    tracker_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    frame_number: int
    timestamp_s: float  # frame_number / fps

    @property
    def centroid(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class TripwireConfig:
    line_a_x: int
    line_b_x: int
    physical_distance_m: float


@dataclass
class CrossingEvent:
    tracker_id: int
    wire: Literal["A", "B"]
    timestamp_s: float
    frame_number: int
    centroid: tuple[int, int]


@dataclass
class VehicleEvent:
    tracker_id: int
    vehicle_class: str
    direction: Literal["left_to_right", "right_to_left"] | None
    speed_kmh: float | None
    entered_at_s: float
    exited_at_s: float
    incomplete: bool
    crossing_a: CrossingEvent | None = None
    crossing_b: CrossingEvent | None = None
