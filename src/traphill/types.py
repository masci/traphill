from dataclasses import dataclass


@dataclass(kw_only=True)
class DetectedObject:
    x: int
    y: int
    width: int
    height: int
    name: str
    conf: float
    centroid: tuple[int, int]


@dataclass(kw_only=True)
class TrackedObject:
    center: tuple[int, int]
    start_x: int
    start_frame: int
    speed_kmh: float | None = None
    detected: bool = True


@dataclass
class TrapArea:
    x1: int
    x2: int
    height: int
