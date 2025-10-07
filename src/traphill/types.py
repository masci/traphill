from dataclasses import dataclass, field


@dataclass(kw_only=True)
class DetectedObject:
    x: int
    y: int
    width: int
    height: int
    name: str
    conf: float
    centroid: tuple[int, int] = field(init=False)

    def __post_init__(self):
        """Calculates the center point (centroid) of the bounding box."""
        self.centroid = (self.x + self.width // 2, self.y + self.height // 2)


@dataclass(kw_only=True)
class TrackedObject:
    center: tuple[int, int]
    start_x: int
    start_frame: int
    last_seen_frame: int
    speed_kmh: float | None = None
    detected: bool = True


@dataclass
class TrapArea:
    x1: int
    x2: int
    height: int
