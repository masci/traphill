from dataclasses import dataclass, field

import numpy as np

from .config import PIXELS_TO_METERS_FACTOR


@dataclass(kw_only=True)
class Detection:
    """A class representing a detected object with its bounding box."""

    id: int
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

    def distance(self, other: "Detection") -> float:
        d = np.linalg.norm(np.array(self.centroid) - np.array(other.centroid))
        return float(d)


@dataclass(kw_only=True)
class Vehicle:
    first_seen: Detection
    current: Detection | None
    first_seen_frame_number: int
    speed: float | None = None

    def travelled_distance(self) -> float:
        return self.first_seen.distance(self.current)

    def frames_elapsed(self, current_frame: int) -> int:
        return current_frame - self.first_seen_frame_number

    def calculate_speed(self, current_frame: int, fps: float) -> float | None:
        """
        Approximates speed (in Km/h) based on pixels traveled over frames elapsed.
        The speed is calculated as the average to cross the trap area.
        """
        distance_pixels = self.travelled_distance()
        frames_elapsed = self.frames_elapsed(current_frame)
        if frames_elapsed <= 0:
            return None  # Cannot divide by zero or zero movement

        # 1. Pixel Speed: distance in pixels / time in seconds
        time_seconds = frames_elapsed / fps
        pixel_speed_per_second = distance_pixels / time_seconds

        # 2. Real-World Speed (Meters/Second)
        meters_per_second = pixel_speed_per_second * PIXELS_TO_METERS_FACTOR

        # 3. Convert to kmh (Meters/Second * 3.6)
        kmh = meters_per_second * 3.6
        self.speed = round(kmh, 1)


@dataclass
class TrapArea:
    x1: int
    x2: int
    height: int

    def contains(self, v: Detection) -> bool:
        centroid_x, _ = v.centroid
        return self.x1 <= centroid_x <= self.x2
