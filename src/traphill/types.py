from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import PIXELS_TO_METERS_FACTOR


class Direction(Enum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2
    STILL = 3


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
    frame_number: int

    def __post_init__(self):
        """Calculates the center point (centroid) of the bounding box."""
        self.centroid = (self.x + self.width // 2, self.y + self.height // 2)

    def distance(self, other: "Detection") -> float:
        d = np.linalg.norm(np.array(self.centroid) - np.array(other.centroid))
        return float(d)


@dataclass(kw_only=True)
class Vehicle:
    speed: float | None = None
    detections: list[Detection] = field(default_factory=list)

    @property
    def first_seen(self) -> Detection:
        return self.detections[0]

    @property
    def last_seen(self) -> Detection:
        try:
            return self.detections[-1]
        except KeyError:
            raise ValueError("No detections found")

    @property
    def first_seen_frame_number(self) -> int:
        try:
            return self.detections[0].frame_number
        except KeyError:
            raise ValueError("No detections found")

    @property
    def direction(self) -> Direction:
        if len(self.detections) < 2:
            return Direction.UNKNOWN

        elif self.detections[0].x < self.detections[-1].x:
            return Direction.RIGHT
        elif self.detections[0].x > self.detections[-1].x:
            return Direction.LEFT
        else:
            return Direction.STILL

    def current(self, current_frame: int) -> Detection | None:
        last_seen = self.last_seen
        if last_seen.frame_number == current_frame:
            return last_seen
        return None

    def travelled_distance(self) -> float:
        return self.first_seen.distance(self.last_seen)

    def frames_elapsed(self, current_frame: int, since_first: bool = True) -> int:
        if since_first:
            return current_frame - self.first_seen_frame_number
        return current_frame - self.last_seen.frame_number

    def calculate_speed(self, current_frame: int, fps: float) -> None:
        """
        Approximates speed (in Km/h) based on pixels traveled over frames elapsed.
        The speed is calculated as the average to cross the trap area.
        """
        try:
            distance_pixels = self.travelled_distance()
            if distance_pixels == 0:
                return None

            frames_elapsed = self.frames_elapsed(current_frame)
            if frames_elapsed == 0:
                return None

        except ValueError:
            return None

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

    def at_border(self, v: Detection) -> bool:
        centroid_x, _ = v.centroid
        return centroid_x == self.x1 or centroid_x == self.x2
