from dataclasses import dataclass, field

import numpy as np


@dataclass(kw_only=True)
class Detection:
    """A class representing a detected object with its bounding box."""

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


class Vehicle:
    """A class representing a tracked vehicle."""

    def __init__(self, detection: Detection, frame_number: int) -> None:
        self._detection = detection
        self._first_seen_detection = detection
        self._first_seen_frame_number = frame_number
        self._last_seen_frame_number = frame_number
        self._speed: float | None = None
        self._detected: bool = True

    def update(self, d: Detection, frame_number: int) -> None:
        self._last_seen_frame_number = frame_number
        self._detection = d
        self._detected = True

    def travelled_distance(self) -> float:
        return self._detection.distance(self._first_seen_detection)

    def frames_elapsed(self, current_frame: int) -> int:
        return current_frame - self._first_seen_frame_number

    @property
    def detected(self) -> bool:
        return self._detected

    @detected.setter
    def detected(self, d: bool) -> None:
        self._detected = d

    @property
    def detection(self) -> Detection:
        return self._detection


@dataclass
class TrapArea:
    x1: int
    x2: int
    height: int
