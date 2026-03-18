from __future__ import annotations

import threading
from collections import deque

import numpy as np


class FrameBuffer:
    """Thread-safe ring buffer of annotated frames for a single camera."""

    def __init__(self, maxsize: int = 10):
        self._deque: deque[bytes] = deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        """Encode frame to JPEG and store. Drops oldest if full."""
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._lock:
            self._deque.append(buf.tobytes())

    def get_latest(self) -> bytes | None:
        """Return the most recently stored JPEG bytes, or None if empty."""
        with self._lock:
            return self._deque[-1] if self._deque else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)
