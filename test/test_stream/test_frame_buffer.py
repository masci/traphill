"""Tests for the FrameBuffer ring buffer."""
import numpy as np
import pytest

from traphill.stream.frame_buffer import FrameBuffer


def make_frame(value: int) -> np.ndarray:
    """Create a small solid-color BGR frame distinguishable by value."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:] = value % 256
    return frame


def test_empty_buffer_returns_none():
    buf = FrameBuffer(maxsize=5)
    assert buf.get_latest() is None


def test_single_put_get():
    buf = FrameBuffer(maxsize=5)
    buf.put(make_frame(100))
    result = buf.get_latest()
    assert result is not None
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_ring_drops_oldest():
    buf = FrameBuffer(maxsize=5)
    for i in range(10):
        buf.put(make_frame(i))
    # Buffer should hold at most 5 frames
    assert len(buf) == 5


def test_get_latest_returns_most_recent():
    buf = FrameBuffer(maxsize=10)
    frames = [make_frame(i * 40) for i in range(5)]
    for f in frames:
        buf.put(f)
    # The latest bytes should decode to the last frame put in
    import cv2
    latest = buf.get_latest()
    decoded = cv2.imdecode(np.frombuffer(latest, dtype=np.uint8), cv2.IMREAD_COLOR)
    last_frame = frames[-1]
    # Check dominant color channel value matches last frame's fill value
    assert decoded is not None
    assert abs(int(decoded[32, 32, 0]) - int(last_frame[32, 32, 0])) < 20  # JPEG lossy tolerance


def test_concurrent_writes_do_not_corrupt():
    """Multiple threads writing simultaneously should not raise or corrupt."""
    import threading
    buf = FrameBuffer(maxsize=10)
    errors = []

    def writer(start):
        try:
            for i in range(50):
                buf.put(make_frame(start + i))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i * 50,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(buf) <= 10
