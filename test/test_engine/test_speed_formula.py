import pytest

from traphill.engine.estimator import compute_speed
from traphill.engine.types import CrossingEvent


def make_crossing(wire, timestamp_s, frame=0):
    return CrossingEvent(
        tracker_id=1,
        wire=wire,
        timestamp_s=timestamp_s,
        frame_number=frame,
        centroid=(0, 0),
    )


def test_speed_basic():
    # 10m in 0.5s = 20 m/s = 72 km/h
    ca = make_crossing("A", 0.0)
    cb = make_crossing("B", 0.5)
    assert compute_speed(ca, cb, 10.0) == pytest.approx(72.0)


def test_speed_order_independent():
    # B crossed before A (right-to-left): time is still abs()
    ca = make_crossing("A", 1.0)
    cb = make_crossing("B", 0.5)
    assert compute_speed(ca, cb, 10.0) == pytest.approx(72.0)


def test_speed_zero_elapsed_raises():
    ca = make_crossing("A", 1.0)
    cb = make_crossing("B", 1.0)
    with pytest.raises(ValueError, match="zero elapsed time"):
        compute_speed(ca, cb, 10.0)


def test_speed_different_distance():
    # 5m in 0.25s = 20 m/s = 72 km/h
    ca = make_crossing("A", 0.0)
    cb = make_crossing("B", 0.25)
    assert compute_speed(ca, cb, 5.0) == pytest.approx(72.0)
