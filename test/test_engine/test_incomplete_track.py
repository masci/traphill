"""
Test that a vehicle crossing only one wire produces an incomplete VehicleEvent
with speed_kmh=None.
"""
from traphill.engine.estimator import SpeedEstimator
from traphill.engine.types import Detection, TripwireConfig

LINE_A = 300
LINE_B = 600
FPS = 25.0


def make_config():
    return TripwireConfig(line_a_x=LINE_A, line_b_x=LINE_B, physical_distance_m=10.0)


def make_detection(cx: int, frame: int) -> Detection:
    half = 20
    return Detection(
        tracker_id=1,
        class_id=2,
        class_name="car",
        confidence=0.9,
        x1=cx - half,
        y1=100,
        x2=cx + half,
        y2=180,
        frame_number=frame,
        timestamp_s=frame / FPS,
    )


def test_only_wire_a_crossed():
    estimator = SpeedEstimator(make_config(), FPS)

    # Vehicle crosses A but disappears before reaching B
    for frame, cx in enumerate([100, 200, 320, 400]):
        estimator.update([make_detection(cx, frame)], frame)

    # Stale eviction: don't feed this tracker for stale_frames
    stale_frames = int(FPS * 2.0)
    events = estimator.update([], stale_frames + 4)

    assert len(events) == 1
    event = events[0]
    assert event.incomplete is True
    assert event.speed_kmh is None
    assert event.crossing_a is not None
    assert event.crossing_b is None


def test_no_wire_crossed():
    estimator = SpeedEstimator(make_config(), FPS)

    # Vehicle never enters the trap
    for frame, cx in enumerate([50, 80, 100, 120]):
        estimator.update([make_detection(cx, frame)], frame)

    stale_frames = int(FPS * 2.0)
    events = estimator.update([], stale_frames + 4)

    assert len(events) == 1
    event = events[0]
    assert event.incomplete is True
    assert event.speed_kmh is None
    assert event.crossing_a is None
    assert event.crossing_b is None
