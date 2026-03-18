"""
Test that SpeedEstimator correctly detects wire crossings and emits a VehicleEvent
when a vehicle crosses both tripwires left-to-right.
"""
import pytest

from traphill.engine.estimator import SpeedEstimator
from traphill.engine.types import Detection, TripwireConfig

LINE_A = 300
LINE_B = 600
DISTANCE_M = 10.0
FPS = 25.0


def make_config():
    return TripwireConfig(line_a_x=LINE_A, line_b_x=LINE_B, physical_distance_m=DISTANCE_M)


def make_detection(tracker_id: int, cx: int, frame_number: int) -> Detection:
    half = 20
    return Detection(
        tracker_id=tracker_id,
        class_id=2,
        class_name="car",
        confidence=0.9,
        x1=cx - half,
        y1=100,
        x2=cx + half,
        y2=180,
        frame_number=frame_number,
        timestamp_s=frame_number / FPS,
    )


def test_left_to_right_emits_event():
    estimator = SpeedEstimator(make_config(), FPS)

    # Move vehicle from left (x=100) through line A (300) and line B (600) to right (x=700)
    positions = [100, 200, 320, 450, 620, 700]
    events = []
    for i, cx in enumerate(positions):
        det = make_detection(1, cx, i)
        events.extend(estimator.update([det], i))

    # Flush remaining
    events.extend(estimator.flush())

    assert len(events) == 1
    event = events[0]
    assert event.tracker_id == 1
    assert event.direction == "left_to_right"
    assert event.speed_kmh is not None
    assert event.speed_kmh > 0
    assert not event.incomplete


def test_speed_value_matches_formula():
    estimator = SpeedEstimator(make_config(), FPS)

    # Frame 0: before A (x=100), Frame 5: after A (x=350), Frame 15: after B (x=650)
    # Crossing A at ~frame 2-3 (between 200 and 350), crossing B at ~frame 10-11
    # We want predictable timing: place crossings exactly

    # Construct so A is crossed between frames 2→3, B between frames 8→9
    # elapsed = (9/25) - (3/25) = 6/25 = 0.24s → speed = 10 / 0.24 * 3.6 = 150 km/h
    positions = [
        (0, 100),   # before A
        (1, 200),
        (2, 280),   # just before A (280 < 300)
        (3, 320),   # just after A (320 > 300) → crossing A at frame 3
        (4, 400),
        (5, 480),
        (6, 550),
        (7, 580),
        (8, 590),   # just before B (590 < 600)
        (9, 620),   # just after B (620 > 600) → crossing B at frame 9
        (10, 700),
    ]

    events = []
    for frame, cx in positions:
        events.extend(estimator.update([make_detection(1, cx, frame)], frame))
    events.extend(estimator.flush())

    assert len(events) == 1
    event = events[0]
    assert event.crossing_a is not None
    assert event.crossing_b is not None
    # crossing A at frame 3, B at frame 9 → elapsed = 6/25 = 0.24s
    expected_speed = (DISTANCE_M / (6 / FPS)) * 3.6
    assert event.speed_kmh == pytest.approx(expected_speed, rel=1e-3)
