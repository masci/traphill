"""
Test that direction is correctly determined from crossing order.
"""
from traphill.engine.estimator import SpeedEstimator
from traphill.engine.types import Detection, TripwireConfig

LINE_A = 300
LINE_B = 600
FPS = 25.0


def make_config():
    return TripwireConfig(line_a_x=LINE_A, line_b_x=LINE_B, physical_distance_m=10.0)


def make_detection(tracker_id: int, cx: int, frame: int) -> Detection:
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
        frame_number=frame,
        timestamp_s=frame / FPS,
    )


def test_left_to_right():
    estimator = SpeedEstimator(make_config(), FPS)
    # Cross A then B
    for frame, cx in enumerate([100, 200, 320, 450, 620, 700]):
        estimator.update([make_detection(1, cx, frame)], frame)
    events = estimator.flush()
    assert events[0].direction == "left_to_right"


def test_right_to_left():
    estimator = SpeedEstimator(make_config(), FPS)
    # Cross B then A (moving right-to-left)
    for frame, cx in enumerate([700, 620, 450, 320, 200, 100]):
        estimator.update([make_detection(1, cx, frame)], frame)
    events = estimator.flush()
    assert events[0].direction == "right_to_left"


def test_two_vehicles_different_directions():
    estimator = SpeedEstimator(make_config(), FPS)

    positions_ltr = [100, 200, 320, 450, 620, 700]
    positions_rtl = [700, 620, 450, 320, 200, 100]

    for frame in range(len(positions_ltr)):
        dets = [
            make_detection(1, positions_ltr[frame], frame),
            make_detection(2, positions_rtl[frame], frame),
        ]
        estimator.update(dets, frame)

    events = estimator.flush()
    by_id = {e.tracker_id: e for e in events}
    assert by_id[1].direction == "left_to_right"
    assert by_id[2].direction == "right_to_left"
