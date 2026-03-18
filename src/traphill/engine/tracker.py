from __future__ import annotations

import numpy as np

from traphill.config import CameraConfig
from traphill.engine.detector import detect
from traphill.engine.estimator import SpeedEstimator
from traphill.engine.types import VehicleEvent


class SessionTracker:
    """Coordinates detection + speed estimation for one camera session."""

    def __init__(self, model, camera_config: CameraConfig, fps: float, stale_timeout_s: float = 2.0):
        self._model = model
        self._config = camera_config
        self._fps = fps
        self._estimator = SpeedEstimator(
            config=_to_engine_tripwire(camera_config),
            fps=fps,
            stale_timeout_s=stale_timeout_s,
        )

    def process_frame(self, frame: np.ndarray, frame_number: int) -> tuple[np.ndarray, list[VehicleEvent]]:
        """
        Run detection + speed estimation on one frame.
        Returns the annotated frame and any VehicleEvents that completed this frame.
        """
        detections = detect(
            self._model,
            frame,
            frame_number,
            self._fps,
            self._config.confidence_threshold,
        )

        events = self._estimator.update(detections, frame_number)
        annotated = _annotate(frame, detections, self._config, events)
        return annotated, events

    def flush(self) -> list[VehicleEvent]:
        """Finalize all active tracks at end of stream."""
        return self._estimator.flush()


def _to_engine_tripwire(camera_config: CameraConfig):
    from traphill.engine.types import TripwireConfig as EngineTripwireConfig
    tw = camera_config.tripwire
    return EngineTripwireConfig(
        line_a_x=tw.line_a_x,
        line_b_x=tw.line_b_x,
        physical_distance_m=tw.physical_distance_m,
    )


def _annotate(frame: np.ndarray, detections, camera_config: CameraConfig, events) -> np.ndarray:
    import cv2

    out = frame.copy()
    h = out.shape[0]
    tw = camera_config.tripwire

    # Draw tripwires
    cv2.line(out, (tw.line_a_x, 0), (tw.line_a_x, h), (0, 255, 0), 2)
    cv2.line(out, (tw.line_b_x, 0), (tw.line_b_x, h), (0, 0, 255), 2)
    cv2.putText(out, "A", (tw.line_a_x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(out, "B", (tw.line_b_x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw bounding boxes
    for det in detections:
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), (255, 200, 0), 2)
        cv2.putText(
            out,
            f"{det.class_name} #{det.tracker_id}",
            (det.x1, det.y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
        )

    # Draw speed for completed events
    for event in events:
        if event.speed_kmh is not None:
            label = f"{event.speed_kmh:.1f} km/h"
            ref = event.crossing_b or event.crossing_a
            if ref:
                cx, cy = ref.centroid
                cv2.putText(out, label, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return out
