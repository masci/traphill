from __future__ import annotations

import numpy as np

from traphill.engine.types import Detection

# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def detect(
    model,
    frame: np.ndarray,
    frame_number: int,
    fps: float,
    confidence_threshold: float = 0.6,
) -> list[Detection]:
    """Run YOLO tracking on a frame and return vehicle detections."""
    results = model.track(frame, persist=True, verbose=False, conf=confidence_threshold)

    detections: list[Detection] = []
    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    if boxes.id is None:
        return detections

    timestamp_s = frame_number / fps if fps > 0 else 0.0

    for box, tracker_id, class_id, conf in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.id.cpu().numpy().astype(int),
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy(),
    ):
        if class_id not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        detections.append(
            Detection(
                tracker_id=int(tracker_id),
                class_id=class_id,
                class_name=VEHICLE_CLASS_NAMES[class_id],
                confidence=float(conf),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_number=frame_number,
                timestamp_s=timestamp_s,
            )
        )

    return detections
