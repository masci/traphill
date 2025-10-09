import cv2
from cv2.typing import MatLike
from ultralytics import YOLO

from .config import VEHICLE_CLASS_IDS
from .types import Detection, TrapArea


def get_trap_area(
    vcap: cv2.VideoCapture, trap_begin: int, trap_end: int | None
) -> TrapArea:
    """Given the size of the video, return the trap area"""
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1 = max(trap_begin, 0)
    x2 = trap_end or width
    return TrapArea(x1, x2, height)


def detect_objects(
    model: YOLO,
    frame: MatLike,
    confidence_treshold: float,
    trap_area: TrapArea,
) -> list[Detection]:
    """Detect objects and return those within the trap area."""
    retval: list[Detection] = []
    results = model.predict(
        source=frame,
        conf=confidence_treshold,
        classes=VEHICLE_CLASS_IDS,
        verbose=False,  # suppress logging for cleaner output
    )[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = round(box.conf[0].item(), 2)
        cls_id = int(box.cls[0].item())
        dt = Detection(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            name=model.names.get(cls_id, "Unknown"),
            conf=conf,
        )
        centroid_x, _ = dt.centroid

        # Filter detections to only include those within the speed tracking zone
        if trap_area.x1 <= centroid_x <= trap_area.x2:
            retval.append(dt)
    return retval
