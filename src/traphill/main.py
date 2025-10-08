import sys

import click
import cv2
import numpy as np
from cv2.typing import MatLike
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from .types import Car, Detection, TrapArea

# YOLO config
YOLO_MODEL = "yolov8n.pt"
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck

# Video config
PIXELS_TO_METERS_FACTOR = 0.05


def get_trap_area(vcap: cv2.VideoCapture, area_percentage: int = 40) -> TrapArea:
    """Given the size of the video, return the trap area"""
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    trap_width = int(width / 100 * area_percentage)
    border = (width - trap_width) // 2
    return TrapArea(border, width - border, height)


def calculate_speed(car: Car, current_frame: int, fps: float) -> float | None:
    """
    Approximates speed (in Km/h) based on pixels traveled over frames elapsed.
    The speed is calculated for the segment within the trap area.
    """
    try:
        distance_pixels = car.travelled_distance()
        frames_elapsed = car.frames_elapsed(current_frame)
        if frames_elapsed <= 0:
            return None  # Cannot divide by zero or zero movement

        # 1. Pixel Speed: distance in pixels / time in seconds
        time_seconds = frames_elapsed / fps
        pixel_speed_per_second = distance_pixels / time_seconds

        # 2. Real-World Speed (Meters/Second)
        meters_per_second = pixel_speed_per_second * PIXELS_TO_METERS_FACTOR

        # 3. Convert to kmh (Meters/Second * 3.6)
        kmh = meters_per_second * 3.6
        return round(kmh, 1)

    except Exception as e:
        print(f"Error calculating speed: {e}")
        return None


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


def main(video_path: str, confidence_treshold: float) -> int:
    """Main function to run the video processing pipeline using Ultralytics YOLO."""
    tracked_cars: dict[int, Car] = {}
    next_car_id = 0
    current_frame_number = 0

    # Load YOLO Model
    try:
        model = YOLO(YOLO_MODEL)
        # Get the list of class names from the loaded model
        print(
            f"YOLO Model loaded. Vehicle classes targeted: {[model.names[i] for i in VEHICLE_CLASS_IDS]}"
        )

    except Exception as e:
        print(f"Error loading YOLO model with Ultralytics: {e}")
        return 1

    # Video Capture Setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}. Check the path.")
        return 1
    trap_area = get_trap_area(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Starting video processing with Ultralytics YOLO...")

    # Process frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed reading frame from capture, exiting...")
            break

        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        detected_objects: list[Detection] = detect_objects(
            model, frame, confidence_treshold, trap_area
        )

        # Mark all existing tracked objects as potentially lost
        for car_id in tracked_cars:
            tracked_cars[car_id]._detected = False

        # If there are no detections, there's nothing to do
        if len(detected_objects) == 0:
            # TODO: if we want to start/stop recording when there's no traffic,
            # we could change state here
            pass
        # If there are no tracked objects, all detections are new
        elif len(tracked_cars) == 0:
            for obj in detected_objects:
                tracked_cars[next_car_id] = Car(
                    detection=obj, frame_number=current_frame_number
                )
                next_car_id += 1
        else:
            # Prepare cost matrix for assignment problem
            #
            # Rows: existing tracked objects
            # Columns: new detected objects
            # Value: Euclidean distance
            car_ids = list(tracked_cars.keys())
            cost_matrix = np.zeros(
                (len(car_ids), len(detected_objects)), dtype=np.float32
            )

            for i, car_id in enumerate(car_ids):
                for j, det_obj in enumerate(detected_objects):
                    cost_matrix[i, j] = det_obj.distance(tracked_cars[car_id].detection)

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Process assignments
            for r, c in zip(row_ind, col_ind):
                car_id = car_ids[r]
                det_obj = detected_objects[c]

                # Check if the match is good enough (e.g., distance threshold)
                if cost_matrix[r, c] < 100:
                    tracked_cars[car_id].update(det_obj, current_frame_number)

            # New objects are those that were not assigned
            assigned_det_indices = set(col_ind)
            for i, det_obj in enumerate(detected_objects):
                if i not in assigned_det_indices:
                    tracked_cars[next_car_id] = Car(
                        detection=det_obj, frame_number=current_frame_number
                    )
                    next_car_id += 1

        # Draw tracked objects
        for car_id, car in tracked_cars.items():
            if not car.detected:
                continue

            # Draw the bounding box on the original frame
            color = (0, 255, 255) if car.detection.name == "car" else (255, 165, 0)
            cv2.rectangle(
                frame,
                (car.detection.x, car.detection.y),
                (
                    car.detection.x + car.detection.width,
                    car.detection.y + car.detection.height,
                ),
                color,
                2,
            )

            speed = calculate_speed(car, current_frame_number, fps)
            if speed is not None:
                tracked_cars[car_id]._speed = speed

            # Display ID, Class, and Speed
            display_text = f"ID:{car_id} ({car.detection.name})"
            if car._speed:
                display_text += f" | {car._speed} Km/h"

            cv2.putText(
                frame,
                display_text,
                (car.detection.x, car.detection.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Remove objects that haven't been seen for a while
        #
        # This is a simple approach. A more robust solution would be to
        # keep the object for a bit longer, in case it reappears.
        objects_to_remove = [
            car_id
            for car_id, car in tracked_cars.items()
            if not car.detected and car.frames_elapsed(current_frame_number) > fps / 2
        ]

        for car_id in objects_to_remove:
            del tracked_cars[car_id]

        # Draw the Speed Trap Lines for visual reference
        cv2.line(
            frame, (trap_area.x1, 0), (trap_area.x1, trap_area.height), (255, 0, 0), 2
        )
        cv2.line(
            frame, (trap_area.x2, 0), (trap_area.x2, trap_area.height), (0, 0, 255), 2
        )

        # Show the result
        cv2.imshow("YOLO Car Tracker and Speed Estimator (Ultralytics)", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


@click.command(
    help="Track cars and estimate their speed in a video.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--confidence-treshold",
    type=float,
    default=0.6,
    help="Minimum confidence to consider a detection",
)
@click.argument(
    "video_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
def cli(video_path: str, confidence_treshold: float):
    """CLI for the YOLO Car Tracker and Speed Estimator."""
    sys.exit(main(video_path, confidence_treshold))
