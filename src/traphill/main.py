import sys

import click
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from .config import PIXELS_TO_METERS_FACTOR, VEHICLE_CLASS_IDS, YOLO_MODEL
from .detection import detect_objects, get_trap_area
from .types import Detection, Vehicle


def calculate_speed(vehicle: Vehicle, current_frame: int, fps: float) -> float | None:
    """
    Approximates speed (in Km/h) based on pixels traveled over frames elapsed.
    The speed is calculated as the average to cross the trap area.
    """
    try:
        distance_pixels = vehicle.travelled_distance()
        frames_elapsed = vehicle.frames_elapsed(current_frame)
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


def main(
    video_path: str, confidence_treshold: float, trap_begin: int, trap_end: int | None
) -> int:
    """Main function to run the video processing pipeline using Ultralytics YOLO."""
    tracked_vehicles: dict[int, Vehicle] = {}
    next_id = 0
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
    trap_area = get_trap_area(cap, trap_begin, trap_end)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Starting video processing...")

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
        for vehicle_id in tracked_vehicles:
            tracked_vehicles[vehicle_id]._detected = False

        # If there are no detections, there's nothing to do
        if len(detected_objects) == 0:
            # TODO: if we want to start/stop recording when there's no traffic,
            # we could change state here
            pass
        # If there are no tracked objects, all detections are new
        elif len(tracked_vehicles) == 0:
            for obj in detected_objects:
                tracked_vehicles[next_id] = Vehicle(
                    detection=obj, frame_number=current_frame_number
                )
                next_id += 1
        else:
            # Prepare cost matrix for assignment problem
            #
            # Rows: existing tracked objects
            # Columns: new detected objects
            # Value: Euclidean distance
            vehicle_ids = list(tracked_vehicles.keys())
            cost_matrix = np.zeros(
                (len(vehicle_ids), len(detected_objects)), dtype=np.float32
            )

            for i, vehicle_id in enumerate(vehicle_ids):
                for j, det_obj in enumerate(detected_objects):
                    cost_matrix[i, j] = det_obj.distance(
                        tracked_vehicles[vehicle_id].detection
                    )

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Process assignments
            for r, c in zip(row_ind, col_ind):
                vehicle_id = vehicle_ids[r]
                det_obj = detected_objects[c]

                # Check if the match is good enough (e.g., distance threshold)
                if cost_matrix[r, c] < 10:
                    tracked_vehicles[vehicle_id].update(det_obj, current_frame_number)

            # New objects are those that were not assigned
            assigned_det_indices = set(col_ind)
            for i, det_obj in enumerate(detected_objects):
                if i not in assigned_det_indices:
                    tracked_vehicles[next_id] = Vehicle(
                        detection=det_obj, frame_number=current_frame_number
                    )
                    next_id += 1

        # Draw tracked objects
        for vehicle_id, vehicle in tracked_vehicles.items():
            if not vehicle.detected:
                continue

            # Draw the bounding box on the original frame
            color = (0, 255, 255) if vehicle.detection.name == "car" else (255, 165, 0)
            cv2.rectangle(
                frame,
                (vehicle.detection.x, vehicle.detection.y),
                (
                    vehicle.detection.x + vehicle.detection.width,
                    vehicle.detection.y + vehicle.detection.height,
                ),
                color,
                2,
            )

            speed = calculate_speed(vehicle, current_frame_number, fps)
            if speed is not None:
                tracked_vehicles[vehicle_id]._speed = speed

            # Display ID, Class, and Speed
            display_text = f"ID:{vehicle_id} ({vehicle.detection.name})"
            if vehicle._speed:
                display_text += f" | {vehicle._speed} Km/h"

            cv2.putText(
                frame,
                display_text,
                (vehicle.detection.x, vehicle.detection.y - 10),
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
            vehicle_id
            for vehicle_id, vehicle in tracked_vehicles.items()
            if not vehicle.detected
            and vehicle.frames_elapsed(current_frame_number) > fps / 2
        ]

        for vehicle_id in objects_to_remove:
            print(
                f"Vehicle {vehicle_id} avg speed: {tracked_vehicles[vehicle_id]._speed}"
            )
            del tracked_vehicles[vehicle_id]

        # Draw the Speed Trap Lines for visual reference
        cv2.line(
            frame, (trap_area.x1, 0), (trap_area.x1, trap_area.height), (255, 0, 0), 2
        )
        cv2.line(
            frame, (trap_area.x2, 0), (trap_area.x2, trap_area.height), (0, 0, 255), 2
        )

        # Show the result
        cv2.imshow("YOLO Vehicle Tracker and Speed Estimator (Ultralytics)", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


@click.command(
    help="Track vehicles and estimate their speed in a video.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--confidence-treshold",
    type=float,
    default=0.6,
    help="Minimum confidence to consider a detection",
)
@click.option(
    "--trap-begin",
    type=int,
    default=0,
    help="X coordinate in pixels of the left bound of the trap area",
)
@click.option(
    "--trap-end",
    type=int,
    default=None,
    help="X coordinate in pixels of the right bound of the trap area",
)
@click.argument(
    "video_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
def cli(
    video_path: str, confidence_treshold: float, trap_begin: int, trap_end: int | None
):
    """CLI for the YOLO Vehicle Tracker and Speed Estimator."""
    sys.exit(main(video_path, confidence_treshold, trap_begin, trap_end))
