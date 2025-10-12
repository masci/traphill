import sys

import click
import cv2
from cv2.typing import MatLike
from ultralytics import YOLO

from .config import VEHICLE_CLASS_IDS, YOLO_MODEL
from .detection import get_trap_area, track_vehicles
from .types import Detection, TrapArea, Vehicle


def draw_tracked_objects(
    tracked_vehicles: dict[int, Vehicle], frame: MatLike, current_frame_number: int
):
    for tracking in tracked_vehicles.values():
        vehicle = tracking.current(current_frame_number)
        if vehicle is None:
            # Nothing to draw, tracking objects has just left the trap area
            continue

        color = (0, 255, 255) if vehicle.name == "car" else (255, 165, 0)
        cv2.rectangle(
            frame,
            (vehicle.x, vehicle.y),
            (
                vehicle.x + vehicle.width,
                vehicle.y + vehicle.height,
            ),
            color,
            2,
        )

        # Display ID, Class, and Speed
        display_text = f"ID:{vehicle.id} ({vehicle.name})"
        if tracking.speed:
            display_text += f" | {tracking.speed} Km/h"

        cv2.putText(
            frame,
            display_text,
            (vehicle.x, vehicle.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def draw_speed_trap_area(frame: MatLike, trap_area: TrapArea):
    cv2.line(frame, (trap_area.x1, 0), (trap_area.x1, trap_area.height), (255, 0, 0), 2)
    cv2.line(frame, (trap_area.x2, 0), (trap_area.x2, trap_area.height), (0, 0, 255), 2)


def main(
    video_path: str, confidence_treshold: float, trap_begin: int, trap_end: int | None
) -> int:
    """Main function to run the video processing pipeline using Ultralytics YOLO."""
    current_frame_number = 0
    tracked_vehicles: dict[int, Vehicle] = {}
    deleted_ids: list[int] = []

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

        # Draw the Speed Trap Lines for visual reference
        draw_speed_trap_area(frame, trap_area)

        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        detected_objects: list[Detection] = track_vehicles(
            model, frame, current_frame_number, confidence_treshold, trap_area
        )

        for obj in detected_objects:
            if obj.id in deleted_ids:
                # Remove ghosts
                continue

            if obj.id not in tracked_vehicles:
                print(f"Tracking vehicle {obj.id}")
                tracked_vehicles[obj.id] = Vehicle()
                tracked_vehicles[obj.id].detections.append(obj)
            else:
                tracked_vehicles[obj.id].detections.append(obj)
                tracked_vehicles[obj.id].calculate_speed(current_frame_number, fps)

            to_remove = []
            for vehicle_id, vehicle in tracked_vehicles.items():
                if vehicle.frames_elapsed(current_frame_number) > fps * 5:
                    print(f"Vehicle {vehicle_id} avg speed: {vehicle.speed}")
                    deleted_ids.append(vehicle_id)
                    to_remove.append(vehicle_id)

            for id in to_remove:
                del tracked_vehicles[id]

        # objects_to_remove = [
        #     vehicle_id
        #     for vehicle_id, vehicle in tracked_vehicles.items()
        #     if vehicle.frames_elapsed(current_frame_number) > fps / 2
        #     and trap_area.at_border(vehicle.last_seen)
        # ]

        # for vehicle_id in objects_to_remove:
        #     if tracked_vehicles[vehicle_id].speed:
        #         print(
        #             f"Vehicle {vehicle_id} avg speed: {tracked_vehicles[vehicle_id].speed}"
        #         )
        #     del tracked_vehicles[vehicle_id]
        #     deleted_ids.append(vehicle_id)

        # Draw tracked objects
        draw_tracked_objects(tracked_vehicles, frame, current_frame_number)

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
