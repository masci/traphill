import sys
from dataclasses import dataclass

import click
import cv2
from ultralytics import YOLO  # New import for modern YOLO usage

# YOLO config
YOLO_MODEL = "yolov8n.pt"
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck

# Video config
FPS = 30.0
PIXELS_TO_METERS_FACTOR = 0.05


@dataclass(kw_only=True)
class DetectedObject:
    x: int
    y: int
    width: int
    height: int
    name: str
    conf: float
    centroid: tuple[int, int]


@dataclass(kw_only=True)
class TrackedObject:
    center: tuple[int, int]
    start_x: int
    start_frame: int
    speed_kmh: float | None = None
    detected: bool = True


def get_trap_boundaries(
    vcap: cv2.VideoCapture, area_percentage: int = 75
) -> tuple[int, int]:
    """Given the size of the video, return the (X1, X2) coordinates of the trap area"""
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # get trap area width
    trap_width = int(width / 100 * area_percentage)
    border = (width - trap_width) // 2
    return border, width - border


def get_centroid(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    """Calculates the center point (centroid) of the bounding box."""
    return (x + w // 2, y + h // 2)


def calculate_speed(car_data: TrackedObject, current_frame: int) -> float | None:
    """
    Approximates speed (in Km/h) based on pixels traveled over frames elapsed.
    The speed is calculated for the segment within the trap area.
    """
    try:
        # Distance in pixels traveled
        distance_pixels = abs(car_data.center[0] - car_data.start_x)

        # Frames elapsed between starting line and finishing line
        frames_elapsed = current_frame - car_data.start_frame

        if frames_elapsed <= 0:
            return None  # Cannot divide by zero or zero movement

        # 1. Pixel Speed: distance in pixels / time in seconds
        time_seconds = frames_elapsed / FPS
        pixel_speed_per_second = distance_pixels / time_seconds

        # 2. Real-World Speed (Meters/Second)
        meters_per_second = pixel_speed_per_second * PIXELS_TO_METERS_FACTOR

        # 3. Convert to kmh (Meters/Second * 3.6)
        kmh = meters_per_second * 3.6
        return round(kmh, 1)

    except Exception as e:
        print(f"Error calculating speed: {e}")
        return None


def main(video_path: str, confidence_treshold: float) -> int:
    """Main function to run the video processing pipeline using Ultralytics YOLO."""
    tracked_objects: dict[int, TrackedObject] = {}
    next_object_id = 0
    current_frame_number = 0

    # Load YOLO Model
    try:
        model = YOLO(YOLO_MODEL)
        # Get the list of class names from the loaded model
        class_names = model.names
        print(
            f"YOLO Model loaded. Vehicle classes targeted: {[class_names[i] for i in VEHICLE_CLASS_IDS]}"
        )

    except Exception as e:
        print(f"Error loading YOLO model with Ultralytics: {e}")
        return 1

    # Video Capture Setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}. Check the path.")
        return 1
    trap_area_x1, trap_area_x2 = get_trap_boundaries(cap)

    print("Starting video processing with Ultralytics YOLO...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed reading frame from capture, exiting...")
            break

        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        height, width, _ = frame.shape

        # Predict on the frame (setting verbose=False suppresses logging for cleaner output)
        results = model.predict(
            source=frame,
            conf=confidence_treshold,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )[0]

        trapped_objects: list[DetectedObject] = []

        # Iterate over bounding boxes and process only the detected vehicles
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(box.conf[0].item(), 2)
            cls_id = int(box.cls[0].item())

            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
            centroid_x, centroid_y = get_centroid(x, y, w, h)

            # Filter detections to only those in the speed tracking zone
            if trap_area_x1 <= centroid_x <= trap_area_x2:
                t = DetectedObject(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    name=class_names.get(cls_id, "Unknown"),
                    conf=conf,
                    centroid=(centroid_x, centroid_y),
                )
                trapped_objects.append(t)

        # Mark all existing tracked objects as potentially lost
        for obj_id in tracked_objects:
            tracked_objects[obj_id].detected = False

        # Process the objects detected within the trap area
        for obj in trapped_objects:
            centroid_x, centroid_y = obj.centroid

            # Draw the bounding box on the original frame
            color = (0, 255, 255) if obj.name == "car" else (255, 165, 0)
            cv2.rectangle(
                frame, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), color, 2
            )

            # Try to match the current car to an existing tracked object using proximity
            matched_id = -1
            for obj_id, tracked_obj in tracked_objects.items():
                # Check proximity based on centroid X position
                if abs(tracked_obj.center[0] - centroid_x) < 80:
                    matched_id = obj_id
                    break

            if matched_id == -1:
                # New object detected: Record starting point for speed calculation
                obj_id = next_object_id
                next_object_id += 1
                matched_id = obj_id
                tracked_objects[obj_id] = TrackedObject(
                    center=(centroid_x, centroid_y),
                    start_x=centroid_x,
                    start_frame=current_frame_number,
                )
            else:
                # Existing object updated
                tracked_objects[matched_id].center = (centroid_x, centroid_y)
                tracked_objects[matched_id].detected = True

            car_data = tracked_objects[matched_id]

            # Check if the object has entered the trap area AND its speed hasn't been calculated yet
            if car_data.speed_kmh is None:
                speed = calculate_speed(car_data, current_frame_number)
                tracked_objects[matched_id].speed_kmh = speed

            # Display ID, Class, and Speed
            display_text = f"ID:{matched_id} ({obj.name})"
            if car_data.speed_kmh:
                display_text += f" | {car_data.speed_kmh} Km/h"

            cv2.putText(
                frame,
                display_text,
                (obj.x, obj.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Remove objects that haven't been seen for a while but have completed their track
        objects_to_remove = [
            obj_id
            for obj_id, data in tracked_objects.items()
            if not data.detected and data.speed_kmh is not None
        ]

        for obj_id in objects_to_remove:
            del tracked_objects[obj_id]

        # Draw the Speed Trap Lines for visual reference
        cv2.line(frame, (trap_area_x1, 0), (trap_area_x1, height), (255, 0, 0), 2)
        cv2.line(frame, (trap_area_x2, 0), (trap_area_x2, height), (0, 0, 255), 2)

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
