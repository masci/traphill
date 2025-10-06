import sys

import cv2
from ultralytics import YOLO  # New import for modern YOLO usage

# YOLO config
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to consider a detection
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Global config
VIDEO_PATH = "cars_passing.mp4"
FPS = 30.0
PIXELS_TO_METERS_FACTOR = 0.05

# Global tracking data
# {id: {'center': (x, y), 'start_y': y_coord, 'start_frame': frame_number, 'speed_kmh': None, 'detected': True}}
tracked_objects = {}
next_object_id = 0
current_frame_number = 0


def get_trap_boundaries(vcap, area_percentage: int = 75) -> tuple[int, int]:
    """Given the size of the video, return the (X1, X2) coordinates of the trap area"""
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # get trap area width
    trap_width = int(width / 100 * area_percentage)
    border = (width - trap_width) // 2
    return border, width - border


def get_centroid(x, y, w, h):
    """Calculates the center point (centroid) of the bounding box."""
    return (x + w // 2, y + h // 2)


def calculate_speed(car_data, current_frame):
    """
    Approximates speed (kmh) based on pixels traveled over frames elapsed.
    The speed is calculated for the segment between SPEED_LINE_Y1 and SPEED_LINE_Y2.
    """
    try:
        # Distance in pixels traveled
        distance_pixels = abs(car_data["center"][0] - car_data["start_x"])

        # Frames elapsed between starting line and finishing line
        frames_elapsed = current_frame - car_data["start_frame"]

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


def main():
    """Main function to run the video processing pipeline using Ultralytics YOLO."""
    global next_object_id, current_frame_number, tracked_objects

    # Load YOLO Model
    try:
        # Ultralytics automatically handles model downloading and loading
        model = YOLO(YOLO_MODEL)
        class_names = model.names  # Get the list of class names from the loaded model
        print(
            f"YOLO Model loaded. Vehicle classes targeted: {[class_names[i] for i in VEHICLE_CLASS_IDS]}"
        )

    except Exception as e:
        print(f"Error loading YOLO model with Ultralytics: {e}")
        return 1

    # Video Capture Setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}. Check the path.")
        return 1

    print("Starting video processing with Ultralytics YOLO...")

    trap_area_x1, trap_area_x2 = get_trap_boundaries(cap)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        height, width, _ = frame.shape

        # Predict on the frame (setting verbose=False suppresses logging for cleaner output)
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )[0]

        final_detections = []

        # Iterate over bounding boxes and process only the detected vehicles
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(box.conf[0].item(), 2)
            cls_id = int(box.cls[0].item())

            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1

            # Filter detections to only those in the speed tracking zone
            if x + w > trap_area_x1:
                final_detections.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "class": class_names.get(cls_id, "Unknown"),
                        "conf": conf,
                    }
                )

        # Mark all existing objects as potentially lost
        for obj_id in tracked_objects:
            tracked_objects[obj_id]["detected"] = False

        for det in final_detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]

            centroid_x, centroid_y = get_centroid(x, y, w, h)

            # Draw the bounding box on the original frame
            class_name = det["class"]
            color = (0, 255, 255) if class_name == "car" else (255, 165, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Try to match the current car to an existing tracked object using proximity
            matched_id = -1
            for obj_id, data in tracked_objects.items():
                # Check proximity based on centroid X position
                if abs(data["center"][0] - centroid_x) < 80:
                    matched_id = obj_id
                    break

            if matched_id == -1:
                # New object detected: Record starting point for speed calculation
                obj_id = next_object_id
                next_object_id += 1
                matched_id = obj_id
                tracked_objects[obj_id] = {
                    "center": (centroid_x, centroid_y),
                    "start_x": centroid_x,
                    "start_frame": current_frame_number,
                    "speed_kmh": None,
                    "detected": True,
                }
            else:
                # Existing object updated
                tracked_objects[matched_id]["center"] = (centroid_x, centroid_y)
                tracked_objects[matched_id]["detected"] = True

            car_data = tracked_objects[matched_id]

            # Check if the object has left the trap area AND its speed hasn't been calculated yet
            if (
                trap_area_x1 <= centroid_x <= trap_area_x2
                and car_data["speed_kmh"] is None
            ):
                speed = calculate_speed(car_data, current_frame_number)
                tracked_objects[matched_id]["speed_kmh"] = speed

            # Display ID, Class, and Speed
            display_text = f"ID:{matched_id} ({class_name})"
            if car_data["speed_kmh"]:
                display_text += f" | {car_data['speed_kmh']} Km/h"

            cv2.putText(
                frame,
                display_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Remove objects that haven't been seen for a while but have completed their track
        objects_to_remove = [
            obj_id
            for obj_id, data in tracked_objects.items()
            if not data["detected"] and data["speed_kmh"] is not None
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


if __name__ == "__main__":
    sys.exit(main())
