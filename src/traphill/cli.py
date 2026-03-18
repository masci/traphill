from __future__ import annotations

import json
import sys

import click
import cv2
from ultralytics import YOLO

from traphill.config import load_config
from traphill.engine.tracker import SessionTracker


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("video_path")
@click.option("--camera-id", default=None, help="Camera ID from config to use (default: first)")
@click.option("--display/--no-display", default=False, help="Show annotated video window")
def run(config_path: str, video_path: str, camera_id: str | None, display: bool):
    """Run speed estimation on a video file or stream, printing events as JSON."""
    app_config = load_config(config_path)

    if camera_id:
        cam_configs = [c for c in app_config.cameras if c.id == camera_id]
        if not cam_configs:
            click.echo(f"Camera '{camera_id}' not found in config.", err=True)
            sys.exit(1)
        cam_config = cam_configs[0]
    else:
        cam_config = app_config.cameras[0]

    model = YOLO(app_config.yolo_model)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        click.echo(f"Cannot open video: {video_path}", err=True)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = SessionTracker(
        model=model,
        camera_config=cam_config,
        fps=fps,
        stale_timeout_s=app_config.stale_track_timeout_s,
    )

    frame_number = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, events = tracker.process_frame(frame, frame_number)
            frame_number += 1

            for event in events:
                print(json.dumps(_event_to_dict(event)), flush=True)

            if display:
                cv2.imshow("traphill", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for event in tracker.flush():
            print(json.dumps(_event_to_dict(event)), flush=True)
        cap.release()
        if display:
            cv2.destroyAllWindows()


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--db", default="traphill.db", show_default=True, help="SQLite database path")
def serve(config_path: str, host: str, port: int, db: str):
    """Start the Traphill HTTP service (API + Web UI)."""
    import uvicorn
    from traphill.api.app import create_app
    app = create_app(config_path=config_path, db_path=db)
    uvicorn.run(app, host=host, port=port)


def _event_to_dict(event) -> dict:
    return {
        "tracker_id": event.tracker_id,
        "vehicle_class": event.vehicle_class,
        "direction": event.direction,
        "speed_kmh": round(event.speed_kmh, 2) if event.speed_kmh is not None else None,
        "entered_at_s": round(event.entered_at_s, 3),
        "exited_at_s": round(event.exited_at_s, 3),
        "incomplete": event.incomplete,
    }
