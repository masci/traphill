from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TripwireConfig(BaseModel):
    line_a_x: int
    line_b_x: int
    physical_distance_m: float = Field(gt=0)


class CameraConfig(BaseModel):
    id: str
    url: str
    name: str
    confidence_threshold: float = 0.6
    tripwire: TripwireConfig


class AppConfig(BaseModel):
    cameras: list[CameraConfig]
    yolo_model: str = "yolo11n.mnn"
    stale_track_timeout_s: float = 2.0


def load_config(path: str | Path) -> AppConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return AppConfig.model_validate(data)


def save_config(path: str | Path, config: AppConfig) -> None:
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, allow_unicode=True)
