from __future__ import annotations

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData()

vehicle_events = Table(
    "vehicle_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("camera_id", String, nullable=False, index=True),
    Column("tracker_id", Integer, nullable=False),
    Column("vehicle_class", String, nullable=False),
    Column("direction", String, nullable=True),       # left_to_right | right_to_left | null
    Column("speed_kmh", Float, nullable=True),        # null if incomplete
    Column("entered_at_s", Float, nullable=False),
    Column("exited_at_s", Float, nullable=False),
    Column("incomplete", Integer, nullable=False),    # 0 or 1
    Column("recorded_at", Text, nullable=False),      # ISO8601 wall clock
    Column("wire_a_frame", Integer, nullable=True),
    Column("wire_b_frame", Integer, nullable=True),
)

cameras = Table(
    "cameras",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("url", Text, nullable=False),
    Column("line_a_x", Integer, nullable=False),
    Column("line_b_x", Integer, nullable=False),
    Column("physical_distance_m", Float, nullable=False),
    Column("created_at", Text, nullable=False),
)
