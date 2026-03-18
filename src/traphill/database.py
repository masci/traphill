from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from traphill.models import metadata


def get_engine(db_path: str = "traphill.db") -> Engine:
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
    return engine


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
