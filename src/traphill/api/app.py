from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from traphill.api.routers import cameras, events, stream
from traphill.config import load_config
from traphill.database import get_engine, init_db
from traphill.storage.events import insert_event
from traphill.stream.manager import CameraManager

logger = logging.getLogger(__name__)


def create_app(config_path: str, db_path: str = "traphill.db") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_config = load_config(config_path)
        engine = get_engine(db_path)
        init_db(engine)

        loop = asyncio.get_event_loop()
        manager = CameraManager(
            app_config=app_config,
            on_event=lambda cid, ev: insert_event(engine, cid, ev),
            event_loop=loop,
        )
        manager.start()

        app.state.config_path = config_path
        app.state.app_config = app_config
        app.state.engine = engine
        app.state.manager = manager

        logger.info("Traphill service started with %d camera(s)", len(app_config.cameras))
        yield

        manager.stop_all()
        engine.dispose()
        logger.info("Traphill service stopped")

    app = FastAPI(title="Traphill", version="2.0.0", lifespan=lifespan)

    app.include_router(cameras.router)
    app.include_router(events.router)
    app.include_router(stream.router)

    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app
