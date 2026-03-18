from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["stream"])


@router.get("/streams/{camera_id}/mjpeg")
async def mjpeg(camera_id: str, request: Request):
    manager = request.app.state.manager
    buf = manager.get_frame_buffer(camera_id)
    if buf is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    async def generate():
        while True:
            if await request.is_disconnected():
                break
            frame_bytes = buf.get_latest()
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            await asyncio.sleep(1 / 30)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.websocket("/streams/{camera_id}/ws")
async def ws_events(websocket: WebSocket, camera_id: str):
    manager = websocket.app.state.manager
    if camera_id not in manager.camera_ids():
        await websocket.close(code=4004)
        return

    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    manager.subscribe(camera_id, queue)

    try:
        while True:
            event_data = await queue.get()
            await websocket.send_text(json.dumps(event_data))
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(camera_id, queue)
