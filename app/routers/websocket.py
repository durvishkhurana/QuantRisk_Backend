import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.services.alerts import socket_manager
from app.services.redis_client import get_redis

router = APIRouter(tags=["websocket"])


def _parse_stream_message(entry_id: str, fields: dict) -> dict | None:
    raw = fields.get("data")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    event_type = payload.get("type", "alert")
    return {
        "stream_id": entry_id,
        "event_type": event_type,
        "data": payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.websocket("/ws/portfolios/{portfolio_id}")
async def portfolio_ws(
    websocket: WebSocket,
    portfolio_id: str,
    since: str = Query("$"),
) -> None:
    await socket_manager.connect(portfolio_id, websocket)
    redis = await get_redis()
    stream_key = f"stream:alerts:{portfolio_id}"
    last_id = since if since else "$"

    try:
        while True:
            streams = await redis.xread({stream_key: last_id}, block=1000, count=10)
            if not streams:
                continue
            for _stream_name, entries in streams:
                for entry_id, fields in entries:
                    message = _parse_stream_message(entry_id, fields)
                    if message:
                        await websocket.send_json(message)
                    last_id = entry_id
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        socket_manager.disconnect(portfolio_id, websocket)
