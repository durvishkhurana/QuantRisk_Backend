import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_user_from_token
from app.database import get_db
from app.models import Portfolio
from app.services.alerts import socket_manager
from app.services.redis_client import get_redis

router = APIRouter(tags=["websocket"])

# WebSocket close codes (4000-4999 is the application-private range).
WS_UNAUTHORIZED = 4401
WS_FORBIDDEN = 4403


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
    token: str = Query(default=""),
    db: AsyncSession = Depends(get_db),
) -> None:
    # Browsers can't send Authorization headers on a WebSocket handshake, so the
    # JWT is passed as a query param. Authenticate and authorize ownership before
    # accepting the socket — otherwise any portfolio UUID could be subscribed to.
    user = await get_user_from_token(token, db) if token else None
    if not user:
        await websocket.close(code=WS_UNAUTHORIZED)
        return
    owns = (
        await db.execute(
            select(Portfolio.id).where(Portfolio.id == portfolio_id, Portfolio.user_id == user.id).limit(1)
        )
    ).scalar_one_or_none()
    if not owns:
        await websocket.close(code=WS_FORBIDDEN)
        return

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
