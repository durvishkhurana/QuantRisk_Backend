from __future__ import annotations

import json
import uuid

from app.services.redis_client import get_redis

TASK_KEY_PREFIX = "task:"
TASK_TTL_SECONDS = 24 * 60 * 60

VALID_STATUSES = frozenset({"PENDING", "RUNNING", "SUCCESS", "FAILED"})


def _task_key(task_id: str) -> str:
    return f"{TASK_KEY_PREFIX}{task_id}"


async def set_task_status(
    task_id: str,
    status: str,
    *,
    result: dict | None = None,
    error: str | None = None,
    user_id: str | None = None,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid task status: {status}")
    redis = await get_redis()
    # Preserve the owning user_id across status transitions even if a later
    # call omits it (only the creating call is required to set it).
    if user_id is None:
        existing = await get_task_status(task_id)
        if existing:
            user_id = existing.get("user_id")
    payload = {"status": status, "result": result, "error": error, "user_id": user_id}
    await redis.set(_task_key(task_id), json.dumps(payload), ex=TASK_TTL_SECONDS)


async def get_task_status(task_id: str) -> dict | None:
    redis = await get_redis()
    raw = await redis.get(_task_key(task_id))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "status" not in data:
        return None
    return {
        "status": data["status"],
        "result": data.get("result"),
        "error": data.get("error"),
        "user_id": data.get("user_id"),
    }


def new_task_id() -> str:
    return str(uuid.uuid4())
