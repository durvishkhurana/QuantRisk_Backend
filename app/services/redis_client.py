import json
from redis.asyncio import Redis
from app.config import get_settings

settings = get_settings()
_redis: Redis | None = None


async def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def cache_set_json(key: str, value: dict, ttl_seconds: int) -> None:
    redis = await get_redis()
    await redis.set(key, json.dumps(value), ex=ttl_seconds)


async def cache_get_json(key: str) -> dict | None:
    redis = await get_redis()
    raw = await redis.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def publish_json(channel: str, payload: dict) -> None:
    """Publish alert payloads to a Redis Stream (replaces legacy pub/sub)."""
    redis = await get_redis()
    stream_key = channel if channel.startswith("stream:") else channel.replace("alerts:", "stream:alerts:")
    await redis.xadd(stream_key, {"data": json.dumps(payload)}, maxlen=1000)
