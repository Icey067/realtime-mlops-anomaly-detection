"""Process-wide Redis client shared by API routes."""

from __future__ import annotations

from functools import lru_cache

import redis.asyncio as aioredis
from src.config import get_settings


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    """Return a lazily created Redis client using the application connection settings."""
    settings = get_settings()
    return aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=10,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
    )


async def close_redis_client() -> None:
    """Close the shared pool during application shutdown."""
    if get_redis_client.cache_info().currsize:
        await get_redis_client().aclose()
        get_redis_client.cache_clear()
