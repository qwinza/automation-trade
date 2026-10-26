"""Redis asynchronous client configuration and lifecycle management for Hermes.

Provides an asynchronous Redis client instance using redis.asyncio,
configured for caching, rate limiting, and concurrency locks.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
import redis.asyncio as aioredis

load_dotenv(override=True)

logger = logging.getLogger("hermes.db.redis")

# Redis configuration from environment variables with fallback defaults
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD") or None
REDIS_URL: Optional[str] = os.getenv("REDIS_URL") or None

# Initialize async Redis client
if REDIS_URL:
    redis_client: aioredis.Redis = aioredis.from_url(
        REDIS_URL,
        decode_responses=True,
    )
else:
    redis_client = aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )


def get_signal_lock_key(asset: str, timeframe: str) -> str:
    """Generate unique Redis lock key for an asset and timeframe.

    Example:
        get_signal_lock_key("XAUUSD", "M15") -> "signal_lock:XAUUSD:M15"
    """
    return f"signal_lock:{asset}:{timeframe}"


async def close_redis() -> None:
    """Safely close the Redis client connection and release connection pool resources."""
    if redis_client is not None:
        try:
            logger.info("Closing Redis connection...")
            await redis_client.close()
            logger.info("Redis connection closed successfully.")
        except Exception as exc:
            logger.warning("Error while closing Redis connection: %s", exc)
