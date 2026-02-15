import asyncio
import logging
from typing import Optional

from redis.asyncio import Redis

logger = logging.getLogger("mlup")


class RedisConnector:
    """
    Helper to manage Redis connection lifecycle:
      - connect()
      - ensure() (ping + reconnect if needed)
      - reconnect_forever() while running
    """

    def __init__(self, redis_url: str, *, decode_responses: bool = True):
        self.redis_url = redis_url
        self.decode_responses = decode_responses
        self.client: Optional[Redis] = None

    async def connect(self) -> Redis:
        self.client = Redis.from_url(self.redis_url, decode_responses=self.decode_responses)
        await self.client.ping()
        return self.client

    async def reconnect_forever(self, *, is_running_fn, who: str = "") -> None:
        backoff = 1.0
        while is_running_fn():
            try:
                logger.warning("%s Reconnecting to Redis: %s", who, self.redis_url)
                self.client = Redis.from_url(self.redis_url, decode_responses=self.decode_responses)
                await self.client.ping()
                logger.info("%s Redis reconnect successful.", who)
                return
            except Exception as e:
                logger.warning("%s Redis reconnect failed: %s (retry in %.1fs)", who, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    async def ensure(self, *, is_running_fn, who: str = "") -> Redis:
        if not self.client:
            await self.reconnect_forever(is_running_fn=is_running_fn, who=who)
            return self.client  # type: ignore

        try:
            await self.client.ping()
            return self.client
        except Exception:
            await self.reconnect_forever(is_running_fn=is_running_fn, who=who)
            return self.client  # type: ignore

    async def close(self) -> None:
        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass
            self.client = None
