import json
import logging
import time
from datetime import datetime
from typing import Optional

from redis.asyncio import Redis

from .redis_keys import RedisKeys

logger = logging.getLogger("mlup")


class LeaderTasks:
    """
    Leader housekeeping:
      - remove dead workers from heartbeats
      - cleanup stale in_progress statuses (dead PID)
      - requeue stale inflight jobs
    """

    def __init__(
        self,
        *,
        redis: Redis,
        keys: RedisKeys,
        queue_name: str,
        inflight_zset_name: str,
        inflight_hash_name: str,
        processing_list_name: str,
        worker_ttl_s: int,
        inflight_stale_s: int,
        ttl_predicted_data: int,
        meta_ttl_s: int,
        leader_cleanup_every_s: float,
    ):
        self.redis = redis
        self.keys = keys
        self.queue_name = queue_name
        self.inflight_zset_name = inflight_zset_name
        self.inflight_hash_name = inflight_hash_name
        self.processing_list_name = processing_list_name

        self.worker_ttl_s = worker_ttl_s
        self.inflight_stale_s = inflight_stale_s
        self.ttl_predicted_data = ttl_predicted_data
        self.meta_ttl_s = meta_ttl_s
        self.leader_cleanup_every_s = leader_cleanup_every_s

    async def run_once(self) -> None:
        removed_workers = await self._remove_inactive_workers()
        removed_in_progress = await self._cleanup_stale_in_progress_statuses()
        requeued = await self._requeue_stale_inflight()

        logger.info(
            "Leader duties completed | time=%sZ | removed_workers=%d | removed_in_progress=%d | requeued_inflight=%d",
            datetime.utcnow().isoformat(),
            removed_workers,
            removed_in_progress,
            requeued,
        )

    async def _remove_inactive_workers(self) -> int:
        workers = await self.redis.hgetall(self.keys.workers_hash)
        if not workers:
            return 0

        removed = 0
        now = time.time()

        for uniq_key, raw in workers.items():
            try:
                data = json.loads(raw)
                ts = float(data.get("timestamp", 0))
                if now - ts > self.worker_ttl_s:
                    await self.redis.hdel(self.keys.workers_hash, uniq_key)
                    removed += 1
                    logger.info("Leader removed dead worker | key=%s | last_seen_age=%.1fs", uniq_key, (now - ts))
            except Exception:
                await self.redis.hdel(self.keys.workers_hash, uniq_key)
                removed += 1

        return removed

    async def _cleanup_stale_in_progress_statuses(self) -> int:
        """
        IMPORTANT:
          - never touch 'wait' statuses (API creates them)
          - only remove 'in_progress' if pid is dead
        """
        workers = await self.redis.hgetall(self.keys.workers_hash)
        active_pids = set()
        now = time.time()

        for _, raw in workers.items():
            try:
                wd = json.loads(raw)
                ts = float(wd.get("timestamp", 0))
                if now - ts <= self.worker_ttl_s:
                    active_pids.add(int(wd.get("pid")))
            except Exception:
                continue

        removed = 0
        pattern = f"{self.keys.status_prefix}*"
        cursor = 0

        while True:
            cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=500)
            for key in keys:
                raw = await self.redis.get(key)
                if not raw:
                    continue
                try:
                    sd = json.loads(raw)
                    status = sd.get("status")
                    pid = sd.get("pid")

                    if status == "in_progress" and pid is not None and int(pid) not in active_pids:
                        await self.redis.delete(key)
                        removed += 1
                        logger.info("Leader removed stale in_progress | key=%s | dead_pid=%s", key, pid)
                except Exception:
                    await self.redis.delete(key)
                    removed += 1

            if cursor == 0:
                break

        return removed

    async def _requeue_stale_inflight(self) -> int:
        now = time.time()
        cutoff = now - self.inflight_stale_s

        stale_ids = await self.redis.zrangebyscore(self.inflight_zset_name, min=0, max=cutoff)
        if not stale_ids:
            return 0

        requeued = 0
        for predict_id in stale_ids:
            payload = await self.redis.hget(self.inflight_hash_name, predict_id)
            if not payload:
                await self.redis.zrem(self.inflight_zset_name, predict_id)
                await self.redis.hdel(self.inflight_hash_name, predict_id)
                continue

            # Requeue back to main queue
            await self.redis.lpush(self.queue_name, payload)

            # Remove from processing/inflight
            await self.redis.lrem(self.processing_list_name, 1, payload)
            await self.redis.hdel(self.inflight_hash_name, predict_id)
            await self.redis.zrem(self.inflight_zset_name, predict_id)

            requeued += 1
            logger.info("Leader requeued stale inflight | predict_id=%s", predict_id)

        return requeued
