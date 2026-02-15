import json
import os
import time
from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from redis.asyncio import Redis

# One registry per process
METRICS_REGISTRY = CollectorRegistry(auto_describe=True)
APP_NAME = os.getenv("METRICS_APP_NAME", "mlup_redis_queue")

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["app", "method", "route", "status"],
    registry=METRICS_REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency (seconds)",
    ["app", "method", "route"],
    registry=METRICS_REGISTRY,
)

QUEUE_LENGTH = Gauge(
    "mlup_queue_length",
    "Redis queue length",
    ["app", "queue"],
    registry=METRICS_REGISTRY,
)

INFLIGHT_COUNT = Gauge(
    "mlup_inflight",
    "Number of inflight jobs (claimed, not acked yet)",
    ["app", "queue"],
    registry=METRICS_REGISTRY,
)

WORKERS_ACTIVE = Gauge(
    "mlup_workers_active",
    "Number of active workers (by heartbeat)",
    ["app"],
    registry=METRICS_REGISTRY,
)


async def update_queue_gauges(
    redis: Redis,
    *,
    queue_name: str,
    inflight_zset_name: str,
    workers_hash_key: str,
    worker_ttl_s: int,
) -> None:
    qlen = await redis.llen(queue_name)
    QUEUE_LENGTH.labels(APP_NAME, queue_name).set(qlen)

    inflight = await redis.zcard(inflight_zset_name)
    INFLIGHT_COUNT.labels(APP_NAME, queue_name).set(inflight)

    workers = await redis.hgetall(workers_hash_key)
    now = time.time()
    active = 0

    for _, raw in workers.items():
        try:
            wd = json.loads(raw)
            ts = float(wd.get("timestamp", 0))
            if now - ts <= worker_ttl_s:
                active += 1
        except Exception:
            continue

    WORKERS_ACTIVE.labels(APP_NAME).set(active)
