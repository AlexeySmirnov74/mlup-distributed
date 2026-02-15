#!/usr/bin/env python3
"""
Stress test for non-blocking MLup Redis-queue API.

What it does:
  - Sends POST /model1/predict with a mix of valid and invalid JSON payloads
  - Collects predict_id for successful POSTs
  - Polls GET /model1/predict/{predict_id} until done/expired/unknown
  - Also sends some invalid GETs (random/garbled ids) to test 404/410 paths
  - Prints a compact summary at the end

Usage examples:
  python stress_test.py --base-url http://localhost:8009 --model-dir /model1 --concurrency 50 --requests 500
  python stress_test.py --base-url http://localhost:8009 --invalid-post-rate 0.15 --invalid-get-rate 0.10

Notes:
  - Uses aiohttp for high concurrency.
  - If you don't have aiohttp: pip install aiohttp
"""

import argparse
import asyncio
import json
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import aiohttp


# -----------------------------
# Helpers
# -----------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def rand_str(n: int = 12) -> str:
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


def make_valid_payload() -> Dict[str, Any]:
    # A generic "data science style" JSON payload.
    # Keep it simple and stable for demos.
    return {
        "request_id": str(uuid.uuid4()),
        "customer_id": random.randint(1, 10_000_000),
        "features": {
            "age": random.randint(18, 80),
            "income": round(random.uniform(20_000, 250_000), 2),
            "country": random.choice(["NL", "DE", "PL", "FR", "US"]),
            "segment": random.choice(["mass", "affluent", "premium"]),
        },
        "timestamp_ms": now_ms(),
    }


def make_invalid_payload() -> Any:
    # Different types of invalid:
    #  - wrong JSON shape
    #  - non-serializable
    #  - wrong content type (we'll also send malformed JSON string)
    choice = random.choice(["string", "number", "array", "missing_fields", "malformed_json"])
    if choice == "string":
        return "not a json object"
    if choice == "number":
        return 12345
    if choice == "array":
        return [1, 2, 3, {"x": "y"}]
    if choice == "missing_fields":
        return {"foo": "bar"}  # still valid JSON, but maybe invalid for model
    return "__MALFORMED_JSON__"  # special marker handled separately


def random_predict_id_like() -> str:
    # valid UUID format but likely unknown
    return str(uuid.uuid4())


def garbage_predict_id() -> str:
    # invalid path segment
    return rand_str(20)


@dataclass
class Stats:
    post_ok: int = 0
    post_fail: int = 0
    post_http: Dict[int, int] = field(default_factory=dict)

    get_done: int = 0
    get_wait: int = 0
    get_404: int = 0
    get_410: int = 0
    get_other: int = 0
    get_http: Dict[int, int] = field(default_factory=dict)

    exceptions: int = 0
    lat_post_ms: List[int] = field(default_factory=list)
    lat_get_ms: List[int] = field(default_factory=list)

    def inc_http(self, bucket: Dict[int, int], status: int) -> None:
        bucket[status] = bucket.get(status, 0) + 1

    def pctl(self, values: List[int], p: float) -> Optional[float]:
        if not values:
            return None
        values_sorted = sorted(values)
        k = int((len(values_sorted) - 1) * p)
        return float(values_sorted[k])


# -----------------------------
# HTTP calls
# -----------------------------

async def do_post(
    session: aiohttp.ClientSession,
    url: str,
    invalid_rate: float,
    stats: Stats,
    timeout_s: float,
) -> Optional[str]:
    """
    Returns predict_id if POST was accepted and response contains it.
    """
    t0 = time.time()
    try:
        is_invalid = random.random() < invalid_rate
        payload = make_invalid_payload() if is_invalid else make_valid_payload()

        # Handle malformed JSON explicitly
        if payload == "__MALFORMED_JSON__":
            data = "{ this is : not json }"
            headers = {"Content-Type": "application/json"}
            async with session.post(url, data=data, headers=headers, timeout=timeout_s) as r:
                stats.inc_http(stats.post_http, r.status)
                body = await r.text()
                dt = int((time.time() - t0) * 1000)
                stats.lat_post_ms.append(dt)
                if 200 <= r.status < 300:
                    # Even if server accepted malformed, try parse
                    try:
                        j = json.loads(body)
                        return j.get("predict_id")
                    except Exception:
                        return None
                stats.post_fail += 1
                return None

        # Normal JSON request
        async with session.post(url, json=payload, timeout=timeout_s) as r:
            stats.inc_http(stats.post_http, r.status)
            dt = int((time.time() - t0) * 1000)
            stats.lat_post_ms.append(dt)

            if 200 <= r.status < 300:
                # In your is_long_predict=True flow response should have predict_id
                try:
                    j = await r.json(content_type=None)
                except Exception:
                    # fallback
                    text = await r.text()
                    try:
                        j = json.loads(text)
                    except Exception:
                        j = {}
                pid = j.get("predict_id")
                if pid:
                    stats.post_ok += 1
                    return pid
                # If no predict_id, still count as ok, but no polling.
                stats.post_ok += 1
                return None
            else:
                stats.post_fail += 1
                return None

    except Exception:
        stats.exceptions += 1
        return None


async def do_get_until_terminal(
    session: aiohttp.ClientSession,
    url_base: str,
    predict_id: str,
    stats: Stats,
    timeout_s: float,
    poll_min_delay: float,
    poll_max_delay: float,
    max_polls: int,
) -> None:
    """
    Polls GET /predict/{id} until terminal:
      - 200 done -> count get_done
      - 404 unknown -> count get_404
      - 410 expired -> count get_410
    Non-terminal:
      - 202 wait/in_progress -> count get_wait and continue
    """
    polls = 0
    while polls < max_polls:
        polls += 1
        t0 = time.time()
        try:
            async with session.get(f"{url_base}/{predict_id}", timeout=timeout_s) as r:
                stats.inc_http(stats.get_http, r.status)
                dt = int((time.time() - t0) * 1000)
                stats.lat_get_ms.append(dt)

                if r.status == 200:
                    stats.get_done += 1
                    return
                if r.status == 202:
                    stats.get_wait += 1
                    # Respect Retry-After if present
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        delay = max(poll_min_delay, min(poll_max_delay, float(ra)))
                    else:
                        delay = random.uniform(poll_min_delay, poll_max_delay)
                    await asyncio.sleep(delay)
                    continue
                if r.status == 404:
                    stats.get_404 += 1
                    return
                if r.status == 410:
                    stats.get_410 += 1
                    return

                # Other unexpected codes
                stats.get_other += 1
                return

        except Exception:
            stats.exceptions += 1
            await asyncio.sleep(random.uniform(poll_min_delay, poll_max_delay))

    # If exceeded max polls, count as other
    stats.get_other += 1


async def do_invalid_gets(
    session: aiohttp.ClientSession,
    url_base: str,
    count: int,
    invalid_get_rate: float,
    stats: Stats,
    timeout_s: float,
) -> None:
    """
    Sends additional invalid GET requests to test 404/410 paths:
      - some random UUIDs
      - some garbage strings
      - optionally some valid-looking but never enqueued
    """
    for _ in range(count):
        if random.random() < invalid_get_rate:
            pid = garbage_predict_id() if random.random() < 0.5 else random_predict_id_like()
            t0 = time.time()
            try:
                async with session.get(f"{url_base}/{pid}", timeout=timeout_s) as r:
                    stats.inc_http(stats.get_http, r.status)
                    dt = int((time.time() - t0) * 1000)
                    stats.lat_get_ms.append(dt)

                    if r.status == 202:
                        stats.get_wait += 1
                    elif r.status == 200:
                        stats.get_done += 1
                    elif r.status == 404:
                        stats.get_404 += 1
                    elif r.status == 410:
                        stats.get_410 += 1
                    else:
                        stats.get_other += 1
            except Exception:
                stats.exceptions += 1


# -----------------------------
# Orchestration
# -----------------------------

async def worker_task(
    name: str,
    session: aiohttp.ClientSession,
    post_url: str,
    get_url_base: str,
    stats: Stats,
    total_posts: int,
    invalid_post_rate: float,
    invalid_get_rate: float,
    timeout_s: float,
    poll_min_delay: float,
    poll_max_delay: float,
    max_polls: int,
) -> None:
    predict_ids: List[str] = []

    # Fire POSTs
    for _ in range(total_posts):
        pid = await do_post(session, post_url, invalid_post_rate, stats, timeout_s)
        if pid:
            predict_ids.append(pid)

    # Poll results
    # (also inject some invalid GETs)
    await do_invalid_gets(
        session,
        get_url_base,
        count=max(1, int(len(predict_ids) * 0.2)),
        invalid_get_rate=invalid_get_rate,
        stats=stats,
        timeout_s=timeout_s,
    )

    # Poll valid predict_ids
    for pid in predict_ids:
        await do_get_until_terminal(
            session=session,
            url_base=get_url_base,
            predict_id=pid,
            stats=stats,
            timeout_s=timeout_s,
            poll_min_delay=poll_min_delay,
            poll_max_delay=poll_max_delay,
            max_polls=max_polls,
        )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8009", help="API base URL")
    ap.add_argument("--model-dir", default="/model1", help="Model route prefix")
    ap.add_argument("--concurrency", type=int, default=20, help="Number of concurrent clients")
    ap.add_argument("--requests", type=int, default=200, help="Total POST requests across all clients")
    ap.add_argument("--invalid-post-rate", type=float, default=0.10, help="Fraction of POSTs that are invalid")
    ap.add_argument("--invalid-get-rate", type=float, default=0.10, help="Fraction of extra GETs that are invalid")
    ap.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    ap.add_argument("--poll-min", type=float, default=0.2, help="Min delay between GET polls")
    ap.add_argument("--poll-max", type=float, default=1.0, help="Max delay between GET polls")
    ap.add_argument("--max-polls", type=int, default=60, help="Max polls per predict_id")
    args = ap.parse_args()

    post_url = f"{args.base_url}{args.model_dir}/predict"
    get_url_base = f"{args.base_url}{args.model_dir}/predict"

    # Distribute total POST requests among concurrent tasks
    per_worker = args.requests // args.concurrency
    remainder = args.requests % args.concurrency

    stats = Stats()

    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(args.concurrency):
            n = per_worker + (1 if i < remainder else 0)
            tasks.append(
                asyncio.create_task(
                    worker_task(
                        name=f"client-{i}",
                        session=session,
                        post_url=post_url,
                        get_url_base=get_url_base,
                        stats=stats,
                        total_posts=n,
                        invalid_post_rate=args.invalid_post_rate,
                        invalid_get_rate=args.invalid_get_rate,
                        timeout_s=args.timeout,
                        poll_min_delay=args.poll_min,
                        poll_max_delay=args.poll_max,
                        max_polls=args.max_polls,
                    )
                )
            )

        t_start = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - t_start

    # Summary
    print("\n=== Stress Test Summary ===")
    print(f"Base URL: {args.base_url}  Model dir: {args.model_dir}")
    print(f"Concurrency: {args.concurrency}  Total POSTs: {args.requests}  Elapsed: {elapsed:.2f}s")
    print("\nPOST:")
    print(f"  ok: {stats.post_ok}  fail: {stats.post_fail}  exceptions: {stats.exceptions}")
    print(f"  status codes: {dict(sorted(stats.post_http.items()))}")
    p50 = stats.pctl(stats.lat_post_ms, 0.50)
    p95 = stats.pctl(stats.lat_post_ms, 0.95)
    if p50 is not None:
        print(f"  latency ms: p50={p50:.0f}  p95={p95:.0f}")

    print("\nGET:")
    print(f"  done(200): {stats.get_done}")
    print(f"  wait(202): {stats.get_wait}")
    print(f"  unknown(404): {stats.get_404}")
    print(f"  expired(410): {stats.get_410}")
    print(f"  other: {stats.get_other}")
    print(f"  status codes: {dict(sorted(stats.get_http.items()))}")
    p50g = stats.pctl(stats.lat_get_ms, 0.50)
    p95g = stats.pctl(stats.lat_get_ms, 0.95)
    if p50g is not None:
        print(f"  latency ms: p50={p50g:.0f}  p95={p95g:.0f}")

    # A simple “health” hint:
    inflight_hint = "If mlup_inflight stays >0 long after workload ends, ACK may be broken."
    print(f"\nHint: {inflight_hint}\n")


if __name__ == "__main__":
    asyncio.run(main())

