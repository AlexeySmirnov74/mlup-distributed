import asyncio
import inspect
import json
import logging
import os
import socket
import time
import traceback
from asyncio import Task
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from mlup.constants import DEFAULT_X_ARG_NAME, WebAppArchitecture
from mlup.errors import WebAppLoadError
from mlup.ml.model import MLupModel
from mlup.utils.loop import create_async_task
from mlup.web.architecture.base import BaseWebAppArchitecture

from .leader_tasks import LeaderTasks
from .queue_metrics import update_queue_gauges
from .redis_client import RedisConnector
from .redis_keys import RedisKeys

logger = logging.getLogger("mlup")


class Architecture(BaseWebAppArchitecture):
    """
    Distributed worker_and_queue implementation backed by Redis.

    Run modes (ENV MLUP_RUN_MODE):
      - api:    enqueue + poll only, NO worker loop
      - worker: worker loop only (no HTTP server, use workers.py)
      - both:   api + worker loop (not recommended with uvicorn workers>1)
    """

    fast_api: FastAPI
    ml: MLupModel

    item_id_col_name: str
    max_queue_size: int
    ttl_predicted_data: int
    ttl_client_wait: float
    is_long_predict: bool

    model_directory_swagger: str
    redis_url: str
    redis_queue_name: str

    get_first_model_argument: bool
    get_formatted_result: bool

    deploy_type: str
    web_description: str

    redis_client: Optional[Redis] = None
    redis: Optional[RedisConnector] = None

    worker_process: Optional[Task] = None
    heartbeat_process: Optional[Task] = None
    leader_election_process: Optional[Task] = None
    leader_cleanup_process: Optional[Task] = None
    metrics_process: Optional[Task] = None

    type: WebAppArchitecture = WebAppArchitecture.worker_and_queue

    _running: bool = False
    is_master: bool = False

    def __init__(self, **configs):
        self.fast_api = configs.pop("fast_api")
        self.ml = configs.pop("ml")
        self.item_id_col_name = configs.pop("item_id_col_name")

        self.max_queue_size = configs.pop("max_queue_size")
        self.ttl_predicted_data = configs.pop("ttl_predicted_data")
        self.ttl_client_wait = configs.pop("ttl_client_wait")
        self.is_long_predict = configs.pop("is_long_predict")

        self.is_use_redis = configs.pop("is_use_redis")
        self.deploy_type = configs.pop("deploy_type")
        self.web_description = configs.pop("web_description")
        self.model_directory_swagger = configs.pop("model_directory_swagger")

        self.redis_url = configs.pop("redis_url")
        self.redis_queue_name = configs.pop("redis_queue_name")

        self.get_first_model_argument = configs.pop("get_first_model_argument")
        self.get_formatted_result = configs.pop("get_formatted_result")

        self.run_mode: str = os.getenv("MLUP_RUN_MODE", "both").lower()
        if self.run_mode not in ("api", "worker", "both"):
            self.run_mode = "both"

        self.heartbeat_interval_s: float = configs.pop("heartbeat_interval_s", 10.0)
        self.worker_ttl_s: int = configs.pop("worker_ttl_s", 20)

        self.metrics_poll_interval_s: float = configs.pop("metrics_poll_interval_s", 2.0)

        self.leader_lock_ttl_s: int = configs.pop("leader_lock_ttl_s", 30)
        self.leader_renew_every_s: int = configs.pop("leader_renew_every_s", 10)
        self.leader_cleanup_every_s: int = configs.pop("leader_cleanup_every_s", 15)

        self.inflight_stale_s: int = configs.pop("inflight_stale_s", 120)
        self.queue_block_timeout_s: int = configs.pop("queue_block_timeout_s", 1)
        self.error_backoff_s: float = configs.pop("error_backoff_s", 1.0)

        self.meta_ttl_s: int = configs.pop("meta_ttl_s", self.ttl_predicted_data + 3600)

        self.node_id = os.getenv("MLUP_NODE_ID") or socket.gethostname() or "node"
        container_id = os.getenv("HOSTNAME", "container")
        self.worker_id = f"worker-{self.node_id}-{container_id}-{os.getpid()}"
        self.unique_worker_key = self.worker_id

        self.keys = RedisKeys()

        if self.ttl_predicted_data <= 0:
            raise ValueError(f"ttl_predicted_data must be > 0, got {self.ttl_predicted_data}")

        self.extra = configs

    # -------------------------------------------------------------------------
    # Derived Redis structure names
    # -------------------------------------------------------------------------
    @property
    def processing_list_name(self) -> str:
        return self.keys.processing_list(self.redis_queue_name)

    @property
    def inflight_zset_name(self) -> str:
        return self.keys.inflight_zset(self.redis_queue_name)

    @property
    def inflight_hash_name(self) -> str:
        return self.keys.inflight_hash(self.redis_queue_name)

    def _is_running(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Redis connection
    # -------------------------------------------------------------------------
    async def connect_redis(self) -> None:
        self.redis = RedisConnector(self.redis_url, decode_responses=True)
        self.redis_client = await self.redis.connect()
        logger.info("%s Connected to Redis (PING OK).", self.worker_id)

    async def _ensure_redis(self) -> None:
        if not self.redis:
            self.redis = RedisConnector(self.redis_url, decode_responses=True)
        self.redis_client = await self.redis.ensure(is_running_fn=self._is_running, who=self.worker_id)

    # -------------------------------------------------------------------------
    # Metrics loop
    # -------------------------------------------------------------------------
    async def metrics_loop(self) -> None:
        while self._running:
            try:
                await self._ensure_redis()
                await update_queue_gauges(
                    self.redis_client,
                    queue_name=self.redis_queue_name,
                    inflight_zset_name=self.inflight_zset_name,
                    workers_hash_key=self.keys.workers_hash,
                    worker_ttl_s=self.worker_ttl_s,
                )
            except Exception as e:
                logger.debug("%s metrics_loop error: %s", self.worker_id, e)

            await asyncio.sleep(self.metrics_poll_interval_s)

    # -------------------------------------------------------------------------
    # Heartbeat
    # -------------------------------------------------------------------------
    async def register_worker_heartbeat(self) -> None:
        data = {
            "timestamp": time.time(),
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "node_id": self.node_id,
        }
        await self.redis_client.hset(self.keys.workers_hash, self.unique_worker_key, json.dumps(data))

    async def heartbeat_loop(self) -> None:
        while self._running:
            try:
                await self._ensure_redis()
                await self.register_worker_heartbeat()
            except Exception as e:
                logger.warning("%s Heartbeat failed: %s", self.worker_id, e)
            await asyncio.sleep(self.heartbeat_interval_s)

    # -------------------------------------------------------------------------
    # Leader election
    # -------------------------------------------------------------------------
    async def leader_election_loop(self) -> None:
        while self._running:
            try:
                await self._ensure_redis()
                acquired = await self.redis_client.set(
                    self.keys.leader_lock,
                    self.unique_worker_key,
                    nx=True,
                    ex=self.leader_lock_ttl_s,
                )
                if acquired:
                    if not self.is_master:
                        logger.info(
                            "%s Became leader | duties=[cleanup_workers,cleanup_in_progress,cleanup_inflight]",
                            self.worker_id,
                        )
                    self.is_master = True
                else:
                    owner = await self.redis_client.get(self.keys.leader_lock)
                    if owner == self.unique_worker_key:
                        await self.redis_client.expire(self.keys.leader_lock, self.leader_lock_ttl_s)
                        self.is_master = True
                    else:
                        if self.is_master:
                            logger.info("%s Lost leader role (owner=%s).", self.worker_id, owner)
                        self.is_master = False
            except Exception as e:
                logger.warning("%s Leader election error: %s", self.worker_id, e)
                self.is_master = False

            await asyncio.sleep(self.leader_renew_every_s)

    # -------------------------------------------------------------------------
    # Leader cleanup loop
    # -------------------------------------------------------------------------
    async def leader_cleanup_loop(self) -> None:
        tasks = LeaderTasks(
            redis=self.redis_client,
            keys=self.keys,
            queue_name=self.redis_queue_name,
            inflight_zset_name=self.inflight_zset_name,
            inflight_hash_name=self.inflight_hash_name,
            processing_list_name=self.processing_list_name,
            worker_ttl_s=self.worker_ttl_s,
            inflight_stale_s=self.inflight_stale_s,
            ttl_predicted_data=self.ttl_predicted_data,
            meta_ttl_s=self.meta_ttl_s,
            leader_cleanup_every_s=self.leader_cleanup_every_s,
        )

        while self._running:
            await asyncio.sleep(self.leader_cleanup_every_s)
            if not self.is_master:
                continue
            try:
                await self._ensure_redis()
                tasks.redis = self.redis_client
                await tasks.run_once()
            except Exception as e:
                logger.warning("%s Leader cleanup error: %s", self.worker_id, e)

    # -------------------------------------------------------------------------
    # Status / Meta / Result
    # -------------------------------------------------------------------------
    async def set_predict_status(self, predict_id: str, status: str, ttl: Optional[int] = None) -> None:
        status_data = {
            "status": status,
            "pid": os.getpid(),
            "node_id": self.node_id,
            "worker_id": self.worker_id,
            "timestamp": time.time(),
        }
        args = {"name": self.keys.status_key(predict_id), "value": json.dumps(status_data)}
        if ttl is not None:
            args["ex"] = ttl
        await self.redis_client.set(**args)

    async def check_predict_status_from_redis(self, predict_id: str) -> Optional[str]:
        raw = await self.redis_client.get(self.keys.status_key(predict_id))
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return data.get("status")
        except Exception:
            return None

    async def set_predict_meta(self, predict_id: str) -> None:
        meta = {"created_at": time.time()}
        await self.redis_client.set(self.keys.meta_key(predict_id), json.dumps(meta), ex=self.meta_ttl_s)

    async def meta_exists(self, predict_id: str) -> bool:
        return (await self.redis_client.exists(self.keys.meta_key(predict_id))) == 1

    def _serialize_error(self, exc: BaseException) -> Dict[str, Any]:
        return {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    async def _save_result(self, predict_id: str, payload: Any) -> None:
        await self.redis_client.set(self.keys.result_key(predict_id), json.dumps(payload), ex=self.ttl_predicted_data)

    async def _load_result(self, predict_id: str) -> Optional[Any]:
        raw = await self.redis_client.get(self.keys.result_key(predict_id))
        if raw is None:
            return None
        return json.loads(raw)

    # -------------------------------------------------------------------------
    # Reliable queue: claim + ack
    # -------------------------------------------------------------------------
    async def _claim_job(self) -> Optional[Tuple[str, Dict[str, Any], str]]:
        raw_payload = await self.redis_client.brpoplpush(
            self.redis_queue_name,
            self.processing_list_name,
            timeout=self.queue_block_timeout_s,
        )
        if not raw_payload:
            return None

        task_data = json.loads(raw_payload)
        predict_id = task_data["alex_predict_id"]

        now = time.time()
        await self.redis_client.hset(self.inflight_hash_name, predict_id, raw_payload)
        await self.redis_client.zadd(self.inflight_zset_name, {predict_id: now})
        return predict_id, task_data, raw_payload

    async def _ack_job(self, predict_id: str, raw_payload: Optional[str] = None) -> None:
        if raw_payload is None:
            raw_payload = await self.redis_client.hget(self.inflight_hash_name, predict_id)

        await self.redis_client.hdel(self.inflight_hash_name, predict_id)
        await self.redis_client.zrem(self.inflight_zset_name, predict_id)

        if raw_payload:
            await self.redis_client.lrem(self.processing_list_name, 1, raw_payload)

    # -------------------------------------------------------------------------
    # Model execution
    # -------------------------------------------------------------------------
    async def _call_model_predict(self, data_for_predict: Dict[str, Any]) -> Any:
        fn = self.ml.predict
        if inspect.iscoroutinefunction(fn):
            return await fn(**data_for_predict)
        return await asyncio.to_thread(fn, **data_for_predict)

    async def _predict(self, predict_id: str, data_for_predict: Dict[str, Any]) -> None:
        predicted_data = None
        err_payload = None

        try:
            predicted_data = await self._call_model_predict(data_for_predict)
        except Exception as e:
            err_payload = self._serialize_error(e)

        if self.get_formatted_result:
            result_payload = {"result": predicted_data, "error": err_payload, "timestamp": time.time()}
        else:
            result_payload = predicted_data if err_payload is None else {"error": err_payload}

        await self._save_result(predict_id, result_payload)
        await self.set_predict_status(predict_id, "done", ttl=self.ttl_predicted_data)

    # -------------------------------------------------------------------------
    # Worker loop
    # -------------------------------------------------------------------------
    async def _start_worker(self) -> None:
        logger.info("%s Worker loop started. Queue=%s", self.worker_id, self.redis_queue_name)

        await self._ensure_redis()
        await self.register_worker_heartbeat()

        self.heartbeat_process = create_async_task(self.heartbeat_loop())
        self.leader_election_process = create_async_task(self.leader_election_loop())
        self.leader_cleanup_process = create_async_task(self.leader_cleanup_loop())

        while self._running:
            try:
                await self._ensure_redis()

                claimed = await self._claim_job()
                if not claimed:
                    continue

                predict_id, task_data, raw_payload = claimed
                await self.set_predict_status(predict_id, "in_progress", ttl=self.ttl_predicted_data)

                data_for_predict = task_data["alex_data_for_predict"]
                if not self.get_first_model_argument:
                    data_for_predict = {DEFAULT_X_ARG_NAME: data_for_predict}

                logger.info("%s Running predict_id=%s", self.worker_id, predict_id)

                try:
                    await self._predict(predict_id, data_for_predict)
                finally:
                    await self._ack_job(predict_id, raw_payload)

            except asyncio.CancelledError:
                logger.info("%s Worker cancelled.", self.worker_id)
                break
            except Exception as e:
                logger.error("%s Worker loop error: %s", self.worker_id, e)
                await asyncio.sleep(self.error_backoff_s)

        logger.info("%s Worker loop stopped.", self.worker_id)

    # -------------------------------------------------------------------------
    # Public lifecycle
    # -------------------------------------------------------------------------
    async def run(self) -> None:
        await self.connect_redis()

        if self.is_running:
            raise WebAppLoadError(f"{self.worker_id} is already running")

        self._running = True

        self.metrics_process = create_async_task(self.metrics_loop())

        if self.run_mode in ("worker", "both"):
            self.worker_process = create_async_task(self._start_worker())
        else:
            logger.info("%s Running in API-only mode (no worker loop).", self.worker_id)

    @property
    def is_running(self) -> bool:
        return self._running

    async def stop(self) -> None:
        if not self.is_running:
            raise WebAppLoadError("Worker not started")

        self._running = False

        for t in (
            self.metrics_process,
            self.worker_process,
            self.heartbeat_process,
            self.leader_election_process,
            self.leader_cleanup_process,
        ):
            if t and not t.cancelled():
                try:
                    t.cancel()
                    await t
                except Exception:
                    pass

        if self.redis:
            await self.redis.close()
        self.redis_client = None

        logger.info("%s Stopped.", self.worker_id)

    # -------------------------------------------------------------------------
    # MLup integration
    # -------------------------------------------------------------------------
    def load(self) -> None:
        if self.is_long_predict:
            self.fast_api.add_api_route(
                self.model_directory_swagger + "/predict/{predict_id}",
                self.get_predict_result_from_redis,
                methods=["GET"],
            )

    async def get_predict_result_from_redis(self, predict_id: str) -> Any:
        await self._ensure_redis()
        status = await self.check_predict_status_from_redis(predict_id)

        if status in ("wait", "in_progress"):
            retry_after = 1
            return JSONResponse(
                status_code=202,
                content={"predict_id": predict_id, "status": status, "retry_after": retry_after},
                headers={"Retry-After": str(retry_after)},
            )

        if status == "done":
            result = await self._load_result(predict_id)
            if result is None:
                raise HTTPException(status_code=410, detail="Result expired")
            return result

        if await self.meta_exists(predict_id):
            raise HTTPException(status_code=410, detail="Predict_id expired")
        raise HTTPException(status_code=404, detail="Unknown predict_id")

    async def _queue_over_limit(self) -> bool:
        size = await self.redis_client.llen(self.redis_queue_name)
        return size >= self.max_queue_size

    async def predict(self, data_for_predict: Dict, predict_id: str) -> Any:
        if not self.is_running:
            raise HTTPException(status_code=503, detail="Service is not running yet")

        if not isinstance(data_for_predict, dict):
            raise HTTPException(status_code=422, detail="Payload must be a JSON object (dictionary)")

        try:
            await self._ensure_redis()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Redis unavailable: {e}")

        if await self._queue_over_limit():
            raise HTTPException(status_code=429, detail="Queue is full, try later")

        task = {"alex_predict_id": predict_id, "alex_data_for_predict": data_for_predict}

        try:
            payload = json.dumps(task)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Payload is not JSON serializable: {e}")

        try:
            await self.set_predict_meta(predict_id)
            await self.redis_client.lpush(self.redis_queue_name, payload)
            await self.set_predict_status(predict_id, "wait", ttl=self.ttl_predicted_data)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Redis enqueue failed: {e}")

        logger.info(
            "POST accepted | time=%sZ | worker=%s | predict_id=%s | queue=%s",
            datetime.utcnow().isoformat(),
            self.worker_id,
            predict_id,
            self.redis_queue_name,
        )

        if self.is_long_predict:
            return {"predict_id": predict_id}
        return None
