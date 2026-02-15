from dataclasses import dataclass


@dataclass(frozen=True)
class RedisKeys:
    # Global keys
    workers_hash: str = "worker_pids"
    leader_lock: str = "mlup:leader_lock"

    # Predict keys
    status_prefix: str = "predict_status:"
    result_prefix: str = "predict_result:"
    meta_prefix: str = "predict_meta:"

    # Queue structures
    processing_suffix: str = ":processing"
    inflight_zset_suffix: str = ":inflight"
    inflight_hash_suffix: str = ":inflight_payload"

    def status_key(self, predict_id: str) -> str:
        return f"{self.status_prefix}{predict_id}"

    def result_key(self, predict_id: str) -> str:
        return f"{self.result_prefix}{predict_id}"

    def meta_key(self, predict_id: str) -> str:
        return f"{self.meta_prefix}{predict_id}"

    def processing_list(self, queue_name: str) -> str:
        return f"{queue_name}{self.processing_suffix}"

    def inflight_zset(self, queue_name: str) -> str:
        return f"{queue_name}{self.inflight_zset_suffix}"

    def inflight_hash(self, queue_name: str) -> str:
        return f"{queue_name}{self.inflight_hash_suffix}"
