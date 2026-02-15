import os
import time
import uuid
from typing import Any, Dict

import mlup
from mlup.constants import ModelDataTransformerType


class DemoStubModelSync:
    """
    Data-Scientist-like synchronous model:
    - time.sleep() simulates heavy work
    - returns JSON-serializable dict
    """
    def predict(self, X: Any) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        started = time.time()

        time.sleep(10)

        elapsed = round(time.time() - started, 3)
        return {
            "request_id": request_id,
            "status": "ok",
            "model": {"name": "Demo Model", "version": "1.0"},
            "timing": {"elapsed_seconds": elapsed},
            "prediction": {"risk_score": 0.42, "decision": "approve", "reason_codes": ["demo_stub"]},
            "echo": X if isinstance(X, (dict, list, str, int, float, bool)) else str(X),
        }


def build_up() -> mlup.UP:
    """
    Single source of truth for mlup.Config.
    Both app.py and workers.py import and use this.
    """
    model_port = int(os.environ.get("MODEL_PORT", 8009))
    ttl_predicted_data = int(os.environ.get("TTL_PREDICTED_DATA", 300))
    max_queue_size = int(os.environ.get("MAX_QUEUE_SIZE", 100))
    redis_queue_name = os.getenv("REDIS_QUEUE_NAME", "predict_queue")

    return mlup.UP(
        ml_model=DemoStubModelSync(),
        conf=mlup.Config(
            mode="mlup.web.architecture.redis_queue.Architecture",
            name="Demo Model",
            web_app_version="1.0",
            version="1.0",
            predict_method_name="predict",

            # Raw JSON in POST body (no wrapper)
            get_first_model_argument=False,
            get_formatted_result=False,

            web_description="Non-blocking API: POST returns ticket, GET polls (202/200).",
            deploy_type="demo",

            data_transformer_for_predict=ModelDataTransformerType.SRC_TYPES,
            data_transformer_for_predicted=ModelDataTransformerType.SRC_TYPES,

            port=model_port,
            ttl_predicted_data=ttl_predicted_data,
            model_directory_swagger="/model1",

            redis_url=mlup.config.get_redis_url(),
            redis_queue_name=redis_queue_name,
            max_queue_size=max_queue_size,
            ttl_client_wait=1,

            is_long_predict=True,
            is_use_redis=True,

            auto_detect_predict_params=True,
            column_validation=False,
            debug=False,
        ),
    )


# Build once per process import
up = build_up()
up.ml.load()
up.web.load()

# FastAPI application (imported by uvicorn)
app = up.web.app
