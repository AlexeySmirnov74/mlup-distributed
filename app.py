import logging
import os
import time

import uvicorn
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from mlup.config import set_logging_settings, LOGGING_CONFIG

# Import app from composition root
from common_app import app

# Import metrics objects from queue_module (single registry per process)
from mlup.web.architecture.redis_queue import (
    METRICS_REGISTRY,
    APP_NAME,
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION,
)

set_logging_settings(LOGGING_CONFIG, level=logging.INFO)
logger = logging.getLogger("mlup")

# IMPORTANT: API-only mode (no queue worker loop in API processes)
os.environ["MLUP_RUN_MODE"] = "api"

MODEL_PORT = int(os.environ.get("MODEL_PORT", 8009))
UVICORN_WORKERS = int(os.environ.get("WORKERS", 4))


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)

        HTTP_REQUESTS_TOTAL.labels(APP_NAME, request.method, route_path, str(response.status_code)).inc()
        HTTP_REQUEST_DURATION.labels(APP_NAME, request.method, route_path).observe(elapsed)

        return response


# Attach middleware once
app.add_middleware(PrometheusMiddleware)


@app.get("/metrics")
def metrics():
    """
    Prometheus scrape endpoint.
    Variant A: expose metrics via API service.
    """
    payload = generate_latest(METRICS_REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    logger.info(
        "Starting API (MLUP_RUN_MODE=api) on port=%s with uvicorn workers=%s",
        MODEL_PORT,
        UVICORN_WORKERS,
    )
    uvicorn.run(
        "common_app:app",   # important: import from common_app
        host="0.0.0.0",
        port=MODEL_PORT,
        workers=UVICORN_WORKERS,
        proxy_headers=True,
        reload=False,
    )
