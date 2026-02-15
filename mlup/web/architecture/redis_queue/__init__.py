from .architecture import Architecture

# Re-export metrics registry for /metrics endpoint (used by app.py)
from .queue_metrics import (
    METRICS_REGISTRY,
    APP_NAME,
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION,
)
