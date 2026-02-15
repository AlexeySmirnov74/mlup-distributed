import os
import pytest
import httpx

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8009")


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus_metrics():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{BASE_URL}/metrics")
        assert r.status_code == 200
        text = r.text

        # Core metrics we expect
        assert "http_requests_total" in text
        assert "http_request_duration_seconds" in text

        # Queue metrics
        assert "mlup_queue_length" in text
        assert "mlup_inflight" in text

