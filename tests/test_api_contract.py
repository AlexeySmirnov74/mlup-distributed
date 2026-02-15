import os
import time
import uuid
import asyncio

import pytest
import httpx


BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8009")
MODEL_DIR = os.getenv("TEST_MODEL_DIR", "/model1")


def post_url() -> str:
    return f"{BASE_URL}{MODEL_DIR}/predict"


def get_url(predict_id: str) -> str:
    return f"{BASE_URL}{MODEL_DIR}/predict/{predict_id}"


def extract_predict_id(response: httpx.Response) -> str:
    """
    Current MLup contract:
      response JSON is usually: {"predict_result": {"predict_id": "..."}}
    Also MLup sets header PREDICT_ID_HEADER (often "X-Predict-Id").
    We'll support both, but prefer body (more explicit for clients).
    """
    j = response.json()

    # Primary: body contract
    if isinstance(j, dict):
        pr = j.get("predict_result")
        if isinstance(pr, dict) and "predict_id" in pr:
            return pr["predict_id"]

        # Fallback: in case someone changed web/app.py to return flat dict
        if "predict_id" in j:
            return j["predict_id"]

    # Secondary: header contract
    for header_name in ("X-Predict-Id", "x-predict-id", "PREDICT_ID", "predict_id"):
        v = response.headers.get(header_name)
        if v:
            return v

    raise AssertionError(f"Cannot extract predict_id from response: body={j}, headers={dict(response.headers)}")


@pytest.mark.asyncio
async def test_post_valid_returns_predict_id():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(post_url(), json={"hello": "world"})
        assert r.status_code == 200, r.text

        predict_id = extract_predict_id(r)
        uuid.UUID(predict_id)  # validates uuid format


@pytest.mark.asyncio
async def test_post_invalid_payload_returns_422():
    # payload is not a JSON object (dict) -> should be 422
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(post_url(), json="not a dict")
        assert r.status_code == 422, r.text


@pytest.mark.asyncio
async def test_get_unknown_id_returns_404():
    unknown_id = str(uuid.uuid4())
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(get_url(unknown_id))
        assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_post_then_get_nonblocking_flow_202_then_200():
    """
    Non-blocking contract:
      - POST returns predict_id quickly (should not sleep 10s in API)
      - GET returns 202 while job is running
      - eventually returns 200 with result
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        t0 = time.perf_counter()
        r = await client.post(post_url(), json={"demo": True})
        t_post = time.perf_counter() - t0

        assert r.status_code == 200, r.text
        assert t_post < 2.0, f"POST took too long ({t_post:.2f}s). API might be blocking."

        predict_id = extract_predict_id(r)

        # Poll for up to ~30 seconds (stub model sleeps 10s)
        deadline = time.time() + 35
        saw_202 = False

        while time.time() < deadline:
            g = await client.get(get_url(predict_id))

            if g.status_code == 202:
                saw_202 = True
                ra = g.headers.get("Retry-After")
                # Retry-After is usually "1"
                delay = float(ra) if (ra and ra.isdigit()) else 1.0
                await asyncio.sleep(delay)
                continue

            if g.status_code == 200:
                assert saw_202, "Expected to see 202 at least once before 200"
                body = g.json()
                assert isinstance(body, (dict, list)), body
                return

            if g.status_code == 410:
                pytest.fail(f"Result expired too early (410): {g.text}")

            pytest.fail(f"Unexpected status {g.status_code}: {g.text}")

        pytest.fail("Timed out waiting for job completion")
