"""Send InferenceRequests to C's service (or a stand-in) over HTTP JSON."""

import time
import urllib.error
import urllib.request

from gigacity_contracts import InferenceRequest, InferenceResult


def now_ms() -> int:
    return int(time.time() * 1000)


def failed_result(request: InferenceRequest, received_at_ms: int) -> InferenceResult:
    return InferenceResult(
        run_id=request.run_id,
        window_id=request.window.window_id,
        status="failed",
        model_id="unknown",
        frames_used=0,
        received_at_unix_ms=received_at_ms,
        is_synthetic=False,
    )


def post_request(
    url: str, request: InferenceRequest, timeout: float = 120.0
) -> tuple[InferenceResult, int]:
    """POST to {url}/v1/infer. Returns (result, round_trip_ms on the local clock)."""
    body = request.model_dump_json().encode("utf-8")
    http = urllib.request.Request(
        url.rstrip("/") + "/v1/infer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(http, timeout=timeout) as response:
            payload = response.read()
        result = InferenceResult.model_validate_json(payload)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        result = failed_result(request, now_ms())
    return result, int((time.perf_counter() - start) * 1000)
