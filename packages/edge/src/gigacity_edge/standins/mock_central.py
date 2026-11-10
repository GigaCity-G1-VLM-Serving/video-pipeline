"""Mock C server returning synthetic results; never a serving measurement.

Run: python -m gigacity_edge.standins.mock_central --port 8100
"""

import argparse
import hashlib
import time
from http.server import ThreadingHTTPServer

from gigacity_contracts import InferenceRequest, InferenceResult
from gigacity_edge.standins._http import make_server, unix_ms

MODEL_ID = "mock-central"


def pseudo_answer(window_id: str) -> str:
    """Deterministic yes/no from the window ID, unrelated to any label."""
    return "yes" if hashlib.sha256(window_id.encode()).digest()[0] % 2 else "no"


def mock_result(
    request: InferenceRequest,
    received_at_unix_ms: int,
    base_ms: int = 50,
    per_frame_ms: int = 20,
) -> InferenceResult:
    delay_ms = base_ms + per_frame_ms * len(request.frames)
    time.sleep(delay_ms / 1000)
    answer = pseudo_answer(request.window.window_id)
    return InferenceResult(
        run_id=request.run_id,
        window_id=request.window.window_id,
        status="completed",
        answer_text=answer.capitalize(),
        answer=answer,
        model_id=MODEL_ID,
        frames_used=len(request.frames),
        prompt_tokens=None,
        completion_tokens=None,
        queue_duration_ms=0,
        inference_duration_ms=delay_ms,
        received_at_unix_ms=received_at_unix_ms,
        completed_at_unix_ms=unix_ms(),
        is_synthetic=True,
    )


def build_server(
    host: str = "127.0.0.1", port: int = 8100, base_ms: int = 50, per_frame_ms: int = 20
) -> ThreadingHTTPServer:
    return make_server(
        host,
        port,
        lambda request, received: mock_result(request, received, base_ms, per_frame_ms),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--base-ms", type=int, default=50)
    parser.add_argument("--per-frame-ms", type=int, default=20)
    args = parser.parse_args()
    server = build_server(args.host, args.port, args.base_ms, args.per_frame_ms)
    print(f"mock-central (SYNTHETIC) on http://{args.host}:{server.server_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
