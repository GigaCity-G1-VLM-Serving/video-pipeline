"""C-owned HTTP entry point for the draft B-to-C inference contract."""

import argparse
import base64
import binascii
import string
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gigacity_contracts import InferenceRequest, InferenceResult

Backend = Callable[[InferenceRequest], str | None]
JPEG_DATA_URI = "data:image/jpeg;base64,"


class BackendError(Exception):
    """Expected inference failure reported by an adapter."""


def validate_frames(request: InferenceRequest) -> None:
    """Validate the proposed inline JPEG transport before calling the backend."""
    for frame in request.frames:
        if frame.codec != "jpeg" or not frame.data_ref.startswith(JPEG_DATA_URI):
            raise ValueError("frames must use inline JPEG data URIs")
        encoded = frame.data_ref.removeprefix(JPEG_DATA_URI)
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("frame data URI must contain valid base64") from error
        if len(payload) != frame.byte_count:
            raise ValueError("frame byte_count must match the decoded payload size")


def build_server(
    backend: Backend,
    *,
    model_id: str,
    is_synthetic: bool,
    host: str = "127.0.0.1",
    port: int = 8200,
) -> ThreadingHTTPServer:
    """Serve the shared contract using an injected inference backend."""

    class Handler(BaseHTTPRequestHandler):
        def send_body(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_body(200, b"ok", "text/plain")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/v1/infer":
                self.send_error(404)
                return

            received_ms = time.time_ns() // 1_000_000
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("request body required")
                request = InferenceRequest.model_validate_json(self.rfile.read(length))
                validate_frames(request)
            except ValueError:
                self.send_body(
                    400, b'{"error":"invalid inference request"}', "application/json"
                )
                return

            started = time.perf_counter()
            try:
                raw_answer = backend(request)
            except BackendError:
                status = "failed"
                raw_answer = None
                answer = None
                frames_used = 0
            else:
                status = "completed"
                words = (raw_answer or "").lower().split(maxsplit=1)
                first_word = words[0].strip(string.punctuation) if words else ""
                answer = first_word if first_word in ("yes", "no") else "unparsed"
                frames_used = len(request.frames)

            result = InferenceResult(
                run_id=request.run_id,
                window_id=request.window.window_id,
                status=status,
                answer_text=raw_answer,
                answer=answer,
                model_id=model_id,
                frames_used=frames_used,
                received_at_unix_ms=received_ms,
                completed_at_unix_ms=time.time_ns() // 1_000_000,
                inference_duration_ms=round((time.perf_counter() - started) * 1000),
                is_synthetic=is_synthetic,
            )
            self.send_body(200, result.model_dump_json().encode(), "application/json")

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--synthetic", action="store_true", required=True)
    args = parser.parse_args()

    def synthetic_answer(_request: InferenceRequest) -> str:
        return "Yes"

    with build_server(
        synthetic_answer,
        model_id="central-synthetic",
        is_synthetic=True,
        port=args.port,
    ) as server:
        print(
            f"Synthetic C endpoint at http://127.0.0.1:{server.server_port}", flush=True
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
