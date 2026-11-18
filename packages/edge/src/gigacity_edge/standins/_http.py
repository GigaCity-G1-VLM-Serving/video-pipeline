"""Shared HTTP plumbing for the C stand-ins: POST /v1/infer and GET /health."""

import json
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import ValidationError

from gigacity_contracts import InferenceRequest, InferenceResult

Handler = Callable[[InferenceRequest, int], InferenceResult]


def unix_ms() -> int:
    return time.time_ns() // 1_000_000


def make_server(host: str, port: int, handle: Handler) -> ThreadingHTTPServer:
    """Build a threaded server that validates requests and calls ``handle``."""

    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # quiet default per-request logging

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/v1/infer":
                self._send(404, b"not found", "text/plain")
                return
            received = unix_ms()
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                request = InferenceRequest.model_validate_json(raw)
            except ValidationError as error:
                body = json.dumps(
                    {"error": error.errors(include_url=False)}, default=str
                )
                self._send(400, body.encode(), "application/json")
                return
            result = handle(request, received)
            self._send(200, result.model_dump_json().encode(), "application/json")

    server = ThreadingHTTPServer((host, port), RequestHandler)
    server.daemon_threads = True
    return server
