"""Exercise C's HTTP boundary through the existing A/B client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gigacity_central.backends import BackendError, vllm_backend
from gigacity_central.server import build_server
from gigacity_contracts import InferenceRequest
from gigacity_edge.client import post_request

EXAMPLE = Path(__file__).resolve().parents[3] / "docs/examples/inference_request.json"


@pytest.fixture
def start_server():
    servers = []
    threads = []

    def start(backend, *, is_synthetic=True):
        server = build_server(
            backend, model_id="test-backend", is_synthetic=is_synthetic, port=0
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        servers.append(server)
        threads.append(thread)
        thread.start()
        return f"http://127.0.0.1:{server.server_port}"

    yield start

    for server, thread in zip(servers, threads, strict=True):
        server.shutdown()
        thread.join()
        server.server_close()


def test_edge_client_gets_correlated_result(start_server):
    request = InferenceRequest.model_validate_json(EXAMPLE.read_bytes())

    def backend(received):
        assert received == request
        return " Yes."

    url = start_server(backend)
    with urlopen(url + "/health", timeout=2) as health:
        assert health.read() == b"ok"

    result, _ = post_request(url, request, timeout=2)
    assert result.status == "completed"
    assert result.answer == "yes"
    assert result.answer_text == " Yes."
    assert result.is_synthetic
    assert result.model_id == "test-backend"
    assert result.run_id == request.run_id
    assert result.window_id == request.window.window_id
    assert result.frames_used == len(request.frames)


def test_backend_failure_is_a_correlated_result(start_server):
    request = InferenceRequest.model_validate_json(EXAMPLE.read_bytes())

    def backend(_request):
        raise BackendError("backend unavailable")

    result, _ = post_request(start_server(backend), request, timeout=2)
    assert result.status == "failed"
    assert result.answer is None
    assert result.frames_used == 0
    assert result.run_id == request.run_id
    assert result.window_id == request.window.window_id
    assert result.model_id == "test-backend"
    assert result.is_synthetic


@pytest.mark.parametrize("invalid", ["json", "uri", "byte_count"])
def test_invalid_request_never_reaches_backend(start_server, invalid):
    def backend(_request):
        pytest.fail("invalid request reached the backend")

    data = json.loads(EXAMPLE.read_bytes())
    if invalid == "uri":
        data["frames"][0]["data_ref"] = "data:image/jpeg;base64,?"
    if invalid == "byte_count":
        data["frames"][0]["byte_count"] += 1
        data["total_bytes"] += 1
    body = b"{" if invalid == "json" else json.dumps(data).encode()
    request = Request(
        start_server(backend) + "/v1/infer",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=2)
    with error.value as response:
        assert response.code == 400
        assert json.load(response) == {"error": "invalid inference request"}


@pytest.mark.parametrize("upstream_status", [200, 503])
def test_edge_client_through_vllm_adapter(start_server, upstream_status):
    request = InferenceRequest.model_validate_json(EXAMPLE.read_bytes())
    received = []

    class FakeVllm(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers.get("Authorization"), payload))
            self.send_response(upstream_status)
            self.end_headers()
            if upstream_status == 200:
                self.wfile.write(b'{"choices":[{"message":{"content":"No."}}]}')

    with ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm) as upstream:
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            backend = vllm_backend(
                f"http://127.0.0.1:{upstream.server_port}",
                "pilot-model",
                api_key="test-token",
                timeout=2,
            )
            url = start_server(backend, is_synthetic=False)
            result, _ = post_request(url, request, timeout=2)
        finally:
            upstream.shutdown()
            thread.join()

    [(path, auth, payload)] = received
    assert path == "/v1/chat/completions"
    assert auth == "Bearer test-token"
    assert payload["model"] == "pilot-model"
    [message] = payload["messages"]
    first_time, first_image, second_time, second_image, query = message["content"]
    assert first_time["text"] == "Frame 1 at 1.2 s:"
    assert second_time["text"] == "Frame 2 at 7.4 s:"
    assert first_image["image_url"]["url"] == request.frames[0].data_ref
    assert second_image["image_url"]["url"] == request.frames[1].data_ref
    assert query["text"] == f"{request.query} Answer only Yes or No."
    assert result.run_id == request.run_id
    assert result.window_id == request.window.window_id
    assert result.model_id == "test-backend"
    assert not result.is_synthetic
    if upstream_status == 200:
        assert result.status == "completed"
        assert result.answer == "no"
    else:
        assert result.status == "failed"
        assert result.answer is None
