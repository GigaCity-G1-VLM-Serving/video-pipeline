"""Exercise C's HTTP boundary through the existing A/B client."""

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gigacity_central.server import BackendError, build_server
from gigacity_contracts import InferenceRequest
from gigacity_edge.client import post_request

EXAMPLE = Path(__file__).resolve().parents[3] / "docs/examples/inference_request.json"


@pytest.fixture
def start_server():
    servers = []
    threads = []

    def start(backend):
        server = build_server(
            backend, model_id="test-backend", is_synthetic=True, port=0
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
