"""Stand-in C server, vLLM shim helpers, and stand-in scorer, using synthetic data."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gigacity_contracts import InferenceRequest, InferenceResult
from gigacity_edge.analysis import score
from gigacity_edge.standins import mock_central, vllm_shim

EXAMPLES = Path(__file__).resolve().parents[3] / "docs/examples"


def example_request() -> InferenceRequest:
    text = (EXAMPLES / "inference_request.json").read_text(encoding="utf-8")
    return InferenceRequest.model_validate_json(text)


@pytest.fixture
def mock_server():
    server = mock_central.build_server(port=0, base_ms=1, per_frame_ms=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def post(url: str, body: bytes):
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    return urllib.request.urlopen(request, timeout=5)


def test_mock_server_returns_synthetic_result(mock_server):
    request = example_request()
    with post(mock_server + "/v1/infer", request.model_dump_json().encode()) as resp:
        result = InferenceResult.model_validate_json(resp.read())
    assert result.is_synthetic
    assert result.window_id == request.window.window_id
    assert result.frames_used == len(request.frames)
    assert result.answer in ("yes", "no")
    with urllib.request.urlopen(mock_server + "/health", timeout=5) as resp:
        assert resp.read() == b"ok"


def test_mock_server_rejects_invalid_request(mock_server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post(mock_server + "/v1/infer", b'{"run_id": "x"}')
    assert error.value.code == 400


def test_chat_request_interleaves_captions_and_images():
    payload = vllm_shim.build_chat_request(example_request(), "qwen2.5-vl-7b")
    content = payload["messages"][0]["content"]
    assert payload["temperature"] == 0 and payload["max_tokens"] == 16
    assert content[0] == {"type": "text", "text": "Frame 1 at 1.2 s:"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,AAECAwQFBgc="
    assert content[2]["text"] == "Frame 2 at 7.4 s:"
    assert content[-1]["text"].endswith("window? Answer only Yes or No.")
    assert len(content) == 2 * 2 + 1


def test_chat_request_without_query_uses_default_question():
    request = example_request().model_copy(update={"query": None})
    content = vllm_shim.build_chat_request(request, "m")["messages"][0]["content"]
    assert content[-1]["text"] == vllm_shim.DEFAULT_QUERY + vllm_shim.ANSWER_SUFFIX


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", "yes"),
        ("  no.", "no"),
        ("YES, a collision occurs", "yes"),
        ("Maybe", "unparsed"),
        ("", "unparsed"),
        (None, "unparsed"),
    ],
)
def test_parse_answer(text, expected):
    assert vllm_shim.parse_answer(text) == expected


def test_shim_reports_failure_when_vllm_unreachable():
    result = vllm_shim.shim_result(
        example_request(), 0, "http://127.0.0.1:9", None, timeout_s=1
    )
    assert result.status == "failed" and result.answer is None
    assert not result.is_synthetic


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def edge_row(question_id: str, round_trip_ms: int | None = 100) -> dict:
    return {
        "run_id": "r1",
        "window": {
            "window_id": f"r1:{question_id}:b4",
            "camera_id": "cam",
            "source_video_id": "vid",
            "start_ms": 0,
            "end_ms": 10000,
        },
        "status": "sent",
        "selector": "uniform",
        "frame_budget": 4,
        "selected_capture_ms": [0, 5000],
        "total_bytes": 200,
        "decode_ms": 1,
        "select_ms": 1,
        "encode_ms": 1,
        "round_trip_ms": round_trip_ms,
        "simulated_network_ms": None,
    }


def result_row(question_id: str, answer: str) -> dict:
    return {
        "run_id": "r1",
        "window_id": f"r1:{question_id}:b4",
        "status": "completed",
        "answer_text": answer,
        "answer": answer,
        "model_id": "mock-central",
        "frames_used": 2,
        "prompt_tokens": 300,
        "completion_tokens": 1,
        "queue_duration_ms": 0,
        "inference_duration_ms": 10,
        "received_at_unix_ms": 0,
        "completed_at_unix_ms": 10,
        "is_synthetic": True,
    }


def test_score_counts_missing_results_as_misses(tmp_path):
    run = tmp_path / "r1"
    run.mkdir()
    write_jsonl(run / "edge.jsonl", [edge_row("q1"), edge_row("q2"), edge_row("q3")])
    write_jsonl(
        run / "results.jsonl", [result_row("q1", "yes"), result_row("q2", "no")]
    )
    labels = {"q1": "Yes", "q2": "Yes", "q3": "No"}

    [summary] = score.summarize(score.load_rows([run], labels, {}))

    assert summary["n"] == 3
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["yes_recall"] == pytest.approx(1 / 2)
    assert summary["no_accuracy"] == 0
    assert summary["failures"] == 1
    assert summary["p50_round_trip_ms"] == 100
    markdown = score.to_markdown([summary], synthetic=True)
    assert "SYNTHETIC" in markdown and "| uniform | 4 | all | 3 |" in markdown


def test_question_id_parsing():
    assert score.question_id_from("run:1:q-7:b16", "run:1") == "q-7"
    with pytest.raises(ValueError):
        score.question_id_from("other:q:b4", "run")
