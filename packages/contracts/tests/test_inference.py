"""Draft B <-> C inference API: examples, invalid inputs, and status rules."""

import base64
import json
import re
from functools import cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from gigacity_contracts import (
    BudgetResponse,
    Codec,
    ErrorResponse,
    InferenceRequest,
    InferenceResult,
    PromptVersion,
    Task,
)

CONTRACTS_DOC = Path(__file__).resolve().parents[3] / "docs/contracts.md"
# A JSON code block in the doc, labeled by a preceding `<!-- example: NAME -->`.
EXAMPLE_BLOCK = re.compile(r"<!-- example: (\w+) -->\n```json\n(.*?)\n```", re.DOTALL)


@cache
def doc_examples() -> dict[str, str]:
    text = CONTRACTS_DOC.read_text(encoding="utf-8")
    return dict(EXAMPLE_BLOCK.findall(text))


def load(name: str) -> dict:
    return json.loads(doc_examples()[name])


def test_doc_examples_are_all_tested():
    assert set(doc_examples()) == {
        "inference_request",
        "inference_result",
        "budget_response",
        "error_response",
    }


def data_uri(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode()


@pytest.fixture
def request_data():
    return load("inference_request")


@pytest.fixture
def result_data():
    return load("inference_result")


@pytest.mark.parametrize(
    ("model", "name"),
    [
        (InferenceRequest, "inference_request"),
        (InferenceResult, "inference_result"),
        (BudgetResponse, "budget_response"),
        (ErrorResponse, "error_response"),
    ],
)
def test_examples_validate_and_round_trip(model, name):
    data = load(name)
    message = model.model_validate(data)
    assert json.loads(message.model_dump_json()) == data
    assert model.model_validate_json(message.model_dump_json()) == message


def test_request_parses_shared_enums(request_data):
    request = InferenceRequest.model_validate_json(json.dumps(request_data))
    assert request.task is Task.COLLISION_YN
    assert request.prompt_version is PromptVersion.V1
    assert all(frame.codec is Codec.JPEG for frame in request.frames)


def test_prompt_version_defaults_to_v1(request_data):
    del request_data["prompt_version"]
    assert InferenceRequest.model_validate(request_data).prompt_version == "v1"


def test_result_echoes_request_ids(request_data, result_data):
    request = InferenceRequest.model_validate(request_data)
    result = InferenceResult.model_validate(result_data)
    assert result.request_id == request.request_id
    assert result.run_id == request.run_id
    assert result.window_id == request.window.window_id


@pytest.mark.parametrize(
    "updates",
    [
        {"task": "summary"},
        {"prompt_version": "v9"},
        {"frame_budget": 6},
        {"frame_budget": "4"},
        {"request_id": " "},
        {"query": ""},
        {"total_bytes": 1},
        {"sent_at_unix_ms": -1},
        {"label": "yes"},
    ],
)
def test_invalid_requests_are_rejected(request_data, updates):
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate({**request_data, **updates})


def test_frames_must_be_ordered_and_inside_window(request_data):
    frames = request_data["frames"]
    with pytest.raises(ValidationError, match="strictly increasing"):
        InferenceRequest.model_validate({**request_data, "frames": frames[::-1]})
    late = {**frames[1], "capture_ms": request_data["window"]["end_ms"]}
    with pytest.raises(ValidationError, match="inside"):
        InferenceRequest.model_validate({**request_data, "frames": [frames[0], late]})


def test_frames_must_not_exceed_budget(request_data):
    frame = request_data["frames"][0]
    frames = [{**frame, "capture_ms": 1000 * i} for i in range(5)]
    data = {
        **request_data,
        "frames": frames,
        "total_bytes": 5 * frame["byte_count"],
    }
    with pytest.raises(ValidationError, match="frame_budget"):
        InferenceRequest.model_validate(data)
    assert len(InferenceRequest.model_validate({**data, "frame_budget": 8}).frames) == 5


@pytest.mark.parametrize(
    "frame_updates",
    [
        {"codec": "png"},
        {"data_ref": "https://example.com/frame.jpg"},
        {"data_ref": "data:image/jpeg;base64,?"},
        {"byte_count": 999},
        {"width": 0},
    ],
)
def test_invalid_frames_are_rejected(request_data, frame_updates):
    frames = [{**request_data["frames"][0], **frame_updates}]
    data = {**request_data, "frames": frames, "total_bytes": frames[0]["byte_count"]}
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(data)


def test_byte_count_matches_decoded_payload(request_data):
    payload = b"\xff\xd8 synthetic \xff\xd9"
    frame = {
        **request_data["frames"][0],
        "byte_count": len(payload),
        "data_ref": data_uri(payload),
    }
    data = {**request_data, "frames": [frame], "total_bytes": len(payload)}
    assert InferenceRequest.model_validate(data).total_bytes == len(payload)


ERROR = {"code": "too_many_frames", "message": "16 frames allowed"}


@pytest.mark.parametrize(
    ("status", "updates"),
    [
        ("rejected", {"answer": None, "answer_text": None, "confidence": None}),
        ("failed", {"answer": None, "answer_text": None, "confidence": None}),
        (
            "completed",
            {"answer": "unparsed", "answer_text": "Maybe", "confidence": None},
        ),
        ("completed", {"answer": "no", "answer_text": "No.", "confidence": 82.0}),
    ],
)
def test_valid_status_combinations(result_data, status, updates):
    error = ERROR if status in ("rejected", "failed") else None
    data = {**result_data, "status": status, "error": error, **updates}
    assert InferenceResult.model_validate(data).status == status


@pytest.mark.parametrize(
    "updates",
    [
        {"answer": None, "confidence": None},
        {"error": ERROR},
        {"confidence": None},
        {"confidence": 101.0},
        {"confidence": -1.0},
        {"answer": "unparsed"},
        {"status": "failed", "error": None, "answer": None, "confidence": None},
        {"status": "rejected", "error": ERROR},
        {"error": {"code": "oops", "message": "x"}, "status": "failed"},
        {"config_version": ""},
        {"is_synthetic": True},
        {"yes_probability": 0.5},
    ],
)
def test_invalid_results_are_rejected(result_data, updates):
    with pytest.raises(ValidationError):
        InferenceResult.model_validate({**result_data, **updates})


@pytest.mark.parametrize("field", ["max_frames", "max_image_side", "max_request_bytes"])
def test_budget_limits_must_be_positive(field):
    data = load("budget_response")
    data["budget"][field] = 0
    with pytest.raises(ValidationError):
        BudgetResponse.model_validate(data)
