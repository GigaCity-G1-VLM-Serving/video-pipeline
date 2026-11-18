"""Executable examples of the proposed inference handoff, using synthetic payloads."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gigacity_contracts import EdgeRecord, InferenceRequest, InferenceResult

EXAMPLES = Path(__file__).resolve().parents[3] / "docs/examples"


def load(name):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("model", "name"),
    [
        (InferenceRequest, "inference_request.json"),
        (InferenceResult, "inference_result.json"),
        (EdgeRecord, "edge_record.json"),
    ],
)
def test_examples_validate_and_round_trip(model, name):
    data = load(name)
    instance = model.model_validate(data)
    assert model.model_validate_json(instance.model_dump_json()) == instance
    assert json.loads(instance.model_dump_json()) == data


def request_with(mutate):
    data = copy.deepcopy(load("inference_request.json"))
    mutate(data)
    return data


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["frames"][0].update(capture_ms=10000),
        lambda d: d["frames"][1].update(capture_ms=1200),
        lambda d: d["frames"][0].update(byte_count=9),
        lambda d: d["frames"][0].update(byte_count=0),
        lambda d: d["frames"][0].update(data_ref="  "),
        lambda d: d["frames"][0].update(codec="gif"),
        lambda d: d.update(total_bytes=21),
        lambda d: d.update(frames=d["frames"] * 3),
        lambda d: d.update(frames=[]),
        lambda d: d.update(query="  "),
        lambda d: d.update(budget_id=""),
        lambda d: d.update(sent_at_unix_ms="1794355200000"),
        lambda d: d.update(answer="yes"),
    ],
)
def test_invalid_requests_are_rejected(mutate):
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(request_with(mutate))


def test_query_budget_and_deadline_are_optional():
    data = request_with(
        lambda d: d.update(query=None, budget_id=None, deadline_unix_ms=None)
    )
    assert InferenceRequest.model_validate(data).query is None


def test_completed_result_requires_parsed_answer():
    data = load("inference_result.json")
    data["answer"] = None
    with pytest.raises(ValidationError):
        InferenceResult.model_validate(data)


def test_failed_result_may_omit_answer():
    data = load("inference_result.json")
    data.update(status="failed", answer=None, answer_text=None)
    assert InferenceResult.model_validate(data).status == "failed"


@pytest.mark.parametrize(
    "update",
    [
        {"selected_capture_ms": [7400, 1200]},
        {"selected_capture_ms": [12000]},
        {"selected_capture_ms": [1, 2, 3, 4, 5]},
        {"status": "lost"},
    ],
)
def test_invalid_edge_records_are_rejected(update):
    with pytest.raises(ValidationError):
        EdgeRecord.model_validate({**load("edge_record.json"), **update})


def test_dropped_window_is_recordable():
    data = {
        **load("edge_record.json"),
        "status": "dropped",
        "selected_capture_ms": [],
        "total_bytes": 0,
        "round_trip_ms": None,
    }
    assert EdgeRecord.model_validate(data).status == "dropped"
