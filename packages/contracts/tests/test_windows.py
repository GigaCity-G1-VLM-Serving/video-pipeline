"""Executable examples of the draft contract, without media or GPU dependencies."""

import json
from importlib.metadata import version
from pathlib import Path

import pytest
from pydantic import ValidationError

from gigacity_contracts import WindowMetadata

EXAMPLE = Path(__file__).resolve().parents[3] / "docs/examples/window.json"


@pytest.fixture
def window_data():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_installed_contracts_distribution_is_importable():
    assert version("gigacity-contracts") == "0.1.0"


def test_documented_example_validates(window_data):
    window = WindowMetadata.model_validate(window_data)
    assert window.end_ms - window.start_ms == 10000
    assert window.camera_id == "demo-camera-01"


def test_json_serialization_round_trip(window_data):
    window = WindowMetadata.model_validate(window_data)
    assert WindowMetadata.model_validate_json(window.model_dump_json()) == window
    assert json.loads(window.model_dump_json()) == window_data


@pytest.mark.parametrize(
    "updates",
    [
        {"start_ms": -1},
        {"end_ms": 0},
        {"start_ms": 10001},
        {"start_ms": "0"},
        {"start_ms": False},
        {"camera_id": ""},
        {"window_id": "   "},
        {"source_video_id": "\t"},
        {"answer": "Yes"},
    ],
)
def test_invalid_examples_are_rejected(window_data, updates):
    with pytest.raises(ValidationError):
        WindowMetadata.model_validate({**window_data, **updates})


def test_required_field_cannot_be_omitted(window_data):
    del window_data["source_video_id"]
    with pytest.raises(ValidationError):
        WindowMetadata.model_validate(window_data)
