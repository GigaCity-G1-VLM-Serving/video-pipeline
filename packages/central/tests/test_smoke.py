"""Unit tests for the serving smoke-test harness; no server or video needed."""

import pytest

from gigacity_central.smoke import (
    ANSWER_INSTRUCTION,
    COLLISION_PROMPT,
    Frame,
    Window,
    build_request,
    parse_answer,
    uniform_capture_times,
)

WINDOW = Window(
    window_id="run1:cam01:10-20",
    camera_id="cam01",
    source_video_id="example.mp4",
    start_ms=10_000,
    end_ms=20_000,
)


@pytest.mark.parametrize("budget", [1, 4, 8, 16])
def test_capture_times_stay_inside_the_half_open_window(budget: int) -> None:
    times = uniform_capture_times(WINDOW, budget)
    assert len(times) == budget
    assert all(WINDOW.start_ms <= t < WINDOW.end_ms for t in times)


def test_capture_times_strictly_increase() -> None:
    times = uniform_capture_times(WINDOW, 16)
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_capture_times_are_evenly_spaced() -> None:
    times = uniform_capture_times(WINDOW, 4)
    assert times == [11_250, 13_750, 16_250, 18_750]


def test_zero_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        uniform_capture_times(WINDOW, 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", "yes"),
        ("no", "no"),
        ("No.", "no"),
        ("Yes, a collision occurred", "yes"),
        ("I cannot determine that", None),
        ("", None),
    ],
)
def test_answer_parsing(text: str, expected: str | None) -> None:
    assert parse_answer(text) == expected


def test_request_is_deterministic_and_carries_every_frame() -> None:
    frames = [Frame(capture_ms=t, jpeg=b"\xff\xd8fake") for t in (0, 1200, 2500)]
    payload = build_request("test-model", frames, COLLISION_PROMPT)

    assert payload["temperature"] == 0.0
    assert payload["seed"] == 0
    assert payload["max_tokens"] == 16
    content = payload["messages"][0]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    texts = [part for part in content if part["type"] == "text"]
    assert len(images) == len(frames)
    assert all(
        part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for part in images
    )
    assert texts[-1]["text"] == f"{COLLISION_PROMPT}\n{ANSWER_INSTRUCTION}"


def test_each_image_is_preceded_by_its_capture_time() -> None:
    """Decision 12: a 'Frame i at t s:' line before each image, in order."""
    frames = [Frame(capture_ms=t, jpeg=b"\xff\xd8fake") for t in (0, 1200, 2500)]
    content = build_request("m", frames, COLLISION_PROMPT)["messages"][0]["content"]

    labels = [p["text"] for p in content[:-1] if p["type"] == "text"]
    assert labels == ["Frame 1 at 0.0 s:", "Frame 2 at 1.2 s:", "Frame 3 at 2.5 s:"]
    kinds = [p["type"] for p in content[:-1]]
    assert kinds == ["text", "image_url"] * len(frames)


def test_frame_byte_count_tracks_the_encoded_payload() -> None:
    frame = Frame(capture_ms=0, jpeg=b"0123456789")
    assert frame.byte_count == 10
