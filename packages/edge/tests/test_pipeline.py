"""Edge pipeline tests on a synthetic generated video; no downloaded media."""

import base64
import socket

import cv2
import numpy as np
import pytest

from gigacity_contracts import InferenceRequest
from gigacity_edge.client import post_request
from gigacity_edge.frames import (
    encode_jpeg,
    model_size,
    probe,
    read_frames_at,
    resize_for_model,
)
from gigacity_edge.run import build_request, edge_record, prepare
from gigacity_edge.selectors import motion, pick_peaks, uniform

FPS = 10
SECONDS = 3
JUMP_MS = 2000


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """640x360, 3 s at 10 fps; a square drifts slowly, then jumps at 2.0 s."""
    path = tmp_path_factory.mktemp("video") / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (640, 360)
    )
    for index in range(FPS * SECONDS):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        x = 40 + index if index < FPS * JUMP_MS // 1000 else 480
        cv2.rectangle(frame, (x, 120), (x + 120, 240), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_probe(video):
    duration_ms, fps, count = probe(video)
    assert count == FPS * SECONDS
    assert fps == pytest.approx(FPS)
    assert duration_ms == SECONDS * 1000


@pytest.mark.parametrize(
    ("size", "expected"),
    [((640, 360), (448, 252)), ((1920, 1080), (448, 252)), ((300, 300), (280, 280))],
)
def test_model_size(size, expected):
    assert model_size(*size) == expected


def test_resize_and_jpeg_round_trip(video):
    image = resize_for_model(read_frames_at(video, [0])[0])
    assert image.shape[:2] == (252, 448)
    data = encode_jpeg(image)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape


def test_uniform_times():
    assert uniform(10000, 4) == [1250, 3750, 6250, 8750]
    assert uniform(10000, 0) == []


def test_motion_picks_the_jump(video):
    times, select_ms = motion(video, 1, SECONDS * 1000, scan_fps=FPS)
    assert select_ms >= 0
    assert times and abs(times[0] - JUMP_MS) <= 200


def test_motion_returns_sorted_unique_times_within_video(video):
    times, _ = motion(video, 8, SECONDS * 1000, scan_fps=FPS)
    assert len(times) == 8
    assert times == sorted(set(times))
    assert all(0 <= t < SECONDS * 1000 for t in times)


def test_pick_peaks_pads_with_uniform_times():
    assert len(pick_peaks([(500, 1.0)], 4, 10000)) == 4


def test_request_from_pipeline_validates(video):
    prepared = prepare(video, "uniform", 4)
    request = build_request(
        "run", "q1", "synthetic.avi", "Is there a crash?", 4, "uniform", prepared
    )
    again = InferenceRequest.model_validate_json(request.model_dump_json())
    assert again == request
    assert request.total_bytes == sum(
        len(base64.b64decode(f.data_b64)) for f in request.frames
    )
    assert edge_record(request, prepared, None).selected_capture_ms == prepared.times_ms


def test_client_failure_returns_failed_result(video):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    prepared = prepare(video, "uniform", 4)
    request = build_request(
        "run", "q1", "v", "Is there a crash?", 4, "uniform", prepared
    )
    result, round_trip_ms = post_request(f"http://127.0.0.1:{port}", request, 2.0)
    assert result.status == "failed"
    assert result.window_id == request.window.window_id
    assert round_trip_ms >= 0
