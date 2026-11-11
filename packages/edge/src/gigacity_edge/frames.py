"""Decode, resize, and JPEG-encode frames at chosen source-video times."""

import base64
from pathlib import Path

import cv2
import numpy as np

from gigacity_contracts import SelectedFrame

MAX_SIDE = 448
PATCH = 28
JPEG_QUALITY = 85


def probe(path: str | Path) -> tuple[int, float, int]:
    """Return (duration_ms, fps, frame_count) for a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f"cannot read fps/frame count: {path}")
    return int(count * 1000 / fps), float(fps), count


def frame_index(time_ms: int, fps: float, frame_count: int) -> int:
    return min(max(round(time_ms * fps / 1000), 0), frame_count - 1)


def read_frames_at(path: str | Path, times_ms: list[int]) -> list[np.ndarray]:
    """Read BGR frames nearest to each time, reading sequentially."""
    _, fps, count = probe(path)
    wanted = [frame_index(t, fps, count) for t in times_ms]
    targets = sorted(set(wanted))
    found: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    try:
        index, last = 0, None
        for target in targets:
            while index <= target:
                ok, frame = cap.read()
                if not ok:
                    break
                last = frame
                index += 1
            if last is None:
                raise ValueError(f"no frames decoded from {path}")
            found[target] = last
    finally:
        cap.release()
    return [found[i] for i in wanted]


def model_size(width: int, height: int) -> tuple[int, int]:
    """Fit inside MAX_SIDE x MAX_SIDE, keep aspect, round down to PATCH multiples."""
    scale = min(MAX_SIDE / width, MAX_SIDE / height, 1.0)
    w = max(PATCH, round(width * scale) // PATCH * PATCH)
    h = max(PATCH, round(height * scale) // PATCH * PATCH)
    return w, h


def resize_for_model(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    w, h = model_size(width, height)
    if (w, h) == (width, height):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()


def data_uri(data: bytes) -> str:
    """Inline JPEG payload for SelectedFrame.data_ref (proposed v1 transport)."""
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def selected_frame(time_ms: int, image: np.ndarray, quality: int = JPEG_QUALITY):
    """Resize and JPEG-encode one decoded frame as a contract SelectedFrame."""
    small = resize_for_model(image)
    data = encode_jpeg(small, quality)
    return SelectedFrame(
        capture_ms=time_ms,
        codec="jpeg",
        width=small.shape[1],
        height=small.shape[0],
        byte_count=len(data),
        data_ref=data_uri(data),
    )


def encode_frames(
    path: str | Path, times_ms: list[int], quality: int = JPEG_QUALITY
) -> list[SelectedFrame]:
    """Decode, resize, and encode frames at times_ms as contract frames."""
    images = read_frames_at(path, times_ms)
    return [selected_frame(t, image, quality) for t, image in zip(times_ms, images)]
