"""Frame selectors. They see only video pixels and timing, never labels."""

import time
from pathlib import Path

import cv2
import numpy as np

from gigacity_edge.frames import probe

SCAN_SIZE = (160, 90)


def uniform(duration_ms: int, n: int) -> list[int]:
    """n times at the centers of n equal segments of [0, duration_ms)."""
    if n <= 0 or duration_ms <= 0:
        return []
    times = sorted({int((i + 0.5) * duration_ms / n) for i in range(n)})
    return [t for t in times if 0 <= t < duration_ms]


def motion_scores(path: str | Path, scan_fps: float = 4.0) -> list[tuple[int, float]]:
    """(time_ms, mean absolute difference vs the previous scanned frame)."""
    duration_ms, fps, count = probe(path)
    step = max(1, round(fps / scan_fps))
    cap = cv2.VideoCapture(str(path))
    scores: list[tuple[int, float]] = []
    previous = None
    try:
        for index in range(count):
            if not cap.grab():
                break
            if index % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, SCAN_SIZE), cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32)
            time_ms = min(int(index * 1000 / fps), duration_ms - 1)
            if previous is not None:
                scores.append((time_ms, float(np.mean(np.abs(gray - previous)))))
            previous = gray
    finally:
        cap.release()
    return scores


def pick_peaks(scores: list[tuple[int, float]], n: int, duration_ms: int) -> list[int]:
    """Top-n times by score with non-max suppression, padded with uniform times."""
    min_gap = duration_ms / (2 * n) if n else 0
    chosen: list[int] = []
    for time_ms, _ in sorted(scores, key=lambda item: (-item[1], item[0])):
        if len(chosen) == n:
            break
        if all(abs(time_ms - other) >= min_gap for other in chosen):
            chosen.append(time_ms)
    for time_ms in uniform(duration_ms, n * 4):
        if len(chosen) == n:
            break
        if all(abs(time_ms - other) >= min_gap / 2 for other in chosen):
            chosen.append(time_ms)
    for time_ms in uniform(duration_ms, n * 16):
        if len(chosen) == n:
            break
        if time_ms not in chosen:
            chosen.append(time_ms)
    return sorted(t for t in set(chosen) if 0 <= t < duration_ms)


def motion(
    path: str | Path, n: int, duration_ms: int, scan_fps: float = 4.0
) -> tuple[list[int], int]:
    """Motion/scene-change selection. Returns (times_ms, select_ms)."""
    start = time.perf_counter()
    times = pick_peaks(motion_scores(path, scan_fps), n, duration_ms)
    return times, int((time.perf_counter() - start) * 1000)


def select(
    name: str, path: str | Path, n: int, duration_ms: int
) -> tuple[list[int], int]:
    """Dispatch by selector name. Returns (times_ms, select_ms)."""
    if name == "uniform":
        start = time.perf_counter()
        times = uniform(duration_ms, n)
        return times, int((time.perf_counter() - start) * 1000)
    if name == "motion":
        return motion(path, n, duration_ms)
    raise ValueError(f"unknown selector: {name}")
