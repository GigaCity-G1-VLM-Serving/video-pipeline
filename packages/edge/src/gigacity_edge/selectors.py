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


GENERIC_PROMPTS = (
    "a traffic accident",
    "a car crash on the road",
    "vehicles colliding at an intersection",
)


def clip_scores(
    path: str | Path, texts: list[str], scan_fps: float = 2.0, scorer=None
) -> list[tuple[int, float]]:
    """(time_ms, max cosine similarity to any text) for frames scanned at scan_fps."""
    if scorer is None:
        from gigacity_edge.clip import get_scorer

        scorer = get_scorer()
    duration_ms, fps, count = probe(path)
    step = max(1, round(fps / scan_fps))
    times: list[int] = []
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(path))
    try:
        for index in range(count):
            if not cap.grab():
                break
            if index % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            times.append(min(int(index * 1000 / fps), duration_ms - 1))
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        return []
    similarity = scorer.encode_images(frames) @ scorer.encode_text(list(texts)).T
    return list(zip(times, similarity.max(axis=1).astype(float).tolist()))


def clip(
    path: str | Path,
    n: int,
    duration_ms: int,
    query: str | None = None,
    scan_fps: float = 2.0,
    scorer=None,
) -> tuple[list[int], int]:
    """CLIP text-frame relevance selection, using the same peak picking as motion.

    Uses the question text when given, otherwise generic collision prompts. Sees
    only video pixels and the question, never labels. Returns (times_ms, select_ms).
    """
    start = time.perf_counter()
    texts = [query] if query else list(GENERIC_PROMPTS)
    scores = clip_scores(path, texts, scan_fps, scorer)
    times = pick_peaks(scores, n, duration_ms)
    return times, int((time.perf_counter() - start) * 1000)


def motion(
    path: str | Path, n: int, duration_ms: int, scan_fps: float = 4.0
) -> tuple[list[int], int]:
    """Motion/scene-change selection. Returns (times_ms, select_ms)."""
    start = time.perf_counter()
    times = pick_peaks(motion_scores(path, scan_fps), n, duration_ms)
    return times, int((time.perf_counter() - start) * 1000)


SELECTORS = ("uniform", "motion", "clip", "clip-generic")


def select(
    name: str, path: str | Path, n: int, duration_ms: int, query: str | None = None
) -> tuple[list[int], int]:
    """Dispatch by selector name. Returns (times_ms, select_ms).

    ``query`` is used only by "clip"; "clip-generic" ignores it (withheld-query mode).
    """
    if name == "uniform":
        start = time.perf_counter()
        times = uniform(duration_ms, n)
        return times, int((time.perf_counter() - start) * 1000)
    if name == "motion":
        return motion(path, n, duration_ms)
    if name == "clip":
        return clip(path, n, duration_ms, query)
    if name == "clip-generic":
        return clip(path, n, duration_ms, None)
    raise ValueError(f"unknown selector: {name}")
