"""Run whole-video pilot requests: select, encode, send, and log JSONL records.

Reads only the pilot questions (no labels). Each question becomes one whole-video
window per budget.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from gigacity_contracts import (
    EdgeRecord,
    InferenceRequest,
    SelectedFrame,
    WindowMetadata,
)
from gigacity_edge.client import now_ms, post_request
from gigacity_edge.frames import probe, read_frames_at, selected_frame
from gigacity_edge.selectors import select


@dataclass(frozen=True)
class Prepared:
    duration_ms: int
    times_ms: list[int]
    frames: list[SelectedFrame]
    decode_ms: int
    select_ms: int
    encode_ms: int


def prepare(video: Path, selector: str, budget: int) -> Prepared:
    duration_ms, _, _ = probe(video)
    times, select_ms = select(selector, video, budget, duration_ms)
    start = time.perf_counter()
    images = read_frames_at(video, times)
    decode_ms = int((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    frames = [selected_frame(t, image) for t, image in zip(times, images)]
    encode_ms = int((time.perf_counter() - start) * 1000)
    return Prepared(duration_ms, times, frames, decode_ms, select_ms, encode_ms)


def build_request(
    run_id: str,
    question_id: str,
    video_id: str,
    question: str,
    budget: int,
    selector: str,
    prepared: Prepared,
) -> InferenceRequest:
    window = WindowMetadata(
        window_id=f"{run_id}:{question_id}:b{budget}",
        camera_id="pilot-cam",
        source_video_id=video_id,
        start_ms=0,
        end_ms=prepared.duration_ms,
    )
    return InferenceRequest(
        run_id=run_id,
        window=window,
        frames=prepared.frames,
        total_bytes=sum(frame.byte_count for frame in prepared.frames),
        selector=selector,
        query=question,
        sent_at_unix_ms=now_ms(),
    )


def edge_record(
    request: InferenceRequest,
    prepared: Prepared,
    budget: int,
    round_trip_ms: int | None,
) -> EdgeRecord:
    return EdgeRecord(
        run_id=request.run_id,
        window=request.window,
        status="sent",
        selector=request.selector,
        frame_budget=budget,
        selected_capture_ms=prepared.times_ms,
        total_bytes=request.total_bytes,
        decode_ms=prepared.decode_ms,
        select_ms=prepared.select_ms,
        encode_ms=prepared.encode_ms,
        round_trip_ms=round_trip_ms,
    )


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_pilot(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(
                    {key: row[key] for key in ("question_id", "video_id", "question")}
                )
    return rows[:limit] if limit else rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--selector", choices=["uniform", "motion"], required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    started = now_ms()
    rows = load_pilot(args.pilot, args.limit)
    cache: dict[tuple[str, int], Prepared] = {}
    jobs, missing = [], []
    for row in rows:
        video = args.videos / row["video_id"]
        if not video.exists():
            missing.append(row["video_id"])
            continue
        for budget in args.budgets:
            key = (row["video_id"], budget)
            if key not in cache:
                cache[key] = prepare(video, args.selector, budget)
            request = build_request(
                args.run_id,
                row["question_id"],
                row["video_id"],
                row["question"],
                budget,
                args.selector,
                cache[key],
            )
            jobs.append((request, cache[key], budget))
    prepared_at = now_ms()

    def send(job):
        request, prepared, budget = job
        request = request.model_copy(update={"sent_at_unix_ms": now_ms()})
        result, round_trip_ms = post_request(args.url, request, args.timeout)
        return edge_record(request, prepared, budget, round_trip_ms), result

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        outcomes = list(pool.map(send, jobs))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "edge.jsonl").open("w", encoding="utf-8") as handle:
        for record, _ in outcomes:
            handle.write(record.model_dump_json() + "\n")
    with (args.out / "results.jsonl").open("w", encoding="utf-8") as handle:
        for _, result in outcomes:
            handle.write(result.model_dump_json() + "\n")
    statuses: dict[str, int] = {}
    for _, result in outcomes:
        statuses[result.status] = statuses.get(result.status, 0) + 1
    summary = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "git_commit": git_commit(),
        "started_unix_ms": started,
        "prepared_unix_ms": prepared_at,
        "finished_unix_ms": now_ms(),
        "requests": len(jobs),
        "statuses": statuses,
        "missing_videos": missing,
    }
    (args.out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("requests", "statuses")}))
    if missing:
        print(f"missing videos: {len(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
