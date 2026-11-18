"""Event-centred diagnostic. **USES EVALUATION LABELS BY DESIGN; NOT A SELECTOR.**

For each pilot question, place 8 frames inside TAR's annotated event interval (±1 s)
for that video and ask the question through the same ``/v1/infer`` service path as
``gigacity_edge.run``. It answers "can the model see the event when the evidence is
certainly there?", which bounds what any selector could achieve. Never use its frame
times, or anything derived from labels, as an online selection policy.

Output matches ``run.py`` (edge.jsonl, results.jsonl, run.json), with selector
``event-oracle`` and window IDs f"{run_id}:{question_id}:b8", so ``score`` works on it.

Run: python -m gigacity_edge.analysis.oracle --pilot data/pilot80/pilot_questions.jsonl \
         --tar data/tar/temporal_localization.json --videos data/videos \
         --url http://127.0.0.1:8200 --run-id q30-oracle --out results/q30-oracle
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gigacity_contracts import EdgeRecord, InferenceRequest, WindowMetadata
from gigacity_edge.analysis.capture import TOLERANCE_MS, load_intervals
from gigacity_edge.client import now_ms, post_request
from gigacity_edge.frames import probe, read_frames_at, selected_frame
from gigacity_edge.run import git_commit, load_server_config

SELECTOR = "event-oracle"
BUDGET = 8


def oracle_times(
    interval: tuple[int, int], duration_ms: int, n: int = BUDGET
) -> list[int]:
    """n evenly spaced times inside the event interval widened by TOLERANCE_MS."""
    start = max(0, interval[0] - TOLERANCE_MS)
    end = min(duration_ms - 1, interval[1] + TOLERANCE_MS)
    if end <= start:
        return []
    return sorted({start + int((i + 0.5) * (end - start) / n) for i in range(n)})


def build(run_id: str, row: dict, video: Path, interval: tuple[int, int]):
    duration_ms, _, _ = probe(video)
    times = oracle_times(interval, duration_ms)
    if not times:
        return None
    started = time.perf_counter()
    images = read_frames_at(video, times)
    decode_ms = int((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    frames = [selected_frame(t, image) for t, image in zip(times, images)]
    encode_ms = int((time.perf_counter() - started) * 1000)
    window = WindowMetadata(
        window_id=f"{run_id}:{row['question_id']}:b{BUDGET}",
        camera_id="pilot-cam",
        source_video_id=row["video_id"],
        start_ms=0,
        end_ms=duration_ms,
    )
    request = InferenceRequest(
        run_id=run_id,
        window=window,
        frames=frames,
        total_bytes=sum(f.byte_count for f in frames),
        selector=SELECTOR,
        query=row["question"],
        sent_at_unix_ms=now_ms(),
    )
    return request, times, decode_ms, encode_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--server-config", type=Path, default=None)
    args = parser.parse_args(argv)

    started = now_ms()
    intervals = load_intervals(args.tar)
    rows = [
        json.loads(line)
        for line in args.pilot.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jobs, skipped = [], []
    for row in rows:
        video = args.videos / row["video_id"]
        interval = intervals.get(row["video_id"])
        built = (
            build(args.run_id, row, video, interval)
            if video.exists() and interval
            else None
        )
        if built is None:
            skipped.append(row["question_id"])
        else:
            jobs.append(built)

    def send(job):
        request, times, decode_ms, encode_ms = job
        request = request.model_copy(update={"sent_at_unix_ms": now_ms()})
        result, round_trip_ms = post_request(args.url, request, args.timeout)
        record = EdgeRecord(
            run_id=args.run_id,
            window=request.window,
            status="sent",
            selector=SELECTOR,
            frame_budget=BUDGET,
            selected_capture_ms=times,
            total_bytes=request.total_bytes,
            decode_ms=decode_ms,
            select_ms=0,
            encode_ms=encode_ms,
            round_trip_ms=round_trip_ms,
        )
        return record, result

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
        "note": "event-oracle diagnostic: frame times come from evaluation labels",
        "git_commit": git_commit(),
        "started_unix_ms": started,
        "finished_unix_ms": now_ms(),
        "requests": len(jobs),
        "statuses": statuses,
        "skipped_questions": skipped,
        "pilot": str(args.pilot),
        "server": load_server_config(args.server_config),
    }
    (args.out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("requests", "statuses")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
