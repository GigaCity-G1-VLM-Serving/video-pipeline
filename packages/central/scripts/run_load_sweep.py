"""Measure client-visible serving load through C with one question per video."""

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from gigacity_contracts import InferenceRequest
from gigacity_edge.analysis.score import percentile, read_jsonl
from gigacity_edge.client import now_ms, post_request
from gigacity_edge.run import build_request, git_commit, load_server_config, prepare
from gigacity_edge.selectors import SELECTORS


@dataclass(frozen=True)
class RequestTiming:
    window_id: str
    status: str
    answer: str | None
    model_id: str
    is_synthetic: bool
    round_trip_ms: int
    frames_used: int


def measure(url: str, request: InferenceRequest, timeout: float) -> RequestTiming:
    outgoing = request.model_copy(update={"sent_at_unix_ms": now_ms()})
    result, rtt = post_request(url, outgoing, timeout)
    if (result.run_id, result.window_id) != (request.run_id, request.window.window_id):
        raise ValueError(f"uncorrelated response for {request.window.window_id}")
    return RequestTiming(
        result.window_id,
        result.status,
        result.answer,
        result.model_id,
        result.is_synthetic,
        rtt,
        result.frames_used,
    )


def check_backend(rows: list[RequestTiming], args: argparse.Namespace) -> None:
    for row in rows:
        if row.status == "completed" and (
            row.model_id != args.model
            or row.is_synthetic != args.synthetic
            or row.frames_used != args.budget
        ):
            raise ValueError(f"unexpected backend or frame count for {row.window_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--videos", type=Path, default=Path("data/videos"))
    parser.add_argument("--url", default="http://127.0.0.1:8200")
    parser.add_argument("--model", default="qwen3-vl-30b-a3b")
    parser.add_argument("--selector", choices=sorted(SELECTORS), default="uniform")
    parser.add_argument("--budget", choices=(4, 8, 16), type=int, default=8)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument(
        "--limit", type=int, default=32, help="Measured videos per level"
    )
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-config", type=Path)
    parser.add_argument(
        "--synthetic", action="store_true", help="Expect a mock backend"
    )
    args = parser.parse_args()
    if min(args.levels) < 1 or len(set(args.levels)) != len(args.levels):
        parser.error("levels must be distinct positive integers")
    if args.limit < max(args.levels) or args.timeout <= 0:
        parser.error(
            "limit must cover the highest concurrency and timeout must be positive"
        )

    unique = {}
    for row in read_jsonl(args.pilot):
        if row.get("split") == args.split:
            unique.setdefault(row["video_id"], row)
    cohort = list(unique.values())[: args.limit + 1]
    if len(cohort) != args.limit + 1:
        parser.error(
            f"need {args.limit + 1} distinct {args.split} clips including warm-up"
        )
    if missing := [
        row["video_id"]
        for row in cohort
        if not (args.videos / row["video_id"]).is_file()
    ]:
        parser.error(f"missing videos: {', '.join(missing)}")

    settings = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    server_config = load_server_config(args.server_config)
    args.out.mkdir(parents=True)
    (args.out / "run.json").write_text(
        json.dumps(
            {"settings": settings, "server": server_config, "git_commit": git_commit()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Keep only inference inputs in the saved workload. Labels are never loaded.
    workload = [
        {name: row[name] for name in ("video_id", "question_id", "question", "split")}
        for row in cohort
    ]
    (args.out / "questions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in workload), encoding="utf-8"
    )
    requests = []
    selection_ms = []
    for row in workload:
        prepared = prepare(
            args.videos / row["video_id"], args.selector, args.budget, row["question"]
        )
        if len(prepared.frames) != args.budget:
            raise ValueError(f"{row['video_id']} did not supply {args.budget} frames")
        requests.append(
            build_request(
                args.out.name,
                row["question_id"],
                row["video_id"],
                row["question"],
                args.budget,
                args.selector,
                prepared,
            )
        )
        selection_ms.append(prepared.select_ms)

    warmup, *measured = requests
    result = measure(args.url, warmup, args.timeout)
    check_backend([result], args)
    if result.status != "completed" or result.answer not in ("yes", "no"):
        raise SystemExit(
            "C warm-up did not return a parsed answer. Check the endpoint before sweeping."
        )

    lines = [
        "# Synthetic load sweep" if args.synthetic else "# C load sweep",
        "",
        f"{args.selector}, {args.budget} frames, {len(measured)} distinct {args.split} clips per level.",
        "RTT and throughput are measured at the client after frame preparation.",
        f"Median edge selection time before dispatch: {statistics.median(selection_ms[1:]):g} ms.",
        "",
        "| Concurrency | Parsed | Unparsed | Failures | Completed/s | p50 RTT ms | p95 RTT ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for level in args.levels:
        run_id = f"{args.out.name}-c{level}"
        batch = [
            request.model_copy(
                update={
                    "run_id": run_id,
                    "window": request.window.model_copy(
                        update={
                            "window_id": f"{run_id}:{request.window.window_id.removeprefix(f'{request.run_id}:')}"
                        }
                    ),
                }
            )
            for request in measured
        ]
        with ThreadPoolExecutor(max_workers=level) as pool:
            started = perf_counter()
            rows = list(
                pool.map(
                    lambda request: measure(args.url, request, args.timeout), batch
                )
            )
            elapsed = perf_counter() - started
        (args.out / f"c{level}.jsonl").write_text(
            "".join(json.dumps(asdict(row)) + "\n" for row in rows), encoding="utf-8"
        )
        check_backend(rows, args)
        completed = [row for row in rows if row.status == "completed"]
        parsed = sum(row.answer in ("yes", "no") for row in completed)
        rtts = [row.round_trip_ms for row in completed]
        p50 = percentile(rtts, 50)
        p95 = percentile(rtts, 95)
        lines.append(
            f"| {level} | {parsed} | {len(completed) - parsed} | {len(rows) - len(completed)} "
            f"| {len(completed) / elapsed:.2f} | {p50 if p50 is not None else '-'} "
            f"| {p95 if p95 is not None else '-'} |"
        )
        (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
