"""Event-capture scoring for selectors: a stand-in for D's evaluation, no VLM needed.

For each pilot question, run each selector at each budget, then (only afterwards)
compare the chosen capture times with TAR's annotated event interval for the video.
Labels never reach the selectors. TAR intervals are AI-generated and may describe a
different event than the question, so treat capture as an approximate signal.

Run: python -m gigacity_edge.analysis.capture --pilot data/pilot80/pilot_questions.jsonl \
         --videos data/videos --tar data/tar/temporal_localization.json \
         --selectors uniform motion clip clip-generic --out results/capture80
"""

import argparse
import functools
import json
import statistics
import time
from pathlib import Path

from gigacity_edge import selectors
from gigacity_edge.frames import probe

TOLERANCE_MS = 1000


def parse_timestamp(stamp: str) -> int:
    """'SS', 'MM:SS(.ff)' or 'HH:MM:SS(.ff)' to integer milliseconds."""
    seconds = 0.0
    for part in stamp.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return round(seconds * 1000)


def load_intervals(path: Path) -> dict[str, tuple[int, int]]:
    """video_id -> (start_ms, end_ms) from TAR temporal_localization.json."""
    intervals = {}
    for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
        span = json.loads(item["answer"])
        start, end = parse_timestamp(span["start"]), parse_timestamp(span["end"])
        if end > start:
            intervals.setdefault(item["video_id"], (start, end))
    return intervals


def capture_stats(times: list[int], interval: tuple[int, int], tolerance: int) -> dict:
    start, end = interval
    inside = [t for t in times if start - tolerance <= t <= end + tolerance]
    return {"hit": bool(inside), "inside": len(inside), "frames": len(times)}


class Scans:
    """Scan each video once per selector, then pick frames for every budget."""

    def __init__(self) -> None:
        self.cache: dict[tuple, tuple[list, float]] = {}

    def scores(self, name: str, path: Path, query: str | None):
        if name == "motion":
            key = (name, path)
            compute = functools.partial(selectors.motion_scores, path)
        else:
            texts = [query] if (name == "clip" and query) else selectors.GENERIC_PROMPTS
            key = ("clip", path, tuple(texts))
            compute = functools.partial(selectors.clip_scores, path, texts)
        if key not in self.cache:
            started = time.perf_counter()
            result = compute()
            self.cache[key] = (result, time.perf_counter() - started)
        return self.cache[key]

    def select(self, name, path, budget, duration_ms, query):
        if name == "uniform":
            return selectors.uniform(duration_ms, budget)
        scores, _ = self.scores(name, path, query)
        return selectors.pick_peaks(scores, budget, duration_ms)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["selector"], row["budget"]), []).append(row)
    table = []
    for (selector, budget), items in sorted(groups.items()):
        table.append(
            {
                "selector": selector,
                "budget": budget,
                "n": len(items),
                "hit_rate": statistics.mean(r["hit"] for r in items),
                "hit_rate_strict": statistics.mean(r["hit_strict"] for r in items),
                "frames_inside": statistics.mean(
                    r["inside"] / r["frames"] for r in items
                ),
            }
        )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--selectors", nargs="+", default=["uniform", "motion"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--split", default=None, help="calibration or evaluation")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    intervals = load_intervals(args.tar)
    questions = [
        json.loads(line)
        for line in args.pilot.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.split:
        questions = [q for q in questions if q.get("split") == args.split]
    scans = Scans()
    rows, skipped = [], []
    for question in questions:
        path = args.videos / question["video_id"]
        interval = intervals.get(question["video_id"])
        if not path.exists() or interval is None:
            skipped.append(question["question_id"])
            continue
        duration_ms, _, _ = probe(path)
        for name in args.selectors:
            for budget in args.budgets:
                times = scans.select(
                    name, path, budget, duration_ms, question["question"]
                )
                loose = capture_stats(times, interval, TOLERANCE_MS)
                strict = capture_stats(times, interval, 0)
                rows.append(
                    {
                        "question_id": question["question_id"],
                        "video_id": question["video_id"],
                        "selector": name,
                        "budget": budget,
                        "times_ms": times,
                        "interval_ms": list(interval),
                        "duration_ms": duration_ms,
                        "hit": loose["hit"],
                        "hit_strict": strict["hit"],
                        "inside": loose["inside"],
                        "frames": loose["frames"],
                    }
                )

    scan_cost = {}
    for key, (_, seconds) in scans.cache.items():
        scan_cost.setdefault(key[0], []).append(seconds)
    table = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "capture.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    lines = [
        (
            "| Selector | Frames | N | Any frame on event (±1 s) | Any frame on event "
            "(strict) | Share of frames on event |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['selector']} | {r['budget']} | {r['n']} | {r['hit_rate']:.0%} | "
            f"{r['hit_rate_strict']:.0%} | {r['frames_inside']:.0%} |"
        )
    lines.append("")
    for name, seconds in sorted(scan_cost.items()):
        lines.append(
            f"- {name} scan: {statistics.mean(seconds):.1f} s per video "
            f"({len(seconds)} scans)"
        )
    if skipped:
        lines.append(f"- skipped {len(skipped)} questions (missing video or interval)")
    summary = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
