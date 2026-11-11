"""Per-question export and paired comparisons of runs (stand-in for D's analysis).

Export: one CSV row per request, small enough to commit (no images or payloads).
Compare: on the questions both runs answered at one budget, count where only one was
right and report an exact two-sided sign test. A missing or failed result is wrong.

Run:
  python -m gigacity_edge.analysis.compare export --labels L --pilot P \
      --runs results/A results/B --csv docs/results/data/answers.csv
  python -m gigacity_edge.analysis.compare paired --labels L --budget 8 \
      --a results/A --b results/B
"""

import argparse
import csv
import json
import math
from pathlib import Path

from gigacity_edge.analysis.score import question_id_from, read_jsonl

COLUMNS = [
    "run_id",
    "question_id",
    "video_id",
    "split",
    "selector",
    "budget",
    "status",
    "answer",
    "yes_probability",
    "label",
    "correct",
    "round_trip_ms",
    "total_bytes",
    "prompt_tokens",
    "model_id",
]


def load_labels(path: Path) -> dict[str, str]:
    """question_id -> 'yes' or 'no'."""
    return {row["question_id"]: row["answer"].lower() for row in read_jsonl(path)}


def load_pilot(path: Path | None) -> dict[str, dict]:
    return {row["question_id"]: row for row in read_jsonl(path)} if path else {}


def run_rows(
    run_dir: Path, labels: dict[str, str], pilot: dict[str, dict]
) -> list[dict]:
    """Join edge records with results by window_id; a missing result is a miss."""
    results = {r["window_id"]: r for r in read_jsonl(run_dir / "results.jsonl")}
    rows = []
    for edge in read_jsonl(run_dir / "edge.jsonl"):
        window = edge["window"]
        qid = question_id_from(window["window_id"], edge["run_id"])
        result = results.get(window["window_id"], {})
        status = result.get("status", "missing")
        answer = result.get("answer") if status == "completed" else None
        label = labels.get(qid)
        rows.append(
            {
                "run_id": edge["run_id"],
                "question_id": qid,
                "video_id": window["source_video_id"],
                "split": pilot.get(qid, {}).get("split", ""),
                "selector": edge["selector"],
                "budget": edge["frame_budget"],
                "status": status,
                "answer": answer or "",
                "yes_probability": result.get("yes_probability"),
                "label": label or "",
                "correct": bool(label) and answer == label,
                "round_trip_ms": edge.get("round_trip_ms"),
                "total_bytes": edge["total_bytes"],
                "prompt_tokens": result.get("prompt_tokens"),
                "model_id": result.get("model_id", ""),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in COLUMNS})


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1))
    return min(1.0, 2 * tail / 2**n)


def paired(rows_a: list[dict], rows_b: list[dict], budget: int) -> dict:
    a = {r["question_id"]: r["correct"] for r in rows_a if r["budget"] == budget}
    b = {r["question_id"]: r["correct"] for r in rows_b if r["budget"] == budget}
    shared = sorted(set(a) & set(b))
    counts = {
        "n": len(shared),
        "both_right": sum(a[q] and b[q] for q in shared),
        "only_a": sum(a[q] and not b[q] for q in shared),
        "only_b": sum(b[q] and not a[q] for q in shared),
        "both_wrong": sum(not a[q] and not b[q] for q in shared),
    }
    counts["p_value"] = sign_test(counts["only_a"], counts["only_b"])
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--labels", type=Path, required=True)
    export.add_argument("--pilot", type=Path, default=None)
    export.add_argument("--runs", type=Path, nargs="+", required=True)
    export.add_argument("--csv", type=Path, required=True)
    pair = sub.add_parser("paired")
    pair.add_argument("--labels", type=Path, required=True)
    pair.add_argument("--budget", type=int, required=True)
    pair.add_argument("--a", type=Path, required=True)
    pair.add_argument("--b", type=Path, required=True)
    args = parser.parse_args(argv)

    labels = load_labels(args.labels)
    if args.command == "export":
        pilot = load_pilot(args.pilot)
        rows = [row for run in args.runs for row in run_rows(run, labels, pilot)]
        write_csv(rows, args.csv)
        print(f"wrote {len(rows)} rows to {args.csv}")
    else:
        result = paired(
            run_rows(args.a, labels, {}), run_rows(args.b, labels, {}), args.budget
        )
        print(json.dumps({"a": str(args.a), "b": str(args.b), **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
