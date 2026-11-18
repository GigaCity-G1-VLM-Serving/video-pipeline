"""Stand-in for D's scorer: join edge records, results, and labels by window_id.

A window with an edge record but no completed result counts as incorrect (a miss).

Run: python -m gigacity_edge.analysis.score --labels data/pilot/pilot_labels.jsonl \
         --runs results/RUN1 results/RUN2 --out results/summary.md
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from gigacity_contracts import EdgeRecord, InferenceResult


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question_id_from(window_id: str, run_id: str) -> str:
    """Window IDs look like f"{run_id}:{question_id}:b{budget}"."""
    prefix = run_id + ":"
    if not window_id.startswith(prefix):
        raise ValueError(f"window_id {window_id!r} does not start with run_id")
    question_id, _, budget = window_id[len(prefix) :].rpartition(":")
    if not question_id or not budget.startswith("b"):
        raise ValueError(f"unexpected window_id format: {window_id!r}")
    return question_id


def percentile(values: list[int], q: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


@dataclass
class Row:
    selector: str
    budget: int
    split: str
    label: str
    answer: str | None
    completed: bool
    round_trip_ms: int | None
    total_bytes: int
    prompt_tokens: int | None
    synthetic: bool


def load_rows(
    run_dirs: list[Path], labels: dict[str, str], splits: dict[str, str]
) -> list[Row]:
    rows: list[Row] = []
    for run_dir in run_dirs:
        results = {
            r.window_id: r
            for r in map(
                InferenceResult.model_validate, read_jsonl(run_dir / "results.jsonl")
            )
        }
        for edge in map(EdgeRecord.model_validate, read_jsonl(run_dir / "edge.jsonl")):
            question_id = question_id_from(edge.window.window_id, edge.run_id)
            if question_id not in labels:
                raise KeyError(f"no label for question_id {question_id!r}")
            result = results.get(edge.window.window_id)
            completed = result is not None and result.status == "completed"
            rows.append(
                Row(
                    selector=edge.selector,
                    budget=edge.frame_budget,
                    split=splits.get(question_id, "all"),
                    label=labels[question_id].strip().lower(),
                    answer=result.answer if completed else None,
                    completed=completed,
                    round_trip_ms=edge.round_trip_ms,
                    total_bytes=edge.total_bytes,
                    prompt_tokens=result.prompt_tokens if result else None,
                    synthetic=bool(result and result.is_synthetic),
                )
            )
    return rows


def summarize(rows: list[Row]) -> list[dict]:
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for row in rows:
        groups[(row.selector, row.budget, row.split)].append(row)
    summaries = []
    for (selector, budget, split), group in sorted(groups.items()):
        yes = [r for r in group if r.label == "yes"]
        no = [r for r in group if r.label == "no"]
        trips = [r.round_trip_ms for r in group if r.round_trip_ms is not None]
        tokens = [r.prompt_tokens for r in group if r.prompt_tokens is not None]
        summaries.append(
            {
                "selector": selector,
                "budget": budget,
                "split": split,
                "n": len(group),
                "accuracy": sum(r.answer == r.label for r in group) / len(group),
                "yes_recall": (
                    sum(r.answer == "yes" for r in yes) / len(yes) if yes else None
                ),
                "no_accuracy": (
                    sum(r.answer == "no" for r in no) / len(no) if no else None
                ),
                "unparsed_rate": sum(r.answer == "unparsed" for r in group)
                / len(group),
                "failures": sum(not r.completed for r in group),
                "p50_round_trip_ms": percentile(trips, 50),
                "p95_round_trip_ms": percentile(trips, 95),
                "mean_total_bytes": statistics.fmean(r.total_bytes for r in group),
                "mean_prompt_tokens": statistics.fmean(tokens) if tokens else None,
            }
        )
    return summaries


COLUMNS = [
    ("selector", "Selector"),
    ("budget", "Budget"),
    ("split", "Split"),
    ("n", "N"),
    ("accuracy", "Accuracy"),
    ("yes_recall", "Yes recall"),
    ("no_accuracy", "No accuracy"),
    ("unparsed_rate", "Unparsed"),
    ("failures", "Failures"),
    ("p50_round_trip_ms", "p50 RTT ms"),
    ("p95_round_trip_ms", "p95 RTT ms"),
    ("mean_total_bytes", "Mean bytes"),
    ("mean_prompt_tokens", "Mean prompt tokens"),
]


def fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key in ("accuracy", "yes_recall", "no_accuracy", "unparsed_rate"):
        return f"{value:.0%}"
    if isinstance(value, float):
        return f"{value:,.0f}"
    return str(value)


def to_markdown(summaries: list[dict], synthetic: bool) -> str:
    lines = []
    if synthetic:
        lines.append(
            "**SYNTHETIC (mock server): not model or serving measurements.**\n"
        )
    lines.append("| " + " | ".join(title for _, title in COLUMNS) + " |")
    lines.append("|" + "---|" * len(COLUMNS))
    for summary in summaries:
        cells = (fmt(key, summary[key]) for key, _ in COLUMNS)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--pilot", type=Path, help="pilot JSONL with question_id/split")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    labels = {r["question_id"]: r["answer"] for r in read_jsonl(args.labels)}
    splits = (
        {r["question_id"]: r["split"] for r in read_jsonl(args.pilot) if "split" in r}
        if args.pilot
        else {}
    )
    rows = load_rows(args.runs, labels, splits)
    markdown = to_markdown(summarize(rows), any(r.synthetic for r in rows))
    print(markdown)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
