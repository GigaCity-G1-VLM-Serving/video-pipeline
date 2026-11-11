"""Yes-threshold calibration on yes_probability (stand-in for D's analysis).

For each (selector, budget), choose the threshold that maximises balanced accuracy on
the CALIBRATION split only (ties go to the threshold closest to 0.5), then report
calibration and evaluation metrics for the parsed answer versus the fitted threshold.
The evaluation split is never used to fit. Questions without yes_probability (failed,
or a server without logprobs) count as wrong under the threshold.

Run: python -m gigacity_edge.analysis.threshold --labels L --pilot P \
         --runs results/A results/B --out results/threshold.md
"""

import argparse
from collections import defaultdict
from pathlib import Path

from gigacity_edge.analysis.compare import load_labels, load_pilot, run_rows


def predict(row: dict, threshold: float | None) -> str | None:
    if threshold is None:
        return row["answer"] or None
    p = row["yes_probability"]
    if p is None:
        return None
    return "yes" if p >= threshold else "no"


def metrics(rows: list[dict], threshold: float | None) -> dict:
    yes = [r for r in rows if r["label"] == "yes"]
    no = [r for r in rows if r["label"] == "no"]
    yes_ok = sum(predict(r, threshold) == "yes" for r in yes)
    no_ok = sum(predict(r, threshold) == "no" for r in no)
    yes_recall = yes_ok / len(yes) if yes else 0.0
    no_accuracy = no_ok / len(no) if no else 0.0
    return {
        "n": len(rows),
        "accuracy": (yes_ok + no_ok) / len(rows) if rows else 0.0,
        "yes_recall": yes_recall,
        "no_accuracy": no_accuracy,
        "balanced": (yes_recall + no_accuracy) / 2,
    }


def fit_threshold(rows: list[dict]) -> float | None:
    """Best balanced-accuracy cut among observed probabilities, on calibration rows."""
    probabilities = sorted(
        {r["yes_probability"] for r in rows if r["yes_probability"] is not None}
    )
    if not probabilities:
        return None
    candidates = [0.5] + probabilities
    return max(candidates, key=lambda t: (metrics(rows, t)["balanced"], -abs(t - 0.5)))


def table(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["selector"], row["budget"])].append(row)
    out = []
    for (selector, budget), items in sorted(groups.items()):
        calibration = [r for r in items if r["split"] == "calibration"]
        evaluation = [r for r in items if r["split"] == "evaluation"]
        threshold = fit_threshold(calibration)
        for split, subset in (("calibration", calibration), ("evaluation", evaluation)):
            for method, t in (("parsed answer", None), ("fitted threshold", threshold)):
                if method == "fitted threshold" and threshold is None:
                    continue
                out.append(
                    {
                        "selector": selector,
                        "budget": budget,
                        "split": split,
                        "method": method,
                        "threshold": t,
                        **metrics(subset, t),
                    }
                )
    return out


def to_markdown(rows: list[dict]) -> str:
    lines = [
        (
            "| Selector | Frames | Split | Method | Threshold | N | Accuracy "
            "| Yes recall | No accuracy | Balanced |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        threshold = "-" if r["threshold"] is None else f"{r['threshold']:.3f}"
        lines.append(
            f"| {r['selector']} | {r['budget']} | {r['split']} | {r['method']} | "
            f"{threshold} | {r['n']} | {r['accuracy']:.0%} | {r['yes_recall']:.0%} | "
            f"{r['no_accuracy']:.0%} | {r['balanced']:.0%} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    labels, pilot = load_labels(args.labels), load_pilot(args.pilot)
    rows = [row for run in args.runs for row in run_rows(run, labels, pilot)]
    markdown = to_markdown(table(rows))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
