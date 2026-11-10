"""Temporary A/B pilot list from TAR train yes/no questions (D's manifest replaces it).

Questions and labels are written to separate files so selection and inference never
read answers. Videos come from the Accident-Bench source used by TAR.
"""

import argparse
import json
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

TAR_BASE = (
    "https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning"
    "/resolve/main/"
)
TAR_FILES = {
    "bcq.json": "train/bcq.json",
    "temporal_localization.json": "train/temporal_localization.json",
}
ACCIDENT_BENCH_BASE = (
    "https://huggingface.co/datasets/Open-Space-Reasoning/AccidentBench/resolve/main/"
)
SOURCE_PREFIX = "Accident-Bench/"
SUFFIX = re.compile(r"\s*\n\s*Answer with Yes or No\.?\s*$", re.IGNORECASE)
NOTE = "temporary A/B pilot list; D's manifest replaces it"


def strip_answer_suffix(question: str) -> str:
    """Remove TAR's trailing answer-format instruction."""
    return SUFFIX.sub("", question).strip()


def eligible_videos(items: list[dict], prefix: str = SOURCE_PREFIX) -> list[str]:
    """Sorted video IDs from one source with at least one Yes and one No question."""
    answers: dict[str, set[str]] = {}
    for item in items:
        if item["video_id"].startswith(prefix):
            answers.setdefault(item["video_id"], set()).add(item["answer"])
    return sorted(v for v, seen in answers.items() if {"Yes", "No"} <= seen)


def shuffled(video_ids: list[str], seed: int) -> list[str]:
    order = sorted(video_ids)
    random.Random(seed).shuffle(order)
    return order


def questions_for(items: list[dict], video_id: str) -> list[dict]:
    """First Yes then first No question for a video, in file order."""
    picked = {}
    for item in items:
        if item["video_id"] == video_id and item["answer"] not in picked:
            picked[item["answer"]] = item
    return [picked["Yes"], picked["No"]]


def assign_splits(video_ids: list[str], calibration_fraction: float) -> dict[str, str]:
    """Split by source video so no video appears in both sets."""
    n_calibration = round(len(video_ids) * calibration_fraction)
    return {
        video: "calibration" if index < n_calibration else "evaluation"
        for index, video in enumerate(video_ids)
    }


def build_rows(
    items: list[dict], video_ids: list[str], calibration_fraction: float
) -> tuple[list[dict], list[dict]]:
    """Question rows (no answers) and label rows for the chosen videos."""
    splits = assign_splits(video_ids, calibration_fraction)
    questions, labels = [], []
    for video_id in video_ids:
        for item in questions_for(items, video_id):
            question_id = f"q{len(questions):04d}"
            questions.append(
                {
                    "question_id": question_id,
                    "video_id": video_id,
                    "question": strip_answer_suffix(item["question"]),
                    "question_raw": item["question"],
                    "split": splits[video_id],
                }
            )
            labels.append(
                {
                    "question_id": question_id,
                    "video_id": video_id,
                    "answer": item["answer"],
                }
            )
    return questions, labels


def download(url: str, dest: Path, attempts: int = 3) -> None:
    """Download to dest unless present; raises FileNotFoundError on HTTP 404."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                partial.write_bytes(response.read())
            if partial.stat().st_size == 0:
                raise OSError("empty download")
            partial.replace(dest)
            return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(url) from error
            if attempt == attempts:
                raise
        except OSError:
            if attempt == attempts:
                raise
        time.sleep(2**attempt)


def duration_seconds(path: Path) -> float:
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(output.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/pilot"))
    parser.add_argument("--videos", type=Path, default=Path("data/videos"))
    parser.add_argument("--tar", type=Path, default=Path("data/tar"))
    parser.add_argument("--n-videos", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.4)
    parser.add_argument("--max-duration-s", type=float, default=300.0)
    args = parser.parse_args(argv)

    for name, remote in TAR_FILES.items():
        download(TAR_BASE + remote, args.tar / name)
    items = json.loads((args.tar / "bcq.json").read_text(encoding="utf-8"))["items"]

    chosen, durations, replaced = [], {}, []
    for video_id in shuffled(eligible_videos(items), args.seed):
        if len(chosen) == args.n_videos:
            break
        dest = args.videos / video_id
        try:
            download(ACCIDENT_BENCH_BASE + video_id.removeprefix(SOURCE_PREFIX), dest)
        except FileNotFoundError:
            replaced.append({"video_id": video_id, "reason": "404"})
            continue
        seconds = duration_seconds(dest)
        if seconds > args.max_duration_s:
            replaced.append(
                {"video_id": video_id, "reason": f"duration {seconds:.0f}s"}
            )
            continue
        chosen.append(video_id)
        durations[video_id] = round(seconds, 3)
    if len(chosen) < args.n_videos:
        sys.exit(f"only {len(chosen)} usable videos; wanted {args.n_videos}")

    questions, labels = build_rows(items, chosen, args.calibration_fraction)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pilot_questions", questions), ("pilot_labels", labels)):
        lines = "".join(json.dumps(row) + "\n" for row in rows)
        (args.out / f"{name}.jsonl").write_text(lines, encoding="utf-8")
    meta = {
        "note": NOTE,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "max_duration_s": args.max_duration_s,
        "videos": len(chosen),
        "questions": len(questions),
        "splits": {
            split: sum(1 for q in questions if q["split"] == split)
            for split in ("calibration", "evaluation")
        },
        "video_durations_s": durations,
        "replaced": replaced,
        "sources": {
            "tar": {name: TAR_BASE + remote for name, remote in TAR_FILES.items()},
            "videos": ACCIDENT_BENCH_BASE,
        },
    }
    (args.out / "pilot_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("videos", "questions", "splits")}))


if __name__ == "__main__":
    main()
