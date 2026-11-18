"""Bandwidth benchmark: selected JPEG requests versus compressed full video.

All numbers are measured file sizes from local encoding; network transfer is not
simulated here.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from statistics import mean

from gigacity_edge.frames import encode_jpeg, probe, read_frames_at, resize_for_model
from gigacity_edge.selectors import uniform

WINDOW_S = 10
LINK_MBPS = 5.0


def jpeg_request_bytes(video: Path, budget: int, duration_ms: int) -> int:
    images = read_frames_at(video, uniform(duration_ms, budget))
    return sum(len(encode_jpeg(resize_for_model(image))) for image in images)


def reencode_bytes(video: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "reencoded.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-an",
                "-vf",
                "scale=448:252",
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-preset",
                "veryfast",
                str(out),
            ],
            check=True,
        )
        return out.stat().st_size


def mbps(byte_count: float, seconds: float) -> float:
    return byte_count * 8 / seconds / 1e6


def measure(videos: list[Path], budgets: list[int]) -> dict:
    per_video = []
    for video in videos:
        duration_ms, _, _ = probe(video)
        seconds = duration_ms / 1000
        original = video.stat().st_size
        reencoded = reencode_bytes(video)
        per_video.append(
            {
                "video": str(video),
                "duration_s": seconds,
                "original_bytes": original,
                "original_mbps": mbps(original, seconds),
                "reencoded_448x252_bytes": reencoded,
                "reencoded_448x252_mbps": mbps(reencoded, seconds),
                "request_bytes": {
                    str(b): jpeg_request_bytes(video, b, duration_ms) for b in budgets
                },
            }
        )
    summary = {}
    for budget in budgets:
        request_kb = mean(v["request_bytes"][str(budget)] for v in per_video) / 1000
        per_camera = request_kb * 1000 * 8 / WINDOW_S / 1e6
        summary[str(budget)] = {
            "mean_request_kb": request_kb,
            "per_camera_mbps_one_request_per_10s": per_camera,
            "cameras_in_5_mbps": int(LINK_MBPS // per_camera) if per_camera else None,
        }
    full_448 = mean(v["reencoded_448x252_mbps"] for v in per_video)
    full_orig = mean(v["original_mbps"] for v in per_video)
    return {
        "note": "measured local file sizes; uniform selection; no transport overhead",
        "videos": len(per_video),
        "budgets": summary,
        "full_video_mbps": {"reencoded_448x252": full_448, "original": full_orig},
        "cameras_in_5_mbps_full_video": {
            "reencoded_448x252": int(LINK_MBPS // full_448) if full_448 else None,
            "original": int(LINK_MBPS // full_orig) if full_orig else None,
        },
        "per_video": per_video,
    }


def markdown(report: dict) -> str:
    lines = [
        "| Evidence | Mean request KB | Mbit/s per camera | Cameras in 5 Mbit/s |",
        "|---|---|---|---|",
    ]
    for budget, row in report["budgets"].items():
        lines.append(
            f"| {budget} JPEG frames / 10 s | {row['mean_request_kb']:.1f} | "
            f"{row['per_camera_mbps_one_request_per_10s']:.3f} | "
            f"{row['cameras_in_5_mbps']} |"
        )
    full = report["full_video_mbps"]
    fit = report["cameras_in_5_mbps_full_video"]
    lines.append(
        f"| Full video, H.264 448x252 crf 28 | - | "
        f"{full['reencoded_448x252']:.3f} | {fit['reencoded_448x252']} |"
    )
    lines.append(
        f"| Full video, original file | - | {full['original']:.3f} | {fit['original']} |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    seen, videos = set(), []
    with args.pilot.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            video_id = json.loads(line)["video_id"]
            path = args.videos / video_id
            if video_id not in seen and path.exists():
                seen.add(video_id)
                videos.append(path)
    if args.limit:
        videos = videos[: args.limit]
    report = measure(videos, args.budgets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
