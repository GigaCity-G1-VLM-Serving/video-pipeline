"""Stand-in client for C: budget sweep against a vLLM Qwen2.5-VL endpoint.

Selects frames uniformly from one source-video window, posts them to a
vLLM OpenAI-compatible server, and records request timing, token counts,
and the Yes/No answer as JSONL.

This is a measurement harness, not the serving path. It deliberately mirrors
the window identity fields of ``gigacity_contracts.WindowMetadata`` so the
payload can move onto the real B -> C contract without changing the
measurement code. Selection here is uniform only: it is a baseline stand-in
for A's selector, never a proposed selection policy.

Results are measured server timings from this client's point of view. There is
no network emulation here, so these are not end-to-end edge numbers.

Usage:
    uv run python -m gigacity_central.smoke \\
        --video /data/tar/HTV/cctv052x2004080519x01683.mp4 \\
        --budgets 4,8,16 --repeats 3 --out results/smoke.jsonl

Set VLLM_API_KEY when the server was started with --api-key.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

COLLISION_PROMPT = (
    "Did any traffic collision occur during the shown observation window? "
    "Answer only Yes or No."
)

YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Window:
    """Source-video window identity; mirrors the draft shared contract."""

    window_id: str
    camera_id: str
    source_video_id: str
    start_ms: int
    end_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "camera_id": self.camera_id,
            "source_video_id": self.source_video_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


@dataclass
class Frame:
    """One encoded frame plus the source-video time it was taken from."""

    capture_ms: int
    jpeg: bytes = field(repr=False)

    @property
    def byte_count(self) -> int:
        return len(self.jpeg)

    def data_url(self) -> str:
        encoded = base64.b64encode(self.jpeg).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def probe_duration_ms(video: Path) -> int:
    """Return the container duration in milliseconds via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return int(float(out.stdout.strip()) * 1000)


def uniform_capture_times(window: Window, budget: int) -> list[int]:
    """Midpoint-uniform capture times inside the half-open window."""
    if budget < 1:
        raise ValueError("budget must be at least 1")
    span = window.end_ms - window.start_ms
    step = span / budget
    times = [int(window.start_ms + (i + 0.5) * step) for i in range(budget)]
    # Guard against rounding landing on or past end_ms.
    return [min(t, window.end_ms - 1) for t in times]


def extract_frame(video: Path, capture_ms: int, quality: int) -> bytes:
    """Decode a single JPEG at capture_ms using a fast ffmpeg seek."""
    out = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            f"{capture_ms / 1000:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            str(quality),
            "-f",
            "image2",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    if not out.stdout:
        raise RuntimeError(f"ffmpeg returned no frame at {capture_ms} ms")
    return out.stdout


def select_frames(
    video: Path, window: Window, budget: int, quality: int
) -> tuple[list[Frame], float]:
    """Extract the budget's frames; also return edge-side decode seconds."""
    started = time.perf_counter()
    frames = [
        Frame(capture_ms=t, jpeg=extract_frame(video, t, quality))
        for t in uniform_capture_times(window, budget)
    ]
    return frames, time.perf_counter() - started


def build_request(model: str, frames: list[Frame], prompt: str) -> dict[str, Any]:
    """Build a pinned, deterministic chat-completions payload."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f.data_url()}} for f in frames
    ]
    content.append({"type": "text", "text": prompt})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
        "temperature": 0.0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def parse_answer(text: str) -> str | None:
    """Return 'yes', 'no', or None when the model answered neither."""
    found = YES_NO.search(text)
    return found.group(1).lower() if found else None


async def send_window(
    client: httpx.AsyncClient,
    *,
    model: str,
    window: Window,
    frames: list[Frame],
    budget: int,
    prompt: str,
    decode_s: float,
    deadline_ms: int | None,
) -> dict[str, Any]:
    """Post one window and record client-observed timing and token counts."""
    payload = build_request(model, frames, prompt)
    total_bytes = sum(f.byte_count for f in frames)
    record: dict[str, Any] = {
        "run_id": RUN_ID,
        "window": window.as_dict(),
        "selector": "uniform",
        "budget_frames": budget,
        "frame_count": len(frames),
        "capture_ms": [f.capture_ms for f in frames],
        "frame_bytes": [f.byte_count for f in frames],
        "total_bytes": total_bytes,
        "edge_decode_s": round(decode_s, 4),
        "deadline_unix_ms": deadline_ms,
        "sent_at_unix_ms": int(time.time() * 1000),
    }

    sent = time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []
    usage: dict[str, Any] | None = None

    try:
        async with client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[len("data: ") :].strip()
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", ()):
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - sent
                        chunks.append(piece)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["total_latency_s"] = round(time.perf_counter() - sent, 4)
        return record

    answer_text = "".join(chunks).strip()
    total_s = time.perf_counter() - sent
    record.update(
        {
            "status": "ok",
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "total_latency_s": round(total_s, 4),
            "answer_raw": answer_text,
            "answer": parse_answer(answer_text),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
        }
    )
    if deadline_ms is not None:
        record["on_time"] = record["sent_at_unix_ms"] + total_s * 1000 <= deadline_ms
    return record


async def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run every budget at the requested concurrency and collect records."""
    video = Path(args.video)
    duration_ms = probe_duration_ms(video)
    window_ms = args.window_s * 1000
    if duration_ms < window_ms:
        raise SystemExit(
            f"video is {duration_ms} ms, shorter than the "
            f"{window_ms} ms observation window"
        )

    window = Window(
        window_id=f"{RUN_ID}:{args.camera_id}:{args.window_start_s}",
        camera_id=args.camera_id,
        source_video_id=video.name,
        start_ms=args.window_start_s * 1000,
        end_ms=args.window_start_s * 1000 + window_ms,
    )

    budgets = [int(b) for b in args.budgets.split(",")]
    records: list[dict[str, Any]] = []

    limits = httpx.Limits(max_connections=max(args.concurrency, 1))
    api_key = os.environ.get("VLLM_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=args.timeout_s,
        limits=limits,
        headers=headers,
    ) as client:
        models = await client.get("/v1/models")
        models.raise_for_status()
        served = [m["id"] for m in models.json().get("data", ())]
        print(f"server reports models: {served}", file=sys.stderr)

        for budget in budgets:
            frames, decode_s = select_frames(video, window, budget, args.jpeg_quality)
            kb = sum(f.byte_count for f in frames) / 1024
            print(
                f"budget={budget:>2} frames  {kb:7.1f} KiB  "
                f"decode {decode_s:5.2f}s  concurrency={args.concurrency}",
                file=sys.stderr,
            )
            for repeat in range(args.repeats):
                deadline = (
                    int(time.time() * 1000) + args.deadline_ms
                    if args.deadline_ms
                    else None
                )
                batch = await asyncio.gather(
                    *(
                        send_window(
                            client,
                            model=args.model,
                            window=window,
                            frames=frames,
                            budget=budget,
                            prompt=args.prompt,
                            decode_s=decode_s,
                            deadline_ms=deadline,
                        )
                        for _ in range(args.concurrency)
                    )
                )
                for item in batch:
                    item["repeat"] = repeat
                    item["concurrency"] = args.concurrency
                records.extend(batch)
                ok = [r for r in batch if r["status"] == "ok"]
                if ok:
                    mean_total = sum(r["total_latency_s"] for r in ok) / len(ok)
                    answers = {r["answer"] for r in ok}
                    tokens = ok[0]["prompt_tokens"]
                    print(
                        f"  repeat {repeat}: {len(ok)}/{len(batch)} ok  "
                        f"mean {mean_total:5.2f}s  prompt_tokens {tokens}  "
                        f"answers {answers}",
                        file=sys.stderr,
                    )
                else:
                    print(f"  repeat {repeat}: all failed", file=sys.stderr)

    return records


def summarize(records: list[dict[str, Any]]) -> None:
    """Print a per-budget summary to stdout."""
    print()
    header = (
        f"{'budget':>6} {'n':>3} {'ok':>3} {'KiB':>8} {'p_tok':>6} "
        f"{'ttft_s':>7} {'total_s':>8} {'answers':>16}"
    )
    print(header)
    print("-" * len(header))
    budgets = sorted({r["budget_frames"] for r in records})
    for budget in budgets:
        rows = [r for r in records if r["budget_frames"] == budget]
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            print(f"{budget:>6} {len(rows):>3} {0:>3}  all requests failed")
            continue
        kib = ok[0]["total_bytes"] / 1024
        tokens = ok[0]["prompt_tokens"] or 0
        ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
        mean_ttft = sum(ttft) / len(ttft) if ttft else float("nan")
        mean_total = sum(r["total_latency_s"] for r in ok) / len(ok)
        counts: dict[str, int] = {}
        for r in ok:
            key = str(r["answer"])
            counts[key] = counts.get(key, 0) + 1
        spread = ",".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(
            f"{budget:>6} {len(rows):>3} {len(ok):>3} {kib:>8.1f} {tokens:>6} "
            f"{mean_ttft:>7.2f} {mean_total:>8.2f} {spread:>16}"
        )
    print(
        "\nThese are measured server timings without network emulation. "
        "Answers come from a pinned uniform selection, not a proposed policy."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="source video path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--budgets", default="4,8,16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--window-start-s", type=int, default=0)
    parser.add_argument("--window-s", type=int, default=10)
    parser.add_argument("--camera-id", default="cam01")
    parser.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg -q:v")
    parser.add_argument("--prompt", default=COLLISION_PROMPT)
    parser.add_argument("--deadline-ms", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--out", default="", help="JSONL output path")
    return parser.parse_args(argv)


RUN_ID = f"smoke-{uuid.uuid4().hex[:8]}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH; install ffmpeg")

    print(f"run_id {RUN_ID}", file=sys.stderr)
    records = asyncio.run(run_sweep(args))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"wrote {len(records)} records to {out}", file=sys.stderr)

    summarize(records)
    failed = [r for r in records if r["status"] != "ok"]
    return 1 if failed and len(failed) == len(records) else 0


if __name__ == "__main__":
    raise SystemExit(main())