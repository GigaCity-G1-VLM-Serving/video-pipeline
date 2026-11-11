# Committed result summaries

Small CSV summaries of A/B test-server runs, so results survive without the raw
outputs. Raw per-request JSONL (`results/<run_id>/edge.jsonl`, `results.jsonl`,
`run.json`) stays local and out of Git, per project rules; it is not backed up
elsewhere. None of these files contain images, payloads, or credentials.

All rows come from the temporary A/B pilot lists (TAR train, AI-generated labels) and
A/B-operated vLLM servers, not C's service. See the matching write-ups in
[`docs/results/`](..) for setup and caveats.

| File | Source runs |
|---|---|
| `2026-11-10-pilot25-answers.csv` | 25-video pilot: Qwen2.5-VL-7B (`real-*`) and Qwen3-VL-8B (`q3-*`), uniform and motion, 4/8/16 frames |
| `2026-11-10-load-sweep.csv` | Qwen2.5-VL-7B, uniform, 8 frames, 1–16 concurrent requests (A40) |
| `2026-11-10-pilot25-bandwidth.csv` | Request bytes per video and budget versus the original and a 448×252 H.264 re-encode |
| `2026-11-11-pilot80-q30-answers.csv` | 80-video pilot: Qwen3-VL-30B-A3B FP8 (H100), uniform, motion, clip-generic, 4/8/16 frames |
| `2026-11-11-pilot80-capture.csv` | Selector event capture, no VLM: uniform, motion, clip, clip-generic |
| `event-oracle-diagnostics.csv` | Eight frames placed on the annotated event (uses labels; diagnostic only) |

## Columns

**`*-answers.csv`** (from `python -m gigacity_edge.analysis.compare export`): one row per
request. `status` is the server status (`missing` if no result arrived); `answer` is the
parsed `yes`/`no`/`unparsed`, empty when not completed; `correct` is false for any
failure, so failures count as misses. `yes_probability` is empty for runs before
November 11, which did not request token logprobs. `round_trip_ms` is measured on the
edge clock and includes the RunPod proxy. `split` is `calibration` or `evaluation`.

**`*-capture.csv`**: `hit_1s` / `hit_strict` mean at least one selected frame fell
inside the annotated event interval with or without 1 s tolerance;
`frames_inside_1s` counts such frames. Frame times are omitted.

**`event-oracle-diagnostics.csv`**: one row per question; `correct` compares the parsed
answer with the label.

Reproduce a paired comparison from the raw outputs with
`python -m gigacity_edge.analysis.compare paired --labels L --budget 8 --a RUN_A --b RUN_B`,
and Yes-threshold fitting with `python -m gigacity_edge.analysis.threshold`.
