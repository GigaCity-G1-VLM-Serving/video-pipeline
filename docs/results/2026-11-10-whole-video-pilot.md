# Whole-video pilot, November 10, 2026

**Status: A/B test-server results, not C's serving measurements.** These runs used an
A/B-operated vLLM server on RunPod to exercise the milestone 1 pipeline before C's
endpoint exists. D should rescore with the official evaluator once the manifest exists.

## Setup

- **Data:** temporary A/B pilot list. 25 Accident-Bench crash videos from TAR train
  (seed 0), 50 yes/no questions (25 Yes, 25 No), split by video into calibration
  (10 videos) and evaluation (15). Video length 10.5–212.5 s, median 62.9 s. TAR labels
  are AI-generated. No highway (HTV) videos, which need a Kaggle key.
- **Edge:** whole-video window; uniform or motion/scene-change selection at 4, 8, or 16
  frames; frames resized to 448×252 and JPEG-encoded at quality 85.
- **Serving:** one A40 48 GB (RunPod Secure Cloud), `vllm/vllm-openai:v0.11.0`, prefix
  caching off, temperature 0, at most 16 output tokens. Frames are sent as separate
  images with capture times in the prompt (decision 9). Model revisions are pinned in
  each result's `model_id`.
- **Code:** `gigacity_edge.run`, `gigacity_edge.standins.vllm_shim`,
  `gigacity_edge.analysis.score`, `gigacity_edge.bandwidth` at commit `2f36d39` plus
  the shim User-Agent fix `69e938d`, before the switch to the shared C/D request
  model. After the switch, the chat payloads sent to vLLM were checked to be
  byte-identical, so these results still apply. Raw outputs are kept out of Git.

## 1. Selection and budgets: Qwen2.5-VL-7B-Instruct

All 300 requests completed (0 failures, 0 unparsed). Accuracy over 50 questions per row:

| Selector | Frames | Accuracy | Yes recall | Says "Yes" | p50 latency | Mean request |
|---|---|---|---|---|---|---|
| uniform | 4 | 56% | 16% | 5/50 | 0.61 s | 87 KB |
| uniform | 8 | 60% | 20% | 5/50 | 0.78 s | 169 KB |
| uniform | 16 | 58% | 20% | 6/50 | 1.24 s | 342 KB |
| motion | 4 | 60% | 20% | 5/50 | 0.55 s | 77 KB |
| motion | 8 | 58% | 16% | 4/50 | 0.74 s | 161 KB |
| motion | 16 | 54% | 8% | 2/50 | 1.13 s | 335 KB |

Latency is the edge round trip at concurrency 1 (proxy and network included). One
question changes accuracy by 2 points; the differences between rows are within noise.

**Finding:** the model answers "No" almost always, as in the earlier laptop pilot.
More frames add latency and bytes without improving answers, and selection makes no
measurable difference while the model rarely says "Yes".

## 2. Can the model see the crash? Event-centred diagnostic

Eight frames placed inside TAR's annotated event interval (±1 s) for each video. This
uses labels and is a diagnostic, not a selector. TAR's interval can describe a
different event than the question, so it is approximate.

| Model | Yes recall, crash frames | Yes recall, uniform 8 | No questions correct |
|---|---|---|---|
| Qwen2.5-VL-7B | 7/25 (28%) | 5/25 (20%) | 25/25 |
| Qwen3-VL-8B | 11/25 (44%) | 8/25 (32%) | 20/25 |

Even with frames on the crash, Qwen2.5-VL-7B mostly answers "No". **The model, not
frame selection, is the current bottleneck.** Qwen3-VL-8B sees more crashes but still
fewer than half; switching models requires team agreement (decision 3).

## 3. Qwen3-VL-8B-Instruct comparison

Same pilot, pipeline, and server image, with `Qwen/Qwen3-VL-8B-Instruct` (revision
`0c351dd0`). Qwen3-VL uses a 32-pixel grid, so 448×252 frames are processed at about
112 visual tokens each instead of 144. All 300 requests completed.

| Selector | Frames | Accuracy | Yes recall | No accuracy | Says "Yes" | p50 latency |
|---|---|---|---|---|---|---|
| uniform | 4 | 54% | 24% | 84% | 10/50 | 0.73 s |
| uniform | 8 | 58% | 32% | 84% | 12/50 | 0.84 s |
| uniform | 16 | 56% | 36% | 76% | 15/50 | 1.20 s |
| motion | 4 | 58% | 28% | 88% | 10/50 | 0.49 s |
| motion | 8 | 56% | 32% | 80% | 13/50 | 0.61 s |
| motion | 16 | 56% | 32% | 80% | 13/50 | 0.85 s |

Compared with Qwen2.5-VL-7B, Qwen3-VL-8B detects more crashes (Yes recall 24–36% versus
8–20%) and, unlike Qwen2.5, improves with more frames, at the cost of more false "Yes"
answers; overall accuracy is unchanged at about 55%. Selection still makes no
measurable difference on 50 questions. Latency differences between the two runs
include proxy variance and are not a model speed comparison.

## 4. Serving load: Qwen2.5-VL-7B, uniform, 8 frames

50 requests per level, all completed.

| Requests at once | Throughput | p50 latency | p95 latency |
|---|---|---|---|
| 1 | 1.54 req/s | 0.64 s | 0.85 s |
| 2 | 2.32 req/s | 0.86 s | 1.25 s |
| 4 | 3.01 req/s | 1.33 s | 1.83 s |
| 8 | 3.57 req/s | 2.05 s | 3.42 s |
| 16 | 3.53 req/s | 3.98 s | 5.85 s |

One A40 saturates around 3.5 requests/s at 8 frames each. At one request per camera
per 10 s that is about 35 cameras at saturation; a 5 s deadline is missed at the p95
by 16 concurrent requests. These include the RunPod proxy round trip.

## 5. Bandwidth: 25 pilot videos

| Evidence per 10 s window | Mean request | Mbit/s per camera | Cameras in 5 Mbit/s |
|---|---|---|---|
| 4 JPEG frames (448×252, q85) | 86.7 KB | 0.069 | 72 |
| 8 JPEG frames | 169.0 KB | 0.135 | 36 |
| 16 JPEG frames | 342.3 KB | 0.274 | 18 |
| Full video, H.264 448×252, CRF 28 | – | 0.089 | 56 |
| Full video, original file | – | 0.593 | 8 |

Compressed full video at the model's resolution costs about as much as 4–5 JPEG frames
per window. From 8 frames up, JPEG frames cost more than the video. Payload bytes only;
transport overhead is excluded.

## Caveats

- Small pilot (50 questions, one dataset source, AI-generated labels); directional only.
- Many Accident-Bench videos are compilations; motion selection favours scene cuts.
- Latency includes the RunPod proxy and is not a C serving measurement.
- The event-centred diagnostic was a one-off script outside the repo; D's planned
  diagnostic should replace it.
- GPU cost for all runs was about $0.45 (one A40 for about 41 minutes, plus a short probe).
