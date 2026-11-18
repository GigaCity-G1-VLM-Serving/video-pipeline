# C serving load: Qwen3-VL-30B-A3B on one L40S, November 17, 2026

**Status: C serving measurements, single source video.** The client ran on the same
pod as the server, so timings contain no proxy or network time. All frames come from
one window of one video, so this characterises serving cost, not answer quality.

## Setup

- **GPU:** one NVIDIA L40S 48 GB, RunPod Secure Cloud (North America), driver
  595.112.07.
- **Serving:** vLLM 0.11.0 (pip install on the RunPod PyTorch 2.8.0 template, not the
  `vllm/vllm-openai:v0.11.0` image of decision 7), transformers 4.57.6, torch
  2.8.0+cu128. Prefix caching off, max model length 16,384, GPU memory utilisation 0.90,
  up to 16 images per request.
- **Model:** `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`, revision `d9748a51ae66`, the same
  revision as A/B's [model and selector screen](2026-11-11-model-and-selector-screen.md).
  Weights take 30.3 GiB, leaving 6.0 GiB of KV cache (66,016 tokens).
- **Requests:** `gigacity_central.smoke` on branch `feat/central-serving-smoke`. Uniform
  frames from `Accident-Bench/land_space/long/videos/000001.mp4`, window 50–60 s,
  448×252 JPEG at quality 85 (decision 10), "Frame i at t s:" prompt (decision 12),
  temperature 0, at most 16 output tokens.
- **Load:** closed-loop bursts. Each repeat sends *k* identical requests at once and waits
  for all of them; five repeats per level. The first repeat of each level is dropped as
  warm-up. Throughput is completed requests divided by the summed burst durations.
- **Deadline:** `on_time` is measured as completion within 5 s of the request being sent.
  Decision 18 measures 5 s from window end, which also includes edge time; see caveats.

## 1. Cost of one request

| Frames | Request size | Prompt tokens | Latency, one at a time |
|---|---|---|---|
| 4 | 72.7 KiB | 525 | 0.12 s |
| 8 | 132.6 KiB | 1,025 | 0.14 s |
| 16 | 271.8 KiB | 2,032 | 0.18 s |

About 125 prompt tokens per frame: Qwen3-VL's ~112 visual tokens for a 448×252 image
plus the timestamp line. Unloaded, more frames cost little time.

## 2. Load sweep

| Frames | Requests at once | p50 latency | p95 latency | Throughput | On time (5 s) |
|---|---|---|---|---|---|
| 4 | 1 | 0.12 s | 0.12 s | 8.3 req/s | 100% |
| 4 | 4 | 0.27 s | 0.29 s | 14.4 req/s | 100% |
| 4 | 16 | 0.71 s | 0.78 s | 20.8 req/s | 100% |
| 4 | 32 | 1.17 s | 1.44 s | 22.3 req/s | 100% |
| 4 | 64 | 2.04 s | 2.74 s | 23.3 req/s | 100% |
| 8 | 1 | 0.14 s | 0.14 s | 7.4 req/s | 100% |
| 8 | 4 | 0.38 s | 0.40 s | 10.2 req/s | 100% |
| 8 | 16 | 0.99 s | 1.19 s | 13.3 req/s | 100% |
| 8 | 32 | 1.63 s | 2.27 s | 13.8 req/s | 100% |
| 8 | 64 | 2.94 s | 4.42 s | 14.4 req/s | 100% |
| 16 | 1 | 0.18 s | 0.18 s | 5.5 req/s | 100% |
| 16 | 4 | 0.56 s | 0.57 s | 7.1 req/s | 100% |
| 16 | 16 | 1.42 s | 2.06 s | 7.6 req/s | 100% |
| 16 | 32 | 2.50 s | 4.06 s | 7.9 req/s | 100% |
| 16 | 64 | 5.02 s | 8.19 s | 7.6 req/s | **46%** |

Concurrency 2 and 8 were also run and fall between their neighbours; all 1,875 requests
completed.

**Finding:** throughput saturates at about 23, 14 and 8 requests/s for 4, 8 and 16
frames. That is roughly 12,000–16,000 prompt tokens/s in every case, so under load the
cost of a request is set by its frame count. Halving the frames roughly doubles the
requests the GPU can serve.

## 3. What this means for load-aware budgets

At one request per camera per 10 s window, saturation corresponds to about **230
cameras at 4 frames, 140 at 8 and 79 at 16** on one L40S. These are upper bounds: they
assume evenly spread arrivals and ignore edge and network time.

The deadline shows the controller's lever directly. With 64 requests in flight, 16-frame
requests reach a p95 of 8.2 s and only 46% finish within 5 s, while 8-frame requests
under the same load reach 4.4 s and all finish. A/B's screen found accuracy of 60%, 63%
and 68% at 4, 8 and 16 uniform frames with this model. So more frames buy accuracy when
the server is idle, and cost deadline misses when it is busy. That is the trade-off a
load-aware budget is meant to manage.

## 4. Comparison with A/B's A40 numbers

A/B measured Qwen2.5-VL-7B on an A40 through the RunPod proxy: about 3.5 requests/s at
8 frames ([November 10 pilot](2026-11-10-whole-video-pilot.md), section 4). This run
reaches 14.4 requests/s at 8 frames. The two differ in model, GPU, visual tokens per
frame (144 versus about 112) and proxy, so the ratio is not a model speed comparison.
Qwen3-VL-30B-A3B is a mixture-of-experts model with about 3B active parameters, which
helps explain why it serves quickly despite its size.

## Caveats

- **One repeated payload.** Every request in a burst carried the same eight images.
  vLLM's multimodal preprocessing cache may have reused processed images across
  requests, which would make these numbers optimistic for real traffic, where every
  window has new frames. A rerun with distinct windows per request, or with that cache
  disabled, is needed before quoting camera counts.
- **Burst load, not live arrivals.** Real cameras send requests spread over each window.
  Bursts are a stress test; a steady-arrival test would give the realistic deadline curve.
- **Deadline definition.** Decision 18 counts from window end. Edge selection, encoding
  and uplink time come before the request is sent, so end-to-end compliance will be
  lower than shown here.
- **Answers changed with load.** At a fixed input and temperature 0, the 8-frame request
  answered "No" when sent alone but mostly "Yes" in large bursts. vLLM's batched
  computation is not bit-identical across batch sizes, and this window sits near the
  model's decision boundary. Answer-quality runs should be done at a fixed, recorded
  concurrency.
- **Image, not container.** vLLM was installed with pip rather than the pinned container
  image. A first attempt failed because pip pulled transformers 5.x; pinning
  `transformers<5` fixed it.

## Reproducing

Follow [the serving runbook](../serving-runbook.md), then run the budget and concurrency
loops in this branch's history. Raw JSONL is kept out of Git.

GPU cost: one L40S for about one hour, about $1.10.
