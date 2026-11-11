# Model and selector screen, November 11, 2026

**Status: A/B test-server results, not C's serving measurements.** Same pipeline and
caveats as the [November 10 pilot](2026-11-10-whole-video-pilot.md), on a larger
temporary pilot. D should rescore once the official manifest exists.

## Setup

- **Data:** temporary A/B pilot, 80 Accident-Bench videos from TAR train (seed 0;
  contains the 25 November 10 videos), 160 yes/no questions (80 Yes, 80 No), split by
  video into calibration (64 questions) and evaluation (96). Labels are AI-generated.
- **Edge:** whole-video window; 448×252 JPEG frames at quality 85, sent as separate
  images with capture times (decision 9), using the shared `InferenceRequest`.
- **Selectors:** `uniform`, `motion`, and `clip-generic` (CLIP ViT-B/32 on ONNX
  Runtime, scored against generic collision captions, never labels or the question).
- **Model:** `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` (revision `d9748a51`) on vLLM v0.11.0,
  one H100 80 GB, prefix caching off, temperature 0. Requests ran four at a time per
  selector with three selectors in parallel, so latencies are not load measurements.
  No Ada GPUs were available for FP8.

## 1. Selector event capture (no VLM)

Share of questions with at least one selected frame inside TAR's annotated event
interval (strict; ±1 s in brackets). The event is a median 4% of each video.

| Frames | Uniform | Motion | CLIP (question) | CLIP (generic) |
|---|---|---|---|---|
| 4 | 38% (50%) | 34% (52%) | 38% (54%) | 40% (50%) |
| 8 | 45% (68%) | 52% (75%) | 51% (72%) | **59% (76%)** |
| 16 | 68% (90%) | 71% (89%) | 71% (91%) | **81% (91%)** |

CPU scan cost on this 4-core machine: motion 0.7 s per video, CLIP 3.8 s per video
(about 22 ms per frame at 2 frames/s). Question-text CLIP is weaker than generic
captions; TAR questions are about details, and half are distractor "No" questions.

## 2. Qwen3-VL-30B-A3B end to end (160 questions)

| Selector | Frames | Accuracy | Yes recall | No accuracy | Mean request |
|---|---|---|---|---|---|
| uniform | 4 | 60% | 28% | 92% | 87 KB |
| uniform | 8 | 63% | 34% | 92% | 170 KB |
| uniform | 16 | 68% | 46% | 90% | 343 KB |
| motion | 4 | 62% | 34% | 91% | 80 KB |
| motion | 8 | 62% | 35% | 90% | 165 KB |
| motion | 16 | 68% | 45% | 91% | 338 KB |
| clip-generic | 4 | 61% | 30% | 91% | 87 KB |
| clip-generic | 8 | 67% | 41% | 92% | 174 KB |
| clip-generic | 16 | 69% | 48% | 90% | 347 KB |

Two requests failed (one each for uniform and motion at 4 frames) and count as misses.
CLIP-generic is ahead at 8 frames, but on the same questions it was right where uniform
was wrong 15 times and the reverse 9 times (sign test p = 0.31; versus motion 14 to 7,
p = 0.19). **No selector difference is statistically reliable at this sample size.**

Crash-frame diagnostic (8 frames inside the annotated interval ±1 s, labels used):
Yes recall 36/80 (45%) versus 27/80 (34%) for uniform 8; No questions correct 73/80.

## 3. Model comparison on the same 50 questions

The 50 November 10 questions are a subset of this pilot. Same frames, prompt, and
resolution; only the model differs.

| Model | Best accuracy (setting) | Yes recall at 16 uniform | No accuracy at 16 uniform |
|---|---|---|---|
| Qwen2.5-VL-7B | 60% (uniform 8, motion 4) | 20% | 96% |
| Qwen3-VL-8B | 58% (uniform 8, motion 4) | 36% | 76% |
| Qwen3-VL-30B-A3B | 66% (uniform 16, motion 16) | 44% | 88% |

Qwen3-VL-30B-A3B is the best tested: it finds more crashes than Qwen2.5-VL-7B without
Qwen3-VL-8B's loss on "No" questions, and it improves with more frames. Fifty questions
are still few (one question is 2 points); the 160-question numbers above agree.

## Takeaways

- **Model:** propose Qwen3-VL-30B-A3B as the primary model for team review (decision 3
  is Qwen2.5-VL-7B). It needs about 30 GB at FP8 and an Ada or Hopper GPU for FP8.
- **Selection:** CLIP-generic puts frames on the crash most often and gives the best
  8-frame result, but answer-level gains are within noise on 160 questions. Even with
  frames on the crash the model finds fewer than half of crashes, so model and prompt
  improvements still matter more than selection.
- **Next:** a Yes-threshold from token probabilities (fit on calibration only), prompt
  variants, and more questions before claiming selector gains.

GPU cost: one H100 for about 45 minutes, about $3.00.
