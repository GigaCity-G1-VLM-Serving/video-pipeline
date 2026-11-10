# Plan and milestones

The canonical plan. Dates follow the proposal timeline (demo December 11, handoff
December 12). Update status in the same change as the work. Decisions referenced
here are in [decisions](decisions.md).

Ownership: A (selection) and B (edge and transport) are the edge pair; C (serving
and budgets) and D (workload and evaluation) are the central pair.

## Milestone 1: whole-video baseline (week of November 10)

Goal: one video produces a logged, scored yes/no answer; then uniform versus motion
selection at 4, 8, and 16 frames on a pilot set.

| Step | Owner | Status |
|---|---|---|
| Agree on draft contracts and defaults with C/D | A/B, C/D | proposed; see [decisions](decisions.md) |
| Draft inference contracts in `packages/contracts` | A/B | done (draft) |
| Temporary pilot list from TAR train yes/no questions, split by source video | A/B, hand to D | in progress |
| Uniform and motion selectors over the whole video | A | in progress |
| Edge pipeline: decode, resize, JPEG, bytes, request, JSONL records | B | in progress |
| Mock C server returning synthetic results | A/B | in progress |
| A/B test server: vLLM + Qwen2.5-VL-7B on RunPod, labeled as A/B test runs | A/B | in progress |
| C's serving endpoint | C | not started in repo |
| Whole-video comparison scored on accuracy and Yes recall | A/B, then D | in progress |

## Milestone 2: bandwidth and load (November 10–16)

| Step | Owner | Status |
|---|---|---|
| Request size at 4, 8, 16 frames versus compressed full video; what fits under 5 Mbit/s | B | in progress |
| Fixed budgets at 1, 2, 4, 8, 16 concurrent requests: throughput, tail latency | B with C | A/B test server first; C's numbers later |

## Milestone 3: live monitoring (November 17–23)

| Step | Owner | Status |
|---|---|---|
| Continuous replay of 10-second windows, several cameras, uplink rate limiting | B | not started |
| Per-window selection; event capture on D's verified windows | A | not started |
| Verified collision and non-collision windows | D | not started |

## Milestone 4: adaptive budgets (November 17–23)

| Step | Owner | Status |
|---|---|---|
| Budget loop with a simulated schedule, then C's load-aware budgets | A/B with C | not started |
| Load-aware budget controller | C | not started |

## Robustness and final (November 24 – December 12)

- Repeat runs, failure analysis, and D's event-centered diagnostic (missed frames
  versus model errors).
- Four-way comparison (uniform or motion × fixed or adaptive budgets) on the held-out
  set; report, demo, and reproduction check.

## Blocked or waiting

- C's serving endpoint and mentor confirmation of vLLM.
- HTV videos need a Kaggle API key; the pilot starts with Accident-Bench only.
- D's official manifest and verified windows.
