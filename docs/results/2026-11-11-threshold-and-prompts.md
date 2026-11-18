# Yes threshold and prompt variants, November 11, 2026

**Status: A/B test-server results, not C's serving measurements.** Same pilot (80
Accident-Bench videos, 160 questions, 64 calibration / 96 evaluation) and pipeline as
the [model and selector screen](2026-11-11-model-and-selector-screen.md), with
`Qwen3-VL-30B-A3B-Instruct-FP8` on vLLM v0.11.0, this time on one L40S 48 GB. Requests
also asked for first-token log probabilities, from which `yes_probability` is computed.
Per-question answers: [CSV](data/2026-11-11-pilot80-q30-l40s-answers.csv).

## 1. Yes threshold fitted on calibration

For each selector and budget, the threshold on `yes_probability` that maximises
balanced accuracy on the calibration split was applied unchanged to the evaluation
split (`gigacity_edge.analysis.threshold`).

| Selector | Frames | Evaluation accuracy: parsed answer → threshold | Yes recall | No accuracy |
|---|---|---|---|---|
| uniform | 8 | 67% → 59% | 42% → 60% | 92% → 58% |
| uniform | 16 | 66% → 62% | 48% → 52% | 83% → 73% |
| motion | 8 | 67% → 64% | 42% → 62% | 92% → 65% |
| motion | 16 | 70% → 64% | 48% → 71% | 92% → 56% |
| clip-generic | 8 | 69% → 69% | 48% → 56% | 90% → 81% |
| clip-generic | 16 | 71% → 70% | 52% → 58% | 90% → 81% |

Calibration accuracy rose 5–14 points by construction, but **held-out accuracy stayed
flat or fell**. The yes probability separates true Yes from true No only moderately
(evaluation AUC 0.63–0.73), so the "No" bias is not a simple offset; a threshold moves
along a weak trade-off. It remains a useful operating-point knob if the team prefers
recall (alerts) over accuracy.

## 2. Prompt variants at 8 frames

`v1` is the default prompt; `v2-context` adds "These are frames from a fixed traffic
CCTV camera, in time order."; `v3-careful` asks the model to look carefully for any
collision, even brief or partly visible.

| Selector | Prompt | Calibration accuracy | Evaluation accuracy | Evaluation Yes recall | Paired vs v1, all 160 (variant only / v1 only, p) |
|---|---|---|---|---|---|
| uniform | v1 | 58% | 67% | 42% | – |
| uniform | v2-context | 58% | 65% | 38% | 3 / 5, p = 0.73 |
| uniform | v3-careful | 53% | 60% | 23% | 6 / 15, p = 0.08 |
| clip-generic | v1 | 66% | 69% | 48% | – |
| clip-generic | v2-context | 67% | 62% | 31% | 5 / 10, p = 0.30 |
| clip-generic | v3-careful | 59% | 57% | 19% | 5 / 20, p < 0.01 |

Neither variant helps. Asking the model to look carefully made it **more** conservative
(fewer "Yes" answers). Keep `v1`.

## 3. Reproducibility

The same configuration as the H100 screen changed 2.1–3.5% of answers on the L40S
(hardware and kernel differences), about ±1–2 accuracy points. The crash-frame
diagnostic gave 37/80 versus 36/80 on the H100. Selector or prompt differences below
roughly 4 points should not be read as real at this sample size.

## Takeaways

- Keep the parsed answer and prompt `v1` as defaults; expose the threshold only as an
  optional recall/false-alarm setting.
- The remaining errors are mostly discrimination, not calibration: the model often
  cannot tell. Larger gains need a stronger model, cleaner questions (see the
  [miss review](2026-11-11-miss-review.md)), or more evidence per crash (resolution).

GPU cost: one L40S for about 52 minutes, about $0.95.
