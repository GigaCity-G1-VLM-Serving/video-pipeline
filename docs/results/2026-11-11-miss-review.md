# Manual review of missed crashes, November 11, 2026

**Scope:** calibration split only, so the evaluation split stays unexamined. Questions
are the 32 calibration "Yes" questions of the 80-video pilot; the 17 reviewed are those
Qwen3-VL-30B-A3B answered "No" even with 8 frames placed inside TAR's annotated event
interval (±1 s). See the [model and selector screen](2026-11-11-model-and-selector-screen.md).
Each case was judged from the question, TAR's temporal-localization text for the video,
and the 8 frames. Frames are not committed.

| Category | Count | Question IDs |
|---|---|---|
| Question is about a different or earlier event than the annotated interval | 9 | q0002, q0016, q0028, q0032, q0038, q0046, q0048, q0052, q0056 |
| Clear collision visible in the frames; model answered "No" | 3 | q0020, q0024, q0058 |
| Visible but subtle (slow reverse into a pillar) | 1 | q0010 |
| Collision small, distant, or at night in rain | 2 | q0026, q0040 |
| Dashcam, not a fixed camera | 1 | q0014 |
| Not reviewed | 1 | q0030 |

## Findings

- **The event-centred diagnostic understates the model.** TAR has one annotated
  interval per video, while yes/no questions often ask about a different collision or
  about behaviour before it (red-light running, wrong-way driving). For those, frames
  on the annotated interval cannot show the asked-about event.
- **Some misses are genuine model errors** on obvious collisions, which supports
  calibrating a Yes threshold and testing prompts.
- **Some collisions are small or dark** in 448×252 frames, which supports one
  resolution test.
- **Data issues for D's manifest:** q0024 and q0058 are two Accident-Bench clips of the
  same incident, so splitting by video does not prevent leakage; the manifest should
  de-duplicate or split by incident. Some clips are dashcam footage rather than fixed
  cameras and should be filtered for the fixed-camera workload.
