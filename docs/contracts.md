# Draft contracts

**Status: proposed for team review, not a frozen API.** Pydantic models in
`packages/contracts/src/gigacity_contracts` define the implemented draft metadata.
This document explains its meaning. [Example JSON](examples/window.json) is loaded
by contract tests so it stays aligned with the model.

## Implemented: window metadata

`WindowMetadata` identifies one observation window without choosing its media format.

- `window_id`: nonempty request/window correlation ID; producers must keep it stable.
- `camera_id`: nonempty logical replay-camera ID.
- `source_video_id`: nonempty source-video identifier, not an evaluation answer.
- `start_ms`, `end_ms`: integer milliseconds relative to the start of the source video.
  The window is half-open `[start_ms, end_ms)`, with `0 <= start_ms < end_ms`.
  These values are source-video times, not wall-clock times or latency measurements.

Values are strict: strings are not coerced to integers, blank IDs and extra fields
are rejected, and instances are immutable. This model cannot prove replay causality,
ID uniqueness, or that evidence exists; producers and evaluation must check those.

Use the shared package at both ends:

```python
from gigacity_contracts import WindowMetadata

window = WindowMetadata.model_validate_json(message)
encoded = window.model_dump_json()
schema = WindowMetadata.model_json_schema()
```

The example contains synthetic metadata only. It is not a model prediction or a
measurement. Invalid inputs and serialization round-trips are exercised in tests.

## Draft: inference request, result, and per-window records

**Status: draft.** Models live in `gigacity_contracts.inference`; rationale and status
are in [decisions](decisions.md).

`InferenceRequest` and `SelectedFrame` (B → C) are C/D's shared request from
`feature/shared-contracts`, documented field by field in
[`b_to_c.md`](../packages/contracts/src/gigacity_contracts/b_to_c.md)
([example](examples/inference_request.json)). A/B have adopted them unchanged:

- `window.window_id` doubles as the request ID; `query` carries the question.
- **Proposed v1 transport:** `data_ref` holds an inline data URI
  (`data:image/jpeg;base64,...`), which C can pass straight to vLLM as an image URL.
  The model does not check `byte_count` against the payload; A/B's edge tests do.
- The request has no frame budget field. For fixed-budget runs `budget_id` is null and
  the budget is recorded in `EdgeRecord.frame_budget` and the window ID.

The request carries frames, not clips. C passes them to the model as **separate
images** with their capture times in the prompt, not as model "video" input: for
Qwen2.5-VL, video input assumes evenly spaced frames and would distort selected,
irregularly spaced frames. See [decisions](decisions.md).

`InferenceResult` and `EdgeRecord` are A/B proposals awaiting C/D review; the shared
request does not define a response yet.

`InferenceResult` (C → B/D, [example](examples/inference_result.json)): `status` is
`completed`, `rejected`, `failed`, or `late`; completed results require a parsed
`answer` (`yes`, `no`, `unparsed`) and may include raw text, token counts, queue and
inference durations, and wall-clock receipt/completion times. `is_synthetic` marks
mock or stand-in servers so they are never mistaken for C's serving measurements.
Optional `yes_probability` is P(yes) / (P(yes) + P(no)) from the first answer token,
so evaluation can set a Yes threshold on calibration data instead of using raw text.

`EdgeRecord` (B → D, [example](examples/edge_record.json)): one per window, including
windows that were never sent (`dropped`, `skipped`) or sent partially
(`link_limited`). It records the selector, budget, selected capture times, bytes,
per-stage edge durations, B-clock round trip, and any simulated network delay.

Clocks: `*_ms` offsets are source-video time; `*_unix_ms` values are wall-clock and
used only for latency; `*_duration_ms` and stage `*_ms` fields are measured on one
machine. End-to-end latency is measured on B's clock to avoid cross-machine skew.
D joins records by `run_id` and `window_id`; a window without a completed result
counts as a miss.

## Handoffs still awaiting review

- **D → B replay input:** source location, replay timing, and window-generation rules.
  A temporary A/B pilot list stands in until D's manifest exists.
- **C → B budget feedback:** proposed as C pushing a frame budget (4, 8, or 16) for
  window N+1 when window N ends, defaulting to 4; no model yet.
- **Deadlines and late work, transport details, and telemetry format** beyond JSONL.

Neither media choice is implied by `WindowMetadata`; the frames proposal above is
labeled as such until both teams agree.

## Changing a contract

Review the producer and consumer together. Update types, this document, example
messages, and tests in the same change. Record units and failure behavior explicitly.
Keep evaluation labels out of online inputs; separate requested budgets from actual
observed work. Run the contract-review skill before handing off a proposed change.
