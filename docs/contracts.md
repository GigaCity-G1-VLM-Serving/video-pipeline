# Draft contracts

**Status: proposed for team review, not a frozen API.** Pydantic models in
`packages/contracts/src/gigacity_contracts` define the implemented draft messages.
This document explains their meaning. [`examples/window.json`](examples/window.json)
and the JSON examples in this document are loaded by contract tests so they stay
aligned with the models.

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

## Proposed: B ↔ C inference API

**Status: proposed for joint A/B and C/D review.** It covers the first experiment:
whole-video collision detection, where each source video is one window
`[0, duration_ms)` and C answers Yes or No with a confidence. Models live in
`gigacity_contracts.inference`; static settings live in `gigacity_contracts.settings`.

### Shared pipeline config (code, not API)

Values that are fixed for a pipeline release are imported by both B and C rather than
served at runtime:

- `Task`: what C decides. Only `collision_yn` for now.
- `PromptVersion` and `prompt_template(task, version)`: the only source of prompt
  text. C builds the model input as, for each frame in order, the frame caption
  (`Frame 1 at 1.2 s:`) and then the image, followed by `question(query)`, which
  is the request's `query` (or the task's default question) plus the answer suffix.
  `v1` matches the November 10–11 A/B baseline prompt.
- `Codec`: accepted frame encodings. Only `jpeg`.
- `MODEL_ID`, `GENERATION` (temperature 0, at most 16 new tokens).
- `CONFIG_VERSION`: bump it whenever any of the above changes. Results record it.
- `HEALTH_PATH`, `BUDGET_PATH`, `INFER_PATH`: C's HTTP paths.

### Endpoints

| Method and path | Body | Response |
|---|---|---|
| `GET /health` | none | `200`, body `ok` |
| `GET /v1/budget` | none | `200`, `BudgetResponse` |
| `POST /v1/infer` | `InferenceRequest` | `200`, `InferenceResult` |

`POST /v1/infer` is synchronous: one request, one result.

- **HTTP 200 + `InferenceResult`** for every body that validates as an
  `InferenceRequest`. A request that breaks C's current budget comes back as
  `status: "rejected"` with an `error`, so evaluation can tell refusals from network
  failures. A backend failure comes back as `status: "failed"` with an `error`.
- **HTTP 400 + `ErrorResponse`** when the body is not valid JSON or fails model
  validation. `request_id` is included when C can read it.
- **HTTP 413 + `ErrorResponse`** (`request_too_large`) when the body exceeds
  `max_request_bytes`.

### `BudgetResponse`

`budget` holds the limits C enforces now: `max_frames`, `max_image_side` (pixels, the
larger of width and height), and `max_request_bytes` (the HTTP body size). B should
read it before a run and keep requests within it. In live-stream mode this response
is where a per-camera budget and `budget_id` will go.

Example:

<!-- example: budget_response -->
```json
{
  "budget": {
    "max_frames": 16,
    "max_image_side": 448,
    "max_request_bytes": 4000000
  }
}
```

### `InferenceRequest`

| Field | Meaning |
|---|---|
| `request_id` | Unique within a run; the correlation key for results and evaluation |
| `run_id` | Experiment or run ID |
| `window` | `WindowMetadata`; `window_id` names only the time window, not the budget or question |
| `task` | A shared `Task` value |
| `prompt_version` | A shared `PromptVersion`; defaults to `v1` |
| `query` | Optional question text; null means the task's default question |
| `frames` | One or more `SelectedFrame`s, strictly increasing in `capture_ms`, inside the window |
| `total_bytes` | Sum of the frames' `byte_count` |
| `selector` | Edge selection method, e.g. `uniform`, `motion` |
| `frame_budget` | Frame budget B applied: 4, 8, or 16; `frames` must not exceed it |
| `sent_at_unix_ms` | When B sent the request, on B's clock |
| `budget_id`, `deadline_unix_ms` | Reserved for live-stream mode; null in this phase |

`SelectedFrame`: `capture_ms` (milliseconds from the start of the source video),
`codec`, `width`, `height` (pixels), `byte_count` (encoded image bytes, excluding
base64 and HTTP overhead), and `data_ref`, an inline `data:image/jpeg;base64,...`
URI. The model checks that the base64 decodes to exactly `byte_count` bytes, so B
and C apply the same transport check.

Example (synthetic payload bytes, not a real image):

<!-- example: inference_request -->
```json
{
  "request_id": "demo-run-01:video-0001:b4",
  "run_id": "demo-run-01",
  "window": {
    "window_id": "demo-camera-01:video-0001:0-10000",
    "camera_id": "demo-camera-01",
    "source_video_id": "synthetic-video-01",
    "start_ms": 0,
    "end_ms": 10000
  },
  "task": "collision_yn",
  "prompt_version": "v1",
  "query": null,
  "frames": [
    {
      "capture_ms": 1200,
      "codec": "jpeg",
      "width": 448,
      "height": 252,
      "byte_count": 12,
      "data_ref": "data:image/jpeg;base64,/9j/4AECAwQFBv/Z"
    },
    {
      "capture_ms": 7400,
      "codec": "jpeg",
      "width": 448,
      "height": 252,
      "byte_count": 14,
      "data_ref": "data:image/jpeg;base64,/9j/4AcICQoLDA0O/9k="
    }
  ],
  "total_bytes": 26,
  "selector": "uniform",
  "frame_budget": 4,
  "sent_at_unix_ms": 1794355200000,
  "budget_id": null,
  "deadline_unix_ms": null
}
```

### `InferenceResult`

| Field | Meaning |
|---|---|
| `request_id`, `run_id`, `window_id` | Echoed from the request |
| `status` | `completed`, `rejected`, `failed`, or `late` (reserved for deadlines) |
| `answer` | `yes`, `no`, or `unparsed` (first word of the model text); required when completed |
| `answer_text` | Raw model text |
| `confidence` | Confidence in `answer`, in percent (0–100); required for `yes`/`no`, null otherwise. "No, 82" means P(no) = 0.82 |
| `error` | `{code, message}`; required for `rejected` and `failed`, absent when completed |
| `config_version` | `CONFIG_VERSION` of the pipeline config that produced the result |
| `frames_used` | Frames passed to the model |
| `prompt_tokens`, `completion_tokens` | As reported by the serving engine |
| `received_at_unix_ms`, `completed_at_unix_ms` | On C's clock |
| `queue_duration_ms`, `inference_duration_ms` | Time waiting in C and time in the model |

Error codes: `invalid_request`, `too_many_frames`, `image_too_large`,
`request_too_large`, `backend_error`, `backend_timeout`.

Example:

<!-- example: inference_result -->
```json
{
  "request_id": "demo-run-01:video-0001:b4",
  "run_id": "demo-run-01",
  "window_id": "demo-camera-01:video-0001:0-10000",
  "status": "completed",
  "answer": "yes",
  "answer_text": "Yes",
  "confidence": 87.5,
  "error": null,
  "config_version": "2026-11-25",
  "frames_used": 2,
  "prompt_tokens": 580,
  "completion_tokens": 2,
  "received_at_unix_ms": 1794355200040,
  "completed_at_unix_ms": 1794355200910,
  "queue_duration_ms": 15,
  "inference_duration_ms": 850
}
```

Example `ErrorResponse` (HTTP 400):

<!-- example: error_response -->
```json
{
  "error": {
    "code": "invalid_request",
    "message": "frame capture_ms must be strictly increasing"
  },
  "request_id": "demo-run-01:video-0001:b4"
}
```

End-to-end latency is measured on B's clock (`sent_at_unix_ms` to receipt) to avoid
cross-machine clock skew. C's timestamps describe work inside C only.

### Not covered yet

- **D → B replay input:** dataset manifest, labels kept separate, and window rules.
- **B → D per-window records**, including windows that were never sent.
- **Live-stream budget feedback** and deadline handling (`budget_id`, `late`).
- **Clips instead of frames:** this proposal uses frames only.

## Changing a contract

Review the producer and consumer together. Update types, this document, example
messages, and tests in the same change. Record units and failure behavior explicitly.
Keep evaluation labels out of online inputs; separate requested budgets from actual
observed work. Run the contract-review skill before handing off a proposed change.
