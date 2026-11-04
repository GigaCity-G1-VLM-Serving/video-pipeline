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

## Handoffs awaiting review

- **D → B replay input:** source location, replay timing, and window-generation rules.
- **B → C evidence request:** window metadata plus selected frames **or** video clips.
  Agree on encoding, capture timestamps, byte accounting, and query/deadline semantics.
- **C → B budget feedback:** applicable camera/future window, budget units, and expiry.
- **B/C → D observations:** correlated result/status, bytes, and stage timings.
  Specify how rejected, failed, late, and dropped work remains visible to evaluation.

There are intentionally no payload, result, or budget Python models until these
shapes are reviewed. Neither media choice is implied by `WindowMetadata`.

## Changing a contract

Review the producer and consumer together. Update types, this document, example
messages, and tests in the same change. Record units and failure behavior explicitly.
Keep evaluation labels out of online inputs; separate requested budgets from actual
observed work. Review both sides of a handoff before proposing a change.
