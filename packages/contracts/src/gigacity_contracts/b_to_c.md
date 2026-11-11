# B → C Inference Request Contract

**Status:** Draft for team review  
**Scope:** Initial frame-based edge-to-central inference interface

## 1. Purpose

This document defines the draft data contract between the edge component (B)
and the central inference component (C).

B selects and encodes visual evidence from one camera window, then sends an
inference request to C. C uses the supplied evidence to answer the query.

The initial draft supports selected image frames. Clip-based evidence is not
included yet.

## 2. Public models

The models are provided by the `gigacity_contracts` Python package:

```python
from gigacity_contracts import (
    InferenceRequest,
    SelectedFrame,
    WindowMetadata,
)
```

### `WindowMetadata`

Reuses the existing shared model. It identifies the camera, source video,
and half-open source-video interval `[start_ms, end_ms)`.

### `SelectedFrame`

Represents one encoded image selected by B.

| Field | Meaning |
|---|---|
| `capture_ms` | Capture time in milliseconds from the start of the source video |
| `codec` | Image codec: `jpeg`, `png`, or `webp` |
| `width` | Image width in pixels |
| `height` | Image height in pixels |
| `byte_count` | Number of encoded image bytes |
| `data_ref` | Reference to the encoded image payload |

### `InferenceRequest`

Represents one request from B to C for one camera window.

| Field | Meaning |
|---|---|
| `run_id` | Experiment/run identifier |
| `window` | `WindowMetadata` identifying the camera and source-video interval |
| `frames` | Selected image frames, in chronological order |
| `total_bytes` | Sum of the encoded image byte counts |
| `selector` | Selection method used by B, e.g. `uniform` or `motion-v1` |
| `budget_id` | ID of the C-issued budget applied by B, or `null` |
| `query` | Optional question/instruction for C |
| `deadline_unix_ms` | Optional Unix timestamp deadline in milliseconds |
| `sent_at_unix_ms` | Unix timestamp when B sent the request, in milliseconds |

## 3. Validation rules

The Pydantic models enforce these rules:

1. Unknown fields are rejected.
2. Fields use strict types; implicit conversions are not accepted.
3. Models are immutable after creation.
4. A request must contain at least one frame.
5. Frame timestamps must be within `[window.start_ms, window.end_ms)`.
6. Frame timestamps must be strictly increasing.
7. Image width, height, and encoded byte count must be positive.
8. `total_bytes` must equal the sum of all frame `byte_count` values.
9. Required identifiers and optional string values, when supplied, must not be blank.

`byte_count` and `total_bytes` refer to encoded image bytes. They do not include
Base64 expansion, JSON overhead, or HTTP headers.

## 4. Image transport — pending agreement

The exact transport mechanism must be agreed by B and C (need to be coordinated with
Aryan). Possible approaches include inline Base64, URI/object-storage references,
or multipart HTTP.

For this draft, `data_ref` is a string placeholder for the agreed payload
reference. The example value is illustrative and is not a claim that a storage
service or URI scheme has been selected.

Before integration, document:
- how B transmits the image bytes;
- how C resolves or receives `data_ref`;
- whether references are inline data, URIs, or multipart part identifiers;
- access control, lifetime, and cleanup expectations if external storage is used.

## 5. Scope boundaries

This contract covers the B → C request only.

It does not define:
- C → B budget updates;
- C's inference response format;
- tracing/span schemas;
- clip-based evidence;
- the HTTP endpoint or deployment configuration.

## 6. Example

See [`docs/examples/inference_request.json`](../examples/inference_request.json).

## 7. Open questions for review

- Should the public names be `InferenceRequest` / `SelectedFrame`?
- Should `query` be a string or a structured `QuerySpec`?
- Which image transport will B and C use?
- Does C need clip-based evidence in the first integration?
