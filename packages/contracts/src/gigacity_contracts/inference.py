"""B <-> C inference API messages (draft, proposed for team review).

``POST /v1/infer``: ``InferenceRequest`` -> ``InferenceResult``.
``GET /v1/budget``: ``BudgetResponse``.
Invalid bodies get HTTP 400 with an ``ErrorResponse``.
"""

import base64
import binascii
from itertools import pairwise
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gigacity_contracts.settings import (
    JPEG_DATA_URI_PREFIX,
    Codec,
    PromptVersion,
    Task,
)
from gigacity_contracts.windows import WindowMetadata

FrameBudget = Literal[4, 8, 16]
InferenceStatus = Literal["completed", "rejected", "failed", "late"]
Answer = Literal["yes", "no", "unparsed"]
ErrorCode = Literal[
    "invalid_request",
    "too_many_frames",
    "image_too_large",
    "request_too_large",
    "backend_error",
    "backend_timeout",
]

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


def _not_blank(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("must not be blank")
    return value


class SelectedFrame(BaseModel):
    """One encoded frame selected by the edge, with its source-video capture time."""

    model_config = _STRICT

    capture_ms: int = Field(ge=0)
    codec: Codec = Field(strict=False)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    byte_count: int = Field(gt=0)
    data_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def data_ref_matches_byte_count(self) -> Self:
        if not self.data_ref.startswith(JPEG_DATA_URI_PREFIX):
            raise ValueError(f"data_ref must start with {JPEG_DATA_URI_PREFIX!r}")
        encoded = self.data_ref.removeprefix(JPEG_DATA_URI_PREFIX)
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except binascii.Error as error:
            raise ValueError("data_ref must contain valid base64") from error
        if len(decoded) != self.byte_count:
            raise ValueError(
                f"byte_count {self.byte_count} does not match the decoded payload "
                f"size {len(decoded)}"
            )
        return self


class InferenceRequest(BaseModel):
    """B -> C: one request per window (one whole video in the first phase)."""

    model_config = _STRICT

    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    window: WindowMetadata
    task: Task = Field(strict=False)
    prompt_version: PromptVersion = Field(default=PromptVersion.V1, strict=False)
    query: str | None = None
    frames: list[SelectedFrame] = Field(min_length=1)
    total_bytes: int = Field(gt=0)
    selector: str = Field(min_length=1)
    frame_budget: FrameBudget
    sent_at_unix_ms: int = Field(ge=0)
    # Reserved for live-stream mode; unused (null) in the whole-video phase.
    budget_id: str | None = None
    deadline_unix_ms: int | None = Field(default=None, ge=0)

    _blank = field_validator("request_id", "run_id", "selector", "query", "budget_id")(
        _not_blank
    )

    @model_validator(mode="after")
    def frames_are_consistent(self) -> Self:
        times = [frame.capture_ms for frame in self.frames]
        if any(not self.window.start_ms <= t < self.window.end_ms for t in times):
            raise ValueError("frame capture_ms must fall inside [start_ms, end_ms)")
        if any(later <= earlier for earlier, later in pairwise(times)):
            raise ValueError("frame capture_ms must be strictly increasing")
        if len(self.frames) > self.frame_budget:
            raise ValueError("number of frames must not exceed frame_budget")
        expected = sum(frame.byte_count for frame in self.frames)
        if self.total_bytes != expected:
            raise ValueError(
                f"total_bytes must equal the sum of byte_count ({expected})"
            )
        return self


class Budget(BaseModel):
    """Limits C currently enforces on a request."""

    model_config = _STRICT

    max_frames: int = Field(gt=0)
    max_image_side: int = Field(gt=0)
    max_request_bytes: int = Field(gt=0)


class BudgetResponse(BaseModel):
    """C -> B: body of ``GET /v1/budget``."""

    model_config = _STRICT

    budget: Budget


class ErrorInfo(BaseModel):
    """Why a request was rejected or failed."""

    model_config = _STRICT

    code: ErrorCode
    message: str = Field(min_length=1)

    _blank = field_validator("message")(_not_blank)


class ErrorResponse(BaseModel):
    """C -> B: HTTP 400 body for a request that is not a valid InferenceRequest."""

    model_config = _STRICT

    error: ErrorInfo
    request_id: str | None = None


class InferenceResult(BaseModel):
    """C -> B/D: outcome of one request, including rejections and failures."""

    model_config = _STRICT

    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    status: InferenceStatus
    answer: Answer | None = None
    answer_text: str | None = None
    # Confidence in ``answer`` (not in "yes"), in percent.
    confidence: float | None = Field(default=None, ge=0, le=100)
    error: ErrorInfo | None = None
    config_version: str = Field(min_length=1)
    frames_used: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    received_at_unix_ms: int = Field(ge=0)
    completed_at_unix_ms: int | None = Field(default=None, ge=0)
    queue_duration_ms: int | None = Field(default=None, ge=0)
    inference_duration_ms: int | None = Field(default=None, ge=0)

    _blank = field_validator("request_id", "run_id", "window_id", "config_version")(
        _not_blank
    )

    @model_validator(mode="after")
    def status_matches_fields(self) -> Self:
        if self.status == "completed":
            if self.answer is None:
                raise ValueError("completed results must include an answer")
            if self.error is not None:
                raise ValueError("completed results must not include an error")
        if self.status in ("rejected", "failed"):
            if self.error is None:
                raise ValueError(f"{self.status} results must include an error")
            if self.answer is not None or self.answer_text is not None:
                raise ValueError(f"{self.status} results must not include an answer")
        if self.answer in ("yes", "no") and self.confidence is None:
            raise ValueError("yes/no answers must include a confidence")
        if self.answer not in ("yes", "no") and self.confidence is not None:
            raise ValueError("confidence is only allowed with a yes/no answer")
        return self
