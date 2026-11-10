"""Proposed B -> C inference handoff and per-window records (draft, not agreed).

These models encode the A/B proposal recorded in ``docs/decisions.md``: JPEG frames
sent as separate images with source-video capture times. C/D review is pending.
"""

import base64
import binascii
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gigacity_contracts.windows import WindowMetadata

FrameBudget = Literal[4, 8, 16]
InferenceStatus = Literal["completed", "rejected", "failed", "late"]
ParsedAnswer = Literal["yes", "no", "unparsed"]
EdgeStatus = Literal["sent", "link_limited", "dropped", "skipped"]

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class EncodedFrame(BaseModel):
    """One selected frame, JPEG-encoded at the edge."""

    model_config = _STRICT

    capture_ms: int = Field(ge=0)
    codec: Literal["jpeg"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    byte_count: int = Field(gt=0)
    data_b64: str = Field(min_length=1)

    @model_validator(mode="after")
    def byte_count_matches_payload(self) -> Self:
        try:
            decoded = base64.b64decode(self.data_b64, validate=True)
        except binascii.Error as error:
            raise ValueError("data_b64 must be valid base64") from error
        if len(decoded) != self.byte_count:
            raise ValueError("byte_count must equal the decoded payload length")
        return self


class InferenceRequest(BaseModel):
    """B -> C: selected evidence and the question for one window."""

    model_config = _STRICT

    run_id: str = Field(min_length=1)
    window: WindowMetadata
    frame_budget: FrameBudget
    frames: list[EncodedFrame] = Field(min_length=1)
    total_bytes: int = Field(gt=0)
    selector: str = Field(min_length=1)
    question: str = Field(min_length=1)
    deadline_ms_after_window_end: int | None = Field(default=None, ge=0)
    sent_at_unix_ms: int = Field(ge=0)

    _blank_check = field_validator("run_id", "selector", "question")(_not_blank)

    @model_validator(mode="after")
    def frames_are_consistent(self) -> Self:
        if len(self.frames) > self.frame_budget:
            raise ValueError("frames must not exceed frame_budget")
        times = [frame.capture_ms for frame in self.frames]
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("capture_ms must be strictly increasing")
        if any(not self.window.start_ms <= t < self.window.end_ms for t in times):
            raise ValueError("capture_ms must fall inside [start_ms, end_ms)")
        if self.total_bytes != sum(frame.byte_count for frame in self.frames):
            raise ValueError("total_bytes must equal the sum of frame byte_count")
        return self


class InferenceResult(BaseModel):
    """C -> B/D: outcome for one request, including failures."""

    model_config = _STRICT

    run_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    status: InferenceStatus
    answer_text: str | None = None
    answer: ParsedAnswer | None = None
    model_id: str = Field(min_length=1)
    frames_used: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    queue_duration_ms: int | None = Field(default=None, ge=0)
    inference_duration_ms: int | None = Field(default=None, ge=0)
    received_at_unix_ms: int = Field(ge=0)
    completed_at_unix_ms: int | None = Field(default=None, ge=0)
    is_synthetic: bool

    _blank_check = field_validator("run_id", "window_id", "model_id")(_not_blank)

    @model_validator(mode="after")
    def completed_results_have_answers(self) -> Self:
        if self.status == "completed" and self.answer is None:
            raise ValueError("completed results must include a parsed answer")
        return self


class EdgeRecord(BaseModel):
    """B -> D: one record per window, including windows that were never sent."""

    model_config = _STRICT

    run_id: str = Field(min_length=1)
    window: WindowMetadata
    status: EdgeStatus
    selector: str = Field(min_length=1)
    frame_budget: FrameBudget
    selected_capture_ms: list[int]
    total_bytes: int = Field(ge=0)
    decode_ms: int = Field(ge=0)
    select_ms: int = Field(ge=0)
    encode_ms: int = Field(ge=0)
    round_trip_ms: int | None = Field(default=None, ge=0)
    simulated_network_ms: int | None = Field(default=None, ge=0)

    _blank_check = field_validator("run_id", "selector")(_not_blank)

    @model_validator(mode="after")
    def selection_is_consistent(self) -> Self:
        times = self.selected_capture_ms
        if len(times) > self.frame_budget:
            raise ValueError("selected frames must not exceed frame_budget")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("selected_capture_ms must be strictly increasing")
        if any(not self.window.start_ms <= t < self.window.end_ms for t in times):
            raise ValueError("selected_capture_ms must fall inside the window")
        return self
