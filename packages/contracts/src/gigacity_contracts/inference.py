"""Edge-to-central inference contracts (draft).

`SelectedFrame` and `InferenceRequest` are the shared B -> C request from C/D's
`feature/shared-contracts` proposal. `InferenceResult` (C -> B/D) and `EdgeRecord`
(B -> D) are A/B proposals for the parts that request does not cover yet.
"""

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

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


class SelectedFrame(BaseModel):
    """One encoded image selected by the edge device."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capture_ms: int = Field(ge=0)
    codec: Literal["jpeg", "png", "webp"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    byte_count: int = Field(gt=0)
    data_ref: str = Field(min_length=1)

    @field_validator("data_ref")
    @classmethod
    def data_ref_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("data_ref must not be blank")
        return value


class InferenceRequest(BaseModel):
    """One request from an edge device to the central inference service."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: str = Field(min_length=1)
    window: WindowMetadata
    frames: list[SelectedFrame] = Field(min_length=1)
    total_bytes: int = Field(ge=0)
    selector: str = Field(min_length=1)
    budget_id: str | None = None
    query: str | None = None
    deadline_unix_ms: int | None = Field(default=None, ge=0)
    sent_at_unix_ms: int = Field(ge=0)

    @field_validator("run_id", "selector")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("budget_id", "query")
    @classmethod
    def optional_strings_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank when provided")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        start_ms = self.window.start_ms
        end_ms = self.window.end_ms
        timestamps = [frame.capture_ms for frame in self.frames]

        if any(ts < start_ms or ts >= end_ms for ts in timestamps):
            raise ValueError(
                "all frame timestamps must fall inside [window.start_ms, window.end_ms)"
            )

        if any(
            current >= following
            for current, following in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("frame timestamps must be strictly increasing")

        expected_bytes = sum(frame.byte_count for frame in self.frames)
        if self.total_bytes != expected_bytes:
            raise ValueError(
                "total_bytes must equal the sum of frame byte_count "
                f"(expected {expected_bytes}, got {self.total_bytes})"
            )

        return self


class InferenceResult(BaseModel):
    """C -> B/D: outcome for one request, including failures."""

    model_config = _STRICT

    run_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    status: InferenceStatus
    answer_text: str | None = None
    answer: ParsedAnswer | None = None
    # P(yes) / (P(yes) + P(no)) from the first answer token, when the server reports it.
    yes_probability: float | None = Field(default=None, ge=0, le=1)
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
