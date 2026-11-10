"""Edge-to-central inference request contract."""

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from gigacity_contracts.windows import WindowMetadata


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
    def optional_strings_must_not_be_blank(
        cls, value: str | None
    ) -> str | None:
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
                "all frame timestamps must fall inside "
                "[window.start_ms, window.end_ms)"
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
