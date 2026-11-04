"""Source-video window identity; independent of the eventual media payload."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WindowMetadata(BaseModel):
    """Draft half-open window with millisecond offsets into a source video."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    window_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    source_video_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @field_validator("window_id", "camera_id", "source_video_id")
    @classmethod
    def identifiers_are_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifiers must not be blank")
        return value

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self
