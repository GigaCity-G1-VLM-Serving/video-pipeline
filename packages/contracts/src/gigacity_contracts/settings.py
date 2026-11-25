"""Static pipeline configuration shared by edge (A/B) and central (C/D).

These values are fixed per pipeline release rather than negotiated at runtime, so
they live in code instead of an HTTP endpoint. Bump ``CONFIG_VERSION`` whenever any
value here changes; every ``InferenceResult`` records the version that produced it.

Status: proposed for team review.
"""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class Task(StrEnum):
    """What C is asked to decide about a window."""

    COLLISION_YN = "collision_yn"


class PromptVersion(StrEnum):
    """Versioned prompt templates; C renders prompts only from these."""

    V1 = "v1"


class Codec(StrEnum):
    """Accepted encodings for selected frames."""

    JPEG = "jpeg"


# Proposed central model (Qwen3-VL-30B-A3B, FP8), pinned to a Hugging Face revision.
MODEL_ID = "qwen3-vl-30b-a3b-instruct-fp8@d9748a51"

CONFIG_VERSION = "2026-11-25"

JPEG_DATA_URI_PREFIX = "data:image/jpeg;base64,"


@dataclass(frozen=True)
class Generation:
    """Decoding settings C applies to every request."""

    temperature: float = 0.0
    max_tokens: int = 16


GENERATION = Generation()


@dataclass(frozen=True)
class PromptTemplate:
    """Engine-independent prompt pieces.

    C builds the model input as: for each frame in order, ``frame_caption(i, t)``
    followed by the image; then ``question(query)``.
    """

    default_query: str
    frame_caption_format: str
    answer_suffix: str

    def frame_caption(self, index: int, capture_ms: int) -> str:
        """Caption for the 1-based ``index``-th frame captured at ``capture_ms``."""
        return self.frame_caption_format.format(index=index, seconds=capture_ms / 1000)

    def question(self, query: str | None) -> str:
        """Final text: the request's query (or the task default) plus the suffix."""
        return (query or self.default_query) + self.answer_suffix


PROMPTS: MappingProxyType[tuple[Task, PromptVersion], PromptTemplate] = (
    MappingProxyType(
        {
            # Identical to the November 10-11 A/B baseline prompt.
            (Task.COLLISION_YN, PromptVersion.V1): PromptTemplate(
                default_query=(
                    "Did any traffic collision occur during the shown observation "
                    "window?"
                ),
                frame_caption_format="Frame {index} at {seconds:.1f} s:",
                answer_suffix=" Answer only Yes or No.",
            ),
        }
    )
)


def prompt_template(task: Task, version: PromptVersion) -> PromptTemplate:
    """Look up a template; raises KeyError for an unsupported combination."""
    return PROMPTS[(task, version)]


# HTTP paths served by C.
HEALTH_PATH = "/health"
BUDGET_PATH = "/v1/budget"
INFER_PATH = "/v1/infer"
