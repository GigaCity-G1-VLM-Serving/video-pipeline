"""Shared, draft data contracts for GigaCity components."""

from gigacity_contracts.inference import (
    Budget,
    BudgetResponse,
    ErrorInfo,
    ErrorResponse,
    InferenceRequest,
    InferenceResult,
    SelectedFrame,
)
from gigacity_contracts.settings import (
    CONFIG_VERSION,
    MODEL_ID,
    Codec,
    PromptTemplate,
    PromptVersion,
    Task,
    prompt_template,
)
from gigacity_contracts.windows import WindowMetadata

__all__ = [
    "CONFIG_VERSION",
    "MODEL_ID",
    "Budget",
    "BudgetResponse",
    "Codec",
    "ErrorInfo",
    "ErrorResponse",
    "InferenceRequest",
    "InferenceResult",
    "PromptTemplate",
    "PromptVersion",
    "SelectedFrame",
    "Task",
    "WindowMetadata",
    "prompt_template",
]
