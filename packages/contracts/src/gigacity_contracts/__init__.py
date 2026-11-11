"""Shared, draft data contracts for GigaCity components."""

from gigacity_contracts.inference import (
    EdgeRecord,
    InferenceRequest,
    InferenceResult,
    SelectedFrame,
)
from gigacity_contracts.windows import WindowMetadata

__all__ = [
    "WindowMetadata",
    "SelectedFrame",
    "InferenceRequest",
    "InferenceResult",
    "EdgeRecord",
]
