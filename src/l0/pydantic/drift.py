"""Pydantic models for L0 drift detection types.

These models mirror the dataclasses in l0.drift for runtime validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DriftType = Literal[
    "tone_shift",
    "meta_commentary",
    "format_collapse",
    "repetition",
    "entropy_spike",
    "markdown_collapse",
    "hedging",
]


class DriftResultModel(BaseModel):
    """Pydantic model for drift detection result."""

    model_config = ConfigDict(extra="forbid")

    detected: bool
    confidence: float
    types: list[DriftType]
    details: str | None = None


class DriftConfigModel(BaseModel):
    """Pydantic model for drift detection configuration."""

    model_config = ConfigDict(extra="forbid")

    detect_tone_shift: bool = True
    detect_meta_commentary: bool = True
    detect_repetition: bool = True
    detect_entropy_spike: bool = True
    repetition_threshold: int = 3
    entropy_threshold: float = 2.5
    entropy_window: int = 50
    sliding_window_size: int = 500
