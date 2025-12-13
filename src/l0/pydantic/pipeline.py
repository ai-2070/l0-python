"""Pydantic models for L0 pipeline types.

These models mirror the dataclasses in l0.pipeline for runtime validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StepContextModel(BaseModel):
    """Pydantic model for context passed to each pipeline step."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    total_steps: int
    previous_results: list["StepResultModel"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cancelled: bool = False


class StepResultModel(BaseModel):
    """Pydantic model for result from a single pipeline step."""

    model_config = ConfigDict(extra="forbid")

    step_name: str
    step_index: int
    input: Any
    output: Any | None
    raw_content: str
    status: Literal["success", "error", "skipped"]
    error: str | None = None  # Serialized as string
    duration: int = 0
    startTime: int = 0
    endTime: int = 0
    token_count: int = 0


# Forward reference update
StepContextModel.model_rebuild()


class PipelineStepModel(BaseModel):
    """Pydantic model for pipeline step configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Functions (fn, transform, condition, etc.) are not serializable


class PipelineOptionsModel(BaseModel):
    """Pydantic model for pipeline configuration options."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    stop_on_error: bool = True
    timeout: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Callbacks are not serializable


class PipelineResultModel(BaseModel):
    """Pydantic model for result from pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    name: str | None
    output: Any | None
    steps: list[StepResultModel]
    status: Literal["success", "error", "partial"]
    error: str | None = None  # Serialized as string
    duration: int = 0
    startTime: int = 0
    endTime: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
