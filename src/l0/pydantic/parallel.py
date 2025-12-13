"""Pydantic models for L0 parallel execution types.

These models mirror the dataclasses in l0.parallel for runtime validation.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class AggregatedTelemetryModel(BaseModel):
    """Pydantic model for aggregated telemetry from parallel operations."""

    model_config = ConfigDict(extra="forbid")

    total_tokens: int = 0
    total_duration: float = 0.0
    total_retries: int = 0
    total_network_errors: int = 0
    total_violations: int = 0
    avg_tokens_per_second: float = 0.0
    avg_time_to_first_token: float = 0.0


class RaceResultModel(BaseModel):
    """Pydantic model for result from race operation."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    winner_index: int


class ParallelResultModel(BaseModel):
    """Pydantic model for result of parallel execution."""

    model_config = ConfigDict(extra="forbid")

    results: list[Any | None]
    errors: list[str | None]  # Serialized as strings
    success_count: int = 0
    failure_count: int = 0
    duration: float = 0.0
    aggregated_telemetry: AggregatedTelemetryModel | None = None


class ParallelOptionsModel(BaseModel):
    """Pydantic model for parallel execution options."""

    model_config = ConfigDict(extra="forbid")

    concurrency: int = 5
    fail_fast: bool = False
    # Callbacks are not serializable, so they are omitted
