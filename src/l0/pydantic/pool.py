"""Pydantic models for L0 pool types.

These models mirror the dataclasses in l0.pool for runtime validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .types import RetryModel, TimeoutModel


class PoolOptionsModel(BaseModel):
    """Pydantic model for operation pool configuration."""

    model_config = ConfigDict(extra="forbid")

    shared_retry: RetryModel | None = None
    shared_timeout: TimeoutModel | None = None
    shared_guardrails: list[Any] | None = None  # GuardrailRule not serializable
    context: dict[str, Any] | None = None
    # on_event callback is not serializable


class PoolStatsModel(BaseModel):
    """Pydantic model for operation pool statistics."""

    model_config = ConfigDict(extra="forbid")

    total_executed: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_duration: float = 0.0
