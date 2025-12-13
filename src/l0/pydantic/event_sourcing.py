"""Pydantic models for L0 event sourcing types.

These models mirror the dataclasses in l0.event_sourcing for runtime validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecordedEventTypeModel(str, Enum):
    """Types of recorded events."""

    START = "START"
    TOKEN = "TOKEN"
    CHECKPOINT = "CHECKPOINT"
    GUARDRAIL = "GUARDRAIL"
    DRIFT = "DRIFT"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    CONTINUATION = "CONTINUATION"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class RecordedEventModel(BaseModel):
    """Pydantic model for a recorded event from L0 execution."""

    model_config = ConfigDict(extra="forbid")

    type: RecordedEventTypeModel
    ts: float
    data: dict[str, Any] = Field(default_factory=dict)


class EventEnvelopeModel(BaseModel):
    """Pydantic model for wrapper around a recorded event."""

    model_config = ConfigDict(extra="forbid")

    stream_id: str
    seq: int
    event: RecordedEventModel


class SnapshotModel(BaseModel):
    """Pydantic model for state snapshot."""

    model_config = ConfigDict(extra="forbid")

    stream_id: str
    seq: int
    state: dict[str, Any]
    ts: float


class SerializedErrorModel(BaseModel):
    """Pydantic model for serialized error."""

    model_config = ConfigDict(extra="forbid")

    name: str
    message: str
    stack: str | None = None
    code: str | None = None


class ReplayedStateModel(BaseModel):
    """Pydantic model for state reconstructed from replay."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    token_count: int = 0
    checkpoint: str = ""
    violations: list[Any] = Field(default_factory=list)
    drift_detected: bool = False
    retry_attempts: int = 0
    network_retry_count: int = 0
    fallback_index: int = 0
    completed: bool = False
    error: SerializedErrorModel | None = None
    start_ts: float = 0
    end_ts: float = 0


class StreamMetadataModel(BaseModel):
    """Pydantic model for stream metadata."""

    model_config = ConfigDict(extra="forbid")

    stream_id: str
    event_count: int
    token_count: int
    start_ts: float
    end_ts: float
    completed: bool
    has_error: bool
    options: dict[str, Any]


class ReplayComparisonModel(BaseModel):
    """Pydantic model for comparing two replays."""

    model_config = ConfigDict(extra="forbid")

    identical: bool
    differences: list[str]
