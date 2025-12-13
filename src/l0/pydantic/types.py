"""Pydantic models for L0 core types.

These models mirror the dataclasses in l0.types for runtime validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventTypeModel(str, Enum):
    """Type of streaming event."""

    TOKEN = "token"
    MESSAGE = "message"
    DATA = "data"
    PROGRESS = "progress"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    COMPLETE = "complete"


class ContentTypeModel(str, Enum):
    """Type of multimodal content."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    JSON = "json"
    BINARY = "binary"


class ErrorCategoryModel(str, Enum):
    """Category of error for retry decisions."""

    NETWORK = "network"
    TRANSIENT = "transient"
    MODEL = "model"
    CONTENT = "content"
    PROVIDER = "provider"
    FATAL = "fatal"
    INTERNAL = "internal"


class BackoffStrategyModel(str, Enum):
    """Backoff strategy for retries."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FULL_JITTER = "full-jitter"
    FIXED_JITTER = "fixed-jitter"


class RetryableErrorTypeModel(str, Enum):
    """Error types that can be retried."""

    ZERO_OUTPUT = "zero_output"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    DRIFT = "drift"
    INCOMPLETE = "incomplete"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"


class DataPayloadModel(BaseModel):
    """Pydantic model for multimodal data payload."""

    model_config = ConfigDict(extra="forbid")

    content_type: ContentTypeModel
    mime_type: str | None = None
    base64: str | None = None
    url: str | None = None
    data: bytes | None = None
    json_data: Any = Field(default=None, alias="json")
    metadata: dict[str, Any] | None = None


class ProgressModel(BaseModel):
    """Pydantic model for progress updates."""

    model_config = ConfigDict(extra="forbid")

    percent: float | None = None
    step: int | None = None
    total_steps: int | None = None
    message: str | None = None
    eta: float | None = None


class EventModel(BaseModel):
    """Pydantic model for unified stream events."""

    model_config = ConfigDict(extra="forbid")

    type: EventTypeModel
    text: str | None = None
    data: dict[str, Any] | None = None
    payload: DataPayloadModel | None = None
    progress: ProgressModel | None = None
    error: str | None = None  # Serialized as string for Pydantic
    usage: dict[str, int] | None = None
    timestamp: float | None = None


class ErrorTypeDelaysModel(BaseModel):
    """Pydantic model for per-error-type delay configuration."""

    model_config = ConfigDict(extra="forbid")

    connection_dropped: float | None = 1.0
    fetch_error: float | None = 0.5
    econnreset: float | None = 1.0
    econnrefused: float | None = 2.0
    sse_aborted: float | None = 0.5
    no_bytes: float | None = 0.5
    partial_chunks: float | None = 0.5
    runtime_killed: float | None = 2.0
    background_throttle: float | None = 5.0
    dns_error: float | None = 3.0
    ssl_error: float | None = 0.0
    timeout: float | None = 1.0
    unknown: float | None = 1.0


class RetryModel(BaseModel):
    """Pydantic model for retry configuration."""

    model_config = ConfigDict(extra="forbid")

    attempts: int | None = 3
    max_retries: int | None = 6
    base_delay: float | None = 1.0
    max_delay: float | None = 10.0
    strategy: BackoffStrategyModel | None = BackoffStrategyModel.FIXED_JITTER
    error_type_delays: ErrorTypeDelaysModel | None = None
    retry_on: list[RetryableErrorTypeModel] | None = None


class TimeoutModel(BaseModel):
    """Pydantic model for timeout configuration."""

    model_config = ConfigDict(extra="forbid")

    initial_token: int = 5000
    inter_token: int = 10000


class CheckIntervalsModel(BaseModel):
    """Pydantic model for check frequency configuration."""

    model_config = ConfigDict(extra="forbid")

    guardrails: int = 15
    drift: int = 25
    checkpoint: int = 20


class StateModel(BaseModel):
    """Pydantic model for runtime state tracking."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    checkpoint: str = ""
    token_count: int = 0
    model_retry_count: int = 0
    network_retry_count: int = 0
    fallback_index: int = 0
    violations: list[Any] = Field(default_factory=list)  # GuardrailViolation
    drift_detected: bool = False
    completed: bool = False
    aborted: bool = False
    first_token_at: float | None = None
    last_token_at: float | None = None
    duration: float | None = None
    resumed: bool = False
    network_errors: list[Any] = Field(default_factory=list)
    data_outputs: list[DataPayloadModel] = Field(default_factory=list)
    last_progress: ProgressModel | None = None
    resume_point: str | None = None
    resume_from: int | None = None
    continuation_used: bool = False
    deduplication_applied: bool = False
    overlap_removed: str | None = None


class TelemetryMetricsModel(BaseModel):
    """Pydantic model for telemetry metrics."""

    model_config = ConfigDict(extra="forbid")

    time_to_first_token: float | None = None
    avg_inter_token_time: float | None = None
    tokens_per_second: float | None = None
    total_tokens: int = 0
    total_retries: int = 0
    network_retry_count: int = 0
    model_retry_count: int = 0


class TelemetryNetworkModel(BaseModel):
    """Pydantic model for network telemetry."""

    model_config = ConfigDict(extra="forbid")

    error_count: int = 0
    errors_by_type: dict[str, int] = Field(default_factory=dict)
    errors: list[dict[str, Any]] | None = None


class TelemetryGuardrailsModel(BaseModel):
    """Pydantic model for guardrail telemetry."""

    model_config = ConfigDict(extra="forbid")

    violation_count: int = 0
    violations_by_rule: dict[str, int] = Field(default_factory=dict)
    violations_by_rule_and_severity: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    violations_by_severity: dict[str, int] = Field(default_factory=dict)


class TelemetryDriftModel(BaseModel):
    """Pydantic model for drift telemetry."""

    model_config = ConfigDict(extra="forbid")

    detected: bool = False
    types: list[str] = Field(default_factory=list)


class TelemetryContinuationModel(BaseModel):
    """Pydantic model for continuation telemetry."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    used: bool = False
    checkpoint_content: str | None = None
    checkpoint_length: int | None = None
    continuation_count: int | None = None


class TelemetryModel(BaseModel):
    """Pydantic model for full telemetry."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    start_time: float
    end_time: float | None = None
    duration: float | None = None
    metrics: TelemetryMetricsModel
    network: TelemetryNetworkModel
    guardrails: TelemetryGuardrailsModel | None = None
    drift: TelemetryDriftModel | None = None
    continuation: TelemetryContinuationModel | None = None
    metadata: dict[str, Any] | None = None
