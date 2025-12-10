"""Pydantic models for L0 events/observability types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ObservabilityEventTypeModel(str, Enum):
    """Observability event types."""

    # Session events
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    SESSION_SUMMARY = "SESSION_SUMMARY"
    ATTEMPT_START = "ATTEMPT_START"

    # Stream events
    STREAM_INIT = "STREAM_INIT"
    STREAM_READY = "STREAM_READY"
    TOKEN = "TOKEN"

    # Adapter events
    ADAPTER_DETECTED = "ADAPTER_DETECTED"
    ADAPTER_WRAP_START = "ADAPTER_WRAP_START"
    ADAPTER_WRAP_END = "ADAPTER_WRAP_END"

    # Timeout events
    TIMEOUT_START = "TIMEOUT_START"
    TIMEOUT_RESET = "TIMEOUT_RESET"
    TIMEOUT_TRIGGERED = "TIMEOUT_TRIGGERED"

    # Network events
    NETWORK_ERROR = "NETWORK_ERROR"
    NETWORK_RECOVERY = "NETWORK_RECOVERY"
    CONNECTION_DROPPED = "CONNECTION_DROPPED"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"

    # Abort events
    ABORT_REQUESTED = "ABORT_REQUESTED"
    ABORT_COMPLETED = "ABORT_COMPLETED"

    # Guardrail events
    GUARDRAIL_PHASE_START = "GUARDRAIL_PHASE_START"
    GUARDRAIL_RULE_START = "GUARDRAIL_RULE_START"
    GUARDRAIL_RULE_RESULT = "GUARDRAIL_RULE_RESULT"
    GUARDRAIL_RULE_END = "GUARDRAIL_RULE_END"
    GUARDRAIL_PHASE_END = "GUARDRAIL_PHASE_END"
    GUARDRAIL_CALLBACK_START = "GUARDRAIL_CALLBACK_START"
    GUARDRAIL_CALLBACK_END = "GUARDRAIL_CALLBACK_END"

    # Drift events
    DRIFT_CHECK_START = "DRIFT_CHECK_START"
    DRIFT_CHECK_RESULT = "DRIFT_CHECK_RESULT"
    DRIFT_CHECK_END = "DRIFT_CHECK_END"
    DRIFT_CHECK_SKIPPED = "DRIFT_CHECK_SKIPPED"

    # Checkpoint events
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    RESUME_START = "RESUME_START"

    # Retry events
    RETRY_START = "RETRY_START"
    RETRY_ATTEMPT = "RETRY_ATTEMPT"
    RETRY_END = "RETRY_END"
    RETRY_GIVE_UP = "RETRY_GIVE_UP"
    RETRY_FN_START = "RETRY_FN_START"
    RETRY_FN_RESULT = "RETRY_FN_RESULT"
    RETRY_FN_ERROR = "RETRY_FN_ERROR"

    # Fallback events
    FALLBACK_START = "FALLBACK_START"
    FALLBACK_MODEL_SELECTED = "FALLBACK_MODEL_SELECTED"
    FALLBACK_END = "FALLBACK_END"

    # Structured events
    STRUCTURED_PARSE_START = "STRUCTURED_PARSE_START"
    STRUCTURED_PARSE_END = "STRUCTURED_PARSE_END"
    STRUCTURED_PARSE_ERROR = "STRUCTURED_PARSE_ERROR"
    STRUCTURED_VALIDATION_START = "STRUCTURED_VALIDATION_START"
    STRUCTURED_VALIDATION_END = "STRUCTURED_VALIDATION_END"
    STRUCTURED_VALIDATION_ERROR = "STRUCTURED_VALIDATION_ERROR"
    STRUCTURED_AUTO_CORRECT_START = "STRUCTURED_AUTO_CORRECT_START"
    STRUCTURED_AUTO_CORRECT_END = "STRUCTURED_AUTO_CORRECT_END"

    # Continuation events
    CONTINUATION_START = "CONTINUATION_START"

    # Tool events
    TOOL_REQUESTED = "TOOL_REQUESTED"
    TOOL_START = "TOOL_START"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_ERROR = "TOOL_ERROR"
    TOOL_COMPLETED = "TOOL_COMPLETED"

    # Completion events
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class ObservabilityEventModel(BaseModel):
    """Pydantic model for observability event."""

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for event-specific data
        populate_by_name=True,  # Accept both "ts" and "timestamp"
    )

    type: ObservabilityEventTypeModel
    ts: float = Field(alias="timestamp")
    stream_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] | None = None

    # Common optional fields for specific event types
    attempt: int | None = None
    is_retry: bool | None = None
    is_fallback: bool | None = None
    duration_ms: float | None = None
    success: bool | None = None
    token_count: int | None = None
    error: str | None = None
    reason: str | None = None


class SessionStartEventModel(BaseModel):
    """Pydantic model for session start event."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: ObservabilityEventTypeModel = ObservabilityEventTypeModel.SESSION_START
    ts: float = Field(alias="timestamp")
    stream_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    attempt: int
    is_retry: bool
    is_fallback: bool


class SessionEndEventModel(BaseModel):
    """Pydantic model for session end event."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: ObservabilityEventTypeModel = ObservabilityEventTypeModel.SESSION_END
    ts: float = Field(alias="timestamp")
    stream_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float
    success: bool
    token_count: int


class RetryAttemptEventModel(BaseModel):
    """Pydantic model for retry attempt event."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: ObservabilityEventTypeModel = ObservabilityEventTypeModel.RETRY_ATTEMPT
    ts: float = Field(alias="timestamp")
    stream_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    index: int | None = None
    attempt: int
    max_attempts: int
    reason: str
    delay_ms: float
    counts_toward_limit: bool | None = None
    is_network: bool | None = None
    is_model_issue: bool | None = None


class CompleteEventModel(BaseModel):
    """Pydantic model for complete event."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: ObservabilityEventTypeModel = ObservabilityEventTypeModel.COMPLETE
    ts: float = Field(alias="timestamp")
    stream_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    token_count: int
    content_length: int
    duration_ms: float
    state: dict[str, Any] | None = None


class ErrorEventModel(BaseModel):
    """Pydantic model for error event."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: ObservabilityEventTypeModel = ObservabilityEventTypeModel.ERROR
    ts: float = Field(alias="timestamp")
    stream_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    error: str
    error_code: str | None = None
    failure_type: str
    recovery_strategy: str
    policy: dict[str, Any]
