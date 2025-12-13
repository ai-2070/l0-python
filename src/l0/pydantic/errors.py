"""Pydantic models for L0 error types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorCodeModel(str, Enum):
    """Error codes for programmatic handling."""

    STREAM_ABORTED = "STREAM_ABORTED"
    INITIAL_TOKEN_TIMEOUT = "INITIAL_TOKEN_TIMEOUT"
    INTER_TOKEN_TIMEOUT = "INTER_TOKEN_TIMEOUT"
    ZERO_OUTPUT = "ZERO_OUTPUT"
    GUARDRAIL_VIOLATION = "GUARDRAIL_VIOLATION"
    FATAL_GUARDRAIL_VIOLATION = "FATAL_GUARDRAIL_VIOLATION"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    INVALID_STREAM = "INVALID_STREAM"
    ADAPTER_NOT_FOUND = "ADAPTER_NOT_FOUND"
    FEATURE_NOT_ENABLED = "FEATURE_NOT_ENABLED"
    ALL_STREAMS_EXHAUSTED = "ALL_STREAMS_EXHAUSTED"
    NETWORK_ERROR = "NETWORK_ERROR"


class FailureTypeModel(str, Enum):
    """What actually went wrong - the root cause of the failure."""

    NETWORK = "network"
    MODEL = "model"
    TOOL = "tool"
    TIMEOUT = "timeout"
    ABORT = "abort"
    ZERO_OUTPUT = "zero_output"
    UNKNOWN = "unknown"


class RecoveryStrategyModel(str, Enum):
    """What L0 decided to do next after an error."""

    RETRY = "retry"
    FALLBACK = "fallback"
    CONTINUE = "continue"
    HALT = "halt"


class NetworkErrorTypeModel(str, Enum):
    """Network error types that L0 can detect."""

    CONNECTION_DROPPED = "connection_dropped"
    FETCH_ERROR = "fetch_error"
    ECONNRESET = "econnreset"
    ECONNREFUSED = "econnrefused"
    SSE_ABORTED = "sse_aborted"
    NO_BYTES = "no_bytes"
    PARTIAL_CHUNKS = "partial_chunks"
    RUNTIME_KILLED = "runtime_killed"
    BACKGROUND_THROTTLE = "background_throttle"
    DNS_ERROR = "dns_error"
    SSL_ERROR = "ssl_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RecoveryPolicyModel(BaseModel):
    """Pydantic model for recovery policy."""

    model_config = ConfigDict(extra="forbid")

    retry_enabled: bool = True
    fallback_enabled: bool = False
    max_retries: int = 3
    max_fallbacks: int = 0
    attempt: int = 1
    fallback_index: int | None = None


class ErrorContextModel(BaseModel):
    """Pydantic model for error context."""

    model_config = ConfigDict(extra="forbid")

    code: ErrorCodeModel
    checkpoint: str | None = None
    token_count: int | None = None
    content_length: int | None = None
    model_retry_count: int | None = None
    network_retry_count: int | None = None
    fallback_index: int | None = None
    metadata: dict[str, Any] | None = None
    context: dict[str, Any] | None = None


class NetworkErrorAnalysisModel(BaseModel):
    """Pydantic model for network error analysis."""

    model_config = ConfigDict(extra="forbid")

    type: NetworkErrorTypeModel
    retryable: bool
    counts_toward_limit: bool
    suggestion: str
    context: dict[str, Any] = Field(default_factory=dict)


class SerializedErrorModel(BaseModel):
    """Pydantic model for serialized error (for transport/storage)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    code: str
    category: str
    message: str
    timestamp: float
    has_checkpoint: bool = Field(alias="hasCheckpoint")
    checkpoint: str | None = None
    token_count: int | None = Field(default=None, alias="tokenCount")
    model_retry_count: int | None = Field(default=None, alias="modelRetryCount")
    network_retry_count: int | None = Field(default=None, alias="networkRetryCount")
    fallback_index: int | None = Field(default=None, alias="fallbackIndex")
    metadata: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
