from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

if TYPE_CHECKING:
    from .events import ObservabilityEvent


# ─────────────────────────────────────────────────────────────────────────────
# Event Types (matches TS: token | message | data | progress | error | complete)
# ─────────────────────────────────────────────────────────────────────────────


class EventType(str, Enum):
    TOKEN = "token"
    MESSAGE = "message"
    DATA = "data"
    PROGRESS = "progress"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class L0Event:
    """Unified event from adapter-normalized LLM stream."""

    type: EventType
    value: str | None = None
    data: dict[str, Any] | None = None
    error: Exception | None = None
    usage: dict[str, int] | None = None
    timestamp: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Error Categories (matches TS ErrorCategory enum)
# ─────────────────────────────────────────────────────────────────────────────


class ErrorCategory(str, Enum):
    NETWORK = "network"
    TRANSIENT = "transient"
    MODEL = "model"
    CONTENT = "content"
    PROVIDER = "provider"
    FATAL = "fatal"
    INTERNAL = "internal"


class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FULL_JITTER = "full-jitter"
    FIXED_JITTER = "fixed-jitter"


# ─────────────────────────────────────────────────────────────────────────────
# State Object (matches TS L0State)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class L0State:
    content: str = ""
    checkpoint: str = ""
    token_count: int = 0
    model_retry_count: int = 0
    network_retry_count: int = 0
    fallback_index: int = 0
    violations: list[Any] = field(default_factory=list)
    drift_detected: bool = False
    completed: bool = False
    aborted: bool = False
    first_token_at: float | None = None
    last_token_at: float | None = None
    duration: float | None = None
    resumed: bool = False
    network_errors: list[Any] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Retry + Timeout Configs (matches TS defaults)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetryConfig:
    attempts: int = 3
    max_retries: int = 6
    base_delay_ms: int = 1000
    max_delay_ms: int = 10000
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER


@dataclass
class TimeoutConfig:
    initial_token_ms: int = 5000
    inter_token_ms: int = 10000


# ─────────────────────────────────────────────────────────────────────────────
# Options + Results
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class L0Options:
    stream: Callable[[], AsyncIterator[Any]]
    fallbacks: list[Callable[[], AsyncIterator[Any]]] = field(default_factory=list)
    guardrails: list[Any] = field(default_factory=list)
    retry: RetryConfig | None = None
    timeout: TimeoutConfig | None = None
    adapter: Any | str | None = None
    on_event: Callable[[ObservabilityEvent], None] | None = None
    meta: dict[str, Any] | None = None


@dataclass
class L0Result:
    stream: AsyncIterator[L0Event]
    state: L0State
    abort: Callable[[], None]
    errors: list[Exception] = field(default_factory=list)
