"""L0 types - clean Pythonic naming without module prefixes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

if TYPE_CHECKING:
    from .events import ObservabilityEvent


# ─────────────────────────────────────────────────────────────────────────────
# Event Types
# ─────────────────────────────────────────────────────────────────────────────


class EventType(str, Enum):
    """Type of streaming event."""

    TOKEN = "token"
    MESSAGE = "message"
    DATA = "data"
    PROGRESS = "progress"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class Event:
    """Unified event from adapter-normalized LLM stream."""

    type: EventType
    value: str | None = None
    data: dict[str, Any] | None = None
    error: Exception | None = None
    usage: dict[str, int] | None = None
    timestamp: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Error Categories
# ─────────────────────────────────────────────────────────────────────────────


class ErrorCategory(str, Enum):
    """Category of error for retry decisions."""

    NETWORK = "network"
    TRANSIENT = "transient"
    MODEL = "model"
    CONTENT = "content"
    PROVIDER = "provider"
    FATAL = "fatal"
    INTERNAL = "internal"


class BackoffStrategy(str, Enum):
    """Backoff strategy for retries."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FULL_JITTER = "full-jitter"
    FIXED_JITTER = "fixed-jitter"


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class State:
    """Runtime state tracking."""

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
# Retry + Timeout
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Retry:
    """Retry configuration."""

    attempts: int = 3
    max_retries: int = 6
    base_delay_ms: int = 1000
    max_delay_ms: int = 10000
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER


@dataclass
class Timeout:
    """Timeout configuration."""

    initial_token_ms: int = 5000
    inter_token_ms: int = 10000


# ─────────────────────────────────────────────────────────────────────────────
# Stream (the result type)
# ─────────────────────────────────────────────────────────────────────────────


class Stream:
    """Async iterator result with state and abort attached.

    Usage:
        result = await l0(stream=my_stream)

        # Iterate directly
        async for event in result:
            print(event)

        # Access state
        print(result.state.content)
        print(result.state.token_count)

        # Get full text
        text = await result.text()

        # Abort if needed
        result.abort()
    """

    __slots__ = ("_iterator", "_consumed", "_text", "state", "abort", "errors")

    def __init__(
        self,
        iterator: AsyncIterator[Event],
        state: State,
        abort: Callable[[], None],
        errors: list[Exception] | None = None,
    ) -> None:
        self._iterator = iterator
        self._consumed = False
        self._text: str | None = None
        self.state = state
        self.abort = abort
        self.errors = errors or []

    def __aiter__(self) -> "Stream":
        return self

    async def __anext__(self) -> Event:
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            self._consumed = True
            raise

    async def text(self) -> str:
        """Consume the stream and return the full text content.

        If already consumed, returns the accumulated state.content.
        """
        if self._consumed or self._text is not None:
            return self._text or self.state.content

        # Consume the stream
        async for event in self:
            pass  # Events are processed, state is updated

        self._text = self.state.content
        return self._text

    @property
    def stream(self) -> "Stream":
        """Backwards compatibility: result.stream returns self."""
        return self
