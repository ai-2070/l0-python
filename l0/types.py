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
    """Unified event from adapter-normalized LLM stream.

    Usage:
        async for event in result:
            if event.is_token:
                print(event.text, end="")
            elif event.is_complete:
                print(f"Done! Usage: {event.usage}")
            elif event.is_error:
                print(f"Error: {event.error}")
    """

    type: EventType
    text: str | None = None  # Token content (renamed from 'value')
    data: dict[str, Any] | None = None  # Tool call / misc data
    error: Exception | None = None  # Error (for error events)
    usage: dict[str, int] | None = None  # Token usage
    timestamp: float | None = None  # Event timestamp

    # ─────────────────────────────────────────────────────────────────────────
    # Type check helpers - beautiful Pythonic API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_token(self) -> bool:
        """Check if this is a token event."""
        return self.type is EventType.TOKEN

    @property
    def is_message(self) -> bool:
        """Check if this is a message event."""
        return self.type is EventType.MESSAGE

    @property
    def is_data(self) -> bool:
        """Check if this is a data event."""
        return self.type is EventType.DATA

    @property
    def is_progress(self) -> bool:
        """Check if this is a progress event."""
        return self.type is EventType.PROGRESS

    @property
    def is_tool_call(self) -> bool:
        """Check if this is a tool call event."""
        return self.type is EventType.TOOL_CALL

    @property
    def is_error(self) -> bool:
        """Check if this is an error event."""
        return self.type is EventType.ERROR

    @property
    def is_complete(self) -> bool:
        """Check if this is a complete event."""
        return self.type is EventType.COMPLETE


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
# Retry + Timeout (seconds, not milliseconds - Pythonic!)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Retry:
    """Retry configuration.

    All delays are in seconds (float), matching Python conventions
    like asyncio.sleep(), time.sleep(), etc.
    """

    attempts: int = 3  # Model errors only
    max_retries: int = 6  # Absolute cap (all errors)
    base_delay: float = 1.0  # Starting delay (seconds)
    max_delay: float = 10.0  # Maximum delay (seconds)
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER


@dataclass
class Timeout:
    """Timeout configuration.

    All timeouts are in seconds (float), matching Python conventions
    like asyncio.wait_for(), socket.settimeout(), etc.
    """

    initial_token: float = 5.0  # Seconds to first token
    inter_token: float = 10.0  # Seconds between tokens


# ─────────────────────────────────────────────────────────────────────────────
# Stream (the result type)
# ─────────────────────────────────────────────────────────────────────────────


class Stream:
    """Async iterator result with state and abort attached.

    Supports both iteration and context manager patterns:

        # Pattern 1: Direct iteration
        result = await l0.run(stream=my_stream)
        async for event in result:
            if event.is_token:
                print(event.text, end="")

        # Pattern 2: Context manager (auto-cleanup)
        async with await l0.run(stream=my_stream) as result:
            async for event in result:
                if event.is_token:
                    print(event.text, end="")

        # Get full text
        text = await result.read()

        # Access state
        print(result.state.content)
        print(result.state.token_count)
    """

    __slots__ = ("_iterator", "_consumed", "_content", "state", "abort", "errors")

    def __init__(
        self,
        iterator: AsyncIterator[Event],
        state: State,
        abort: Callable[[], None],
        errors: list[Exception] | None = None,
    ) -> None:
        self._iterator = iterator
        self._consumed = False
        self._content: str | None = None
        self.state = state
        self.abort = abort
        self.errors = errors or []

    # ─────────────────────────────────────────────────────────────────────────
    # Async iterator protocol
    # ─────────────────────────────────────────────────────────────────────────

    def __aiter__(self) -> "Stream":
        return self

    async def __anext__(self) -> Event:
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            self._consumed = True
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Context manager protocol
    # ─────────────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "Stream":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.abort()
        return False  # Don't suppress exceptions

    # ─────────────────────────────────────────────────────────────────────────
    # Read interface
    # ─────────────────────────────────────────────────────────────────────────

    async def read(self) -> str:
        """Consume the stream and return the full text content.

        Pythonic interface matching file.read(), stream.read(), etc.
        If already consumed, returns the accumulated state.content.
        """
        if self._consumed or self._content is not None:
            return self._content or self.state.content

        # Consume the stream
        async for event in self:
            pass  # Events are processed, state is updated

        self._content = self.state.content
        return self._content
