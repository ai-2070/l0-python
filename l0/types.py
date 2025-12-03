"""L0 types - clean Pythonic naming without module prefixes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

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


class ContentType(str, Enum):
    """Type of multimodal content."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    JSON = "json"
    BINARY = "binary"


@dataclass
class DataPayload:
    """Multimodal data payload.

    Carries image, audio, video, file, or structured data from
    multimodal AI outputs.

    Attributes:
        content_type: Type of content (image, audio, video, file, json, binary)
        mime_type: MIME type (e.g., "image/png", "audio/mp3")
        base64: Base64-encoded data
        url: URL to content
        data: Raw bytes
        json: Structured JSON data
        metadata: Additional metadata (dimensions, duration, etc.)
    """

    content_type: ContentType
    mime_type: str | None = None
    base64: str | None = None
    url: str | None = None
    data: bytes | None = None
    json: Any | None = None
    metadata: dict[str, Any] | None = None

    # Convenience properties for common metadata
    @property
    def width(self) -> int | None:
        """Image/video width."""
        return self.metadata.get("width") if self.metadata else None

    @property
    def height(self) -> int | None:
        """Image/video height."""
        return self.metadata.get("height") if self.metadata else None

    @property
    def duration(self) -> float | None:
        """Audio/video duration in seconds."""
        return self.metadata.get("duration") if self.metadata else None

    @property
    def size(self) -> int | None:
        """File size in bytes."""
        return self.metadata.get("size") if self.metadata else None

    @property
    def filename(self) -> str | None:
        """Filename if available."""
        return self.metadata.get("filename") if self.metadata else None

    @property
    def seed(self) -> int | None:
        """Generation seed for reproducibility."""
        return self.metadata.get("seed") if self.metadata else None

    @property
    def model(self) -> str | None:
        """Model used for generation."""
        return self.metadata.get("model") if self.metadata else None


@dataclass
class Progress:
    """Progress update for long-running operations.

    Attributes:
        percent: Progress percentage (0-100)
        step: Current step number
        total_steps: Total number of steps
        message: Status message
        eta: Estimated time remaining in seconds
    """

    percent: float | None = None
    step: int | None = None
    total_steps: int | None = None
    message: str | None = None
    eta: float | None = None


@dataclass
class Event:
    """Unified event from adapter-normalized LLM stream.

    Usage:
        async for event in result:
            if event.is_token:
                print(event.text, end="")
            elif event.is_data:
                if event.payload.content_type == ContentType.IMAGE:
                    save_image(event.payload.base64)
            elif event.is_progress:
                print(f"Progress: {event.progress.percent}%")
            elif event.is_complete:
                print(f"Done! Usage: {event.usage}")
            elif event.is_error:
                print(f"Error: {event.error}")
    """

    type: EventType
    text: str | None = None  # Token content
    data: dict[str, Any] | None = None  # Tool call / misc data
    payload: DataPayload | None = None  # Multimodal data payload
    progress: Progress | None = None  # Progress update
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
    # Multimodal state
    data_outputs: list[DataPayload] = field(default_factory=list)
    last_progress: Progress | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Retry + Timeout (seconds, not milliseconds - Pythonic!)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ErrorTypeDelays:
    """Per-error-type delay configuration.

    All delays are in seconds (float), matching Python conventions.
    """

    connection_dropped: float = 1.0
    fetch_error: float = 0.5
    econnreset: float = 1.0
    econnrefused: float = 2.0
    sse_aborted: float = 0.5
    no_bytes: float = 0.5
    partial_chunks: float = 0.5
    runtime_killed: float = 2.0
    background_throttle: float = 5.0
    dns_error: float = 3.0
    ssl_error: float = 2.0
    timeout: float = 1.0
    unknown: float = 1.0


@dataclass
class Retry:
    """Retry configuration.

    All delays are in seconds (float), matching Python conventions
    like asyncio.sleep(), time.sleep(), etc.

    Usage:
        from l0 import Retry

        # Use presets
        retry = Retry.recommended()
        retry = Retry.mobile()
        retry = Retry.edge()

        # Or customize
        retry = Retry(
            attempts=5,
            base_delay=2.0,
            error_type_delays=ErrorTypeDelays(
                timeout=3.0,
                connection_dropped=2.0,
            ),
        )
    """

    attempts: int = 3  # Model errors only
    max_retries: int = 6  # Absolute cap (all errors)
    base_delay: float = 1.0  # Starting delay (seconds)
    max_delay: float = 10.0  # Maximum delay (seconds)
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER
    error_type_delays: ErrorTypeDelays | None = None  # Per-error-type delays

    @classmethod
    def recommended(cls) -> Retry:
        """Get recommended retry configuration.

        Handles all network errors automatically with sensible defaults:
        - 3 model error retries
        - 6 max total retries
        - Fixed-jitter backoff strategy
        - Per-error-type delays for network errors

        Returns:
            Retry configuration optimized for most use cases.
        """
        return cls(
            attempts=3,
            max_retries=6,
            base_delay=1.0,
            max_delay=10.0,
            strategy=BackoffStrategy.FIXED_JITTER,
            error_type_delays=ErrorTypeDelays(),
        )

    @classmethod
    def mobile(cls) -> Retry:
        """Get retry configuration optimized for mobile environments.

        Higher delays for background throttling and connection issues.
        """
        return cls(
            attempts=3,
            max_retries=6,
            base_delay=1.0,
            max_delay=15.0,
            strategy=BackoffStrategy.FULL_JITTER,
            error_type_delays=ErrorTypeDelays(
                background_throttle=15.0,
                timeout=3.0,
                connection_dropped=2.5,
            ),
        )

    @classmethod
    def edge(cls) -> Retry:
        """Get retry configuration optimized for edge runtimes.

        Shorter delays to stay within edge runtime limits.
        """
        return cls(
            attempts=3,
            max_retries=6,
            base_delay=0.5,
            max_delay=5.0,
            strategy=BackoffStrategy.FIXED_JITTER,
            error_type_delays=ErrorTypeDelays(
                runtime_killed=2.0,
                timeout=1.5,
            ),
        )


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

        # Pattern 1: Direct iteration (l0.wrap - no await needed!)
        result = l0.wrap(stream)
        async for event in result:
            if event.is_token:
                print(event.text, end="")

        # Pattern 2: Context manager (auto-cleanup)
        async with l0.wrap(stream) as result:
            async for event in result:
                if event.is_token:
                    print(event.text, end="")

        # Get full text
        text = await result.read()

        # Access state (after iteration)
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

    def __aiter__(self) -> Stream:
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

    async def __aenter__(self) -> Stream:
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
        async for _ in self:
            pass  # Events are processed, state is updated

        self._content = self.state.content
        return self._content


class LazyStream:
    """Lazy stream wrapper - no await needed on creation.

    Like httpx.AsyncClient() or aiohttp.ClientSession(), this returns
    immediately and only does async work when you iterate or read.

    Usage:
        # Simple - no double await!
        result = l0.wrap(stream)
        text = await result.read()

        # Streaming
        async for event in l0.wrap(stream):
            print(event.text)

        # Context manager
        async with l0.wrap(stream) as result:
            async for event in result:
                print(event.text)
    """

    __slots__ = (
        "_stream",
        "_guardrails",
        "_retry",
        "_timeout",
        "_adapter",
        "_on_event",
        "_meta",
        "_buffer_tool_calls",
        "_runner",
        "_started",
    )

    def __init__(
        self,
        stream: AsyncIterator[Any],
        *,
        guardrails: list[Any] | None = None,
        retry: Retry | None = None,
        timeout: Timeout | None = None,
        adapter: Any | str | None = None,
        on_event: Callable[[ObservabilityEvent], None] | None = None,
        meta: dict[str, Any] | None = None,
        buffer_tool_calls: bool = False,
    ) -> None:
        self._stream = stream
        self._guardrails = guardrails
        self._retry = retry
        self._timeout = timeout
        self._adapter = adapter
        self._on_event = on_event
        self._meta = meta
        self._buffer_tool_calls = buffer_tool_calls
        self._runner: Stream | None = None
        self._started = False

    async def _ensure_started(self) -> Stream:
        """Lazily start the L0 runtime."""
        if self._runner is None:
            # Import here to avoid circular import
            from .runtime import _internal_run

            # Wrap stream in factory
            stream = self._stream

            def stream_factory() -> AsyncIterator[Any]:
                return stream

            self._runner = await _internal_run(
                stream=stream_factory,
                fallbacks=None,
                guardrails=self._guardrails,
                retry=self._retry,
                timeout=self._timeout,
                adapter=self._adapter,
                on_event=self._on_event,
                meta=self._meta,
                buffer_tool_calls=self._buffer_tool_calls,
            )
            self._started = True
        return self._runner

    # ─────────────────────────────────────────────────────────────────────────
    # Proxy properties (delegate to runner once started)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        """Get state (only valid after iteration starts)."""
        if self._runner is None:
            # Return empty state before started
            return State()
        return self._runner.state

    @property
    def errors(self) -> list[Exception]:
        """Get errors list."""
        if self._runner is None:
            return []
        return self._runner.errors

    def abort(self) -> None:
        """Abort the stream."""
        if self._runner is not None:
            self._runner.abort()

    # ─────────────────────────────────────────────────────────────────────────
    # Async iterator protocol
    # ─────────────────────────────────────────────────────────────────────────

    def __aiter__(self) -> LazyStream:
        return self

    async def __anext__(self) -> Event:
        runner = await self._ensure_started()
        return await runner.__anext__()

    # ─────────────────────────────────────────────────────────────────────────
    # Context manager protocol
    # ─────────────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> LazyStream:
        await self._ensure_started()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.abort()
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Read interface
    # ─────────────────────────────────────────────────────────────────────────

    async def read(self) -> str:
        """Consume the stream and return the full text content."""
        runner = await self._ensure_started()
        return await runner.read()
