"""L0 stream adapters.

Adapters convert provider-specific streams into L0's unified event format.
L0 handles all reliability concerns (retries, timeouts, guardrails),
so adapters can focus purely on format conversion.

Official first-party adapters:
- OpenAI - Direct OpenAI SDK streams
- LiteLLM - Unified interface for 100+ providers (uses OpenAI format)
- Anthropic - Anthropic SDK streams (reference implementation)

Usage:
    ```python
    from l0 import Adapters

    # Explicit adapter (recommended)
    result = await l0.run(stream, adapter=Adapters.openai())

    # Adapter by name
    result = await l0.run(stream, adapter="openai")

    # Auto-detection
    result = await l0.run(stream)  # L0 detects the adapter
    ```
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from .logging import logger
from .types import ContentType, DataPayload, Event, EventType, Progress

# TypeVar for adapter chunk types
AdapterChunkT = TypeVar("AdapterChunkT", covariant=True)
AdapterOptionsT = TypeVar("AdapterOptionsT", contravariant=True)


@dataclass
class AdaptedEvent(Generic[AdapterChunkT]):
    """Event with associated raw chunk from the provider.

    Wraps an L0 Event with the original raw chunk for provider-specific access.
    """

    event: Event
    raw_chunk: AdapterChunkT | None = None


@runtime_checkable
class Adapter(Protocol[AdapterChunkT, AdapterOptionsT]):
    """Protocol for stream adapters (L0Adapter interface).

    Adapters convert raw LLM provider streams into L0 Events.
    L0 handles all reliability concerns, so adapters focus on format conversion.

    Attributes:
        name: Unique identifier for this adapter

    Methods:
        detect: Optional type guard for auto-detection (only needed for registry)
        wrap: Convert provider stream to L0 Events
    """

    name: str

    def detect(self, stream: Any) -> bool:
        """Check if this adapter can handle the given stream.

        Optional - only required for registerAdapter() auto-detection.
        Not needed for explicit `adapter=myAdapter` usage.

        Args:
            stream: The stream to check

        Returns:
            True if this adapter can handle the stream
        """
        ...

    def wrap(
        self,
        stream: AsyncIterator[Any],
        options: AdapterOptionsT | None = None,
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        """Wrap raw stream into AdaptedEvent stream.

        Yields AdaptedEvent objects containing both the normalized Event
        and the original raw chunk for provider-specific access.

        Args:
            stream: The raw provider stream
            options: Optional adapter-specific options

        Yields:
            AdaptedEvent objects with normalized events and raw chunks
        """
        ...


@dataclass
class OpenAIAdapterOptions:
    """Options for OpenAI adapter.

    Attributes:
        include_usage: Include usage in complete event (default: True)
        include_tool_calls: Include tool calls as events (default: True)
        emit_function_calls_as_tokens: Emit function args as tokens (default: False)
        choice_index: Which choice to use when n > 1 (default: 0, or "all")
    """

    include_usage: bool = True
    include_tool_calls: bool = True
    emit_function_calls_as_tokens: bool = False
    choice_index: int | str = 0  # 0 or "all"


class OpenAIAdapter:
    """Adapter for OpenAI SDK streams. Also works with LiteLLM.

    Handles ChatCompletionChunk objects from OpenAI and LiteLLM,
    preserving raw chunks for provider-specific access.
    """

    name = "openai"

    def detect(self, stream: Any) -> bool:
        """Detect OpenAI or LiteLLM streams."""
        import inspect

        type_name = type(stream).__module__

        # Check direct module match
        if "openai" in type_name or "litellm" in type_name:
            return True

        # Check for coroutines from openai/litellm (unawaited streams)
        if inspect.iscoroutine(stream):
            # Check the coroutine's code object for openai/litellm origin
            coro_module = getattr(stream, "cr_code", None)
            if coro_module:
                code_file = getattr(coro_module, "co_filename", "")
                if "openai" in code_file or "litellm" in code_file:
                    return True

        return False

    async def wrap(
        self,
        stream: Any,
        options: OpenAIAdapterOptions | None = None,
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        """Wrap OpenAI/LiteLLM stream into AdaptedEvents."""
        import inspect

        opts = options or OpenAIAdapterOptions()
        usage = None

        # Handle coroutines (unawaited stream factories)
        if inspect.iscoroutine(stream):
            stream = await stream

        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                # Handle choice_index option
                if opts.choice_index == "all":
                    choices = chunk.choices
                else:
                    idx = opts.choice_index if isinstance(opts.choice_index, int) else 0
                    choices = [chunk.choices[idx]] if len(chunk.choices) > idx else []

                for choice in choices:
                    delta = getattr(choice, "delta", None)

                    if delta:
                        # Text content
                        if hasattr(delta, "content") and delta.content:
                            yield AdaptedEvent(
                                event=Event(type=EventType.TOKEN, text=delta.content),
                                raw_chunk=chunk,
                            )

                        # Tool calls
                        if (
                            opts.include_tool_calls
                            and hasattr(delta, "tool_calls")
                            and delta.tool_calls
                        ):
                            for tc in delta.tool_calls:
                                tool_data = {
                                    "index": getattr(tc, "index", None),
                                    "id": getattr(tc, "id", None),
                                    "name": (
                                        getattr(tc.function, "name", None)
                                        if hasattr(tc, "function")
                                        else None
                                    ),
                                    "arguments": (
                                        getattr(tc.function, "arguments", None)
                                        if hasattr(tc, "function")
                                        else None
                                    ),
                                }

                                # Optionally emit function args as tokens
                                if (
                                    opts.emit_function_calls_as_tokens
                                    and tool_data["arguments"]
                                ):
                                    yield AdaptedEvent(
                                        event=Event(
                                            type=EventType.TOKEN,
                                            text=tool_data["arguments"],
                                        ),
                                        raw_chunk=chunk,
                                    )

                                yield AdaptedEvent(
                                    event=Event(
                                        type=EventType.TOOL_CALL,
                                        data=tool_data,
                                    ),
                                    raw_chunk=chunk,
                                )

            # Extract usage if present (typically on last chunk)
            if opts.include_usage and hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "input_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(chunk.usage, "completion_tokens", 0),
                }

        yield AdaptedEvent(
            event=Event(type=EventType.COMPLETE, usage=usage),
            raw_chunk=None,
        )


# Alias for clarity
LiteLLMAdapter = OpenAIAdapter  # LiteLLM uses OpenAI-compatible format


class EventPassthroughAdapter:
    """Adapter for raw Event async iterators.

    This adapter handles async iterators that yield Event objects directly,
    wrapping them in AdaptedEvent for consistency with the runtime.
    """

    name = "event"

    def detect(self, stream: Any) -> bool:
        """Detect async iterators (fallback adapter)."""
        # This is a fallback - detect any async iterator
        return hasattr(stream, "__aiter__")

    async def wrap(
        self,
        stream: Any,
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        """Wrap raw Event stream into AdaptedEvents."""
        async for event in stream:
            if isinstance(event, Event):
                yield AdaptedEvent(event=event, raw_chunk=None)
            else:
                # If it's not an Event, skip it
                pass


# Registry - OpenAI first (more specific), passthrough last (fallback)
_adapters: list[Adapter] = [
    OpenAIAdapter(),
    EventPassthroughAdapter(),
]


# ─────────────────────────────────────────────────────────────────────────────
# Adapters Class - Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class Adapters:
    """Scoped API for stream adapter operations.

    Usage:
        from l0 import Adapters

        # Detect adapter for a stream
        adapter = Adapters.detect(stream)

        # Detect with hint
        adapter = Adapters.detect(stream, hint="openai")

        # Register a custom adapter
        Adapters.register(my_adapter)

        # List registered adapters
        names = Adapters.registered()

        # Unregister an adapter
        Adapters.unregister("my_adapter")

        # Clear all adapters (for testing)
        Adapters.clear()

        # Get built-in adapters
        adapter = Adapters.openai()
        adapter = Adapters.litellm()  # Alias for openai
    """

    @staticmethod
    def detect(stream: Any, hint: Adapter | str | None = None) -> Adapter:
        """Detect or lookup adapter for stream.

        Args:
            stream: The stream to detect adapter for
            hint: Optional adapter instance or name hint

        Returns:
            Adapter instance that can handle the stream

        Raises:
            ValueError: If no adapter found or unknown hint
        """
        if hint is not None and not isinstance(hint, str):
            return hint

        if isinstance(hint, str):
            # Handle litellm hint -> use OpenAI adapter
            if hint == "litellm":
                hint = "openai"
            for a in _adapters:
                if a.name == hint:
                    return a
            raise ValueError(f"Unknown adapter: {hint}")

        for a in _adapters:
            if a.detect(stream):
                logger.debug(f"Detected adapter: {a.name}")
                return a

        raise ValueError("No adapter found for stream")

    @staticmethod
    def register(adapter: Adapter, *, silent: bool = False) -> None:
        """Register a custom adapter (takes priority).

        Args:
            adapter: Adapter instance to register
            silent: If True, suppress warning when adapter lacks detect() method

        Note:
            In development mode, registering an adapter without detect() logs a warning.
            Use silent=True to suppress this warning, or set NODE_ENV=production.
        """
        # Check for detect method and warn if missing (unless silent or production)
        if not silent and os.environ.get("NODE_ENV") != "production":
            if not hasattr(adapter, "detect") or adapter.detect is None:
                warnings.warn(
                    f'Adapter "{adapter.name}" has no detect() method. '
                    "It will not be used for auto-detection. "
                    "Use explicit `adapter=myAdapter` instead, or add a detect() method.",
                    UserWarning,
                    stacklevel=2,
                )
        _adapters.insert(0, adapter)

    @staticmethod
    def unregister(name: str) -> bool:
        """Unregister an adapter by name.

        Args:
            name: Name of the adapter to remove

        Returns:
            True if adapter was found and removed, False otherwise
        """
        global _adapters
        for i, adapter in enumerate(_adapters):
            if adapter.name == name:
                _adapters.pop(i)
                return True
        return False

    @staticmethod
    def clear() -> None:
        """Clear all registered adapters.

        Useful for testing. After clearing, you may want to
        re-register the default OpenAI adapter.
        """
        global _adapters
        _adapters.clear()

    @staticmethod
    def reset() -> None:
        """Reset to default adapters (OpenAI and Event passthrough).

        Useful for testing cleanup.
        """
        global _adapters
        _adapters.clear()
        _adapters.append(OpenAIAdapter())
        _adapters.append(EventPassthroughAdapter())

    @staticmethod
    def openai() -> OpenAIAdapter:
        """Get the OpenAI adapter instance."""
        return OpenAIAdapter()

    @staticmethod
    def litellm() -> OpenAIAdapter:
        """Get the LiteLLM adapter instance (alias for OpenAI)."""
        return OpenAIAdapter()

    @staticmethod
    def get(name: str) -> Adapter | None:
        """Get adapter by name.

        Args:
            name: Name of the adapter

        Returns:
            Adapter instance if found, None otherwise
        """
        for a in _adapters:
            if a.name == name:
                return a
        return None

    @staticmethod
    def unregister_all_except(names: list[str] | None = None) -> list[str]:
        """Remove all adapters except those in the specified list.

        Useful for testing to ensure a clean adapter state.

        Args:
            names: List of adapter names to keep (default: keep none)

        Returns:
            List of removed adapter names
        """
        global _adapters
        names = names or []
        removed = []
        new_adapters = []
        for a in _adapters:
            if a.name in names:
                new_adapters.append(a)
            else:
                removed.append(a.name)
        _adapters = new_adapters
        return removed

    @staticmethod
    def has_matching(stream: Any) -> bool:
        """Check if exactly one adapter matches the stream.

        Args:
            stream: The stream to check

        Returns:
            True if exactly one adapter matches
        """
        matches = [a for a in _adapters if a.detect(stream)]
        return len(matches) == 1

    @staticmethod
    def detect_adapter(stream: Any) -> Adapter | None:
        """Auto-detect adapter for stream.

        Unlike detect(), this returns None instead of raising if no adapter found.

        Args:
            stream: The stream to detect adapter for

        Returns:
            Adapter instance if found, None otherwise
        """
        for a in _adapters:
            if a.detect(stream):
                return a
        return None

    @staticmethod
    def registered() -> list[str]:
        """List names of all registered adapters.

        Matches TS getRegisteredStreamAdapters().

        Returns:
            List of adapter names in priority order
        """
        return [a.name for a in _adapters]

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Functions (scoped under Adapters)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def to_l0_events(
        stream: AsyncIterator[Any],
        extract_text: Callable[[Any], str | None],
    ) -> AsyncIterator[Event]:
        """Convert any async iterable stream to L0 Events.

        This helper makes it easier to build custom adapters by handling:
        - Error conversion to L0 error events
        - Automatic complete event emission

        Args:
            stream: The source async iterable stream
            extract_text: Function to extract text from a chunk (return None to skip)

        Yields:
            L0 Event objects

        Example:
            ```python
            from l0 import Adapters

            async def my_adapter(stream):
                async for event in Adapters.to_l0_events(stream, lambda chunk: chunk.text):
                    yield event
            ```
        """
        async for event in to_l0_events(stream, extract_text):
            yield event

    @staticmethod
    async def to_l0_events_with_messages(
        stream: AsyncIterator[Any],
        extract_text: Callable[[Any], str | None],
        extract_message: Callable[[Any], dict[str, Any] | None] | None = None,
    ) -> AsyncIterator[Event]:
        """Convert a stream with message events to L0 Events.

        Use this when your stream emits both text tokens and structured messages
        (e.g., tool calls, function calls).

        Args:
            stream: The source async iterable stream
            extract_text: Function to extract text from a chunk
            extract_message: Function to extract message from a chunk

        Yields:
            L0 Event objects
        """
        async for event in to_l0_events_with_messages(
            stream, extract_text, extract_message
        ):
            yield event

    @staticmethod
    async def to_multimodal_l0_events(
        stream: AsyncIterator[Any],
        extract_text: Callable[[Any], str | None] | None = None,
        extract_data: Callable[[Any], DataPayload | None] | None = None,
        extract_progress: Callable[[Any], Progress | None] | None = None,
        extract_message: Callable[[Any], dict[str, Any] | None] | None = None,
    ) -> AsyncIterator[Event]:
        """Convert multimodal stream to L0 Events with support for text and data.

        Args:
            stream: The source async iterable stream
            extract_text: Function to extract text from a chunk
            extract_data: Function to extract multimodal data from a chunk
            extract_progress: Function to extract progress from a chunk
            extract_message: Function to extract message from a chunk

        Yields:
            L0 Event objects
        """
        async for event in to_multimodal_l0_events(
            stream, extract_text, extract_data, extract_progress, extract_message
        ):
            yield event

    @staticmethod
    def token_event(value: str) -> Event:
        """Create an L0 token event.

        Args:
            value: The token text

        Returns:
            Event of type TOKEN
        """
        return create_token_event(value)

    @staticmethod
    def complete_event(usage: dict[str, int] | None = None) -> Event:
        """Create an L0 complete event.

        Args:
            usage: Optional usage information

        Returns:
            Event of type COMPLETE
        """
        return create_complete_event(usage)

    @staticmethod
    def error_event(error: Exception | str) -> Event:
        """Create an L0 error event.

        Args:
            error: The error (will be wrapped if string)

        Returns:
            Event of type ERROR
        """
        return create_error_event(error)

    @staticmethod
    def data_event(payload: DataPayload) -> Event:
        """Create an L0 data event for multimodal content.

        Args:
            payload: The data payload

        Returns:
            Event of type DATA
        """
        return create_data_event(payload)

    @staticmethod
    def progress_event(progress: Progress) -> Event:
        """Create an L0 progress event.

        Args:
            progress: Progress information

        Returns:
            Event of type PROGRESS
        """
        return create_progress_event(progress)

    @staticmethod
    def image_event(
        url: str | None = None,
        base64: str | None = None,
        mime_type: str = "image/png",
        width: int | None = None,
        height: int | None = None,
        **metadata: Any,
    ) -> Event:
        """Create an image data event with convenience parameters.

        Args:
            url: Image URL
            base64: Base64-encoded image data
            mime_type: MIME type (default: image/png)
            width: Image width
            height: Image height
            **metadata: Additional metadata

        Returns:
            Event of type DATA with image content
        """
        return create_image_event(url, base64, mime_type, width, height, **metadata)

    @staticmethod
    def audio_event(
        url: str | None = None,
        base64: str | None = None,
        mime_type: str = "audio/mp3",
        duration: float | None = None,
        **metadata: Any,
    ) -> Event:
        """Create an audio data event with convenience parameters.

        Args:
            url: Audio URL
            base64: Base64-encoded audio data
            mime_type: MIME type (default: audio/mp3)
            duration: Audio duration in seconds
            **metadata: Additional metadata

        Returns:
            Event of type DATA with audio content
        """
        return create_audio_event(url, base64, mime_type, duration, **metadata)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions (kept for internal use)
# ─────────────────────────────────────────────────────────────────────────────


def register_adapter(adapter: Adapter, *, silent: bool = False) -> None:
    """Register a custom adapter (takes priority).

    Args:
        adapter: Adapter instance to register
        silent: If True, suppress warning when adapter lacks detect() method
    """
    Adapters.register(adapter, silent=silent)


def unregister_adapter(name: str) -> bool:
    """Unregister an adapter by name.

    Args:
        name: Name of the adapter to remove

    Returns:
        True if adapter was found and removed, False otherwise
    """
    return Adapters.unregister(name)


def unregister_all_except(names: list[str] | None = None) -> list[str]:
    """Remove all adapters except those in the specified list.

    Args:
        names: List of adapter names to keep (default: keep none)

    Returns:
        List of removed adapter names
    """
    return Adapters.unregister_all_except(names)


def get_adapter(name: str) -> Adapter | None:
    """Get adapter by name.

    Args:
        name: Name of the adapter

    Returns:
        Adapter instance if found, None otherwise
    """
    return Adapters.get(name)


def get_registered_adapters() -> list[str]:
    """List names of all registered adapters.

    Returns:
        List of adapter names in priority order
    """
    return Adapters.registered()


def clear_adapters() -> None:
    """Clear all registered adapters."""
    Adapters.clear()


def detect_adapter(stream: Any) -> Adapter | None:
    """Auto-detect adapter for stream.

    Args:
        stream: The stream to detect adapter for

    Returns:
        Adapter instance if found, None otherwise
    """
    return Adapters.detect_adapter(stream)


def has_matching_adapter(stream: Any) -> bool:
    """Check if exactly one adapter matches the stream.

    Args:
        stream: The stream to check

    Returns:
        True if exactly one adapter matches
    """
    return Adapters.has_matching(stream)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")


async def to_l0_events(
    stream: AsyncIterator[T],
    extract_text: Callable[[T], str | None],
) -> AsyncIterator[Event]:
    """Convert any async iterable stream to L0 Events.

    This helper makes it easier to build custom adapters by handling:
    - Error conversion to L0 error events
    - Automatic complete event emission

    You only need to provide an extraction function that pulls text from chunks.

    Args:
        stream: The source async iterable stream
        extract_text: Function to extract text from a chunk (return None to skip)

    Yields:
        L0 Event objects

    Example:
        ```python
        from l0.adapters import to_l0_events

        async def my_adapter(stream):
            async for event in to_l0_events(stream, lambda chunk: chunk.text):
                yield event
        ```
    """
    try:
        async for chunk in stream:
            text = extract_text(chunk)
            if text is not None:
                yield Event(type=EventType.TOKEN, text=text)
        yield Event(type=EventType.COMPLETE)
    except Exception as e:
        yield Event(type=EventType.ERROR, error=e)


async def to_l0_events_with_messages(
    stream: AsyncIterator[T],
    extract_text: Callable[[T], str | None],
    extract_message: Callable[[T], dict[str, Any] | None] | None = None,
) -> AsyncIterator[Event]:
    """Convert a stream with message events to L0 Events.

    Use this when your stream emits both text tokens and structured messages
    (e.g., tool calls, function calls).

    Args:
        stream: The source async iterable stream
        extract_text: Function to extract text from a chunk (return None to skip)
        extract_message: Function to extract message from a chunk (return None to skip)

    Yields:
        L0 Event objects

    Example:
        ```python
        from l0.adapters import to_l0_events_with_messages

        async def tool_adapter(stream):
            async for event in to_l0_events_with_messages(
                stream,
                extract_text=lambda c: c.text if c.type == "text" else None,
                extract_message=lambda c: {"value": c.tool_call} if c.type == "tool" else None,
            ):
                yield event
        ```
    """
    try:
        async for chunk in stream:
            # Check for text content
            text = extract_text(chunk)
            if text is not None:
                yield Event(type=EventType.TOKEN, text=text)
                continue

            # Check for message content
            if extract_message:
                message = extract_message(chunk)
                if message is not None:
                    yield Event(
                        type=EventType.MESSAGE,
                        data=message,
                    )
        yield Event(type=EventType.COMPLETE)
    except Exception as e:
        yield Event(type=EventType.ERROR, error=e)


async def to_multimodal_l0_events(
    stream: AsyncIterator[T],
    extract_text: Callable[[T], str | None] | None = None,
    extract_data: Callable[[T], DataPayload | None] | None = None,
    extract_progress: Callable[[T], Progress | None] | None = None,
    extract_message: Callable[[T], dict[str, Any] | None] | None = None,
) -> AsyncIterator[Event]:
    """Convert multimodal stream to L0 Events with support for text and data.

    Args:
        stream: The source async iterable stream
        extract_text: Function to extract text from a chunk
        extract_data: Function to extract multimodal data from a chunk
        extract_progress: Function to extract progress from a chunk
        extract_message: Function to extract message from a chunk

    Yields:
        L0 Event objects

    Example:
        ```python
        from l0.adapters import to_multimodal_l0_events
        from l0 import DataPayload

        async def image_adapter(stream):
            async for event in to_multimodal_l0_events(
                stream,
                extract_data=lambda c: DataPayload(
                    content_type="image",
                    mime_type="image/png",
                    base64=c.image,
                ) if c.type == "image" else None,
                extract_progress=lambda c: Progress(
                    percent=c.percent,
                    message=c.status,
                ) if c.type == "progress" else None,
            ):
                yield event
        ```
    """
    try:
        async for chunk in stream:
            # Try each extractor in order

            # Text tokens
            if extract_text:
                text = extract_text(chunk)
                if text is not None:
                    yield Event(type=EventType.TOKEN, text=text)
                    continue

            # Multimodal data
            if extract_data:
                payload = extract_data(chunk)
                if payload is not None:
                    yield Event(type=EventType.DATA, payload=payload)
                    continue

            # Progress updates
            if extract_progress:
                progress = extract_progress(chunk)
                if progress is not None:
                    yield Event(type=EventType.PROGRESS, progress=progress)
                    continue

            # Messages
            if extract_message:
                message = extract_message(chunk)
                if message is not None:
                    yield Event(type=EventType.MESSAGE, data=message)
                    continue

        yield Event(type=EventType.COMPLETE)
    except Exception as e:
        yield Event(type=EventType.ERROR, error=e)


# ─────────────────────────────────────────────────────────────────────────────
# Event Creation Helpers
# ─────────────────────────────────────────────────────────────────────────────


def create_token_event(value: str) -> Event:
    """Create an L0 token event.

    Args:
        value: The token text

    Returns:
        Event of type TOKEN
    """
    return Event(type=EventType.TOKEN, text=value)


def create_complete_event(usage: dict[str, int] | None = None) -> Event:
    """Create an L0 complete event.

    Args:
        usage: Optional usage information

    Returns:
        Event of type COMPLETE
    """
    return Event(type=EventType.COMPLETE, usage=usage)


def create_error_event(error: Exception | str) -> Event:
    """Create an L0 error event.

    Args:
        error: The error (will be wrapped if string)

    Returns:
        Event of type ERROR
    """
    if isinstance(error, str):
        error = Exception(error)
    return Event(type=EventType.ERROR, error=error)


def create_data_event(payload: DataPayload) -> Event:
    """Create an L0 data event for multimodal content.

    Args:
        payload: The data payload

    Returns:
        Event of type DATA
    """
    return Event(type=EventType.DATA, payload=payload)


def create_progress_event(progress: Progress) -> Event:
    """Create an L0 progress event.

    Args:
        progress: Progress information

    Returns:
        Event of type PROGRESS
    """
    return Event(type=EventType.PROGRESS, progress=progress)


def create_image_event(
    url: str | None = None,
    base64: str | None = None,
    mime_type: str = "image/png",
    width: int | None = None,
    height: int | None = None,
    **metadata: Any,
) -> Event:
    """Create an image data event with convenience parameters.

    Args:
        url: Image URL
        base64: Base64-encoded image data
        mime_type: MIME type (default: image/png)
        width: Image width
        height: Image height
        **metadata: Additional metadata

    Returns:
        Event of type DATA with image content
    """
    meta = {k: v for k, v in metadata.items() if v is not None}
    if width is not None:
        meta["width"] = width
    if height is not None:
        meta["height"] = height

    payload = DataPayload(
        content_type=ContentType.IMAGE,
        mime_type=mime_type,
        url=url,
        base64=base64,
        metadata=meta if meta else None,
    )
    return Event(type=EventType.DATA, payload=payload)


def create_audio_event(
    url: str | None = None,
    base64: str | None = None,
    mime_type: str = "audio/mp3",
    duration: float | None = None,
    **metadata: Any,
) -> Event:
    """Create an audio data event with convenience parameters.

    Args:
        url: Audio URL
        base64: Base64-encoded audio data
        mime_type: MIME type (default: audio/mp3)
        duration: Audio duration in seconds
        **metadata: Additional metadata

    Returns:
        Event of type DATA with audio content
    """
    meta = {k: v for k, v in metadata.items() if v is not None}
    if duration is not None:
        meta["duration"] = duration

    payload = DataPayload(
        content_type=ContentType.AUDIO,
        mime_type=mime_type,
        url=url,
        base64=base64,
        metadata=meta if meta else None,
    )
    return Event(type=EventType.DATA, payload=payload)
