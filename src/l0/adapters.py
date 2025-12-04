"""L0 stream adapters.

Two adapters only:
- OpenAI - Direct OpenAI SDK streams
- LiteLLM - Unified interface for 100+ providers (Anthropic, Cohere, etc.)

LiteLLM uses OpenAI-compatible format, so the OpenAI adapter handles both.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from .logging import logger
from .types import Event, EventType

# TypeVar for adapter chunk types
AdapterChunkT = TypeVar("AdapterChunkT")


@dataclass
class AdaptedEvent(Generic[AdapterChunkT]):
    """Event with associated raw chunk from the provider.

    Wraps an L0 Event with the original raw chunk for provider-specific access.
    """

    event: Event
    raw_chunk: AdapterChunkT | None = None


@runtime_checkable
class Adapter(Protocol):
    """Protocol for stream adapters.

    Adapters convert raw LLM provider streams into AdaptedEvent streams,
    preserving raw chunks for provider-specific access.
    """

    name: str

    def detect(self, stream: Any) -> bool:
        """Check if this adapter can handle the given stream."""
        ...

    def wrap(self, stream: AsyncIterator[Any]) -> AsyncIterator[AdaptedEvent[Any]]:
        """Wrap raw stream into AdaptedEvent stream.

        Yields AdaptedEvent objects containing both the normalized Event
        and the original raw chunk for provider-specific access.
        """
        ...


class OpenAIAdapter:
    """Adapter for OpenAI SDK streams. Also works with LiteLLM.

    Handles ChatCompletionChunk objects from OpenAI and LiteLLM,
    preserving raw chunks for provider-specific access.
    """

    name = "openai"

    def detect(self, stream: Any) -> bool:
        """Detect OpenAI or LiteLLM streams."""
        type_name = type(stream).__module__
        return "openai" in type_name or "litellm" in type_name

    async def wrap(self, stream: Any) -> AsyncIterator[AdaptedEvent[Any]]:
        """Wrap OpenAI/LiteLLM stream into AdaptedEvents."""
        usage = None
        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)

                if delta:
                    # Text content
                    if hasattr(delta, "content") and delta.content:
                        yield AdaptedEvent(
                            event=Event(type=EventType.TOKEN, text=delta.content),
                            raw_chunk=chunk,
                        )

                    # Tool calls
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            yield AdaptedEvent(
                                event=Event(
                                    type=EventType.TOOL_CALL,
                                    data={
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
                                    },
                                ),
                                raw_chunk=chunk,
                            )

            # Extract usage if present (typically on last chunk)
            if hasattr(chunk, "usage") and chunk.usage:
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


# Registry
_adapters: list[Adapter] = [OpenAIAdapter()]


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
        names = Adapters.list()

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
    def register(adapter: Adapter) -> None:
        """Register a custom adapter (takes priority).

        Args:
            adapter: Adapter instance to register
        """
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
    def list() -> list[str]:
        """List names of all registered adapters.

        Returns:
            List of adapter names in priority order
        """
        return [a.name for a in _adapters]

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
        """Reset to default adapters (OpenAI only).

        Useful for testing cleanup.
        """
        global _adapters
        _adapters.clear()
        _adapters.append(OpenAIAdapter())

    @staticmethod
    def openai() -> OpenAIAdapter:
        """Get the OpenAI adapter instance."""
        return OpenAIAdapter()

    @staticmethod
    def litellm() -> OpenAIAdapter:
        """Get the LiteLLM adapter instance (alias for OpenAI)."""
        return OpenAIAdapter()
