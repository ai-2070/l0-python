"""L0 stream adapters.

Two adapters only:
- OpenAI - Direct OpenAI SDK streams
- LiteLLM - Unified interface for 100+ providers (Anthropic, Cohere, Bedrock, Vertex, etc.)

LiteLLM uses OpenAI-compatible format, so the OpenAI adapter handles both.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from .logging import logger
from .types import EventType, L0Event


@runtime_checkable
class Adapter(Protocol):
    """Protocol for stream adapters."""

    name: str

    def detect(self, stream: Any) -> bool:
        """Check if this adapter can handle the given stream."""
        ...

    def wrap(self, stream: Any) -> AsyncIterator[L0Event]:
        """Wrap raw stream into L0Event stream."""
        ...


class OpenAIAdapter:
    """Adapter for OpenAI SDK streams. Also works with LiteLLM (OpenAI-compatible format)."""

    name = "openai"

    def detect(self, stream: Any) -> bool:
        """Detect OpenAI or LiteLLM streams."""
        type_name = type(stream).__module__
        return "openai" in type_name or "litellm" in type_name

    async def wrap(self, stream: Any) -> AsyncIterator[L0Event]:
        """Wrap OpenAI/LiteLLM stream into L0Events."""
        usage = None
        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)

                if delta:
                    # Text content
                    if hasattr(delta, "content") and delta.content:
                        yield L0Event(type=EventType.TOKEN, value=delta.content)

                    # Tool calls
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            yield L0Event(
                                type=EventType.TOOL_CALL,
                                data={
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
                            )

            # Extract usage if present (typically on last chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "input_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(chunk.usage, "completion_tokens", 0),
                }

        yield L0Event(type=EventType.COMPLETE, usage=usage)


# Alias for clarity
LiteLLMAdapter = OpenAIAdapter  # LiteLLM uses OpenAI-compatible format


# Registry
_adapters: list[Adapter] = [OpenAIAdapter()]


def register_adapter(adapter: Adapter) -> None:
    """Register a custom adapter (takes priority)."""
    _adapters.insert(0, adapter)


def detect_adapter(stream: Any, hint: Adapter | str | None = None) -> Adapter:
    """Detect or lookup adapter for stream."""
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
