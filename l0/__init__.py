"""L0 - Reliability layer for AI/LLM streaming."""

from typing import Any, AsyncIterator, Callable

from .adapters import (
    Adapter,
    LiteLLMAdapter,
    OpenAIAdapter,
    detect_adapter,
    register_adapter,
)
from .consensus import consensus
from .events import EventBus, ObservabilityEvent, ObservabilityEventType
from .guardrails import (
    GuardrailRule,
    Guardrails,
    GuardrailViolation,
    check_guardrails,
    json_rule,
    pattern_rule,
    recommended_guardrails,
    repetition_rule,
    stall_rule,
    strict_guardrails,
    strict_json_rule,
    zero_output_rule,
)
from .logging import enable_debug
from .parallel import batched, parallel, race
from .runtime import _internal_run
from .stream import consume_stream, get_text
from .structured import structured
from .types import (
    BackoffStrategy,
    ErrorCategory,
    Event,
    EventType,
    Retry,
    State,
    Stream,
    Timeout,
)
from .version import __version__

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


async def wrap(
    stream: AsyncIterator[Any],
    *,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: Any | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
) -> Stream:
    """Wrap a raw LLM stream with L0 reliability.

    This is the preferred API - takes a raw stream directly, no lambda needed.

    Args:
        stream: Raw async iterator from OpenAI/LiteLLM/etc.
        guardrails: Optional list of guardrail rules to apply
        retry: Optional retry configuration
        timeout: Optional timeout configuration
        adapter: Optional adapter hint ("openai", "litellm", or Adapter instance)
        on_event: Optional callback for observability events
        meta: Optional metadata attached to all events

    Returns:
        Stream - async iterator with .state, .abort(), and .read()

    Example:
        ```python
        import l0
        import litellm

        raw = litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        async with await l0.wrap(raw, guardrails=l0.Guardrails.recommended()) as result:
            async for event in result:
                if event.is_token:
                    print(event.text, end="")

        text = await result.read()
        ```
    """

    # Wrap the stream in a lambda for internal use
    # The stream is already created, so we just return it
    async def stream_factory() -> AsyncIterator[Any]:
        # If it's a coroutine (awaitable), await it first
        if hasattr(stream, "__await__"):
            return await stream  # type: ignore
        return stream

    return await _internal_run(
        stream=stream_factory,
        fallbacks=None,
        guardrails=guardrails,
        retry=retry,
        timeout=timeout,
        adapter=adapter,
        on_event=on_event,
        meta=meta,
    )


async def run(
    stream: Callable[[], AsyncIterator[Any]],
    *,
    fallbacks: list[Callable[[], AsyncIterator[Any]]] | None = None,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: Any | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
) -> Stream:
    """Run L0 with a stream factory (supports retries and fallbacks).

    Use this when you need retry/fallback support, which requires re-creating
    the stream. For simple cases, prefer l0.wrap().

    Args:
        stream: Factory function that returns an async LLM stream
        fallbacks: Optional list of fallback stream factories
        guardrails: Optional list of guardrail rules to apply
        retry: Optional retry configuration
        timeout: Optional timeout configuration
        adapter: Optional adapter hint ("openai", "litellm", or Adapter instance)
        on_event: Optional callback for observability events
        meta: Optional metadata attached to all events

    Returns:
        Stream - async iterator with .state, .abort(), and .read()

    Example:
        ```python
        import l0
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            ),
            fallbacks=[
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                ),
            ],
            guardrails=l0.Guardrails.recommended(),
            retry=l0.Retry(max_attempts=3),
        )

        async for event in result:
            if event.is_token:
                print(event.text, end="")
        ```
    """
    return await _internal_run(
        stream=stream,
        fallbacks=fallbacks,
        guardrails=guardrails,
        retry=retry,
        timeout=timeout,
        adapter=adapter,
        on_event=on_event,
        meta=meta,
    )


# Legacy alias
l0 = run


__all__ = [
    # Version
    "__version__",
    # Core API
    "wrap",  # Preferred - takes raw stream
    "run",  # With factory - supports retries/fallbacks
    "l0",  # Alias to run
    # Types
    "Stream",
    "Event",
    "State",
    "EventType",
    # Config
    "Retry",
    "Timeout",
    "BackoffStrategy",
    "ErrorCategory",
    # Events
    "ObservabilityEvent",
    "ObservabilityEventType",
    "EventBus",
    # Stream utilities
    "consume_stream",
    "get_text",
    # Adapters
    "Adapter",
    "OpenAIAdapter",
    "LiteLLMAdapter",
    "register_adapter",
    "detect_adapter",
    # Guardrails
    "Guardrails",  # Class with .recommended(), .strict()
    "GuardrailRule",
    "GuardrailViolation",
    "check_guardrails",
    "json_rule",
    "strict_json_rule",
    "pattern_rule",
    "zero_output_rule",
    "stall_rule",
    "repetition_rule",
    "recommended_guardrails",  # Legacy
    "strict_guardrails",  # Legacy
    # Structured
    "structured",
    # Parallel
    "parallel",
    "race",
    "batched",
    # Consensus
    "consensus",
    # Debug
    "enable_debug",
]
