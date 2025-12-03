"""L0 - Reliability layer for AI/LLM streaming."""

from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

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
from .runtime import _internal_run, _internal_run_with_options
from .stream import consume_stream, get_text
from .structured import structured
from .types import (
    BackoffStrategy,
    ErrorCategory,
    EventType,
    L0Event,
    L0Options,
    L0Result,
    L0State,
    L0Stream,
    RetryConfig,
    TimeoutConfig,
)
from .version import __version__

# ─────────────────────────────────────────────────────────────────────────────
# Public API: run() is the main entrypoint, l0() is an alias
# ─────────────────────────────────────────────────────────────────────────────


async def run(
    stream: Callable[[], AsyncIterator[Any]],
    *,
    fallbacks: list[Callable[[], AsyncIterator[Any]]] | None = None,
    guardrails: list[GuardrailRule] | None = None,
    retry: RetryConfig | None = None,
    timeout: TimeoutConfig | None = None,
    adapter: Any | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
) -> L0Stream:
    """Main L0 streaming runtime with guardrails and retry logic.

    This is the primary entrypoint for the L0 library. It wraps LLM streams
    with deterministic behavior, retry logic, fallbacks, and guardrails.

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
        L0Stream - async iterator with .state, .abort(), and .text()

    Example:
        ```python
        import l0

        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            ),
            guardrails=[l0.json_rule()],
        )

        # Iterate directly (Pythonic!)
        async for event in result:
            if event.type == l0.EventType.TOKEN:
                print(event.value, end="")

        # Or get full text
        text = await result.text()

        # Access state
        print(result.state.token_count)
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


# Alias for convenience
l0 = run


__all__ = [
    # Version
    "__version__",
    # Core
    "run",
    "l0",  # Alias to run()
    "L0Stream",
    "L0Event",
    "L0State",
    "L0Options",  # Legacy, kept for backwards compatibility
    "L0Result",  # Legacy, kept for backwards compatibility
    "RetryConfig",
    "TimeoutConfig",
    "EventType",
    "ErrorCategory",
    "BackoffStrategy",
    # Events
    "ObservabilityEvent",
    "ObservabilityEventType",
    "EventBus",
    # Stream
    "consume_stream",
    "get_text",
    # Adapters
    "Adapter",
    "OpenAIAdapter",
    "LiteLLMAdapter",
    "register_adapter",
    "detect_adapter",
    # Guardrails
    "GuardrailRule",
    "GuardrailViolation",
    "check_guardrails",
    "json_rule",
    "strict_json_rule",
    "pattern_rule",
    "zero_output_rule",
    "stall_rule",
    "repetition_rule",
    "recommended_guardrails",
    "strict_guardrails",
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
