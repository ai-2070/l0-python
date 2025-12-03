"""L0 - Reliability layer for AI/LLM streaming."""

from collections.abc import AsyncIterator, Callable
from typing import Any

from .adapters import (
    Adapter,
    Adapters,
    LiteLLMAdapter,
    OpenAIAdapter,
)
from .consensus import (
    Agreement,
    Consensus,
    ConsensusAnalysis,
    ConsensusOutput,
    ConsensusPreset,
    ConsensusResult,
    Disagreement,
    DisagreementValue,
    FieldConsensus,
    FieldConsensusInfo,
    consensus,
)
from .errors import (
    # L0 Error (with .categorize() and .is_retryable() static methods)
    Error,
    ErrorCode,
    ErrorContext,
    # Failure and recovery
    FailureType,
    # Network errors (scoped API)
    NetworkError,
    NetworkErrorAnalysis,
    NetworkErrorType,
    RecoveryPolicy,
    RecoveryStrategy,
)
from .events import EventBus, ObservabilityEvent, ObservabilityEventType
from .format import Format
from .guardrails import (
    GuardrailRule,
    Guardrails,
    GuardrailViolation,
    JsonAnalysis,
    LatexAnalysis,
    MarkdownAnalysis,
)
from .logging import enable_debug
from .multimodal import Multimodal
from .parallel import (
    ParallelOptions,
    ParallelResult,
    batched,
    parallel,
    race,
    sequential,
)
from .runtime import TimeoutError, _internal_run
from .stream import consume_stream, get_text
from .structured import (
    AutoCorrectInfo,
    StructuredResult,
    StructuredStreamResult,
    structured,
    structured_stream,
)
from .types import (
    BackoffStrategy,
    ContentType,
    DataPayload,
    ErrorCategory,
    ErrorTypeDelays,
    Event,
    EventType,
    LazyStream,
    Progress,
    Retry,
    State,
    Stream,
    Timeout,
)
from .version import __version__
from .window import (
    ChunkProcessConfig,
    ChunkResult,
    DocumentChunk,
    DocumentWindow,
    Window,
    WindowConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def wrap(
    stream: AsyncIterator[Any],
    *,
    guardrails: list[GuardrailRule] | None = None,
    timeout: Timeout | None = None,
    adapter: Any | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
    buffer_tool_calls: bool = False,
) -> LazyStream:
    """Wrap a raw LLM stream with L0 reliability.

    This is the preferred API - returns immediately, no await needed!
    Like httpx.AsyncClient() or aiohttp.ClientSession().

    Note: For retry support, use l0.run() with a factory function instead,
    since a raw stream cannot be recreated after consumption.

    Args:
        stream: Raw async iterator from OpenAI/LiteLLM/etc.
        guardrails: Optional list of guardrail rules to apply
        timeout: Optional timeout configuration
        adapter: Optional adapter hint ("openai", "litellm", or Adapter instance)
        on_event: Optional callback for observability events
        meta: Optional metadata attached to all events
        buffer_tool_calls: Buffer tool call arguments until complete (default: False)

    Returns:
        LazyStream - async iterator with .state, .abort(), and .read()

    Example:
        ```python
        import l0
        import litellm

        raw = litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        # Simple - no double await!
        result = l0.wrap(raw)
        text = await result.read()

        # Streaming
        async for event in l0.wrap(raw):
            if event.is_token:
                print(event.text, end="")

        # Context manager
        async with l0.wrap(raw, guardrails=l0.Guardrails.recommended()) as result:
            async for event in result:
                if event.is_token:
                    print(event.text, end="")
        ```
    """
    return LazyStream(
        stream=stream,
        guardrails=guardrails,
        retry=None,
        timeout=timeout,
        adapter=adapter,
        on_event=on_event,
        meta=meta,
        buffer_tool_calls=buffer_tool_calls,
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
    buffer_tool_calls: bool = False,
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
        buffer_tool_calls: Buffer tool call arguments until complete (default: False)

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
        buffer_tool_calls=buffer_tool_calls,
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
    "LazyStream",
    "Event",
    "State",
    "EventType",
    # Config
    "Retry",
    "Timeout",
    "TimeoutError",
    "BackoffStrategy",
    "ErrorCategory",
    "ErrorTypeDelays",
    # Errors
    "Error",  # L0 error with code, context, checkpoint
    "ErrorCode",  # ZERO_OUTPUT, GUARDRAIL_VIOLATION, etc.
    "ErrorContext",
    # Failure and recovery
    "FailureType",  # network, model, timeout, abort, etc.
    "RecoveryStrategy",  # retry, fallback, continue, halt
    "RecoveryPolicy",
    # Network errors (scoped API)
    "NetworkError",  # Class with .check(), .analyze(), .is_timeout(), etc.
    "NetworkErrorType",
    "NetworkErrorAnalysis",
    # Events
    "ObservabilityEvent",
    "ObservabilityEventType",
    "EventBus",
    # Stream utilities
    "consume_stream",
    "get_text",
    # Adapters (scoped API)
    "Adapters",  # Class with .detect(), .register(), .openai(), .litellm()
    "Adapter",
    "OpenAIAdapter",
    "LiteLLMAdapter",
    # Guardrails (scoped API)
    "Guardrails",  # Class with .recommended(), .strict(), .json(), .check(), .analyze_json(), etc.
    "GuardrailRule",
    "GuardrailViolation",
    "JsonAnalysis",
    "MarkdownAnalysis",
    "LatexAnalysis",
    # Structured
    "structured",
    "structured_stream",
    "StructuredResult",
    "StructuredStreamResult",
    "AutoCorrectInfo",
    # Parallel
    "parallel",
    "race",
    "sequential",
    "batched",
    "ParallelResult",
    "ParallelOptions",
    # Consensus (scoped API)
    "Consensus",  # Class with .run(), .strict(), .standard(), .lenient(), .best(), .quick(), .get_value(), .validate()
    "consensus",  # Convenience alias for Consensus.run()
    "ConsensusResult",
    "ConsensusOutput",
    "ConsensusAnalysis",
    "ConsensusPreset",
    "Agreement",
    "Disagreement",
    "DisagreementValue",
    "FieldConsensus",
    "FieldConsensusInfo",
    # Window (scoped API)
    "Window",  # Class with .create(), .small(), .medium(), .large(), .paragraph(), .sentence(), .chunk(), .estimate_tokens()
    "DocumentWindow",
    "DocumentChunk",
    "WindowConfig",
    "ChunkProcessConfig",
    "ChunkResult",
    # Debug
    "enable_debug",
    # Formatting
    "Format",
    # Multimodal (scoped API)
    "Multimodal",  # Class with .image(), .audio(), .video(), .from_stream(), etc.
    "ContentType",
    "DataPayload",
    "Progress",
]
