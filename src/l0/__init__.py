"""L0 - Reliability layer for AI/LLM streaming."""

from collections.abc import AsyncIterator, Callable
from typing import Any

from .adapters import (
    Adapter,
    Adapters,
    LiteLLMAdapter,
    OpenAIAdapter,
)
from .client import WrappedClient, wrap_client
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
from .continuation import (
    ContinuationConfig,
    DeduplicationOptions,
    OverlapResult,
    deduplicate_continuation,
    detect_overlap,
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
    RawStream,
    Retry,
    State,
    Stream,
    StreamFactory,
    StreamSource,
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
    client_or_stream: Any,
    *,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: Any | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: Callable[[str], str] | None = None,
) -> WrappedClient | LazyStream:
    """Wrap an OpenAI/LiteLLM client or raw stream with L0 reliability.

    This is the preferred API. Pass a client for full retry support,
    or a raw stream for simple cases.

    Args:
        client_or_stream: OpenAI/LiteLLM client or raw async iterator
        guardrails: Optional guardrail rules to apply
        retry: Retry configuration (default: Retry.recommended() for clients)
        timeout: Timeout configuration
        adapter: Adapter hint ("openai", "litellm", or Adapter instance)
        on_event: Observability event callback
        meta: Metadata attached to all events
        buffer_tool_calls: Buffer tool calls until complete (default: False)
        continue_from_last_good_token: Resume from checkpoint on retry (default: False)
        build_continuation_prompt: Callback to modify prompt for continuation

    Returns:
        WrappedClient (for clients) or LazyStream (for raw streams)

    Example - Wrap a client (recommended):
        ```python
        import l0
        from openai import AsyncOpenAI

        # Wrap the client once
        client = l0.wrap(AsyncOpenAI())

        # Use normally - L0 reliability is automatic
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        # Iterate with L0 events
        async for event in response:
            if event.is_token:
                print(event.text, end="")

        # Or read all at once
        text = await response.read()
        ```

    Example - Wrap a raw stream (no retry support):
        ```python
        import l0

        raw = await client.chat.completions.create(..., stream=True)
        result = l0.wrap(raw)
        text = await result.read()
        ```
    """
    # Detect if this is a client (has .chat.completions) or a raw stream
    if hasattr(client_or_stream, "chat") and hasattr(
        client_or_stream.chat, "completions"
    ):
        # It's an OpenAI-style client
        return wrap_client(
            client_or_stream,
            guardrails=guardrails,
            retry=retry,
            timeout=timeout,
            adapter=adapter,
            on_event=on_event,
            meta=meta,
            buffer_tool_calls=buffer_tool_calls,
            continue_from_last_good_token=continue_from_last_good_token,
            build_continuation_prompt=build_continuation_prompt,
        )
    else:
        # It's a raw stream - wrap with LazyStream (no retry support)
        return LazyStream(
            stream=client_or_stream,
            guardrails=guardrails,
            timeout=timeout,
            adapter=adapter,
            on_event=on_event,
            meta=meta,
            buffer_tool_calls=buffer_tool_calls,
        )


async def run(
    stream: StreamFactory,
    *,
    fallbacks: list[StreamFactory] | None = None,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: Adapter | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: Callable[[str], str] | None = None,
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
        continue_from_last_good_token: Resume from checkpoint on retry (default: False)
        build_continuation_prompt: Callback to modify prompt for continuation

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
            continue_from_last_good_token=True,  # Resume from checkpoint on retry
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
        continue_from_last_good_token=continue_from_last_good_token,
        build_continuation_prompt=build_continuation_prompt,
    )


# Legacy alias
l0 = run


__all__ = [
    # Version
    "__version__",
    # Core API
    "wrap",  # Preferred - wraps client or raw stream
    "run",  # With factory - supports retries/fallbacks
    "l0",  # Alias to run
    # Types
    "Stream",
    "LazyStream",
    "WrappedClient",  # Wrapped OpenAI/LiteLLM client
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
    # Continuation (checkpoint resumption)
    "ContinuationConfig",
    "DeduplicationOptions",
    "OverlapResult",
    "detect_overlap",
    "deduplicate_continuation",
]
