"""L0 - Reliability layer for AI/LLM streaming."""

from collections.abc import AsyncIterator, Callable
from typing import Any

from .adapters import (
    Adapter,
    LiteLLMAdapter,
    OpenAIAdapter,
    detect_adapter,
    register_adapter,
)
from .consensus import (
    Agreement,
    ConsensusAnalysis,
    ConsensusOutput,
    ConsensusPreset,
    ConsensusResult,
    Disagreement,
    DisagreementValue,
    FieldConsensus,
    FieldConsensusInfo,
    best_consensus,
    consensus,
    get_consensus_value,
    lenient_consensus,
    quick_consensus,
    standard_consensus,
    strict_consensus,
    validate_consensus,
)
from .errors import NetworkError, NetworkErrorAnalysis, NetworkErrorType
from .events import EventBus, ObservabilityEvent, ObservabilityEventType
from .format import Format
from .guardrails import (
    # Patterns
    BAD_PATTERNS,
    BadPatterns,
    # Core
    GuardrailRule,
    Guardrails,
    GuardrailViolation,
    # Analysis result types
    JsonAnalysis,
    LatexAnalysis,
    MarkdownAnalysis,
    # Analysis
    analyze_json_structure,
    analyze_latex_structure,
    analyze_markdown_structure,
    check_guardrails,
    custom_pattern_rule,
    find_bad_patterns,
    is_noise_only,
    is_zero_output,
    # Rules
    json_rule,
    latex_rule,
    looks_like_json,
    looks_like_latex,
    looks_like_markdown,
    markdown_rule,
    pattern_rule,
    # Presets (legacy)
    recommended_guardrails,
    repetition_rule,
    stall_rule,
    strict_guardrails,
    strict_json_rule,
    zero_output_rule,
)
from .logging import enable_debug
from .parallel import batched, parallel, race
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
    ErrorCategory,
    ErrorTypeDelays,
    Event,
    EventType,
    LazyStream,
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
    WindowConfig,
    chunk_document,
    create_window,
    estimate_tokens,
    large_window,
    medium_window,
    paragraph_window,
    sentence_window,
    small_window,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def wrap(
    stream: AsyncIterator[Any],
    *,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: Any | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
) -> LazyStream:
    """Wrap a raw LLM stream with L0 reliability.

    This is the preferred API - returns immediately, no await needed!
    Like httpx.AsyncClient() or aiohttp.ClientSession().

    Args:
        stream: Raw async iterator from OpenAI/LiteLLM/etc.
        guardrails: Optional list of guardrail rules to apply
        retry: Optional retry configuration
        timeout: Optional timeout configuration
        adapter: Optional adapter hint ("openai", "litellm", or Adapter instance)
        on_event: Optional callback for observability events
        meta: Optional metadata attached to all events

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
    # Adapters
    "Adapter",
    "OpenAIAdapter",
    "LiteLLMAdapter",
    "register_adapter",
    "detect_adapter",
    # Guardrails
    "Guardrails",  # Class with .recommended(), .strict(), .minimal(), etc.
    "GuardrailRule",
    "GuardrailViolation",
    "check_guardrails",
    # Guardrail rules
    "json_rule",
    "strict_json_rule",
    "markdown_rule",
    "latex_rule",
    "pattern_rule",
    "custom_pattern_rule",
    "zero_output_rule",
    "stall_rule",
    "repetition_rule",
    "recommended_guardrails",  # Legacy
    "strict_guardrails",  # Legacy
    # Guardrail analysis
    "analyze_json_structure",
    "analyze_markdown_structure",
    "analyze_latex_structure",
    "looks_like_json",
    "looks_like_markdown",
    "looks_like_latex",
    "is_zero_output",
    "is_noise_only",
    "find_bad_patterns",
    "JsonAnalysis",
    "MarkdownAnalysis",
    "LatexAnalysis",
    "BAD_PATTERNS",
    "BadPatterns",
    # Structured
    "structured",
    "structured_stream",
    "StructuredResult",
    "StructuredStreamResult",
    "AutoCorrectInfo",
    # Parallel
    "parallel",
    "race",
    "batched",
    # Consensus
    "consensus",
    "ConsensusResult",
    "ConsensusOutput",
    "ConsensusAnalysis",
    "ConsensusPreset",
    "Agreement",
    "Disagreement",
    "DisagreementValue",
    "FieldConsensus",
    "FieldConsensusInfo",
    "quick_consensus",
    "get_consensus_value",
    "validate_consensus",
    "strict_consensus",
    "standard_consensus",
    "lenient_consensus",
    "best_consensus",
    # Window (document chunking)
    "create_window",
    "DocumentWindow",
    "DocumentChunk",
    "WindowConfig",
    "ChunkProcessConfig",
    "ChunkResult",
    "chunk_document",
    "estimate_tokens",
    "small_window",
    "medium_window",
    "large_window",
    "paragraph_window",
    "sentence_window",
    # Debug
    "enable_debug",
    # Formatting
    "Format",
]
