"""L0 - Reliability layer for AI/LLM streaming."""

from collections.abc import AsyncIterator, Callable
from typing import Any

from ._utils import (
    CorrectionType,
    extract_json,
    is_valid_json,
    safe_json_parse,
)
from .adapters import (
    # Types for custom adapters
    AdaptedEvent,
    Adapter,
    # Main class (scoped API) - all utilities accessible via Adapters.*
    Adapters,
    LiteLLMAdapter,
    OpenAIAdapter,
    OpenAIAdapterOptions,
)
from .client import WrappedClient, wrap_client
from .consensus import (
    # Result types (needed for type hints)
    Agreement,
    # Main class (scoped API) - all utilities accessible via Consensus.*
    Consensus,
    ConsensusAnalysis,
    ConsensusOutput,
    ConsensusPreset,
    ConsensusResult,
    Disagreement,
    DisagreementValue,
    FieldAgreement,
    FieldConsensus,
    FieldConsensusInfo,
    # Convenience alias for Consensus.run()
    consensus,
)
from .continuation import (
    ContinuationConfig,
    DeduplicationOptions,
    OverlapResult,
    deduplicate_continuation,
    detect_overlap,
)
from .drift import (
    DriftConfig,
    DriftDetector,
    DriftResult,
    check_drift,
    create_drift_detector,
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
from .json_schema import (
    JSONSchemaAdapter,
    JSONSchemaDefinition,
    JSONSchemaValidationError,
    JSONSchemaValidationFailure,
    JSONSchemaValidationSuccess,
    SimpleJSONSchemaAdapter,
    UnifiedSchema,
    create_simple_json_schema_adapter,
    get_json_schema_adapter,
    has_json_schema_adapter,
    is_json_schema,
    register_json_schema_adapter,
    unregister_json_schema_adapter,
    validate_json_schema,
    wrap_json_schema,
)
from .logging import enable_debug
from .multimodal import Multimodal
from .parallel import (
    AggregatedTelemetry,
    ParallelOptions,
    ParallelResult,
    RaceResult,
    batched,
    parallel,
    race,
    sequential,
)
from .pipeline import (
    FAST_PIPELINE,
    PRODUCTION_PIPELINE,
    RELIABLE_PIPELINE,
    Pipeline,
    PipelineOptions,
    PipelineResult,
    PipelineStep,
    StepContext,
    StepResult,
    chain_pipelines,
    create_branch_step,
    create_pipeline,
    create_step,
    parallel_pipelines,
    pipe,
)
from .pool import OperationPool, PoolOptions, PoolStats, create_pool
from .runtime import LifecycleCallbacks, TimeoutError, _internal_run
from .stream import consume_stream, get_text
from .structured import (
    MINIMAL_STRUCTURED,
    RECOMMENDED_STRUCTURED,
    STRICT_STRUCTURED,
    AutoCorrectInfo,
    StructuredConfig,
    StructuredResult,
    StructuredState,
    StructuredStreamResult,
    StructuredTelemetry,
    structured,
    structured_array,
    structured_object,
    structured_stream,
)
from .types import (
    BackoffStrategy,
    CheckIntervals,
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
    RetryableErrorType,
    State,
    Stream,
    StreamFactory,
    StreamSource,
    Timeout,
)
from .version import __version__
from .window import (
    ChunkingStrategy,
    ChunkProcessConfig,
    ChunkResult,
    ContextRestorationOptions,
    ContextRestorationStrategy,
    DocumentChunk,
    DocumentWindow,
    ProcessingStats,
    Window,
    WindowConfig,
    WindowStats,
    get_processing_stats,
    l0_with_window,
    merge_chunks,
    merge_results,
    process_with_window,
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
) -> "WrappedClient | LazyStream[Any]":
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
    drift_detector: DriftDetector | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    check_intervals: "CheckIntervals | None" = None,
    adapter: Adapter | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: Callable[[str], str] | None = None,
    # Lifecycle callbacks
    callbacks: "LifecycleCallbacks | None" = None,
    on_start: Callable[[int, bool, bool], None] | None = None,
    on_complete: Callable[[State], None] | None = None,
    on_error: Callable[[Exception, bool, bool], None] | None = None,
    on_stream_event: Callable[[Event], None] | None = None,
    on_violation: Callable[..., None] | None = None,
    on_retry: Callable[[int, str], None] | None = None,
    on_fallback: Callable[[int, str], None] | None = None,
    on_resume: Callable[[str, int], None] | None = None,
    on_checkpoint: Callable[[str, int], None] | None = None,
    on_timeout: Callable[[str, float], None] | None = None,
    on_abort: Callable[[int, int], None] | None = None,
    on_drift: Callable[[list[str], float | None], None] | None = None,
    on_tool_call: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> "Stream[Any]":
    """Run L0 with a stream factory (supports retries and fallbacks).

    Use this when you need retry/fallback support, which requires re-creating
    the stream. For simple cases, prefer l0.wrap().

    Args:
        stream: Factory function that returns an async LLM stream
        fallbacks: Optional list of fallback stream factories
        guardrails: Optional list of guardrail rules to apply
        drift_detector: Optional drift detector for detecting model derailment
        retry: Optional retry configuration
        timeout: Optional timeout configuration
        check_intervals: Optional check intervals for guardrails/drift/checkpoint
        adapter: Optional adapter hint ("openai", "litellm", or Adapter instance)
        on_event: Optional callback for observability events
        meta: Optional metadata attached to all events
        buffer_tool_calls: Buffer tool call arguments until complete (default: False)
        continue_from_last_good_token: Resume from checkpoint on retry (default: False)
        build_continuation_prompt: Callback to modify prompt for continuation
        callbacks: Optional LifecycleCallbacks object with all callbacks
        on_start: Called when execution attempt begins (attempt, is_retry, is_fallback)
        on_complete: Called when stream completes (state)
        on_error: Called when error occurs (error, will_retry, will_fallback)
        on_stream_event: Called for every L0 event (event)
        on_violation: Called when guardrail violation detected (violation)
        on_retry: Called when retry triggered (attempt, reason)
        on_fallback: Called when switching to fallback (index, reason)
        on_resume: Called when resuming from checkpoint (checkpoint, token_count)
        on_checkpoint: Called when checkpoint saved (checkpoint, token_count)
        on_timeout: Called when timeout occurs (type, elapsed_seconds)
        on_abort: Called when stream aborted (token_count, content_length)
        on_drift: Called when drift detected (drift_types, confidence)
        on_tool_call: Called when tool call detected (name, id, args)

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
            retry=l0.Retry(attempts=3),
            check_intervals=l0.CheckIntervals(guardrails=5, checkpoint=10),
            continue_from_last_good_token=True,
            on_start=lambda attempt, is_retry, is_fallback: print(f"Starting attempt {attempt}"),
            on_complete=lambda state: print(f"Done: {state.token_count} tokens"),
            on_error=lambda e, will_retry, will_fallback: print(f"Error: {e}"),
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
        drift_detector=drift_detector,
        retry=retry,
        timeout=timeout,
        check_intervals=check_intervals,
        adapter=adapter,
        on_event=on_event,
        meta=meta,
        buffer_tool_calls=buffer_tool_calls,
        continue_from_last_good_token=continue_from_last_good_token,
        build_continuation_prompt=build_continuation_prompt,
        callbacks=callbacks,
        on_start=on_start,
        on_complete=on_complete,
        on_error=on_error,
        on_stream_event=on_stream_event,
        on_violation=on_violation,
        on_retry=on_retry,
        on_fallback=on_fallback,
        on_resume=on_resume,
        on_checkpoint=on_checkpoint,
        on_timeout=on_timeout,
        on_abort=on_abort,
        on_drift=on_drift,
        on_tool_call=on_tool_call,
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
    "RetryableErrorType",
    "Timeout",
    "TimeoutError",
    "CheckIntervals",
    "BackoffStrategy",
    "ErrorCategory",
    "ErrorTypeDelays",
    "LifecycleCallbacks",
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
    # All utilities accessible via Adapters.* (e.g., Adapters.register(), Adapters.to_l0_events(), etc.)
    "Adapters",
    # Adapter types (for custom adapters)
    "Adapter",
    "AdaptedEvent",
    "OpenAIAdapter",
    "OpenAIAdapterOptions",
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
    "structured_object",
    "structured_array",
    "StructuredResult",
    "StructuredStreamResult",
    "StructuredState",
    "StructuredTelemetry",
    "StructuredConfig",
    "AutoCorrectInfo",
    "MINIMAL_STRUCTURED",
    "RECOMMENDED_STRUCTURED",
    "STRICT_STRUCTURED",
    # JSON utilities
    "CorrectionType",
    "extract_json",
    "is_valid_json",
    "safe_json_parse",
    # Parallel
    "parallel",
    "race",
    "sequential",
    "batched",
    "ParallelResult",
    "ParallelOptions",
    "RaceResult",
    "AggregatedTelemetry",
    # Pool (dynamic workload)
    "OperationPool",
    "PoolOptions",
    "PoolStats",
    "create_pool",
    # Pipeline (multi-step workflows)
    "pipe",
    "Pipeline",
    "PipelineStep",
    "PipelineOptions",
    "PipelineResult",
    "StepContext",
    "StepResult",
    "create_pipeline",
    "create_step",
    "chain_pipelines",
    "parallel_pipelines",
    "create_branch_step",
    "FAST_PIPELINE",
    "RELIABLE_PIPELINE",
    "PRODUCTION_PIPELINE",
    # Consensus (scoped API)
    # All utilities accessible via Consensus.* (e.g., Consensus.quick(), Consensus.STRICT, etc.)
    "Consensus",
    "consensus",  # Convenience alias for Consensus.run()
    # Consensus result types (for type hints)
    "ConsensusResult",
    "ConsensusOutput",
    "ConsensusAnalysis",
    "ConsensusPreset",
    "Agreement",
    "Disagreement",
    "DisagreementValue",
    "FieldAgreement",
    "FieldConsensus",
    "FieldConsensusInfo",  # Alias for FieldAgreement (backwards compat)
    # Window (scoped API)
    "Window",  # Class with .create(), .small(), .medium(), .large(), .paragraph(), .sentence(), .chunk(), .estimate_tokens()
    "DocumentWindow",
    "DocumentChunk",
    "WindowConfig",
    "WindowStats",
    "ChunkProcessConfig",
    "ChunkResult",
    "ChunkingStrategy",
    "ProcessingStats",
    "ContextRestorationOptions",
    "ContextRestorationStrategy",
    # Window helper functions
    "process_with_window",
    "merge_results",
    "merge_chunks",
    "get_processing_stats",
    "l0_with_window",
    # Debug
    "enable_debug",
    # Formatting
    "Format",
    # JSON Schema (adapter-based validation)
    "JSONSchemaAdapter",
    "JSONSchemaDefinition",
    "JSONSchemaValidationError",
    "JSONSchemaValidationFailure",
    "JSONSchemaValidationSuccess",
    "SimpleJSONSchemaAdapter",
    "UnifiedSchema",
    "create_simple_json_schema_adapter",
    "get_json_schema_adapter",
    "has_json_schema_adapter",
    "is_json_schema",
    "register_json_schema_adapter",
    "unregister_json_schema_adapter",
    "validate_json_schema",
    "wrap_json_schema",
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
    # Drift detection
    "DriftDetector",
    "DriftConfig",
    "DriftResult",
    "check_drift",
    "create_drift_detector",
]
