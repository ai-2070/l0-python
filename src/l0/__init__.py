"""L0 - Reliability layer for AI/LLM streaming."""

from collections.abc import AsyncIterator as _AsyncIterator
from collections.abc import Callable as _Callable
from typing import Any as _Any
from typing import Protocol as _Protocol
from typing import overload as _overload
from typing import runtime_checkable as _runtime_checkable

from ._utils import (
    # Scoped API
    JSON,
    # Types (also available via JSON.*)
    AutoCorrectResult,
    CorrectionType,
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
from .comparison import (
    # Scoped API
    Compare,
    # Types (for type hints, also available via Compare.*)
    Difference,
    DifferenceSeverity,
    DifferenceType,
    ObjectComparisonOptions,
    StringComparisonOptions,
)
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
    # Scoped API
    Continuation,
    # Types (also available via Continuation.*)
    ContinuationConfig,
    DeduplicationOptions,
    OverlapResult,
)
from .drift import (
    # Scoped API
    Drift,
    # Types (also available via Drift.*)
    DriftConfig,
    DriftDetector,
    DriftResult,
)
from .errors import (
    # L0 Error (with scoped API: .categorize(), .is_retryable(), .is_error(), .get_category())
    Error,
    ErrorCode,
    ErrorContext,
    # Failure and recovery
    FailureType,
    # Network errors (scoped API: .check(), .analyze(), .is_timeout(), .suggest_delay(), etc.)
    NetworkError,
    NetworkErrorAnalysis,
    NetworkErrorType,
    RecoveryPolicy,
    RecoveryStrategy,
)
from .event_sourcing import (
    # Types (also available via EventSourcing.*)
    EventEnvelope,
    EventRecorder,
    EventReplayer,
    # Scoped API
    EventSourcing,
    EventStore,
    EventStoreWithSnapshots,
    InMemoryEventStore,
    RecordedEvent,
    RecordedEventType,
    ReplayCallbacks,
    ReplayComparison,
    ReplayedState,
    ReplayResult,
    SerializedError,
    Snapshot,
    StreamMetadata,
)
from .events import EventBus, ObservabilityEvent, ObservabilityEventType
from .format import Format
from .guardrails import (
    JSON_ONLY_GUARDRAILS,
    LATEX_ONLY_GUARDRAILS,
    MARKDOWN_ONLY_GUARDRAILS,
    MINIMAL_GUARDRAILS,
    RECOMMENDED_GUARDRAILS,
    STRICT_GUARDRAILS,
    GuardrailRule,
    Guardrails,
    GuardrailViolation,
    JsonAnalysis,
    LatexAnalysis,
    MarkdownAnalysis,
)
from .json_schema import (
    # Scoped API
    JSONSchema,
    # Types (also available via JSONSchema.*)
    JSONSchemaAdapter,
    JSONSchemaDefinition,
    JSONSchemaValidationError,
    JSONSchemaValidationFailure,
    JSONSchemaValidationSuccess,
    SimpleJSONSchemaAdapter,
    UnifiedSchema,
)
from .logging import enable_debug
from .metrics import (
    Metrics,
    MetricsSnapshot,
    create_metrics,
    get_global_metrics,
    reset_global_metrics,
)
from .monitoring import (
    # Scoped API
    Monitoring,
    # OpenTelemetry
    OpenTelemetry,
    OpenTelemetryConfig,
    OpenTelemetryExporter,
    OpenTelemetryExporterConfig,
    SemanticAttributes,
    # Sentry
    Sentry,
    SentryConfig,
    SentryExporter,
    SentryExporterConfig,
)
from .multimodal import Multimodal
from .normalize import (
    # Types (also available via Text.*)
    NormalizeOptions,
    # Scoped API
    Text,
    WhitespaceOptions,
)
from .parallel import (
    # Types (also available via Parallel.*)
    AggregatedTelemetry,
    # Scoped API
    Parallel,
    ParallelOptions,
    ParallelResult,
    RaceResult,
    # Legacy functions (kept for convenience)
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
from .state_machine import (
    RuntimeState,
    RuntimeStates,
    StateMachine,
    StateTransition,
    create_state_machine,
)
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
    ERROR_TYPE_DELAY_DEFAULTS,
    EXPONENTIAL_RETRY,
    MINIMAL_RETRY,
    RECOMMENDED_RETRY,
    RETRY_DEFAULTS,
    STRICT_RETRY,
    BackoffStrategy,
    CheckIntervals,
    ContentType,
    DataPayload,
    ErrorCategory,
    ErrorTypeDelayDefaults,
    ErrorTypeDelays,
    Event,
    EventType,
    LazyStream,
    Progress,
    RawStream,
    Retry,
    RetryableErrorType,
    RetryDefaults,
    State,
    Stream,
    StreamFactory,
    StreamSource,
    Timeout,
)
from .version import __version__
from .window import (
    # Types (also available via Window.*)
    ChunkingStrategy,
    ChunkProcessConfig,
    ChunkResult,
    ContextRestorationOptions,
    ContextRestorationStrategy,
    DocumentChunk,
    DocumentWindow,
    ProcessingStats,
    # Scoped API
    Window,
    WindowConfig,
    WindowStats,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


# Protocol for OpenAI-style clients (has .chat.completions)
@_runtime_checkable
class _OpenAILikeClient(_Protocol):
    """Protocol matching OpenAI/LiteLLM client structure."""

    @property
    def chat(self) -> _Any: ...


# Overloads for wrap() to provide accurate return types
@_overload
def wrap(
    client_or_stream: _OpenAILikeClient,
    *,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: _Any | str | None = None,
    on_event: _Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: _Callable[[str], str] | None = None,
) -> WrappedClient: ...


@_overload
def wrap(
    client_or_stream: _AsyncIterator[_Any],
    *,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: _Any | str | None = None,
    on_event: _Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: _Callable[[str], str] | None = None,
) -> "LazyStream[_Any]": ...


def wrap(
    client_or_stream: _Any,
    *,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: _Any | str | None = None,
    on_event: _Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: _Callable[[str], str] | None = None,
) -> "WrappedClient | LazyStream[_Any]":
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
    on_event: _Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: _Callable[[str], str] | None = None,
    # Lifecycle callbacks
    callbacks: "LifecycleCallbacks | None" = None,
    on_start: _Callable[[int, bool, bool], None] | None = None,
    on_complete: _Callable[[State], None] | None = None,
    on_error: _Callable[[Exception, bool, bool], None] | None = None,
    on_stream_event: _Callable[[Event], None] | None = None,
    on_violation: _Callable[..., None] | None = None,
    on_retry: _Callable[[int, str], None] | None = None,
    on_fallback: _Callable[[int, str], None] | None = None,
    on_resume: _Callable[[str, int], None] | None = None,
    on_checkpoint: _Callable[[str, int], None] | None = None,
    on_timeout: _Callable[[str, float], None] | None = None,
    on_abort: _Callable[[int, int], None] | None = None,
    on_drift: _Callable[[list[str], float | None], None] | None = None,
    on_tool_call: _Callable[[str, str, dict[str, _Any]], None] | None = None,
) -> "Stream[_Any]":
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

# ─────────────────────────────────────────────────────────────────────────────
# Namespace Cleanup
# ─────────────────────────────────────────────────────────────────────────────
# Remove submodule references to avoid namespace pollution.
# Users should import from submodules directly if needed (e.g., from l0.adapters import ...)

import sys as _sys

# List of all submodules to clean from namespace
_submodules = [
    "_utils",
    "adapters",
    "client",
    "comparison",
    "consensus",
    "continuation",
    "drift",
    "errors",
    "event_sourcing",
    "events",
    "format",
    "guardrails",
    "json_schema",
    "logging",
    "metrics",
    "monitoring",
    "multimodal",
    "normalize",
    "parallel",
    "pipeline",
    "pool",
    "runtime",
    "state_machine",
    "stream",
    "structured",
    "types",
    "version",
    "window",
    # Transitive imports that may leak
    "formatting",
    "retry",
    "state",
]

# Clean submodules from namespace and sys.modules cache
_this_module = _sys.modules[__name__]
_mod: str = ""
_full: str = ""
for _mod in _submodules:
    # Remove from namespace if present
    if hasattr(_this_module, _mod):
        delattr(_this_module, _mod)
    # Clean from sys.modules cache
    _full = f"{__name__}.{_mod}"
    if _full in _sys.modules:
        del _sys.modules[_full]

# Hide wrap_client (internal, use wrap() instead)
if hasattr(_this_module, "wrap_client"):
    delattr(_this_module, "wrap_client")

# Clean up cleanup variables
del _sys, _submodules, _this_module, _mod, _full

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
    "StreamFactory",
    "StreamSource",
    "RawStream",
    # Config
    "Retry",
    "RetryDefaults",
    "RETRY_DEFAULTS",
    "MINIMAL_RETRY",
    "RECOMMENDED_RETRY",
    "STRICT_RETRY",
    "EXPONENTIAL_RETRY",
    "RetryableErrorType",
    "Timeout",
    "TimeoutError",
    "CheckIntervals",
    "BackoffStrategy",
    "ErrorCategory",
    "ErrorTypeDelays",
    "ErrorTypeDelayDefaults",
    "ERROR_TYPE_DELAY_DEFAULTS",
    "LifecycleCallbacks",
    # Errors
    "Error",  # L0 error with code, context, checkpoint
    "ErrorCode",  # ZERO_OUTPUT, GUARDRAIL_VIOLATION, etc.
    "ErrorContext",
    # Failure and recovery
    "FailureType",  # network, model, timeout, abort, etc.
    "RecoveryStrategy",  # retry, fallback, continue, halt
    "RecoveryPolicy",
    # Network errors (scoped API: NetworkError.check(), .analyze(), .is_timeout(), .suggest_delay(), etc.)
    "NetworkError",
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
    # Guardrail presets (TypeScript parity)
    "MINIMAL_GUARDRAILS",
    "RECOMMENDED_GUARDRAILS",
    "STRICT_GUARDRAILS",
    "JSON_ONLY_GUARDRAILS",
    "MARKDOWN_ONLY_GUARDRAILS",
    "LATEX_ONLY_GUARDRAILS",
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
    # JSON (scoped API)
    "JSON",  # Class with .extract(), .is_valid(), .parse(), .auto_correct(), .extract_from_markdown()
    # JSON types (also available via JSON.*)
    "AutoCorrectResult",
    "CorrectionType",
    # Parallel (scoped API + legacy functions)
    "Parallel",  # Class with .run(), .race(), .sequential(), .batched()
    # Parallel types (also available via Parallel.*)
    "ParallelResult",
    "ParallelOptions",
    "RaceResult",
    "AggregatedTelemetry",
    # Parallel legacy functions (kept for convenience)
    "parallel",
    "race",
    "sequential",
    "batched",
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
    "Window",  # Class with .create(), .small(), .medium(), .large(), .paragraph(), .sentence(), .chunk(), .estimate_tokens(), .process(), .merge_results(), .merge_chunks(), .get_stats()
    # Window types (also available via Window.*)
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
    # Debug
    "enable_debug",
    # Monitoring (scoped API)
    "Monitoring",  # Class with .combine(), .filter(), .exclude(), .debounce(), .batch(), .sample(), .tap(), .opentelemetry(), .sentry()
    # OpenTelemetry
    "OpenTelemetry",
    "OpenTelemetryConfig",
    "OpenTelemetryExporter",
    "OpenTelemetryExporterConfig",
    "SemanticAttributes",
    # Sentry
    "Sentry",
    "SentryConfig",
    "SentryExporter",
    "SentryExporterConfig",
    # Formatting
    "Format",
    # JSON Schema (scoped API)
    "JSONSchema",  # Class with .register(), .unregister(), .has_adapter(), .get_adapter(), .is_schema(), .validate(), .wrap(), .create_simple_adapter()
    # JSON Schema types (also available via JSONSchema.*)
    "JSONSchemaAdapter",
    "JSONSchemaDefinition",
    "JSONSchemaValidationError",
    "JSONSchemaValidationFailure",
    "JSONSchemaValidationSuccess",
    "SimpleJSONSchemaAdapter",
    "UnifiedSchema",
    # Multimodal (scoped API)
    "Multimodal",  # Class with .image(), .audio(), .video(), .from_stream(), etc.
    "ContentType",
    "DataPayload",
    "Progress",
    # Continuation (scoped API)
    "Continuation",  # Class with .detect_overlap(), .deduplicate()
    # Continuation types (also available via Continuation.*)
    "ContinuationConfig",
    "DeduplicationOptions",
    "OverlapResult",
    # Drift (scoped API)
    "Drift",  # Class with .check(), .create_detector()
    # Drift types (also available via Drift.*)
    "DriftDetector",
    "DriftConfig",
    "DriftResult",
    # State machine
    "StateMachine",
    "RuntimeState",
    "RuntimeStates",
    "StateTransition",
    "create_state_machine",
    # Event Sourcing (scoped API)
    "EventSourcing",  # Class with .replay(), .get_metadata(), .compare(), .create_store(), .create_recorder(), .create_replayer(), .generate_stream_id()
    # Event Sourcing types (also available via EventSourcing.*)
    "EventStore",
    "EventStoreWithSnapshots",
    "InMemoryEventStore",
    "EventRecorder",
    "EventReplayer",
    "EventEnvelope",
    "RecordedEvent",
    "RecordedEventType",
    "Snapshot",
    "SerializedError",
    "ReplayResult",
    "ReplayCallbacks",
    "ReplayedState",
    "ReplayComparison",
    "StreamMetadata",
    # Metrics
    "Metrics",
    "MetricsSnapshot",
    "create_metrics",
    "get_global_metrics",
    "reset_global_metrics",
    # Text (scoped API)
    "Text",  # Class with .normalize_newlines(), .normalize_whitespace(), .dedent(), .indent(), .trim(), .for_model(), etc.
    # Text types (also available via Text.*)
    "NormalizeOptions",
    "WhitespaceOptions",
    # Comparison (scoped API)
    "Compare",  # Class with .strings(), .levenshtein(), .jaro_winkler(), .cosine(), .numbers(), .deep_equal(), .objects(), .arrays(), .values(), etc.
    # Comparison types (also available via Compare.*)
    "Difference",
    "DifferenceSeverity",
    "DifferenceType",
    "StringComparisonOptions",
    "ObjectComparisonOptions",
]
