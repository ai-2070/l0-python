"""L0 - Reliability layer for AI/LLM streaming.

This module uses lazy imports to avoid loading all submodules on import.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

# Version is always available
from .version import __version__

# Clean up version module from namespace
del version  # type: ignore[name-defined]  # noqa: F821

# Define what can be imported - used by __getattr__ for lazy loading
_API_MODULES = {
    # api/core.py
    "WrappedClient": "api.core",
    "LifecycleCallbacks": "api.core",
    "TimeoutError": "api.core",
    "_internal_run": "api.core",
    "consume_stream": "api.core",
    "get_text": "api.core",
    "ERROR_TYPE_DELAY_DEFAULTS": "api.core",
    "EXPONENTIAL_RETRY": "api.core",
    "MINIMAL_RETRY": "api.core",
    "RECOMMENDED_RETRY": "api.core",
    "RETRY_DEFAULTS": "api.core",
    "STRICT_RETRY": "api.core",
    "BackoffStrategy": "api.core",
    "CheckIntervals": "api.core",
    "ContentType": "api.core",
    "DataPayload": "api.core",
    "ErrorCategory": "api.core",
    "ErrorTypeDelayDefaults": "api.core",
    "ErrorTypeDelays": "api.core",
    "Event": "api.core",
    "EventType": "api.core",
    "LazyStream": "api.core",
    "Progress": "api.core",
    "RawStream": "api.core",
    "Retry": "api.core",
    "RetryableErrorType": "api.core",
    "RetryDefaults": "api.core",
    "State": "api.core",
    "Stream": "api.core",
    "AwaitableStream": "api.core",
    "AwaitableStreamFactory": "api.core",
    "AwaitableStreamSource": "api.core",
    "StreamFactory": "api.core",
    "StreamSource": "api.core",
    "Timeout": "api.core",
    # api/errors.py
    "Error": "api.errors",
    "ErrorCode": "api.errors",
    "ErrorContext": "api.errors",
    "FailureType": "api.errors",
    "NetworkError": "api.errors",
    "NetworkErrorAnalysis": "api.errors",
    "NetworkErrorType": "api.errors",
    "RecoveryPolicy": "api.errors",
    "RecoveryStrategy": "api.errors",
    # api/adapters.py
    "AdaptedEvent": "api.adapters",
    "Adapter": "api.adapters",
    "Adapters": "api.adapters",
    "LiteLLMAdapter": "api.adapters",
    "OpenAIAdapter": "api.adapters",
    "OpenAIAdapterOptions": "api.adapters",
    # api/events.py
    "EventBus": "api.events",
    "ObservabilityEvent": "api.events",
    "ObservabilityEventType": "api.events",
    # api/guardrails.py
    "JSON_ONLY_GUARDRAILS": "api.guardrails",
    "LATEX_ONLY_GUARDRAILS": "api.guardrails",
    "MARKDOWN_ONLY_GUARDRAILS": "api.guardrails",
    "MINIMAL_GUARDRAILS": "api.guardrails",
    "RECOMMENDED_GUARDRAILS": "api.guardrails",
    "STRICT_GUARDRAILS": "api.guardrails",
    "GuardrailRule": "api.guardrails",
    "Guardrails": "api.guardrails",
    "GuardrailViolation": "api.guardrails",
    "Violation": "api.guardrails",  # Alias for GuardrailViolation
    "custom_pattern_rule": "api.guardrails",
    "pattern_rule": "api.guardrails",
    "JsonAnalysis": "api.guardrails",
    "LatexAnalysis": "api.guardrails",
    "MarkdownAnalysis": "api.guardrails",
    # api/structured.py
    "MINIMAL_STRUCTURED": "api.structured",
    "RECOMMENDED_STRUCTURED": "api.structured",
    "STRICT_STRUCTURED": "api.structured",
    "AutoCorrectInfo": "api.structured",
    "StructuredConfig": "api.structured",
    "StructuredResult": "api.structured",
    "StructuredState": "api.structured",
    "StructuredStreamResult": "api.structured",
    "StructuredTelemetry": "api.structured",
    "structured": "api.structured",
    "structured_array": "api.structured",
    "structured_object": "api.structured",
    "structured_stream": "api.structured",
    # api/json.py
    "JSON": "api.json",
    "AutoCorrectResult": "api.json",
    "CorrectionType": "api.json",
    # api/json_schema.py
    "JSONSchema": "api.json_schema",
    "JSONSchemaAdapter": "api.json_schema",
    "JSONSchemaDefinition": "api.json_schema",
    "JSONSchemaValidationError": "api.json_schema",
    "JSONSchemaValidationFailure": "api.json_schema",
    "JSONSchemaValidationSuccess": "api.json_schema",
    "SimpleJSONSchemaAdapter": "api.json_schema",
    "UnifiedSchema": "api.json_schema",
    # api/parallel.py
    "AggregatedTelemetry": "api.parallel",
    "Parallel": "api.parallel",
    "ParallelOptions": "api.parallel",
    "ParallelResult": "api.parallel",
    "RaceResult": "api.parallel",
    "batched": "api.parallel",
    "parallel": "api.parallel",
    "race": "api.parallel",
    "sequential": "api.parallel",
    # api/pipeline.py
    "FAST_PIPELINE": "api.pipeline",
    "PRODUCTION_PIPELINE": "api.pipeline",
    "RELIABLE_PIPELINE": "api.pipeline",
    "Pipeline": "api.pipeline",
    "PipelineOptions": "api.pipeline",
    "PipelineResult": "api.pipeline",
    "PipelineStep": "api.pipeline",
    "StepContext": "api.pipeline",
    "StepResult": "api.pipeline",
    "chain_pipelines": "api.pipeline",
    "create_branch_step": "api.pipeline",
    "create_pipeline": "api.pipeline",
    "create_step": "api.pipeline",
    "parallel_pipelines": "api.pipeline",
    "pipe": "api.pipeline",
    # api/pool.py
    "OperationPool": "api.pool",
    "PoolOptions": "api.pool",
    "PoolStats": "api.pool",
    "create_pool": "api.pool",
    # api/consensus.py
    "Agreement": "api.consensus",
    "Consensus": "api.consensus",
    "ConsensusAnalysis": "api.consensus",
    "ConsensusOutput": "api.consensus",
    "ConsensusPreset": "api.consensus",
    "ConsensusResult": "api.consensus",
    "Disagreement": "api.consensus",
    "DisagreementValue": "api.consensus",
    "FieldAgreement": "api.consensus",
    "FieldConsensus": "api.consensus",
    "FieldConsensusInfo": "api.consensus",
    "consensus": "api.consensus",
    # api/window.py
    "ChunkingStrategy": "api.window",
    "ChunkProcessConfig": "api.window",
    "ChunkResult": "api.window",
    "ContextRestorationOptions": "api.window",
    "ContextRestorationStrategy": "api.window",
    "DocumentChunk": "api.window",
    "DocumentWindow": "api.window",
    "ProcessingStats": "api.window",
    "Window": "api.window",
    "WindowConfig": "api.window",
    "WindowStats": "api.window",
    # api/monitoring.py
    "Monitoring": "api.monitoring",
    "OpenTelemetry": "api.monitoring",
    "OpenTelemetryConfig": "api.monitoring",
    "OpenTelemetryExporter": "api.monitoring",
    "OpenTelemetryExporterConfig": "api.monitoring",
    "SemanticAttributes": "api.monitoring",
    "Sentry": "api.monitoring",
    "SentryConfig": "api.monitoring",
    "SentryExporter": "api.monitoring",
    "SentryExporterConfig": "api.monitoring",
    # api/text.py
    "NormalizeOptions": "api.text",
    "Text": "api.text",
    "WhitespaceOptions": "api.text",
    # api/comparison.py
    "Compare": "api.comparison",
    "Difference": "api.comparison",
    "DifferenceSeverity": "api.comparison",
    "DifferenceType": "api.comparison",
    "ObjectComparisonOptions": "api.comparison",
    "StringComparisonOptions": "api.comparison",
    # api/continuation.py
    "Continuation": "api.continuation",
    "ContinuationConfig": "api.continuation",
    "DeduplicationOptions": "api.continuation",
    "OverlapResult": "api.continuation",
    # api/drift.py
    "Drift": "api.drift",
    "DriftConfig": "api.drift",
    "DriftDetector": "api.drift",
    "DriftResult": "api.drift",
    # api/event_sourcing.py
    "EventEnvelope": "api.event_sourcing",
    "EventRecorder": "api.event_sourcing",
    "EventReplayer": "api.event_sourcing",
    "EventSourcing": "api.event_sourcing",
    "EventStore": "api.event_sourcing",
    "EventStoreWithSnapshots": "api.event_sourcing",
    "InMemoryEventStore": "api.event_sourcing",
    "RecordedEvent": "api.event_sourcing",
    "RecordedEventType": "api.event_sourcing",
    "ReplayCallbacks": "api.event_sourcing",
    "ReplayComparison": "api.event_sourcing",
    "ReplayedState": "api.event_sourcing",
    "ReplayResult": "api.event_sourcing",
    "SerializedError": "api.event_sourcing",
    "Snapshot": "api.event_sourcing",
    "StreamMetadata": "api.event_sourcing",
    # api/state_machine.py
    "RuntimeState": "api.state_machine",
    "RuntimeStates": "api.state_machine",
    "StateMachine": "api.state_machine",
    "StateTransition": "api.state_machine",
    "create_state_machine": "api.state_machine",
    # api/metrics.py
    "Metrics": "api.metrics",
    "MetricsSnapshot": "api.metrics",
    "create_metrics": "api.metrics",
    "get_global_metrics": "api.metrics",
    "reset_global_metrics": "api.metrics",
    # api/format.py
    "Format": "api.format",
    # api/multimodal.py
    "Multimodal": "api.multimodal",
    # api/logging.py
    "enable_debug": "api.logging",
}

# Cache for imported modules
_imported: dict[str, _Any] = {}


def __getattr__(name: str) -> _Any:
    """Lazy import handler."""
    if name in _API_MODULES:
        module_path = _API_MODULES[name]
        if module_path not in _imported:
            import importlib

            _imported[module_path] = importlib.import_module(
                f".{module_path}", __name__
            )
        return getattr(_imported[module_path], name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For type checkers - these imports are not executed at runtime
if _TYPE_CHECKING:
    from .api.adapters import (
        AdaptedEvent,
        Adapter,
        Adapters,
        LiteLLMAdapter,
        OpenAIAdapter,
        OpenAIAdapterOptions,
    )
    from .api.comparison import (
        Compare,
        Difference,
        DifferenceSeverity,
        DifferenceType,
        ObjectComparisonOptions,
        StringComparisonOptions,
    )
    from .api.consensus import (
        Agreement,
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
        consensus,
    )
    from .api.continuation import (
        Continuation,
        ContinuationConfig,
        DeduplicationOptions,
        OverlapResult,
    )
    from .api.core import (
        ERROR_TYPE_DELAY_DEFAULTS,
        EXPONENTIAL_RETRY,
        MINIMAL_RETRY,
        RECOMMENDED_RETRY,
        RETRY_DEFAULTS,
        STRICT_RETRY,
        AwaitableStream,
        AwaitableStreamFactory,
        AwaitableStreamSource,
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
        LifecycleCallbacks,
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
        TimeoutError,
        WrappedClient,
        _internal_run,
        consume_stream,
        get_text,
    )
    from .api.drift import (
        Drift,
        DriftConfig,
        DriftDetector,
        DriftResult,
    )
    from .api.errors import (
        Error,
        ErrorCode,
        ErrorContext,
        FailureType,
        NetworkError,
        NetworkErrorAnalysis,
        NetworkErrorType,
        RecoveryPolicy,
        RecoveryStrategy,
    )
    from .api.event_sourcing import (
        EventEnvelope,
        EventRecorder,
        EventReplayer,
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
    from .api.events import (
        EventBus,
        ObservabilityEvent,
        ObservabilityEventType,
    )
    from .api.format import Format
    from .api.guardrails import (
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
        Violation,
        custom_pattern_rule,
        pattern_rule,
    )
    from .api.json import (
        JSON,
        AutoCorrectResult,
        CorrectionType,
    )
    from .api.json_schema import (
        JSONSchema,
        JSONSchemaAdapter,
        JSONSchemaDefinition,
        JSONSchemaValidationError,
        JSONSchemaValidationFailure,
        JSONSchemaValidationSuccess,
        SimpleJSONSchemaAdapter,
        UnifiedSchema,
    )
    from .api.logging import enable_debug
    from .api.metrics import (
        Metrics,
        MetricsSnapshot,
        create_metrics,
        get_global_metrics,
        reset_global_metrics,
    )
    from .api.monitoring import (
        Monitoring,
        OpenTelemetry,
        OpenTelemetryConfig,
        OpenTelemetryExporter,
        OpenTelemetryExporterConfig,
        SemanticAttributes,
        Sentry,
        SentryConfig,
        SentryExporter,
        SentryExporterConfig,
    )
    from .api.multimodal import Multimodal
    from .api.parallel import (
        AggregatedTelemetry,
        Parallel,
        ParallelOptions,
        ParallelResult,
        RaceResult,
        batched,
        parallel,
        race,
        sequential,
    )
    from .api.pipeline import (
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
    from .api.pool import (
        OperationPool,
        PoolOptions,
        PoolStats,
        create_pool,
    )
    from .api.state_machine import (
        RuntimeState,
        RuntimeStates,
        StateMachine,
        StateTransition,
        create_state_machine,
    )
    from .api.structured import (
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
    from .api.text import (
        NormalizeOptions,
        Text,
        WhitespaceOptions,
    )
    from .api.window import (
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
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API Functions (wrap and run)
# These are defined here to avoid circular imports and provide the main entry points
# ─────────────────────────────────────────────────────────────────────────────

from collections.abc import AsyncIterator as _AsyncIterator
from collections.abc import Callable as _Callable
from collections.abc import Coroutine as _Coroutine
from typing import Protocol as _Protocol
from typing import overload as _overload
from typing import runtime_checkable as _runtime_checkable


class _ChatCompletions(_Protocol):
    """Protocol for the completions namespace."""

    def create(self, *args: _Any, **kwargs: _Any) -> _Any: ...


class _ChatNamespace(_Protocol):
    """Protocol for the chat namespace with completions."""

    @property
    def completions(self) -> _ChatCompletions: ...


@_runtime_checkable
class _OpenAILikeClient(_Protocol):
    """Protocol matching OpenAI/LiteLLM client structure with .chat.completions."""

    @property
    def chat(self) -> _ChatNamespace: ...


@_overload
def wrap(
    client_or_stream: _OpenAILikeClient,
    *,
    guardrails: "list[GuardrailRule] | None" = None,
    retry: "Retry | None" = None,
    timeout: "Timeout | None" = None,
    adapter: _Any | str | None = None,
    on_event: "_Callable[[ObservabilityEvent], None] | None" = None,
    on_token: "_Callable[[str], None] | None" = None,
    on_tool_call: "_Callable[[str, str, dict[str, _Any]], None] | None" = None,
    on_violation: "_Callable[[GuardrailViolation], None] | None" = None,
    context: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: "_Callable[[str], str] | None" = None,
) -> "WrappedClient": ...


@_overload
def wrap(
    client_or_stream: _AsyncIterator[_Any],
    *,
    guardrails: "list[GuardrailRule] | None" = None,
    retry: "Retry | None" = None,
    timeout: "Timeout | None" = None,
    adapter: _Any | str | None = None,
    on_event: "_Callable[[ObservabilityEvent], None] | None" = None,
    on_token: "_Callable[[str], None] | None" = None,
    on_tool_call: "_Callable[[str, str, dict[str, _Any]], None] | None" = None,
    on_violation: "_Callable[[GuardrailViolation], None] | None" = None,
    context: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: "_Callable[[str], str] | None" = None,
) -> "LazyStream[_Any]": ...


@_overload
def wrap(
    client_or_stream: _Coroutine[_Any, _Any, _AsyncIterator[_Any]],
    *,
    guardrails: "list[GuardrailRule] | None" = None,
    retry: "Retry | None" = None,
    timeout: "Timeout | None" = None,
    adapter: _Any | str | None = None,
    on_event: "_Callable[[ObservabilityEvent], None] | None" = None,
    on_token: "_Callable[[str], None] | None" = None,
    on_tool_call: "_Callable[[str, str, dict[str, _Any]], None] | None" = None,
    on_violation: "_Callable[[GuardrailViolation], None] | None" = None,
    context: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: "_Callable[[str], str] | None" = None,
) -> "LazyStream[_Any]": ...


def wrap(
    client_or_stream: _Any,
    *,
    guardrails: "list[GuardrailRule] | None" = None,
    retry: "Retry | None" = None,
    timeout: "Timeout | None" = None,
    adapter: _Any | str | None = None,
    on_event: "_Callable[[ObservabilityEvent], None] | None" = None,
    on_token: "_Callable[[str], None] | None" = None,
    on_tool_call: "_Callable[[str, str, dict[str, _Any]], None] | None" = None,
    on_violation: "_Callable[[GuardrailViolation], None] | None" = None,
    context: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: "_Callable[[str], str] | None" = None,
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
        on_token: Callback for each token received (text: str)
        on_tool_call: Callback for tool calls (name: str, id: str, args: dict)
        on_violation: Callback for guardrail violations
        context: User context attached to all events (request_id, tenant, etc.)
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
    # Import lazily to avoid loading everything
    from .client import wrap_client
    from .types import LazyStream

    if hasattr(client_or_stream, "chat") and hasattr(
        client_or_stream.chat, "completions"
    ):
        return wrap_client(
            client_or_stream,
            guardrails=guardrails,
            retry=retry,
            timeout=timeout,
            adapter=adapter,
            on_event=on_event,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_violation=on_violation,
            context=context,
            buffer_tool_calls=buffer_tool_calls,
            continue_from_last_good_token=continue_from_last_good_token,
            build_continuation_prompt=build_continuation_prompt,
        )
    else:
        return LazyStream(
            stream=client_or_stream,
            guardrails=guardrails,
            timeout=timeout,
            adapter=adapter,
            on_event=on_event,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_violation=on_violation,
            context=context,
            buffer_tool_calls=buffer_tool_calls,
        )


async def run(
    stream: "AwaitableStreamFactory",
    *,
    fallbacks: "list[AwaitableStreamFactory] | None" = None,
    guardrails: "list[GuardrailRule] | None" = None,
    drift_detector: "DriftDetector | None" = None,
    retry: "Retry | None" = None,
    timeout: "Timeout | None" = None,
    check_intervals: "CheckIntervals | None" = None,
    adapter: "Adapter | str | None" = None,
    on_event: "_Callable[[ObservabilityEvent], None] | None" = None,
    context: dict[str, _Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: "ContinuationConfig | bool" = False,
    build_continuation_prompt: "_Callable[[str], str] | None" = None,
    callbacks: "LifecycleCallbacks | None" = None,
    on_start: "_Callable[[int, bool, bool], None] | None" = None,
    on_complete: "_Callable[[State], None] | None" = None,
    on_error: "_Callable[[Exception, bool, bool], None] | None" = None,
    on_stream_event: "_Callable[[Event], None] | None" = None,
    on_violation: "_Callable[[GuardrailViolation], None] | None" = None,
    on_retry: "_Callable[[int, str], None] | None" = None,
    on_fallback: "_Callable[[int, str], None] | None" = None,
    on_resume: "_Callable[[str, int], None] | None" = None,
    on_checkpoint: "_Callable[[str, int], None] | None" = None,
    on_timeout: "_Callable[[str, float], None] | None" = None,
    on_abort: "_Callable[[int, int], None] | None" = None,
    on_drift: "_Callable[[list[str], float | None], None] | None" = None,
    on_tool_call: "_Callable[[str, str, dict[str, _Any]], None] | None" = None,
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
        context: Optional user context attached to all events (request_id, tenant, etc.)
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
        )

        async for event in result:
            if event.is_token:
                print(event.text, end="")
        ```
    """
    from .runtime import _internal_run

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
        context=context,
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
    "wrap",
    "run",
    "l0",
    # Types
    "Stream",
    "LazyStream",
    "WrappedClient",
    "Event",
    "State",
    "EventType",
    "AwaitableStream",
    "AwaitableStreamFactory",
    "AwaitableStreamSource",
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
    "Error",
    "ErrorCode",
    "ErrorContext",
    "FailureType",
    "RecoveryStrategy",
    "RecoveryPolicy",
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
    # Adapters
    "Adapters",
    "Adapter",
    "AdaptedEvent",
    "OpenAIAdapter",
    "OpenAIAdapterOptions",
    "LiteLLMAdapter",
    # Guardrails
    "Guardrails",
    "GuardrailRule",
    "GuardrailViolation",
    "Violation",
    "JsonAnalysis",
    "MarkdownAnalysis",
    "LatexAnalysis",
    "custom_pattern_rule",
    "pattern_rule",
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
    # JSON
    "JSON",
    "AutoCorrectResult",
    "CorrectionType",
    # Parallel
    "Parallel",
    "ParallelResult",
    "ParallelOptions",
    "RaceResult",
    "AggregatedTelemetry",
    "parallel",
    "race",
    "sequential",
    "batched",
    # Pool
    "OperationPool",
    "PoolOptions",
    "PoolStats",
    "create_pool",
    # Pipeline
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
    # Consensus
    "Consensus",
    "consensus",
    "ConsensusResult",
    "ConsensusOutput",
    "ConsensusAnalysis",
    "ConsensusPreset",
    "Agreement",
    "Disagreement",
    "DisagreementValue",
    "FieldAgreement",
    "FieldConsensus",
    "FieldConsensusInfo",
    # Window
    "Window",
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
    # Monitoring
    "Monitoring",
    "OpenTelemetry",
    "OpenTelemetryConfig",
    "OpenTelemetryExporter",
    "OpenTelemetryExporterConfig",
    "SemanticAttributes",
    "Sentry",
    "SentryConfig",
    "SentryExporter",
    "SentryExporterConfig",
    # Formatting
    "Format",
    # JSON Schema
    "JSONSchema",
    "JSONSchemaAdapter",
    "JSONSchemaDefinition",
    "JSONSchemaValidationError",
    "JSONSchemaValidationFailure",
    "JSONSchemaValidationSuccess",
    "SimpleJSONSchemaAdapter",
    "UnifiedSchema",
    # Multimodal
    "Multimodal",
    "ContentType",
    "DataPayload",
    "Progress",
    # Continuation
    "Continuation",
    "ContinuationConfig",
    "DeduplicationOptions",
    "OverlapResult",
    # Drift
    "Drift",
    "DriftDetector",
    "DriftConfig",
    "DriftResult",
    # State machine
    "StateMachine",
    "RuntimeState",
    "RuntimeStates",
    "StateTransition",
    "create_state_machine",
    # Event Sourcing
    "EventSourcing",
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
    # Text
    "Text",
    "NormalizeOptions",
    "WhitespaceOptions",
    # Comparison
    "Compare",
    "Difference",
    "DifferenceSeverity",
    "DifferenceType",
    "StringComparisonOptions",
    "ObjectComparisonOptions",
]
