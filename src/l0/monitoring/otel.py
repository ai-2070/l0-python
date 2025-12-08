"""OpenTelemetry integration for L0 monitoring.

This module provides OpenTelemetry integration for distributed tracing and metrics.
It follows the same patterns as the TypeScript implementation for API parity.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..events import ObservabilityEvent, ObservabilityEventType
from .telemetry import Telemetry

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter, UpDownCounter
    from opentelemetry.trace import Span, SpanContext, Tracer


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Attributes (matches TS SemanticAttributes)
# ─────────────────────────────────────────────────────────────────────────────


class SemanticAttributes:
    """Semantic convention attribute names for LLM operations.

    Following OpenTelemetry semantic conventions for GenAI.
    """

    # General LLM attributes
    LLM_SYSTEM = "gen_ai.system"
    LLM_REQUEST_MODEL = "gen_ai.request.model"
    LLM_RESPONSE_MODEL = "gen_ai.response.model"
    LLM_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    LLM_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    LLM_REQUEST_TOP_P = "gen_ai.request.top_p"
    LLM_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    LLM_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    LLM_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # L0-specific attributes
    L0_SESSION_ID = "l0.session_id"
    L0_STREAM_COMPLETED = "l0.stream.completed"
    L0_FALLBACK_INDEX = "l0.fallback.index"
    L0_RETRY_COUNT = "l0.retry.count"
    L0_NETWORK_ERROR_COUNT = "l0.network.error_count"
    L0_GUARDRAIL_VIOLATION_COUNT = "l0.guardrail.violation_count"
    L0_DRIFT_DETECTED = "l0.drift.detected"
    L0_TIME_TO_FIRST_TOKEN = "l0.time_to_first_token_ms"
    L0_TOKENS_PER_SECOND = "l0.tokens_per_second"


# ─────────────────────────────────────────────────────────────────────────────
# Span Protocol and NoOp Span
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class SpanProtocol(Protocol):
    """Protocol for OpenTelemetry Span compatibility."""

    def set_attribute(self, key: str, value: Any) -> "SpanProtocol": ...
    def set_attributes(self, attributes: dict[str, Any]) -> "SpanProtocol": ...
    def add_event(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> "SpanProtocol": ...
    def set_status(
        self, status: Any, description: str | None = None
    ) -> "SpanProtocol": ...
    def record_exception(self, exception: BaseException) -> None: ...
    def end(self) -> None: ...
    def is_recording(self) -> bool: ...


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def span_context(self) -> dict[str, Any]:
        return {"trace_id": "", "span_id": "", "trace_flags": 0}

    def set_attribute(self, key: str, value: Any) -> "NoOpSpan":
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "NoOpSpan":
        return self

    def add_event(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> "NoOpSpan":
        return self

    def add_link(self, link: Any) -> "NoOpSpan":
        return self

    def set_status(self, status: Any, description: str | None = None) -> "NoOpSpan":
        return self

    def update_name(self, name: str) -> "NoOpSpan":
        return self

    def record_exception(self, exception: BaseException) -> None:
        pass

    def end(self) -> None:
        pass

    def is_recording(self) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# OpenTelemetry Configuration
# ─────────────────────────────────────────────────────────────────────────────


class OpenTelemetryExporterConfig(BaseModel):
    """OpenTelemetry exporter configuration.

    Usage:
        ```python
        from l0.monitoring import OpenTelemetryExporterConfig, OpenTelemetryExporter

        config = OpenTelemetryExporterConfig(
            service_name="my-llm-app",
            endpoint="http://localhost:4317",
        )

        exporter = OpenTelemetryExporter(config)
        exporter.export(telemetry)
        ```

    Attributes:
        service_name: Service name for traces and metrics
        endpoint: OTLP endpoint URL
        headers: Additional headers for OTLP requests
        insecure: Use insecure connection (no TLS)
        timeout: Request timeout in seconds
        resource_attributes: Additional resource attributes
        enabled: Enable/disable OpenTelemetry export
        trace_enabled: Enable trace export
        metrics_enabled: Enable metrics export
        batch_export: Use batch export (vs immediate)
        export_interval: Batch export interval in seconds
    """

    service_name: str = "l0"
    endpoint: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    insecure: bool = False
    timeout: float = Field(default=30.0, ge=1.0)
    resource_attributes: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    trace_enabled: bool = True
    metrics_enabled: bool = True
    batch_export: bool = True
    export_interval: float = Field(default=5.0, ge=1.0)

    @classmethod
    def from_env(cls) -> OpenTelemetryExporterConfig:
        """Create config from environment variables.

        Reads:
            - OTEL_SERVICE_NAME
            - OTEL_EXPORTER_OTLP_ENDPOINT
            - OTEL_EXPORTER_OTLP_HEADERS
            - OTEL_EXPORTER_OTLP_INSECURE

        Returns:
            OpenTelemetryExporterConfig from environment
        """
        import os

        headers = {}
        headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        if headers_str:
            for pair in headers_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers[key.strip()] = value.strip()

        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "l0"),
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            headers=headers,
            insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "").lower() == "true",
        )


class OpenTelemetryConfig(BaseModel):
    """Configuration for OpenTelemetry class.

    This is for runtime configuration, separate from export configuration.
    """

    service_name: str = "l0"
    trace_tokens: bool = False
    """Whether to create span events for individual tokens (can be noisy)."""

    record_token_content: bool = False
    """Whether to record token content in spans (privacy consideration)."""

    record_guardrail_violations: bool = True
    """Whether to record guardrail violations as span events."""

    default_attributes: dict[str, Any] = Field(default_factory=dict)
    """Custom attributes to add to all spans."""


# ─────────────────────────────────────────────────────────────────────────────
# OpenTelemetry Class (matches TS L0OpenTelemetry)
# ─────────────────────────────────────────────────────────────────────────────


class OpenTelemetry:
    """OpenTelemetry integration for distributed tracing and metrics.

    This class provides the same API as the TypeScript L0OpenTelemetry class.

    Example:
        ```python
        from opentelemetry import trace, metrics
        from l0.monitoring import OpenTelemetry

        otel = OpenTelemetry(
            tracer=trace.get_tracer("l0"),
            meter=metrics.get_meter("l0"),
        )

        # Trace a stream operation
        result = await otel.trace_stream(
            "chat-completion",
            lambda span: l0(stream=lambda: stream_text(model, prompt)),
        )
        ```
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        meter: Meter | None = None,
        config: OpenTelemetryConfig | None = None,
    ) -> None:
        """Initialize OpenTelemetry.

        Args:
            tracer: OpenTelemetry tracer instance
            meter: OpenTelemetry meter instance
            config: Configuration options
        """
        self._tracer = tracer
        self._meter = meter
        self._config = config or OpenTelemetryConfig()

        # Metrics instruments
        self._request_counter: Counter | None = None
        self._token_counter: Counter | None = None
        self._retry_counter: Counter | None = None
        self._error_counter: Counter | None = None
        self._duration_histogram: Histogram | None = None
        self._ttft_histogram: Histogram | None = None
        self._active_streams_gauge: UpDownCounter | None = None

        self._active_streams = 0
        self._metrics_initialized = False

        if self._meter:
            self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize OpenTelemetry metrics instruments."""
        if not self._meter or self._metrics_initialized:
            return

        self._request_counter = self._meter.create_counter(
            "l0.requests",
            description="Total number of L0 stream requests",
            unit="1",
        )

        self._token_counter = self._meter.create_counter(
            "l0.tokens",
            description="Total number of tokens processed",
            unit="1",
        )

        self._retry_counter = self._meter.create_counter(
            "l0.retries",
            description="Total number of retry attempts",
            unit="1",
        )

        self._error_counter = self._meter.create_counter(
            "l0.errors",
            description="Total number of errors",
            unit="1",
        )

        self._duration_histogram = self._meter.create_histogram(
            "l0.duration",
            description="Stream duration in milliseconds",
            unit="ms",
        )

        self._ttft_histogram = self._meter.create_histogram(
            "l0.time_to_first_token",
            description="Time to first token in milliseconds",
            unit="ms",
        )

        self._active_streams_gauge = self._meter.create_up_down_counter(
            "l0.active_streams",
            description="Number of currently active streams",
            unit="1",
        )

        self._metrics_initialized = True

    async def trace_stream(
        self,
        name: str,
        fn: Callable[..., Any],
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Trace an L0 stream operation.

        Args:
            name: Span name
            fn: Async function that returns an L0 result
            attributes: Additional span attributes

        Returns:
            Result from fn
        """
        if not self._tracer:
            return await fn(NoOpSpan())

        from opentelemetry.trace import SpanKind, StatusCode

        span_attributes = {
            **self._config.default_attributes,
            **(attributes or {}),
        }

        span = self._tracer.start_span(
            f"{self._config.service_name}.{name}",
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
        )

        self._active_streams += 1
        if self._active_streams_gauge:
            self._active_streams_gauge.add(1)

        try:
            result = await fn(span)
            span.set_status(StatusCode.OK)
            return result
        except Exception as error:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
            if self._error_counter:
                self._error_counter.add(1, {"type": "stream_error"})
            raise
        finally:
            self._active_streams -= 1
            if self._active_streams_gauge:
                self._active_streams_gauge.add(-1)
            span.end()

    def record_telemetry(self, telemetry: Telemetry, span: Any | None = None) -> None:
        """Record telemetry from a completed L0 operation.

        This is the primary method for recording metrics. All metric counters
        are updated here using the aggregated data to ensure accurate counting.

        Args:
            telemetry: L0 telemetry data
            span: Optional span to add attributes to
        """
        attributes: dict[str, Any] = {
            SemanticAttributes.L0_SESSION_ID: telemetry.session_id
            or telemetry.stream_id,
        }

        # Record request completion
        if self._request_counter:
            self._request_counter.add(1, {"status": "completed"})

        # Record tokens
        if self._token_counter and telemetry.metrics.token_count > 0:
            self._token_counter.add(telemetry.metrics.token_count, attributes)

        # Record retries
        if self._retry_counter and telemetry.retries.total_retries > 0:
            self._retry_counter.add(
                telemetry.retries.total_retries,
                {**attributes, "type": "total"},
            )

        # Record network retries separately
        if self._retry_counter and telemetry.retries.network_retries > 0:
            self._retry_counter.add(
                telemetry.retries.network_retries,
                {**attributes, "type": "network"},
            )

        # Record model retries separately
        if self._retry_counter and telemetry.retries.model_retries > 0:
            self._retry_counter.add(
                telemetry.retries.model_retries,
                {**attributes, "type": "model"},
            )

        # Record errors
        if self._error_counter and telemetry.error.occurred:
            error_attrs = {**attributes, "type": "error"}
            if telemetry.error.category:
                error_attrs["category"] = telemetry.error.category.value
            self._error_counter.add(1, error_attrs)

        # Record guardrail violations
        if self._error_counter and telemetry.guardrails.violations:
            for violation in telemetry.guardrails.violations:
                self._error_counter.add(
                    1,
                    {
                        **attributes,
                        "type": "guardrail_violation",
                        "rule": violation.get("rule", "unknown"),
                        "severity": violation.get("severity", "unknown"),
                    },
                )

        # Record duration
        if self._duration_histogram and telemetry.timing.duration is not None:
            self._duration_histogram.record(
                telemetry.timing.duration * 1000,  # Convert to ms
                attributes,
            )

        # Record time to first token
        if self._ttft_histogram and telemetry.metrics.time_to_first_token is not None:
            self._ttft_histogram.record(
                telemetry.metrics.time_to_first_token * 1000,  # Convert to ms
                attributes,
            )

        # Add span attributes if span is recording
        if span and hasattr(span, "is_recording") and span.is_recording():
            span.set_attributes(
                {
                    SemanticAttributes.L0_SESSION_ID: telemetry.session_id
                    or telemetry.stream_id,
                    SemanticAttributes.LLM_USAGE_OUTPUT_TOKENS: telemetry.metrics.token_count,
                    SemanticAttributes.L0_RETRY_COUNT: telemetry.retries.total_retries,
                    SemanticAttributes.L0_NETWORK_ERROR_COUNT: telemetry.retries.network_retries,
                }
            )

            if telemetry.guardrails.violations:
                span.set_attribute(
                    SemanticAttributes.L0_GUARDRAIL_VIOLATION_COUNT,
                    len(telemetry.guardrails.violations),
                )

            if not telemetry.guardrails.passed:
                span.set_attribute(SemanticAttributes.L0_DRIFT_DETECTED, True)

            if telemetry.metrics.time_to_first_token is not None:
                span.set_attribute(
                    SemanticAttributes.L0_TIME_TO_FIRST_TOKEN,
                    telemetry.metrics.time_to_first_token * 1000,
                )

            if telemetry.metrics.tokens_per_second is not None:
                span.set_attribute(
                    SemanticAttributes.L0_TOKENS_PER_SECOND,
                    telemetry.metrics.tokens_per_second,
                )

            if telemetry.timing.duration is not None:
                span.set_attribute("duration_ms", telemetry.timing.duration * 1000)

    def record_token(self, span: Any | None = None, content: str | None = None) -> None:
        """Record a token event (span event only).

        Note: Metric counters are updated via record_telemetry() to avoid double-counting.

        Args:
            span: Span to add event to
            content: Token content (only recorded if record_token_content is True)
        """
        if not self._config.trace_tokens:
            return

        if span and hasattr(span, "is_recording") and span.is_recording():
            event_attributes: dict[str, Any] = {}
            if self._config.record_token_content and content:
                event_attributes["token.content"] = content
            span.add_event("token", event_attributes)

    def record_retry(
        self,
        reason: str,
        attempt: int,
        span: Any | None = None,
    ) -> None:
        """Record a retry attempt (span event only).

        Note: Metric counters are updated via record_telemetry() to avoid double-counting.

        Args:
            reason: Reason for retry
            attempt: Retry attempt number
            span: Span to add event to
        """
        if span and hasattr(span, "is_recording") and span.is_recording():
            span.add_event(
                "retry",
                {
                    "retry.reason": reason,
                    "retry.attempt": attempt,
                },
            )

    def record_network_error(
        self,
        error: Exception,
        error_type: str,
        span: Any | None = None,
    ) -> None:
        """Record a network error (span event only).

        Note: Metric counters are updated via record_telemetry() to avoid double-counting.

        Args:
            error: The error that occurred
            error_type: Type of network error
            span: Span to add event to
        """
        if span and hasattr(span, "is_recording") and span.is_recording():
            span.add_event(
                "network_error",
                {
                    "error.type": error_type,
                    "error.message": str(error),
                },
            )

    def record_guardrail_violation(
        self,
        violation: dict[str, Any],
        span: Any | None = None,
    ) -> None:
        """Record a guardrail violation (span event only).

        Note: Metric counters are updated via record_telemetry() to avoid double-counting.

        Args:
            violation: Guardrail violation details
            span: Span to add event to
        """
        if not self._config.record_guardrail_violations:
            return

        if span and hasattr(span, "is_recording") and span.is_recording():
            span.add_event(
                "guardrail_violation",
                {
                    "guardrail.rule": violation.get("rule", "unknown"),
                    "guardrail.severity": violation.get("severity", "unknown"),
                    "guardrail.message": violation.get("message", ""),
                },
            )

    def record_drift(
        self,
        drift_type: str,
        confidence: float,
        span: Any | None = None,
    ) -> None:
        """Record drift detection (span event only).

        Note: Metric counters are updated via record_telemetry() to avoid double-counting.

        Args:
            drift_type: Type of drift detected
            confidence: Confidence score
            span: Span to add event to
        """
        if span and hasattr(span, "is_recording") and span.is_recording():
            span.set_attribute(SemanticAttributes.L0_DRIFT_DETECTED, True)
            span.add_event(
                "drift_detected",
                {
                    "drift.type": drift_type,
                    "drift.confidence": confidence,
                },
            )

    def create_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Create a child span for a sub-operation.

        Args:
            name: Span name
            attributes: Span attributes

        Returns:
            Span instance (or NoOpSpan if tracing disabled)
        """
        if not self._tracer:
            return NoOpSpan()

        from opentelemetry.trace import SpanKind

        return self._tracer.start_span(
            f"{self._config.service_name}.{name}",
            kind=SpanKind.INTERNAL,
            attributes={
                **self._config.default_attributes,
                **(attributes or {}),
            },
        )

    def get_active_streams(self) -> int:
        """Get current active stream count."""
        return self._active_streams


# ─────────────────────────────────────────────────────────────────────────────
# Event Handler Factory (matches TS createOpenTelemetryHandler)
# ─────────────────────────────────────────────────────────────────────────────


def create_opentelemetry_handler(
    tracer: Tracer | None = None,
    meter: Meter | None = None,
    config: OpenTelemetryConfig | None = None,
) -> Callable[[ObservabilityEvent], None]:
    """Create an OpenTelemetry event handler for L0 observability.

    This is the recommended way to integrate OpenTelemetry with L0.
    The handler subscribes to L0 events and records traces/metrics.

    Example:
        ```python
        from opentelemetry import trace, metrics
        from l0 import l0
        from l0.monitoring import create_opentelemetry_handler, combine_events

        result = await l0(
            stream=lambda: stream_text(model, prompt),
            on_event=create_opentelemetry_handler(
                tracer=trace.get_tracer("my-app"),
                meter=metrics.get_meter("my-app"),
            ),
        )

        # Or combine with other handlers:
        result = await l0(
            stream=lambda: stream_text(model, prompt),
            on_event=combine_events(
                create_opentelemetry_handler(tracer=tracer, meter=meter),
                create_sentry_handler(sentry=sentry),
            ),
        )
        ```

    Args:
        tracer: OpenTelemetry tracer instance
        meter: OpenTelemetry meter instance
        config: Configuration options

    Returns:
        Event handler function
    """
    otel = OpenTelemetry(tracer=tracer, meter=meter, config=config)
    current_span: Any = None

    def handler(event: ObservabilityEvent) -> None:
        nonlocal current_span

        event_type = event.type
        meta = event.meta

        if event_type == ObservabilityEventType.SESSION_START:
            # Start a new span for the session
            current_span = otel.create_span("stream")
            if hasattr(current_span, "set_attribute"):
                current_span.set_attribute("l0.attempt", meta.get("attempt", 1))
                current_span.set_attribute("l0.is_retry", meta.get("is_retry", False))
                current_span.set_attribute(
                    "l0.is_fallback", meta.get("is_fallback", False)
                )

        elif event_type == ObservabilityEventType.RETRY_ATTEMPT:
            otel.record_retry(
                reason=meta.get("reason", "unknown"),
                attempt=meta.get("attempt", 1),
                span=current_span,
            )

        elif event_type == ObservabilityEventType.ERROR:
            error_msg = meta.get("error", "Unknown error")
            otel.record_network_error(
                error=Exception(error_msg),
                error_type=meta.get("failure_type", "unknown"),
                span=current_span,
            )

        elif event_type == ObservabilityEventType.GUARDRAIL_RULE_RESULT:
            violation = meta.get("violation")
            if violation:
                otel.record_guardrail_violation(violation, span=current_span)

        elif event_type == ObservabilityEventType.DRIFT_CHECK_RESULT:
            if meta.get("detected"):
                metrics = meta.get("metrics", {})
                drift_types = list(metrics.keys()) if metrics else []
                otel.record_drift(
                    drift_type=",".join(drift_types) if drift_types else "unknown",
                    confidence=meta.get("score", 0.0),
                    span=current_span,
                )

        elif event_type == ObservabilityEventType.TOKEN:
            otel.record_token(
                span=current_span,
                content=meta.get("token") or meta.get("content"),
            )

        elif event_type == ObservabilityEventType.COMPLETE:
            if current_span and hasattr(current_span, "set_attribute"):
                current_span.set_attribute("l0.token_count", meta.get("token_count", 0))
                current_span.set_attribute(
                    "l0.content_length", meta.get("content_length", 0)
                )
                current_span.set_attribute("l0.duration_ms", meta.get("duration_ms", 0))

                from opentelemetry.trace import StatusCode

                current_span.set_status(StatusCode.OK)
                current_span.end()
                current_span = None

    return handler


# ─────────────────────────────────────────────────────────────────────────────
# Async Context Manager for Tracing
# ─────────────────────────────────────────────────────────────────────────────


async def with_opentelemetry(
    tracer: Tracer | None,
    meter: Meter | None,
    fn: Callable[..., Any],
    name: str = "l0.stream",
    config: OpenTelemetryConfig | None = None,
) -> Any:
    """Execute a function with OpenTelemetry tracing.

    Example:
        ```python
        from opentelemetry import trace, metrics
        from l0.monitoring import with_opentelemetry

        result = await with_opentelemetry(
            tracer=trace.get_tracer("my-app"),
            meter=metrics.get_meter("my-app"),
            fn=lambda span: l0(stream=lambda: stream_text(model, prompt)),
            name="chat-completion",
        )
        ```

    Args:
        tracer: OpenTelemetry tracer instance
        meter: OpenTelemetry meter instance
        fn: Async function to execute
        name: Span name
        config: Configuration options

    Returns:
        Result from fn
    """
    otel = OpenTelemetry(tracer=tracer, meter=meter, config=config)
    return await otel.trace_stream(name, fn)


# ─────────────────────────────────────────────────────────────────────────────
# OpenTelemetry Exporter (existing implementation)
# ─────────────────────────────────────────────────────────────────────────────


class OpenTelemetryExporter:
    """Export L0 telemetry to OpenTelemetry.

    Usage:
        ```python
        from l0.monitoring import OpenTelemetryExporterConfig, OpenTelemetryExporter

        config = OpenTelemetryExporterConfig(
            service_name="my-llm-app",
            endpoint="http://localhost:4317",
        )

        exporter = OpenTelemetryExporter(config)

        # Export telemetry
        exporter.export(telemetry)

        # Or use as callback
        monitor = Monitor(on_complete=exporter.export)
        ```

    Requires:
        pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
    """

    def __init__(self, config: OpenTelemetryExporterConfig) -> None:
        """Initialize OpenTelemetry exporter.

        Args:
            config: OpenTelemetry configuration
        """
        self.config = config
        self._tracer: Tracer | None = None
        self._meter: Meter | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize OpenTelemetry components."""
        if self._initialized:
            return

        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError as e:
            raise ImportError(
                "OpenTelemetry packages not installed. "
                "Install with: pip install ai2070-l0[observability]"
            ) from e

        # Build resource
        resource_attrs = {
            "service.name": self.config.service_name,
            **self.config.resource_attributes,
        }
        resource = Resource.create(resource_attrs)

        # Set up tracer if enabled
        if self.config.trace_enabled:
            tracer_provider = TracerProvider(resource=resource)

            if self.config.endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                span_exporter = OTLPSpanExporter(
                    endpoint=self.config.endpoint,
                    headers=self.config.headers or None,
                    insecure=self.config.insecure,
                    timeout=int(self.config.timeout),
                )
                tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

            trace.set_tracer_provider(tracer_provider)
            self._tracer = trace.get_tracer(self.config.service_name)

        # Set up meter if enabled
        if self.config.metrics_enabled:
            meter_provider = MeterProvider(resource=resource)

            if self.config.endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.sdk.metrics.export import (
                    PeriodicExportingMetricReader,
                )

                metric_exporter = OTLPMetricExporter(
                    endpoint=self.config.endpoint,
                    headers=self.config.headers or None,
                    insecure=self.config.insecure,
                    timeout=int(self.config.timeout),
                )
                reader = PeriodicExportingMetricReader(
                    metric_exporter,
                    export_interval_millis=int(self.config.export_interval * 1000),
                )
                meter_provider = MeterProvider(
                    resource=resource, metric_readers=[reader]
                )

            metrics.set_meter_provider(meter_provider)
            self._meter = metrics.get_meter(self.config.service_name)

        self._initialized = True

    def export(self, telemetry: Telemetry) -> None:
        """Export telemetry to OpenTelemetry.

        Args:
            telemetry: Telemetry data to export
        """
        if not self.config.enabled:
            return

        self._ensure_initialized()

        if self._tracer and self.config.trace_enabled:
            self._export_trace(telemetry)

        if self._meter and self.config.metrics_enabled:
            self._export_metrics(telemetry)

    def _export_trace(self, telemetry: Telemetry) -> None:
        """Export telemetry as a trace span."""
        if not self._tracer:
            return

        from opentelemetry import trace
        from opentelemetry.trace import StatusCode

        # Create span with timing
        with self._tracer.start_as_current_span(
            name="l0.stream",
            kind=trace.SpanKind.CLIENT,
        ) as span:
            # Set attributes
            span.set_attribute("l0.stream_id", telemetry.stream_id)
            if telemetry.session_id:
                span.set_attribute("l0.session_id", telemetry.session_id)
            if telemetry.model:
                span.set_attribute("l0.model", telemetry.model)

            # Timing attributes
            if telemetry.timing.duration is not None:
                span.set_attribute("l0.duration_ms", telemetry.timing.duration * 1000)
            if telemetry.metrics.time_to_first_token is not None:
                span.set_attribute(
                    "l0.ttft_ms", telemetry.metrics.time_to_first_token * 1000
                )

            # Token attributes
            span.set_attribute("l0.token_count", telemetry.metrics.token_count)
            if telemetry.metrics.tokens_per_second is not None:
                span.set_attribute(
                    "l0.tokens_per_second", telemetry.metrics.tokens_per_second
                )

            # Retry attributes
            span.set_attribute("l0.retries.total", telemetry.retries.total_retries)
            span.set_attribute("l0.retries.model", telemetry.retries.model_retries)
            span.set_attribute("l0.retries.network", telemetry.retries.network_retries)

            # Guardrail attributes
            span.set_attribute(
                "l0.guardrails.checked", telemetry.guardrails.rules_checked
            )
            span.set_attribute(
                "l0.guardrails.violations", len(telemetry.guardrails.violations)
            )
            span.set_attribute("l0.guardrails.passed", telemetry.guardrails.passed)

            # Status
            if telemetry.error.occurred:
                span.set_status(
                    StatusCode.ERROR, telemetry.error.message or "Unknown error"
                )
                if telemetry.error.category:
                    span.set_attribute(
                        "l0.error.category", telemetry.error.category.value
                    )
                if telemetry.error.code:
                    span.set_attribute("l0.error.code", telemetry.error.code)
            elif telemetry.aborted:
                span.set_status(StatusCode.OK, "Aborted")
                span.set_attribute("l0.aborted", True)
            elif telemetry.completed:
                span.set_status(StatusCode.OK)
            else:
                span.set_status(StatusCode.UNSET)

    def _export_metrics(self, telemetry: Telemetry) -> None:
        """Export telemetry as metrics."""
        if not self._meter:
            return

        # Create instruments (cached after first call)
        if not hasattr(self, "_instruments"):
            self._instruments = {
                "token_count": self._meter.create_counter(
                    "l0.tokens",
                    description="Total tokens generated",
                    unit="tokens",
                ),
                "duration": self._meter.create_histogram(
                    "l0.duration",
                    description="Stream duration",
                    unit="s",
                ),
                "ttft": self._meter.create_histogram(
                    "l0.ttft",
                    description="Time to first token",
                    unit="s",
                ),
                "tokens_per_second": self._meter.create_histogram(
                    "l0.tokens_per_second",
                    description="Token generation rate",
                    unit="tokens/s",
                ),
                "retries": self._meter.create_counter(
                    "l0.retries",
                    description="Total retries",
                    unit="retries",
                ),
                "errors": self._meter.create_counter(
                    "l0.errors",
                    description="Total errors",
                    unit="errors",
                ),
                "guardrail_violations": self._meter.create_counter(
                    "l0.guardrail_violations",
                    description="Guardrail violations",
                    unit="violations",
                ),
            }

        # Build labels
        labels: dict[str, str] = {}
        if telemetry.model:
            labels["model"] = telemetry.model
        if telemetry.session_id:
            labels["session_id"] = telemetry.session_id

        # Record metrics
        self._instruments["token_count"].add(telemetry.metrics.token_count, labels)

        if telemetry.timing.duration is not None:
            self._instruments["duration"].record(telemetry.timing.duration, labels)

        if telemetry.metrics.time_to_first_token is not None:
            self._instruments["ttft"].record(
                telemetry.metrics.time_to_first_token, labels
            )

        if telemetry.metrics.tokens_per_second is not None:
            self._instruments["tokens_per_second"].record(
                telemetry.metrics.tokens_per_second, labels
            )

        if telemetry.retries.total_retries > 0:
            self._instruments["retries"].add(telemetry.retries.total_retries, labels)

        if telemetry.error.occurred:
            error_labels = {**labels}
            if telemetry.error.category:
                error_labels["category"] = telemetry.error.category.value
            self._instruments["errors"].add(1, error_labels)

        if telemetry.guardrails.violations:
            self._instruments["guardrail_violations"].add(
                len(telemetry.guardrails.violations), labels
            )

    def create_span(self, name: str, **attributes: Any) -> Any:
        """Create a custom span for manual instrumentation.

        Args:
            name: Span name
            **attributes: Span attributes

        Returns:
            Span context manager, or None if tracing disabled
        """
        if not self.config.enabled or not self.config.trace_enabled:
            return None

        self._ensure_initialized()

        if not self._tracer:
            return None

        return self._tracer.start_as_current_span(name, attributes=attributes)

    def shutdown(self) -> None:
        """Shutdown OpenTelemetry providers."""
        if not self._initialized:
            return

        try:
            from opentelemetry import metrics, trace

            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "shutdown"):
                tracer_provider.shutdown()

            meter_provider = metrics.get_meter_provider()
            if hasattr(meter_provider, "shutdown"):
                meter_provider.shutdown()
        except Exception:
            pass
