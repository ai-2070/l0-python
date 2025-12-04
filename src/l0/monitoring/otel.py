"""OpenTelemetry integration for L0 monitoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .telemetry import Telemetry

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Span, Tracer


class OpenTelemetryConfig(BaseModel):
    """OpenTelemetry configuration.

    Usage:
        ```python
        from l0.monitoring import OpenTelemetryConfig, OpenTelemetryExporter

        config = OpenTelemetryConfig(
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
    def from_env(cls) -> OpenTelemetryConfig:
        """Create config from environment variables.

        Reads:
            - OTEL_SERVICE_NAME
            - OTEL_EXPORTER_OTLP_ENDPOINT
            - OTEL_EXPORTER_OTLP_HEADERS
            - OTEL_EXPORTER_OTLP_INSECURE

        Returns:
            OpenTelemetryConfig from environment
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


class OpenTelemetryExporter:
    """Export L0 telemetry to OpenTelemetry.

    Usage:
        ```python
        from l0.monitoring import OpenTelemetryConfig, OpenTelemetryExporter

        config = OpenTelemetryConfig(
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

    def __init__(self, config: OpenTelemetryConfig) -> None:
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
                "Install with: pip install l0[observability]"
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

    def create_span(self, name: str, **attributes: Any) -> Span | None:
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
