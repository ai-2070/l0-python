"""L0 Monitoring & Telemetry.

Production-ready observability with OpenTelemetry and Sentry support.

Usage:
    ```python
    import l0
    from l0.monitoring import Monitor, MonitoringConfig, Telemetry

    # Simple usage with defaults
    monitor = Monitor()

    result = await l0.run(
        stream=lambda: client.chat.completions.create(...),
        on_event=monitor.handle_event,
    )

    # Get telemetry
    telemetry = monitor.get_telemetry()
    print(f"TTFT: {telemetry.metrics.time_to_first_token}s")
    print(f"Tokens/sec: {telemetry.metrics.tokens_per_second}")

    # Export
    from l0.monitoring import TelemetryExporter
    json_data = TelemetryExporter.to_json(telemetry)
    csv_data = TelemetryExporter.to_csv([telemetry])

    # With OpenTelemetry
    from l0.monitoring import OpenTelemetryConfig, OpenTelemetryExporter

    otel_config = OpenTelemetryConfig(
        service_name="my-llm-app",
        endpoint="http://localhost:4317",
    )
    otel = OpenTelemetryExporter(otel_config)
    otel.export(telemetry)

    # With event-based OpenTelemetry handler
    from opentelemetry import trace, metrics
    from l0.monitoring import create_opentelemetry_handler

    result = await l0.run(
        stream=lambda: client.chat.completions.create(...),
        on_event=create_opentelemetry_handler(
            tracer=trace.get_tracer("my-app"),
            meter=metrics.get_meter("my-app"),
        ),
    )

    # With Sentry
    from l0.monitoring import SentryConfig, SentryExporter

    sentry_config = SentryConfig(
        dsn="https://...",
        environment="production",
    )
    sentry = SentryExporter(sentry_config)
    sentry.capture_error(error, telemetry)

    # With event-based Sentry handler
    import sentry_sdk
    from l0.monitoring import create_sentry_handler

    result = await l0.run(
        stream=lambda: client.chat.completions.create(...),
        on_event=create_sentry_handler(sentry_sdk),
    )

    # Combine multiple handlers
    from l0.monitoring import combine_events

    result = await l0.run(
        stream=lambda: client.chat.completions.create(...),
        on_event=combine_events(
            create_opentelemetry_handler(tracer=tracer, meter=meter),
            create_sentry_handler(sentry_sdk),
            monitor.handle_event,
        ),
    )
    ```
"""

from .config import (
    MetricsConfig,
    MonitoringConfig,
    SamplingConfig,
)
from .dispatcher import EventDispatcher
from .exporter import TelemetryExporter
from .handlers import (
    # Scoped API
    Monitoring,
    # Legacy functions
    batch_events,
    combine_events,
    debounce_events,
    exclude_events,
    filter_events,
    sample_events,
    tap_events,
)
from .monitor import Monitor
from .normalize import (
    L0Event,
    L0EventType,
    create_complete_event,
    create_error_event,
    create_message_event,
    create_token_event,
    extract_tokens,
    normalize_stream_event,
    reconstruct_text,
)
from .otel import (
    NoOpSpan,
    OpenTelemetry,
    OpenTelemetryConfig,
    OpenTelemetryExporter,
    OpenTelemetryExporterConfig,
    SemanticAttributes,
    create_opentelemetry_handler,
    with_opentelemetry,
)
from .sentry import (
    Sentry,
    SentryConfig,
    SentryExporter,
    SentryExporterConfig,
    create_sentry_handler,
    with_sentry,
)
from .telemetry import (
    ErrorInfo,
    GuardrailInfo,
    Metrics,
    RetryInfo,
    Telemetry,
    TimingInfo,
)

__all__ = [
    # Scoped API
    "Monitoring",  # Class with .combine(), .filter(), .exclude(), .debounce(), .batch(), .sample(), .tap()
    # Config
    "MonitoringConfig",
    "MetricsConfig",
    "SamplingConfig",
    # Telemetry
    "Telemetry",
    "Metrics",
    "TimingInfo",
    "RetryInfo",
    "GuardrailInfo",
    "ErrorInfo",
    # Monitor
    "Monitor",
    # Exporter
    "TelemetryExporter",
    # Event Dispatcher
    "EventDispatcher",
    # Event Normalization
    "L0Event",
    "L0EventType",
    "normalize_stream_event",
    "create_token_event",
    "create_message_event",
    "create_complete_event",
    "create_error_event",
    "extract_tokens",
    "reconstruct_text",
    # Event Handlers
    "combine_events",
    "filter_events",
    "exclude_events",
    "debounce_events",
    "batch_events",
    "sample_events",
    "tap_events",
    # OpenTelemetry
    "OpenTelemetry",
    "OpenTelemetryConfig",
    "OpenTelemetryExporter",
    "OpenTelemetryExporterConfig",
    "SemanticAttributes",
    "NoOpSpan",
    "create_opentelemetry_handler",
    "with_opentelemetry",
    # Sentry
    "Sentry",
    "SentryConfig",
    "SentryExporter",
    "SentryExporterConfig",
    "create_sentry_handler",
    "with_sentry",
]
