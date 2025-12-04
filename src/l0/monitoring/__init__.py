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

    # With Sentry
    from l0.monitoring import SentryConfig, SentryExporter

    sentry_config = SentryConfig(
        dsn="https://...",
        environment="production",
    )
    sentry = SentryExporter(sentry_config)
    sentry.capture_error(error, telemetry)
    ```
"""

from .config import (
    MetricsConfig,
    MonitoringConfig,
    SamplingConfig,
)
from .exporter import TelemetryExporter
from .monitor import Monitor
from .otel import OpenTelemetryConfig, OpenTelemetryExporter
from .sentry import SentryConfig, SentryExporter
from .telemetry import (
    ErrorInfo,
    GuardrailInfo,
    Metrics,
    RetryInfo,
    Telemetry,
    TimingInfo,
)

__all__ = [
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
    # OpenTelemetry
    "OpenTelemetryConfig",
    "OpenTelemetryExporter",
    # Sentry
    "SentryConfig",
    "SentryExporter",
]
