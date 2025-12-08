# Monitoring & Telemetry

L0 provides production-ready observability with OpenTelemetry and Sentry support, telemetry collection, and flexible event handling.

## Quick Start

```python
import l0
from l0.monitoring import Monitor

# Create a monitor
monitor = Monitor()

# Use with l0.run
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o", messages=messages, stream=True
    ),
    on_event=monitor.handle_event,
)

# Get telemetry
telemetry = monitor.get_telemetry()
print(f"TTFT: {telemetry.metrics.time_to_first_token}s")
print(f"Tokens/sec: {telemetry.metrics.tokens_per_second}")
print(f"Total tokens: {telemetry.metrics.token_count}")
```

---

## Monitor Class

The `Monitor` class collects telemetry from L0 observability events:

```python
from l0.monitoring import Monitor, MonitoringConfig

# Simple usage
monitor = Monitor()

# With custom config
config = MonitoringConfig.production()
monitor = Monitor(config)

# With completion callback
def on_complete(telemetry):
    print(f"Stream completed: {telemetry.stream_id}")
    print(f"Duration: {telemetry.timing.duration}s")

monitor = Monitor(on_complete=on_complete)
```

### Monitor Methods

| Method | Description |
|--------|-------------|
| `handle_event(event)` | Process an observability event |
| `get_telemetry(stream_id=None)` | Get telemetry for a stream (or most recent) |
| `get_all_telemetry()` | Get all buffered telemetry records |
| `get_aggregate_metrics()` | Get aggregate metrics across all telemetry |
| `clear()` | Clear all telemetry data |

---

## Configuration

### MonitoringConfig

```python
from l0.monitoring import MonitoringConfig, SamplingConfig, MetricsConfig

# Default config
config = MonitoringConfig()

# Production config (10% sampling, errors always sampled)
config = MonitoringConfig.production()

# Development config (100% sampling, full metrics)
config = MonitoringConfig.development()

# Minimal config (errors only)
config = MonitoringConfig.minimal()

# Custom config
config = MonitoringConfig(
    enabled=True,
    sampling=SamplingConfig(
        rate=0.1,              # 10% sampling
        min_duration=0.0,      # Min duration to sample
        sample_errors=True,    # Always sample errors
        sample_slow=True,      # Always sample slow requests
        slow_threshold=5.0,    # Slow threshold in seconds
    ),
    metrics=MetricsConfig(
        collect_tokens=True,
        collect_timing=True,
        collect_retries=True,
        collect_guardrails=True,
        collect_errors=True,
        inter_token_latency=False,  # Adds overhead
        percentiles=[0.5, 0.9, 0.95, 0.99],
    ),
    buffer_size=100,
    flush_interval=0.0,      # 0 = disabled
    log_level="info",
)
```

---

## Telemetry Data Structure

### Telemetry

```python
from l0.monitoring import Telemetry

telemetry = monitor.get_telemetry()

# Stream identification
telemetry.stream_id        # Unique stream identifier
telemetry.session_id       # Session identifier (if grouped)
telemetry.model            # Model name/identifier

# Timing information
telemetry.timing.started_at          # Start timestamp (datetime)
telemetry.timing.completed_at        # Completion timestamp (datetime)
telemetry.timing.duration            # Total duration in seconds
telemetry.timing.time_to_first_token # TTFT in seconds
telemetry.timing.inter_token_latencies  # List of latencies

# Pre-calculated metrics
telemetry.metrics.token_count            # Total tokens
telemetry.metrics.tokens_per_second      # Token generation rate
telemetry.metrics.time_to_first_token    # TTFT in seconds
telemetry.metrics.avg_inter_token_latency
telemetry.metrics.p50_inter_token_latency
telemetry.metrics.p90_inter_token_latency
telemetry.metrics.p99_inter_token_latency

# Retry information
telemetry.retries.attempt           # Current attempt number
telemetry.retries.max_attempts      # Max attempts configured
telemetry.retries.model_retries     # Model error retries
telemetry.retries.network_retries   # Network error retries
telemetry.retries.total_retries     # Total retries
telemetry.retries.last_error        # Last error message
telemetry.retries.last_error_category  # Error category

# Guardrail information
telemetry.guardrails.rules_checked  # Number of rules checked
telemetry.guardrails.violations     # List of violation details
telemetry.guardrails.passed         # Whether all guardrails passed

# Error information
telemetry.error.occurred       # Whether an error occurred
telemetry.error.message        # Error message
telemetry.error.category       # Error category
telemetry.error.code           # Error code
telemetry.error.stack          # Stack trace (if available)
telemetry.error.recoverable    # Whether error was recoverable

# Status
telemetry.content_length  # Length of generated content
telemetry.completed       # Whether stream completed
telemetry.aborted         # Whether stream was aborted
telemetry.metadata        # Custom metadata dict
```

### Aggregate Metrics

```python
metrics = monitor.get_aggregate_metrics()

print(metrics["count"])            # Number of streams
print(metrics["total_tokens"])     # Total tokens across all
print(metrics["total_duration"])   # Total duration
print(metrics["total_retries"])    # Total retries
print(metrics["error_count"])      # Error count
print(metrics["error_rate"])       # Error rate (0-1)
print(metrics["completed_count"])  # Completed count
print(metrics["completion_rate"])  # Completion rate (0-1)
print(metrics["avg_tokens"])       # Average tokens per stream
print(metrics["avg_duration"])     # Average duration
print(metrics["avg_ttft"])         # Average time to first token
print(metrics["avg_tokens_per_sec"])  # Average tokens/second
```

---

## Exporting Telemetry

### TelemetryExporter

```python
from l0.monitoring import TelemetryExporter

telemetry = monitor.get_telemetry()

# To JSON
json_str = TelemetryExporter.to_json(telemetry)
json_str = TelemetryExporter.to_json(telemetry, indent=None)  # Compact

# To dictionary
data = TelemetryExporter.to_dict(telemetry)

# To CSV (multiple records)
csv_str = TelemetryExporter.to_csv(monitor.get_all_telemetry())

# To JSON Lines (multiple records)
jsonl_str = TelemetryExporter.to_jsonl(monitor.get_all_telemetry())

# To log format
log_str = TelemetryExporter.to_log_format(telemetry)
# Output: stream_id=abc model=gpt-4o duration=1.234s ttft=0.456s tokens=100 ...

# To Prometheus-compatible metrics
metrics = TelemetryExporter.to_metrics(
    telemetry,
    prefix="l0",
    labels={"environment": "production"},
)
# Returns dict of metric names to {value, type, labels}
```

---

## Event Handler Utilities

### Combining Handlers

```python
from l0.monitoring import Monitoring, Monitor

monitor = Monitor()

# Combine multiple handlers
combined = Monitoring.combine(
    monitor.handle_event,
    lambda e: print(f"Event: {e.type}"),
    custom_handler,
)

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_event=combined,
)
```

### Filtering Events

```python
from l0.monitoring import Monitoring
from l0.events import ObservabilityEventType

# Only handle specific event types
errors_only = Monitoring.filter(
    [ObservabilityEventType.ERROR, ObservabilityEventType.RETRY_GIVE_UP],
    lambda e: send_alert(e),
)

# Exclude noisy events
quiet_handler = Monitoring.exclude(
    [ObservabilityEventType.TOKEN],  # Exclude individual tokens
    lambda e: print(e.type),
)
```

### Debouncing and Batching

```python
from l0.monitoring import Monitoring

# Debounce high-frequency events
throttled = Monitoring.debounce(
    0.1,  # 100ms debounce interval
    lambda e: print(f"Latest: {e.type}"),
)

# Batch events for efficient processing
batched = Monitoring.batch(
    size=10,              # Batch size
    max_wait_seconds=1.0, # Max wait before flush
    handler=lambda events: send_to_analytics(events),
)
```

### Sampling Events

```python
from l0.monitoring import Monitoring

# Sample 10% of events
sampled = Monitoring.sample(
    0.1,  # 10% sampling rate
    lambda e: log_event(e),
)
```

### Tap (Pass-Through)

```python
from l0.monitoring import Monitoring

# Observe events without modifying flow
on_event = Monitoring.combine(
    Monitoring.tap(lambda e: print(f"DEBUG: {e.type}")),
    main_handler,
)
```

---

## OpenTelemetry Integration

### Quick Start

```python
from opentelemetry import trace, metrics
from l0.monitoring import Monitoring

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_event=Monitoring.opentelemetry(
        tracer=trace.get_tracer("my-app"),
        meter=metrics.get_meter("my-app"),
    ),
)
```

### Event Handler Configuration

```python
from l0.monitoring import Monitoring

handler = Monitoring.opentelemetry(
    tracer=trace.get_tracer("my-app"),
    meter=metrics.get_meter("my-app"),
    service_name="l0",
    trace_tokens=False,              # Create spans for individual tokens
    record_token_content=False,      # Record token content (privacy)
    record_guardrail_violations=True,
    default_attributes={"env": "production"},
)
```

### Using create_opentelemetry_handler

```python
from opentelemetry import trace, metrics
from l0.monitoring import create_opentelemetry_handler, OpenTelemetryConfig

config = OpenTelemetryConfig(
    service_name="my-llm-app",
    trace_tokens=False,
    record_token_content=False,
    record_guardrail_violations=True,
    default_attributes={"deployment": "prod"},
)

handler = create_opentelemetry_handler(
    tracer=trace.get_tracer("my-app"),
    meter=metrics.get_meter("my-app"),
    config=config,
)

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_event=handler,
)
```

### OpenTelemetry Class

For more control, use the `OpenTelemetry` class directly:

```python
from opentelemetry import trace, metrics
from l0.monitoring import OpenTelemetry, OpenTelemetryConfig

otel = OpenTelemetry(
    tracer=trace.get_tracer("l0"),
    meter=metrics.get_meter("l0"),
    config=OpenTelemetryConfig(service_name="my-app"),
)

# Trace a stream operation
result = await otel.trace_stream(
    "chat-completion",
    lambda span: l0.run(stream=lambda: ...),
)

# Record telemetry manually
otel.record_telemetry(telemetry, span=current_span)

# Create child spans
span = otel.create_span("sub-operation", {"key": "value"})
```

### OpenTelemetry Exporter

For exporting telemetry to OTLP endpoints:

```python
from l0.monitoring import OpenTelemetryExporterConfig, OpenTelemetryExporter

config = OpenTelemetryExporterConfig(
    service_name="my-llm-app",
    endpoint="http://localhost:4317",
    headers={"Authorization": "Bearer token"},
    insecure=False,
    timeout=30.0,
    trace_enabled=True,
    metrics_enabled=True,
    batch_export=True,
    export_interval=5.0,
)

# Or from environment variables
config = OpenTelemetryExporterConfig.from_env()

exporter = OpenTelemetryExporter(config)
exporter.export(telemetry)

# Cleanup
exporter.shutdown()
```

### Semantic Attributes

```python
from l0.monitoring import SemanticAttributes

# Available attributes
SemanticAttributes.LLM_SYSTEM                # "gen_ai.system"
SemanticAttributes.LLM_REQUEST_MODEL         # "gen_ai.request.model"
SemanticAttributes.LLM_RESPONSE_MODEL        # "gen_ai.response.model"
SemanticAttributes.LLM_USAGE_INPUT_TOKENS    # "gen_ai.usage.input_tokens"
SemanticAttributes.LLM_USAGE_OUTPUT_TOKENS   # "gen_ai.usage.output_tokens"

# L0-specific attributes
SemanticAttributes.L0_SESSION_ID             # "l0.session_id"
SemanticAttributes.L0_STREAM_COMPLETED       # "l0.stream.completed"
SemanticAttributes.L0_RETRY_COUNT            # "l0.retry.count"
SemanticAttributes.L0_NETWORK_ERROR_COUNT    # "l0.network.error_count"
SemanticAttributes.L0_GUARDRAIL_VIOLATION_COUNT  # "l0.guardrail.violation_count"
SemanticAttributes.L0_DRIFT_DETECTED         # "l0.drift.detected"
SemanticAttributes.L0_TIME_TO_FIRST_TOKEN    # "l0.time_to_first_token_ms"
SemanticAttributes.L0_TOKENS_PER_SECOND      # "l0.tokens_per_second"
```

---

## Sentry Integration

### Quick Start

```python
import sentry_sdk
from l0.monitoring import Monitoring

sentry_sdk.init(dsn="https://...")

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_event=Monitoring.sentry(sentry_sdk),
)
```

### Event Handler Configuration

```python
from l0.monitoring import Monitoring

handler = Monitoring.sentry(
    sentry_sdk,
    capture_network_errors=True,
    capture_guardrail_violations=True,
    min_guardrail_severity="error",  # "debug" | "info" | "warning" | "error" | "fatal"
    breadcrumbs_for_tokens=False,
    enable_tracing=True,
    tags={"model": "gpt-4o"},
    environment="production",
)
```

### Using create_sentry_handler

```python
import sentry_sdk
from l0.monitoring import create_sentry_handler, SentryConfig

sentry_sdk.init(dsn="https://...")

config = SentryConfig(
    capture_network_errors=True,
    capture_guardrail_violations=True,
    min_guardrail_severity="error",
    breadcrumbs_for_tokens=False,
    enable_tracing=True,
    tags={"service": "llm-api"},
    environment="production",
)

handler = create_sentry_handler(sentry_sdk, config)

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_event=handler,
)
```

### Sentry Class

For more control, use the `Sentry` class directly:

```python
import sentry_sdk
from l0.monitoring import Sentry, SentryConfig

sentry_sdk.init(dsn="https://...")

sentry_integration = Sentry(
    sentry=sentry_sdk,
    config=SentryConfig(
        capture_network_errors=True,
        breadcrumbs_for_tokens=False,
    ),
)

# Manual tracking
sentry_integration.start_execution("chat-completion", {"model": "gpt-4"})
sentry_integration.start_stream()

# Record events
sentry_integration.record_token(token)
sentry_integration.record_first_token(ttft_ms=456)
sentry_integration.record_retry(attempt=2, reason="timeout", is_network_error=True)
sentry_integration.record_network_error(error, "timeout", retried=True)
sentry_integration.record_guardrail_violations(violations)
sentry_integration.record_drift(detected=True, types=["repetition"])

# Complete
sentry_integration.complete_stream(token_count=100)
sentry_integration.complete_execution(telemetry)
```

### Sentry Exporter

For exporting telemetry errors to Sentry:

```python
from l0.monitoring import SentryExporterConfig, SentryExporter

config = SentryExporterConfig(
    dsn="https://xxx@sentry.io/123",
    environment="production",
    release="1.0.0",
    sample_rate=1.0,
    traces_sample_rate=0.1,
    tags={"service": "llm-api"},
)

# Or from environment
config = SentryExporterConfig.from_env()

exporter = SentryExporter(config)
exporter.init()

# Capture errors with telemetry context
try:
    result = await l0.run(...)
except Exception as e:
    exporter.capture_error(e, monitor.get_telemetry())

# Capture from telemetry with error
if telemetry.error.occurred:
    exporter.capture_telemetry_error(telemetry)

# Add breadcrumbs
exporter.add_breadcrumb("Processing started", category="l0", level="info")

# Set user context
exporter.set_user(user_id="123", email="user@example.com")

# Cleanup
exporter.flush()
exporter.close()
```

### What Gets Tracked

**Breadcrumbs:**
- L0 execution start/complete
- Stream start/complete
- Tokens (if `breadcrumbs_for_tokens=True`)
- First token (TTFT)
- Retry attempts
- Network errors
- Guardrail violations
- Drift detection

**Captured Events:**
- Network errors (final failures, not retried)
- Guardrail violations (above severity threshold)
- Execution failures

**Context:**
- L0 telemetry (duration, tokens, retries, etc.)
- Error details (category, code, recoverable)

---

## Combining Integrations

```python
from l0.monitoring import Monitoring, Monitor
import sentry_sdk

monitor = Monitor()

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_event=Monitoring.combine(
        monitor.handle_event,
        Monitoring.opentelemetry(tracer=tracer, meter=meter),
        Monitoring.sentry(sentry_sdk),
        lambda e: print(f"Event: {e.type}"),
    ),
)
```

---

## Best Practices

### Production Configuration

```python
from l0.monitoring import MonitoringConfig, Monitor

config = MonitoringConfig.production()  # 10% sampling, errors always sampled
monitor = Monitor(config)
```

### Development Configuration

```python
from l0.monitoring import MonitoringConfig, Monitor

config = MonitoringConfig.development()  # 100% sampling, full metrics
monitor = Monitor(config)
```

### Add Contextual Metadata

```python
from l0.monitoring import Monitor

def on_complete(telemetry):
    telemetry.metadata["user_id"] = current_user.id
    telemetry.metadata["request_id"] = request.id
    export_telemetry(telemetry)

monitor = Monitor(on_complete=on_complete)
```

### Export Telemetry Asynchronously

```python
import asyncio
from l0.monitoring import Monitor, TelemetryExporter

async def export_async(telemetry):
    # Don't block on export
    await asyncio.to_thread(send_to_analytics, telemetry)

monitor = Monitor(on_complete=lambda t: asyncio.create_task(export_async(t)))
```

### Monitor Key Metrics

Focus on these key metrics for production monitoring:

| Metric | Description | Target |
|--------|-------------|--------|
| `time_to_first_token` | User-perceived latency | < 1s |
| `tokens_per_second` | Generation throughput | > 20 |
| `error_rate` | Stream failures | < 1% |
| `retry_rate` | Retry frequency | < 5% |
| `guardrail_violations` | Content issues | < 1% |

---

## API Reference

### Configuration

| Class | Description |
|-------|-------------|
| `MonitoringConfig` | Main monitoring configuration |
| `SamplingConfig` | Sampling configuration |
| `MetricsConfig` | Metrics collection configuration |

### Telemetry

| Class | Description |
|-------|-------------|
| `Telemetry` | Complete telemetry data |
| `Metrics` | Pre-calculated metrics |
| `TimingInfo` | Timing information |
| `RetryInfo` | Retry information |
| `GuardrailInfo` | Guardrail information |
| `ErrorInfo` | Error information |

### Monitor & Export

| Class | Description |
|-------|-------------|
| `Monitor` | Collects telemetry from events |
| `TelemetryExporter` | Export telemetry to various formats |

### Event Handlers

| Function | Description |
|----------|-------------|
| `Monitoring.combine(*handlers)` | Combine multiple handlers |
| `Monitoring.filter(types, handler)` | Filter by event types |
| `Monitoring.exclude(types, handler)` | Exclude event types |
| `Monitoring.debounce(seconds, handler)` | Debounce events |
| `Monitoring.batch(size, max_wait, handler)` | Batch events |
| `Monitoring.sample(rate, handler)` | Sample events |
| `Monitoring.tap(handler)` | Pass-through handler |
| `Monitoring.opentelemetry(...)` | Create OpenTelemetry handler |
| `Monitoring.sentry(...)` | Create Sentry handler |

### OpenTelemetry

| Class/Function | Description |
|----------------|-------------|
| `OpenTelemetry` | OpenTelemetry integration class |
| `OpenTelemetryConfig` | Runtime configuration |
| `OpenTelemetryExporter` | Export to OTLP endpoints |
| `OpenTelemetryExporterConfig` | Exporter configuration |
| `create_opentelemetry_handler()` | Create event handler |
| `with_opentelemetry()` | Context manager for tracing |
| `SemanticAttributes` | Semantic convention attributes |
| `NoOpSpan` | No-op span for disabled tracing |

### Sentry

| Class/Function | Description |
|----------------|-------------|
| `Sentry` | Sentry integration class |
| `SentryConfig` | Handler configuration |
| `SentryExporter` | Export errors to Sentry |
| `SentryExporterConfig` | Exporter configuration |
| `create_sentry_handler()` | Create event handler |
| `with_sentry()` | Context manager for tracking |
