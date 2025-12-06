"""Tests for L0 monitoring system."""

from datetime import datetime, timezone
from typing import Any

import pytest

from l0.events import ObservabilityEvent, ObservabilityEventType
from l0.monitoring import (
    MetricsConfig,
    Monitor,
    MonitoringConfig,
    SamplingConfig,
    Telemetry,
    TelemetryExporter,
)
from l0.monitoring.telemetry import (
    ErrorInfo,
    GuardrailInfo,
    Metrics,
    RetryInfo,
    TimingInfo,
)
from l0.types import ErrorCategory


class TestMonitoringConfig:
    """Tests for MonitoringConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MonitoringConfig.default()
        assert config.enabled is True
        assert config.sampling.rate == 1.0
        assert config.metrics.collect_tokens is True

    def test_production_config(self):
        """Test production configuration."""
        config = MonitoringConfig.production()
        assert config.sampling.rate == 0.1
        assert config.sampling.sample_errors is True
        assert config.metrics.inter_token_latency is False

    def test_development_config(self):
        """Test development configuration."""
        config = MonitoringConfig.development()
        assert config.sampling.rate == 1.0
        assert config.metrics.inter_token_latency is True
        assert config.log_level == "debug"

    def test_minimal_config(self):
        """Test minimal configuration."""
        config = MonitoringConfig.minimal()
        assert config.sampling.rate == 0.0
        assert config.sampling.sample_errors is True
        assert config.metrics.collect_tokens is False

    def test_sampling_config_validation(self):
        """Test sampling config validation."""
        with pytest.raises(ValueError):
            SamplingConfig(rate=1.5)  # > 1.0

        with pytest.raises(ValueError):
            SamplingConfig(rate=-0.1)  # < 0.0


class TestTelemetry:
    """Tests for Telemetry data structures."""

    def test_telemetry_creation(self):
        """Test basic telemetry creation."""
        telemetry = Telemetry(stream_id="test-123")
        assert telemetry.stream_id == "test-123"
        assert telemetry.completed is False
        assert telemetry.metrics.token_count == 0

    def test_metrics_calculation(self):
        """Test metrics calculation."""
        metrics = Metrics.calculate(
            token_count=100,
            duration=2.0,
            time_to_first_token=0.5,
            inter_token_latencies=[0.02, 0.03, 0.025, 0.028, 0.022],
        )

        assert metrics.token_count == 100
        assert metrics.tokens_per_second == 50.0
        assert metrics.time_to_first_token == 0.5
        assert metrics.avg_inter_token_latency is not None
        assert 0.02 < metrics.avg_inter_token_latency < 0.03

    def test_metrics_with_empty_latencies(self):
        """Test metrics with no inter-token latencies."""
        metrics = Metrics.calculate(
            token_count=50,
            duration=1.0,
            time_to_first_token=0.3,
            inter_token_latencies=[],
        )

        assert metrics.tokens_per_second == 50.0
        assert metrics.avg_inter_token_latency is None
        assert metrics.p50_inter_token_latency is None

    def test_telemetry_finalize(self):
        """Test telemetry finalization."""
        telemetry = Telemetry(stream_id="test-456")
        telemetry.metrics = Metrics(token_count=200)
        telemetry.timing.duration = 4.0
        telemetry.timing.time_to_first_token = 0.8

        telemetry.finalize()

        assert telemetry.metrics.tokens_per_second == 50.0
        assert telemetry.metrics.time_to_first_token == 0.8


class TestMonitor:
    """Tests for Monitor class."""

    def test_monitor_creation(self):
        """Test monitor creation."""
        monitor = Monitor()
        assert monitor.config.enabled is True

    def test_monitor_with_config(self):
        """Test monitor with custom config."""
        config = MonitoringConfig(enabled=False)
        monitor = Monitor(config)
        assert monitor.config.enabled is False

    def test_handle_stream_init_event(self):
        """Test handling STREAM_INIT event."""
        monitor = Monitor()

        event = ObservabilityEvent(
            type=ObservabilityEventType.STREAM_INIT,
            ts=1000.0,
            stream_id="stream-1",
            meta={"model": "gpt-4"},
        )

        monitor.handle_event(event)

        telemetry = monitor.get_telemetry("stream-1")
        assert telemetry is not None
        assert telemetry.model == "gpt-4"
        assert telemetry.timing.started_at is not None

    def test_handle_retry_events(self):
        """Test handling retry events."""
        monitor = Monitor()

        # Start event
        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.RETRY_START,
                ts=1000.0,
                stream_id="stream-2",
                meta={"max_attempts": 3},
            )
        )

        # Retry attempt
        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.RETRY_ATTEMPT,
                ts=1001.0,
                stream_id="stream-2",
                meta={
                    "attempt": 2,
                    "category": ErrorCategory.NETWORK,
                    "error": "Connection failed",
                },
            )
        )

        telemetry = monitor.get_telemetry("stream-2")
        assert telemetry is not None
        assert telemetry.retries.max_attempts == 3
        assert telemetry.retries.total_retries == 1
        assert telemetry.retries.network_retries == 1
        assert telemetry.retries.last_error == "Connection failed"

    def test_handle_guardrail_events(self):
        """Test handling guardrail events."""
        monitor = Monitor()

        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.GUARDRAIL_RULE_START,
                ts=1000.0,
                stream_id="stream-3",
                meta={"rule": "json"},
            )
        )

        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.GUARDRAIL_RULE_RESULT,
                ts=1001.0,
                stream_id="stream-3",
                meta={"violations": [{"rule": "json", "message": "Invalid JSON"}]},
            )
        )

        telemetry = monitor.get_telemetry("stream-3")
        assert telemetry is not None
        assert telemetry.guardrails.rules_checked == 1
        assert len(telemetry.guardrails.violations) == 1
        assert telemetry.guardrails.passed is False

    def test_handle_error_event(self):
        """Test handling ERROR event."""
        monitor = Monitor()

        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.ERROR,
                ts=1000.0,
                stream_id="stream-4",
                meta={
                    "error": ValueError("Something went wrong"),
                    "category": ErrorCategory.MODEL,
                    "code": "MODEL_ERROR",
                },
            )
        )

        telemetry = monitor.get_telemetry("stream-4")
        assert telemetry is not None
        assert telemetry.error.occurred is True
        assert telemetry.error.message is not None
        assert "Something went wrong" in telemetry.error.message
        assert telemetry.error.category == ErrorCategory.MODEL

    def test_handle_complete_event(self):
        """Test handling COMPLETE event."""
        monitor = Monitor()
        completed_telemetry = []

        def on_complete(t: Telemetry) -> None:
            completed_telemetry.append(t)

        monitor = Monitor(on_complete=on_complete)

        # Init
        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.STREAM_INIT,
                ts=1000.0,
                stream_id="stream-5",
                meta={},
            )
        )

        # Complete
        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.COMPLETE,
                ts=2000.0,
                stream_id="stream-5",
                meta={},
            )
        )

        telemetry = monitor.get_telemetry("stream-5")
        assert telemetry is not None
        assert telemetry.completed is True
        assert telemetry.timing.completed_at is not None
        assert len(completed_telemetry) == 1

    def test_sampling(self):
        """Test sampling behavior."""
        config = MonitoringConfig(
            sampling=SamplingConfig(rate=0.0, sample_errors=False)
        )
        monitor = Monitor(config)

        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.STREAM_INIT,
                ts=1000.0,
                stream_id="stream-6",
                meta={},
            )
        )

        # Should not be sampled
        telemetry = monitor.get_telemetry("stream-6")
        assert telemetry is None

    def test_force_sample_errors(self):
        """Test force sampling of errors."""
        config = MonitoringConfig(sampling=SamplingConfig(rate=0.0, sample_errors=True))
        monitor = Monitor(config)

        # Initial event - not sampled
        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.STREAM_INIT,
                ts=1000.0,
                stream_id="stream-7",
                meta={},
            )
        )

        # Error event - should force sampling
        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.ERROR,
                ts=1001.0,
                stream_id="stream-7",
                meta={"error": "Test error"},
            )
        )

        telemetry = monitor.get_telemetry("stream-7")
        assert telemetry is not None
        assert telemetry.error.occurred is True

    def test_get_all_telemetry(self):
        """Test getting all telemetry."""
        monitor = Monitor()

        for i in range(3):
            monitor.handle_event(
                ObservabilityEvent(
                    type=ObservabilityEventType.STREAM_INIT,
                    ts=1000.0 + i,
                    stream_id=f"stream-{i}",
                    meta={},
                )
            )
            monitor.handle_event(
                ObservabilityEvent(
                    type=ObservabilityEventType.COMPLETE,
                    ts=2000.0 + i,
                    stream_id=f"stream-{i}",
                    meta={},
                )
            )

        all_telemetry = monitor.get_all_telemetry()
        assert len(all_telemetry) == 3

    def test_clear(self):
        """Test clearing monitor data."""
        monitor = Monitor()

        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.STREAM_INIT,
                ts=1000.0,
                stream_id="stream-clear",
                meta={},
            )
        )

        monitor.clear()

        assert monitor.get_telemetry() is None
        assert len(monitor.get_all_telemetry()) == 0

    def test_aggregate_metrics(self):
        """Test aggregate metrics calculation."""
        monitor = Monitor()

        # Create multiple completed streams
        for i in range(5):
            stream_id = f"agg-stream-{i}"
            monitor.handle_event(
                ObservabilityEvent(
                    type=ObservabilityEventType.STREAM_INIT,
                    ts=1000.0,
                    stream_id=stream_id,
                    meta={},
                )
            )
            monitor.handle_event(
                ObservabilityEvent(
                    type=ObservabilityEventType.COMPLETE,
                    ts=2000.0,
                    stream_id=stream_id,
                    meta={},
                )
            )

        aggregates = monitor.get_aggregate_metrics()
        assert aggregates["count"] == 5
        assert aggregates["completed_count"] == 5
        assert aggregates["error_count"] == 0


class TestTelemetryExporter:
    """Tests for TelemetryExporter."""

    @pytest.fixture
    def sample_telemetry(self) -> Telemetry:
        """Create sample telemetry for testing."""
        telemetry = Telemetry(
            stream_id="export-test-1",
            session_id="session-1",
            model="gpt-4",
            timing=TimingInfo(
                started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                completed_at=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
                duration=5.0,
                time_to_first_token=0.5,
            ),
            retries=RetryInfo(total_retries=1, network_retries=1),
            guardrails=GuardrailInfo(rules_checked=2, passed=True),
            error=ErrorInfo(occurred=False),
            metrics=Metrics(
                token_count=100,
                tokens_per_second=20.0,
                time_to_first_token=0.5,
            ),
            completed=True,
            content_length=500,
        )
        return telemetry

    def test_to_json(self, sample_telemetry: Telemetry):
        """Test JSON export."""
        json_str = TelemetryExporter.to_json(sample_telemetry)
        assert "export-test-1" in json_str
        assert "gpt-4" in json_str

    def test_to_dict(self, sample_telemetry: Telemetry):
        """Test dict export."""
        data = TelemetryExporter.to_dict(sample_telemetry)
        assert data["stream_id"] == "export-test-1"
        assert data["model"] == "gpt-4"
        assert data["metrics"]["token_count"] == 100

    def test_to_csv(self, sample_telemetry: Telemetry):
        """Test CSV export."""
        csv_str = TelemetryExporter.to_csv([sample_telemetry])
        assert "stream_id" in csv_str  # Header
        assert "export-test-1" in csv_str
        assert "gpt-4" in csv_str

    def test_to_csv_empty(self):
        """Test CSV export with empty list."""
        csv_str = TelemetryExporter.to_csv([])
        assert csv_str == ""

    def test_to_log_format(self, sample_telemetry: Telemetry):
        """Test log format export."""
        log_str = TelemetryExporter.to_log_format(sample_telemetry)
        assert "stream_id=export-test-1" in log_str
        assert "model=gpt-4" in log_str
        assert "tokens=100" in log_str
        assert "completed=True" in log_str

    def test_to_metrics(self, sample_telemetry: Telemetry):
        """Test Prometheus metrics export."""
        metrics = TelemetryExporter.to_metrics(sample_telemetry, prefix="l0")

        assert "l0_tokens_total" in metrics
        assert metrics["l0_tokens_total"]["value"] == 100
        assert metrics["l0_tokens_total"]["type"] == "counter"

        assert "l0_duration_seconds" in metrics
        assert metrics["l0_duration_seconds"]["value"] == 5.0

        assert "l0_ttft_seconds" in metrics
        assert metrics["l0_ttft_seconds"]["value"] == 0.5

    def test_to_jsonl(self, sample_telemetry: Telemetry):
        """Test JSONL export."""
        telemetry2 = Telemetry(stream_id="export-test-2", completed=True)
        jsonl_str = TelemetryExporter.to_jsonl([sample_telemetry, telemetry2])

        lines = jsonl_str.strip().split("\n")
        assert len(lines) == 2
        assert "export-test-1" in lines[0]
        assert "export-test-2" in lines[1]


class TestDisabledMonitoring:
    """Tests for disabled monitoring."""

    def test_disabled_monitor_no_events(self):
        """Test that disabled monitor doesn't collect events."""
        config = MonitoringConfig(enabled=False)
        monitor = Monitor(config)

        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.STREAM_INIT,
                ts=1000.0,
                stream_id="disabled-stream",
                meta={},
            )
        )
        monitor.handle_event(
            ObservabilityEvent(
                type=ObservabilityEventType.STREAM_INIT,
                ts=1000.0,
                stream_id="disabled-stream",
                meta={},
            )
        )

        assert monitor.get_telemetry() is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests for EventDispatcher
# ─────────────────────────────────────────────────────────────────────────────


class TestEventDispatcher:
    """Tests for EventDispatcher class."""

    def test_dispatcher_creation(self):
        """Test creating an event dispatcher."""
        from l0.monitoring import EventDispatcher

        dispatcher = EventDispatcher()
        assert dispatcher.stream_id is not None
        assert len(dispatcher.stream_id) > 0

    def test_dispatcher_with_meta(self):
        """Test dispatcher with custom metadata."""
        from l0.monitoring import EventDispatcher

        dispatcher = EventDispatcher(meta={"app": "test", "version": "1.0"})
        assert dispatcher.meta["app"] == "test"
        assert dispatcher.meta["version"] == "1.0"

    def test_emit_with_no_handlers(self):
        """Test that emit does nothing when no handlers registered."""
        from l0.monitoring import EventDispatcher

        dispatcher = EventDispatcher()
        # Should not raise
        dispatcher.emit(ObservabilityEventType.TOKEN, token="hello")

    def test_emit_with_handler(self):
        """Test emitting events to a handler."""
        from l0.monitoring import EventDispatcher

        events = []

        def handler(event: ObservabilityEvent) -> None:
            events.append(event)

        dispatcher = EventDispatcher()
        dispatcher.on_event(handler)
        dispatcher.emit(ObservabilityEventType.TOKEN, token="hello")

        # Allow async handler to complete
        import time

        time.sleep(0.1)

        assert len(events) == 1
        assert events[0].type == ObservabilityEventType.TOKEN
        assert events[0].meta["token"] == "hello"
        assert events[0].stream_id == dispatcher.stream_id

    def test_emit_sync(self):
        """Test synchronous event emission."""
        from l0.monitoring import EventDispatcher

        events = []

        def handler(event: ObservabilityEvent) -> None:
            events.append(event)

        dispatcher = EventDispatcher()
        dispatcher.on_event(handler)
        dispatcher.emit_sync(ObservabilityEventType.STREAM_INIT, model="gpt-4")

        # Sync should be immediate
        assert len(events) == 1
        assert events[0].type == ObservabilityEventType.STREAM_INIT
        assert events[0].meta["model"] == "gpt-4"

    def test_remove_handler(self):
        """Test removing a handler."""
        from l0.monitoring import EventDispatcher

        events = []

        def handler(event: ObservabilityEvent) -> None:
            events.append(event)

        dispatcher = EventDispatcher()
        dispatcher.on_event(handler)
        dispatcher.off_event(handler)
        dispatcher.emit_sync(ObservabilityEventType.TOKEN, token="hello")

        assert len(events) == 0

    def test_handler_error_isolation(self):
        """Test that handler errors don't affect other handlers."""
        from l0.monitoring import EventDispatcher

        events = []

        def bad_handler(event: ObservabilityEvent) -> None:
            raise ValueError("Handler error")

        def good_handler(event: ObservabilityEvent) -> None:
            events.append(event)

        dispatcher = EventDispatcher()
        dispatcher.on_event(bad_handler)
        dispatcher.on_event(good_handler)

        # Should not raise
        dispatcher.emit_sync(ObservabilityEventType.TOKEN, token="hello")

        # Good handler should still receive event
        assert len(events) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Event Normalization
# ─────────────────────────────────────────────────────────────────────────────


class TestEventNormalization:
    """Tests for event normalization utilities."""

    def test_create_token_event(self):
        """Test creating a token event."""
        from l0.monitoring import L0EventType, create_token_event

        event = create_token_event("Hello")
        assert event.type == L0EventType.TOKEN
        assert event.value == "Hello"
        assert event.timestamp > 0

    def test_create_message_event(self):
        """Test creating a message event."""
        from l0.monitoring import L0EventType, create_message_event

        event = create_message_event("Hello, world!", role="assistant")
        assert event.type == L0EventType.MESSAGE
        assert event.value == "Hello, world!"
        assert event.role == "assistant"

    def test_create_complete_event(self):
        """Test creating a complete event."""
        from l0.monitoring import L0EventType, create_complete_event

        event = create_complete_event()
        assert event.type == L0EventType.COMPLETE
        assert event.value is None

    def test_create_error_event(self):
        """Test creating an error event."""
        from l0.monitoring import L0EventType, create_error_event

        error = ValueError("Test error")
        event = create_error_event(error)
        assert event.type == L0EventType.ERROR
        assert event.error is error

    def test_extract_tokens(self):
        """Test extracting tokens from events."""
        from l0.monitoring import (
            create_complete_event,
            create_token_event,
            extract_tokens,
        )

        events = [
            create_token_event("Hello"),
            create_token_event(" "),
            create_token_event("world"),
            create_complete_event(),
        ]
        tokens = extract_tokens(events)
        assert tokens == ["Hello", " ", "world"]

    def test_reconstruct_text(self):
        """Test reconstructing text from events."""
        from l0.monitoring import create_token_event, reconstruct_text

        events = [
            create_token_event("Hello"),
            create_token_event(" "),
            create_token_event("world"),
            create_token_event("!"),
        ]
        text = reconstruct_text(events)
        assert text == "Hello world!"

    def test_normalize_stream_event_vercel_format(self):
        """Test normalizing Vercel AI SDK format."""
        from l0.monitoring import L0EventType, normalize_stream_event

        # Text delta format
        chunk = {"type": "text-delta", "textDelta": "Hello"}
        event = normalize_stream_event(chunk)
        assert event.type == L0EventType.TOKEN
        assert event.value == "Hello"

    def test_normalize_stream_event_openai_format(self):
        """Test normalizing OpenAI format."""
        from l0.monitoring import L0EventType, normalize_stream_event

        # OpenAI-style chunk with delta (dict format)
        chunk = {
            "choices": [
                {
                    "delta": {"content": "Hello"},
                    "index": 0,
                }
            ]
        }
        event = normalize_stream_event(chunk)
        assert event.type == L0EventType.TOKEN
        assert event.value == "Hello"

    def test_normalize_stream_event_string(self):
        """Test normalizing plain string."""
        from l0.monitoring import L0EventType, normalize_stream_event

        event = normalize_stream_event("Hello")
        assert event.type == L0EventType.TOKEN
        assert event.value == "Hello"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Sentry
# ─────────────────────────────────────────────────────────────────────────────


class MockSentryClient:
    """Mock Sentry client for testing."""

    def __init__(self) -> None:
        self.exceptions: list[dict] = []
        self.messages: list[dict] = []
        self.breadcrumbs: list[dict] = []
        self.tags: dict[str, str] = {}
        self.extras: dict[str, Any] = {}
        self.contexts: dict[str, dict] = {}

    def capture_exception(
        self, error: Exception | None = None, **kwargs: Any
    ) -> str | None:
        self.exceptions.append({"error": error, **kwargs})
        return "mock-event-id"

    def capture_message(
        self, message: str, level: str | None = None, **kwargs: Any
    ) -> str | None:
        self.messages.append({"message": message, "level": level, **kwargs})
        return "mock-event-id"

    def add_breadcrumb(self, **kwargs: Any) -> None:
        self.breadcrumbs.append(kwargs)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def set_extra(self, key: str, value: Any) -> None:
        self.extras[key] = value

    def set_context(self, name: str, context: dict[str, Any]) -> None:
        self.contexts[name] = context


class TestSentry:
    """Tests for Sentry class."""

    def test_l0_sentry_creation(self):
        """Test creating Sentry instance."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)
        assert l0_sentry is not None

    def test_start_stream(self):
        """Test starting a stream."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)
        l0_sentry.start_stream()

        assert len(sentry.breadcrumbs) == 1
        assert sentry.breadcrumbs[0]["category"] == "l0.stream"
        assert sentry.breadcrumbs[0]["message"] == "Stream started"

    def test_record_token(self):
        """Test recording tokens."""
        from l0.monitoring import Sentry, SentryConfig

        sentry = MockSentryClient()
        config = SentryConfig(breadcrumbs_for_tokens=True)
        l0_sentry = Sentry(sentry, config)

        l0_sentry.record_token("Hello")
        l0_sentry.record_token(" world")

        # With breadcrumbs_for_tokens=True, should add breadcrumbs
        assert len(sentry.breadcrumbs) == 2

    def test_record_first_token(self):
        """Test recording first token timing."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)
        l0_sentry.record_first_token(250.5)

        assert len(sentry.breadcrumbs) == 1
        assert "First token" in sentry.breadcrumbs[0]["message"]
        assert sentry.breadcrumbs[0]["data"]["ttft_ms"] == 250.5

    def test_record_network_error(self):
        """Test recording network errors."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)

        error = ConnectionError("Connection failed")
        l0_sentry.record_network_error(error, "connection", retried=True)

        assert len(sentry.breadcrumbs) == 1
        assert "Network error" in sentry.breadcrumbs[0]["message"]
        assert (
            sentry.breadcrumbs[0]["level"] == "error"
        )  # Level is "error" for network errors

    def test_record_retry(self):
        """Test recording retries."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)

        l0_sentry.record_retry(attempt=2, reason="rate_limit", is_network_error=False)

        assert len(sentry.breadcrumbs) == 1
        assert "Retry attempt 2" in sentry.breadcrumbs[0]["message"]

    def test_record_guardrail_violations(self):
        """Test recording guardrail violations."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)

        violations = [
            {"rule": "pii", "severity": "error", "message": "PII detected"},
            {"rule": "length", "severity": "warning", "message": "Too long"},
        ]
        l0_sentry.record_guardrail_violations(violations)

        # Should capture message for error severity
        assert len(sentry.messages) == 1
        # Message includes the violation message
        assert "PII detected" in sentry.messages[0]["message"]

    def test_record_guardrail_violations_with_debug_severity(self):
        """Test that debug/info severity levels don't raise ValueError."""
        from l0.monitoring.sentry import Sentry, SentryConfig

        sentry = MockSentryClient()

        # Test with debug severity - should not raise
        config = SentryConfig(min_guardrail_severity="debug")
        l0_sentry = Sentry(sentry, config=config)

        violations = [
            {"rule": "test", "severity": "debug", "message": "Debug message"},
            {"rule": "test2", "severity": "info", "message": "Info message"},
            {"rule": "test3", "severity": "warning", "message": "Warning message"},
        ]
        l0_sentry.record_guardrail_violations(violations)

        # All violations should be captured since min_severity is "debug"
        assert len(sentry.messages) == 3

    def test_record_guardrail_violations_with_info_severity(self):
        """Test filtering with info as minimum severity."""
        from l0.monitoring.sentry import Sentry, SentryConfig

        sentry = MockSentryClient()

        config = SentryConfig(min_guardrail_severity="info")
        l0_sentry = Sentry(sentry, config=config)

        violations = [
            {"rule": "test", "severity": "debug", "message": "Debug message"},
            {"rule": "test2", "severity": "info", "message": "Info message"},
            {"rule": "test3", "severity": "warning", "message": "Warning message"},
        ]
        l0_sentry.record_guardrail_violations(violations)

        # Only info and above should be captured (debug is below threshold)
        assert len(sentry.messages) == 2

    def test_record_drift(self):
        """Test recording drift detection."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)

        l0_sentry.record_drift(detected=True, types=["semantic", "format"])

        assert len(sentry.breadcrumbs) == 1
        assert "Drift detected" in sentry.breadcrumbs[0]["message"]

    def test_complete_stream(self):
        """Test completing a stream."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)
        # Record tokens to set the token count
        for _ in range(100):
            l0_sentry.record_token("x")

        l0_sentry.complete_stream(token_count=100)

        assert len(sentry.breadcrumbs) == 1
        assert "Stream completed" in sentry.breadcrumbs[0]["message"]

    def test_record_failure(self):
        """Test recording a failure."""
        from l0.monitoring import Sentry

        sentry = MockSentryClient()
        l0_sentry = Sentry(sentry)

        error = ValueError("Something went wrong")
        l0_sentry.record_failure(error, telemetry=None)

        assert len(sentry.exceptions) == 1
        assert sentry.exceptions[0]["error"] is error


class TestCreateSentryHandler:
    """Tests for create_sentry_handler factory."""

    def test_create_handler(self):
        """Test creating a Sentry handler."""
        from l0.monitoring import create_sentry_handler

        sentry = MockSentryClient()
        handler = create_sentry_handler(sentry)
        assert callable(handler)

    def test_handler_processes_events(self):
        """Test handler processes observability events."""
        from l0.monitoring import create_sentry_handler

        sentry = MockSentryClient()
        handler = create_sentry_handler(sentry)

        # Session start
        handler(
            ObservabilityEvent(
                type=ObservabilityEventType.SESSION_START,
                ts=1000.0,
                stream_id="test-1",
                meta={},
            )
        )

        # Should have breadcrumb for stream start
        assert len(sentry.breadcrumbs) >= 1

    def test_handler_captures_errors(self):
        """Test handler captures error events."""
        from l0.monitoring import create_sentry_handler

        sentry = MockSentryClient()
        handler = create_sentry_handler(sentry)

        handler(
            ObservabilityEvent(
                type=ObservabilityEventType.ERROR,
                ts=1000.0,
                stream_id="test-1",
                meta={"error": "Test error", "failure_type": "model"},
            )
        )

        # Should capture the error
        assert len(sentry.exceptions) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests for OpenTelemetry
# ─────────────────────────────────────────────────────────────────────────────


class TestSemanticAttributes:
    """Tests for SemanticAttributes constants."""

    def test_semantic_attributes_exist(self):
        """Test that semantic attributes are defined."""
        from l0.monitoring import SemanticAttributes

        assert SemanticAttributes.LLM_SYSTEM == "gen_ai.system"
        assert SemanticAttributes.LLM_REQUEST_MODEL == "gen_ai.request.model"
        assert SemanticAttributes.L0_SESSION_ID == "l0.session_id"
        assert SemanticAttributes.L0_RETRY_COUNT == "l0.retry.count"
        assert SemanticAttributes.L0_TIME_TO_FIRST_TOKEN == "l0.time_to_first_token_ms"


class TestNoOpSpan:
    """Tests for NoOpSpan class."""

    def test_noop_span_methods(self):
        """Test NoOpSpan methods don't raise."""
        from l0.monitoring import NoOpSpan

        span = NoOpSpan()

        # All methods should return self or None without raising
        assert span.set_attribute("key", "value") is span
        assert span.set_attributes({"a": 1}) is span
        assert span.add_event("test") is span
        assert span.set_status("OK") is span
        span.record_exception(ValueError("test"))
        span.end()
        assert span.is_recording() is False


class TestOpenTelemetry:
    """Tests for OpenTelemetry class."""

    def test_create_without_tracer(self):
        """Test creating OpenTelemetry without tracer/meter."""
        from l0.monitoring import OpenTelemetry

        otel = OpenTelemetry()
        assert otel is not None
        assert otel.get_active_streams() == 0

    def test_create_span_without_tracer(self):
        """Test creating span without tracer returns NoOpSpan."""
        from l0.monitoring import NoOpSpan, OpenTelemetry

        otel = OpenTelemetry()
        span = otel.create_span("test")
        assert isinstance(span, NoOpSpan)

    def test_record_token_no_op(self):
        """Test record_token without tracer."""
        from l0.monitoring import OpenTelemetry

        otel = OpenTelemetry()
        # Should not raise
        otel.record_token(span=None, content="hello")

    def test_record_retry_no_op(self):
        """Test record_retry without tracer."""
        from l0.monitoring import OpenTelemetry

        otel = OpenTelemetry()
        # Should not raise
        otel.record_retry(reason="rate_limit", attempt=1, span=None)

    def test_record_network_error_no_op(self):
        """Test record_network_error without tracer."""
        from l0.monitoring import OpenTelemetry

        otel = OpenTelemetry()
        # Should not raise
        otel.record_network_error(
            error=ConnectionError("test"),
            error_type="connection",
            span=None,
        )

    def test_record_guardrail_violation_no_op(self):
        """Test record_guardrail_violation without tracer."""
        from l0.monitoring import OpenTelemetry

        otel = OpenTelemetry()
        # Should not raise
        otel.record_guardrail_violation(
            violation={"rule": "pii", "severity": "error"},
            span=None,
        )

    def test_record_drift_no_op(self):
        """Test record_drift without tracer."""
        from l0.monitoring import OpenTelemetry

        otel = OpenTelemetry()
        # Should not raise
        otel.record_drift(drift_type="semantic", confidence=0.9, span=None)


class TestCreateOpenTelemetryHandler:
    """Tests for create_opentelemetry_handler factory."""

    def test_create_handler(self):
        """Test creating an OpenTelemetry handler."""
        from l0.monitoring import create_opentelemetry_handler

        handler = create_opentelemetry_handler()
        assert callable(handler)

    def test_handler_processes_session_start(self):
        """Test handler processes SESSION_START events."""
        from l0.monitoring import create_opentelemetry_handler

        handler = create_opentelemetry_handler()

        # Should not raise
        handler(
            ObservabilityEvent(
                type=ObservabilityEventType.SESSION_START,
                ts=1000.0,
                stream_id="test-1",
                meta={"attempt": 1, "is_retry": False},
            )
        )

    def test_handler_processes_retry_attempt(self):
        """Test handler processes RETRY_ATTEMPT events."""
        from l0.monitoring import create_opentelemetry_handler

        handler = create_opentelemetry_handler()

        handler(
            ObservabilityEvent(
                type=ObservabilityEventType.RETRY_ATTEMPT,
                ts=1000.0,
                stream_id="test-1",
                meta={"attempt": 2, "reason": "rate_limit"},
            )
        )

    def test_handler_processes_complete(self):
        """Test handler processes COMPLETE events."""
        from l0.monitoring import create_opentelemetry_handler

        handler = create_opentelemetry_handler()

        # Start session first
        handler(
            ObservabilityEvent(
                type=ObservabilityEventType.SESSION_START,
                ts=1000.0,
                stream_id="test-1",
                meta={},
            )
        )

        # Complete - without opentelemetry installed, should not raise
        # The handler gracefully handles missing opentelemetry module
        try:
            handler(
                ObservabilityEvent(
                    type=ObservabilityEventType.COMPLETE,
                    ts=2000.0,
                    stream_id="test-1",
                    meta={
                        "token_count": 100,
                        "content_length": 500,
                        "duration_ms": 1000,
                    },
                )
            )
        except ModuleNotFoundError:
            # Expected when opentelemetry is not installed
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Tests for Event Handler Combinators
# ─────────────────────────────────────────────────────────────────────────────


class TestEventHandlerCombinators:
    """Tests for event handler combinator functions."""

    def test_combine_events(self):
        """Test combining multiple handlers."""
        from l0.monitoring import combine_events

        results1 = []
        results2 = []

        def handler1(event: ObservabilityEvent) -> None:
            results1.append(event)

        def handler2(event: ObservabilityEvent) -> None:
            results2.append(event)

        combined = combine_events(handler1, handler2)

        event = ObservabilityEvent(
            type=ObservabilityEventType.TOKEN,
            ts=1000.0,
            stream_id="test",
            meta={},
        )
        combined(event)

        assert len(results1) == 1
        assert len(results2) == 1

    def test_filter_events(self):
        """Test filtering events by type."""
        from l0.monitoring import filter_events

        results = []

        def handler(event: ObservabilityEvent) -> None:
            results.append(event)

        # filter_events takes (types, handler) - types first
        filtered = filter_events([ObservabilityEventType.TOKEN], handler)

        # Token event - should pass
        filtered(
            ObservabilityEvent(
                type=ObservabilityEventType.TOKEN,
                ts=1000.0,
                stream_id="test",
                meta={},
            )
        )

        # Error event - should be filtered
        filtered(
            ObservabilityEvent(
                type=ObservabilityEventType.ERROR,
                ts=1001.0,
                stream_id="test",
                meta={},
            )
        )

        assert len(results) == 1
        assert results[0].type == ObservabilityEventType.TOKEN

    def test_exclude_events(self):
        """Test excluding events by type."""
        from l0.monitoring import exclude_events

        results = []

        def handler(event: ObservabilityEvent) -> None:
            results.append(event)

        # exclude_events takes (types, handler) - types first
        excluded = exclude_events([ObservabilityEventType.TOKEN], handler)

        # Token event - should be excluded
        excluded(
            ObservabilityEvent(
                type=ObservabilityEventType.TOKEN,
                ts=1000.0,
                stream_id="test",
                meta={},
            )
        )

        # Error event - should pass
        excluded(
            ObservabilityEvent(
                type=ObservabilityEventType.ERROR,
                ts=1001.0,
                stream_id="test",
                meta={},
            )
        )

        assert len(results) == 1
        assert results[0].type == ObservabilityEventType.ERROR

    def test_tap_events(self):
        """Test tapping events for side effects."""
        from l0.monitoring import tap_events

        tapped = []

        def tap_fn(event: ObservabilityEvent) -> None:
            tapped.append(event)

        # tap_events takes only a handler and returns a pass-through handler
        tapped_handler = tap_events(tap_fn)

        event = ObservabilityEvent(
            type=ObservabilityEventType.TOKEN,
            ts=1000.0,
            stream_id="test",
            meta={},
        )
        tapped_handler(event)

        # tap_events observes events
        assert len(tapped) == 1

    @pytest.mark.asyncio
    async def test_batch_events(self):
        """Test batching events."""
        from l0.monitoring import batch_events

        batches: list[list[ObservabilityEvent]] = []

        def handler(events: list[ObservabilityEvent]) -> None:
            batches.append(events)

        # batch_events takes (size, max_wait_seconds, handler)
        batched = batch_events(3, 1.0, handler)

        for i in range(5):
            batched(
                ObservabilityEvent(
                    type=ObservabilityEventType.TOKEN,
                    ts=1000.0 + i,
                    stream_id="test",
                    meta={"i": i},
                )
            )

        # First batch of 3 should be complete
        assert len(batches) >= 1
        assert len(batches[0]) == 3

    @pytest.mark.asyncio
    async def test_batch_events_timer_flush(self):
        """Test that partial batches flush after max_wait_seconds."""
        import asyncio

        from l0.monitoring import batch_events

        batches: list[list[ObservabilityEvent]] = []

        def handler(events: list[ObservabilityEvent]) -> None:
            batches.append(events.copy())

        # batch_events with short timeout
        batched = batch_events(10, 0.05, handler)  # batch size 10, 50ms timeout

        # Send only 2 events (less than batch size)
        for i in range(2):
            batched(
                ObservabilityEvent(
                    type=ObservabilityEventType.TOKEN,
                    ts=1000.0 + i,
                    stream_id="test",
                    meta={"i": i},
                )
            )

        # No batch should be flushed yet (not full, timer pending)
        assert len(batches) == 0

        # Wait for timer to trigger
        await asyncio.sleep(0.1)

        # Now the partial batch should be flushed
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_batch_events_no_event_loop(self):
        """Test batching events without event loop flushes immediately to avoid data loss."""
        from l0.monitoring import batch_events

        batches: list[list[ObservabilityEvent]] = []

        def handler(events: list[ObservabilityEvent]) -> None:
            batches.append(events)

        # batch_events takes (size, max_wait_seconds, handler)
        batched = batch_events(3, 1.0, handler)

        # Send 2 events (less than batch size)
        for i in range(2):
            batched(
                ObservabilityEvent(
                    type=ObservabilityEventType.TOKEN,
                    ts=1000.0 + i,
                    stream_id="test",
                    meta={"i": i},
                )
            )

        # Without event loop, events should be flushed immediately to avoid loss
        # Each event triggers an immediate flush since no timer can be scheduled
        assert len(batches) == 2

    def test_sample_events(self):
        """Test sampling events."""
        from l0.monitoring import sample_events

        results = []

        def handler(event: ObservabilityEvent) -> None:
            results.append(event)

        # sample_events takes (rate, handler)
        # Sample rate of 0 should filter everything
        sampled = sample_events(0.0, handler)

        for i in range(10):
            sampled(
                ObservabilityEvent(
                    type=ObservabilityEventType.TOKEN,
                    ts=1000.0 + i,
                    stream_id="test",
                    meta={},
                )
            )

        assert len(results) == 0

        # Sample rate of 1 should pass everything
        results.clear()
        sampled_all = sample_events(1.0, handler)

        for i in range(10):
            sampled_all(
                ObservabilityEvent(
                    type=ObservabilityEventType.TOKEN,
                    ts=1000.0 + i,
                    stream_id="test",
                    meta={},
                )
            )

        assert len(results) == 10
