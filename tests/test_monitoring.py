"""Tests for L0 monitoring system."""

from datetime import datetime, timezone

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
        monitor.handle_event(ObservabilityEvent(
            type=ObservabilityEventType.STREAM_INIT,
            ts=1000.0,
            stream_id="disabled-stream",
            meta={},
        ))

        assert monitor.get_telemetry() is None
