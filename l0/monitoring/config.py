"""Monitoring configuration with Pydantic models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    """Sampling configuration for telemetry collection.

    Attributes:
        rate: Sample rate between 0.0 and 1.0 (1.0 = 100% sampling)
        min_duration: Only sample requests longer than this (seconds)
        sample_errors: Always sample error cases regardless of rate
        sample_slow: Always sample slow requests regardless of rate
        slow_threshold: Threshold for slow request detection (seconds)
    """

    rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_duration: float = Field(default=0.0, ge=0.0)
    sample_errors: bool = True
    sample_slow: bool = True
    slow_threshold: float = Field(default=5.0, ge=0.0)


class MetricsConfig(BaseModel):
    """Configuration for metrics collection.

    Attributes:
        collect_tokens: Collect token counts and rates
        collect_timing: Collect timing metrics (TTFT, latency)
        collect_retries: Collect retry information
        collect_guardrails: Collect guardrail violations
        collect_errors: Collect error information
        inter_token_latency: Calculate inter-token latency (adds overhead)
        percentiles: Percentiles to calculate for aggregated metrics
    """

    collect_tokens: bool = True
    collect_timing: bool = True
    collect_retries: bool = True
    collect_guardrails: bool = True
    collect_errors: bool = True
    inter_token_latency: bool = False
    percentiles: list[float] = Field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])


class MonitoringConfig(BaseModel):
    """Main monitoring configuration.

    Usage:
        ```python
        from l0.monitoring import MonitoringConfig, Monitor

        # Default config
        config = MonitoringConfig()

        # Custom config
        config = MonitoringConfig(
            enabled=True,
            sampling=SamplingConfig(rate=0.1, sample_errors=True),
            metrics=MetricsConfig(inter_token_latency=True),
            buffer_size=1000,
        )

        monitor = Monitor(config)
        ```

    Attributes:
        enabled: Enable/disable monitoring globally
        sampling: Sampling configuration
        metrics: Metrics collection configuration
        buffer_size: Max number of telemetry records to buffer
        flush_interval: Auto-flush interval in seconds (0 = disabled)
        log_level: Log level for monitoring output
    """

    enabled: bool = True
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    buffer_size: int = Field(default=100, ge=1)
    flush_interval: float = Field(default=0.0, ge=0.0)
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    @classmethod
    def default(cls) -> MonitoringConfig:
        """Get default monitoring configuration."""
        return cls()

    @classmethod
    def production(cls) -> MonitoringConfig:
        """Get production-optimized configuration.

        - 10% sampling rate
        - Always samples errors and slow requests
        - No inter-token latency (reduces overhead)
        """
        return cls(
            sampling=SamplingConfig(
                rate=0.1,
                sample_errors=True,
                sample_slow=True,
                slow_threshold=5.0,
            ),
            metrics=MetricsConfig(
                inter_token_latency=False,
            ),
            buffer_size=1000,
            flush_interval=30.0,
        )

    @classmethod
    def development(cls) -> MonitoringConfig:
        """Get development configuration.

        - 100% sampling
        - Full metrics including inter-token latency
        - Debug logging
        """
        return cls(
            sampling=SamplingConfig(rate=1.0),
            metrics=MetricsConfig(inter_token_latency=True),
            log_level="debug",
        )

    @classmethod
    def minimal(cls) -> MonitoringConfig:
        """Get minimal configuration.

        - Errors only
        - No timing metrics
        - Low overhead
        """
        return cls(
            sampling=SamplingConfig(
                rate=0.0,
                sample_errors=True,
                sample_slow=False,
            ),
            metrics=MetricsConfig(
                collect_tokens=False,
                collect_timing=False,
                collect_retries=False,
                collect_guardrails=False,
                collect_errors=True,
            ),
        )
