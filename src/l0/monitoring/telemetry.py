"""Telemetry data structures for L0 monitoring."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..types import ErrorCategory


class TimingInfo(BaseModel):
    """Timing information for a stream.

    All times are in seconds (float), matching Python conventions.

    Attributes:
        started_at: When the stream started (ISO timestamp)
        completed_at: When the stream completed (ISO timestamp)
        duration: Total duration in seconds
        time_to_first_token: Time to first token (TTFT) in seconds
        inter_token_latencies: List of inter-token latencies in seconds
    """

    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration: float | None = None
    time_to_first_token: float | None = None
    inter_token_latencies: list[float] = Field(default_factory=list)


class RetryInfo(BaseModel):
    """Retry information for a stream.

    Attributes:
        attempt: Current attempt number (1-indexed)
        max_attempts: Maximum attempts configured
        model_retries: Number of model error retries
        network_retries: Number of network error retries
        total_retries: Total number of retries
        last_error: Last error message
        last_error_category: Category of last error
    """

    attempt: int = 1
    max_attempts: int = 1
    model_retries: int = 0
    network_retries: int = 0
    total_retries: int = 0
    last_error: str | None = None
    last_error_category: ErrorCategory | None = None


class GuardrailInfo(BaseModel):
    """Guardrail information for a stream.

    Attributes:
        rules_checked: Number of rules checked
        violations: List of violation details
        passed: Whether all guardrails passed
    """

    rules_checked: int = 0
    violations: list[dict[str, Any]] = Field(default_factory=list)
    passed: bool = True


class ErrorInfo(BaseModel):
    """Error information for a stream.

    Attributes:
        occurred: Whether an error occurred
        message: Error message
        category: Error category
        code: Error code
        stack: Stack trace (if available)
        recoverable: Whether the error was recoverable
    """

    occurred: bool = False
    message: str | None = None
    category: ErrorCategory | None = None
    code: str | None = None
    stack: str | None = None
    recoverable: bool = False


class Metrics(BaseModel):
    """Pre-calculated metrics for a stream.

    Attributes:
        token_count: Total number of tokens
        tokens_per_second: Token generation rate
        time_to_first_token: TTFT in seconds
        avg_inter_token_latency: Average inter-token latency
        p50_inter_token_latency: 50th percentile inter-token latency
        p90_inter_token_latency: 90th percentile inter-token latency
        p99_inter_token_latency: 99th percentile inter-token latency
    """

    token_count: int = 0
    tokens_per_second: float | None = None
    time_to_first_token: float | None = None
    avg_inter_token_latency: float | None = None
    p50_inter_token_latency: float | None = None
    p90_inter_token_latency: float | None = None
    p99_inter_token_latency: float | None = None

    @classmethod
    def calculate(
        cls,
        token_count: int,
        duration: float | None,
        time_to_first_token: float | None,
        inter_token_latencies: list[float],
    ) -> Metrics:
        """Calculate metrics from raw data.

        Args:
            token_count: Total number of tokens
            duration: Total duration in seconds
            time_to_first_token: TTFT in seconds
            inter_token_latencies: List of inter-token latencies

        Returns:
            Calculated metrics
        """
        tokens_per_second = None
        if duration and duration > 0 and token_count > 0:
            tokens_per_second = token_count / duration

        avg_latency = None
        p50_latency = None
        p90_latency = None
        p99_latency = None

        if inter_token_latencies:
            avg_latency = sum(inter_token_latencies) / len(inter_token_latencies)
            sorted_latencies = sorted(inter_token_latencies)
            n = len(sorted_latencies)
            p50_latency = sorted_latencies[int(n * 0.5)]
            p90_latency = sorted_latencies[int(n * 0.9)] if n >= 10 else None
            p99_latency = sorted_latencies[int(n * 0.99)] if n >= 100 else None

        return cls(
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            time_to_first_token=time_to_first_token,
            avg_inter_token_latency=avg_latency,
            p50_inter_token_latency=p50_latency,
            p90_inter_token_latency=p90_latency,
            p99_inter_token_latency=p99_latency,
        )


class Telemetry(BaseModel):
    """Complete telemetry data for a stream.

    Usage:
        ```python
        from l0.monitoring import Monitoring

        monitoring = Monitoring()
        result = await l0.run(..., on_event=monitoring.handle_event)

        telemetry = monitoring.get_telemetry()
        print(f"Stream ID: {telemetry.stream_id}")
        print(f"Tokens: {telemetry.metrics.token_count}")
        print(f"TTFT: {telemetry.metrics.time_to_first_token}s")
        print(f"Tokens/sec: {telemetry.metrics.tokens_per_second}")
        ```

    Attributes:
        stream_id: Unique stream identifier
        session_id: Session identifier (if grouped)
        model: Model name/identifier
        timing: Timing information
        retries: Retry information
        guardrails: Guardrail information
        error: Error information
        metrics: Pre-calculated metrics
        metadata: Additional custom metadata
        content_length: Length of generated content
        completed: Whether the stream completed successfully
        aborted: Whether the stream was aborted
    """

    stream_id: str
    session_id: str | None = None
    model: str | None = None
    timing: TimingInfo = Field(default_factory=TimingInfo)
    retries: RetryInfo = Field(default_factory=RetryInfo)
    guardrails: GuardrailInfo = Field(default_factory=GuardrailInfo)
    error: ErrorInfo = Field(default_factory=ErrorInfo)
    metrics: Metrics = Field(default_factory=Metrics)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_length: int = 0
    completed: bool = False
    aborted: bool = False

    def finalize(self) -> Telemetry:
        """Finalize telemetry by calculating metrics.

        Call this after stream completion to compute derived metrics.

        Returns:
            Self with metrics calculated
        """
        self.metrics = Metrics.calculate(
            token_count=self.metrics.token_count,
            duration=self.timing.duration,
            time_to_first_token=self.timing.time_to_first_token,
            inter_token_latencies=self.timing.inter_token_latencies,
        )
        return self
