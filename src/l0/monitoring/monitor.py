"""Monitor class for collecting L0 telemetry."""

from __future__ import annotations

import random
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..events import ObservabilityEvent, ObservabilityEventType
from ..types import ErrorCategory
from .config import MonitoringConfig
from .telemetry import (
    ErrorInfo,
    GuardrailInfo,
    Metrics,
    RetryInfo,
    Telemetry,
    TimingInfo,
)


class Monitor:
    """Collects telemetry from L0 observability events.

    Usage:
        ```python
        from l0.monitoring import Monitor, MonitoringConfig

        # Simple usage
        monitor = Monitor()

        result = await l0.run(
            stream=lambda: client.chat.completions.create(...),
            on_event=monitor.handle_event,
        )

        # Get telemetry for last stream
        telemetry = monitor.get_telemetry()

        # Get all telemetry
        all_telemetry = monitor.get_all_telemetry()

        # With custom config
        config = MonitoringConfig.production()
        monitor = Monitor(config)

        # With callback for real-time export
        def on_complete(telemetry: Telemetry) -> None:
            print(f"Stream completed: {telemetry.stream_id}")

        monitor = Monitor(on_complete=on_complete)
        ```

    Attributes:
        config: Monitoring configuration
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
        *,
        on_complete: Callable[[Telemetry], None] | None = None,
    ) -> None:
        """Initialize monitor.

        Args:
            config: Monitoring configuration (defaults to MonitoringConfig.default())
            on_complete: Callback invoked when a stream completes
        """
        self.config = config or MonitoringConfig.default()
        self._on_complete = on_complete
        self._telemetry: dict[str, Telemetry] = {}
        self._buffer: list[Telemetry] = []
        self._current_stream_id: str | None = None
        self._last_token_time: float | None = None
        self._should_sample: dict[str, bool] = {}

    def handle_event(self, event: ObservabilityEvent) -> None:
        """Handle an L0 observability event.

        This is the main entry point - pass this to l0.run(on_event=...).

        Args:
            event: The observability event to process
        """
        if not self.config.enabled:
            return

        stream_id = event.stream_id

        # Determine if we should sample this stream
        if stream_id not in self._should_sample:
            self._should_sample[stream_id] = self._decide_sampling()

        if not self._should_sample.get(stream_id, True):
            # Check if we should force-sample based on event type
            if not self._should_force_sample(event):
                return

        # Get or create telemetry for this stream
        if stream_id not in self._telemetry:
            self._telemetry[stream_id] = Telemetry(stream_id=stream_id)

        telemetry = self._telemetry[stream_id]
        self._current_stream_id = stream_id

        # Route to appropriate handler
        self._process_event(event, telemetry)

    def _decide_sampling(self) -> bool:
        """Decide whether to sample based on config."""
        return random.random() < self.config.sampling.rate

    def _should_force_sample(self, event: ObservabilityEvent) -> bool:
        """Check if event should force sampling."""
        # Always sample errors if configured
        if self.config.sampling.sample_errors:
            if event.type in (
                ObservabilityEventType.ERROR,
                ObservabilityEventType.RETRY_GIVE_UP,
            ):
                self._should_sample[event.stream_id] = True
                return True

        return False

    def _process_event(self, event: ObservabilityEvent, telemetry: Telemetry) -> None:
        """Process an event and update telemetry."""
        event_type = event.type
        meta = event.meta

        # Session events
        if event_type == ObservabilityEventType.SESSION_START:
            telemetry.session_id = meta.get("session_id")
            telemetry.timing.started_at = datetime.now(timezone.utc)
            if "model" in meta:
                telemetry.model = meta["model"]

        elif event_type == ObservabilityEventType.STREAM_INIT:
            if telemetry.timing.started_at is None:
                telemetry.timing.started_at = datetime.now(timezone.utc)
            if "model" in meta:
                telemetry.model = meta["model"]

        # Token events - track timing
        elif event_type == ObservabilityEventType.STREAM_READY:
            # First token received
            if self.config.metrics.collect_timing:
                now = time.time()
                if telemetry.timing.started_at:
                    start_ts = telemetry.timing.started_at.timestamp()
                    telemetry.timing.time_to_first_token = now - start_ts
                self._last_token_time = now

        # We need to track tokens for metrics
        # The runtime calls append_token which we can't intercept directly,
        # so we rely on state from COMPLETE event or count via specific events

        # Retry events
        elif event_type == ObservabilityEventType.RETRY_START:
            if self.config.metrics.collect_retries:
                telemetry.retries.max_attempts = meta.get("max_attempts", 1)

        elif event_type == ObservabilityEventType.RETRY_ATTEMPT:
            if self.config.metrics.collect_retries:
                telemetry.retries.attempt = meta.get("attempt", 1)
                telemetry.retries.total_retries += 1
                category = meta.get("category")
                if category == ErrorCategory.NETWORK or category == "network":
                    telemetry.retries.network_retries += 1
                else:
                    telemetry.retries.model_retries += 1
                if "error" in meta:
                    telemetry.retries.last_error = str(meta["error"])
                if "category" in meta:
                    cat = meta["category"]
                    if isinstance(cat, ErrorCategory):
                        telemetry.retries.last_error_category = cat
                    elif isinstance(cat, str):
                        try:
                            telemetry.retries.last_error_category = ErrorCategory(cat)
                        except ValueError:
                            pass

        elif event_type == ObservabilityEventType.RETRY_GIVE_UP:
            if self.config.metrics.collect_retries:
                if "error" in meta:
                    telemetry.retries.last_error = str(meta["error"])

        # Guardrail events
        elif event_type == ObservabilityEventType.GUARDRAIL_RULE_START:
            if self.config.metrics.collect_guardrails:
                telemetry.guardrails.rules_checked += 1

        elif event_type == ObservabilityEventType.GUARDRAIL_RULE_RESULT:
            if self.config.metrics.collect_guardrails:
                violations = meta.get("violations", [])
                if violations:
                    telemetry.guardrails.passed = False
                    for v in violations:
                        if hasattr(v, "model_dump"):
                            telemetry.guardrails.violations.append(v.model_dump())
                        elif isinstance(v, dict):
                            telemetry.guardrails.violations.append(v)
                        else:
                            telemetry.guardrails.violations.append({"message": str(v)})

        # Error events
        elif event_type == ObservabilityEventType.ERROR:
            if self.config.metrics.collect_errors:
                telemetry.error.occurred = True
                if "error" in meta:
                    err = meta["error"]
                    telemetry.error.message = str(err)
                    if hasattr(err, "__traceback__"):
                        telemetry.error.stack = "".join(
                            traceback.format_exception(
                                type(err), err, err.__traceback__
                            )
                        )
                if "category" in meta:
                    cat = meta["category"]
                    if isinstance(cat, ErrorCategory):
                        telemetry.error.category = cat
                    elif isinstance(cat, str):
                        try:
                            telemetry.error.category = ErrorCategory(cat)
                        except ValueError:
                            pass
                if "code" in meta:
                    telemetry.error.code = meta["code"]
                telemetry.error.recoverable = meta.get("recoverable", False)

                # Force sample errors
                if self.config.sampling.sample_errors:
                    self._should_sample[telemetry.stream_id] = True

        # Completion events
        elif event_type == ObservabilityEventType.COMPLETE:
            telemetry.completed = True
            telemetry.timing.completed_at = datetime.now(timezone.utc)

            # Extract state info if available
            if "state" in meta:
                state = meta["state"]
                if hasattr(state, "token_count"):
                    telemetry.metrics = Metrics(token_count=state.token_count)
                if hasattr(state, "content"):
                    telemetry.content_length = len(state.content)
                if hasattr(state, "duration") and state.duration is not None:
                    telemetry.timing.duration = state.duration

            # Calculate duration if not set
            if telemetry.timing.duration is None:
                if telemetry.timing.started_at and telemetry.timing.completed_at:
                    telemetry.timing.duration = (
                        telemetry.timing.completed_at.timestamp()
                        - telemetry.timing.started_at.timestamp()
                    )

            # Check slow threshold for force-sampling
            if self.config.sampling.sample_slow:
                if (
                    telemetry.timing.duration
                    and telemetry.timing.duration > self.config.sampling.slow_threshold
                ):
                    self._should_sample[telemetry.stream_id] = True

            # Finalize and notify
            self._finalize_telemetry(telemetry)

        elif event_type == ObservabilityEventType.ABORT_COMPLETED:
            telemetry.aborted = True
            telemetry.timing.completed_at = datetime.now(timezone.utc)
            self._finalize_telemetry(telemetry)

        # Store metadata from any event
        if "model" in meta and not telemetry.model:
            telemetry.model = meta["model"]

    def _finalize_telemetry(self, telemetry: Telemetry) -> None:
        """Finalize telemetry and trigger callbacks."""
        # Calculate final metrics
        telemetry.finalize()

        # Add to buffer
        self._buffer.append(telemetry)
        if len(self._buffer) > self.config.buffer_size:
            self._buffer.pop(0)

        # Trigger callback
        if self._on_complete:
            self._on_complete(telemetry)

    def get_telemetry(self, stream_id: str | None = None) -> Telemetry | None:
        """Get telemetry for a specific stream or the most recent.

        Args:
            stream_id: Stream ID to get telemetry for (None = most recent)

        Returns:
            Telemetry for the stream, or None if not found
        """
        if stream_id:
            return self._telemetry.get(stream_id)

        if self._current_stream_id:
            return self._telemetry.get(self._current_stream_id)

        if self._buffer:
            return self._buffer[-1]

        return None

    def get_all_telemetry(self) -> list[Telemetry]:
        """Get all buffered telemetry records.

        Returns:
            List of telemetry records
        """
        return list(self._buffer)

    def clear(self) -> None:
        """Clear all telemetry data."""
        self._telemetry.clear()
        self._buffer.clear()
        self._current_stream_id = None
        self._should_sample.clear()

    def get_aggregate_metrics(self) -> dict[str, Any]:
        """Get aggregate metrics across all buffered telemetry.

        Returns:
            Dictionary with aggregate metrics
        """
        if not self._buffer:
            return {}

        total_tokens = sum(t.metrics.token_count for t in self._buffer)
        total_duration = sum(t.timing.duration or 0 for t in self._buffer)
        total_retries = sum(t.retries.total_retries for t in self._buffer)
        error_count = sum(1 for t in self._buffer if t.error.occurred)
        completed_count = sum(1 for t in self._buffer if t.completed)

        ttft_values = [
            t.metrics.time_to_first_token
            for t in self._buffer
            if t.metrics.time_to_first_token is not None
        ]

        tokens_per_sec_values = [
            t.metrics.tokens_per_second
            for t in self._buffer
            if t.metrics.tokens_per_second is not None
        ]

        return {
            "count": len(self._buffer),
            "total_tokens": total_tokens,
            "total_duration": total_duration,
            "total_retries": total_retries,
            "error_count": error_count,
            "error_rate": error_count / len(self._buffer) if self._buffer else 0,
            "completed_count": completed_count,
            "completion_rate": completed_count / len(self._buffer)
            if self._buffer
            else 0,
            "avg_tokens": total_tokens / len(self._buffer) if self._buffer else 0,
            "avg_duration": total_duration / len(self._buffer) if self._buffer else 0,
            "avg_ttft": sum(ttft_values) / len(ttft_values) if ttft_values else None,
            "avg_tokens_per_sec": (
                sum(tokens_per_sec_values) / len(tokens_per_sec_values)
                if tokens_per_sec_values
                else None
            ),
        }
