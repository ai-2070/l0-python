"""Event Handler Utilities.

Helpers for combining and composing event handlers for the L0 observability pipeline.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from ..events import ObservabilityEvent

# Event handler type
EventHandler = Callable[[ObservabilityEvent], None]
BatchEventHandler = Callable[[list[ObservabilityEvent]], None]


def combine_events(*handlers: EventHandler | None) -> EventHandler:
    """Combine multiple event handlers into a single handler.

    This is the recommended way to use multiple observability integrations
    (OpenTelemetry, Sentry, custom loggers) together.

    Args:
        handlers: Event handlers to combine (None values are filtered out)

    Returns:
        A single event handler that calls all provided handlers

    Example:
        ```python
        from l0.monitoring import combine_events, Monitor

        monitor = Monitor()

        result = await l0.run(
            stream=lambda: client.chat.completions.create(...),
            on_event=combine_events(
                monitor.handle_event,
                lambda e: print(e.type),  # custom handler
            ),
        )
        ```
    """
    # Filter out None handlers
    valid_handlers = [h for h in handlers if h is not None]

    if len(valid_handlers) == 0:
        # Return no-op if no handlers
        return lambda event: None

    if len(valid_handlers) == 1:
        # Optimization: return single handler directly
        return valid_handlers[0]

    # Return combined handler that calls all handlers
    def combined_handler(event: ObservabilityEvent) -> None:
        for handler in valid_handlers:
            try:
                handler(event)
            except Exception as e:
                # Log but don't throw - one handler failing shouldn't break others
                print(f"Event handler error for {event.type}: {e}")

    return combined_handler


def filter_events(
    types: list[str],
    handler: EventHandler,
) -> EventHandler:
    """Create a filtered event handler that only receives specific event types.

    Args:
        types: Event types to include
        handler: Handler to call for matching events

    Returns:
        Filtered event handler

    Example:
        ```python
        from l0.monitoring import filter_events
        from l0.events import ObservabilityEventType

        error_handler = filter_events(
            [ObservabilityEventType.ERROR],
            lambda event: send_to_alert_system(event),
        )
        ```
    """
    type_set = set(types)

    def filtered_handler(event: ObservabilityEvent) -> None:
        if event.type in type_set:
            handler(event)

    return filtered_handler


def exclude_events(
    types: list[str],
    handler: EventHandler,
) -> EventHandler:
    """Create an event handler that excludes specific event types.

    Useful for filtering out noisy events like individual tokens.

    Args:
        types: Event types to exclude
        handler: Handler to call for non-excluded events

    Returns:
        Filtered event handler

    Example:
        ```python
        from l0.monitoring import exclude_events
        from l0.events import ObservabilityEventType

        quiet_handler = exclude_events(
            [ObservabilityEventType.TOKEN],  # Exclude token events
            lambda event: print(event.type),
        )
        ```
    """
    type_set = set(types)

    def excluded_handler(event: ObservabilityEvent) -> None:
        if event.type not in type_set:
            handler(event)

    return excluded_handler


def debounce_events(
    seconds: float,
    handler: EventHandler,
) -> EventHandler:
    """Create a debounced event handler for high-frequency events.

    Useful for token events when you want periodic updates instead of every token.

    Args:
        seconds: Debounce interval in seconds
        handler: Handler to call with latest event

    Returns:
        Debounced event handler

    Example:
        ```python
        from l0.monitoring import debounce_events

        throttled_logger = debounce_events(
            0.1,  # 100ms debounce
            lambda event: print(f"Latest: {event.type}"),
        )
        ```
    """
    last_call_time: float = 0
    pending_event: ObservabilityEvent | None = None
    timer_handle: asyncio.TimerHandle | None = None

    def debounced_handler(event: ObservabilityEvent) -> None:
        nonlocal last_call_time, pending_event, timer_handle

        current_time = time.time()
        pending_event = event

        # If enough time has passed, call immediately
        if current_time - last_call_time >= seconds:
            last_call_time = current_time
            handler(event)
            pending_event = None
        else:
            # Schedule a call for later if not already scheduled
            if timer_handle is None:
                try:
                    loop = asyncio.get_running_loop()

                    def flush() -> None:
                        nonlocal last_call_time, pending_event, timer_handle
                        if pending_event is not None:
                            last_call_time = time.time()
                            handler(pending_event)
                            pending_event = None
                        timer_handle = None

                    remaining = seconds - (current_time - last_call_time)
                    timer_handle = loop.call_later(remaining, flush)
                except RuntimeError:
                    # No event loop running, just call the handler
                    last_call_time = current_time
                    handler(event)
                    pending_event = None

    return debounced_handler


def batch_events(
    size: int,
    max_wait_seconds: float,
    handler: BatchEventHandler,
) -> EventHandler:
    """Create a batched event handler that collects events and processes them in batches.

    Args:
        size: Maximum batch size
        max_wait_seconds: Maximum time to wait before flushing partial batch
        handler: Handler to call with batched events

    Returns:
        Batching event handler

    Example:
        ```python
        from l0.monitoring import batch_events

        batched_handler = batch_events(
            10,   # Batch size
            1.0,  # Max wait time (seconds)
            lambda events: send_to_analytics(events),
        )
        ```
    """
    batch: list[ObservabilityEvent] = []
    timer_handle: asyncio.TimerHandle | None = None

    def flush() -> None:
        nonlocal batch, timer_handle
        if batch:
            handler(batch.copy())
            batch.clear()
        if timer_handle is not None:
            timer_handle.cancel()
            timer_handle = None

    def batched_handler(event: ObservabilityEvent) -> None:
        nonlocal batch, timer_handle

        batch.append(event)

        if len(batch) >= size:
            flush()
        elif timer_handle is None:
            try:
                loop = asyncio.get_running_loop()
                timer_handle = loop.call_later(max_wait_seconds, flush)
            except RuntimeError:
                # No event loop running, flush immediately to avoid losing events
                flush()

    return batched_handler


def sample_events(
    rate: float,
    handler: EventHandler,
) -> EventHandler:
    """Create a sampling event handler that only processes a fraction of events.

    Args:
        rate: Sampling rate between 0.0 and 1.0 (e.g., 0.1 = 10% of events)
        handler: Handler to call for sampled events

    Returns:
        Sampling event handler

    Example:
        ```python
        from l0.monitoring import sample_events

        sampled_handler = sample_events(
            0.1,  # Sample 10% of events
            lambda event: log_event(event),
        )
        ```
    """
    import random

    def sampled_handler(event: ObservabilityEvent) -> None:
        if random.random() < rate:
            handler(event)

    return sampled_handler


def tap_events(handler: EventHandler) -> EventHandler:
    """Create a pass-through handler that observes events without modifying them.

    Useful for logging or debugging without affecting the event flow.

    Args:
        handler: Handler to call for each event

    Returns:
        Pass-through event handler

    Example:
        ```python
        from l0.monitoring import tap_events, combine_events

        on_event = combine_events(
            tap_events(lambda e: print(f"DEBUG: {e.type}")),
            main_handler,
        )
        ```
    """

    def tap_handler(event: ObservabilityEvent) -> None:
        try:
            handler(event)
        except Exception:
            # Silently ignore errors in tap handlers
            pass

    return tap_handler


# ─────────────────────────────────────────────────────────────────────────────
# Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class Monitoring:
    """Scoped API for monitoring and event handler utilities.

    Provides utilities for combining, filtering, and transforming event handlers
    for the L0 observability pipeline.

    Usage:
        ```python
        from l0 import Monitoring

        # Combine multiple handlers
        combined = Monitoring.combine(
            handler1,
            handler2,
            lambda e: print(e.type),
        )

        # Filter events by type
        errors_only = Monitoring.filter(
            ["ERROR"],
            error_handler,
        )

        # Exclude noisy events
        quiet = Monitoring.exclude(
            ["TOKEN"],
            logger_handler,
        )

        # Batch events for efficient processing
        batched = Monitoring.batch(
            size=10,
            max_wait_seconds=1.0,
            handler=lambda events: send_to_analytics(events),
        )

        # Sample events for high-volume streams
        sampled = Monitoring.sample(0.1, handler)  # 10% sampling
        ```
    """

    @staticmethod
    def combine(*handlers: EventHandler | None) -> EventHandler:
        """Combine multiple event handlers into a single handler.

        Args:
            handlers: Event handlers to combine (None values are filtered out)

        Returns:
            A single event handler that calls all provided handlers
        """
        return combine_events(*handlers)

    @staticmethod
    def filter(
        types: list[str],
        handler: EventHandler,
    ) -> EventHandler:
        """Create a filtered event handler that only receives specific event types.

        Args:
            types: Event types to include
            handler: Handler to call for matching events

        Returns:
            Filtered event handler
        """
        return filter_events(types, handler)

    @staticmethod
    def exclude(
        types: list[str],
        handler: EventHandler,
    ) -> EventHandler:
        """Create an event handler that excludes specific event types.

        Args:
            types: Event types to exclude
            handler: Handler to call for non-excluded events

        Returns:
            Filtered event handler
        """
        return exclude_events(types, handler)

    @staticmethod
    def debounce(
        seconds: float,
        handler: EventHandler,
    ) -> EventHandler:
        """Create a debounced event handler for high-frequency events.

        Args:
            seconds: Debounce interval in seconds
            handler: Handler to call with latest event

        Returns:
            Debounced event handler
        """
        return debounce_events(seconds, handler)

    @staticmethod
    def batch(
        size: int,
        max_wait_seconds: float,
        handler: BatchEventHandler,
    ) -> EventHandler:
        """Create a batched event handler that collects events and processes them in batches.

        Args:
            size: Maximum batch size
            max_wait_seconds: Maximum time to wait before flushing partial batch
            handler: Handler to call with batched events

        Returns:
            Batching event handler
        """
        return batch_events(size, max_wait_seconds, handler)

    @staticmethod
    def sample(
        rate: float,
        handler: EventHandler,
    ) -> EventHandler:
        """Create a sampling event handler that only processes a fraction of events.

        Args:
            rate: Sampling rate between 0.0 and 1.0 (e.g., 0.1 = 10% of events)
            handler: Handler to call for sampled events

        Returns:
            Sampling event handler
        """
        return sample_events(rate, handler)

    @staticmethod
    def tap(handler: EventHandler) -> EventHandler:
        """Create a pass-through handler that observes events without modifying them.

        Args:
            handler: Handler to call for each event

        Returns:
            Pass-through event handler
        """
        return tap_events(handler)

    @staticmethod
    def opentelemetry(
        tracer: Any = None,
        meter: Any = None,
        *,
        service_name: str = "l0",
        trace_tokens: bool = False,
        record_token_content: bool = False,
        record_guardrail_violations: bool = True,
        default_attributes: dict[str, Any] | None = None,
    ) -> EventHandler:
        """Create an OpenTelemetry event handler.

        Args:
            tracer: OpenTelemetry tracer instance.
            meter: OpenTelemetry meter instance.
            service_name: Service name for spans.
            trace_tokens: Whether to create spans for individual tokens.
            record_token_content: Whether to record token content in spans.
            record_guardrail_violations: Whether to record guardrail violations.
            default_attributes: Custom attributes to add to all spans.

        Returns:
            Event handler function.
        """
        from .otel import OpenTelemetryConfig, create_opentelemetry_handler

        config = OpenTelemetryConfig(
            service_name=service_name,
            trace_tokens=trace_tokens,
            record_token_content=record_token_content,
            record_guardrail_violations=record_guardrail_violations,
            default_attributes=default_attributes or {},
        )
        return create_opentelemetry_handler(
            tracer=tracer,
            meter=meter,
            config=config,
        )

    @staticmethod
    def sentry(
        sentry: Any,
        *,
        capture_network_errors: bool = True,
        capture_guardrail_violations: bool = True,
        min_guardrail_severity: str = "error",
        breadcrumbs_for_tokens: bool = False,
        enable_tracing: bool = True,
        tags: dict[str, str] | None = None,
        environment: str | None = None,
    ) -> EventHandler:
        """Create a Sentry event handler.

        Args:
            sentry: Sentry client instance (import sentry_sdk).
            capture_network_errors: Whether to capture network errors.
            capture_guardrail_violations: Whether to capture guardrail violations.
            min_guardrail_severity: Minimum severity to capture.
            breadcrumbs_for_tokens: Whether to add breadcrumbs for tokens.
            enable_tracing: Whether to enable performance monitoring.
            tags: Custom tags to add to all events.
            environment: Environment name.

        Returns:
            Event handler function.
        """
        from .sentry import SentryConfig, create_sentry_handler

        config = SentryConfig(
            capture_network_errors=capture_network_errors,
            capture_guardrail_violations=capture_guardrail_violations,
            min_guardrail_severity=min_guardrail_severity,  # type: ignore[arg-type]
            breadcrumbs_for_tokens=breadcrumbs_for_tokens,
            enable_tracing=enable_tracing,
            tags=tags or {},
            environment=environment,
        )
        return create_sentry_handler(
            sentry=sentry,
            config=config,
        )
