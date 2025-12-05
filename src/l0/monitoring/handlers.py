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
            [ObservabilityEventType.ERROR, ObservabilityEventType.NETWORK_ERROR],
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
                # No event loop running, flush immediately when batch is full
                pass

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
