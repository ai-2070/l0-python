"""L0 Event Dispatcher.

Centralized event emission for all L0 lifecycle events.
- Adds ts, stream_id, meta automatically to all events
- Calls handlers via asyncio (fire-and-forget)
- Never throws from handler failures
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from uuid6 import uuid7

from ..events import ObservabilityEvent, ObservabilityEventType

# Event handler type
EventHandler = Callable[[ObservabilityEvent], None | Awaitable[None]]


class EventDispatcher:
    """Centralized event emission for L0.

    Usage:
        ```python
        from l0.monitoring import EventDispatcher

        dispatcher = EventDispatcher(meta={"user_id": "123"})

        def my_handler(event):
            print(f"Got event: {event.type}")

        dispatcher.on_event(my_handler)
        dispatcher.emit(ObservabilityEventType.SESSION_START, attempt=1)
        ```

    Features:
        - Automatically adds ts, stream_id, meta to all events
        - Handlers are called asynchronously (fire-and-forget)
        - Handler failures are silently ignored
        - Zero overhead when no handlers registered
    """

    def __init__(self, meta: dict[str, Any] | None = None) -> None:
        """Initialize event dispatcher.

        Args:
            meta: Default metadata to include in all events
        """
        self._handlers: list[EventHandler] = []
        self._stream_id = str(uuid7())
        self._meta: dict[str, Any] = dict(meta) if meta else {}

    @property
    def stream_id(self) -> str:
        """Get the stream ID for this dispatcher."""
        return self._stream_id

    @property
    def meta(self) -> dict[str, Any]:
        """Get the metadata for this dispatcher."""
        return self._meta

    @property
    def handler_count(self) -> int:
        """Get the number of registered handlers."""
        return len(self._handlers)

    def on_event(self, handler: EventHandler) -> None:
        """Register an event handler.

        Args:
            handler: Function to call for each event
        """
        self._handlers.append(handler)

    def off_event(self, handler: EventHandler) -> None:
        """Remove an event handler.

        Args:
            handler: Handler to remove
        """
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass  # Handler not found, ignore

    def emit(
        self,
        event_type: ObservabilityEventType,
        **payload: Any,
    ) -> None:
        """Emit an event to all handlers.

        Adds ts, stream_id, meta automatically.
        Calls handlers asynchronously via asyncio (fire-and-forget).
        Never throws from handler failures.

        Args:
            event_type: The type of event to emit
            **payload: Additional event data (merged into meta)
        """
        # Skip event creation if no handlers registered (zero overhead)
        if not self._handlers:
            return

        import time

        event = ObservabilityEvent(
            type=event_type,
            ts=time.time() * 1000,  # Unix timestamp in milliseconds
            stream_id=self._stream_id,
            meta={**self._meta, **payload},
        )

        # Fire handlers asynchronously
        # Snapshot handlers to avoid issues if handlers modify the list during dispatch
        for handler in list(self._handlers):
            self._schedule_handler(handler, event)

    def emit_sync(
        self,
        event_type: ObservabilityEventType,
        **payload: Any,
    ) -> None:
        """Emit an event synchronously (for critical path events).

        Use sparingly - prefer emit() for most cases.

        Args:
            event_type: The type of event to emit
            **payload: Additional event data (merged into meta)
        """
        # Skip event creation if no handlers registered (zero overhead)
        if not self._handlers:
            return

        import time

        event = ObservabilityEvent(
            type=event_type,
            ts=time.time() * 1000,
            stream_id=self._stream_id,
            meta={**self._meta, **payload},
        )

        # Call handlers synchronously
        for handler in list(self._handlers):
            try:
                result = handler(event)
                # Handle async handlers
                if asyncio.iscoroutine(result):
                    # Schedule the coroutine but don't wait
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._run_async_handler(result))
                    except RuntimeError:
                        # No event loop running, skip async handler
                        pass
            except Exception:
                # Silently ignore handler errors
                pass

    def _schedule_handler(
        self,
        handler: EventHandler,
        event: ObservabilityEvent,
    ) -> None:
        """Schedule a handler to be called asynchronously."""
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(self._call_handler, handler, event)
        except RuntimeError:
            # No event loop running, call synchronously
            self._call_handler(handler, event)

    def _call_handler(
        self,
        handler: EventHandler,
        event: ObservabilityEvent,
    ) -> None:
        """Call a handler with error handling."""
        try:
            result = handler(event)
            # Handle async handlers
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._run_async_handler(result))
                except RuntimeError:
                    # No event loop running, skip async handler
                    pass
        except Exception:
            # Silently ignore handler errors - fire and forget
            pass

    async def _run_async_handler(self, coro: Awaitable[None]) -> None:
        """Run an async handler with error handling."""
        try:
            await coro
        except Exception:
            # Silently ignore async handler errors
            pass


def create_event_dispatcher(meta: dict[str, Any] | None = None) -> EventDispatcher:
    """Create an event dispatcher with the given meta.

    Args:
        meta: Default metadata to include in all events

    Returns:
        EventDispatcher instance
    """
    return EventDispatcher(meta)
