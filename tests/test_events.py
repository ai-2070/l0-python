"""Tests for l0.events module."""

import pytest

from l0.events import EventBus, ObservabilityEvent, ObservabilityEventType


class TestObservabilityEventType:
    def test_event_types_exist(self):
        assert ObservabilityEventType.SESSION_START == "SESSION_START"
        assert ObservabilityEventType.STREAM_INIT == "STREAM_INIT"
        assert ObservabilityEventType.RETRY_ATTEMPT == "RETRY_ATTEMPT"
        assert ObservabilityEventType.FALLBACK_START == "FALLBACK_START"
        assert ObservabilityEventType.GUARDRAIL_PHASE_START == "GUARDRAIL_PHASE_START"
        assert ObservabilityEventType.COMPLETE == "COMPLETE"
        assert ObservabilityEventType.ERROR == "ERROR"


class TestObservabilityEvent:
    def test_create_event(self):
        event = ObservabilityEvent(
            type=ObservabilityEventType.STREAM_INIT,
            ts=1234567890.0,
            stream_id="test-id",
            context={"requestId": "req-123"},
            meta={"key": "value"},
        )
        assert event.type == ObservabilityEventType.STREAM_INIT
        assert event.ts == 1234567890.0
        assert event.stream_id == "test-id"
        assert event.context == {"requestId": "req-123"}
        assert event.meta == {"key": "value"}


class TestEventBus:
    def test_creates_stream_id(self):
        bus = EventBus()
        assert bus.stream_id is not None
        assert len(bus.stream_id) == 36  # UUID format

    def test_stream_id_is_uuid7(self):
        """Stream ID should be a valid UUIDv7."""
        bus = EventBus()
        # UUIDv7 format: xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx
        parts = bus.stream_id.split("-")
        assert len(parts) == 5
        assert parts[2][0] == "7"  # Version 7

    def test_emit_without_handler(self):
        """Emit should not fail without handler."""
        bus = EventBus()
        # Should not raise
        bus.emit(ObservabilityEventType.STREAM_INIT)

    def test_emit_with_handler(self):
        events = []

        def handler(event: ObservabilityEvent):
            events.append(event)

        bus = EventBus(handler=handler)
        bus.emit(ObservabilityEventType.STREAM_INIT)

        assert len(events) == 1
        assert events[0].type == ObservabilityEventType.STREAM_INIT
        assert events[0].stream_id == bus.stream_id

    def test_emit_with_context(self):
        events = []

        def handler(event: ObservabilityEvent):
            events.append(event)

        bus = EventBus(handler=handler, context={"session": "abc"})
        bus.emit(ObservabilityEventType.RETRY_ATTEMPT, attempt=3)

        assert len(events) == 1
        assert events[0].context["session"] == "abc"
        assert events[0].meta["attempt"] == 3

    def test_emit_timestamp_in_milliseconds(self):
        events = []

        def handler(event: ObservabilityEvent):
            events.append(event)

        bus = EventBus(handler=handler)
        bus.emit(ObservabilityEventType.STREAM_INIT)

        # Timestamp should be in milliseconds (> 1 trillion)
        assert events[0].ts > 1_000_000_000_000
