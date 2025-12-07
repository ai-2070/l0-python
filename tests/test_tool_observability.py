"""Tests for tool call observability events.

L0 detects tool calls and emits observability events:
- TOOL_REQUESTED: Tool call detected
- TOOL_START: Tool execution began
- TOOL_COMPLETED: Tool lifecycle finished

The on_tool_call callback fires when tool calls are detected.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from l0.events import ObservabilityEventType
from l0.runtime import _internal_run
from l0.types import Event, EventType


class TestOnToolCallCallback:
    """Tests for on_tool_call lifecycle callback."""

    @pytest.mark.asyncio
    async def test_on_tool_call_fires_for_tool_call_event(self) -> None:
        """on_tool_call should fire when a tool call event is detected."""
        on_tool_call = MagicMock()

        async def stream_with_tool_call():
            yield Event(type=EventType.TOKEN, text="Before tool call. ")
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "Seattle"}',
                },
            )
            yield Event(type=EventType.TOKEN, text="After tool call.")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_tool_call,
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert on_tool_call.call_count == 1
        on_tool_call.assert_called_once_with(
            "get_weather",
            "call_123",
            {"location": "Seattle"},
        )
        assert result.state.completed

    @pytest.mark.asyncio
    async def test_on_tool_call_fires_for_buffered_tool_calls(self) -> None:
        """on_tool_call should fire for buffered tool calls at stream end."""
        on_tool_call = MagicMock()

        async def stream_with_chunked_tool_call():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": "call_456",
                    "name": "search_web",
                    "arguments": '{"query":',
                },
            )
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "arguments": ' "L0 library"}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_chunked_tool_call,
            buffer_tool_calls=True,
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert on_tool_call.call_count == 1
        on_tool_call.assert_called_once_with(
            "search_web",
            "call_456",
            {"query": "L0 library"},
        )

    @pytest.mark.asyncio
    async def test_on_tool_call_multiple_tool_calls(self) -> None:
        """on_tool_call should fire for each tool call."""
        on_tool_call = MagicMock()

        async def stream_with_multiple_tools():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "multi_1",
                    "name": "tool_a",
                    "arguments": '{"a": 1}',
                },
            )
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "multi_2",
                    "name": "tool_b",
                    "arguments": '{"b": 2}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_multiple_tools,
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert on_tool_call.call_count == 2
        calls = on_tool_call.call_args_list
        assert calls[0][0] == ("tool_a", "multi_1", {"a": 1})
        assert calls[1][0] == ("tool_b", "multi_2", {"b": 2})

    @pytest.mark.asyncio
    async def test_on_tool_call_with_empty_arguments(self) -> None:
        """on_tool_call should work with empty arguments."""
        on_tool_call = MagicMock()

        async def stream_with_no_args_tool():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "empty_args",
                    "name": "no_params",
                    "arguments": "",
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_no_args_tool,
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert on_tool_call.call_count == 1
        on_tool_call.assert_called_once_with(
            "no_params",
            "empty_args",
            {},
        )

    @pytest.mark.asyncio
    async def test_on_tool_call_not_fired_for_non_tool_events(self) -> None:
        """on_tool_call should not fire for token or other events."""
        on_tool_call = MagicMock()

        async def stream_without_tools():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.TOKEN, text=" World")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_without_tools,
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert on_tool_call.call_count == 0
        assert result.state.completed


class TestToolObservabilityEvents:
    """Tests for tool-related observability events."""

    @pytest.mark.asyncio
    async def test_tool_requested_event_emitted(self) -> None:
        """TOOL_REQUESTED event should be emitted for tool calls."""
        events: list[Any] = []

        def on_event(event: Any) -> None:
            events.append(event)

        async def stream_with_tool():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "call_req",
                    "name": "test_tool",
                    "arguments": "{}",
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_tool,
            on_event=on_event,
        )

        async for _ in result:
            pass

        tool_events = [
            e for e in events if e.type == ObservabilityEventType.TOOL_REQUESTED
        ]
        assert len(tool_events) == 1
        assert tool_events[0].meta["toolName"] == "test_tool"
        assert tool_events[0].meta["toolCallId"] == "call_req"

    @pytest.mark.asyncio
    async def test_tool_start_event_emitted(self) -> None:
        """TOOL_START event should be emitted for tool calls."""
        events: list[Any] = []

        def on_event(event: Any) -> None:
            events.append(event)

        async def stream_with_tool():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "call_start",
                    "name": "start_tool",
                    "arguments": "{}",
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_tool,
            on_event=on_event,
        )

        async for _ in result:
            pass

        tool_events = [e for e in events if e.type == ObservabilityEventType.TOOL_START]
        assert len(tool_events) == 1
        assert tool_events[0].meta["toolName"] == "start_tool"
        assert tool_events[0].meta["toolCallId"] == "call_start"

    @pytest.mark.asyncio
    async def test_tool_completed_event_emitted(self) -> None:
        """TOOL_COMPLETED event should be emitted at stream end."""
        events: list[Any] = []

        def on_event(event: Any) -> None:
            events.append(event)

        async def stream_with_tool():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "call_complete",
                    "name": "complete_tool",
                    "arguments": "{}",
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_tool,
            on_event=on_event,
        )

        async for _ in result:
            pass

        tool_events = [
            e for e in events if e.type == ObservabilityEventType.TOOL_COMPLETED
        ]
        assert len(tool_events) == 1
        assert tool_events[0].meta["toolCallId"] == "call_complete"
        assert tool_events[0].meta["status"] == "success"


class TestToolCallsWithBuffering:
    """Tests for tool calls with buffering enabled."""

    @pytest.mark.asyncio
    async def test_buffered_tool_calls_emit_observability_events(self) -> None:
        """Buffered tool calls should emit proper observability events."""
        events: list[Any] = []

        def on_event(event: Any) -> None:
            events.append(event)

        async def stream_with_chunked_tool():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": "buf_call",
                    "name": "buffered_tool",
                    "arguments": '{"part1":',
                },
            )
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "arguments": ' "value"}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_chunked_tool,
            buffer_tool_calls=True,
            on_event=on_event,
        )

        async for _ in result:
            pass

        # Check that TOOL_REQUESTED, TOOL_START, and TOOL_COMPLETED were emitted
        tool_requested = [
            e for e in events if e.type == ObservabilityEventType.TOOL_REQUESTED
        ]
        tool_start = [e for e in events if e.type == ObservabilityEventType.TOOL_START]
        tool_completed = [
            e for e in events if e.type == ObservabilityEventType.TOOL_COMPLETED
        ]

        assert len(tool_requested) == 1
        assert len(tool_start) == 1
        assert len(tool_completed) == 1

        # Verify the buffered arguments are complete (spec requires dict, not string)
        assert tool_requested[0].meta["arguments"] == {"part1": "value"}


class TestToolCallsWithOtherFeatures:
    """Tests for tool calls working with other L0 features."""

    @pytest.mark.asyncio
    async def test_tool_calls_with_guardrails(self) -> None:
        """Tool calls should work alongside guardrails."""
        on_tool_call = MagicMock()

        async def stream_with_tool():
            yield Event(type=EventType.TOKEN, text="Hello ")
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "guard_call",
                    "name": "safe_tool",
                    "arguments": '{"safe": true}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_tool,
            guardrails=[],  # Empty guardrails list
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert on_tool_call.call_count == 1
        assert result.state.completed

    @pytest.mark.asyncio
    async def test_tool_calls_with_retry_config(self) -> None:
        """Tool calls should work with retry configuration."""
        from l0.types import Retry

        on_tool_call = MagicMock()

        async def stream_with_tool():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "retry_call",
                    "name": "reliable_tool",
                    "arguments": "{}",
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_tool,
            retry=Retry(attempts=2),
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert on_tool_call.call_count == 1
        assert result.state.completed

    @pytest.mark.asyncio
    async def test_tool_calls_with_timeout(self) -> None:
        """Tool calls should work with timeout configuration."""
        from l0.types import Timeout

        on_tool_call = MagicMock()

        async def stream_with_tool():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "timeout_call",
                    "name": "fast_tool",
                    "arguments": "{}",
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_tool,
            timeout=Timeout(initial_token=5000, inter_token=5000),
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert on_tool_call.call_count == 1
        assert result.state.completed


class TestConcurrentToolCalls:
    """Tests for handling multiple concurrent tool calls."""

    @pytest.mark.asyncio
    async def test_interleaved_tool_call_chunks(self) -> None:
        """Interleaved tool call chunks should be correctly accumulated."""
        on_tool_call = MagicMock()

        async def stream_with_interleaved_tools():
            # Tool call 0 - first chunk
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": "concurrent_1",
                    "name": "tool_a",
                    "arguments": '{"a":',
                },
            )
            # Tool call 1 - first chunk (interleaved)
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 1,
                    "id": "concurrent_2",
                    "name": "tool_b",
                    "arguments": '{"b":',
                },
            )
            # Tool call 0 - continuation
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "arguments": " 1}",
                },
            )
            # Tool call 1 - continuation
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 1,
                    "arguments": " 2}",
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_interleaved_tools,
            buffer_tool_calls=True,
            on_tool_call=on_tool_call,
        )

        tool_calls = []
        async for event in result:
            if event.is_tool_call:
                tool_calls.append(event.data)

        assert len(tool_calls) == 2
        assert on_tool_call.call_count == 2

        # Verify both tool calls have complete arguments
        assert tool_calls[0]["arguments"] == '{"a": 1}'
        assert tool_calls[1]["arguments"] == '{"b": 2}'
