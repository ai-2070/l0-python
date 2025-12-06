"""Tests for l0.stream module."""

import pytest

from l0.stream import consume_stream, get_text
from l0.types import Event, EventType, State, Stream


class TestConsumeStream:
    @pytest.mark.asyncio
    async def test_consume_stream_collects_tokens(self):
        """Test that consume_stream collects all token text."""

        async def token_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.TOKEN, text=" ")
            yield Event(type=EventType.TOKEN, text="world")
            yield Event(type=EventType.COMPLETE)

        result = await consume_stream(token_stream())
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_consume_stream_ignores_non_tokens(self):
        """Test that consume_stream ignores non-token events."""

        async def mixed_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.DATA, data={"key": "value"})
            yield Event(type=EventType.TOKEN, text=" world")
            yield Event(type=EventType.TOOL_CALL, data={"name": "test"})
            yield Event(type=EventType.COMPLETE)

        result = await consume_stream(mixed_stream())
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_consume_stream_handles_empty(self):
        """Test that consume_stream handles empty stream."""

        async def empty_stream():
            yield Event(type=EventType.COMPLETE)

        result = await consume_stream(empty_stream())
        assert result == ""

    @pytest.mark.asyncio
    async def test_consume_stream_handles_none_text(self):
        """Test that consume_stream handles None text in tokens."""

        async def stream_with_none():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.TOKEN, text=None)
            yield Event(type=EventType.TOKEN, text=" world")
            yield Event(type=EventType.COMPLETE)

        result = await consume_stream(stream_with_none())
        assert result == "Hello world"


class TestGetText:
    @pytest.mark.asyncio
    async def test_get_text_returns_content(self):
        """Test that get_text returns stream content."""

        async def token_stream():
            yield Event(type=EventType.TOKEN, text="Test content")
            yield Event(type=EventType.COMPLETE)

        state = State()

        async def tracking_stream():
            async for event in token_stream():
                if event.is_token and event.text:
                    state.content += event.text
                yield event

        stream = Stream(
            iterator=tracking_stream(),
            state=state,
            abort=lambda: None,
        )

        result = await get_text(stream)
        assert result == "Test content"
