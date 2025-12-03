"""Tests for l0.runtime module."""

import asyncio
from typing import Any, AsyncIterator

import pytest

from l0 import Timeout, TimeoutError
from l0.adapters import register_adapter
from l0.runtime import _internal_run
from l0.types import Event, EventType


class PassthroughAdapter:
    """Test adapter that passes through Event objects directly."""

    name = "passthrough"

    def detect(self, stream: Any) -> bool:
        """Detect async generators (our test streams)."""
        return hasattr(stream, "__anext__")

    async def wrap(self, stream: Any) -> AsyncIterator[Event]:
        """Pass through events directly."""
        async for event in stream:
            yield event


# Register the test adapter
register_adapter(PassthroughAdapter())


class TestTimeout:
    @pytest.mark.asyncio
    async def test_initial_token_timeout(self):
        """Test that initial_token timeout is enforced."""

        async def slow_start_stream():
            # Wait longer than the timeout before yielding first token
            await asyncio.sleep(0.5)
            yield Event(type=EventType.TOKEN, text="hello")
            yield Event(type=EventType.COMPLETE)

        with pytest.raises(TimeoutError) as exc_info:
            result = await _internal_run(
                stream=slow_start_stream,
                timeout=Timeout(initial_token=0.1, inter_token=1.0),
            )
            async for _ in result:
                pass

        assert exc_info.value.timeout_type == "initial_token"
        assert exc_info.value.timeout_seconds == 0.1

    @pytest.mark.asyncio
    async def test_inter_token_timeout(self):
        """Test that inter_token timeout is enforced."""

        async def stalling_stream():
            yield Event(type=EventType.TOKEN, text="first")
            # Wait longer than inter_token timeout
            await asyncio.sleep(0.5)
            yield Event(type=EventType.TOKEN, text="second")
            yield Event(type=EventType.COMPLETE)

        with pytest.raises(TimeoutError) as exc_info:
            result = await _internal_run(
                stream=stalling_stream,
                timeout=Timeout(initial_token=1.0, inter_token=0.1),
            )
            async for _ in result:
                pass

        assert exc_info.value.timeout_type == "inter_token"
        assert exc_info.value.timeout_seconds == 0.1

    @pytest.mark.asyncio
    async def test_no_timeout_when_fast(self):
        """Test that fast streams don't timeout."""

        async def fast_stream():
            yield Event(type=EventType.TOKEN, text="hello")
            yield Event(type=EventType.TOKEN, text=" world")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=fast_stream,
            timeout=Timeout(initial_token=1.0, inter_token=1.0),
        )

        tokens = []
        async for event in result:
            if event.is_token:
                tokens.append(event.text)

        assert tokens == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_no_timeout_config(self):
        """Test that no timeout config means no timeout enforcement."""

        async def slow_stream():
            await asyncio.sleep(0.1)
            yield Event(type=EventType.TOKEN, text="hello")
            await asyncio.sleep(0.1)
            yield Event(type=EventType.TOKEN, text=" world")
            yield Event(type=EventType.COMPLETE)

        # No timeout config - should not raise
        result = await _internal_run(
            stream=slow_stream,
            timeout=None,
        )

        tokens = []
        async for event in result:
            if event.is_token:
                tokens.append(event.text)

        assert tokens == ["hello", " world"]
