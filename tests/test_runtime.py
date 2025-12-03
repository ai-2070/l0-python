"""Tests for l0.runtime module."""

import asyncio
from typing import Any, AsyncIterator

import pytest

from l0 import Retry, Timeout, TimeoutError
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


class TestLazyWrap:
    """Test that l0.wrap() returns immediately (no await needed)."""

    @pytest.mark.asyncio
    async def test_wrap_returns_immediately(self):
        """Test that wrap() is sync and returns LazyStream."""
        import l0

        async def my_stream():
            yield Event(type=EventType.TOKEN, text="hello")
            yield Event(type=EventType.COMPLETE)

        # No await needed!
        result = l0.wrap(my_stream())
        assert isinstance(result, l0.LazyStream)

    @pytest.mark.asyncio
    async def test_wrap_read_works(self):
        """Test that await result.read() works."""
        import l0

        async def my_stream():
            yield Event(type=EventType.TOKEN, text="hello")
            yield Event(type=EventType.COMPLETE)

        result = l0.wrap(my_stream())
        text = await result.read()
        assert text == "hello"

    @pytest.mark.asyncio
    async def test_wrap_iteration_works(self):
        """Test that async for works directly."""
        import l0

        async def my_stream():
            yield Event(type=EventType.TOKEN, text="hello")
            yield Event(type=EventType.TOKEN, text=" world")
            yield Event(type=EventType.COMPLETE)

        tokens = []
        async for event in l0.wrap(my_stream()):
            if event.is_token:
                tokens.append(event.text)

        assert tokens == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_wrap_context_manager_works(self):
        """Test that async with works without double await."""
        import l0

        async def my_stream():
            yield Event(type=EventType.TOKEN, text="test")
            yield Event(type=EventType.COMPLETE)

        # No double await!
        async with l0.wrap(my_stream()) as result:
            tokens = []
            async for event in result:
                if event.is_token:
                    tokens.append(event.text)

        assert tokens == ["test"]


class TestFallback:
    @pytest.mark.asyncio
    async def test_fallback_end_emitted_on_success(self):
        """Test that FALLBACK_END is emitted when fallback succeeds."""
        events_received = []

        def on_event(event):
            events_received.append(event.type.value)

        async def failing_stream():
            raise ValueError("Primary failed")
            yield  # Make it a generator

        async def working_stream():
            yield Event(type=EventType.TOKEN, text="fallback")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=failing_stream,
            fallbacks=[working_stream],
            on_event=on_event,
            retry=Retry(attempts=1, max_retries=1),
        )

        async for _ in result:
            pass

        # Should have both FALLBACK_START and FALLBACK_END
        assert "FALLBACK_START" in events_received
        assert "FALLBACK_END" in events_received


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
                retry=Retry(attempts=1, max_retries=1),  # No retries
            )
            async for _ in result:
                pass

        assert isinstance(exc_info.value, TimeoutError)
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
                retry=Retry(attempts=1, max_retries=1),  # No retries
            )
            async for _ in result:
                pass

        assert isinstance(exc_info.value, TimeoutError)
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
