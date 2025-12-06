"""Tests for l0.runtime module."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from l0 import Retry, Timeout, TimeoutError
from l0.adapters import AdaptedEvent, Adapters
from l0.guardrails import GuardrailRule, GuardrailViolation
from l0.runtime import _internal_run
from l0.types import Event, EventType, State


class PassthroughAdapter:
    """Test adapter that passes through Event objects directly."""

    name = "passthrough"

    def detect(self, stream: Any) -> bool:
        """Detect async generators (our test streams)."""
        return hasattr(stream, "__anext__")

    async def wrap(
        self, stream: Any, options: Any = None
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        """Pass through events wrapped in AdaptedEvent."""
        async for event in stream:
            yield AdaptedEvent(event=event, raw_chunk=None)


@pytest.fixture(autouse=True)
def register_passthrough_adapter():
    """Register and cleanup the passthrough adapter for tests."""
    Adapters.register(PassthroughAdapter())
    yield
    Adapters.reset()


class TestLazyWrap:
    """Test that l0.wrap() returns immediately (no await needed)."""

    @pytest.mark.asyncio
    async def test_wrap_returns_immediately(self):
        """Test that wrap() is sync and returns LazyStream."""
        import l0 as l0

        async def my_stream():
            yield Event(type=EventType.TOKEN, text="hello")
            yield Event(type=EventType.COMPLETE)

        # No await needed!
        result = l0.wrap(my_stream())
        assert isinstance(result, l0.LazyStream)

    @pytest.mark.asyncio
    async def test_wrap_read_works(self):
        """Test that await result.read() works."""
        import l0 as l0

        async def my_stream():
            yield Event(type=EventType.TOKEN, text="hello")
            yield Event(type=EventType.COMPLETE)

        result = l0.wrap(my_stream())
        text = await result.read()
        assert text == "hello"

    @pytest.mark.asyncio
    async def test_wrap_iteration_works(self):
        """Test that async for works directly."""
        import l0 as l0

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
        import l0 as l0

        async def my_stream():
            yield Event(type=EventType.TOKEN, text="test")
            yield Event(type=EventType.COMPLETE)

        # No double await!
        tokens = []
        async with l0.wrap(my_stream()) as result:
            async for event in result:
                if event.is_token:
                    tokens.append(event.text)

        assert tokens == ["test"]


class TestCompletionGuardrails:
    """Test that completion-only guardrails are executed after stream completes."""

    @pytest.mark.asyncio
    async def test_completion_only_guardrail_runs_after_complete(self):
        """Test that guardrails with streaming=False run after completion."""
        check_calls = []

        def completion_check(state: State) -> list[GuardrailViolation]:
            check_calls.append({"completed": state.completed, "content": state.content})
            if state.completed and len(state.content) < 10:
                return [
                    GuardrailViolation(
                        rule="min_length",
                        message="Output too short",
                        severity="error",
                        recoverable=False,
                    )
                ]
            return []

        completion_rule = GuardrailRule(
            name="min_length",
            check=completion_check,
            streaming=False,
            description="Check minimum length on completion",
        )

        async def short_stream():
            yield Event(type=EventType.TOKEN, text="Hi")
            yield Event(type=EventType.COMPLETE)

        from l0.errors import Error, ErrorCode

        result = await _internal_run(
            stream=short_stream,
            guardrails=[completion_rule],
        )

        # Fatal guardrail violation (recoverable=False) should raise error
        with pytest.raises(Error) as exc_info:
            async for _ in result:
                pass

        assert exc_info.value.code == ErrorCode.FATAL_GUARDRAIL_VIOLATION

        # Should have been called with completed=True at least once
        completed_calls = [c for c in check_calls if c["completed"]]
        assert len(completed_calls) > 0, "Guardrail should run after completion"

    @pytest.mark.asyncio
    async def test_zero_output_rule_detects_empty(self):
        """Test that zero_output_rule works on completion."""
        from l0.errors import Error, ErrorCode
        from l0.guardrails import zero_output_rule

        async def empty_stream():
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=empty_stream,
            guardrails=[zero_output_rule()],
        )

        # Recoverable guardrail violation should raise error (triggers retry attempt)
        with pytest.raises(Error) as exc_info:
            async for _ in result:
                pass

        # Zero output is a recoverable error that triggers retry
        assert exc_info.value.code == ErrorCode.GUARDRAIL_VIOLATION
        assert "Empty or whitespace-only output" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_strict_json_rule_validates_on_completion(self):
        """Test that strict_json_rule validates complete JSON."""
        from l0.errors import Error, ErrorCode
        from l0.guardrails import strict_json_rule

        async def invalid_json_stream():
            yield Event(
                type=EventType.TOKEN, text='{"key": "value"'
            )  # Missing closing brace
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=invalid_json_stream,
            guardrails=[strict_json_rule()],
        )

        # Recoverable guardrail violation should raise error (triggers retry attempt)
        with pytest.raises(Error) as exc_info:
            async for _ in result:
                pass

        # Invalid JSON is a recoverable error that triggers retry
        assert exc_info.value.code == ErrorCode.GUARDRAIL_VIOLATION
        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_valid_json_passes_strict_rule(self):
        """Test that valid JSON passes strict_json_rule."""
        from l0.guardrails import strict_json_rule

        async def valid_json_stream():
            yield Event(type=EventType.TOKEN, text='{"key": "value"}')
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=valid_json_stream,
            guardrails=[strict_json_rule()],
        )

        async for _ in result:
            pass

        # Should have no violations
        json_violations = [
            v for v in result.state.violations if v.rule == "strict_json"
        ]
        assert len(json_violations) == 0, "Valid JSON should pass"


class TestFallback:
    @pytest.mark.asyncio
    async def test_fallback_end_emitted_on_success(self):
        """Test that FALLBACK_END is emitted when fallback succeeds."""
        events_received = []

        def on_event(event: Any) -> None:
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
                timeout=Timeout(initial_token=100, inter_token=1000),
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
                timeout=Timeout(initial_token=1000, inter_token=100),
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
            timeout=Timeout(initial_token=1000, inter_token=1000),
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

    @pytest.mark.asyncio
    async def test_inter_token_timeout_with_continuation(self):
        """Test that inter_token timeout triggers retry with continuation."""
        call_count = 0

        async def stalling_then_succeeding_stream():
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: yield some tokens then stall
                yield Event(type=EventType.TOKEN, text="Hello ")
                yield Event(type=EventType.TOKEN, text="world")
                # Stall - will trigger inter_token timeout
                await asyncio.sleep(0.5)
                yield Event(type=EventType.TOKEN, text=" never")
                yield Event(type=EventType.COMPLETE)
            else:
                # Second call (after retry): complete successfully
                # With continuation, we resume from checkpoint
                yield Event(type=EventType.TOKEN, text="Hello ")
                yield Event(type=EventType.TOKEN, text="world")
                yield Event(type=EventType.TOKEN, text="!")
                yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stalling_then_succeeding_stream,
            timeout=Timeout(initial_token=1000, inter_token=100),
            retry=Retry(attempts=3, max_retries=3),
            continue_from_last_good_token=True,
        )

        tokens = []
        async for event in result:
            if event.is_token:
                tokens.append(event.text)

        # Should have retried and succeeded
        assert call_count == 2
        # Content should be from second successful call
        # (with deduplication removing overlap if any)
        assert result.state.completed is True
        assert result.state.resumed is True


class TestToolCallBuffering:
    """Test tool call buffering feature."""

    @pytest.mark.asyncio
    async def test_buffer_tool_calls_accumulates_arguments(self):
        """Test that buffer_tool_calls=True accumulates chunked arguments."""

        async def stream_with_chunked_tool_call():
            # Simulates OpenAI streaming pattern:
            # First chunk: index, id, name, partial arguments
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"loc',
                },
            )
            # Subsequent chunks: just more arguments
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": None,
                    "name": None,
                    "arguments": 'ation": "',
                },
            )
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": None,
                    "name": None,
                    "arguments": 'NYC"}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_chunked_tool_call,
            buffer_tool_calls=True,
        )

        tool_calls = []
        async for event in result:
            if event.is_tool_call:
                tool_calls.append(event.data)

        # Should get one complete tool call
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["arguments"] == '{"location": "NYC"}'

    @pytest.mark.asyncio
    async def test_buffer_tool_calls_multiple_tools(self):
        """Test buffering multiple tool calls."""

        async def stream_with_multiple_tool_calls():
            # Tool call 0 - first chunk
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":',
                },
            )
            # Tool call 1 - first chunk (interleaved)
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 1,
                    "id": "call_2",
                    "name": "get_time",
                    "arguments": '{"tz":',
                },
            )
            # Tool call 0 - continuation
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": None,
                    "name": None,
                    "arguments": ' "LA"}',
                },
            )
            # Tool call 1 - continuation
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 1,
                    "id": None,
                    "name": None,
                    "arguments": ' "UTC"}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_multiple_tool_calls,
            buffer_tool_calls=True,
        )

        tool_calls = []
        async for event in result:
            if event.is_tool_call:
                tool_calls.append(event.data)

        # Should get two complete tool calls in order
        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] == "call_1"
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["arguments"] == '{"city": "LA"}'
        assert tool_calls[1]["id"] == "call_2"
        assert tool_calls[1]["name"] == "get_time"
        assert tool_calls[1]["arguments"] == '{"tz": "UTC"}'

    @pytest.mark.asyncio
    async def test_no_buffering_by_default(self):
        """Test that tool calls are emitted immediately without buffering."""

        async def stream_with_chunked_tool_call():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"loc',
                },
            )
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": None,
                    "name": None,
                    "arguments": 'ation": "NYC"}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_chunked_tool_call,
            buffer_tool_calls=False,  # Default
        )

        tool_calls = []
        async for event in result:
            if event.is_tool_call:
                tool_calls.append(event.data)

        # Should get two separate (partial) tool call events
        assert len(tool_calls) == 2
        assert tool_calls[0]["arguments"] == '{"loc'
        assert tool_calls[1]["arguments"] == 'ation": "NYC"}'

    @pytest.mark.asyncio
    async def test_buffer_tool_calls_with_wrap(self):
        """Test that buffer_tool_calls works with l0.wrap()."""
        import l0 as l0

        async def stream_with_chunked_tool_call():
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": "call_abc",
                    "name": "search",
                    "arguments": '{"q',
                },
            )
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "index": 0,
                    "id": None,
                    "name": None,
                    "arguments": 'uery": "test"}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        tool_calls = []
        async for event in l0.wrap(
            stream_with_chunked_tool_call(), buffer_tool_calls=True
        ):
            if event.is_tool_call:
                tool_calls.append(event.data)

        assert len(tool_calls) == 1
        assert tool_calls[0]["arguments"] == '{"query": "test"}'
