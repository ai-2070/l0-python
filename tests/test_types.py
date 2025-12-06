"""Tests for l0.types module."""

import pytest

from l0.types import (
    BackoffStrategy,
    ErrorCategory,
    Event,
    EventType,
    Retry,
    State,
    Timeout,
)


class TestEventType:
    def test_event_types_exist(self):
        assert EventType.TOKEN == "token"
        assert EventType.MESSAGE == "message"
        assert EventType.DATA == "data"
        assert EventType.PROGRESS == "progress"
        assert EventType.TOOL_CALL == "tool_call"
        assert EventType.ERROR == "error"
        assert EventType.COMPLETE == "complete"


class TestErrorCategory:
    def test_error_categories_exist(self):
        assert ErrorCategory.NETWORK == "network"
        assert ErrorCategory.TRANSIENT == "transient"
        assert ErrorCategory.MODEL == "model"
        assert ErrorCategory.CONTENT == "content"
        assert ErrorCategory.PROVIDER == "provider"
        assert ErrorCategory.FATAL == "fatal"
        assert ErrorCategory.INTERNAL == "internal"


class TestBackoffStrategy:
    def test_backoff_strategies_exist(self):
        assert BackoffStrategy.EXPONENTIAL == "exponential"
        assert BackoffStrategy.LINEAR == "linear"
        assert BackoffStrategy.FIXED == "fixed"
        assert BackoffStrategy.FULL_JITTER == "full-jitter"
        assert BackoffStrategy.FIXED_JITTER == "fixed-jitter"


class TestEvent:
    def test_create_token_event(self):
        event = Event(type=EventType.TOKEN, text="hello")
        assert event.type == EventType.TOKEN
        assert event.text == "hello"
        assert event.data is None
        assert event.error is None

    def test_create_tool_call_event(self):
        event = Event(
            type=EventType.TOOL_CALL,
            data={"id": "call_123", "name": "get_weather"},
        )
        assert event.type == EventType.TOOL_CALL
        assert event.data is not None
        assert event.data["id"] == "call_123"

    def test_create_complete_event_with_usage(self):
        event = Event(
            type=EventType.COMPLETE,
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert event.type == EventType.COMPLETE
        assert event.usage is not None
        assert event.usage["input_tokens"] == 100

    def test_is_token_property(self):
        event = Event(type=EventType.TOKEN, text="hello")
        assert event.is_token is True
        assert event.is_error is False
        assert event.is_complete is False

    def test_is_error_property(self):
        event = Event(type=EventType.ERROR, error=Exception("test"))
        assert event.is_error is True
        assert event.is_token is False

    def test_is_complete_property(self):
        event = Event(type=EventType.COMPLETE)
        assert event.is_complete is True
        assert event.is_token is False

    def test_is_tool_call_property(self):
        event = Event(type=EventType.TOOL_CALL, data={"name": "test"})
        assert event.is_tool_call is True


class TestState:
    def test_default_state(self):
        state = State()
        assert state.content == ""
        assert state.checkpoint == ""
        assert state.token_count == 0
        assert state.model_retry_count == 0
        assert state.network_retry_count == 0
        assert state.fallback_index == 0
        assert state.violations == []
        assert state.drift_detected is False
        assert state.completed is False
        assert state.aborted is False


class TestRetry:
    def test_default_values(self):
        config = Retry()
        assert config.attempts == 3
        assert config.max_retries == 6
        assert config.base_delay == 1.0  # seconds
        assert config.max_delay == 10.0  # seconds
        assert config.strategy == BackoffStrategy.FIXED_JITTER

    def test_custom_values(self):
        config = Retry(
            attempts=5,
            base_delay=0.5,
            strategy=BackoffStrategy.EXPONENTIAL,
        )
        assert config.attempts == 5
        assert config.base_delay == 0.5
        assert config.strategy == BackoffStrategy.EXPONENTIAL


class TestTimeout:
    def test_default_values(self):
        config = Timeout()
        assert config.initial_token == 5000  # milliseconds
        assert config.inter_token == 10000  # milliseconds


class TestStream:
    @pytest.mark.asyncio
    async def test_read_returns_content(self):
        """Test that read() returns accumulated content."""
        from l0.types import Stream

        # Create a simple async iterator
        async def token_iterator():
            for token in ["Hello", " ", "world", "!"]:
                yield Event(type=EventType.TOKEN, text=token)
            yield Event(type=EventType.COMPLETE)

        state = State()
        stream = Stream(
            iterator=token_iterator(),
            state=state,
            abort=lambda: None,
        )

        # Manually consume to populate state
        async for event in stream:
            if event.is_token and event.text:
                state.content += event.text

        result = await stream.read()
        assert result == "Hello world!"

    @pytest.mark.asyncio
    async def test_read_consumes_stream(self):
        """Test that read() consumes the stream and returns text."""
        from l0.types import Stream

        async def token_iterator():
            for token in ["Test", " ", "content"]:
                yield Event(type=EventType.TOKEN, text=token)
            yield Event(type=EventType.COMPLETE)

        state = State()

        # We need to manually append tokens since we're testing the Stream directly
        original_iterator = token_iterator()

        async def tracking_iterator():
            async for event in original_iterator:
                if event.is_token and event.text:
                    state.content += event.text
                yield event

        stream = Stream(
            iterator=tracking_iterator(),
            state=state,
            abort=lambda: None,
        )

        result = await stream.read()
        assert result == "Test content"
