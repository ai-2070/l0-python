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
        event = Event(type=EventType.TOKEN, value="hello")
        assert event.type == EventType.TOKEN
        assert event.value == "hello"
        assert event.data is None
        assert event.error is None

    def test_create_tool_call_event(self):
        event = Event(
            type=EventType.TOOL_CALL,
            data={"id": "call_123", "name": "get_weather"},
        )
        assert event.type == EventType.TOOL_CALL
        assert event.data["id"] == "call_123"

    def test_create_complete_event_with_usage(self):
        event = Event(
            type=EventType.COMPLETE,
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert event.type == EventType.COMPLETE
        assert event.usage["input_tokens"] == 100


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
        assert config.base_delay_ms == 1000
        assert config.max_delay_ms == 10000
        assert config.strategy == BackoffStrategy.FIXED_JITTER

    def test_custom_values(self):
        config = Retry(
            attempts=5,
            base_delay_ms=500,
            strategy=BackoffStrategy.EXPONENTIAL,
        )
        assert config.attempts == 5
        assert config.base_delay_ms == 500
        assert config.strategy == BackoffStrategy.EXPONENTIAL


class TestTimeout:
    def test_default_values(self):
        config = Timeout()
        assert config.initial_token_ms == 5000
        assert config.inter_token_ms == 10000
