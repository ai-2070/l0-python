"""Tests for l0.structured module."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from l0._structured import (
    AutoCorrectInfo,
    StructuredResult,
    StructuredState,
    StructuredTelemetry,
    structured,
    structured_array,
    structured_object,
    structured_stream,
)
from l0.adapters import AdaptedEvent, Adapters
from l0.events import ObservabilityEventType
from l0.types import Event, EventType, Retry, Timeout


# Test adapter that passes through Event objects
class PassthroughAdapter:
    """Test adapter that passes through Event objects directly."""

    name = "passthrough_structured"

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


class UserProfile(BaseModel):
    name: str
    age: int
    email: str


class SimpleModel(BaseModel):
    value: str


class TestStructured:
    @pytest.mark.asyncio
    async def test_structured_parses_json(self):
        """Test that structured parses valid JSON."""

        async def json_stream():
            yield Event(
                type=EventType.TOKEN,
                text='{"name": "John", "age": 30, "email": "john@example.com"}',
            )
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=UserProfile,
            stream=json_stream,
        )

        assert isinstance(result, StructuredResult)
        assert result.data.name == "John"
        assert result.data.age == 30
        assert result.data.email == "john@example.com"
        assert result.raw is not None

    @pytest.mark.asyncio
    async def test_structured_extracts_from_markdown(self):
        """Test that structured extracts JSON from markdown."""

        async def markdown_stream():
            yield Event(
                type=EventType.TOKEN,
                text='Here is the data:\n```json\n{"value": "test"}\n```',
            )
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=markdown_stream,
        )

        assert result.data.value == "test"

    @pytest.mark.asyncio
    async def test_structured_auto_corrects_json(self):
        """Test that structured auto-corrects malformed JSON."""

        async def malformed_stream():
            # Missing closing brace
            yield Event(type=EventType.TOKEN, text='{"value": "test"')
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=malformed_stream,
            auto_correct=True,
        )

        assert result.data.value == "test"
        assert result.corrected is True
        assert len(result.corrections) > 0

    @pytest.mark.asyncio
    async def test_structured_raises_on_invalid(self):
        """Test that structured raises on invalid JSON."""

        async def invalid_stream():
            yield Event(type=EventType.TOKEN, text='{"wrong_field": "value"}')
            yield Event(type=EventType.COMPLETE)

        with pytest.raises(ValueError, match="Schema validation failed"):
            await structured(
                schema=UserProfile,
                stream=invalid_stream,
            )

    @pytest.mark.asyncio
    async def test_structured_reraises_non_validation_errors(self):
        """Test that non-validation errors are re-raised as-is."""

        class CustomError(Exception):
            pass

        async def failing_stream():
            raise CustomError("Something went wrong")
            yield  # Make it a generator  # noqa: B030

        with pytest.raises(CustomError, match="Something went wrong"):
            await structured(
                schema=UserProfile,
                stream=failing_stream,
            )

    @pytest.mark.asyncio
    async def test_structured_without_auto_correct(self):
        """Test structured with auto_correct disabled."""

        async def valid_stream():
            yield Event(type=EventType.TOKEN, text='{"value": "test"}')
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=valid_stream,
            auto_correct=False,
        )

        assert result.data.value == "test"
        assert result.corrected is False

    @pytest.mark.asyncio
    async def test_structured_accumulates_tokens(self):
        """Test that structured accumulates multiple tokens."""

        async def multi_token_stream():
            yield Event(type=EventType.TOKEN, text='{"value":')
            yield Event(type=EventType.TOKEN, text=' "hello')
            yield Event(type=EventType.TOKEN, text=' world"}')
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=multi_token_stream,
        )

        assert result.data.value == "hello world"

    @pytest.mark.asyncio
    async def test_structured_on_auto_correct_callback(self):
        """Test on_auto_correct callback is called."""
        callback_called = False
        callback_info = None

        def on_auto_correct(info: AutoCorrectInfo):
            nonlocal callback_called, callback_info
            callback_called = True
            callback_info = info

        async def malformed_stream():
            yield Event(type=EventType.TOKEN, text='{"value": "test",}')
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=malformed_stream,
            auto_correct=True,
            on_auto_correct=on_auto_correct,
        )

        assert result.data.value == "test"
        assert callback_called
        assert callback_info is not None
        assert len(callback_info.corrections) > 0

    @pytest.mark.asyncio
    async def test_structured_on_validation_error_callback(self):
        """Test on_validation_error callback is called on failures."""
        errors_received = []

        def on_validation_error(error: ValidationError, attempt: int):
            errors_received.append((error, attempt))

        async def invalid_stream():
            yield Event(type=EventType.TOKEN, text='{"wrong": "field"}')
            yield Event(type=EventType.COMPLETE)

        with pytest.raises(ValueError):
            await structured(
                schema=UserProfile,
                stream=invalid_stream,
                retry=Retry(attempts=2),
                on_validation_error=on_validation_error,
            )

        # Should have been called for each attempt
        assert len(errors_received) == 2
        assert errors_received[0][1] == 1
        assert errors_received[1][1] == 2

    @pytest.mark.asyncio
    async def test_structured_emits_parse_end_on_validation_failure(self):
        """Test that PARSE_END is emitted even when validation fails."""
        events_emitted = []

        def on_event(event: Any) -> None:
            events_emitted.append(event.type)

        async def invalid_stream():
            yield Event(type=EventType.TOKEN, text='{"wrong": "field"}')
            yield Event(type=EventType.COMPLETE)

        with pytest.raises(ValueError):
            await structured(
                schema=UserProfile,
                stream=invalid_stream,
                on_event=on_event,
            )

        # Verify PARSE_START and PARSE_END are both emitted
        assert ObservabilityEventType.PARSE_START in events_emitted
        assert ObservabilityEventType.PARSE_END in events_emitted
        # Verify exactly one SCHEMA_VALIDATION_END (no duplicates)
        schema_validation_end_count = events_emitted.count(
            ObservabilityEventType.SCHEMA_VALIDATION_END
        )
        assert schema_validation_end_count == 1


class TestStructuredStream:
    """Test structured_stream() function."""

    @pytest.mark.asyncio
    async def test_structured_stream_basic(self):
        """Test basic structured streaming."""

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='{"value":')
            yield Event(type=EventType.TOKEN, text=' "streamed"}')
            yield Event(type=EventType.COMPLETE)

        stream, result_holder = await structured_stream(
            schema=SimpleModel,
            stream=json_stream,
        )

        tokens = []
        async for event in stream:
            if event.is_token and event.text:
                tokens.append(event.text)

        # Validate after consuming stream
        result = await result_holder.validate()
        assert result.data.value == "streamed"
        assert len(tokens) == 2

    @pytest.mark.asyncio
    async def test_structured_stream_with_auto_correct(self):
        """Test structured streaming with auto-correction."""

        async def malformed_stream():
            yield Event(type=EventType.TOKEN, text='{"value": "test"')
            yield Event(type=EventType.COMPLETE)

        stream, result_holder = await structured_stream(
            schema=SimpleModel,
            stream=malformed_stream,
            auto_correct=True,
        )

        async for _ in stream:
            pass

        result = await result_holder.validate()
        assert result.data.value == "test"
        assert result.corrected is True

    @pytest.mark.asyncio
    async def test_structured_stream_validate_emits_events(self):
        """Test that validate() emits observability events via on_event callback."""
        from l0.events import ObservabilityEventType

        events_received = []

        def on_event(event: Any) -> None:
            events_received.append(event.type)

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='{"value": "test"}')
            yield Event(type=EventType.COMPLETE)

        stream, result_holder = await structured_stream(
            schema=SimpleModel,
            stream=json_stream,
            on_event=on_event,
        )

        async for _ in stream:
            pass

        # Validate should emit parse/validation events
        result = await result_holder.validate()
        assert result.data.value == "test"

        # Check that validation events were emitted
        assert ObservabilityEventType.PARSE_START in events_received
        assert ObservabilityEventType.PARSE_END in events_received

    @pytest.mark.asyncio
    async def test_structured_stream_validate_raises_valueerror(self):
        """Test that validate() raises ValueError, not ValidationError."""

        async def invalid_stream():
            yield Event(type=EventType.TOKEN, text='{"wrong": "field"}')
            yield Event(type=EventType.COMPLETE)

        stream, result_holder = await structured_stream(
            schema=UserProfile,
            stream=invalid_stream,
        )

        async for _ in stream:
            pass

        # Should raise ValueError, not pydantic.ValidationError
        with pytest.raises(ValueError, match="Schema validation failed"):
            await result_holder.validate()


class TestStructuredIteratorValidation:
    """Test validation of async iterator vs factory usage."""

    @pytest.mark.asyncio
    async def test_direct_iterator_with_retry_auto_buffers(self):
        """Test that direct async iterator is auto-buffered for retries."""
        from l0 import Retry

        attempt_count = 0

        async def json_gen():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                # First attempt returns invalid JSON
                yield Event(type=EventType.TOKEN, text='{"wrong": "field"}')
            else:
                # This won't be reached since we buffer - but validation will retry
                yield Event(type=EventType.TOKEN, text='{"value": "test"}')
            yield Event(type=EventType.COMPLETE)

        # Create a direct async iterator (not a factory)
        direct_iterator = json_gen()

        # This should work - iterator is auto-buffered
        # Note: retries will replay the same buffered content, so validation
        # will fail on all attempts with the same invalid JSON
        with pytest.raises(ValueError):
            await structured(
                schema=SimpleModel,
                stream=direct_iterator,
                retry=Retry(attempts=2),
            )

        # Iterator was consumed once (buffered), then replayed
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_direct_iterator_works_without_retry(self):
        """Test that direct async iterator works fine without retry."""

        async def json_gen():
            yield Event(type=EventType.TOKEN, text='{"value": "test"}')
            yield Event(type=EventType.COMPLETE)

        direct_iterator = json_gen()

        result = await structured(
            schema=SimpleModel,
            stream=direct_iterator,
        )

        assert result.data.value == "test"

    @pytest.mark.asyncio
    async def test_factory_with_retry_works(self):
        """Test that using a factory function with retry works."""
        from l0 import Retry

        async def json_gen():
            yield Event(type=EventType.TOKEN, text='{"value": "test"}')
            yield Event(type=EventType.COMPLETE)

        # Pass factory function (not direct iterator)
        result = await structured(
            schema=SimpleModel,
            stream=json_gen,  # Factory, not json_gen()
            retry=Retry(attempts=2),
        )

        assert result.data.value == "test"

    @pytest.mark.asyncio
    async def test_direct_iterator_with_fallback_auto_buffers(self):
        """Test that direct async iterators in fallbacks are also buffered."""

        async def failing_gen():
            yield Event(type=EventType.TOKEN, text='{"wrong": "field"}')
            yield Event(type=EventType.COMPLETE)

        async def fallback_gen():
            yield Event(type=EventType.TOKEN, text='{"value": "from_fallback"}')
            yield Event(type=EventType.COMPLETE)

        # Both are direct iterators
        direct_main = failing_gen()
        direct_fallback = fallback_gen()

        result = await structured(
            schema=SimpleModel,
            stream=direct_main,
            fallbacks=[direct_fallback],
        )

        assert result.data.value == "from_fallback"


class TestStructuredState:
    """Test StructuredState tracking."""

    @pytest.mark.asyncio
    async def test_structured_state_on_success(self):
        """Test that structured_state is populated on success."""

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='{"value": "test"}')
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=json_stream,
        )

        assert result.structured_state is not None
        assert result.structured_state.validation_failures == 0
        assert result.structured_state.validation_time_ms is not None
        assert result.structured_state.validation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_structured_state_tracks_corrections(self):
        """Test that corrections are tracked in structured_state."""

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='{"value": "test",}')
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=json_stream,
            auto_correct=True,
        )

        assert result.structured_state is not None
        assert result.structured_state.auto_corrections >= 1
        assert len(result.structured_state.correction_types) > 0


class TestStructuredTelemetry:
    """Test StructuredTelemetry when monitoring is enabled."""

    @pytest.mark.asyncio
    async def test_telemetry_when_monitoring_enabled(self):
        """Test that telemetry is populated when monitoring is enabled."""

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='{"value": "test"}')
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=json_stream,
            monitoring=True,
        )

        assert result.telemetry is not None
        assert result.telemetry.schema_name == "SimpleModel"
        assert result.telemetry.validation_attempts >= 1
        assert result.telemetry.validation_success is True
        assert result.telemetry.validation_time_ms is not None

    @pytest.mark.asyncio
    async def test_telemetry_not_present_when_monitoring_disabled(self):
        """Test that telemetry is None when monitoring is disabled."""

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='{"value": "test"}')
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            schema=SimpleModel,
            stream=json_stream,
            monitoring=False,
        )

        assert result.telemetry is None


class TestStructuredOnRetryCallback:
    """Test on_retry callback."""

    @pytest.mark.asyncio
    async def test_on_retry_called_on_validation_failure(self):
        """Test that on_retry is called when validation fails."""
        retry_calls = []

        def on_retry(attempt: int, reason: str):
            retry_calls.append((attempt, reason))

        async def invalid_stream():
            yield Event(type=EventType.TOKEN, text='{"wrong": "field"}')
            yield Event(type=EventType.COMPLETE)

        with pytest.raises(ValueError):
            await structured(
                schema=SimpleModel,
                stream=invalid_stream,
                retry=Retry(attempts=2),
                on_retry=on_retry,
            )

        # Should have been called once (after first failure, before second attempt)
        assert len(retry_calls) >= 1
        assert retry_calls[0][0] == 1  # First retry attempt


class TestStructuredDetectZeroTokens:
    """Test detect_zero_tokens option."""

    @pytest.mark.asyncio
    async def test_detect_zero_tokens_raises_on_empty(self):
        """Test that zero-token detection raises error on empty output."""

        async def empty_stream():
            yield Event(type=EventType.TOKEN, text="")
            yield Event(type=EventType.COMPLETE)

        with pytest.raises(ValueError, match="Zero-token output"):
            await structured(
                schema=SimpleModel,
                stream=empty_stream,
                detect_zero_tokens=True,
            )

    @pytest.mark.asyncio
    async def test_detect_zero_tokens_disabled_by_default(self):
        """Test that zero-token detection is disabled by default."""

        async def whitespace_stream():
            yield Event(type=EventType.TOKEN, text="   ")
            yield Event(type=EventType.COMPLETE)

        # Should not raise for zero tokens, but will fail validation
        with pytest.raises(ValueError, match="Schema validation failed"):
            await structured(
                schema=SimpleModel,
                stream=whitespace_stream,
                detect_zero_tokens=False,
            )


class TestStructuredObject:
    """Test structured_object helper."""

    @pytest.mark.asyncio
    async def test_structured_object_basic(self):
        """Test basic structured_object usage."""

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='{"name": "Alice", "age": 30}')
            yield Event(type=EventType.COMPLETE)

        result = await structured_object(
            {"name": str, "age": int},
            stream=json_stream,
        )

        assert result.data.name == "Alice"
        assert result.data.age == 30

    @pytest.mark.asyncio
    async def test_structured_object_with_defaults(self):
        """Test structured_object with default values."""

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='{"name": "Bob"}')
            yield Event(type=EventType.COMPLETE)

        result = await structured_object(
            {"name": str, "active": (bool, True)},
            stream=json_stream,
        )

        assert result.data.name == "Bob"
        assert result.data.active is True


class TestStructuredArray:
    """Test structured_array helper."""

    @pytest.mark.asyncio
    async def test_structured_array_basic(self):
        """Test basic structured_array usage."""

        async def json_stream():
            yield Event(
                type=EventType.TOKEN,
                text='[{"value": "a"}, {"value": "b"}]',
            )
            yield Event(type=EventType.COMPLETE)

        result = await structured_array(
            SimpleModel,
            stream=json_stream,
        )

        assert len(result.data) == 2
        assert result.data[0].value == "a"
        assert result.data[1].value == "b"

    @pytest.mark.asyncio
    async def test_structured_array_with_telemetry(self):
        """Test structured_array with monitoring enabled."""

        async def json_stream():
            yield Event(type=EventType.TOKEN, text='[{"value": "test"}]')
            yield Event(type=EventType.COMPLETE)

        result = await structured_array(
            SimpleModel,
            stream=json_stream,
            monitoring=True,
        )

        assert result.telemetry is not None
        assert result.telemetry.schema_name is not None
        assert "list[SimpleModel]" in result.telemetry.schema_name


class TestStructuredStrictMode:
    """Test strict_mode parameter for rejecting extra fields."""

    @pytest.mark.asyncio
    async def test_strict_mode_rejects_extra_fields(self):
        """Test that strict_mode rejects extra fields not in schema."""

        async def json_stream():
            # JSON with extra field "extra_field" not in SimpleModel
            yield Event(
                type=EventType.TOKEN,
                text='{"value": "test", "extra_field": "should_fail"}',
            )
            yield Event(type=EventType.COMPLETE)

        with pytest.raises(ValueError, match="Extra field"):
            await structured(
                SimpleModel,
                stream=json_stream,
                strict_mode=True,
            )

    @pytest.mark.asyncio
    async def test_strict_mode_allows_valid_fields(self):
        """Test that strict_mode allows JSON with only valid fields."""

        async def json_stream():
            yield Event(
                type=EventType.TOKEN,
                text='{"value": "test"}',
            )
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            SimpleModel,
            stream=json_stream,
            strict_mode=True,
        )

        assert result.data.value == "test"

    @pytest.mark.asyncio
    async def test_non_strict_mode_allows_extra_fields(self):
        """Test that non-strict mode (default) ignores extra fields."""

        async def json_stream():
            # JSON with extra field - should be ignored in non-strict mode
            yield Event(
                type=EventType.TOKEN,
                text='{"value": "test", "extra_field": "ignored"}',
            )
            yield Event(type=EventType.COMPLETE)

        result = await structured(
            SimpleModel,
            stream=json_stream,
            strict_mode=False,
        )

        assert result.data.value == "test"
