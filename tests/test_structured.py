"""Tests for l0.structured module."""

import pytest
from pydantic import BaseModel

from l0.adapters import register_adapter
from l0.structured import structured
from l0.types import Event, EventType


# Test adapter that passes through Event objects
class PassthroughAdapter:
    name = "passthrough_structured"

    def detect(self, stream):
        return hasattr(stream, "__anext__")

    async def wrap(self, stream):
        async for event in stream:
            yield event


# Ensure adapter is registered
register_adapter(PassthroughAdapter())


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

        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

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

        assert result.value == "test"

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

        assert result.value == "test"

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

        assert result.value == "test"

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

        assert result.value == "hello world"
