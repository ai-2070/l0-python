"""Tests for l0.adapters module."""

import pytest

from l0.adapters import OpenAIAdapter, detect_adapter, register_adapter
from l0.types import Event, EventType


class MockDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockToolCallFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, id=None, name=None, arguments=None):
        self.id = id
        self.function = MockToolCallFunction(name, arguments)


class MockChoice:
    def __init__(self, delta=None):
        self.delta = delta


class MockUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockChunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


# Make it look like openai module
MockChunk.__module__ = "openai.types.chat"


class TestOpenAIAdapter:
    def test_detect_openai_stream(self):
        """Test that adapter detects OpenAI streams."""
        adapter = OpenAIAdapter()
        chunk = MockChunk()
        assert adapter.detect(chunk) is True

    def test_detect_non_openai_stream(self):
        """Test that adapter rejects non-OpenAI streams."""
        adapter = OpenAIAdapter()

        class OtherStream:
            pass

        assert adapter.detect(OtherStream()) is False

    @pytest.mark.asyncio
    async def test_wrap_text_tokens(self):
        """Test that adapter wraps text tokens correctly."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(delta=MockDelta(content="Hello"))])
            yield MockChunk(choices=[MockChoice(delta=MockDelta(content=" world"))])
            yield MockChunk(choices=[MockChoice(delta=MockDelta())])

        events = []
        async for event in adapter.wrap(mock_stream()):
            events.append(event)

        assert len(events) == 3
        assert events[0].type == EventType.TOKEN
        assert events[0].text == "Hello"
        assert events[1].type == EventType.TOKEN
        assert events[1].text == " world"
        assert events[2].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_wrap_tool_calls(self):
        """Test that adapter wraps tool calls correctly."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            yield MockChunk(
                choices=[
                    MockChoice(
                        delta=MockDelta(
                            tool_calls=[
                                MockToolCall(
                                    id="call_123",
                                    name="get_weather",
                                    arguments='{"location": "NYC"}',
                                )
                            ]
                        )
                    )
                ]
            )
            yield MockChunk(choices=[MockChoice(delta=MockDelta())])

        events = []
        async for event in adapter.wrap(mock_stream()):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == EventType.TOOL_CALL
        assert events[0].data["id"] == "call_123"
        assert events[0].data["name"] == "get_weather"
        assert events[0].data["arguments"] == '{"location": "NYC"}'
        assert events[1].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_wrap_with_usage(self):
        """Test that adapter captures usage on completion."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(content="Hi"))],
            )
            yield MockChunk(
                choices=[],
                usage=MockUsage(prompt_tokens=10, completion_tokens=5),
            )

        events = []
        async for event in adapter.wrap(mock_stream()):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == EventType.TOKEN
        assert events[1].type == EventType.COMPLETE
        assert events[1].usage == {"input_tokens": 10, "output_tokens": 5}

    @pytest.mark.asyncio
    async def test_wrap_empty_stream(self):
        """Test that adapter handles empty stream."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            if False:
                yield  # Make it an async generator

        events = []
        async for event in adapter.wrap(mock_stream()):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == EventType.COMPLETE


class TestDetectAdapter:
    def test_detect_by_hint(self):
        """Test adapter detection by hint."""
        adapter = detect_adapter(object(), hint="openai")
        assert adapter.name == "openai"

    def test_detect_litellm_hint(self):
        """Test that litellm hint maps to openai adapter."""
        adapter = detect_adapter(object(), hint="litellm")
        assert adapter.name == "openai"

    def test_detect_unknown_hint_raises(self):
        """Test that unknown hint raises ValueError."""
        with pytest.raises(ValueError, match="Unknown adapter"):
            detect_adapter(object(), hint="unknown")

    def test_detect_adapter_instance(self):
        """Test that adapter instance is returned directly."""
        custom = OpenAIAdapter()
        adapter = detect_adapter(object(), hint=custom)
        assert adapter is custom

    def test_detect_no_match_raises(self):
        """Test that no match raises ValueError."""

        class UnknownStream:
            pass

        with pytest.raises(ValueError, match="No adapter found"):
            detect_adapter(UnknownStream())


class TestRegisterAdapter:
    def test_register_custom_adapter(self):
        """Test registering a custom adapter."""

        class CustomAdapter:
            name = "custom"

            def detect(self, stream):
                return hasattr(stream, "_custom_marker")

            async def wrap(self, stream):
                yield Event(type=EventType.COMPLETE)

        register_adapter(CustomAdapter())

        class CustomStream:
            _custom_marker = True

        adapter = detect_adapter(CustomStream())
        assert adapter.name == "custom"
