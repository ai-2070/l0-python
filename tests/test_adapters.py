"""Tests for l0.adapters module."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from l0.adapters import Adapters, OpenAIAdapter
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

        adapted_events = []
        async for adapted in adapter.wrap(mock_stream()):
            adapted_events.append(adapted)

        assert len(adapted_events) == 3
        assert adapted_events[0].event.type == EventType.TOKEN
        assert adapted_events[0].event.text == "Hello"
        assert adapted_events[0].raw_chunk is not None  # Raw chunk preserved
        assert adapted_events[1].event.type == EventType.TOKEN
        assert adapted_events[1].event.text == " world"
        assert adapted_events[2].event.type == EventType.COMPLETE

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

        adapted_events = []
        async for adapted in adapter.wrap(mock_stream()):
            adapted_events.append(adapted)

        assert len(adapted_events) == 2
        assert adapted_events[0].event.type == EventType.TOOL_CALL
        assert adapted_events[0].event.data["id"] == "call_123"
        assert adapted_events[0].event.data["name"] == "get_weather"
        assert adapted_events[0].event.data["arguments"] == '{"location": "NYC"}'
        assert adapted_events[1].event.type == EventType.COMPLETE

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

        adapted_events = []
        async for adapted in adapter.wrap(mock_stream()):
            adapted_events.append(adapted)

        assert len(adapted_events) == 2
        assert adapted_events[0].event.type == EventType.TOKEN
        assert adapted_events[1].event.type == EventType.COMPLETE
        assert adapted_events[1].event.usage == {"input_tokens": 10, "output_tokens": 5}

    @pytest.mark.asyncio
    async def test_wrap_empty_stream(self):
        """Test that adapter handles empty stream."""
        adapter = OpenAIAdapter()

        async def mock_stream():
            if False:
                yield  # Make it an async generator

        adapted_events = []
        async for adapted in adapter.wrap(mock_stream()):
            adapted_events.append(adapted)

        assert len(adapted_events) == 1
        assert adapted_events[0].event.type == EventType.COMPLETE


class TestAdaptersDetect:
    def test_detect_by_hint(self):
        """Test adapter detection by hint."""
        adapter = Adapters.detect(object(), hint="openai")
        assert adapter.name == "openai"

    def test_detect_litellm_hint(self):
        """Test that litellm hint maps to openai adapter."""
        adapter = Adapters.detect(object(), hint="litellm")
        assert adapter.name == "openai"

    def test_detect_unknown_hint_raises(self):
        """Test that unknown hint raises ValueError."""
        with pytest.raises(ValueError, match="Unknown adapter"):
            Adapters.detect(object(), hint="unknown")

    def test_detect_adapter_instance(self):
        """Test that adapter instance is returned directly."""
        custom = OpenAIAdapter()
        adapter = Adapters.detect(object(), hint=custom)
        assert adapter is custom

    def test_detect_no_match_raises(self):
        """Test that no match raises ValueError."""

        class UnknownStream:
            pass

        with pytest.raises(ValueError, match="No adapter found"):
            Adapters.detect(UnknownStream())


class TestAdaptersRegister:
    def setup_method(self):
        """Reset adapters before each test."""
        Adapters.reset()

    def teardown_method(self):
        """Reset adapters after each test."""
        Adapters.reset()

    def test_register_custom_adapter(self):
        """Test registering a custom adapter."""

        class CustomAdapter:
            name = "custom"

            def detect(self, stream):
                return hasattr(stream, "_custom_marker")

            async def wrap(self, stream):
                yield Event(type=EventType.COMPLETE)

        Adapters.register(CustomAdapter())

        class CustomStream:
            _custom_marker = True

        adapter = Adapters.detect(CustomStream())
        assert adapter.name == "custom"


class TestAdaptersList:
    def setup_method(self):
        """Reset adapters before each test."""
        Adapters.reset()

    def teardown_method(self):
        """Reset adapters after each test."""
        Adapters.reset()

    def test_list_default(self):
        """Test listing default adapters."""
        names = Adapters.list()
        assert names == ["openai", "event"]

    def test_list_after_register(self):
        """Test listing after registering an adapter."""

        class FakeAdapter:
            name: str = "fake"

            def detect(self, stream: Any) -> bool:
                return False

            async def wrap(self, stream: Any) -> AsyncIterator[Event]:
                yield Event(type=EventType.COMPLETE)

        Adapters.register(FakeAdapter())
        names = Adapters.list()
        assert names == ["fake", "openai", "event"]


class TestAdaptersUnregister:
    def setup_method(self):
        """Reset adapters before each test."""
        Adapters.reset()

    def teardown_method(self):
        """Reset adapters after each test."""
        Adapters.reset()

    def test_unregister_existing(self):
        """Test unregistering an existing adapter."""

        class FakeAdapter:
            name: str = "fake"

            def detect(self, stream: Any) -> bool:
                return False

            async def wrap(self, stream: Any) -> AsyncIterator[Event]:
                yield Event(type=EventType.COMPLETE)

        Adapters.register(FakeAdapter())
        assert "fake" in Adapters.list()

        result = Adapters.unregister("fake")
        assert result is True
        assert "fake" not in Adapters.list()

    def test_unregister_nonexistent(self):
        """Test unregistering a non-existent adapter."""
        result = Adapters.unregister("nonexistent")
        assert result is False


class TestAdaptersClear:
    def setup_method(self):
        """Reset adapters before each test."""
        Adapters.reset()

    def teardown_method(self):
        """Reset adapters after each test."""
        Adapters.reset()

    def test_clear(self):
        """Test clearing all adapters."""
        assert len(Adapters.list()) > 0
        Adapters.clear()
        assert Adapters.list() == []


class TestAdaptersReset:
    def test_reset_restores_default(self):
        """Test that reset restores default adapters."""
        Adapters.clear()
        assert Adapters.list() == []

        Adapters.reset()
        assert Adapters.list() == ["openai", "event"]

    def test_reset_removes_custom(self):
        """Test that reset removes custom adapters."""

        class FakeAdapter:
            name: str = "fake"

            def detect(self, stream: Any) -> bool:
                return False

            async def wrap(self, stream: Any) -> AsyncIterator[Event]:
                yield Event(type=EventType.COMPLETE)

        Adapters.register(FakeAdapter())
        assert "fake" in Adapters.list()

        Adapters.reset()
        assert Adapters.list() == ["openai", "event"]


class TestAdaptersFactories:
    def test_openai_factory(self):
        """Test Adapters.openai() factory."""
        adapter = Adapters.openai()
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.name == "openai"

    def test_litellm_factory(self):
        """Test Adapters.litellm() factory."""
        adapter = Adapters.litellm()
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.name == "openai"
