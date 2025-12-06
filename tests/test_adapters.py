"""Tests for l0.adapters module."""

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from l0.adapters import (
    AdaptedEvent,
    Adapters,
    EventPassthroughAdapter,
    OpenAIAdapter,
    OpenAIAdapterOptions,
    create_audio_event,
    create_complete_event,
    create_data_event,
    create_error_event,
    create_image_event,
    create_progress_event,
    create_token_event,
    to_l0_events,
    to_l0_events_with_messages,
    to_multimodal_l0_events,
)
from l0.types import ContentType, DataPayload, Event, EventType, Progress


class MockDelta:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None):
        self.content = content
        self.tool_calls = tool_calls


class MockToolCallFunction:
    def __init__(self, name: str | None = None, arguments: str | None = None):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ):
        self.id = id
        self.function = MockToolCallFunction(name, arguments)


class MockChoice:
    def __init__(self, delta: MockDelta | None = None):
        self.delta = delta


class MockUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockChunk:
    def __init__(
        self, choices: list[MockChoice] | None = None, usage: MockUsage | None = None
    ):
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

            def detect(self, stream: Any) -> bool:
                return hasattr(stream, "_custom_marker")

            async def wrap(
                self, stream: Any, options: Any = None
            ) -> AsyncIterator[Any]:
                from l0.adapters import AdaptedEvent

                yield AdaptedEvent(event=Event(type=EventType.COMPLETE), raw_chunk=None)

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
        names = Adapters.registered()
        assert names == ["openai", "event"]

    def test_list_after_register(self):
        """Test listing after registering an adapter."""

        class FakeAdapter:
            name: str = "fake"

            def detect(self, stream: Any) -> bool:
                return False

            async def wrap(
                self, stream: Any, options: Any = None
            ) -> AsyncIterator[Any]:
                from l0.adapters import AdaptedEvent

                yield AdaptedEvent(event=Event(type=EventType.COMPLETE), raw_chunk=None)

        Adapters.register(FakeAdapter())
        names = Adapters.registered()
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

            async def wrap(
                self, stream: Any, options: Any = None
            ) -> AsyncIterator[Any]:
                from l0.adapters import AdaptedEvent

                yield AdaptedEvent(event=Event(type=EventType.COMPLETE), raw_chunk=None)

        Adapters.register(FakeAdapter())
        assert "fake" in Adapters.registered()

        result = Adapters.unregister("fake")
        assert result is True
        assert "fake" not in Adapters.registered()

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
        assert len(Adapters.registered()) > 0
        Adapters.clear()
        assert Adapters.registered() == []


class TestAdaptersReset:
    def test_reset_restores_default(self):
        """Test that reset restores default adapters."""
        Adapters.clear()
        assert Adapters.registered() == []

        Adapters.reset()
        assert Adapters.registered() == ["openai", "event"]

    def test_reset_removes_custom(self):
        """Test that reset removes custom adapters."""

        class FakeAdapter:
            name: str = "fake"

            def detect(self, stream: Any) -> bool:
                return False

            async def wrap(
                self, stream: Any, options: Any = None
            ) -> AsyncIterator[Any]:
                from l0.adapters import AdaptedEvent

                yield AdaptedEvent(event=Event(type=EventType.COMPLETE), raw_chunk=None)

        Adapters.register(FakeAdapter())
        assert "fake" in Adapters.registered()

        Adapters.reset()
        assert Adapters.registered() == ["openai", "event"]


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


class TestAdaptedEvent:
    """Tests for AdaptedEvent dataclass."""

    def test_adapted_event_with_raw_chunk(self):
        """Test AdaptedEvent with raw chunk."""
        event = Event(type=EventType.TOKEN, text="Hello")
        raw = {"raw": "data"}
        adapted = AdaptedEvent(event=event, raw_chunk=raw)

        assert adapted.event == event
        assert adapted.raw_chunk == raw

    def test_adapted_event_without_raw_chunk(self):
        """Test AdaptedEvent without raw chunk."""
        event = Event(type=EventType.COMPLETE)
        adapted = AdaptedEvent(event=event)

        assert adapted.event == event
        assert adapted.raw_chunk is None


class TestOpenAIAdapterOptions:
    """Tests for OpenAIAdapterOptions."""

    def test_default_options(self):
        """Test default option values."""
        opts = OpenAIAdapterOptions()
        assert opts.include_usage is True
        assert opts.include_tool_calls is True
        assert opts.emit_function_calls_as_tokens is False
        assert opts.choice_index == 0

    def test_custom_options(self):
        """Test custom option values."""
        opts = OpenAIAdapterOptions(
            include_usage=False,
            include_tool_calls=False,
            emit_function_calls_as_tokens=True,
            choice_index="all",
        )
        assert opts.include_usage is False
        assert opts.include_tool_calls is False
        assert opts.emit_function_calls_as_tokens is True
        assert opts.choice_index == "all"


class TestOpenAIAdapterChoiceIndex:
    """Tests for OpenAI adapter choice_index option."""

    @pytest.mark.asyncio
    async def test_choice_index_all(self):
        """Test processing all choices."""
        adapter = OpenAIAdapter()
        opts = OpenAIAdapterOptions(choice_index="all")

        async def mock_stream():
            yield MockChunk(
                choices=[
                    MockChoice(delta=MockDelta(content="Choice 0")),
                    MockChoice(delta=MockDelta(content="Choice 1")),
                ]
            )

        events = []
        async for adapted in adapter.wrap(mock_stream(), opts):
            events.append(adapted)

        # Should have 2 tokens + 1 complete
        assert len(events) == 3
        token_events = [e for e in events if e.event.type == EventType.TOKEN]
        assert len(token_events) == 2
        assert token_events[0].event.text == "Choice 0"
        assert token_events[1].event.text == "Choice 1"

    @pytest.mark.asyncio
    async def test_choice_index_specific(self):
        """Test processing specific choice index."""
        adapter = OpenAIAdapter()
        opts = OpenAIAdapterOptions(choice_index=1)

        async def mock_stream():
            yield MockChunk(
                choices=[
                    MockChoice(delta=MockDelta(content="Choice 0")),
                    MockChoice(delta=MockDelta(content="Choice 1")),
                ]
            )

        events = []
        async for adapted in adapter.wrap(mock_stream(), opts):
            events.append(adapted)

        # Should have 1 token + 1 complete
        token_events = [e for e in events if e.event.type == EventType.TOKEN]
        assert len(token_events) == 1
        assert token_events[0].event.text == "Choice 1"

    @pytest.mark.asyncio
    async def test_emit_function_calls_as_tokens(self):
        """Test emitting function call arguments as tokens."""
        adapter = OpenAIAdapter()
        opts = OpenAIAdapterOptions(emit_function_calls_as_tokens=True)

        async def mock_stream():
            yield MockChunk(
                choices=[
                    MockChoice(
                        delta=MockDelta(
                            tool_calls=[
                                MockToolCall(
                                    id="call_1",
                                    name="func",
                                    arguments='{"arg": 1}',
                                )
                            ]
                        )
                    )
                ]
            )

        events = []
        async for adapted in adapter.wrap(mock_stream(), opts):
            events.append(adapted)

        # Should emit token for arguments then tool_call
        token_events = [e for e in events if e.event.type == EventType.TOKEN]
        tool_events = [e for e in events if e.event.type == EventType.TOOL_CALL]

        assert len(token_events) == 1
        assert token_events[0].event.text == '{"arg": 1}'
        assert len(tool_events) == 1


class TestEventPassthroughAdapter:
    """Tests for EventPassthroughAdapter."""

    def test_detect(self):
        """Test passthrough adapter detects async iterators."""
        adapter = EventPassthroughAdapter()

        async def async_gen():
            yield Event(type=EventType.TOKEN, text="test")

        assert adapter.detect(async_gen()) is True
        assert adapter.detect("not an iterator") is False

    @pytest.mark.asyncio
    async def test_wrap_events(self):
        """Test passthrough adapter wraps Event objects."""
        adapter = EventPassthroughAdapter()

        async def event_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        events = []
        async for adapted in adapter.wrap(event_stream()):
            events.append(adapted)

        assert len(events) == 2
        assert events[0].event.type == EventType.TOKEN
        assert events[0].event.text == "Hello"
        assert events[1].event.type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_wrap_skips_non_events(self):
        """Test passthrough adapter skips non-Event objects."""
        adapter = EventPassthroughAdapter()

        async def mixed_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield "not an event"
            yield {"also": "not an event"}
            yield Event(type=EventType.COMPLETE)

        events = []
        async for adapted in adapter.wrap(mixed_stream()):
            events.append(adapted)

        assert len(events) == 2
        assert events[0].event.type == EventType.TOKEN
        assert events[1].event.type == EventType.COMPLETE


class TestToL0Events:
    """Tests for to_l0_events helper function."""

    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        """Test basic text extraction from chunks."""

        async def mock_stream():
            yield {"text": "Hello"}
            yield {"text": " "}
            yield {"text": "World"}

        events = []
        async for event in to_l0_events(mock_stream(), lambda c: c.get("text")):
            events.append(event)

        assert len(events) == 4  # 3 tokens + 1 complete
        assert events[0].type == EventType.TOKEN
        assert events[0].text == "Hello"
        assert events[3].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_skip_none_extraction(self):
        """Test that None extractions are skipped."""

        async def mock_stream():
            yield {"text": "Hello"}
            yield {"other": "data"}
            yield {"text": "World"}

        events = []
        async for event in to_l0_events(mock_stream(), lambda c: c.get("text")):
            events.append(event)

        # Should have 2 tokens + 1 complete (skipped middle chunk)
        assert len(events) == 3
        assert events[0].text == "Hello"
        assert events[1].text == "World"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during extraction."""

        async def failing_stream():
            yield {"text": "Hello"}
            raise ValueError("Stream error")

        events = []
        async for event in to_l0_events(failing_stream(), lambda c: c.get("text")):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == EventType.TOKEN
        assert events[1].type == EventType.ERROR
        assert isinstance(events[1].error, ValueError)


class TestToL0EventsWithMessages:
    """Tests for to_l0_events_with_messages helper function."""

    @pytest.mark.asyncio
    async def test_text_and_message_extraction(self):
        """Test extracting both text and message events."""

        async def mock_stream():
            yield {"type": "text", "content": "Hello"}
            yield {"type": "tool", "tool_call": {"name": "func"}}
            yield {"type": "text", "content": "World"}

        events = []
        async for event in to_l0_events_with_messages(
            mock_stream(),
            extract_text=lambda c: str(c["content"]) if c["type"] == "text" else None,
            extract_message=lambda c: cast(dict[str, Any], c["tool_call"])
            if c["type"] == "tool"
            else None,
        ):
            events.append(event)

        assert len(events) == 4  # 2 tokens + 1 message + 1 complete
        assert events[0].type == EventType.TOKEN
        assert events[0].text == "Hello"
        assert events[1].type == EventType.MESSAGE
        assert events[1].data == {"name": "func"}
        assert events[2].type == EventType.TOKEN
        assert events[3].type == EventType.COMPLETE


class TestToMultimodalL0Events:
    """Tests for to_multimodal_l0_events helper function."""

    @pytest.mark.asyncio
    async def test_multimodal_extraction(self):
        """Test extracting multiple content types."""

        async def mock_stream():
            yield {"type": "text", "content": "Hello"}
            yield {"type": "image", "data": "base64data"}
            yield {"type": "progress", "percent": 50}
            yield {"type": "text", "content": "Done"}

        events = []
        async for event in to_multimodal_l0_events(
            mock_stream(),
            extract_text=lambda c: str(c["content"]) if c["type"] == "text" else None,
            extract_data=lambda c: (
                DataPayload(
                    content_type=ContentType.IMAGE,
                    mime_type="image/png",
                    base64=str(c["data"]),
                )
                if c["type"] == "image"
                else None
            ),
            extract_progress=lambda c: (
                Progress(percent=float(c["percent"]))
                if c["type"] == "progress"
                else None
            ),
        ):
            events.append(event)

        assert len(events) == 5  # 2 tokens + 1 data + 1 progress + 1 complete
        assert events[0].type == EventType.TOKEN
        assert events[1].type == EventType.DATA
        assert events[1].payload.base64 == "base64data"
        assert events[2].type == EventType.PROGRESS
        assert events[2].progress.percent == 50
        assert events[4].type == EventType.COMPLETE


class TestCreateEventHelpers:
    """Tests for event creation helper functions."""

    def test_create_token_event(self):
        """Test creating token event."""
        event = create_token_event("Hello")
        assert event.type == EventType.TOKEN
        assert event.text == "Hello"

    def test_create_complete_event(self):
        """Test creating complete event."""
        event = create_complete_event()
        assert event.type == EventType.COMPLETE
        assert event.usage is None

    def test_create_complete_event_with_usage(self):
        """Test creating complete event with usage."""
        usage = {"input_tokens": 10, "output_tokens": 5}
        event = create_complete_event(usage)
        assert event.type == EventType.COMPLETE
        assert event.usage == usage

    def test_create_error_event_from_exception(self):
        """Test creating error event from exception."""
        error = ValueError("test error")
        event = create_error_event(error)
        assert event.type == EventType.ERROR
        assert event.error is error

    def test_create_error_event_from_string(self):
        """Test creating error event from string."""
        event = create_error_event("test error")
        assert event.type == EventType.ERROR
        assert isinstance(event.error, Exception)
        assert str(event.error) == "test error"

    def test_create_data_event(self):
        """Test creating data event."""
        payload = DataPayload(
            content_type=ContentType.IMAGE,
            mime_type="image/png",
            base64="abc123",
        )
        event = create_data_event(payload)
        assert event.type == EventType.DATA
        assert event.payload == payload

    def test_create_progress_event(self):
        """Test creating progress event."""
        progress = Progress(percent=75, message="Processing...")
        event = create_progress_event(progress)
        assert event.type == EventType.PROGRESS
        assert event.progress == progress

    def test_create_image_event(self):
        """Test creating image data event."""
        event = create_image_event(
            url="https://example.com/image.png",
            mime_type="image/png",
            width=800,
            height=600,
        )
        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.IMAGE
        assert event.payload.url == "https://example.com/image.png"
        assert event.payload.metadata is not None
        assert event.payload.metadata["width"] == 800
        assert event.payload.metadata["height"] == 600

    def test_create_image_event_with_base64(self):
        """Test creating image event with base64 data."""
        event = create_image_event(base64="abc123==", mime_type="image/jpeg")
        assert event.payload is not None
        assert event.payload.base64 == "abc123=="
        assert event.payload.mime_type == "image/jpeg"

    def test_create_audio_event(self):
        """Test creating audio data event."""
        event = create_audio_event(
            url="https://example.com/audio.mp3",
            duration=120.5,
        )
        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.AUDIO
        assert event.payload.url == "https://example.com/audio.mp3"
        assert event.payload.metadata is not None
        assert event.payload.metadata["duration"] == 120.5

    def test_create_audio_event_with_base64(self):
        """Test creating audio event with base64 data."""
        event = create_audio_event(
            base64="audio_data==",
            mime_type="audio/wav",
        )
        assert event.payload is not None
        assert event.payload.base64 == "audio_data=="
        assert event.payload.mime_type == "audio/wav"


class TestAdaptersHelperMethods:
    """Tests for Adapters class helper methods."""

    def test_token_event(self):
        """Test Adapters.token_event helper."""
        event = Adapters.token_event("test")
        assert event.type == EventType.TOKEN
        assert event.text == "test"

    def test_complete_event(self):
        """Test Adapters.complete_event helper."""
        event = Adapters.complete_event()
        assert event.type == EventType.COMPLETE

    def test_error_event(self):
        """Test Adapters.error_event helper."""
        event = Adapters.error_event("error message")
        assert event.type == EventType.ERROR

    def test_data_event(self):
        """Test Adapters.data_event helper."""
        payload = DataPayload(content_type=ContentType.IMAGE, mime_type="image/png")
        event = Adapters.data_event(payload)
        assert event.type == EventType.DATA

    def test_progress_event(self):
        """Test Adapters.progress_event helper."""
        progress = Progress(percent=50)
        event = Adapters.progress_event(progress)
        assert event.type == EventType.PROGRESS

    def test_image_event(self):
        """Test Adapters.image_event helper."""
        event = Adapters.image_event(url="https://example.com/img.png")
        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.IMAGE

    def test_audio_event(self):
        """Test Adapters.audio_event helper."""
        event = Adapters.audio_event(url="https://example.com/audio.mp3")
        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.AUDIO


class TestAdaptersAdditionalMethods:
    """Tests for additional Adapters class methods."""

    def setup_method(self):
        """Reset adapters before each test."""
        Adapters.reset()

    def teardown_method(self):
        """Reset adapters after each test."""
        Adapters.reset()

    def test_get_existing(self):
        """Test Adapters.get for existing adapter."""
        adapter = Adapters.get("openai")
        assert adapter is not None
        assert adapter.name == "openai"

    def test_get_nonexistent(self):
        """Test Adapters.get for non-existent adapter."""
        adapter = Adapters.get("nonexistent")
        assert adapter is None

    def test_has_matching_true(self):
        """Test Adapters.has_matching when one adapter matches."""
        chunk = MockChunk()
        assert Adapters.has_matching(chunk) is True

    def test_detect_adapter_found(self):
        """Test Adapters.detect_adapter when adapter is found."""
        chunk = MockChunk()
        adapter = Adapters.detect_adapter(chunk)
        assert adapter is not None
        assert adapter.name == "openai"

    def test_detect_adapter_not_found(self):
        """Test Adapters.detect_adapter when no adapter matches."""

        class UnmatchedStream:
            pass

        # Need to clear adapters so nothing matches
        Adapters.clear()
        adapter = Adapters.detect_adapter(UnmatchedStream())
        assert adapter is None

    def test_unregister_all_except(self):
        """Test Adapters.unregister_all_except."""
        # Keep only openai
        removed = Adapters.unregister_all_except(["openai"])
        assert "event" in removed
        assert "openai" not in removed
        assert Adapters.registered() == ["openai"]

    def test_unregister_all_except_none(self):
        """Test Adapters.unregister_all_except with no exceptions."""
        removed = Adapters.unregister_all_except()
        assert "openai" in removed
        assert "event" in removed
        assert Adapters.registered() == []
