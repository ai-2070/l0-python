"""Tests for multimodal support."""

import pytest

from l0 import (
    ContentType,
    DataPayload,
    Event,
    EventType,
    Progress,
    State,
    create_audio_event,
    create_audio_payload,
    create_complete_event,
    create_data_event,
    create_error_event,
    create_file_event,
    create_image_event,
    create_image_payload,
    create_json_event,
    create_progress_event,
    create_video_event,
    to_multimodal_events,
)

# ─────────────────────────────────────────────────────────────────────────────
# ContentType Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_type_values(self):
        """Test all content type values."""
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.AUDIO == "audio"
        assert ContentType.VIDEO == "video"
        assert ContentType.FILE == "file"
        assert ContentType.JSON == "json"
        assert ContentType.BINARY == "binary"

    def test_content_type_is_string_enum(self):
        """Test that ContentType is a string enum."""
        assert isinstance(ContentType.IMAGE, str)
        assert ContentType.IMAGE == "image"


# ─────────────────────────────────────────────────────────────────────────────
# DataPayload Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDataPayload:
    """Tests for DataPayload dataclass."""

    def test_basic_payload(self):
        """Test creating a basic payload."""
        payload = DataPayload(
            content_type=ContentType.IMAGE,
            mime_type="image/png",
            base64="abc123",
        )
        assert payload.content_type == ContentType.IMAGE
        assert payload.mime_type == "image/png"
        assert payload.base64 == "abc123"

    def test_payload_with_metadata(self):
        """Test payload with metadata."""
        payload = DataPayload(
            content_type=ContentType.IMAGE,
            metadata={"width": 1024, "height": 768, "seed": 42},
        )
        assert payload.width == 1024
        assert payload.height == 768
        assert payload.seed == 42

    def test_payload_metadata_properties(self):
        """Test all metadata convenience properties."""
        payload = DataPayload(
            content_type=ContentType.VIDEO,
            metadata={
                "width": 1920,
                "height": 1080,
                "duration": 30.5,
                "size": 1024000,
                "filename": "output.mp4",
                "seed": 123,
                "model": "stable-video",
            },
        )
        assert payload.width == 1920
        assert payload.height == 1080
        assert payload.duration == 30.5
        assert payload.size == 1024000
        assert payload.filename == "output.mp4"
        assert payload.seed == 123
        assert payload.model == "stable-video"

    def test_payload_missing_metadata(self):
        """Test metadata properties when metadata is None."""
        payload = DataPayload(content_type=ContentType.IMAGE)
        assert payload.width is None
        assert payload.height is None
        assert payload.duration is None
        assert payload.size is None
        assert payload.filename is None
        assert payload.seed is None
        assert payload.model is None

    def test_payload_with_url(self):
        """Test payload with URL."""
        payload = DataPayload(
            content_type=ContentType.IMAGE,
            url="https://example.com/image.png",
        )
        assert payload.url == "https://example.com/image.png"

    def test_payload_with_bytes(self):
        """Test payload with raw bytes."""
        raw_data = b"\x89PNG\r\n\x1a\n"
        payload = DataPayload(
            content_type=ContentType.IMAGE,
            data=raw_data,
        )
        assert payload.data == raw_data

    def test_payload_with_json(self):
        """Test payload with JSON data."""
        json_data = {"result": "success", "items": [1, 2, 3]}
        payload = DataPayload(
            content_type=ContentType.JSON,
            json=json_data,
        )
        assert payload.json == json_data


# ─────────────────────────────────────────────────────────────────────────────
# Progress Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProgress:
    """Tests for Progress dataclass."""

    def test_basic_progress(self):
        """Test creating basic progress."""
        progress = Progress(percent=50.0)
        assert progress.percent == 50.0

    def test_progress_with_steps(self):
        """Test progress with steps."""
        progress = Progress(step=3, total_steps=10)
        assert progress.step == 3
        assert progress.total_steps == 10

    def test_progress_with_message(self):
        """Test progress with message."""
        progress = Progress(message="Generating image...")
        assert progress.message == "Generating image..."

    def test_progress_with_eta(self):
        """Test progress with ETA."""
        progress = Progress(percent=75.0, eta=5.5)
        assert progress.eta == 5.5

    def test_full_progress(self):
        """Test progress with all fields."""
        progress = Progress(
            percent=50.0,
            step=5,
            total_steps=10,
            message="Processing...",
            eta=10.0,
        )
        assert progress.percent == 50.0
        assert progress.step == 5
        assert progress.total_steps == 10
        assert progress.message == "Processing..."
        assert progress.eta == 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Event Tests (Multimodal)
# ─────────────────────────────────────────────────────────────────────────────


class TestEventMultimodal:
    """Tests for Event with multimodal support."""

    def test_event_with_payload(self):
        """Test event with data payload."""
        payload = DataPayload(content_type=ContentType.IMAGE, base64="abc")
        event = Event(type=EventType.DATA, payload=payload)
        assert event.is_data
        assert event.payload is payload

    def test_event_with_progress(self):
        """Test event with progress."""
        progress = Progress(percent=50.0)
        event = Event(type=EventType.PROGRESS, progress=progress)
        assert event.is_progress
        assert event.progress is progress
        assert event.progress.percent == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# State Tests (Multimodal)
# ─────────────────────────────────────────────────────────────────────────────


class TestStateMultimodal:
    """Tests for State with multimodal support."""

    def test_state_data_outputs(self):
        """Test state tracks data outputs."""
        state = State()
        assert state.data_outputs == []

        payload = DataPayload(content_type=ContentType.IMAGE, base64="abc")
        state.data_outputs.append(payload)
        assert len(state.data_outputs) == 1
        assert state.data_outputs[0].content_type == ContentType.IMAGE

    def test_state_last_progress(self):
        """Test state tracks last progress."""
        state = State()
        assert state.last_progress is None

        state.last_progress = Progress(percent=75.0)
        assert state.last_progress.percent == 75.0


# ─────────────────────────────────────────────────────────────────────────────
# Event Creator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEventCreators:
    """Tests for event creation helpers."""

    def test_create_data_event(self):
        """Test create_data_event."""
        payload = DataPayload(content_type=ContentType.IMAGE, base64="abc")
        event = create_data_event(payload)
        assert event.type == EventType.DATA
        assert event.payload is payload

    def test_create_progress_event(self):
        """Test create_progress_event."""
        event = create_progress_event(percent=50.0, message="Working...")
        assert event.type == EventType.PROGRESS
        assert event.progress.percent == 50.0
        assert event.progress.message == "Working..."

    def test_create_progress_event_with_steps(self):
        """Test create_progress_event with steps."""
        event = create_progress_event(step=3, total_steps=10, eta=5.0)
        assert event.progress.step == 3
        assert event.progress.total_steps == 10
        assert event.progress.eta == 5.0

    def test_create_image_event(self):
        """Test create_image_event."""
        event = create_image_event(
            base64="abc123",
            width=1024,
            height=768,
            seed=42,
            model="dall-e-3",
        )
        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.IMAGE
        assert event.payload.base64 == "abc123"
        assert event.payload.mime_type == "image/png"
        assert event.payload.width == 1024
        assert event.payload.height == 768
        assert event.payload.seed == 42
        assert event.payload.model == "dall-e-3"

    def test_create_image_event_with_url(self):
        """Test create_image_event with URL."""
        event = create_image_event(url="https://example.com/image.png")
        assert event.payload.url == "https://example.com/image.png"

    def test_create_image_event_custom_mime_type(self):
        """Test create_image_event with custom MIME type."""
        event = create_image_event(base64="abc", mime_type="image/jpeg")
        assert event.payload.mime_type == "image/jpeg"

    def test_create_image_event_extra_metadata(self):
        """Test create_image_event with extra metadata."""
        event = create_image_event(base64="abc", prompt="a cat", steps=50)
        assert event.payload.metadata["prompt"] == "a cat"
        assert event.payload.metadata["steps"] == 50

    def test_create_audio_event(self):
        """Test create_audio_event."""
        event = create_audio_event(
            base64="audio_data",
            duration=10.5,
            model="tts-1",
        )
        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.AUDIO
        assert event.payload.base64 == "audio_data"
        assert event.payload.mime_type == "audio/mp3"
        assert event.payload.duration == 10.5
        assert event.payload.model == "tts-1"

    def test_create_video_event(self):
        """Test create_video_event."""
        event = create_video_event(
            url="https://example.com/video.mp4",
            width=1920,
            height=1080,
            duration=30.0,
            model="sora",
        )
        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.VIDEO
        assert event.payload.url == "https://example.com/video.mp4"
        assert event.payload.mime_type == "video/mp4"
        assert event.payload.width == 1920
        assert event.payload.height == 1080
        assert event.payload.duration == 30.0
        assert event.payload.model == "sora"

    def test_create_file_event(self):
        """Test create_file_event."""
        event = create_file_event(
            base64="file_data",
            filename="output.pdf",
            size=1024,
            mime_type="application/pdf",
        )
        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.FILE
        assert event.payload.base64 == "file_data"
        assert event.payload.filename == "output.pdf"
        assert event.payload.size == 1024
        assert event.payload.mime_type == "application/pdf"

    def test_create_json_event(self):
        """Test create_json_event."""
        data = {"result": "success", "count": 42}
        event = create_json_event(data, model="gpt-4o")
        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.JSON
        assert event.payload.json == data
        assert event.payload.mime_type == "application/json"
        assert event.payload.model == "gpt-4o"

    def test_create_complete_event(self):
        """Test create_complete_event."""
        event = create_complete_event()
        assert event.type == EventType.COMPLETE
        assert event.usage is None

    def test_create_complete_event_with_usage(self):
        """Test create_complete_event with usage."""
        event = create_complete_event(
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
        assert event.type == EventType.COMPLETE
        assert event.usage["prompt_tokens"] == 10
        assert event.usage["completion_tokens"] == 20

    def test_create_error_event(self):
        """Test create_error_event."""
        error = ValueError("Something went wrong")
        event = create_error_event(error)
        assert event.type == EventType.ERROR
        assert event.error is error


# ─────────────────────────────────────────────────────────────────────────────
# Payload Creator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPayloadCreators:
    """Tests for payload creation helpers."""

    def test_create_image_payload(self):
        """Test create_image_payload."""
        payload = create_image_payload(
            base64="abc",
            width=512,
            height=512,
            seed=42,
        )
        assert payload.content_type == ContentType.IMAGE
        assert payload.base64 == "abc"
        assert payload.width == 512
        assert payload.height == 512
        assert payload.seed == 42

    def test_create_audio_payload(self):
        """Test create_audio_payload."""
        payload = create_audio_payload(
            base64="audio",
            duration=5.0,
            model="whisper",
        )
        assert payload.content_type == ContentType.AUDIO
        assert payload.base64 == "audio"
        assert payload.duration == 5.0
        assert payload.model == "whisper"


# ─────────────────────────────────────────────────────────────────────────────
# Stream Converter Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestToMultimodalEvents:
    """Tests for to_multimodal_events helper."""

    @pytest.mark.asyncio
    async def test_basic_stream_conversion(self):
        """Test basic stream conversion."""

        async def mock_stream():
            yield {"type": "progress", "percent": 50}
            yield {"type": "image", "data": "abc123"}

        events = []
        async for event in to_multimodal_events(
            mock_stream(),
            extract_progress=lambda c: Progress(percent=c["percent"])
            if c.get("type") == "progress"
            else None,
            extract_data=lambda c: create_image_payload(base64=c["data"])
            if c.get("type") == "image"
            else None,
        ):
            events.append(event)

        assert len(events) == 3  # progress, data, complete
        assert events[0].is_progress
        assert events[0].progress.percent == 50
        assert events[1].is_data
        assert events[1].payload.base64 == "abc123"
        assert events[2].is_complete

    @pytest.mark.asyncio
    async def test_text_extraction(self):
        """Test text extraction."""

        async def mock_stream():
            yield {"text": "Hello"}
            yield {"text": " World"}

        events = []
        async for event in to_multimodal_events(
            mock_stream(),
            extract_text=lambda c: c.get("text"),
        ):
            events.append(event)

        assert len(events) == 3  # 2 tokens + complete
        assert events[0].is_token
        assert events[0].text == "Hello"
        assert events[1].is_token
        assert events[1].text == " World"

    @pytest.mark.asyncio
    async def test_error_extraction(self):
        """Test error extraction."""

        async def mock_stream():
            yield {"error": ValueError("test error")}

        events = []
        async for event in to_multimodal_events(
            mock_stream(),
            extract_error=lambda c: c.get("error"),
        ):
            events.append(event)

        assert len(events) == 2  # error + complete
        assert events[0].is_error
        assert isinstance(events[0].error, ValueError)

    @pytest.mark.asyncio
    async def test_stream_exception_handling(self):
        """Test exception handling in stream."""

        async def failing_stream():
            yield {"text": "ok"}
            raise ConnectionError("Stream failed")

        events = []
        async for event in to_multimodal_events(
            failing_stream(),
            extract_text=lambda c: c.get("text"),
        ):
            events.append(event)

        assert len(events) == 2  # token + error
        assert events[0].is_token
        assert events[1].is_error
        assert isinstance(events[1].error, ConnectionError)

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test empty stream."""

        async def empty_stream():
            return
            yield  # Make it a generator

        events = []
        async for event in to_multimodal_events(empty_stream()):
            events.append(event)

        assert len(events) == 1
        assert events[0].is_complete

    @pytest.mark.asyncio
    async def test_mixed_content(self):
        """Test mixed content extraction."""

        async def mock_stream():
            yield {"progress": 25}
            yield {"text": "Processing..."}
            yield {"progress": 75}
            yield {"image": "result"}

        events = []
        async for event in to_multimodal_events(
            mock_stream(),
            extract_progress=lambda c: Progress(percent=c["progress"])
            if "progress" in c
            else None,
            extract_text=lambda c: c.get("text"),
            extract_data=lambda c: create_image_payload(base64=c["image"])
            if "image" in c
            else None,
        ):
            events.append(event)

        # Should have: progress, token, progress, data, complete
        assert len(events) == 5
        assert events[0].is_progress
        assert events[0].progress.percent == 25
        assert events[1].is_token
        assert events[1].text == "Processing..."
        assert events[2].is_progress
        assert events[2].progress.percent == 75
        assert events[3].is_data
        assert events[3].payload.base64 == "result"
        assert events[4].is_complete
