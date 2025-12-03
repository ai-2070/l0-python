"""Tests for multimodal support."""

from l0 import (
    ContentType,
    DataPayload,
    Event,
    EventType,
    Multimodal,
    Progress,
    State,
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
# Multimodal Scoped API Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultimodal:
    """Tests for Multimodal scoped API."""

    def test_data(self):
        """Test Multimodal.data()."""
        payload = DataPayload(content_type=ContentType.IMAGE, base64="abc")
        event = Multimodal.data(payload)
        assert event.type == EventType.DATA
        assert event.payload is payload

    def test_progress(self):
        """Test Multimodal.progress()."""
        event = Multimodal.progress(percent=50.0, message="Working...")
        assert event.type == EventType.PROGRESS
        assert event.progress.percent == 50.0
        assert event.progress.message == "Working..."

    def test_progress_with_steps(self):
        """Test Multimodal.progress() with steps."""
        event = Multimodal.progress(step=3, total_steps=10, eta=5.0)
        assert event.progress.step == 3
        assert event.progress.total_steps == 10
        assert event.progress.eta == 5.0

    def test_image(self):
        """Test Multimodal.image()."""
        event = Multimodal.image(
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

    def test_image_with_url(self):
        """Test Multimodal.image() with URL."""
        event = Multimodal.image(url="https://example.com/image.png")
        assert event.payload is not None
        assert event.payload.url == "https://example.com/image.png"

    def test_image_custom_mime_type(self):
        """Test Multimodal.image() with custom MIME type."""
        event = Multimodal.image(base64="abc", mime_type="image/jpeg")
        assert event.payload is not None
        assert event.payload.mime_type == "image/jpeg"

    def test_image_extra_metadata(self):
        """Test Multimodal.image() with extra metadata."""
        event = Multimodal.image(base64="abc", prompt="a cat", steps=50)
        assert event.payload is not None
        assert event.payload.metadata is not None
        assert event.payload.metadata["prompt"] == "a cat"
        assert event.payload.metadata["steps"] == 50

    def test_audio(self):
        """Test Multimodal.audio()."""
        event = Multimodal.audio(
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

    def test_video(self):
        """Test Multimodal.video()."""
        event = Multimodal.video(
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

    def test_file(self):
        """Test Multimodal.file()."""
        event = Multimodal.file(
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

    def test_json(self):
        """Test Multimodal.json()."""
        data = {"result": "success", "count": 42}
        event = Multimodal.json(data, model="gpt-4o")
        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.JSON
        assert event.payload.json == data
        assert event.payload.mime_type == "application/json"
        assert event.payload.model == "gpt-4o"

    def test_complete(self):
        """Test Multimodal.complete()."""
        event = Multimodal.complete()
        assert event.type == EventType.COMPLETE
        assert event.usage is None

    def test_complete_with_usage(self):
        """Test Multimodal.complete() with usage."""
        event = Multimodal.complete(
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
        assert event.usage["prompt_tokens"] == 10
        assert event.usage["completion_tokens"] == 20

    def test_error(self):
        """Test Multimodal.error()."""
        error = ValueError("Something went wrong")
        event = Multimodal.error(error)
        assert event.type == EventType.ERROR
        assert event.error is error
