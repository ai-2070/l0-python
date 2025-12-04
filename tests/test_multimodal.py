"""Tests for L0 multimodal helpers."""

import pytest

from l0 import ContentType, DataPayload, EventType, Multimodal, Progress
from l0.types import Event


class TestMultimodalEventCreation:
    """Tests for Multimodal event creation methods."""

    def test_image_event(self):
        """Test creating an image event."""
        event = Multimodal.image(
            base64="abc123",
            width=1024,
            height=768,
            seed=42,
            model="flux-schnell",
        )

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.IMAGE
        assert event.payload.mime_type == "image/png"
        assert event.payload.base64 == "abc123"
        assert event.payload.metadata["width"] == 1024
        assert event.payload.metadata["height"] == 768
        assert event.payload.metadata["seed"] == 42
        assert event.payload.metadata["model"] == "flux-schnell"

    def test_image_event_with_url(self):
        """Test creating an image event with URL."""
        event = Multimodal.image(url="https://example.com/image.png")

        assert event.type == EventType.DATA
        assert event.payload.url == "https://example.com/image.png"
        assert event.payload.base64 is None

    def test_image_event_custom_mime_type(self):
        """Test image event with custom MIME type."""
        event = Multimodal.image(base64="abc", mime_type="image/webp")
        assert event.payload.mime_type == "image/webp"

    def test_audio_event(self):
        """Test creating an audio event."""
        event = Multimodal.audio(
            base64="audio_data",
            duration=120.5,
            model="whisper",
        )

        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.AUDIO
        assert event.payload.mime_type == "audio/mp3"
        assert event.payload.base64 == "audio_data"
        assert event.payload.metadata["duration"] == 120.5
        assert event.payload.metadata["model"] == "whisper"

    def test_video_event(self):
        """Test creating a video event."""
        event = Multimodal.video(
            url="https://example.com/video.mp4",
            width=1920,
            height=1080,
            duration=60.0,
        )

        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.VIDEO
        assert event.payload.mime_type == "video/mp4"
        assert event.payload.url == "https://example.com/video.mp4"
        assert event.payload.metadata["width"] == 1920
        assert event.payload.metadata["height"] == 1080
        assert event.payload.metadata["duration"] == 60.0

    def test_file_event(self):
        """Test creating a file event."""
        event = Multimodal.file(
            base64="file_content",
            filename="document.pdf",
            size=1024,
            mime_type="application/pdf",
        )

        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.FILE
        assert event.payload.mime_type == "application/pdf"
        assert event.payload.metadata["filename"] == "document.pdf"
        assert event.payload.metadata["size"] == 1024

    def test_json_event(self):
        """Test creating a JSON event."""
        data = {"key": "value", "numbers": [1, 2, 3]}
        event = Multimodal.json(data, model="gpt-4")

        assert event.type == EventType.DATA
        assert event.payload.content_type == ContentType.JSON
        assert event.payload.mime_type == "application/json"
        assert event.payload.json == data
        assert event.payload.metadata["model"] == "gpt-4"

    def test_progress_event(self):
        """Test creating a progress event."""
        event = Multimodal.progress(
            percent=50.0,
            step=5,
            total_steps=10,
            message="Processing...",
            eta=30.0,
        )

        assert event.type == EventType.PROGRESS
        assert event.progress is not None
        assert event.progress.percent == 50.0
        assert event.progress.step == 5
        assert event.progress.total_steps == 10
        assert event.progress.message == "Processing..."
        assert event.progress.eta == 30.0

    def test_complete_event(self):
        """Test creating a complete event."""
        event = Multimodal.complete(usage={"tokens": 100})

        assert event.type == EventType.COMPLETE
        assert event.usage == {"tokens": 100}

    def test_complete_event_no_usage(self):
        """Test complete event without usage."""
        event = Multimodal.complete()
        assert event.type == EventType.COMPLETE
        assert event.usage is None

    def test_error_event(self):
        """Test creating an error event."""
        error = ValueError("Something went wrong")
        event = Multimodal.error(error)

        assert event.type == EventType.ERROR
        assert event.error is error

    def test_data_event_with_payload(self):
        """Test creating a data event with full payload."""
        payload = DataPayload(
            content_type=ContentType.BINARY,
            mime_type="application/octet-stream",
            data=b"binary data",
        )
        event = Multimodal.data(payload)

        assert event.type == EventType.DATA
        assert event.payload is payload

    def test_extra_metadata(self):
        """Test that extra metadata is passed through."""
        event = Multimodal.image(
            base64="abc",
            width=100,
            custom_field="custom_value",
            another_field=42,
        )

        assert event.payload.metadata["custom_field"] == "custom_value"
        assert event.payload.metadata["another_field"] == 42


class TestMultimodalToEvents:
    """Tests for Multimodal.to_events() stream converter."""

    @pytest.mark.asyncio
    async def test_to_events_with_progress(self):
        """Test converting stream with progress extraction."""

        async def source_stream():
            yield {"type": "progress", "percent": 25}
            yield {"type": "progress", "percent": 50}
            yield {"type": "progress", "percent": 100}

        def extract_progress(chunk):
            if chunk["type"] == "progress":
                return {"percent": chunk["percent"]}
            return None

        events = []
        async for event in Multimodal.to_events(
            source_stream(),
            extract_progress=extract_progress,
        ):
            events.append(event)

        assert len(events) == 4  # 3 progress + 1 complete
        assert events[0].type == EventType.PROGRESS
        assert events[0].progress.percent == 25
        assert events[1].progress.percent == 50
        assert events[2].progress.percent == 100
        assert events[3].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_to_events_with_data(self):
        """Test converting stream with data extraction."""

        async def source_stream():
            yield {"type": "image", "base64": "img1", "width": 512}
            yield {"type": "image", "base64": "img2", "width": 1024}

        def extract_data(chunk):
            if chunk["type"] == "image":
                return Multimodal.image(
                    base64=chunk["base64"],
                    width=chunk["width"],
                ).payload
            return None

        events = []
        async for event in Multimodal.to_events(
            source_stream(),
            extract_data=extract_data,
        ):
            events.append(event)

        assert len(events) == 3  # 2 data + 1 complete
        assert events[0].type == EventType.DATA
        assert events[0].payload.base64 == "img1"
        assert events[0].payload.metadata["width"] == 512
        assert events[1].payload.base64 == "img2"
        assert events[2].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_to_events_with_text(self):
        """Test converting stream with text extraction."""

        async def source_stream():
            yield {"type": "text", "content": "Hello "}
            yield {"type": "text", "content": "World"}

        def extract_text(chunk):
            if chunk["type"] == "text":
                return chunk["content"]
            return None

        events = []
        async for event in Multimodal.to_events(
            source_stream(),
            extract_text=extract_text,
        ):
            events.append(event)

        assert len(events) == 3  # 2 tokens + 1 complete
        assert events[0].type == EventType.TOKEN
        assert events[0].text == "Hello "
        assert events[1].text == "World"
        assert events[2].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_to_events_with_error_extraction(self):
        """Test converting stream with error extraction."""

        async def source_stream():
            yield {"type": "progress", "percent": 50}
            yield {"type": "error", "message": "Generation failed"}

        def extract_progress(chunk):
            if chunk["type"] == "progress":
                return {"percent": chunk["percent"]}
            return None

        def extract_error(chunk):
            if chunk["type"] == "error":
                return RuntimeError(chunk["message"])
            return None

        events = []
        async for event in Multimodal.to_events(
            source_stream(),
            extract_progress=extract_progress,
            extract_error=extract_error,
        ):
            events.append(event)

        assert len(events) == 2  # 1 progress + 1 error
        assert events[0].type == EventType.PROGRESS
        assert events[1].type == EventType.ERROR
        assert str(events[1].error) == "Generation failed"

    @pytest.mark.asyncio
    async def test_to_events_handles_stream_exception(self):
        """Test that stream exceptions are converted to error events."""

        async def failing_stream():
            yield {"type": "progress", "percent": 25}
            raise RuntimeError("Stream crashed")

        def extract_progress(chunk):
            if chunk["type"] == "progress":
                return {"percent": chunk["percent"]}
            return None

        events = []
        async for event in Multimodal.to_events(
            failing_stream(),
            extract_progress=extract_progress,
        ):
            events.append(event)

        assert len(events) == 2  # 1 progress + 1 error
        assert events[0].type == EventType.PROGRESS
        assert events[1].type == EventType.ERROR
        assert "Stream crashed" in str(events[1].error)

    @pytest.mark.asyncio
    async def test_to_events_with_progress_object(self):
        """Test that Progress objects are passed through directly."""

        async def source_stream():
            yield {"progress": Progress(percent=75, message="Almost done")}

        def extract_progress(chunk):
            return chunk.get("progress")

        events = []
        async for event in Multimodal.to_events(
            source_stream(),
            extract_progress=extract_progress,
        ):
            events.append(event)

        assert events[0].type == EventType.PROGRESS
        assert events[0].progress.percent == 75
        assert events[0].progress.message == "Almost done"

    @pytest.mark.asyncio
    async def test_to_events_mixed_content(self):
        """Test stream with mixed progress and data."""

        async def flux_stream():
            yield {"type": "queued", "position": 5}
            yield {"type": "progress", "percent": 25}
            yield {"type": "progress", "percent": 75}
            yield {"type": "result", "image": "base64data", "seed": 42}

        def extract_progress(chunk):
            if chunk["type"] == "queued":
                return {"percent": 0, "message": f"Queue position: {chunk['position']}"}
            if chunk["type"] == "progress":
                return {"percent": chunk["percent"]}
            return None

        def extract_data(chunk):
            if chunk["type"] == "result":
                return Multimodal.image(
                    base64=chunk["image"],
                    seed=chunk["seed"],
                ).payload
            return None

        events = []
        async for event in Multimodal.to_events(
            flux_stream(),
            extract_progress=extract_progress,
            extract_data=extract_data,
        ):
            events.append(event)

        assert len(events) == 5  # 3 progress + 1 data + 1 complete
        assert events[0].type == EventType.PROGRESS
        assert events[0].progress.message == "Queue position: 5"
        assert events[1].progress.percent == 25
        assert events[2].progress.percent == 75
        assert events[3].type == EventType.DATA
        assert events[3].payload.metadata["seed"] == 42
        assert events[4].type == EventType.COMPLETE


class TestMultimodalFromStream:
    """Tests for Multimodal.from_stream() convenience method."""

    @pytest.mark.asyncio
    async def test_from_stream_basic(self):
        """Test from_stream as a direct wrapper."""

        async def source():
            yield {"image": "data"}

        def extract_data(chunk):
            if chunk.get("image"):
                return Multimodal.image(base64=chunk["image"]).payload
            return None

        events = []
        async for event in Multimodal.from_stream(
            source(),
            extract_data=extract_data,
        ):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == EventType.DATA
        assert events[1].type == EventType.COMPLETE


class TestDataPayloadProperties:
    """Tests for DataPayload convenience properties."""

    def test_payload_width_height(self):
        """Test width/height properties."""
        payload = DataPayload(
            content_type=ContentType.IMAGE,
            metadata={"width": 1024, "height": 768},
        )
        assert payload.width == 1024
        assert payload.height == 768

    def test_payload_duration(self):
        """Test duration property."""
        payload = DataPayload(
            content_type=ContentType.AUDIO,
            metadata={"duration": 120.5},
        )
        assert payload.duration == 120.5

    def test_payload_size_filename(self):
        """Test size and filename properties."""
        payload = DataPayload(
            content_type=ContentType.FILE,
            metadata={"size": 1024, "filename": "doc.pdf"},
        )
        assert payload.size == 1024
        assert payload.filename == "doc.pdf"

    def test_payload_seed_model(self):
        """Test seed and model properties."""
        payload = DataPayload(
            content_type=ContentType.IMAGE,
            metadata={"seed": 42, "model": "flux"},
        )
        assert payload.seed == 42
        assert payload.model == "flux"

    def test_payload_none_metadata(self):
        """Test properties return None when no metadata."""
        payload = DataPayload(content_type=ContentType.IMAGE)
        assert payload.width is None
        assert payload.height is None
        assert payload.duration is None
