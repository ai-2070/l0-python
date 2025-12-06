"""Tests for L0 multimodal helpers."""

from typing import Any

import pytest

from l0 import ContentType, DataPayload, EventType, Multimodal, Progress


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
        payload = event.payload
        assert payload.metadata is not None
        assert payload.metadata["width"] == 1024
        assert payload.metadata["height"] == 768
        assert payload.metadata["seed"] == 42
        assert payload.metadata["model"] == "flux-schnell"

    def test_image_event_with_url(self):
        """Test creating an image event with URL."""
        event = Multimodal.image(url="https://example.com/image.png")

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.url == "https://example.com/image.png"
        assert event.payload.base64 is None

    def test_image_event_custom_mime_type(self):
        """Test image event with custom MIME type."""
        event = Multimodal.image(base64="abc", mime_type="image/webp")
        assert event.payload is not None
        assert event.payload.mime_type == "image/webp"

    def test_audio_event(self):
        """Test creating an audio event."""
        event = Multimodal.audio(
            base64="audio_data",
            duration=120.5,
            model="whisper",
        )

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.AUDIO
        assert event.payload.mime_type == "audio/mp3"
        assert event.payload.base64 == "audio_data"
        payload = event.payload
        assert payload.metadata is not None
        assert payload.metadata["duration"] == 120.5
        assert payload.metadata["model"] == "whisper"

    def test_video_event(self):
        """Test creating a video event."""
        event = Multimodal.video(
            url="https://example.com/video.mp4",
            width=1920,
            height=1080,
            duration=60.0,
        )

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.VIDEO
        assert event.payload.mime_type == "video/mp4"
        assert event.payload.url == "https://example.com/video.mp4"
        payload = event.payload
        assert payload.metadata is not None
        assert payload.metadata["width"] == 1920
        assert payload.metadata["height"] == 1080
        assert payload.metadata["duration"] == 60.0

    def test_file_event(self):
        """Test creating a file event."""
        event = Multimodal.file(
            base64="file_content",
            filename="document.pdf",
            size=1024,
            mime_type="application/pdf",
        )

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.FILE
        assert event.payload.mime_type == "application/pdf"
        payload = event.payload
        assert payload.metadata is not None
        assert payload.metadata["filename"] == "document.pdf"
        assert payload.metadata["size"] == 1024

    def test_json_event(self):
        """Test creating a JSON event."""
        data = {"key": "value", "numbers": [1, 2, 3]}
        event = Multimodal.json(data, model="gpt-4")

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.JSON
        assert event.payload.mime_type == "application/json"
        assert event.payload.json == data
        assert event.payload.metadata is not None
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
        progress = event.progress
        assert progress.percent == 50.0
        assert progress.step == 5
        assert progress.total_steps == 10
        assert progress.message == "Processing..."
        assert progress.eta == 30.0

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

        assert event.payload is not None
        payload = event.payload
        assert payload.metadata is not None
        assert payload.metadata["custom_field"] == "custom_value"
        assert payload.metadata["another_field"] == 42


class TestMultimodalToEvents:
    """Tests for Multimodal.to_events() stream converter."""

    @pytest.mark.asyncio
    async def test_to_events_with_progress(self):
        """Test converting stream with progress extraction."""

        async def source_stream():
            yield {"type": "progress", "percent": 25}
            yield {"type": "progress", "percent": 50}
            yield {"type": "progress", "percent": 100}

        def extract_progress(chunk: dict[str, Any]) -> dict[str, Any] | None:
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
        assert events[0].progress is not None
        assert events[0].progress.percent == 25
        assert events[1].progress is not None
        assert events[1].progress.percent == 50
        assert events[2].progress is not None
        assert events[2].progress.percent == 100
        assert events[3].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_to_events_with_data(self):
        """Test converting stream with data extraction."""

        async def source_stream():
            yield {"type": "image", "base64": "img1", "width": 512}
            yield {"type": "image", "base64": "img2", "width": 1024}

        def extract_data(chunk: dict[str, Any]) -> DataPayload | None:
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
        assert events[0].payload is not None
        assert events[0].payload.base64 == "img1"
        assert events[0].payload.metadata["width"] == 512
        assert events[1].payload is not None
        assert events[1].payload.base64 == "img2"
        assert events[2].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_to_events_with_text(self):
        """Test converting stream with text extraction."""

        async def source_stream():
            yield {"type": "text", "content": "Hello "}
            yield {"type": "text", "content": "World"}

        def extract_text(chunk: dict[str, Any]) -> str | None:
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

        def extract_progress(chunk: dict[str, Any]) -> dict[str, Any] | None:
            if chunk["type"] == "progress":
                return {"percent": chunk["percent"]}
            return None

        def extract_error(chunk: dict[str, Any]) -> RuntimeError | None:
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

        def extract_progress(chunk: dict[str, Any]) -> dict[str, Any] | None:
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
        assert events[1].error is not None
        assert "Stream crashed" in str(events[1].error)

    @pytest.mark.asyncio
    async def test_to_events_with_progress_object(self):
        """Test that Progress objects are passed through directly."""

        async def source_stream():
            yield {"progress": Progress(percent=75, message="Almost done")}

        def extract_progress(chunk: dict[str, Any]) -> Progress | None:
            return chunk.get("progress")

        events = []
        async for event in Multimodal.to_events(
            source_stream(),
            extract_progress=extract_progress,
        ):
            events.append(event)

        assert events[0].type == EventType.PROGRESS
        assert events[0].progress is not None
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

        def extract_progress(chunk: dict[str, Any]) -> dict[str, Any] | None:
            if chunk["type"] == "queued":
                return {"percent": 0, "message": f"Queue position: {chunk['position']}"}
            if chunk["type"] == "progress":
                return {"percent": chunk["percent"]}
            return None

        def extract_data(chunk: dict[str, Any]) -> DataPayload | None:
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
        assert events[0].progress is not None
        assert events[0].progress.message == "Queue position: 5"
        assert events[1].progress is not None
        assert events[1].progress.percent == 25
        assert events[2].progress is not None
        assert events[2].progress.percent == 75
        assert events[3].type == EventType.DATA
        assert events[3].payload is not None
        assert events[3].payload.metadata["seed"] == 42
        assert events[4].type == EventType.COMPLETE


class TestMultimodalFromStream:
    """Tests for Multimodal.from_stream() convenience method."""

    @pytest.mark.asyncio
    async def test_from_stream_basic(self):
        """Test from_stream as a direct wrapper."""

        async def source():
            yield {"image": "data"}

        def extract_data(chunk: dict[str, Any]) -> DataPayload | None:
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


# ─────────────────────────────────────────────────────────────────────────────
# New Event Creation Methods
# ─────────────────────────────────────────────────────────────────────────────


class TestMultimodalBinaryEvent:
    """Tests for Multimodal.binary() method."""

    def test_binary_event(self):
        """Test creating a binary data event."""
        event = Multimodal.binary(
            base64="YmluYXJ5IGRhdGE=",
            size=12,
        )

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.BINARY
        assert event.payload.mime_type == "application/octet-stream"
        assert event.payload.base64 == "YmluYXJ5IGRhdGE="
        payload = event.payload
        assert payload.metadata is not None
        assert payload.metadata["size"] == 12

    def test_binary_event_with_url(self):
        """Test binary event with URL."""
        event = Multimodal.binary(url="https://example.com/file.bin")
        assert event.payload is not None
        assert event.payload.url == "https://example.com/file.bin"

    def test_binary_event_custom_mime_type(self):
        """Test binary event with custom MIME type."""
        event = Multimodal.binary(
            base64="abc",
            mime_type="application/x-custom",
        )
        assert event.payload is not None
        assert event.payload.mime_type == "application/x-custom"


class TestMultimodalTokenEvent:
    """Tests for Multimodal.token() method."""

    def test_token_event(self):
        """Test creating a token event."""
        event = Multimodal.token("Hello")

        assert event.type == EventType.TOKEN
        assert event.text == "Hello"

    def test_token_event_with_timestamp(self):
        """Test token event with timestamp."""
        event = Multimodal.token("World", timestamp=1234567890.123)

        assert event.type == EventType.TOKEN
        assert event.text == "World"
        assert event.timestamp == 1234567890.123


class TestMultimodalMessageEvent:
    """Tests for Multimodal.message() method."""

    def test_message_event(self):
        """Test creating a message event."""
        event = Multimodal.message("Hello, how can I help?")

        assert event.type == EventType.MESSAGE
        assert event.data is not None
        assert event.data["value"] == "Hello, how can I help?"
        assert event.data["role"] is None

    def test_message_event_with_role(self):
        """Test message event with role."""
        event = Multimodal.message("I need help", role="user")

        assert event.type == EventType.MESSAGE
        assert event.data is not None
        assert event.data["value"] == "I need help"
        assert event.data["role"] == "user"

    def test_message_event_with_timestamp(self):
        """Test message event with timestamp."""
        event = Multimodal.message(
            "Response",
            role="assistant",
            timestamp=1234567890.123,
        )

        assert event.type == EventType.MESSAGE
        assert event.timestamp == 1234567890.123


# ─────────────────────────────────────────────────────────────────────────────
# Module-level Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


class TestModuleLevelHelpers:
    """Tests for module-level helper functions."""

    def test_create_image_event(self):
        """Test create_image_event function."""
        from l0.multimodal import create_image_event

        event = create_image_event(
            base64="abc123",
            width=512,
            height=512,
            model="dall-e",
        )

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.IMAGE
        assert event.payload.metadata is not None
        assert event.payload.metadata["width"] == 512

    def test_create_audio_event(self):
        """Test create_audio_event function."""
        from l0.multimodal import create_audio_event

        event = create_audio_event(
            base64="audio_data",
            duration=5.0,
            model="tts-1",
        )

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.AUDIO
        assert event.payload.metadata is not None
        assert event.payload.metadata["duration"] == 5.0

    def test_create_video_event(self):
        """Test create_video_event function."""
        from l0.multimodal import create_video_event

        event = create_video_event(
            url="https://example.com/video.mp4",
            duration=60.0,
            width=1920,
            height=1080,
        )

        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.VIDEO
        assert event.payload.metadata is not None
        assert event.payload.metadata["duration"] == 60.0

    def test_create_progress_event(self):
        """Test create_progress_event function."""
        from l0.multimodal import create_progress_event

        event = create_progress_event(
            percent=75,
            step=3,
            total_steps=4,
            message="Processing",
        )

        assert event.type == EventType.PROGRESS
        assert event.progress is not None
        assert event.progress.percent == 75
        assert event.progress.step == 3

    def test_create_token_event(self):
        """Test create_token_event function."""
        from l0.multimodal import create_token_event

        event = create_token_event("Hello ")
        assert event.type == EventType.TOKEN
        assert event.text == "Hello "

    def test_create_message_event(self):
        """Test create_message_event function."""
        from l0.multimodal import create_message_event

        event = create_message_event("tool response", role="assistant")
        assert event.type == EventType.MESSAGE
        assert event.data is not None
        assert event.data["value"] == "tool response"
        assert event.data["role"] == "assistant"

    def test_create_complete_event(self):
        """Test create_complete_event function."""
        from l0.multimodal import create_complete_event

        event = create_complete_event(usage={"tokens": 100})
        assert event.type == EventType.COMPLETE
        assert event.usage == {"tokens": 100}

    def test_create_error_event(self):
        """Test create_error_event function."""
        from l0.multimodal import create_error_event

        error = ValueError("Something failed")
        event = create_error_event(error)
        assert event.type == EventType.ERROR
        assert event.error is error

    def test_create_json_event(self):
        """Test create_json_event function."""
        from l0.multimodal import create_json_event

        data = {"key": "value"}
        event = create_json_event(data, model="gpt-4")
        assert event.type == EventType.DATA
        assert event.payload is not None
        assert event.payload.content_type == ContentType.JSON
        assert event.payload.json == data

    def test_create_data_event(self):
        """Test create_data_event function."""
        from l0.multimodal import create_data_event

        payload = DataPayload(
            content_type=ContentType.BINARY,
            data=b"raw bytes",
        )
        event = create_data_event(payload)
        assert event.type == EventType.DATA
        assert event.payload is payload


# ─────────────────────────────────────────────────────────────────────────────
# Stream Conversion with Messages
# ─────────────────────────────────────────────────────────────────────────────


class TestToEventsWithMessages:
    """Tests for to_events with extract_message handler."""

    @pytest.mark.asyncio
    async def test_to_events_with_message(self):
        """Test extracting messages from stream."""

        async def source_stream():
            yield {"type": "message", "content": "tool call", "role": "assistant"}
            yield {"type": "text", "content": "response"}

        def extract_text(chunk: dict[str, Any]) -> str | None:
            if chunk["type"] == "text":
                return chunk["content"]
            return None

        def extract_message(chunk: dict[str, Any]) -> dict[str, Any] | None:
            if chunk["type"] == "message":
                return {"value": chunk["content"], "role": chunk["role"]}
            return None

        events = []
        async for event in Multimodal.to_events(
            source_stream(),
            extract_text=extract_text,
            extract_message=extract_message,
        ):
            events.append(event)

        assert len(events) == 3  # 1 message + 1 token + 1 complete
        assert events[0].type == EventType.MESSAGE
        assert events[0].data is not None
        assert events[0].data["value"] == "tool call"
        assert events[0].data["role"] == "assistant"
        assert events[1].type == EventType.TOKEN
        assert events[1].text == "response"
        assert events[2].type == EventType.COMPLETE

    @pytest.mark.asyncio
    async def test_to_events_handler_priority(self):
        """Test that handlers are tried in order: text → data → progress → message."""

        async def source_stream():
            # Chunk that matches multiple handlers - text should win
            yield {"type": "both", "text": "hello", "message": "msg"}

        def extract_text(chunk: dict[str, Any]) -> str | None:
            if "text" in chunk:
                return chunk["text"]
            return None

        def extract_message(chunk: dict[str, Any]) -> dict[str, Any] | None:
            if "message" in chunk:
                return {"value": chunk["message"]}
            return None

        events = []
        async for event in Multimodal.to_events(
            source_stream(),
            extract_text=extract_text,
            extract_message=extract_message,
        ):
            events.append(event)

        assert len(events) == 2  # 1 token (text wins) + 1 complete
        assert events[0].type == EventType.TOKEN
        assert events[0].text == "hello"

    @pytest.mark.asyncio
    async def test_to_events_with_messages_function(self):
        """Test the to_events_with_messages helper function."""
        from l0.multimodal import to_events_with_messages

        async def source_stream():
            yield {"type": "text", "content": "Hello "}
            yield {"type": "tool", "call": "search()"}
            yield {"type": "text", "content": "world"}

        def extract_text(chunk: dict[str, Any]) -> str | None:
            if chunk["type"] == "text":
                return chunk["content"]
            return None

        def extract_message(chunk: dict[str, Any]) -> dict[str, Any] | None:
            if chunk["type"] == "tool":
                return {"value": chunk["call"], "role": "assistant"}
            return None

        events = []
        async for event in to_events_with_messages(
            source_stream(),
            extract_text=extract_text,
            extract_message=extract_message,
        ):
            events.append(event)

        assert len(events) == 4  # 2 tokens + 1 message + 1 complete
        assert events[0].type == EventType.TOKEN
        assert events[1].type == EventType.MESSAGE
        assert events[2].type == EventType.TOKEN


class TestToMultimodalEvents:
    """Tests for to_multimodal_events helper function."""

    @pytest.mark.asyncio
    async def test_to_multimodal_events(self):
        """Test the to_multimodal_events helper function."""
        from l0.multimodal import to_multimodal_events

        async def flux_stream():
            yield {"type": "progress", "percent": 50}
            yield {"type": "image", "base64": "abc123", "width": 512}

        def extract_progress(chunk: dict[str, Any]) -> dict[str, Any] | None:
            if chunk["type"] == "progress":
                return {"percent": chunk["percent"]}
            return None

        def extract_data(chunk: dict[str, Any]) -> DataPayload | None:
            if chunk["type"] == "image":
                return Multimodal.image(
                    base64=chunk["base64"],
                    width=chunk["width"],
                ).payload
            return None

        events = []
        async for event in to_multimodal_events(
            flux_stream(),
            extract_progress=extract_progress,
            extract_data=extract_data,
        ):
            events.append(event)

        assert len(events) == 3  # 1 progress + 1 data + 1 complete
        assert events[0].type == EventType.PROGRESS
        assert events[0].progress is not None
        assert events[0].progress.percent == 50
        assert events[1].type == EventType.DATA
        assert events[1].payload is not None
        assert events[1].payload.base64 == "abc123"


class TestToEvents:
    """Tests for to_events helper function."""

    @pytest.mark.asyncio
    async def test_to_events_simple(self):
        """Test the to_events helper for simple text streams."""
        from l0.multimodal import to_events

        async def text_stream():
            yield {"text": "Hello "}
            yield {"text": "World"}

        events = []
        async for event in to_events(
            text_stream(),
            extract_text=lambda c: c.get("text"),
        ):
            events.append(event)

        assert len(events) == 3  # 2 tokens + 1 complete
        assert events[0].type == EventType.TOKEN
        assert events[0].text == "Hello "
        assert events[1].text == "World"
        assert events[2].type == EventType.COMPLETE
