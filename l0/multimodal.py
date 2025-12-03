"""Multimodal adapter helpers for L0.

Helpers for building adapters that handle image, audio, video, and other
non-text AI outputs.

Example:
    ```python
    from l0 import Multimodal, Progress

    # Using the high-level helper
    async def wrap(stream):
        async for event in Multimodal.from_stream(
            stream,
            extract_progress=lambda chunk: Progress(percent=chunk.progress),
            extract_data=lambda chunk: Multimodal.image_payload(
                base64=chunk.image,
                width=chunk.width,
                height=chunk.height,
            ) if chunk.image else None,
        ):
            yield event

    # Or using individual helpers
    async def wrap(stream):
        for chunk in stream:
            if chunk.type == "progress":
                yield Multimodal.progress(percent=chunk.percent)
            elif chunk.type == "image":
                yield Multimodal.image(
                    base64=chunk.image,
                    width=chunk.width,
                    height=chunk.height,
                )
        yield Multimodal.complete()
    ```
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from .types import ContentType, DataPayload, Event, EventType, Progress

T = TypeVar("T")


class Multimodal:
    """Scoped API for multimodal adapter helpers.

    Provides static methods for creating multimodal events and payloads,
    and a stream converter for building custom adapters.

    Usage:
        from l0 import Multimodal, Progress

        # Create events
        event = Multimodal.image(base64="...", width=1024, height=768)
        event = Multimodal.audio(url="https://...")
        event = Multimodal.progress(percent=50, message="Processing...")

        # Create payloads (for use with from_stream)
        payload = Multimodal.image_payload(base64="...")

        # Convert streams
        async for event in Multimodal.from_stream(stream, ...):
            yield event
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Event Creation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def data(payload: DataPayload) -> Event:
        """Create a data event with a full payload."""
        return Event(type=EventType.DATA, payload=payload)

    @staticmethod
    def progress(
        percent: float | None = None,
        step: int | None = None,
        total_steps: int | None = None,
        message: str | None = None,
        eta: float | None = None,
    ) -> Event:
        """Create a progress event.

        Args:
            percent: Progress percentage (0-100)
            step: Current step number
            total_steps: Total number of steps
            message: Status message
            eta: Estimated time remaining in seconds
        """
        return Event(
            type=EventType.PROGRESS,
            progress=Progress(
                percent=percent,
                step=step,
                total_steps=total_steps,
                message=message,
                eta=eta,
            ),
        )

    @staticmethod
    def image(
        base64: str | None = None,
        url: str | None = None,
        data: bytes | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        model: str | None = None,
        mime_type: str = "image/png",
        **extra_metadata: Any,
    ) -> Event:
        """Create an image data event.

        Args:
            base64: Base64-encoded image data
            url: URL to image
            data: Raw image bytes
            width: Image width
            height: Image height
            seed: Generation seed
            model: Model used
            mime_type: MIME type (default: image/png)
            **extra_metadata: Additional metadata
        """
        return Event(
            type=EventType.DATA,
            payload=Multimodal.image_payload(
                base64=base64,
                url=url,
                data=data,
                width=width,
                height=height,
                seed=seed,
                model=model,
                mime_type=mime_type,
                **extra_metadata,
            ),
        )

    @staticmethod
    def audio(
        base64: str | None = None,
        url: str | None = None,
        data: bytes | None = None,
        duration: float | None = None,
        model: str | None = None,
        mime_type: str = "audio/mp3",
        **extra_metadata: Any,
    ) -> Event:
        """Create an audio data event.

        Args:
            base64: Base64-encoded audio data
            url: URL to audio
            data: Raw audio bytes
            duration: Audio duration in seconds
            model: Model used
            mime_type: MIME type (default: audio/mp3)
            **extra_metadata: Additional metadata
        """
        return Event(
            type=EventType.DATA,
            payload=Multimodal.audio_payload(
                base64=base64,
                url=url,
                data=data,
                duration=duration,
                model=model,
                mime_type=mime_type,
                **extra_metadata,
            ),
        )

    @staticmethod
    def video(
        base64: str | None = None,
        url: str | None = None,
        data: bytes | None = None,
        width: int | None = None,
        height: int | None = None,
        duration: float | None = None,
        model: str | None = None,
        mime_type: str = "video/mp4",
        **extra_metadata: Any,
    ) -> Event:
        """Create a video data event.

        Args:
            base64: Base64-encoded video data
            url: URL to video
            data: Raw video bytes
            width: Video width
            height: Video height
            duration: Video duration in seconds
            model: Model used
            mime_type: MIME type (default: video/mp4)
            **extra_metadata: Additional metadata
        """
        metadata = {
            k: v
            for k, v in {
                "width": width,
                "height": height,
                "duration": duration,
                "model": model,
                **extra_metadata,
            }.items()
            if v is not None
        }

        return Event(
            type=EventType.DATA,
            payload=DataPayload(
                content_type=ContentType.VIDEO,
                mime_type=mime_type,
                base64=base64,
                url=url,
                data=data,
                metadata=metadata or None,
            ),
        )

    @staticmethod
    def file(
        base64: str | None = None,
        url: str | None = None,
        data: bytes | None = None,
        filename: str | None = None,
        size: int | None = None,
        mime_type: str | None = None,
        **extra_metadata: Any,
    ) -> Event:
        """Create a file data event.

        Args:
            base64: Base64-encoded file data
            url: URL to file
            data: Raw file bytes
            filename: Filename
            size: File size in bytes
            mime_type: MIME type
            **extra_metadata: Additional metadata
        """
        metadata = {
            k: v
            for k, v in {
                "filename": filename,
                "size": size,
                **extra_metadata,
            }.items()
            if v is not None
        }

        return Event(
            type=EventType.DATA,
            payload=DataPayload(
                content_type=ContentType.FILE,
                mime_type=mime_type,
                base64=base64,
                url=url,
                data=data,
                metadata=metadata or None,
            ),
        )

    @staticmethod
    def json(
        json_data: Any,
        model: str | None = None,
        **extra_metadata: Any,
    ) -> Event:
        """Create a JSON data event.

        Args:
            json_data: The JSON-serializable data
            model: Model used
            **extra_metadata: Additional metadata
        """
        metadata = {
            k: v
            for k, v in {
                "model": model,
                **extra_metadata,
            }.items()
            if v is not None
        }

        return Event(
            type=EventType.DATA,
            payload=DataPayload(
                content_type=ContentType.JSON,
                mime_type="application/json",
                json=json_data,
                metadata=metadata or None,
            ),
        )

    @staticmethod
    def complete(usage: dict[str, int] | None = None) -> Event:
        """Create a completion event."""
        return Event(type=EventType.COMPLETE, usage=usage)

    @staticmethod
    def error(error: Exception) -> Event:
        """Create an error event."""
        return Event(type=EventType.ERROR, error=error)

    # ─────────────────────────────────────────────────────────────────────────
    # Payload Creation (for use with from_stream extract_data)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def image_payload(
        base64: str | None = None,
        url: str | None = None,
        data: bytes | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        model: str | None = None,
        mime_type: str = "image/png",
        **extra_metadata: Any,
    ) -> DataPayload:
        """Create an image payload (for use with extract_data)."""
        metadata = {
            k: v
            for k, v in {
                "width": width,
                "height": height,
                "seed": seed,
                "model": model,
                **extra_metadata,
            }.items()
            if v is not None
        }

        return DataPayload(
            content_type=ContentType.IMAGE,
            mime_type=mime_type,
            base64=base64,
            url=url,
            data=data,
            metadata=metadata or None,
        )

    @staticmethod
    def audio_payload(
        base64: str | None = None,
        url: str | None = None,
        data: bytes | None = None,
        duration: float | None = None,
        model: str | None = None,
        mime_type: str = "audio/mp3",
        **extra_metadata: Any,
    ) -> DataPayload:
        """Create an audio payload (for use with extract_data)."""
        metadata = {
            k: v
            for k, v in {
                "duration": duration,
                "model": model,
                **extra_metadata,
            }.items()
            if v is not None
        }

        return DataPayload(
            content_type=ContentType.AUDIO,
            mime_type=mime_type,
            base64=base64,
            url=url,
            data=data,
            metadata=metadata or None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stream Converter
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def from_stream(
        stream: AsyncIterator[T],
        *,
        extract_progress: Callable[[T], Progress | None] | None = None,
        extract_data: Callable[[T], DataPayload | None] | None = None,
        extract_text: Callable[[T], str | None] | None = None,
        extract_error: Callable[[T], Exception | None] | None = None,
    ) -> AsyncIterator[Event]:
        """Convert a multimodal stream to L0 events.

        High-level helper that extracts progress, data, text, and errors
        from arbitrary stream chunks.

        Args:
            stream: The source async stream
            extract_progress: Function to extract Progress from a chunk
            extract_data: Function to extract DataPayload from a chunk
            extract_text: Function to extract text content from a chunk
            extract_error: Function to extract errors from a chunk

        Yields:
            L0 Event objects

        Example:
            ```python
            async def wrap(stream):
                async for event in Multimodal.from_stream(
                    stream,
                    extract_progress=lambda c: Progress(percent=c.progress) if c.progress else None,
                    extract_data=lambda c: Multimodal.image_payload(base64=c.image) if c.image else None,
                ):
                    yield event
            ```
        """
        try:
            async for chunk in stream:
                # Check for errors first
                if extract_error:
                    error = extract_error(chunk)
                    if error:
                        yield Multimodal.error(error)
                        continue

                # Check for progress
                if extract_progress:
                    progress = extract_progress(chunk)
                    if progress:
                        yield Event(type=EventType.PROGRESS, progress=progress)

                # Check for data
                if extract_data:
                    payload = extract_data(chunk)
                    if payload:
                        yield Event(type=EventType.DATA, payload=payload)

                # Check for text
                if extract_text:
                    text = extract_text(chunk)
                    if text:
                        yield Event(type=EventType.TOKEN, text=text)

            yield Multimodal.complete()

        except Exception as e:
            yield Multimodal.error(e)
