"""Multimodal adapter helpers for L0.

Helpers for building adapters that handle image, audio, video, and other
non-text AI outputs.

Example - Manual event creation:
    ```python
    from l0 import Multimodal

    async def wrap_dalle(stream):
        async for chunk in stream:
            if chunk.type == "progress":
                yield Multimodal.progress(percent=chunk.percent)
            elif chunk.type == "result":
                yield Multimodal.image(
                    base64=chunk.b64_json,
                    width=chunk.width,
                    height=chunk.height,
                    model="dall-e-3",
                )
        yield Multimodal.complete()
    ```

Example - Using to_events() converter:
    ```python
    from l0 import Multimodal

    # Define extractors for your stream format
    def extract_progress(chunk):
        if chunk.type == "progress":
            return {"percent": chunk.percent, "message": chunk.status}
        return None

    def extract_data(chunk):
        if chunk.type == "image":
            return Multimodal.image(
                base64=chunk.image,
                width=chunk.width,
                height=chunk.height,
            ).payload
        return None

    # Use in adapter
    class FluxAdapter:
        name = "flux"

        def detect(self, stream):
            return hasattr(stream, "__flux__")

        def wrap(self, stream):
            return Multimodal.to_events(
                stream,
                extract_progress=extract_progress,
                extract_data=extract_data,
            )
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
        from l0 import Multimodal

        # Create events
        event = Multimodal.image(base64="...", width=1024, height=768)
        event = Multimodal.audio(url="https://...")
        event = Multimodal.progress(percent=50, message="Processing...")
        event = Multimodal.complete()
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

        return Event(
            type=EventType.DATA,
            payload=DataPayload(
                content_type=ContentType.IMAGE,
                mime_type=mime_type,
                base64=base64,
                url=url,
                data=data,
                metadata=metadata or None,
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
        metadata = {
            k: v
            for k, v in {
                "duration": duration,
                "model": model,
                **extra_metadata,
            }.items()
            if v is not None
        }

        return Event(
            type=EventType.DATA,
            payload=DataPayload(
                content_type=ContentType.AUDIO,
                mime_type=mime_type,
                base64=base64,
                url=url,
                data=data,
                metadata=metadata or None,
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
    def binary(
        base64: str | None = None,
        url: str | None = None,
        data: bytes | None = None,
        mime_type: str = "application/octet-stream",
        size: int | None = None,
        **extra_metadata: Any,
    ) -> Event:
        """Create a binary data event.

        Args:
            base64: Base64-encoded binary data
            url: URL to binary data
            data: Raw bytes
            mime_type: MIME type (default: application/octet-stream)
            size: Size in bytes
            **extra_metadata: Additional metadata
        """
        metadata = {
            k: v
            for k, v in {
                "size": size,
                **extra_metadata,
            }.items()
            if v is not None
        }

        return Event(
            type=EventType.DATA,
            payload=DataPayload(
                content_type=ContentType.BINARY,
                mime_type=mime_type,
                base64=base64,
                url=url,
                data=data,
                metadata=metadata or None,
            ),
        )

    @staticmethod
    def token(text: str, timestamp: float | None = None) -> Event:
        """Create a token event.

        Args:
            text: The text token content
            timestamp: Optional timestamp for the event
        """
        return Event(type=EventType.TOKEN, text=text, timestamp=timestamp)

    @staticmethod
    def message(
        value: str,
        role: str | None = None,
        timestamp: float | None = None,
    ) -> Event:
        """Create a message event.

        Args:
            value: The message content
            role: Optional role (e.g., "assistant", "user")
            timestamp: Optional timestamp for the event
        """
        return Event(
            type=EventType.MESSAGE,
            data={"value": value, "role": role},
            timestamp=timestamp,
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
    # Stream Conversion
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def to_events(
        stream: AsyncIterator[T],
        *,
        extract_text: Callable[[T], str | None] | None = None,
        extract_data: Callable[[T], DataPayload | None] | None = None,
        extract_progress: Callable[[T], dict[str, Any] | Progress | None] | None = None,
        extract_message: Callable[[T], dict[str, Any] | None] | None = None,
        extract_error: Callable[[T], Exception | None] | None = None,
    ) -> AsyncIterator[Event]:
        """Convert a multimodal stream to L0 events using extractors.

        This is the easiest way to build a multimodal adapter. Provide
        extractor functions that know how to pull text, data, progress,
        messages, or errors from your stream's chunk format.

        Handlers are tried in order: text → data → progress → message.
        The first handler that returns a non-null value creates the event,
        then processing continues to the next chunk.

        Args:
            stream: The source async iterator
            extract_text: Extract text token from a chunk.
                Return the text string.
                Return None if chunk has no text.
            extract_data: Extract data payload from a chunk.
                Return a DataPayload object.
                Return None if chunk has no data.
            extract_progress: Extract progress info from a chunk.
                Return dict with keys: percent, step, total_steps, message, eta
                Or return a Progress object directly.
                Return None if chunk has no progress.
            extract_message: Extract message from a chunk.
                Return dict with keys: value (required), role (optional).
                Return None if chunk has no message.
            extract_error: Extract error from a chunk.
                Return an Exception.
                Return None if chunk has no error.

        Yields:
            L0 Event objects

        Example:
            ```python
            async def wrap(stream):
                async for event in Multimodal.to_events(
                    stream,
                    extract_progress=lambda c: {"percent": c.progress} if c.type == "progress" else None,
                    extract_data=lambda c: Multimodal.image(base64=c.image).payload if c.type == "image" else None,
                ):
                    yield event
            ```
        """
        try:
            async for chunk in stream:
                # Try to extract text (first priority)
                if extract_text:
                    text = extract_text(chunk)
                    if text is not None:
                        yield Event(type=EventType.TOKEN, text=text)
                        continue

                # Try to extract data
                if extract_data:
                    data_payload = extract_data(chunk)
                    if data_payload is not None:
                        yield Event(type=EventType.DATA, payload=data_payload)
                        continue

                # Try to extract progress
                if extract_progress:
                    progress_info = extract_progress(chunk)
                    if progress_info is not None:
                        if isinstance(progress_info, Progress):
                            yield Event(type=EventType.PROGRESS, progress=progress_info)
                        else:
                            yield Event(
                                type=EventType.PROGRESS,
                                progress=Progress(
                                    percent=progress_info.get("percent"),
                                    step=progress_info.get("step"),
                                    total_steps=progress_info.get("total_steps"),
                                    message=progress_info.get("message"),
                                    eta=progress_info.get("eta"),
                                ),
                            )
                        continue

                # Try to extract message
                if extract_message:
                    message_info = extract_message(chunk)
                    if message_info is not None:
                        yield Event(
                            type=EventType.MESSAGE,
                            data={
                                "value": message_info.get("value"),
                                "role": message_info.get("role"),
                            },
                        )
                        continue

                # Try to extract error (always checked, stops stream)
                if extract_error:
                    error = extract_error(chunk)
                    if error is not None:
                        yield Event(type=EventType.ERROR, error=error)
                        return

            # Stream completed successfully
            yield Event(type=EventType.COMPLETE)

        except Exception as e:
            yield Event(type=EventType.ERROR, error=e)

    @staticmethod
    def from_stream(
        stream: AsyncIterator[T],
        *,
        extract_text: Callable[[T], str | None] | None = None,
        extract_data: Callable[[T], DataPayload | None] | None = None,
        extract_progress: Callable[[T], dict[str, Any] | Progress | None] | None = None,
        extract_message: Callable[[T], dict[str, Any] | None] | None = None,
        extract_error: Callable[[T], Exception | None] | None = None,
    ) -> AsyncIterator[Event]:
        """Convenience wrapper for to_events().

        Same as to_events() but can be used directly without async iteration.

        Args:
            stream: The source async iterator
            extract_text: Extract text token from a chunk
            extract_data: Extract data payload from a chunk
            extract_progress: Extract progress info from a chunk
            extract_message: Extract message from a chunk
            extract_error: Extract error from a chunk

        Returns:
            Async iterator of L0 Events

        Example:
            ```python
            # In an adapter
            def wrap(self, stream):
                return Multimodal.from_stream(
                    stream,
                    extract_data=lambda c: Multimodal.image(url=c.url).payload if c.url else None,
                )
            ```
        """
        return Multimodal.to_events(
            stream,
            extract_text=extract_text,
            extract_data=extract_data,
            extract_progress=extract_progress,
            extract_message=extract_message,
            extract_error=extract_error,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level Helper Functions (TypeScript API Parity)
# ─────────────────────────────────────────────────────────────────────────────


def create_image_event(
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

    Module-level function for TypeScript API parity.
    Same as Multimodal.image().

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
    return Multimodal.image(
        base64=base64,
        url=url,
        data=data,
        width=width,
        height=height,
        seed=seed,
        model=model,
        mime_type=mime_type,
        **extra_metadata,
    )


def create_audio_event(
    base64: str | None = None,
    url: str | None = None,
    data: bytes | None = None,
    duration: float | None = None,
    model: str | None = None,
    mime_type: str = "audio/mp3",
    **extra_metadata: Any,
) -> Event:
    """Create an audio data event.

    Module-level function for TypeScript API parity.
    Same as Multimodal.audio().

    Args:
        base64: Base64-encoded audio data
        url: URL to audio
        data: Raw audio bytes
        duration: Audio duration in seconds
        model: Model used
        mime_type: MIME type (default: audio/mp3)
        **extra_metadata: Additional metadata
    """
    return Multimodal.audio(
        base64=base64,
        url=url,
        data=data,
        duration=duration,
        model=model,
        mime_type=mime_type,
        **extra_metadata,
    )


def create_video_event(
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

    Module-level function for TypeScript API parity.
    Same as Multimodal.video().

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
    return Multimodal.video(
        base64=base64,
        url=url,
        data=data,
        width=width,
        height=height,
        duration=duration,
        model=model,
        mime_type=mime_type,
        **extra_metadata,
    )


def create_data_event(payload: DataPayload) -> Event:
    """Create a data event with a full payload.

    Module-level function for TypeScript API parity.
    Same as Multimodal.data().
    """
    return Multimodal.data(payload)


def create_progress_event(
    percent: float | None = None,
    step: int | None = None,
    total_steps: int | None = None,
    message: str | None = None,
    eta: float | None = None,
) -> Event:
    """Create a progress event.

    Module-level function for TypeScript API parity.
    Same as Multimodal.progress().

    Args:
        percent: Progress percentage (0-100)
        step: Current step number
        total_steps: Total number of steps
        message: Status message
        eta: Estimated time remaining in seconds
    """
    return Multimodal.progress(
        percent=percent,
        step=step,
        total_steps=total_steps,
        message=message,
        eta=eta,
    )


def create_token_event(text: str, timestamp: float | None = None) -> Event:
    """Create a token event.

    Module-level function for TypeScript API parity.
    Same as Multimodal.token().

    Args:
        text: The text token content
        timestamp: Optional timestamp for the event
    """
    return Multimodal.token(text, timestamp)


def create_message_event(
    value: str,
    role: str | None = None,
    timestamp: float | None = None,
) -> Event:
    """Create a message event.

    Module-level function for TypeScript API parity.
    Same as Multimodal.message().

    Args:
        value: The message content
        role: Optional role (e.g., "assistant", "user")
        timestamp: Optional timestamp for the event
    """
    return Multimodal.message(value, role, timestamp)


def create_complete_event(usage: dict[str, int] | None = None) -> Event:
    """Create a completion event.

    Module-level function for TypeScript API parity.
    Same as Multimodal.complete().
    """
    return Multimodal.complete(usage)


def create_error_event(error: Exception) -> Event:
    """Create an error event.

    Module-level function for TypeScript API parity.
    Same as Multimodal.error().
    """
    return Multimodal.error(error)


def create_json_event(
    json_data: Any,
    model: str | None = None,
    **extra_metadata: Any,
) -> Event:
    """Create a JSON data event.

    Module-level function for TypeScript API parity.
    Same as Multimodal.json().

    Args:
        json_data: The JSON-serializable data
        model: Model used
        **extra_metadata: Additional metadata
    """
    return Multimodal.json(json_data, model, **extra_metadata)


async def to_multimodal_events(
    stream: AsyncIterator[T],
    *,
    extract_text: Callable[[T], str | None] | None = None,
    extract_data: Callable[[T], DataPayload | None] | None = None,
    extract_progress: Callable[[T], dict[str, Any] | Progress | None] | None = None,
    extract_message: Callable[[T], dict[str, Any] | None] | None = None,
    extract_error: Callable[[T], Exception | None] | None = None,
) -> AsyncIterator[Event]:
    """Convert a multimodal stream to L0 events using extractors.

    Module-level function for TypeScript API parity.
    Same as Multimodal.to_events().

    Args:
        stream: The source async iterator
        extract_text: Extract text token from a chunk
        extract_data: Extract data payload from a chunk
        extract_progress: Extract progress info from a chunk
        extract_message: Extract message from a chunk
        extract_error: Extract error from a chunk

    Yields:
        L0 Event objects
    """
    async for event in Multimodal.to_events(
        stream,
        extract_text=extract_text,
        extract_data=extract_data,
        extract_progress=extract_progress,
        extract_message=extract_message,
        extract_error=extract_error,
    ):
        yield event


async def to_events(
    stream: AsyncIterator[T],
    extract_text: Callable[[T], str | None],
) -> AsyncIterator[Event]:
    """Convert a simple text stream to L0 events.

    Module-level function for TypeScript API parity (toL0Events).

    Args:
        stream: The source async iterator
        extract_text: Function to extract text from each chunk

    Yields:
        L0 Event objects (token events + complete)
    """
    async for event in Multimodal.to_events(
        stream,
        extract_text=extract_text,
    ):
        yield event


async def to_events_with_messages(
    stream: AsyncIterator[T],
    *,
    extract_text: Callable[[T], str | None] | None = None,
    extract_message: Callable[[T], dict[str, Any] | None] | None = None,
) -> AsyncIterator[Event]:
    """Convert a stream with text and messages to L0 events.

    Module-level function for TypeScript API parity (toL0EventsWithMessages).

    Args:
        stream: The source async iterator
        extract_text: Function to extract text from each chunk
        extract_message: Function to extract message from each chunk
            Return dict with keys: value (required), role (optional)

    Yields:
        L0 Event objects (token events, message events, complete)
    """
    async for event in Multimodal.to_events(
        stream,
        extract_text=extract_text,
        extract_message=extract_message,
    ):
        yield event
