"""Event normalization utilities.

Functions for normalizing stream events from various providers
into unified L0 event format.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class L0EventType(str, Enum):
    """L0 unified event types."""

    TOKEN = "token"
    MESSAGE = "message"
    DATA = "data"
    PROGRESS = "progress"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class L0Event:
    """Unified L0 event format.

    All stream events are normalized to this format.
    """

    type: L0EventType
    timestamp: float
    value: str | None = None
    role: str | None = None
    error: Exception | None = None

    @property
    def is_token(self) -> bool:
        """Check if this is a token event."""
        return self.type == L0EventType.TOKEN

    @property
    def is_complete(self) -> bool:
        """Check if this is a complete event."""
        return self.type == L0EventType.COMPLETE

    @property
    def is_error(self) -> bool:
        """Check if this is an error event."""
        return self.type == L0EventType.ERROR


def _is_l0_event(obj: Any) -> bool:
    """Check if object is already an L0 event."""
    if not isinstance(obj, dict):
        return False

    event_type = obj.get("type")
    return event_type in (
        "token",
        "message",
        "data",
        "progress",
        "error",
        "complete",
        L0EventType.TOKEN,
        L0EventType.MESSAGE,
        L0EventType.DATA,
        L0EventType.PROGRESS,
        L0EventType.ERROR,
        L0EventType.COMPLETE,
    )


def _extract_text_from_chunk(chunk: Any) -> str | None:
    """Try to extract text from various chunk formats."""
    if not isinstance(chunk, dict):
        return None

    # Try common field names
    text_fields = ["text", "content", "delta", "textDelta", "token", "message", "data"]

    for field in text_fields:
        value = chunk.get(field)
        if value and isinstance(value, str):
            text: str = value
            return text

    # Try nested delta/content
    delta = chunk.get("delta")
    if isinstance(delta, dict):
        for field in text_fields:
            value = delta.get(field)
            if value and isinstance(value, str):
                text = value
                return text

    return None


def normalize_stream_event(chunk: Any) -> L0Event:
    """Normalize a stream event from various providers into unified L0 event format.

    Supports:
    - Vercel AI SDK format (text-delta, finish, etc.)
    - OpenAI streaming format (choices[0].delta.content)
    - Anthropic streaming format (delta.text)
    - Simple string chunks
    - Already-normalized L0 events

    Args:
        chunk: Raw stream chunk from provider

    Returns:
        Normalized L0Event
    """
    timestamp = time.time() * 1000

    # Handle None/empty
    if chunk is None:
        return L0Event(
            type=L0EventType.ERROR,
            timestamp=timestamp,
            error=ValueError("Received None chunk"),
        )

    # If already L0Event dataclass
    if isinstance(chunk, L0Event):
        return chunk

    # If already in L0 dict format
    if isinstance(chunk, dict) and _is_l0_event(chunk):
        event_type = chunk.get("type")
        if isinstance(event_type, str):
            try:
                event_type = L0EventType(event_type)
            except ValueError:
                pass

        return L0Event(
            type=event_type
            if isinstance(event_type, L0EventType)
            else L0EventType.TOKEN,
            timestamp=chunk.get("timestamp", timestamp),
            value=chunk.get("value"),
            role=chunk.get("role"),
            error=chunk.get("error"),
        )

    # Handle dict with type field (Vercel AI SDK format)
    if isinstance(chunk, dict) and "type" in chunk:
        chunk_type = chunk["type"]

        if chunk_type in ("text-delta", "content-delta"):
            text = (
                chunk.get("textDelta")
                or chunk.get("delta")
                or chunk.get("content")
                or ""
            )
            return L0Event(
                type=L0EventType.TOKEN,
                timestamp=timestamp,
                value=text,
            )

        if chunk_type in ("finish", "complete", "message_stop", "content_block_stop"):
            return L0Event(
                type=L0EventType.COMPLETE,
                timestamp=timestamp,
            )

        if chunk_type == "error":
            error = chunk.get("error") or Exception(
                chunk.get("message", "Stream error")
            )
            if not isinstance(error, Exception):
                error = Exception(str(error))
            return L0Event(
                type=L0EventType.ERROR,
                timestamp=timestamp,
                error=error,
            )

        if chunk_type in ("tool-call", "function-call"):
            # Convert tool call to message event
            import json

            return L0Event(
                type=L0EventType.MESSAGE,
                timestamp=timestamp,
                value=json.dumps(chunk),
                role="assistant",
            )

        # Unknown type, try to extract text
        text = _extract_text_from_chunk(chunk)
        if text:
            return L0Event(
                type=L0EventType.TOKEN,
                timestamp=timestamp,
                value=text,
            )

        return L0Event(
            type=L0EventType.ERROR,
            timestamp=timestamp,
            error=ValueError(f"Unknown chunk type: {chunk_type}"),
        )

    # Handle OpenAI streaming format
    if isinstance(chunk, dict) and "choices" in chunk:
        choices = chunk.get("choices", [])
        if choices and isinstance(choices, list):
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta", {})
                if isinstance(delta, dict) and delta.get("content"):
                    return L0Event(
                        type=L0EventType.TOKEN,
                        timestamp=timestamp,
                        value=delta["content"],
                    )
                if choice.get("finish_reason"):
                    return L0Event(
                        type=L0EventType.COMPLETE,
                        timestamp=timestamp,
                    )

    # Handle Anthropic streaming format
    if isinstance(chunk, dict):
        delta = chunk.get("delta")
        if isinstance(delta, dict) and delta.get("text"):
            return L0Event(
                type=L0EventType.TOKEN,
                timestamp=timestamp,
                value=delta["text"],
            )

        chunk_type = chunk.get("type")
        if chunk_type in ("message_stop", "content_block_stop"):
            return L0Event(
                type=L0EventType.COMPLETE,
                timestamp=timestamp,
            )

    # Handle simple string chunks
    if isinstance(chunk, str):
        if chunk:  # Non-empty string
            return L0Event(
                type=L0EventType.TOKEN,
                timestamp=timestamp,
                value=chunk,
            )
        else:
            return L0Event(
                type=L0EventType.ERROR,
                timestamp=timestamp,
                error=ValueError("Received empty string chunk"),
            )

    # Try to extract any text content
    if isinstance(chunk, dict):
        text = _extract_text_from_chunk(chunk)
        if text:
            return L0Event(
                type=L0EventType.TOKEN,
                timestamp=timestamp,
                value=text,
            )

    # Unknown format
    return L0Event(
        type=L0EventType.ERROR,
        timestamp=timestamp,
        error=ValueError(f"Unable to normalize chunk: {chunk!r}"),
    )


def normalize_error(error: Exception | str | Any) -> L0Event:
    """Normalize an error into L0 event format.

    Args:
        error: Error to normalize

    Returns:
        L0Event with type=error
    """
    if isinstance(error, Exception):
        err = error
    else:
        err = Exception(str(error))

    return L0Event(
        type=L0EventType.ERROR,
        timestamp=time.time() * 1000,
        error=err,
    )


def create_token_event(value: str) -> L0Event:
    """Create a token event.

    Args:
        value: Token value

    Returns:
        L0Event with type=token
    """
    return L0Event(
        type=L0EventType.TOKEN,
        timestamp=time.time() * 1000,
        value=value,
    )


def create_message_event(
    value: str,
    role: str = "assistant",
) -> L0Event:
    """Create a message event.

    Args:
        value: Message content
        role: Message role (user, assistant, system)

    Returns:
        L0Event with type=message
    """
    return L0Event(
        type=L0EventType.MESSAGE,
        timestamp=time.time() * 1000,
        value=value,
        role=role,
    )


def create_complete_event() -> L0Event:
    """Create a complete event.

    Returns:
        L0Event with type=complete
    """
    return L0Event(
        type=L0EventType.COMPLETE,
        timestamp=time.time() * 1000,
    )


def create_error_event(error: Exception) -> L0Event:
    """Create an error event.

    Args:
        error: The error

    Returns:
        L0Event with type=error
    """
    return L0Event(
        type=L0EventType.ERROR,
        timestamp=time.time() * 1000,
        error=error,
    )


def normalize_stream_events(chunks: list[Any]) -> list[L0Event]:
    """Batch normalize multiple chunks.

    Args:
        chunks: Array of chunks to normalize

    Returns:
        Array of normalized L0Events
    """
    return [normalize_stream_event(chunk) for chunk in chunks]


def filter_events_by_type(
    events: list[L0Event],
    event_type: L0EventType,
) -> list[L0Event]:
    """Filter events by type.

    Args:
        events: Events to filter
        event_type: Event type to filter for

    Returns:
        Filtered events
    """
    return [e for e in events if e.type == event_type]


def extract_tokens(events: list[L0Event]) -> list[str]:
    """Get all token values from events.

    Args:
        events: Events to extract tokens from

    Returns:
        Array of token values
    """
    return [e.value for e in events if e.type == L0EventType.TOKEN and e.value]


def reconstruct_text(events: list[L0Event]) -> str:
    """Reconstruct text from token events.

    Args:
        events: Events to reconstruct from

    Returns:
        Reconstructed text
    """
    return "".join(extract_tokens(events))


def is_error_event(event: L0Event) -> bool:
    """Check if event is an error event.

    Args:
        event: Event to check

    Returns:
        True if error event
    """
    return event.type == L0EventType.ERROR


def is_complete_event(event: L0Event) -> bool:
    """Check if event is a complete event.

    Args:
        event: Event to check

    Returns:
        True if complete event
    """
    return event.type == L0EventType.COMPLETE


def is_token_event(event: L0Event) -> bool:
    """Check if event is a token event.

    Args:
        event: Event to check

    Returns:
        True if token event
    """
    return event.type == L0EventType.TOKEN


def get_first_error(events: list[L0Event]) -> Exception | None:
    """Get first error from events.

    Args:
        events: Events to search

    Returns:
        First error or None
    """
    for event in events:
        if event.type == L0EventType.ERROR and event.error:
            return event.error
    return None
