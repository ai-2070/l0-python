"""Stream utilities for L0."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .types import Event, EventType

if TYPE_CHECKING:
    from .types import Stream


async def consume_stream(stream: AsyncIterator[Event]) -> str:
    """Consume stream and return full text."""
    parts: list[str] = []
    async for event in stream:
        if event.type == EventType.TOKEN and event.text:
            parts.append(event.text)
    return "".join(parts)


async def get_text(result: "Stream[Any]") -> str:
    """Helper to get text from Stream result."""
    return await result.read()
