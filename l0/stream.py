"""Stream utilities for L0."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from .types import Event, EventType

if TYPE_CHECKING:
    from .types import Stream


async def consume_stream(stream: AsyncIterator[Event]) -> str:
    """Consume stream and return full text."""
    content = ""
    async for event in stream:
        if event.type == EventType.TOKEN and event.value:
            content += event.value
    return content


async def get_text(result: Stream) -> str:
    """Helper to get text from Stream result."""
    return await result.text()
