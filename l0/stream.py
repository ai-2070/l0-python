"""Stream utilities for L0."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from .types import EventType, L0Event

if TYPE_CHECKING:
    from .types import L0Result


async def consume_stream(stream: AsyncIterator[L0Event]) -> str:
    """Consume stream and return full text."""
    content = ""
    async for event in stream:
        if event.type == EventType.TOKEN and event.value:
            content += event.value
    return content


async def get_text(result: L0Result) -> str:
    """Helper to get text from L0Result."""
    return await consume_stream(result.stream)
