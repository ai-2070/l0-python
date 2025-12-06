"""Integration tests for multimodal features with real API calls.

These tests require OPENAI_API_KEY to be set.
Run with: pytest tests/integration/test_multimodal.py -v
"""

from typing import TYPE_CHECKING

import pytest

import l0
from l0 import EventType
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@requires_openai
class TestMultimodalWithTextStreaming:
    """Tests combining multimodal helpers with text streaming."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_text_stream_with_progress_simulation(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test simulating progress during text generation."""
        events_collected = []

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)

        token_count = 0
        async for event in result:
            events_collected.append(event.type)
            if event.is_token:
                token_count += 1

        assert EventType.TOKEN in events_collected
        assert EventType.COMPLETE in events_collected
        assert token_count > 0

    @pytest.mark.asyncio
    async def test_multimodal_event_types_in_callback(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test that multimodal event types work in observability callbacks."""
        event_types_received = []

        def on_event(event: l0.ObservabilityEvent) -> None:
            event_types_received.append(event.type.value)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(stream, on_event=on_event)
        await result.read()

        assert "STREAM_INIT" in event_types_received
        assert "COMPLETE" in event_types_received
