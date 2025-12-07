"""Integration tests with OpenAI API.

These tests require OPENAI_API_KEY to be set in environment or .env file.
Run with: pytest tests/integration -v
"""

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

import l0 as l0

# Import the marker from conftest
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@requires_openai
class TestOpenAIIntegration:
    """Integration tests using real OpenAI API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        """Create OpenAI client."""
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_basic_wrap(self, client: "AsyncOpenAI") -> None:
        """Test basic l0.wrap() with OpenAI."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            stream=True,
            max_tokens=10,
        )

        # wrap() returns immediately - no await!
        result = l0.wrap(stream)
        text = await result.read()

        assert "hello" in text.lower()
        assert result.state.token_count > 0
        assert result.state.completed

    @pytest.mark.asyncio
    async def test_streaming_events(self, client: "AsyncOpenAI") -> None:
        """Test streaming individual events."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            max_tokens=20,
        )

        tokens = []
        async for event in l0.wrap(stream):
            if event.is_token:
                tokens.append(event.text)
            elif event.is_complete:
                break

        assert len(tokens) > 0
        full_text = "".join(t for t in tokens if t)
        assert any(c in full_text for c in ["1", "2", "3"])

    @pytest.mark.asyncio
    async def test_with_guardrails(self, client: "AsyncOpenAI") -> None:
        """Test streaming with guardrails."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello world'."}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, guardrails=l0.Guardrails.recommended())
        text = await result.read()

        assert len(text) > 0
        # Check for no error-level violations (warnings like "instant completion" are ok)
        error_violations = [v for v in result.state.violations if v.severity == "error"]
        assert len(error_violations) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self, client: "AsyncOpenAI") -> None:
        """Test async context manager pattern."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hi'."}],
            stream=True,
            max_tokens=5,
        )

        tokens: list[str] = []
        async with l0.wrap(stream) as result:
            async for event in result:
                if event.is_token and event.text:
                    tokens.append(event.text)

        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_observability_callback(self, client: "AsyncOpenAI") -> None:
        """Test observability event callback."""
        events_received = []

        def on_event(event: l0.ObservabilityEvent) -> None:
            events_received.append(event.type)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(stream, on_event=on_event)
        await result.read()

        assert len(events_received) > 0
        assert l0.ObservabilityEventType.STREAM_INIT in events_received
        assert l0.ObservabilityEventType.COMPLETE in events_received

    @pytest.mark.asyncio
    async def test_with_timeout(self, client: "AsyncOpenAI") -> None:
        """Test that fast responses don't timeout."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'ok'."}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(
            stream,
            timeout=l0.Timeout(initial_token=30000, inter_token=30000),
        )
        text = await result.read()

        assert len(text) > 0


@requires_openai
class TestRunWithFallbacks:
    """Test l0.run() with fallbacks (requires lambda for retries)."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_fallback_succeeds(self, client: "AsyncOpenAI") -> None:
        """Test that fallback works when using valid models."""
        # run() needs lambdas for retry/fallback support
        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'primary'."}],
                stream=True,
                max_tokens=10,
            ),
            fallbacks=[
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say 'fallback'."}],
                    stream=True,
                    max_tokens=10,
                ),
            ],
        )

        text = await result.read()
        # Primary should succeed
        assert result.state.fallback_index == 0
        assert "primary" in text.lower()


@requires_openai
class TestStructuredOutput:
    """Test structured output with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_structured_json(self, client: "AsyncOpenAI") -> None:
        """Test structured output parsing."""

        class Person(BaseModel):
            name: str
            age: int

        # structured() needs lambda for potential retries
        result = await l0.structured(
            schema=Person,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON: {"name": "Alice", "age": 30}',
                    }
                ],
                stream=True,
                max_tokens=50,
            ),
        )

        assert result.data.name == "Alice"
        assert result.data.age == 30
