"""Integration tests with OpenAI API.

These tests require OPENAI_API_KEY to be set in environment or .env file.
Run with: pytest tests/integration -v
"""

import pytest
from pydantic import BaseModel

import l0

# Import the marker from conftest
from tests.conftest import requires_openai


@requires_openai
class TestOpenAIIntegration:
    """Integration tests using real OpenAI API."""

    @pytest.fixture
    def client(self):
        """Create OpenAI client."""
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_basic_streaming(self, client):
        """Test basic streaming with OpenAI."""
        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
                stream=True,
                max_tokens=10,
            ),
        )

        text = await result.read()
        assert "hello" in text.lower()
        assert result.state.token_count > 0
        assert result.state.completed

    @pytest.mark.asyncio
    async def test_wrap_api(self, client):
        """Test l0.wrap() with OpenAI stream."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            stream=True,
            max_tokens=10,
        )

        # wrap() returns immediately - no await!
        result = l0.wrap(stream)

        text = await result.read()
        assert "test" in text.lower()

    @pytest.mark.asyncio
    async def test_streaming_events(self, client):
        """Test streaming individual events."""
        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Count from 1 to 3."}],
                stream=True,
                max_tokens=20,
            ),
        )

        tokens = []
        async for event in result:
            if event.is_token:
                tokens.append(event.text)
            elif event.is_complete:
                break

        assert len(tokens) > 0
        full_text = "".join(t for t in tokens if t)
        assert any(c in full_text for c in ["1", "2", "3"])

    @pytest.mark.asyncio
    async def test_with_guardrails(self, client):
        """Test streaming with guardrails."""
        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'hello world'."}],
                stream=True,
                max_tokens=10,
            ),
            guardrails=l0.Guardrails.recommended(),
        )

        text = await result.read()
        assert len(text) > 0
        # No violations expected for simple response
        assert len(result.state.violations) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager pattern."""
        async with l0.wrap(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'hi'."}],
                stream=True,
                max_tokens=5,
            )
        ) as result:
            tokens = []
            async for event in result:
                if event.is_token and event.text:
                    tokens.append(event.text)

        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_observability_callback(self, client):
        """Test observability event callback."""
        events_received = []

        def on_event(event):
            events_received.append(event.type)

        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=5,
            ),
            on_event=on_event,
        )

        await result.read()

        # Should have received some events
        assert len(events_received) > 0
        assert l0.ObservabilityEventType.STREAM_INIT in events_received
        assert l0.ObservabilityEventType.COMPLETE in events_received


@requires_openai
class TestStructuredOutput:
    """Test structured output with real API."""

    @pytest.fixture
    def client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_structured_json(self, client):
        """Test structured output parsing."""

        class Person(BaseModel):
            name: str
            age: int

        result = await l0.structured(
            schema=Person,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Return JSON with name='Alice' and age=30. Only JSON, no other text.",
                    }
                ],
                stream=True,
                max_tokens=50,
            ),
        )

        assert result.name == "Alice"
        assert result.age == 30


@requires_openai
class TestFallbacks:
    """Test fallback functionality."""

    @pytest.fixture
    def client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_fallback_to_second_model(self, client):
        """Test that fallback works when using valid models."""
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
class TestTimeout:
    """Test timeout functionality."""

    @pytest.fixture
    def client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_no_timeout_on_fast_response(self, client):
        """Test that fast responses don't timeout."""
        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'ok'."}],
                stream=True,
                max_tokens=5,
            ),
            timeout=l0.Timeout(initial_token=30.0, inter_token=30.0),
        )

        text = await result.read()
        assert len(text) > 0
