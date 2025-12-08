"""Integration tests with LiteLLM.

These tests require litellm to be installed and an API key to be set.
LiteLLM supports 100+ providers - tests use OpenAI by default but can use any.

Run with: pytest tests/integration/test_litellm.py -v

Note: LiteLLM's CustomStreamWrapper implements AsyncIterator at runtime but isn't
typed as such. We use cast() to satisfy the type checker.
"""

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from pydantic import BaseModel

import l0 as l0
from tests.conftest import requires_litellm


def as_stream(stream: Any) -> AsyncIterator[Any]:
    """Cast LiteLLM stream to AsyncIterator for type checking."""
    return cast(AsyncIterator[Any], stream)


def as_stream_factory(factory: Any) -> l0.AwaitableStreamFactory:
    """Cast LiteLLM stream factory for type checking."""
    return cast(l0.AwaitableStreamFactory, factory)


@requires_litellm
class TestLiteLLMIntegration:
    """Integration tests using LiteLLM."""

    @pytest.mark.asyncio
    async def test_basic_wrap(self):
        """Test basic l0.wrap() with LiteLLM."""
        import litellm

        stream = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(as_stream(stream), adapter="litellm")
        text = await result.read()

        assert "hello" in text.lower()
        assert result.state.token_count > 0
        assert result.state.completed

    @pytest.mark.asyncio
    async def test_streaming_events(self):
        """Test streaming individual events."""
        import litellm

        stream = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            max_tokens=20,
        )

        tokens = []
        async for event in l0.wrap(as_stream(stream), adapter="litellm"):
            if event.is_token:
                tokens.append(event.text)
            elif event.is_complete:
                break

        assert len(tokens) > 0
        full_text = "".join(t for t in tokens if t)
        assert any(c in full_text for c in ["1", "2", "3"])

    @pytest.mark.asyncio
    async def test_with_guardrails(self):
        """Test streaming with guardrails."""
        import litellm

        stream = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello world'."}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(
            as_stream(stream), adapter="litellm", guardrails=l0.Guardrails.recommended()
        )
        text = await result.read()

        assert len(text) > 0
        assert len(result.state.violations) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager pattern."""
        import litellm

        stream = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hi'."}],
            stream=True,
            max_tokens=5,
        )

        tokens: list[str] = []
        async with l0.wrap(as_stream(stream), adapter="litellm") as result:
            async for event in result:
                if event.is_token and event.text:
                    tokens.append(event.text)

        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_observability_callback(self):
        """Test observability event callback."""
        import litellm

        events_received = []

        def on_event(event: l0.ObservabilityEvent) -> None:
            events_received.append(event.type)

        stream = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(as_stream(stream), adapter="litellm", on_event=on_event)
        await result.read()

        assert len(events_received) > 0
        assert l0.ObservabilityEventType.STREAM_INIT in events_received
        assert l0.ObservabilityEventType.COMPLETE in events_received

    @pytest.mark.asyncio
    async def test_with_timeout(self):
        """Test that fast responses don't timeout."""
        import litellm

        stream = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'ok'."}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(
            as_stream(stream),
            adapter="litellm",
            timeout=l0.Timeout(initial_token=30000, inter_token=30000),
        )
        text = await result.read()

        assert len(text) > 0


@requires_litellm
class TestLiteLLMRunWithFallbacks:
    """Test l0.run() with fallbacks using LiteLLM."""

    @pytest.mark.asyncio
    async def test_fallback_succeeds(self):
        """Test that fallback works when using valid models."""
        import litellm

        # run() needs lambdas for retry/fallback support
        result = await l0.run(
            stream=as_stream_factory(
                lambda: litellm.acompletion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say 'primary'."}],
                    stream=True,
                    max_tokens=10,
                )
            ),
            fallbacks=[
                as_stream_factory(
                    lambda: litellm.acompletion(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "Say 'fallback'."}],
                        stream=True,
                        max_tokens=10,
                    )
                ),
            ],
            adapter="litellm",
        )

        text = await result.read()
        # Primary should succeed
        assert result.state.fallback_index == 0
        assert "primary" in text.lower()


@requires_litellm
class TestLiteLLMStructuredOutput:
    """Test structured output with LiteLLM."""

    @pytest.mark.asyncio
    async def test_structured_json(self):
        """Test structured output parsing."""
        import litellm

        class Person(BaseModel):
            name: str
            age: int

        # structured() needs lambda for potential retries
        result = await l0.structured(
            schema=Person,
            stream=as_stream_factory(
                lambda: litellm.acompletion(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": 'Return JSON: {"name": "Alice", "age": 30}',
                        }
                    ],
                    stream=True,
                    max_tokens=50,
                )
            ),
            adapter="litellm",
        )

        assert result.data.name == "Alice"
        assert result.data.age == 30
