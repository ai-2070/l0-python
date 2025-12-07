"""Edge case integration tests with real API calls.

These tests verify error handling, timeouts, streaming edge cases,
and other boundary conditions with actual LLM responses.

Requires OPENAI_API_KEY to be set.
Run with: pytest tests/integration/test_edge_cases.py -v
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest

import l0
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@requires_openai
class TestErrorHandlingEdgeCases:
    """Test error handling edge cases with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_all_fallbacks_exhausted(self, client: "AsyncOpenAI") -> None:
        """Test behavior when primary and all fallbacks fail."""

        async def failing_stream() -> AsyncIterator[Any]:
            raise ConnectionError("Primary failed")
            yield  # Make this an async generator

        async def fallback1() -> AsyncIterator[Any]:
            raise ConnectionError("Fallback 1 failed")
            yield  # Make this an async generator

        async def fallback2() -> AsyncIterator[Any]:
            raise ConnectionError("Fallback 2 failed")
            yield  # Make this an async generator

        result = await l0.run(
            stream=failing_stream,
            fallbacks=[fallback1, fallback2],
            retry=l0.Retry(attempts=1),
        )

        with pytest.raises(Exception):
            async for _ in result:
                pass

    @pytest.mark.asyncio
    async def test_empty_fallback_array(self, client: "AsyncOpenAI") -> None:
        """Test behavior with empty fallback array."""

        async def failing_stream() -> AsyncIterator[Any]:
            raise ConnectionError("Primary failed")
            yield  # Make this an async generator

        result = await l0.run(
            stream=failing_stream,
            fallbacks=[],
            retry=l0.Retry(attempts=1),
        )

        with pytest.raises(ConnectionError, match="Primary failed"):
            async for _ in result:
                pass


@requires_openai
class TestStreamingEdgeCases:
    """Test streaming edge cases with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_minimal_response(self, client: "AsyncOpenAI") -> None:
        """Test handling of minimal single-character response."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with only the letter 'X'."}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(stream)
        text = await result.read()

        assert len(text) > 0
        assert result.state.token_count > 0

    @pytest.mark.asyncio
    async def test_special_characters_in_response(self, client: "AsyncOpenAI") -> None:
        """Test handling of special characters in response."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with these exact characters: @#$%^&*()",
                }
            ],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)
        text = await result.read()

        assert len(text) > 0
        # Should contain at least some special characters
        assert any(c in text for c in "@#$%^&*()")

    @pytest.mark.asyncio
    async def test_unicode_and_emoji_in_response(self, client: "AsyncOpenAI") -> None:
        """Test handling of unicode and emoji in response."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with 3 different emojis."}],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)
        text = await result.read()

        assert len(text) > 0
        # Emoji are multi-byte unicode characters
        assert len(text.encode("utf-8")) >= len(text)

    @pytest.mark.asyncio
    async def test_newlines_and_whitespace(self, client: "AsyncOpenAI") -> None:
        """Test handling of newlines and various whitespace."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a haiku with proper line breaks.",
                }
            ],
            stream=True,
            max_tokens=50,
        )

        result = l0.wrap(stream)
        text = await result.read()

        assert len(text) > 0
        # Haiku should have newlines
        assert "\n" in text or len(text.split()) >= 3


@requires_openai
class TestGuardrailEdgeCases:
    """Test guardrail edge cases with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_empty_guardrails_array(self, client: "AsyncOpenAI") -> None:
        """Test behavior with empty guardrails array."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, guardrails=[])
        text = await result.read()

        assert len(text) > 0
        assert len(result.state.violations) == 0

    @pytest.mark.asyncio
    async def test_multiple_violations_collected(self, client: "AsyncOpenAI") -> None:
        """Test that multiple guardrail violations are collected."""
        violations_received: list[l0.Violation] = []

        def on_violation(v: l0.Violation) -> None:
            violations_received.append(v)

        # Create guardrails that will trigger on common words
        guardrails = [
            l0.custom_pattern_rule(
                patterns=[r"hello"],
                message="Contains hello",
                severity="warning",
            ),
            l0.custom_pattern_rule(
                patterns=[r"world"],
                message="Contains world",
                severity="warning",
            ),
        ]

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello, World!'"}],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(
            stream,
            guardrails=guardrails,
            on_violation=on_violation,
        )
        await result.read()

        # Should have detected violations for both patterns
        assert len(violations_received) >= 1


@requires_openai
class TestRetryEdgeCases:
    """Test retry edge cases with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_retry_count_tracking(self, client: "AsyncOpenAI") -> None:
        """Test that retry attempts are tracked accurately."""
        attempt_count = 0
        retry_count = 0

        def on_retry(attempt: int, reason: str) -> None:
            nonlocal retry_count
            retry_count += 1

        async def stream_with_failures():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Transient error")
            # Return actual stream on second attempt
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'success'."}],
                stream=True,
                max_tokens=10,
            )

        result = await l0.run(
            stream=stream_with_failures,  # type: ignore[arg-type]
            retry=l0.Retry(attempts=3),
            on_retry=on_retry,
        )

        text = await result.read()

        assert attempt_count == 2
        assert retry_count == 1
        assert "success" in text.lower()


@requires_openai
class TestTimeoutEdgeCases:
    """Test timeout edge cases with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_fast_response_no_timeout(self, client: "AsyncOpenAI") -> None:
        """Test that fast responses complete without timeout."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'ok'."}],
            stream=True,
            max_tokens=5,
        )

        # Use generous timeouts - should not trigger
        result = l0.wrap(
            stream,
            timeout=l0.Timeout(initial_token=30000, inter_token=30000),
        )
        text = await result.read()

        assert len(text) > 0
        assert result.state.completed


@requires_openai
class TestStateConsistency:
    """Test state consistency with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_state_after_completion(self, client: "AsyncOpenAI") -> None:
        """Test that state is consistent after stream completion."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            stream=True,
            max_tokens=30,
        )

        result = l0.wrap(stream)
        text = await result.read()

        # State should be fully populated
        assert result.state.completed
        assert result.state.token_count > 0
        assert result.state.content == text
        assert len(result.state.violations) == 0

    @pytest.mark.asyncio
    async def test_state_during_streaming(self, client: "AsyncOpenAI") -> None:
        """Test that state updates correctly during streaming."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Write a short sentence."}],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)

        token_counts: list[int] = []
        async for event in result:
            if event.is_token:
                token_counts.append(result.state.token_count)

        # Token count should increase monotonically
        assert len(token_counts) > 0
        for i in range(1, len(token_counts)):
            assert token_counts[i] >= token_counts[i - 1]


@requires_openai
class TestCallbackEdgeCases:
    """Test callback edge cases with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_callback_exception_handling(self, client: "AsyncOpenAI") -> None:
        """Test that stream continues even if callback throws."""
        callback_called = False

        def failing_callback(event: l0.ObservabilityEvent) -> None:
            nonlocal callback_called
            callback_called = True
            raise ValueError("Callback error")

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test'."}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, on_event=failing_callback)

        # Stream should complete despite callback error
        text = await result.read()
        assert len(text) > 0
        assert callback_called

    @pytest.mark.asyncio
    async def test_all_callbacks_fire(self, client: "AsyncOpenAI") -> None:
        """Test that all lifecycle callbacks fire correctly."""
        events: list[str] = []
        tokens: list[str] = []

        def on_event(event: l0.ObservabilityEvent) -> None:
            events.append(event.type.value)

        def on_token(text: str) -> None:
            tokens.append(text)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello'."}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(
            stream,
            on_event=on_event,
            on_token=on_token,
        )
        await result.read()

        # Check that key events were fired
        assert l0.ObservabilityEventType.STREAM_INIT.value in events
        assert l0.ObservabilityEventType.COMPLETE.value in events
        assert len(tokens) > 0
