"""Integration tests for custom adapters with real API calls.

These tests require OPENAI_API_KEY to be set.
Run with: pytest tests/integration/test_custom_adapters.py -v
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest

import l0
from l0.adapters import AdaptedEvent, Adapters, OpenAIAdapter
from l0.types import Event, EventType
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI


# ─────────────────────────────────────────────────────────────────────────────
# Custom Adapter Examples
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIWrapperAdapter:
    """Adapter that wraps OpenAI but adds custom processing."""

    name = "openai_wrapper"

    def __init__(self, prefix: str = "[AI] "):
        self.prefix = prefix
        self._openai_adapter = OpenAIAdapter()

    def detect(self, stream: Any) -> bool:
        return self._openai_adapter.detect(stream)

    async def wrap(
        self,
        stream: AsyncIterator[Any],
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        first_token = True
        async for adapted in self._openai_adapter.wrap(stream, options):
            if adapted.event.type == EventType.TOKEN and adapted.event.text:
                if first_token:
                    adapted.event.text = self.prefix + adapted.event.text
                    first_token = False
            yield adapted


# ─────────────────────────────────────────────────────────────────────────────
# Explicit Adapter Usage Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestExplicitAdapterUsage:
    """Tests for using adapters explicitly with adapter= parameter."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_explicit_openai_adapter(self, client: "AsyncOpenAI") -> None:
        """Test explicitly specifying OpenAI adapter."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'explicit adapter test'"}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, adapter=Adapters.openai())
        text = await result.read()

        assert len(text) > 0
        assert result.state.completed

    @pytest.mark.asyncio
    async def test_adapter_by_name(self, client: "AsyncOpenAI") -> None:
        """Test specifying adapter by string name."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'named adapter'"}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, adapter="openai")
        text = await result.read()

        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_litellm_adapter_alias(self, client: "AsyncOpenAI") -> None:
        """Test that litellm adapter name works (aliases to openai)."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'litellm alias'"}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, adapter="litellm")
        text = await result.read()

        assert len(text) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Custom Adapter Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestCustomAdapterIntegration:
    """Tests for custom adapter integration with L0."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.fixture(autouse=True)
    def reset_adapters(self):
        Adapters.reset()
        yield
        Adapters.reset()

    @pytest.mark.asyncio
    async def test_custom_wrapper_adapter(self, client: "AsyncOpenAI") -> None:
        """Test custom adapter that wraps OpenAI with prefix."""
        wrapper_adapter = OpenAIWrapperAdapter(prefix="[BOT] ")

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello'"}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, adapter=wrapper_adapter)
        text = await result.read()

        assert text.startswith("[BOT] ")

    @pytest.mark.asyncio
    async def test_custom_adapter_with_run(self, client: "AsyncOpenAI") -> None:
        """Test custom adapter with l0.run() for retry support."""
        wrapper_adapter = OpenAIWrapperAdapter(prefix=">>> ")

        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'test'"}],
                stream=True,
                max_tokens=10,
            ),
            adapter=wrapper_adapter,
        )

        text = await result.read()
        assert text.startswith(">>> ")

    @pytest.mark.asyncio
    async def test_custom_adapter_with_guardrails(self, client: "AsyncOpenAI") -> None:
        """Test custom adapter combined with guardrails."""
        wrapper_adapter = OpenAIWrapperAdapter(prefix="[SAFE] ")

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello world'"}],
            stream=True,
            max_tokens=15,
        )

        result = l0.wrap(
            stream,
            adapter=wrapper_adapter,
            guardrails=l0.Guardrails.recommended(),
        )
        text = await result.read()

        assert text.startswith("[SAFE] ")
        assert len(result.state.violations) == 0

    @pytest.mark.asyncio
    async def test_custom_adapter_with_timeout(self, client: "AsyncOpenAI") -> None:
        """Test custom adapter with timeout configuration."""
        wrapper_adapter = OpenAIWrapperAdapter(prefix="[T] ")

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'quick'"}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(
            stream,
            adapter=wrapper_adapter,
            timeout=l0.Timeout(initial_token=30000, inter_token=30000),
        )
        text = await result.read()

        assert text.startswith("[T] ")


# ─────────────────────────────────────────────────────────────────────────────
# Adapter Registration Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestAdapterRegistration:
    """Tests for adapter registration and auto-detection."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.fixture(autouse=True)
    def reset_adapters(self):
        Adapters.reset()
        yield
        Adapters.reset()

    @pytest.mark.asyncio
    async def test_register_custom_adapter(self, client: "AsyncOpenAI") -> None:
        """Test registering a custom adapter."""
        custom = OpenAIWrapperAdapter(prefix="[REG] ")
        Adapters.register(custom)

        assert "openai_wrapper" in Adapters.registered()

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'registered'"}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, adapter=custom)
        text = await result.read()

        assert text.startswith("[REG] ")


# ─────────────────────────────────────────────────────────────────────────────
# Adapter Auto-Detection Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestAdapterAutoDetection:
    """Tests for automatic adapter detection."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.fixture(autouse=True)
    def reset_adapters(self):
        Adapters.reset()
        yield
        Adapters.reset()

    @pytest.mark.asyncio
    async def test_auto_detect_openai(self, client: "AsyncOpenAI") -> None:
        """Test auto-detection of OpenAI streams."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'auto'"}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(stream)
        text = await result.read()

        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_detect_adapter_method(self, client: "AsyncOpenAI") -> None:
        """Test Adapters.detect() method."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            max_tokens=5,
        )

        detected = Adapters.detect(stream)
        assert detected.name == "openai"

    @pytest.mark.asyncio
    async def test_detect_with_hint(self, client: "AsyncOpenAI") -> None:
        """Test Adapters.detect() with hint."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            max_tokens=5,
        )

        detected = Adapters.detect(stream, hint="openai")
        assert detected.name == "openai"

        custom = OpenAIWrapperAdapter()
        detected = Adapters.detect(stream, hint=custom)
        assert detected.name == "openai_wrapper"


# ─────────────────────────────────────────────────────────────────────────────
# Streaming with Custom Adapters
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestCustomAdapterStreaming:
    """Tests for streaming with custom adapters."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.fixture(autouse=True)
    def reset_adapters(self):
        Adapters.reset()
        yield
        Adapters.reset()

    @pytest.mark.asyncio
    async def test_stream_events_with_custom_adapter(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test streaming individual events with custom adapter."""
        wrapper = OpenAIWrapperAdapter(prefix="[S] ")

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count: 1, 2, 3"}],
            stream=True,
            max_tokens=20,
        )

        tokens = []
        async for event in l0.wrap(stream, adapter=wrapper):
            if event.is_token and event.text:
                tokens.append(event.text)
            elif event.is_complete:
                break

        full_text = "".join(tokens)
        assert full_text.startswith("[S] ")
        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_observability_with_custom_adapter(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test observability callbacks with custom adapter."""
        events_received = []
        wrapper = OpenAIWrapperAdapter(prefix="[O] ")

        def on_event(event: l0.ObservabilityEvent) -> None:
            events_received.append(event.type)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'observable'"}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, adapter=wrapper, on_event=on_event)
        await result.read()

        assert l0.ObservabilityEventType.STREAM_INIT in events_received
        assert l0.ObservabilityEventType.COMPLETE in events_received
