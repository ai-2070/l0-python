"""Tests for l0.client module - wrapped client functionality."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from l0.client import (
    ClientConfig,
    WrappedChat,
    WrappedClient,
    WrappedCompletions,
    wrap_client,
)
from l0.types import Retry, Timeout


class TestClientConfig:
    def test_default_config(self):
        config = ClientConfig()
        assert config.fallbacks is None
        assert config.guardrails is None
        assert config.retry is not None  # Defaults to Retry.recommended()
        assert config.timeout is None
        assert config.adapter is None
        assert config.on_event is None
        assert config.context is None
        assert config.buffer_tool_calls is False
        # Default is False per client.py
        assert config.continue_from_last_good_token is False
        assert config.build_continuation_prompt is None

    def test_config_with_all_options(self):
        fallbacks = [lambda: "fallback"]
        guardrails = []
        retry = Retry(attempts=5)
        timeout = Timeout(initial_token=10000, inter_token=5000)
        on_event = lambda e: None
        context = {"request_id": "test-123"}

        config = ClientConfig(
            fallbacks=fallbacks,
            guardrails=guardrails,
            retry=retry,
            timeout=timeout,
            adapter="openai",
            on_event=on_event,
            context=context,
            buffer_tool_calls=True,
            continue_from_last_good_token=True,
        )

        assert config.fallbacks == fallbacks
        assert config.guardrails == guardrails
        assert config.retry == retry
        assert config.timeout == timeout
        assert config.adapter == "openai"
        assert config.on_event == on_event
        assert config.context == context
        assert config.buffer_tool_calls is True
        assert config.continue_from_last_good_token is True


class TestWrappedCompletions:
    @pytest.fixture
    def mock_completions(self):
        completions = MagicMock()
        completions.create = AsyncMock()
        return completions

    @pytest.fixture
    def mock_client(self) -> Any:
        return MagicMock(spec=WrappedClient)

    @pytest.fixture
    def config(self) -> ClientConfig:
        return ClientConfig()

    def test_init(
        self, mock_client: Any, mock_completions: Any, config: ClientConfig
    ) -> None:
        wrapped = WrappedCompletions(mock_client, mock_completions, config)
        assert wrapped._client == mock_client
        assert wrapped._completions == mock_completions
        assert wrapped._config == config

    @pytest.mark.asyncio
    async def test_create_non_streaming_passthrough(
        self, mock_client: Any, mock_completions: Any, config: ClientConfig
    ) -> None:
        """Non-streaming requests should pass through to underlying client."""
        mock_completions.create.return_value = {"content": "response"}
        wrapped = WrappedCompletions(mock_client, mock_completions, config)

        result = await wrapped.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )

        mock_completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert result == {"content": "response"}

    @pytest.mark.asyncio
    async def test_create_streaming_uses_l0(
        self, mock_client: Any, mock_completions: Any, config: ClientConfig
    ) -> None:
        """Streaming requests should use L0 runtime."""

        async def mock_stream():
            yield {"choices": [{"delta": {"content": "Hello"}}]}

        mock_completions.create.return_value = mock_stream()
        wrapped = WrappedCompletions(mock_client, mock_completions, config)

        with patch("l0.runtime._internal_run") as mock_run:
            mock_run.return_value = AsyncMock()

            await wrapped.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            mock_run.assert_called_once()
            # Check that stream factory was passed
            call_kwargs = mock_run.call_args.kwargs
            assert "stream" in call_kwargs
            assert call_kwargs["retry"] == config.retry


class TestWrappedChat:
    @pytest.fixture
    def mock_chat(self) -> Any:
        chat = MagicMock()
        chat.completions = MagicMock()
        return chat

    @pytest.fixture
    def mock_client(self) -> Any:
        return MagicMock(spec=WrappedClient)

    @pytest.fixture
    def config(self) -> ClientConfig:
        return ClientConfig()

    def test_init(self, mock_client: Any, mock_chat: Any, config: ClientConfig) -> None:
        wrapped = WrappedChat(mock_client, mock_chat, config)
        assert wrapped._client == mock_client
        assert wrapped._chat == mock_chat
        assert wrapped._config == config
        assert isinstance(wrapped.completions, WrappedCompletions)


class TestWrappedClient:
    @pytest.fixture
    def mock_underlying_client(self) -> Any:
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        return client

    @pytest.fixture
    def config(self) -> ClientConfig:
        return ClientConfig()

    def test_init_with_chat(
        self, mock_underlying_client: Any, config: ClientConfig
    ) -> None:
        wrapped = WrappedClient(mock_underlying_client, config)
        assert wrapped._client == mock_underlying_client
        assert wrapped._config == config
        assert hasattr(wrapped, "chat")
        assert isinstance(wrapped.chat, WrappedChat)

    def test_init_without_chat(self, config: ClientConfig) -> None:
        client = MagicMock(spec=[])  # No chat attribute
        wrapped = WrappedClient(client, config)
        assert wrapped._client == client
        assert not hasattr(wrapped, "chat") or wrapped.chat is None

    def test_unwrapped_property(
        self, mock_underlying_client: Any, config: ClientConfig
    ) -> None:
        wrapped = WrappedClient(mock_underlying_client, config)
        assert wrapped.unwrapped == mock_underlying_client

    def test_with_options_creates_new_client(
        self, mock_underlying_client: Any, config: ClientConfig
    ) -> None:
        wrapped = WrappedClient(mock_underlying_client, config)
        new_retry = Retry(attempts=10)

        new_wrapped = wrapped.with_options(retry=new_retry)

        assert new_wrapped is not wrapped
        assert new_wrapped._config.retry == new_retry
        # Original unchanged
        assert wrapped._config.retry == config.retry

    def test_with_options_inherits_unspecified(
        self, mock_underlying_client: Any
    ) -> None:
        original_config = ClientConfig(
            retry=Retry(attempts=5),
            timeout=Timeout(initial_token=10000, inter_token=5000),
            adapter="openai",
            context={"original": True},
        )
        wrapped = WrappedClient(mock_underlying_client, original_config)

        # Only change retry
        new_wrapped = wrapped.with_options(retry=Retry(attempts=10))

        # Changed
        assert new_wrapped._config.retry.attempts == 10
        # Inherited
        assert new_wrapped._config.timeout == original_config.timeout
        assert new_wrapped._config.adapter == original_config.adapter
        assert new_wrapped._config.context == original_config.context

    def test_with_options_all_parameters(
        self, mock_underlying_client: Any, config: ClientConfig
    ) -> None:
        wrapped = WrappedClient(mock_underlying_client, config)

        new_fallbacks = [lambda: "fb"]
        new_guardrails = []
        new_retry = Retry(attempts=7)
        new_timeout = Timeout(initial_token=15000, inter_token=8000)
        new_on_event = lambda e: None
        new_context = {"new": True}
        new_continuation_prompt = lambda s: f"Continue: {s}"

        new_wrapped = wrapped.with_options(
            fallbacks=new_fallbacks,
            guardrails=new_guardrails,
            retry=new_retry,
            timeout=new_timeout,
            adapter="litellm",
            on_event=new_on_event,
            context=new_context,
            buffer_tool_calls=True,
            continue_from_last_good_token=True,
            build_continuation_prompt=new_continuation_prompt,
        )

        assert new_wrapped._config.fallbacks == new_fallbacks
        assert new_wrapped._config.guardrails == new_guardrails
        assert new_wrapped._config.retry == new_retry
        assert new_wrapped._config.timeout == new_timeout
        assert new_wrapped._config.adapter == "litellm"
        assert new_wrapped._config.on_event == new_on_event
        assert new_wrapped._config.context == new_context
        assert new_wrapped._config.buffer_tool_calls is True
        assert new_wrapped._config.continue_from_last_good_token is True
        assert new_wrapped._config.build_continuation_prompt == new_continuation_prompt


class TestWrapClient:
    @pytest.fixture
    def mock_client(self) -> Any:
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        return client

    def test_wrap_returns_wrapped_client(self, mock_client: Any) -> None:
        wrapped = wrap_client(mock_client)
        assert isinstance(wrapped, WrappedClient)
        assert wrapped.unwrapped == mock_client

    def test_wrap_with_default_options(self, mock_client: Any) -> None:
        wrapped = wrap_client(mock_client)
        # Default retry is applied
        assert wrapped._config.retry is not None
        # Default continue_from_last_good_token is True per wrap_client signature
        assert wrapped._config.continue_from_last_good_token is True

    def test_wrap_with_custom_options(self, mock_client: Any) -> None:
        retry = Retry(attempts=5)
        timeout = Timeout(initial_token=10000, inter_token=5000)
        context = {"tenant_id": "abc"}

        wrapped = wrap_client(
            mock_client,
            retry=retry,
            timeout=timeout,
            adapter="openai",
            context=context,
            buffer_tool_calls=True,
        )

        assert wrapped._config.retry == retry
        assert wrapped._config.timeout == timeout
        assert wrapped._config.adapter == "openai"
        assert wrapped._config.context == context
        assert wrapped._config.buffer_tool_calls is True

    def test_wrap_with_guardrails(self, mock_client: Any) -> None:
        from l0.guardrails import zero_output_rule

        rules = [zero_output_rule()]
        wrapped = wrap_client(mock_client, guardrails=rules)
        assert wrapped._config.guardrails == rules

    def test_wrap_with_fallbacks(self, mock_client: Any) -> None:
        fallback1 = MagicMock()
        fallback2 = MagicMock()
        fallbacks = [lambda: fallback1, lambda: fallback2]

        wrapped = wrap_client(mock_client, fallbacks=fallbacks)
        assert wrapped._config.fallbacks == fallbacks

    def test_wrap_with_on_event_callback(self, mock_client: Any) -> None:
        events: list[Any] = []

        def on_event(event: Any) -> None:
            events.append(event)

        wrapped = wrap_client(mock_client, on_event=on_event)
        assert wrapped._config.on_event == on_event

    def test_wrap_with_continuation_config(self, mock_client: Any) -> None:
        from l0.continuation import ContinuationConfig

        continuation = ContinuationConfig(
            enabled=True,
            checkpoint_interval=5,
        )

        wrapped = wrap_client(
            mock_client,
            continue_from_last_good_token=continuation,
        )
        assert wrapped._config.continue_from_last_good_token == continuation

    def test_wrap_with_custom_continuation_prompt(self, mock_client: Any) -> None:
        def build_prompt(content: str) -> str:
            return f"Please continue from: {content}"

        wrapped = wrap_client(
            mock_client,
            build_continuation_prompt=build_prompt,
        )
        assert wrapped._config.build_continuation_prompt == build_prompt


class TestWrappedClientIntegration:
    """Integration tests for the wrapped client flow."""

    @pytest.fixture
    def mock_openai_client(self) -> Any:
        """Create a mock that mimics AsyncOpenAI structure."""
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_non_streaming_flow(self, mock_openai_client: Any) -> None:
        """Test that non-streaming calls pass through correctly."""
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello!"))]
        )

        wrapped = wrap_client(mock_openai_client)

        result = await wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        mock_openai_client.chat.completions.create.assert_called_once()
        assert result.choices[0].message.content == "Hello!"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_chained_with_options(self, mock_openai_client: Any) -> None:
        """Test that with_options creates independent configurations."""
        wrapped = wrap_client(mock_openai_client, retry=Retry(attempts=3))

        # Create variant with different retry
        high_retry = wrapped.with_options(retry=Retry(attempts=10))

        # Create variant with different timeout
        with_timeout = wrapped.with_options(
            timeout=Timeout(initial_token=20000, inter_token=15000)
        )

        # All three should be independent
        assert wrapped._config.retry.attempts == 3
        assert high_retry._config.retry.attempts == 10
        assert with_timeout._config.retry.attempts == 3
        assert with_timeout._config.timeout is not None
        assert with_timeout._config.timeout.initial_token == 20000

    def test_slots_optimization(self):
        """Verify __slots__ are used for memory efficiency."""
        assert hasattr(WrappedCompletions, "__slots__")
        assert hasattr(WrappedChat, "__slots__")
        assert hasattr(WrappedClient, "__slots__")
        assert hasattr(ClientConfig, "__slots__")
