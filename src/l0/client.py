"""Client wrapper for seamless L0 integration.

Wrap an OpenAI or LiteLLM client to get automatic reliability:

    ```python
    import asyncio
    import l0
    from openai import AsyncOpenAI

    # Wrap once
    client = l0.wrap(AsyncOpenAI())

    async def main():
        # Use normally - L0 reliability is automatic
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        async for chunk in response:
            print(chunk.choices[0].delta.content or "", end="")

    asyncio.run(main())
    ```
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from .continuation import ContinuationConfig
from .types import Retry, Stream, Timeout

if TYPE_CHECKING:
    from .adapters import Adapter
    from .events import ObservabilityEvent
    from .guardrails import GuardrailRule, GuardrailViolation


class WrappedCompletions:
    """Wrapped completions endpoint with L0 reliability."""

    __slots__ = ("_client", "_completions", "_config")

    def __init__(
        self,
        client: WrappedClient,
        completions: Any,
        config: ClientConfig,
    ) -> None:
        self._client = client
        self._completions = completions
        self._config = config

    async def create(self, **kwargs: Any) -> "Stream[Any]" | Any:
        """Create a chat completion with L0 reliability.

        When stream=True, returns an L0 Stream with automatic retry,
        fallbacks, guardrails, and continuation support.

        When stream=False, passes through to the underlying client.
        """
        is_streaming = kwargs.get("stream", False)

        if not is_streaming:
            # Non-streaming - pass through directly
            return await self._completions.create(**kwargs)

        # Streaming - wrap with L0
        from .runtime import _internal_run

        def stream_factory() -> Any:
            return self._completions.create(**kwargs)

        return await _internal_run(
            stream=stream_factory,
            fallbacks=self._config.fallbacks,
            guardrails=self._config.guardrails,
            retry=self._config.retry,
            timeout=self._config.timeout,
            adapter=self._config.adapter,
            on_event=self._config.on_event,
            on_token=self._config.on_token,
            on_tool_call=self._config.on_tool_call,
            on_violation=self._config.on_violation,
            context=self._config.context,
            buffer_tool_calls=self._config.buffer_tool_calls,
            continue_from_last_good_token=self._config.continue_from_last_good_token,
            build_continuation_prompt=self._config.build_continuation_prompt,
        )


class WrappedChat:
    """Wrapped chat namespace."""

    __slots__ = ("_client", "_chat", "_config", "completions")

    def __init__(
        self,
        client: WrappedClient,
        chat: Any,
        config: ClientConfig,
    ) -> None:
        self._client = client
        self._chat = chat
        self._config = config
        self.completions = WrappedCompletions(client, chat.completions, config)


class ClientConfig:
    """Configuration for wrapped client."""

    __slots__ = (
        "fallbacks",
        "guardrails",
        "retry",
        "timeout",
        "adapter",
        "on_event",
        "on_token",
        "on_tool_call",
        "on_violation",
        "context",
        "buffer_tool_calls",
        "continue_from_last_good_token",
        "build_continuation_prompt",
    )

    def __init__(
        self,
        *,
        fallbacks: list[Callable[[], Any]] | None = None,
        guardrails: list[GuardrailRule] | None = None,
        retry: Retry | None = None,
        timeout: Timeout | None = None,
        adapter: "Adapter | str | None" = None,
        on_event: Callable[[ObservabilityEvent], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str, dict[str, Any]], None] | None = None,
        on_violation: "Callable[[GuardrailViolation], None] | None" = None,
        context: dict[str, Any] | None = None,
        buffer_tool_calls: bool = False,
        continue_from_last_good_token: ContinuationConfig | bool = False,
        build_continuation_prompt: Callable[[str], str] | None = None,
    ) -> None:
        self.fallbacks = fallbacks
        self.guardrails = guardrails
        self.retry = retry if retry is not None else Retry.recommended()
        self.timeout = timeout
        self.adapter = adapter
        self.on_event = on_event
        self.on_token = on_token
        self.on_tool_call = on_tool_call
        self.on_violation = on_violation
        self.context = context
        self.buffer_tool_calls = buffer_tool_calls
        self.continue_from_last_good_token = continue_from_last_good_token
        self.build_continuation_prompt = build_continuation_prompt


class WrappedClient:
    """Wrapped OpenAI/LiteLLM client with L0 reliability.

    Usage:
        ```python
        import l0
        from openai import AsyncOpenAI

        # Basic wrapping
        client = l0.wrap(AsyncOpenAI())

        # With configuration
        client = l0.wrap(
            AsyncOpenAI(),
            retry=l0.Retry(max_attempts=5),
            guardrails=l0.Guardrails.recommended(),
        )

        # Use normally
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        # Iterate over L0 Stream
        async for event in response:
            if event.is_token:
                print(event.text, end="")

        # Or read all at once
        text = await response.read()
        ```
    """

    __slots__ = ("_client", "_config", "chat")

    def __init__(self, client: Any, config: ClientConfig) -> None:
        self._client = client
        self._config = config

        # Wrap chat namespace if it exists
        if hasattr(client, "chat"):
            self.chat = WrappedChat(self, client.chat, config)

    @property
    def unwrapped(self) -> Any:
        """Access the underlying unwrapped client."""
        return self._client

    def with_options(
        self,
        *,
        fallbacks: list[Callable[[], Any]] | None = None,
        guardrails: list[GuardrailRule] | None = None,
        retry: Retry | None = None,
        timeout: Timeout | None = None,
        adapter: "Adapter | str | None" = None,
        on_event: Callable[[ObservabilityEvent], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str, dict[str, Any]], None] | None = None,
        on_violation: "Callable[[GuardrailViolation], None] | None" = None,
        context: dict[str, Any] | None = None,
        buffer_tool_calls: bool | None = None,
        continue_from_last_good_token: ContinuationConfig | bool | None = None,
        build_continuation_prompt: Callable[[str], str] | None = None,
    ) -> WrappedClient:
        """Create a new wrapped client with updated options.

        Returns a new WrappedClient with merged configuration.
        Unspecified options inherit from the current client.
        """
        new_config = ClientConfig(
            fallbacks=fallbacks if fallbacks is not None else self._config.fallbacks,
            guardrails=guardrails
            if guardrails is not None
            else self._config.guardrails,
            retry=retry if retry is not None else self._config.retry,
            timeout=timeout if timeout is not None else self._config.timeout,
            adapter=adapter if adapter is not None else self._config.adapter,
            on_event=on_event if on_event is not None else self._config.on_event,
            on_token=on_token if on_token is not None else self._config.on_token,
            on_tool_call=on_tool_call
            if on_tool_call is not None
            else self._config.on_tool_call,
            on_violation=on_violation
            if on_violation is not None
            else self._config.on_violation,
            context=context if context is not None else self._config.context,
            buffer_tool_calls=buffer_tool_calls
            if buffer_tool_calls is not None
            else self._config.buffer_tool_calls,
            continue_from_last_good_token=continue_from_last_good_token
            if continue_from_last_good_token is not None
            else self._config.continue_from_last_good_token,
            build_continuation_prompt=build_continuation_prompt
            if build_continuation_prompt is not None
            else self._config.build_continuation_prompt,
        )
        return WrappedClient(self._client, new_config)


def wrap_client(
    client: Any,
    *,
    fallbacks: list[Callable[[], Any]] | None = None,
    guardrails: "list[GuardrailRule] | None" = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: "Adapter | str | None" = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    on_tool_call: Callable[[str, str, dict[str, Any]], None] | None = None,
    on_violation: "Callable[[GuardrailViolation], None] | None" = None,
    context: dict[str, Any] | None = None,
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: ContinuationConfig | bool = True,
    build_continuation_prompt: Callable[[str], str] | None = None,
) -> WrappedClient:
    """Wrap an OpenAI or LiteLLM client with L0 reliability.

    Args:
        client: AsyncOpenAI, OpenAI, or LiteLLM client instance
        fallbacks: Optional fallback stream factories
        guardrails: Optional guardrail rules
        retry: Retry configuration (default: Retry.recommended())
        timeout: Timeout configuration
        adapter: Adapter hint ("openai", "litellm", or Adapter instance)
        on_event: Observability event callback
        on_token: Callback for each token received (text: str)
        on_tool_call: Callback for tool calls (name: str, id: str, args: dict)
        on_violation: Callback for guardrail violations
        context: User context attached to all events (request_id, tenant, etc.)
        buffer_tool_calls: Buffer tool calls until complete
        continue_from_last_good_token: Resume from checkpoint on retry (default: True)
        build_continuation_prompt: Callback to modify prompt for continuation

    Returns:
        WrappedClient with L0 reliability features

    Example:
        ```python
        import l0
        from openai import AsyncOpenAI

        client = l0.wrap(AsyncOpenAI())

        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        text = await response.read()
        ```
    """
    config = ClientConfig(
        fallbacks=fallbacks,
        guardrails=guardrails,
        retry=retry,
        timeout=timeout,
        adapter=adapter,
        on_event=on_event,
        on_token=on_token,
        on_tool_call=on_tool_call,
        on_violation=on_violation,
        context=context,
        buffer_tool_calls=buffer_tool_calls,
        continue_from_last_good_token=continue_from_last_good_token,
        build_continuation_prompt=build_continuation_prompt,
    )
    return WrappedClient(client, config)
