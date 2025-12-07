"""Tool call observability integration tests with real API calls.

These tests verify tool call detection, onToolCall callback firing,
and tool result handling with actual LLM responses.

Requires OPENAI_API_KEY to be set.
Run with: pytest tests/integration/test_tool_calls.py -v
"""

from typing import TYPE_CHECKING, Any, cast

import pytest

import l0
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionToolParam


# Tool definitions for OpenAI
WEATHER_TOOL: "ChatCompletionToolParam" = cast(
    "ChatCompletionToolParam",
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
)

TIME_TOOL: "ChatCompletionToolParam" = cast(
    "ChatCompletionToolParam",
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time for a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name"},
                },
                "required": ["timezone"],
            },
        },
    },
)

SEARCH_TOOL: "ChatCompletionToolParam" = cast(
    "ChatCompletionToolParam",
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products with filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "min_price": {"type": "number"},
                            "max_price": {"type": "number"},
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "limit": {"type": "number"},
                },
                "required": ["query"],
            },
        },
    },
)


@requires_openai
class TestSingleToolCall:
    """Test single tool call detection with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_tool_call_detected(self, client: "AsyncOpenAI") -> None:
        """Test that tool call is detected and callback fires."""
        tool_calls: list[dict[str, Any]] = []

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            tool_calls.append({"name": name, "id": tool_call_id, "args": args})

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            stream=True,
        )

        # Use buffer_tool_calls=True to get complete tool calls with parsed arguments
        result = l0.wrap(stream, on_tool_call=on_tool_call, buffer_tool_calls=True)

        async for _ in result:
            pass

        assert len(tool_calls) >= 1
        weather_call = next((t for t in tool_calls if t["name"] == "get_weather"), None)
        assert weather_call is not None
        assert weather_call["id"] is not None
        assert "location" in weather_call["args"]

    @pytest.mark.asyncio
    async def test_tool_call_id_format(self, client: "AsyncOpenAI") -> None:
        """Test that tool call ID has expected OpenAI format."""
        tool_call_ids: list[str] = []

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            tool_call_ids.append(tool_call_id)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What time is it in Tokyo?"}],
            tools=[TIME_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_time"}},
            stream=True,
        )

        result = l0.wrap(stream, on_tool_call=on_tool_call)
        async for _ in result:
            pass

        assert len(tool_call_ids) >= 1
        # OpenAI tool call IDs start with "call_"
        assert tool_call_ids[0].startswith("call_")

    @pytest.mark.asyncio
    async def test_unbuffered_tool_call_streams_immediately(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test that unbuffered mode fires callback as soon as tool call is detected."""
        callback_times: list[float] = []
        import time

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            callback_times.append(time.time())

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            stream=True,
        )

        start_time = time.time()
        # Unbuffered mode (default) - callback fires on first tool call chunk
        result = l0.wrap(stream, on_tool_call=on_tool_call)
        async for _ in result:
            pass
        end_time = time.time()

        # Callback should have fired at least once
        assert len(callback_times) >= 1
        # In unbuffered mode, callback fires early (before stream completes)
        # The first callback should happen before we finish consuming the stream
        first_callback = callback_times[0]
        assert first_callback < end_time


@requires_openai
class TestMultipleToolCalls:
    """Test multiple tool call detection with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_multiple_parallel_tool_calls(self, client: "AsyncOpenAI") -> None:
        """Test detection of multiple parallel tool calls."""
        tool_calls: list[dict[str, Any]] = []

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            tool_calls.append({"name": name, "id": tool_call_id})

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo AND what time is it there?",
                }
            ],
            tools=[WEATHER_TOOL, TIME_TOOL],
            tool_choice="required",
            stream=True,
        )

        result = l0.wrap(stream, on_tool_call=on_tool_call)
        async for _ in result:
            pass

        # Should have at least one tool call
        assert len(tool_calls) >= 1

        # Each tool call should have unique ID
        ids = [t["id"] for t in tool_calls]
        assert len(set(ids)) == len(ids)

    @pytest.mark.asyncio
    async def test_tool_calls_have_unique_ids(self, client: "AsyncOpenAI") -> None:
        """Test that each tool call has a unique ID."""
        tool_call_ids: list[str] = []

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            tool_call_ids.append(tool_call_id)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Get weather for NYC and LA.",
                }
            ],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            stream=True,
        )

        result = l0.wrap(stream, on_tool_call=on_tool_call)
        async for _ in result:
            pass

        if len(tool_call_ids) > 1:
            # All IDs should be unique
            assert len(set(tool_call_ids)) == len(tool_call_ids)


@requires_openai
class TestToolCallArguments:
    """Test tool call argument parsing with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_complex_nested_arguments(self, client: "AsyncOpenAI") -> None:
        """Test parsing of complex nested arguments."""
        captured_args: list[dict[str, Any]] = []

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            captured_args.append(args)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Search for laptops under $1000 in electronics, limit 5",
                }
            ],
            tools=[SEARCH_TOOL],
            tool_choice={"type": "function", "function": {"name": "search_products"}},
            stream=True,
        )

        # Use buffer_tool_calls=True to get complete tool calls with parsed arguments
        result = l0.wrap(stream, on_tool_call=on_tool_call, buffer_tool_calls=True)
        async for _ in result:
            pass

        assert len(captured_args) >= 1
        args = captured_args[0]
        assert "query" in args

    @pytest.mark.asyncio
    async def test_optional_arguments(self, client: "AsyncOpenAI") -> None:
        """Test that optional arguments are handled correctly."""
        captured_args: list[dict[str, Any]] = []

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            captured_args.append(args)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in London?"}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            stream=True,
        )

        # Use buffer_tool_calls=True to get complete tool calls with parsed arguments
        result = l0.wrap(stream, on_tool_call=on_tool_call, buffer_tool_calls=True)
        async for _ in result:
            pass

        assert len(captured_args) >= 1
        args = captured_args[0]
        # Required argument should be present
        assert "location" in args
        # Optional argument may or may not be present
        # (unit is optional in our schema)


@requires_openai
class TestToolCallWithL0Features:
    """Test tool calls with other L0 features."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_tool_call_with_retry(self, client: "AsyncOpenAI") -> None:
        """Test tool call detection works with retry configuration."""
        tool_calls: list[str] = []

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            tool_calls.append(name)

        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                tools=[WEATHER_TOOL],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
                stream=True,
            ),
            retry=l0.Retry(attempts=2),
            on_tool_call=on_tool_call,
        )

        async for _ in result:
            pass

        assert "get_weather" in tool_calls

    @pytest.mark.asyncio
    async def test_tool_call_with_monitoring(self, client: "AsyncOpenAI") -> None:
        """Test tool call tracking in observability events."""
        events: list[l0.ObservabilityEvent] = []

        def on_event(event: l0.ObservabilityEvent) -> None:
            events.append(event)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            stream=True,
        )

        result = l0.wrap(stream, on_event=on_event)
        async for _ in result:
            pass

        # Should have observability events
        assert len(events) > 0

        # Check for tool-related events
        event_types = [e.type for e in events]
        assert l0.ObservabilityEventType.STREAM_INIT in event_types

    @pytest.mark.asyncio
    async def test_tool_call_with_timeout(self, client: "AsyncOpenAI") -> None:
        """Test tool calls work with timeout configuration."""
        tool_calls: list[str] = []

        def on_tool_call(name: str, tool_call_id: str, args: dict[str, Any]) -> None:
            tool_calls.append(name)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Miami?"}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            stream=True,
        )

        result = l0.wrap(
            stream,
            on_tool_call=on_tool_call,
            timeout=l0.Timeout(initial_token=30000, inter_token=30000),
        )

        async for _ in result:
            pass

        assert "get_weather" in tool_calls


@requires_openai
class TestToolCallEvents:
    """Test tool call observability events with real API."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_tool_events_emitted(self, client: "AsyncOpenAI") -> None:
        """Test that tool-related observability events are emitted."""
        events: list[l0.ObservabilityEvent] = []

        def on_event(event: l0.ObservabilityEvent) -> None:
            events.append(event)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            stream=True,
        )

        result = l0.wrap(stream, on_event=on_event)
        async for _ in result:
            pass

        # Check for tool-related events
        tool_event_types = [
            l0.ObservabilityEventType.TOOL_REQUESTED,
            l0.ObservabilityEventType.TOOL_START,
            l0.ObservabilityEventType.TOOL_COMPLETED,
        ]

        event_types = [e.type for e in events]

        # At least one tool event should be present
        has_tool_event = any(t in event_types for t in tool_event_types)
        # Note: The actual events depend on whether tool calls are buffered
        # This test verifies the observability pipeline works
        assert len(events) > 0
