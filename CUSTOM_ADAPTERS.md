# Custom Adapters (BYOA - Bring Your Own Adapter)

L0 supports custom adapters for integrating any LLM provider or streaming source. This guide covers everything you need to build production-ready adapters.

## Adapter Scope

L0 provides **official first-party adapters** for:

- **OpenAI SDK** - `OpenAIAdapter` (also works with LiteLLM)
- **LiteLLM** - Uses OpenAI-compatible format via `LiteLLMAdapter` (alias for OpenAI)

These are the only integrations maintained within the core project.
Support for additional providers is out of scope - use custom adapters.

---

## Table of Contents

- [Overview](#overview)
- [The Adapter Protocol](#the-adapter-protocol)
- [Usage Modes](#usage-modes)
- [Building Adapters](#building-adapters)
- [Adapter Invariants](#adapter-invariants)
- [Helper Functions](#helper-functions)
- [Adapter Registry](#adapter-registry)
- [Built-in Adapters](#built-in-adapters)
- [Complete Examples](#complete-examples)
- [Testing Adapters](#testing-adapters)
- [Best Practices](#best-practices)

## Overview

Adapters convert provider-specific streams into L0's unified event format. L0 handles all reliability concerns (retries, timeouts, guardrails), so adapters can focus purely on format conversion.

```
Provider Stream → Adapter → L0 Events → L0 Runtime → Reliable Output
```

L0 ships with built-in support for:

- **OpenAI SDK** - `OpenAIAdapter`
- **LiteLLM** - `LiteLLMAdapter` (uses OpenAI-compatible format)

For other providers, create a custom adapter.

## The Adapter Protocol

```python
from typing import Any, Protocol, AsyncIterator
from l0 import Event
from l0.adapters import AdaptedEvent

class Adapter(Protocol):
    """Protocol for stream adapters."""
    
    name: str
    """Unique identifier for this adapter."""

    def detect(self, stream: Any) -> bool:
        """Check if this adapter can handle the given stream.
        
        Optional - only required for auto-detection via Adapters.register().
        Not needed for explicit `adapter=my_adapter` usage.
        
        Args:
            stream: The stream to check
            
        Returns:
            True if this adapter can handle the stream
        """
        ...

    async def wrap(
        self,
        stream: AsyncIterator[Any],
        options: Any | None = None,
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        """Wrap raw stream into AdaptedEvent stream.
        
        Yields AdaptedEvent objects containing both the normalized Event
        and the original raw chunk for provider-specific access.
        
        Args:
            stream: The raw provider stream
            options: Optional adapter-specific options
            
        Yields:
            AdaptedEvent objects with normalized events and raw chunks
        """
        ...
```

### Event Types

```python
from l0 import Event, EventType

# Event types
class EventType(str, Enum):
    TOKEN = "token"        # Text content
    MESSAGE = "message"    # Structured message (tool calls, etc.)
    DATA = "data"          # Multimodal data (images, audio, etc.)
    PROGRESS = "progress"  # Progress updates
    TOOL_CALL = "tool_call"  # Tool call
    ERROR = "error"        # Error
    COMPLETE = "complete"  # Stream complete

# Event dataclass
@dataclass
class Event:
    type: EventType
    text: str | None = None           # Token content
    data: dict[str, Any] | None = None  # Tool call / misc data
    payload: DataPayload | None = None  # Multimodal data payload
    progress: Progress | None = None    # Progress update
    error: Exception | None = None      # Error (for error events)
    usage: dict[str, int] | None = None  # Token usage
    timestamp: float | None = None      # Event timestamp
```

### Multimodal Data Types

```python
from l0 import DataPayload, ContentType, Progress

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    JSON = "json"
    BINARY = "binary"

@dataclass
class DataPayload:
    content_type: ContentType
    mime_type: str | None = None
    base64: str | None = None
    url: str | None = None
    data: bytes | None = None
    json: Any | None = None
    metadata: dict[str, Any] | None = None

@dataclass
class Progress:
    percent: float | None = None
    step: int | None = None
    total_steps: int | None = None
    message: str | None = None
    eta: float | None = None
```

### AdaptedEvent

```python
from l0.adapters import AdaptedEvent

@dataclass
class AdaptedEvent(Generic[ChunkT]):
    """Event with associated raw chunk from the provider."""
    event: Event
    raw_chunk: ChunkT | None = None
```

## Usage Modes

### 1. Explicit Adapter (Recommended)

Pass the adapter directly. No `detect()` needed.

```python
import l0
from l0 import Adapters
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Option 1: Use the Adapters class
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ),
    adapter=Adapters.openai(),
)

# Option 2: Pass adapter by name
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ),
    adapter="openai",
)
```

### 2. Auto-Detection

L0 auto-detects the adapter based on the stream type:

```python
import l0
from openai import AsyncOpenAI

client = AsyncOpenAI()

# L0 auto-detects OpenAI stream format
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ),
    # No adapter specified - auto-detected!
)
```

### 3. Register Custom Adapter

Register adapters for auto-detection:

```python
from l0 import Adapters

# Register at startup
Adapters.register(my_custom_adapter)

# Now L0 can auto-detect your custom streams
result = await l0.run(
    stream=lambda: my_custom_provider.stream("Hello!"),
    # No adapter needed - auto-detected via registry
)
```

### Stream Resolution Order

When L0 receives a stream, it resolves the adapter in this order:

1. **Explicit adapter object** - `adapter=my_adapter`
2. **Adapter by name** - `adapter="openai"` → lookup in registry
3. **Auto-detection** - Call `detect()` on registered adapters

## Building Adapters

### Minimal Adapter

```python
from typing import Any, AsyncIterator
from l0 import Event, EventType
from l0.adapters import AdaptedEvent

class MyAdapter:
    """Adapter for my custom LLM provider."""
    
    name = "my-provider"
    
    def detect(self, stream: Any) -> bool:
        """Check if this is a MyProvider stream."""
        if not hasattr(stream, "__aiter__"):
            return False
        # Check for provider-specific markers
        return hasattr(stream, "__my_provider_stream__")
    
    async def wrap(
        self,
        stream: AsyncIterator[Any],
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        """Convert MyProvider stream to L0 events."""
        try:
            async for chunk in stream:
                if chunk.text:
                    yield AdaptedEvent(
                        event=Event(type=EventType.TOKEN, text=chunk.text),
                        raw_chunk=chunk,
                    )
            
            yield AdaptedEvent(
                event=Event(type=EventType.COMPLETE),
                raw_chunk=None,
            )
        except Exception as e:
            yield AdaptedEvent(
                event=Event(type=EventType.ERROR, error=e),
                raw_chunk=None,
            )

# Create instance
my_adapter = MyAdapter()
```

### Adapter with Options

```python
from dataclasses import dataclass
from typing import Any, AsyncIterator
from l0 import Event, EventType
from l0.adapters import AdaptedEvent

@dataclass
class MyAdapterOptions:
    """Options for MyAdapter."""
    include_usage: bool = True
    include_metadata: bool = False

class MyAdapter:
    name = "my-provider"
    
    def detect(self, stream: Any) -> bool:
        return hasattr(stream, "__my_provider_stream__")
    
    async def wrap(
        self,
        stream: AsyncIterator[Any],
        options: MyAdapterOptions | None = None,
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        opts = options or MyAdapterOptions()
        usage = None
        
        async for chunk in stream:
            if chunk.text:
                yield AdaptedEvent(
                    event=Event(type=EventType.TOKEN, text=chunk.text),
                    raw_chunk=chunk,
                )
            
            # Track usage if enabled
            if opts.include_usage and hasattr(chunk, "usage"):
                usage = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
        
        yield AdaptedEvent(
            event=Event(type=EventType.COMPLETE, usage=usage),
            raw_chunk=None,
        )
```

### Using Helper Functions

L0 provides helpers to simplify adapter creation:

```python
from l0 import Adapters, Event, EventType

class SimpleAdapter:
    name = "simple"
    
    def detect(self, stream: Any) -> bool:
        return hasattr(stream, "__simple_stream__")
    
    async def wrap(self, stream, options=None):
        # Use the to_l0_events helper
        async for event in Adapters.to_l0_events(
            stream,
            extract_text=lambda chunk: chunk.text if hasattr(chunk, "text") else None,
        ):
            yield AdaptedEvent(event=event, raw_chunk=None)
```

## Adapter Invariants

Adapters MUST follow these rules. L0 depends on them for reliability.

### MUST Do

| Requirement                    | Description                                        |
| ------------------------------ | -------------------------------------------------- |
| **Preserve text exactly**      | Never trim, modify, or transform text content      |
| **Emit events in order**       | Yield events in exact order received from provider |
| **Convert errors to events**   | Catch all errors, yield `Event(type=ERROR)`        |
| **Emit complete exactly once** | Always yield `Event(type=COMPLETE)` at stream end  |
| **Be synchronous iteration**   | Only async operation is `async for` on the stream  |

### MUST NOT Do

| Forbidden            | Reason                                          |
| -------------------- | ----------------------------------------------- |
| **Modify text**      | L0 guardrails need exact text for validation    |
| **Buffer chunks**    | Breaks streaming, L0 handles batching if needed |
| **Retry internally** | L0 handles all retry logic                      |
| **Throw exceptions** | Convert to error events instead                 |
| **Skip chunks**      | Unless they contain no text (metadata-only)     |
| **Perform I/O**      | No HTTP calls, file reads, etc.                 |

### Example: Correct vs Incorrect

```python
# WRONG - modifies text
yield AdaptedEvent(
    event=Event(type=EventType.TOKEN, text=chunk.text.strip()),
    raw_chunk=chunk,
)

# CORRECT - preserves text exactly
yield AdaptedEvent(
    event=Event(type=EventType.TOKEN, text=chunk.text),
    raw_chunk=chunk,
)

# WRONG - raises on error
if chunk.error:
    raise Exception(chunk.error)

# CORRECT - converts to error event
if chunk.error:
    yield AdaptedEvent(
        event=Event(type=EventType.ERROR, error=Exception(chunk.error)),
        raw_chunk=chunk,
    )
    return
```

## Helper Functions

L0 provides helpers via the `Adapters` class to make building correct adapters easier.

### to_l0_events

The simplest way to build an adapter:

```python
from l0 import Adapters

async def my_adapter_wrap(stream):
    async for event in Adapters.to_l0_events(
        stream,
        extract_text=lambda chunk: chunk.text if chunk.text else None,
    ):
        yield event
```

`to_l0_events` handles:

- Error conversion to error events
- Automatic complete event emission
- None/null filtering

### to_l0_events_with_messages

For streams with both text and structured messages (tool calls, etc.):

```python
from l0 import Adapters

async def tool_adapter_wrap(stream):
    async for event in Adapters.to_l0_events_with_messages(
        stream,
        extract_text=lambda c: c.content if c.type == "text" else None,
        extract_message=lambda c: {"value": c.tool_call} if c.type == "tool" else None,
    ):
        yield event
```

### to_multimodal_l0_events

For streams with multimodal content (images, audio, etc.):

```python
from l0 import Adapters, DataPayload, ContentType, Progress

async def image_adapter_wrap(stream):
    async for event in Adapters.to_multimodal_l0_events(
        stream,
        extract_text=lambda c: c.text if c.type == "text" else None,
        extract_data=lambda c: DataPayload(
            content_type=ContentType.IMAGE,
            mime_type="image/png",
            base64=c.image,
            metadata={"width": c.width, "height": c.height},
        ) if c.type == "image" else None,
        extract_progress=lambda c: Progress(
            percent=c.percent,
            message=c.status,
        ) if c.type == "progress" else None,
    ):
        yield event
```

### Event Creation Helpers

For manual adapter implementations:

```python
from l0 import Adapters, DataPayload, ContentType

# Create events using Adapters helpers
token_event = Adapters.token_event("Hello")
complete_event = Adapters.complete_event(usage={"output_tokens": 100})
error_event = Adapters.error_event(Exception("Stream failed"))

# Multimodal events
image_event = Adapters.image_event(
    base64="...",
    mime_type="image/png",
    width=512,
    height=512,
)

audio_event = Adapters.audio_event(
    base64="...",
    mime_type="audio/mp3",
    duration=30.5,
)

# Generic data event
data_event = Adapters.data_event(DataPayload(
    content_type=ContentType.JSON,
    json={"key": "value"},
))

# Progress event
progress_event = Adapters.progress_event(Progress(
    percent=50,
    message="Processing...",
))
```

## Adapter Registry

### Registering Adapters

```python
from l0 import Adapters

# Register for auto-detection
Adapters.register(my_adapter)

# Silence warning for adapters without detect()
Adapters.register(adapter_without_detect, silent=True)

# Unregister by name
Adapters.unregister("my-provider")

# Unregister all adapters except specified ones (useful for testing)
removed = Adapters.unregister_all_except(["openai"])
print(removed)  # ["my-provider", "other"]

# Clear all (useful in tests)
Adapters.clear()

# Reset to defaults
Adapters.reset()
```

### Registry Functions

| Function                                | Description                                   |
| --------------------------------------- | --------------------------------------------- |
| `Adapters.register(adapter, silent=False)` | Register for auto-detection                |
| `Adapters.unregister(name)`             | Remove by name                                |
| `Adapters.unregister_all_except(names)` | Remove all adapters except those in the list  |
| `Adapters.get(name)`                    | Get adapter by name                           |
| `Adapters.registered()`                 | List all registered adapter names             |
| `Adapters.clear()`                      | Remove all adapters                           |
| `Adapters.reset()`                      | Reset to default adapters                     |
| `Adapters.detect(stream, hint=None)`    | Detect or lookup adapter for stream           |
| `Adapters.detect_adapter(stream)`       | Auto-detect adapter (returns None if not found) |
| `Adapters.has_matching(stream)`         | Check if exactly one adapter matches          |

### DX Warning

In development mode, registering an adapter without `detect()` logs a warning:

```
UserWarning: Adapter "my-provider" has no detect() method.
It will not be used for auto-detection.
Use explicit `adapter=myAdapter` instead, or add a detect() method.
```

Suppress with `silent=True` or set `NODE_ENV=production`.

## Built-in Adapters

### OpenAI Adapter

```python
import l0
from l0 import Adapters
from l0.adapters import OpenAIAdapter, OpenAIAdapterOptions
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Option 1: Explicit adapter via Adapters class
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ),
    adapter=Adapters.openai(),
)

# Option 2: Adapter by name
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ),
    adapter="openai",
)

# Option 3: Auto-detection (OpenAI streams are auto-detected)
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ),
)

# Option 4: Use l0.wrap() for wrapped client
wrapped_client = l0.wrap(client)
response = await wrapped_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
text = await response.read()
```

#### OpenAI Adapter Options

```python
from l0.adapters import OpenAIAdapterOptions

@dataclass
class OpenAIAdapterOptions:
    include_usage: bool = True           # Include usage in complete event
    include_tool_calls: bool = True      # Include tool calls as events
    emit_function_calls_as_tokens: bool = False  # Emit function args as tokens
    choice_index: int | str = 0          # Which choice to use when n > 1 (0 or "all")
```

### LiteLLM Adapter

LiteLLM uses OpenAI-compatible format, so the same adapter works:

```python
import l0
import litellm

# LiteLLM streams are auto-detected (uses OpenAI format)
result = await l0.run(
    stream=lambda: litellm.acompletion(
        model="anthropic/claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ),
)

# Or explicitly
result = await l0.run(
    stream=lambda: litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ),
    adapter="litellm",  # Alias for "openai"
)
```

## Complete Examples

### Custom Provider Adapter

```python
from dataclasses import dataclass
from typing import Any, AsyncIterator
from l0 import Event, EventType
from l0.adapters import AdaptedEvent, Adapters

# Define the provider's stream types
@dataclass
class CustomProviderChunk:
    type: str  # "text", "metadata", "end"
    content: str | None = None
    tokens: int | None = None

class CustomProviderAdapter:
    """Adapter for Custom Provider streams."""
    
    name = "custom-provider"
    
    def detect(self, stream: Any) -> bool:
        """Detect Custom Provider streams."""
        if not hasattr(stream, "__aiter__"):
            return False
        # Check for provider-specific marker
        return hasattr(stream, "__custom_provider__")
    
    async def wrap(
        self,
        stream: AsyncIterator[CustomProviderChunk],
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[CustomProviderChunk]]:
        """Convert Custom Provider stream to L0 events."""
        try:
            async for chunk in stream:
                if chunk.type == "text" and chunk.content:
                    yield AdaptedEvent(
                        event=Event(type=EventType.TOKEN, text=chunk.content),
                        raw_chunk=chunk,
                    )
                # Skip non-text chunks (metadata, etc.)
            
            yield AdaptedEvent(
                event=Event(type=EventType.COMPLETE),
                raw_chunk=None,
            )
        except Exception as e:
            yield AdaptedEvent(
                event=Event(type=EventType.ERROR, error=e),
                raw_chunk=None,
            )

# Register the adapter
custom_adapter = CustomProviderAdapter()
Adapters.register(custom_adapter)
```

### Adapter with Tool Support

Custom adapters that emit tool calls should use L0's format:

```python
from dataclasses import dataclass
from typing import Any, AsyncIterator
import json
from l0 import Event, EventType
from l0.adapters import AdaptedEvent

@dataclass
class ToolProviderChunk:
    type: str  # "text", "tool_call", "tool_result", "complete"
    text: str | None = None
    tool: dict | None = None  # {"id": str, "name": str, "arguments": dict}
    result: dict | None = None  # {"id": str, "output": Any, "error": str | None}

class ToolProviderAdapter:
    name = "tool-provider"
    
    def detect(self, stream: Any) -> bool:
        return hasattr(stream, "__tool_provider__")
    
    async def wrap(
        self,
        stream: AsyncIterator[ToolProviderChunk],
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[ToolProviderChunk]]:
        try:
            async for chunk in stream:
                if chunk.type == "text" and chunk.text:
                    yield AdaptedEvent(
                        event=Event(type=EventType.TOKEN, text=chunk.text),
                        raw_chunk=chunk,
                    )
                
                elif chunk.type == "tool_call" and chunk.tool:
                    # Emit as TOOL_CALL event
                    yield AdaptedEvent(
                        event=Event(
                            type=EventType.TOOL_CALL,
                            data={
                                "id": chunk.tool["id"],
                                "name": chunk.tool["name"],
                                "arguments": json.dumps(chunk.tool["arguments"]),
                            },
                        ),
                        raw_chunk=chunk,
                    )
                
                elif chunk.type == "complete":
                    yield AdaptedEvent(
                        event=Event(type=EventType.COMPLETE),
                        raw_chunk=chunk,
                    )
                    return
            
            # Ensure complete is emitted
            yield AdaptedEvent(
                event=Event(type=EventType.COMPLETE),
                raw_chunk=None,
            )
        except Exception as e:
            yield AdaptedEvent(
                event=Event(type=EventType.ERROR, error=e),
                raw_chunk=None,
            )
```

### Multimodal Adapter (Image Generation)

```python
from dataclasses import dataclass
from typing import Any, AsyncIterator
from l0 import Event, EventType, DataPayload, ContentType, Progress
from l0.adapters import AdaptedEvent

@dataclass
class ImageGenChunk:
    type: str  # "progress", "image", "complete"
    percent: float | None = None
    message: str | None = None
    image: str | None = None  # base64
    width: int | None = None
    height: int | None = None
    seed: int | None = None

class ImageGenAdapter:
    name = "image-gen"
    
    def detect(self, stream: Any) -> bool:
        return hasattr(stream, "__image_gen__")
    
    async def wrap(
        self,
        stream: AsyncIterator[ImageGenChunk],
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[ImageGenChunk]]:
        try:
            async for chunk in stream:
                if chunk.type == "progress":
                    yield AdaptedEvent(
                        event=Event(
                            type=EventType.PROGRESS,
                            progress=Progress(
                                percent=chunk.percent,
                                message=chunk.message,
                            ),
                        ),
                        raw_chunk=chunk,
                    )
                
                elif chunk.type == "image" and chunk.image:
                    yield AdaptedEvent(
                        event=Event(
                            type=EventType.DATA,
                            payload=DataPayload(
                                content_type=ContentType.IMAGE,
                                mime_type="image/png",
                                base64=chunk.image,
                                metadata={
                                    "width": chunk.width,
                                    "height": chunk.height,
                                    "seed": chunk.seed,
                                },
                            ),
                        ),
                        raw_chunk=chunk,
                    )
            
            yield AdaptedEvent(
                event=Event(type=EventType.COMPLETE),
                raw_chunk=None,
            )
        except Exception as e:
            yield AdaptedEvent(
                event=Event(type=EventType.ERROR, error=e),
                raw_chunk=None,
            )
```

### Wrapping a REST API with SSE

```python
from typing import Any, AsyncIterator
import json
from l0 import Event, EventType
from l0.adapters import AdaptedEvent

async def parse_sse(response) -> AsyncIterator[dict]:
    """Parse SSE stream from httpx response."""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return
            yield json.loads(data)

class RestApiAdapter:
    name = "rest-api"
    
    def detect(self, stream: Any) -> bool:
        # Check for httpx response
        return hasattr(stream, "aiter_lines")
    
    async def wrap(
        self,
        response: Any,
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[dict]]:
        try:
            async for message in parse_sse(response):
                if "text" in message:
                    yield AdaptedEvent(
                        event=Event(type=EventType.TOKEN, text=message["text"]),
                        raw_chunk=message,
                    )
            
            yield AdaptedEvent(
                event=Event(type=EventType.COMPLETE),
                raw_chunk=None,
            )
        except Exception as e:
            yield AdaptedEvent(
                event=Event(type=EventType.ERROR, error=e),
                raw_chunk=None,
            )

# Usage
import httpx
import l0

rest_adapter = RestApiAdapter()

result = await l0.run(
    stream=lambda: httpx.AsyncClient().stream(
        "POST",
        "https://api.example.com/stream",
        json={"prompt": "Hello!"},
    ),
    adapter=rest_adapter,
)
```

## Testing Adapters

### Unit Test Structure

```python
import pytest
from l0 import Event, EventType
from l0.adapters import AdaptedEvent, Adapters

# Helper to collect events
async def collect_events(adapter, stream) -> list[Event]:
    events = []
    async for adapted in adapter.wrap(stream):
        events.append(adapted.event)
    return events

# Helper to create mock stream
async def mock_stream(chunks):
    for chunk in chunks:
        yield chunk

class TestMyAdapter:
    @pytest.fixture
    def adapter(self):
        return MyAdapter()
    
    @pytest.fixture(autouse=True)
    def reset_adapters(self):
        """Reset adapter registry between tests."""
        Adapters.reset()
        yield
        Adapters.reset()
    
    @pytest.mark.asyncio
    async def test_preserves_exact_text(self, adapter):
        """Text content must be preserved exactly."""
        stream = mock_stream([
            MockChunk(text="  Hello  "),
            MockChunk(text="\n\nWorld\n\n"),
        ])
        
        events = await collect_events(adapter, stream)
        
        assert events[0].type == EventType.TOKEN
        assert events[0].text == "  Hello  "
        assert events[1].type == EventType.TOKEN
        assert events[1].text == "\n\nWorld\n\n"
    
    @pytest.mark.asyncio
    async def test_emits_complete_once(self, adapter):
        """Complete event must be emitted exactly once."""
        stream = mock_stream([
            MockChunk(text="A"),
            MockChunk(text="B"),
        ])
        
        events = await collect_events(adapter, stream)
        
        complete_events = [e for e in events if e.type == EventType.COMPLETE]
        assert len(complete_events) == 1
    
    @pytest.mark.asyncio
    async def test_converts_errors_to_events(self, adapter):
        """Errors must become error events, not exceptions."""
        async def error_stream():
            yield MockChunk(text="Hello")
            raise Exception("Stream failed")
        
        events = await collect_events(adapter, error_stream())
        
        assert events[0].type == EventType.TOKEN
        assert events[0].text == "Hello"
        assert events[1].type == EventType.ERROR
        assert str(events[1].error) == "Stream failed"
    
    @pytest.mark.asyncio
    async def test_detection(self, adapter):
        """Detection must correctly identify valid streams."""
        valid_stream = create_my_stream()
        invalid_stream = object()
        
        assert adapter.detect(valid_stream) is True
        assert adapter.detect(invalid_stream) is False
        assert adapter.detect(None) is False
```

### Key Test Cases

1. **Text preservation** - Exact text including whitespace, newlines, special chars
2. **Complete event** - Emitted exactly once at end
3. **Error handling** - Errors become error events, never thrown
4. **Event ordering** - Events emitted in receive order
5. **Empty streams** - Still emit complete event
6. **Detection** - Type guard returns correct boolean

## Best Practices

### DO

- Use `Adapters.to_l0_events()` helper when possible
- Test with various chunk shapes from your provider
- Handle all edge cases (empty text, missing fields)
- Keep `detect()` fast and synchronous
- Document provider-specific behavior
- Return `AdaptedEvent` with raw chunks for provider-specific access

### DON'T

- Don't trim or normalize text
- Don't add artificial delays
- Don't buffer chunks for batching
- Don't make HTTP calls in `wrap()`
- Don't assume chunk structure without checking
- Don't throw exceptions - convert to error events

### Performance Tips

1. **Avoid allocations in hot path** - Reuse objects where possible
2. **Keep detect() O(1)** - Only check object properties
3. **Don't parse JSON unnecessarily** - Pass through raw text
4. **Let L0 handle batching** - Yield events immediately

### Error Messages

Provide helpful detection:

```python
def detect(self, stream: Any) -> bool:
    if not stream or not hasattr(stream, "__aiter__"):
        return False
    if not hasattr(stream, "__my_marker__"):
        return False
    return True
```

If detection fails, L0 shows:

```
ValueError: No adapter found for stream
```
