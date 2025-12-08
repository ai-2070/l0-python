# Multimodal Support

L0 supports multimodal AI outputs including images, audio, video, and structured data. Build adapters for image generation models like Flux, Stable Diffusion, DALL-E, or audio models like TTS.

## Quick Start

```python
from l0 import Multimodal

# Create events for different content types
image_event = Multimodal.image(
    base64="iVBORw0KGgo...",
    width=1024,
    height=768,
    model="dall-e-3",
)

audio_event = Multimodal.audio(
    url="https://example.com/audio.mp3",
    duration=30.5,
)

progress_event = Multimodal.progress(
    percent=50,
    message="Generating image...",
)

# Complete the stream
complete_event = Multimodal.complete()
```

## Event Types

L0 extends the standard event system with multimodal-specific events:

| Event Type | Description                                            |
| ---------- | ------------------------------------------------------ |
| `token`    | Text token (standard LLM streaming)                    |
| `message`  | Structured message (tool calls, etc.)                  |
| `data`     | Multimodal content (images, audio, video, files, JSON) |
| `progress` | Progress updates for long-running operations           |
| `error`    | Error event                                            |
| `complete` | Stream completion                                      |

```python
from l0.types import EventType

# Event types as enum
EventType.TOKEN      # "token"
EventType.MESSAGE    # "message"
EventType.DATA       # "data"
EventType.PROGRESS   # "progress"
EventType.ERROR      # "error"
EventType.COMPLETE   # "complete"
```

## Content Types

L0 defines content types for multimodal data:

```python
from l0.types import ContentType

ContentType.TEXT    # "text"
ContentType.IMAGE   # "image"
ContentType.AUDIO   # "audio"
ContentType.VIDEO   # "video"
ContentType.FILE    # "file"
ContentType.JSON    # "json"
ContentType.BINARY  # "binary"
```

## Data Payload

The `data` event carries a `DataPayload`:

```python
from dataclasses import dataclass
from typing import Any
from l0.types import ContentType

@dataclass
class DataPayload:
    """Multimodal data payload."""
    
    content_type: ContentType
    mime_type: str | None = None      # e.g., "image/png", "audio/mp3"
    base64: str | None = None         # Base64-encoded data
    url: str | None = None            # URL to content
    data: bytes | None = None         # Raw bytes
    json: Any | None = None           # Structured JSON data
    metadata: dict[str, Any] | None = None  # Additional metadata

    # Convenience properties
    @property
    def width(self) -> int | None: ...      # Image/video width
    @property
    def height(self) -> int | None: ...     # Image/video height
    @property
    def duration(self) -> float | None: ... # Audio/video duration (seconds)
    @property
    def size(self) -> int | None: ...       # File size in bytes
    @property
    def filename(self) -> str | None: ...   # Filename if available
    @property
    def seed(self) -> int | None: ...       # Generation seed
    @property
    def model(self) -> str | None: ...      # Model used for generation
```

### Metadata Fields

Common metadata fields stored in `metadata` dict:

| Field      | Type    | Description                          |
| ---------- | ------- | ------------------------------------ |
| `width`    | `int`   | Width in pixels (images/video)       |
| `height`   | `int`   | Height in pixels (images/video)      |
| `duration` | `float` | Duration in seconds (audio/video)    |
| `size`     | `int`   | File size in bytes                   |
| `filename` | `str`   | Original filename                    |
| `seed`     | `int`   | Generation seed (for reproducibility)|
| `model`    | `str`   | Model used for generation            |

## Progress Updates

The `progress` event carries a `Progress` dataclass:

```python
from dataclasses import dataclass

@dataclass
class Progress:
    """Progress update for long-running operations."""
    
    percent: float | None = None      # Progress percentage (0-100)
    step: int | None = None           # Current step number
    total_steps: int | None = None    # Total number of steps
    message: str | None = None        # Status message
    eta: float | None = None          # Estimated time remaining (seconds)
```

## Multimodal Class

The `Multimodal` class provides a scoped API for creating multimodal events:

### Event Creation Methods

```python
from l0 import Multimodal

# Create image event
event = Multimodal.image(
    base64="...",           # Base64-encoded image
    url="...",              # OR URL to image
    data=b"...",            # OR raw bytes
    width=1024,
    height=768,
    seed=42,
    model="flux-1.1-pro",
    mime_type="image/png",  # Default: "image/png"
    custom_field="value",   # Extra metadata via **kwargs
)

# Create audio event
event = Multimodal.audio(
    base64="...",
    url="...",
    data=b"...",
    duration=30.5,
    model="tts-1",
    mime_type="audio/mp3",  # Default: "audio/mp3"
)

# Create video event
event = Multimodal.video(
    base64="...",
    url="...",
    data=b"...",
    width=1920,
    height=1080,
    duration=60.0,
    model="sora",
    mime_type="video/mp4",  # Default: "video/mp4"
)

# Create file event
event = Multimodal.file(
    base64="...",
    url="...",
    data=b"...",
    filename="document.pdf",
    size=1024000,
    mime_type="application/pdf",
)

# Create JSON data event
event = Multimodal.json(
    json_data={"key": "value", "nested": {"data": [1, 2, 3]}},
    model="gpt-4",
)

# Create binary data event
event = Multimodal.binary(
    base64="...",
    url="...",
    data=b"...",
    mime_type="application/octet-stream",  # Default
    size=4096,
)

# Create progress event
event = Multimodal.progress(
    percent=75,
    step=3,
    total_steps=4,
    message="Finalizing...",
    eta=5.0,
)

# Create raw data event with full payload
from l0.types import DataPayload, ContentType
event = Multimodal.data(DataPayload(
    content_type=ContentType.IMAGE,
    mime_type="image/webp",
    url="https://example.com/image.webp",
))

# Create token event
event = Multimodal.token("Hello")

# Create message event
event = Multimodal.message(value="Tool response", role="assistant")

# Create complete event
event = Multimodal.complete(usage={"input_tokens": 100, "output_tokens": 50})

# Create error event
event = Multimodal.error(ValueError("Something went wrong"))
```

## Building a Multimodal Adapter

### Using to_events() Converter

The simplest way to build a multimodal adapter:

```python
from l0 import Multimodal
from l0.adapters import Adapter, AdaptedEvent
from collections.abc import AsyncIterator
from typing import Any

class FluxChunk:
    type: str  # "progress" | "image"
    percent: float | None = None
    status: str | None = None
    image: str | None = None  # base64
    width: int | None = None
    height: int | None = None
    seed: int | None = None

class FluxAdapter:
    name = "flux"
    
    def detect(self, stream: Any) -> bool:
        return hasattr(stream, "__flux__")
    
    async def wrap(
        self,
        stream: AsyncIterator[FluxChunk],
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[FluxChunk]]:
        
        def extract_progress(chunk: FluxChunk):
            if chunk.type == "progress":
                return {"percent": chunk.percent, "message": chunk.status}
            return None
        
        def extract_data(chunk: FluxChunk):
            if chunk.type == "image" and chunk.image:
                return Multimodal.image(
                    base64=chunk.image,
                    width=chunk.width,
                    height=chunk.height,
                    seed=chunk.seed,
                    model="flux-schnell",
                ).payload  # Get the DataPayload from the event
            return None
        
        async for event in Multimodal.to_events(
            stream,
            extract_progress=extract_progress,
            extract_data=extract_data,
        ):
            yield AdaptedEvent(event=event, raw_chunk=None)
```

### to_events() Handlers

The `Multimodal.to_events()` method accepts the following extractors:

```python
async for event in Multimodal.to_events(
    stream,
    # Extract text from chunk (for token events)
    extract_text=lambda chunk: chunk.text if hasattr(chunk, 'text') else None,
    
    # Extract multimodal data from chunk
    extract_data=lambda chunk: DataPayload(...) if chunk.has_data else None,
    
    # Extract progress from chunk (dict or Progress object)
    extract_progress=lambda chunk: {"percent": chunk.pct} if chunk.is_progress else None,
    
    # Extract message from chunk
    extract_message=lambda chunk: {"value": chunk.msg, "role": "assistant"} if chunk.is_msg else None,
    
    # Extract error from chunk (stops stream)
    extract_error=lambda chunk: chunk.error if chunk.has_error else None,
):
    yield event
```

Handlers are tried in order: `text` -> `data` -> `progress` -> `message`. The first handler that returns a non-null value creates the event, then processing continues to the next chunk.

### Using Manual Event Creation

For more control, create events directly:

```python
from l0 import Multimodal
from l0.adapters import AdaptedEvent
from l0.types import Event, EventType
from collections.abc import AsyncIterator

class FluxAdapter:
    name = "flux"
    
    def detect(self, stream):
        return hasattr(stream, "__flux__")
    
    async def wrap(self, stream, options=None) -> AsyncIterator[AdaptedEvent]:
        try:
            async for chunk in stream:
                if chunk.type == "progress":
                    event = Multimodal.progress(
                        percent=chunk.percent,
                        message=chunk.status,
                    )
                    yield AdaptedEvent(event=event, raw_chunk=chunk)
                    
                elif chunk.type == "image":
                    event = Multimodal.image(
                        base64=chunk.image,
                        width=chunk.width,
                        height=chunk.height,
                        seed=chunk.seed,
                        model="flux-schnell",
                    )
                    yield AdaptedEvent(event=event, raw_chunk=chunk)
            
            yield AdaptedEvent(event=Multimodal.complete(), raw_chunk=None)
            
        except Exception as e:
            yield AdaptedEvent(event=Multimodal.error(e), raw_chunk=None)
```

## Module-Level Helper Functions

For TypeScript API parity, module-level functions are also available:

```python
from l0.multimodal import (
    create_image_event,
    create_audio_event,
    create_video_event,
    create_data_event,
    create_progress_event,
    create_token_event,
    create_message_event,
    create_complete_event,
    create_error_event,
    create_json_event,
    to_multimodal_events,
    to_events,
    to_events_with_messages,
)

# These mirror the Multimodal class methods
event = create_image_event(base64="...", width=512, height=512)
event = create_audio_event(url="https://...", duration=10.0)
event = create_progress_event(percent=50, message="Processing...")
```

## Adapters Class Helpers

The `Adapters` class also provides multimodal helpers:

```python
from l0 import Adapters
from l0.types import DataPayload, Progress, ContentType

# Create events via Adapters
event = Adapters.image_event(base64="...", width=512, height=512)
event = Adapters.audio_event(url="...", duration=10.0)
event = Adapters.data_event(DataPayload(content_type=ContentType.IMAGE, ...))
event = Adapters.progress_event(Progress(percent=50))

# Stream conversion helpers
async for event in Adapters.to_multimodal_l0_events(
    stream,
    extract_data=lambda c: DataPayload(...) if c.has_image else None,
    extract_progress=lambda c: Progress(percent=c.pct) if c.is_progress else None,
):
    yield event
```

## Consuming Multimodal Streams

```python
from l0 import l0
from l0.types import ContentType

result = await l0(
    stream=lambda: flux_generate(prompt="A cat in space"),
    adapter=flux_adapter,
)

async for event in result:
    if event.is_progress:
        print(f"Progress: {event.progress.percent}%")
        
    elif event.is_data:
        if event.payload.content_type == ContentType.IMAGE:
            # Save or display the image
            image_data = event.payload.base64
            width = event.payload.width
            height = event.payload.height
            print(f"Generated {width}x{height} image")
            
    elif event.is_complete:
        print("Generation complete")

# Access all generated data from state
print(f"Total images: {len(result.state.data_outputs)}")
```

## State Tracking

L0 automatically tracks multimodal outputs in the state:

```python
from dataclasses import dataclass, field
from l0.types import DataPayload, Progress

@dataclass
class State:
    # ... existing fields ...
    
    # Multimodal state
    data_outputs: list[DataPayload] = field(default_factory=list)
    last_progress: Progress | None = None
```

After streaming completes:

```python
result = await l0(stream=..., adapter=...)

# All data payloads collected during streaming
for payload in result.state.data_outputs:
    if payload.content_type == ContentType.IMAGE:
        save_image(payload.base64, payload.filename or "output.png")

# Last progress update received
if result.state.last_progress:
    print(f"Final progress: {result.state.last_progress.percent}%")
```

## Important Notes

### Zero Token Detection

For streams that only produce `data` or `progress` events (no text tokens), disable zero token detection:

```python
result = await l0(
    stream=lambda: image_generator.generate(prompt),
    adapter=image_adapter,
    detect_zero_tokens=False,  # Required for non-text streams
)
```

By default, `detect_zero_tokens` is `True`, which will raise an error if no tokens are received. Set it to `False` for multimodal-only streams.

### Checkpoint Continuation

`continue_from_last_known_good_token` only works with text content. It has no effect on data-only streams since there's no text to checkpoint. For multimodal streams that include text, only the text portion will be checkpointed and resumed.

## Complete Example: Flux Image Generation

```python
from l0 import l0, Multimodal
from l0.adapters import AdaptedEvent
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

# Define the Flux stream types
@dataclass
class FluxChunk:
    type: str  # "queued" | "processing" | "completed" | "error"
    progress: float | None = None
    image_url: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    error: str | None = None

# Create the adapter
class FluxAdapter:
    name = "flux"
    
    def detect(self, stream: Any) -> bool:
        return hasattr(stream, "__flux__")
    
    async def wrap(
        self,
        stream: AsyncIterator[FluxChunk],
        options: Any = None,
    ) -> AsyncIterator[AdaptedEvent[FluxChunk]]:
        
        def extract_progress(chunk: FluxChunk):
            if chunk.type == "queued":
                return {"percent": 0, "message": "Queued"}
            if chunk.type == "processing":
                return {"percent": chunk.progress or 50, "message": "Generating"}
            return None
        
        def extract_data(chunk: FluxChunk):
            if chunk.type == "completed" and chunk.image_url:
                return Multimodal.image(
                    url=chunk.image_url,
                    width=chunk.width,
                    height=chunk.height,
                    seed=chunk.seed,
                    model="flux-1.1-pro",
                ).payload
            return None
        
        async for event in Multimodal.to_events(
            stream,
            extract_progress=extract_progress,
            extract_data=extract_data,
        ):
            yield AdaptedEvent(event=event, raw_chunk=None)

# Use with L0
async def generate_image(prompt: str):
    flux_adapter = FluxAdapter()
    
    result = await l0(
        stream=lambda: flux_api.generate(prompt=prompt),
        adapter=flux_adapter,
        detect_zero_tokens=False,  # Required for image-only streams
        timeout={
            "initial_token": 30000,  # 30s for queue
            "inter_token": 60000,    # 60s between updates
        },
        retry={"attempts": 2},
    )
    
    async for event in result:
        if event.is_progress:
            update_progress_bar(event.progress.percent or 0)
    
    return result.state.data_outputs[0]  # First generated image
```

## Text Stream Helpers

For simpler text-only adapters, use the basic helpers:

### to_events()

Convert a simple text stream:

```python
from l0.multimodal import to_events

async def wrap(stream):
    async for event in to_events(stream, lambda chunk: chunk.text):
        yield event
```

### to_events_with_messages()

Convert a stream with both text and messages:

```python
from l0.multimodal import to_events_with_messages

async def wrap(stream):
    async for event in to_events_with_messages(
        stream,
        extract_text=lambda c: c.content if c.type == "text" else None,
        extract_message=lambda c: {"value": c.tool_call, "role": "assistant"} if c.type == "tool" else None,
    ):
        yield event
```

## API Reference

### Multimodal Class

| Method | Description |
| ------ | ----------- |
| `Multimodal.image(...)` | Create image data event |
| `Multimodal.audio(...)` | Create audio data event |
| `Multimodal.video(...)` | Create video data event |
| `Multimodal.file(...)` | Create file data event |
| `Multimodal.json(...)` | Create JSON data event |
| `Multimodal.binary(...)` | Create binary data event |
| `Multimodal.data(payload)` | Create data event with full payload |
| `Multimodal.progress(...)` | Create progress event |
| `Multimodal.token(text)` | Create token event |
| `Multimodal.message(value, role)` | Create message event |
| `Multimodal.complete(usage)` | Create complete event |
| `Multimodal.error(error)` | Create error event |
| `Multimodal.to_events(stream, ...)` | Convert stream to events with extractors |
| `Multimodal.from_stream(stream, ...)` | Alias for `to_events()` |

### Data Types

| Type | Description |
| ---- | ----------- |
| `ContentType` | Enum: `TEXT`, `IMAGE`, `AUDIO`, `VIDEO`, `FILE`, `JSON`, `BINARY` |
| `EventType` | Enum: `TOKEN`, `MESSAGE`, `DATA`, `PROGRESS`, `ERROR`, `COMPLETE` |
| `DataPayload` | Multimodal data payload dataclass |
| `Progress` | Progress update dataclass |
| `Event` | Unified event dataclass |

### Module-Level Functions

| Function | Description |
| -------- | ----------- |
| `create_image_event(...)` | Create image data event |
| `create_audio_event(...)` | Create audio data event |
| `create_video_event(...)` | Create video data event |
| `create_data_event(payload)` | Create data event |
| `create_progress_event(...)` | Create progress event |
| `create_token_event(text)` | Create token event |
| `create_message_event(value, role)` | Create message event |
| `create_complete_event(usage)` | Create complete event |
| `create_error_event(error)` | Create error event |
| `create_json_event(json_data, model)` | Create JSON data event |
| `to_multimodal_events(stream, ...)` | Convert multimodal stream to events |
| `to_events(stream, extract_text)` | Convert text stream to events |
| `to_events_with_messages(stream, ...)` | Convert stream with messages to events |
