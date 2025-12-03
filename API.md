# L0 API Reference

Complete API reference for L0 Python.

## Table of Contents

- [Core Functions](#core-functions)
- [Structured Output](#structured-output)
- [Parallel Operations](#parallel-operations)
- [Consensus](#consensus)
- [Guardrails](#guardrails)
- [Retry Configuration](#retry-configuration)
- [Error Handling](#error-handling)
- [Stream Utilities](#stream-utilities)
- [Adapters](#adapters)
- [Observability](#observability)
- [Types](#types)

---

## Core Functions

### l0(options)

Main streaming runtime with guardrails and retry logic.

```python
import l0

result = await l0.l0(l0.L0Options(
    # Required: Stream factory
    stream=lambda: client.chat.completions.create(..., stream=True),

    # Optional: Fallback streams
    fallbacks=[
        lambda: fallback_client.chat.completions.create(..., stream=True),
    ],

    # Optional: Guardrails
    guardrails=l0.recommended_guardrails(),

    # Optional: Retry configuration
    retry=l0.RetryConfig(
        attempts=3,
        max_retries=6,
        base_delay_ms=1000,
        max_delay_ms=10000,
        strategy=l0.BackoffStrategy.FIXED_JITTER,
    ),

    # Optional: Timeout configuration
    timeout=l0.TimeoutConfig(
        initial_token_ms=5000,   # 5s to first token
        inter_token_ms=10000,    # 10s between tokens
    ),

    # Optional: Adapter hint
    adapter="openai",  # or "litellm", or custom Adapter instance

    # Optional: Event callback
    on_event=lambda event: print(event.type),

    # Optional: Metadata for events
    meta={"user_id": "123"},
))

# Consume stream
async for event in result.stream:
    match event.type:
        case l0.EventType.TOKEN:
            print(event.value, end="")
        case l0.EventType.COMPLETE:
            print("\nComplete")
        case l0.EventType.ERROR:
            print(f"Error: {event.error}")

# Access final state
print(result.state.content)
print(result.state.token_count)
```

**Returns:** `L0Result`

| Property | Type                     | Description   |
| -------- | ------------------------ | ------------- |
| `stream` | `AsyncIterator[L0Event]` | Event stream  |
| `state`  | `L0State`                | Runtime state |
| `abort`  | `Callable[[], None]`     | Abort function |
| `errors` | `list[Exception]`        | Errors encountered |

---

## Structured Output

### structured(schema, options, auto_correct)

Guaranteed valid JSON matching a Pydantic schema.

```python
from pydantic import BaseModel
import l0

class UserProfile(BaseModel):
    name: str
    age: int
    email: str

result = await l0.structured(
    schema=UserProfile,
    options=l0.L0Options(
        stream=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Generate user data as JSON"}],
            stream=True,
        ),
    ),
    auto_correct=True,  # Fix trailing commas, missing braces, etc.
)

# Type-safe access
print(result.name)   # str
print(result.age)    # int
print(result.email)  # str
```

**Parameters:**

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `schema` | `Type[BaseModel]` | required | Pydantic model class |
| `options` | `L0Options` | required | L0 configuration |
| `auto_correct` | `bool` | `True` | Auto-fix common JSON errors |

---

## Parallel Operations

### parallel(tasks, concurrency)

Run tasks with concurrency limit.

```python
import l0

async def process_item(item: str) -> str:
    # ... process item
    return result

results = await l0.parallel(
    tasks=[
        lambda: process_item("a"),
        lambda: process_item("b"),
        lambda: process_item("c"),
    ],
    concurrency=2,  # Max 2 concurrent
)
```

### race(tasks)

Return first successful result, cancel remaining.

```python
import l0

result = await l0.race([
    lambda: fast_model(),
    lambda: slow_model(),
    lambda: backup_model(),
])
# Returns first to complete, cancels others
```

### batched(items, handler, batch_size)

Process items in batches.

```python
import l0

async def process(item: str) -> str:
    return item.upper()

results = await l0.batched(
    items=["a", "b", "c", "d", "e"],
    handler=process,
    batch_size=2,
)
# Processes in batches of 2
```

---

## Consensus

### consensus(tasks, strategy)

Multi-generation consensus for high-confidence results.

```python
import l0

result = await l0.consensus(
    tasks=[
        lambda: model_a(),
        lambda: model_b(),
        lambda: model_c(),
    ],
    strategy="majority",  # "unanimous" | "majority" | "best"
)
```

**Strategies:**

| Strategy | Description |
| -------- | ----------- |
| `unanimous` | All results must match exactly |
| `majority` | Most common result wins (>50%) |
| `best` | Return first result |

---

## Guardrails

### Built-in Rules

```python
from l0 import (
    json_rule,           # JSON structure validation
    strict_json_rule,    # Strict JSON (complete only)
    pattern_rule,        # Known bad patterns ("As an AI...")
    zero_output_rule,    # Zero/empty output detection
    stall_rule,          # Token stall detection
    repetition_rule,     # Repetitive output detection
)
```

### Presets

```python
from l0 import (
    recommended_guardrails,  # json + pattern + zero_output
    strict_guardrails,       # All rules including drift
)

# Usage
result = await l0.l0(l0.L0Options(
    stream=my_stream,
    guardrails=recommended_guardrails(),
))
```

### Custom Guardrails

```python
from l0 import GuardrailRule, GuardrailViolation
from l0.types import L0State

def max_tokens_rule(limit: int = 500) -> GuardrailRule:
    def check(state: L0State) -> list[GuardrailViolation]:
        if state.token_count > limit:
            return [GuardrailViolation(
                rule="max_tokens",
                message=f"Exceeded {limit} tokens",
                severity="error",
                recoverable=False,
            )]
        return []
    
    return GuardrailRule(
        name="max_tokens",
        check=check,
        streaming=True,
        severity="error",
    )
```

### GuardrailRule

```python
@dataclass
class GuardrailRule:
    name: str                                    # Unique rule name
    check: Callable[[L0State], list[GuardrailViolation]]
    description: str | None = None
    streaming: bool = True                       # Check during streaming
    severity: Severity = "error"                 # Default severity
    recoverable: bool = True                     # Can retry on violation
```

### GuardrailViolation

```python
@dataclass
class GuardrailViolation:
    rule: str                         # Rule name
    message: str                      # Human-readable message
    severity: Severity                # "warning" | "error" | "fatal"
    recoverable: bool = True
    position: int | None = None       # Position in content
    timestamp: float | None = None
    context: dict[str, Any] | None = None
    suggestion: str | None = None
```

---

## Retry Configuration

### RetryConfig

```python
@dataclass
class RetryConfig:
    attempts: int = 3                 # Model errors only
    max_retries: int = 6              # Absolute cap (all errors)
    base_delay_ms: int = 1000         # Starting delay
    max_delay_ms: int = 10000         # Maximum delay
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER
```

### BackoffStrategy

```python
class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"    # delay * 2^attempt
    LINEAR = "linear"              # delay * attempt
    FIXED = "fixed"                # constant delay
    FULL_JITTER = "full-jitter"    # random(0, exponential)
    FIXED_JITTER = "fixed-jitter"  # base + random jitter (default)
```

### Error Categories

```python
class ErrorCategory(str, Enum):
    NETWORK = "network"      # Retry forever, doesn't count
    TRANSIENT = "transient"  # Retry forever (429, 503)
    MODEL = "model"          # Counts toward limit
    CONTENT = "content"      # Counts toward limit
    PROVIDER = "provider"    # May retry
    FATAL = "fatal"          # Don't retry
    INTERNAL = "internal"    # Don't retry (bugs)
```

### Retry Behavior

| Error Type | Retries | Counts Toward `attempts` | Counts Toward `max_retries` |
| ---------- | ------- | ------------------------ | --------------------------- |
| Network    | Yes     | No                       | Yes                         |
| Timeout    | Yes     | No                       | Yes                         |
| 429/503    | Yes     | No                       | Yes                         |
| Model      | Yes     | **Yes**                  | Yes                         |
| Content    | Yes     | **Yes**                  | Yes                         |
| Fatal      | No      | -                        | -                           |

---

## Error Handling

### categorize_error(error)

Categorize an exception for retry decisions.

```python
from l0.errors import categorize_error
from l0.types import ErrorCategory

category = categorize_error(error)

if category == ErrorCategory.NETWORK:
    print("Network error, will retry")
elif category == ErrorCategory.FATAL:
    print("Fatal error, cannot retry")
```

### Network Error Patterns

L0 automatically detects these network errors:

- Connection reset/refused/timeout
- DNS resolution failures
- SSL errors
- Socket errors
- EOF/broken pipe
- Network/host unreachable

---

## Stream Utilities

### consume_stream(stream)

Consume stream and return full text.

```python
import l0

result = await l0.l0(options)
text = await l0.consume_stream(result.stream)
print(text)
```

### get_text(result)

Helper to get text from L0Result.

```python
import l0

result = await l0.l0(options)
text = await l0.get_text(result)
print(text)
```

---

## Adapters

### Built-in Adapters

L0 includes adapters for:

- **OpenAI** - Direct OpenAI SDK streams
- **LiteLLM** - 100+ providers (Anthropic, Cohere, Bedrock, Vertex, etc.)

Both use the same OpenAI-compatible format.

### Auto-Detection

L0 automatically detects the stream type:

```python
# OpenAI - auto-detected
result = await l0.l0(l0.L0Options(
    stream=lambda: openai_client.chat.completions.create(..., stream=True),
))

# LiteLLM - auto-detected
result = await l0.l0(l0.L0Options(
    stream=lambda: litellm.acompletion(..., stream=True),
))
```

### Explicit Adapter

```python
import l0

result = await l0.l0(l0.L0Options(
    stream=lambda: my_stream(),
    adapter="openai",  # or "litellm"
))
```

### Custom Adapters

```python
from l0 import Adapter, register_adapter, L0Event, EventType
from typing import Any, AsyncIterator

class MyProviderAdapter:
    name = "my_provider"
    
    def detect(self, stream: Any) -> bool:
        """Check if this adapter can handle the stream."""
        return "my_provider" in type(stream).__module__
    
    async def wrap(self, stream: Any) -> AsyncIterator[L0Event]:
        """Convert provider stream to L0 events."""
        async for chunk in stream:
            if chunk.text:
                yield L0Event(type=EventType.TOKEN, value=chunk.text)
        
        yield L0Event(type=EventType.COMPLETE, usage={
            "input_tokens": 100,
            "output_tokens": 50,
        })

# Register for auto-detection
register_adapter(MyProviderAdapter())
```

### Adapter Protocol

```python
from typing import Protocol, Any, AsyncIterator

class Adapter(Protocol):
    name: str
    
    def detect(self, stream: Any) -> bool:
        """Return True if this adapter can handle the stream."""
        ...
    
    def wrap(self, stream: Any) -> AsyncIterator[L0Event]:
        """Wrap raw stream into L0Event stream."""
        ...
```

---

## Observability

### EventBus

Central event bus for all L0 observability.

```python
from l0 import EventBus, ObservabilityEventType

def handler(event):
    print(f"[{event.type}] stream={event.stream_id}")
    print(f"  ts={event.ts}, meta={event.meta}")

bus = EventBus(
    handler=handler,
    meta={"session": "abc123"},
)

# Events are emitted automatically during l0() execution
result = await l0.l0(l0.L0Options(
    stream=my_stream,
    on_event=handler,
    meta={"user_id": "123"},
))
```

### ObservabilityEvent

```python
@dataclass
class ObservabilityEvent:
    type: ObservabilityEventType     # Event type
    ts: float                        # Unix epoch milliseconds
    stream_id: str                   # UUIDv7 stream identifier
    meta: dict[str, Any]             # Event metadata
```

### Event Types

```python
class ObservabilityEventType(str, Enum):
    # Session
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    
    # Stream
    STREAM_INIT = "STREAM_INIT"
    STREAM_READY = "STREAM_READY"
    
    # Retry
    RETRY_START = "RETRY_START"
    RETRY_ATTEMPT = "RETRY_ATTEMPT"
    RETRY_END = "RETRY_END"
    RETRY_GIVE_UP = "RETRY_GIVE_UP"
    
    # Fallback
    FALLBACK_START = "FALLBACK_START"
    FALLBACK_END = "FALLBACK_END"
    
    # Guardrail
    GUARDRAIL_PHASE_START = "GUARDRAIL_PHASE_START"
    GUARDRAIL_RULE_RESULT = "GUARDRAIL_RULE_RESULT"
    GUARDRAIL_PHASE_END = "GUARDRAIL_PHASE_END"
    
    # Drift
    DRIFT_CHECK_RESULT = "DRIFT_CHECK_RESULT"
    
    # Network
    NETWORK_ERROR = "NETWORK_ERROR"
    NETWORK_RECOVERY = "NETWORK_RECOVERY"
    
    # Checkpoint
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    
    # Completion
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
```

---

## Types

### L0Options

```python
@dataclass
class L0Options:
    stream: Callable[[], AsyncIterator[Any]]    # Stream factory
    fallbacks: list[Callable[[], AsyncIterator[Any]]] = field(default_factory=list)
    guardrails: list[GuardrailRule] = field(default_factory=list)
    retry: RetryConfig | None = None
    timeout: TimeoutConfig | None = None
    adapter: Adapter | str | None = None
    on_event: Callable[[ObservabilityEvent], None] | None = None
    meta: dict[str, Any] | None = None
```

### L0Result

```python
@dataclass
class L0Result:
    stream: AsyncIterator[L0Event]
    state: L0State
    abort: Callable[[], None]
    errors: list[Exception] = field(default_factory=list)
```

### L0State

```python
@dataclass
class L0State:
    content: str = ""                           # Accumulated content
    checkpoint: str = ""                        # Last checkpoint
    token_count: int = 0                        # Total tokens
    model_retry_count: int = 0                  # Model error retries
    network_retry_count: int = 0                # Network error retries
    fallback_index: int = 0                     # Current fallback (0=primary)
    violations: list[GuardrailViolation] = field(default_factory=list)
    drift_detected: bool = False
    completed: bool = False
    aborted: bool = False
    first_token_at: float | None = None
    last_token_at: float | None = None
    duration: float | None = None
    resumed: bool = False
    network_errors: list[Any] = field(default_factory=list)
```

### L0Event

```python
@dataclass
class L0Event:
    type: EventType
    value: str | None = None                    # Token content
    data: dict[str, Any] | None = None          # Tool call data
    error: Exception | None = None
    usage: dict[str, int] | None = None         # Token usage
    timestamp: float | None = None
```

### EventType

```python
class EventType(str, Enum):
    TOKEN = "token"
    MESSAGE = "message"
    DATA = "data"
    PROGRESS = "progress"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    COMPLETE = "complete"
```

### TimeoutConfig

```python
@dataclass
class TimeoutConfig:
    initial_token_ms: int = 5000     # Max wait for first token
    inter_token_ms: int = 10000      # Max wait between tokens
```

---

## Utility Functions

### JSON Utilities

```python
from l0._utils import auto_correct_json, extract_json_from_markdown

# Fix common JSON errors
fixed = auto_correct_json('{"a": 1,}')  # '{"a": 1}'

# Extract JSON from markdown code blocks
json_str = extract_json_from_markdown('''
```json
{"key": "value"}
```
''')
```

### Debug Logging

```python
import l0

# Enable debug logging
l0.enable_debug()
```

---

## See Also

- [README.md](./README.md) - Quick start guide
- [PLAN.md](./PLAN.md) - Implementation plan and design decisions
