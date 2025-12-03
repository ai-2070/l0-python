# L0 API Reference

Complete API reference for L0 Python.

> Most applications should simply use `import l0`.
> See [Imports](#imports) for details on available exports.

## Table of Contents

- [Core Functions](#core-functions)
- [Lifecycle Callbacks](#lifecycle-callbacks)
- [Streaming Runtime](#streaming-runtime)
- [Retry Configuration](#retry-configuration)
- [Network Protection](#network-protection)
- [Structured Output](#structured-output)
- [Fallback Models](#fallback-models)
- [Guardrails](#guardrails)
- [Consensus](#consensus)
- [Parallel Operations](#parallel-operations)
- [Custom Adapters](#custom-adapters)
- [Observability](#observability)
- [Error Handling](#error-handling)
- [State Machine](#state-machine)
- [Metrics](#metrics)
- [Stream Utilities](#stream-utilities)
- [Utility Functions](#utility-functions)
- [Types](#types)
- [Imports](#imports)

---

## Core Functions

### wrap(stream, *, guardrails, retry, timeout, adapter, on_event, meta)

Wrap a raw LLM stream with L0 reliability. **Returns immediately (no await needed).**

This is the preferred API for simple cases - takes a raw stream directly, no lambda needed. Like `httpx.AsyncClient()` or `aiohttp.ClientSession()`.

```python
import l0

# Create your stream
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

# wrap() returns immediately - no await!
result = l0.wrap(stream, guardrails=l0.Guardrails.recommended())

# Read full text
text = await result.read()

# Or stream events
async for event in l0.wrap(stream):
    if event.is_token:
        print(event.text, end="")

# Or use context manager
async with l0.wrap(stream) as result:
    async for event in result:
        if event.is_token:
            print(event.text, end="")
```

**Parameters:**

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `stream` | `AsyncIterator` | required | Raw async LLM stream |
| `guardrails` | `list[GuardrailRule]` | `None` | Guardrail rules to apply |
| `retry` | `Retry` | `None` | Retry configuration |
| `timeout` | `Timeout` | `None` | Timeout configuration |
| `adapter` | `str \| Adapter` | `None` | Adapter hint or instance |
| `on_event` | `Callable` | `None` | Observability callback |
| `meta` | `dict` | `None` | Metadata for events |

**Returns:** `LazyStream` - Async iterator with attached state (starts lazily on first use)

---

### run(stream, *, fallbacks, guardrails, retry, timeout, adapter, on_event, meta)

Run L0 with a stream factory. Use when you need **retries or fallbacks** (which require re-creating the stream).

> **Note:** `l0()` is an alias to `run()` for convenience. Both work identically.

```python
import l0

result = await l0.run(
    # Required: Stream factory (lambda for retries)
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ),

    # Optional: Fallback streams
    fallbacks=[
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
    ],

    # Optional: Guardrails
    guardrails=l0.Guardrails.recommended(),

    # Optional: Retry configuration (defaults shown)
    retry=l0.Retry(
        attempts=3,                              # LLM errors only
        max_retries=6,                           # Total (LLM + network)
        base_delay=1.0,                          # Seconds
        max_delay=10.0,                          # Seconds
        strategy=l0.BackoffStrategy.FIXED_JITTER,
    ),

    # Optional: Timeout configuration (defaults shown)
    timeout=l0.Timeout(
        initial_token=5.0,   # Seconds to first token
        inter_token=10.0,    # Seconds between tokens
    ),

    # Optional: Adapter hint
    adapter="openai",  # or "litellm", or Adapter instance

    # Optional: Event callback
    on_event=lambda event: print(f"[{event.type}]"),

    # Optional: Metadata for events
    meta={"user_id": "123"},
)

# Iterate with Pythonic event properties
async for event in result:
    if event.is_token:
        print(event.text, end="")
    elif event.is_tool_call:
        print(f"Tool call: {event.data}")
    elif event.is_complete:
        print("\nComplete")
        print(f"Usage: {event.usage}")
    elif event.is_error:
        print(f"Error: {event.error}")

# Or get full text directly
text = await result.read()

# Access state anytime
print(result.state.content)       # Full accumulated content
print(result.state.token_count)   # Total tokens received
print(result.state.checkpoint)    # Last stable checkpoint
print(result.state.duration)      # Duration in seconds
```

**Parameters:**

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `stream` | `Callable[[], AsyncIterator]` | required | Factory returning async LLM stream |
| `fallbacks` | `list[Callable]` | `None` | Fallback stream factories |
| `guardrails` | `list[GuardrailRule]` | `None` | Guardrail rules to apply |
| `retry` | `Retry` | `None` | Retry configuration |
| `timeout` | `Timeout` | `None` | Timeout configuration |
| `adapter` | `str \| Adapter` | `None` | Adapter hint or instance |
| `on_event` | `Callable` | `None` | Observability callback |
| `meta` | `dict` | `None` | Metadata for events |

**Returns:** `Stream` - Async iterator with attached state

| Property/Method | Type | Description |
| --------------- | ---- | ----------- |
| `__aiter__` | - | Iterate directly over events |
| `state` | `State` | Runtime state |
| `abort()` | `Callable[[], None]` | Abort the stream |
| `read()` | `async -> str` | Consume stream, return full text |
| `errors` | `list[Exception]` | Errors encountered |

### wrap() vs run()

| Function | When to Use | Stream Argument |
| -------- | ----------- | --------------- |
| `wrap()` | Simple cases, no retries/fallbacks | Raw stream directly |
| `run()` | Need retries or fallbacks | Lambda factory |

```python
# Simple - use wrap()
result = l0.wrap(stream)
text = await result.read()

# With retries/fallbacks - use run()
result = await l0.run(
    stream=lambda: create_stream(),  # Lambda for retries
    fallbacks=[lambda: backup_stream()],
)
```

---

## Lifecycle Callbacks

L0 provides lifecycle callbacks for monitoring and responding to runtime events. All callbacks are optional and are pure side-effect handlers (they don't affect execution flow).

### Callback Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            L0 LIFECYCLE FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

                                ┌──────────┐
                                │  START   │
                                └────┬─────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │       on_event(event)        │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              STREAMING PHASE                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         on_event(event)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  During streaming, events fire as conditions occur:                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  CHECKPOINT  │  │  TOOL_CALL   │  │    DRIFT     │  │   TIMEOUT    │   │
│  │    SAVED     │  │   detected   │  │   detected   │  │   occurred   │   │
│  └──────────────┘  └──────────────┘  └──────┬───────┘  └──────┬───────┘   │
│                                             │                  │           │
│                                             └────────┬─────────┘           │
│                                                      │ triggers retry      │
└──────────────────────────────────────────────────────┼─────────────────────┘
                                                       │
              ┌────────────────────────────────────────┼────────────────┐
              │                    │                   │                │
              ▼                    ▼                   ▼                ▼
        ┌─────────┐          ┌───────────┐      ┌──────────┐      ┌─────────┐
        │ SUCCESS │          │   ERROR   │      │VIOLATION │      │  ABORT  │
        └────┬────┘          └─────┬─────┘      └────┬─────┘      └────┬────┘
             │                     │                 │                 │
             │                     ▼                 ▼                 ▼
             │              ┌────────────────────────────────┐   ┌───────────┐
             │              │      on_event(ERROR)           │   │ ABORTED   │
             │              └──────────────┬─────────────────┘   └───────────┘
             │                             │
             │                 ┌───────────┼───────────┐
             │                 │           │           │
             │                 ▼           ▼           ▼
             │           ┌──────────┐ ┌──────────┐ ┌──────────┐
             │           │  RETRY   │ │ FALLBACK │ │  FATAL   │
             │           └────┬─────┘ └────┬─────┘ └────┬─────┘
             │                │            │            │
             │                │    ┌───────┘            │
             │                │    │                    │
             │                ▼    ▼                    │
             │          ┌─────────────────────┐         │
             │          │  Has checkpoint?    │         │
             │          └──────────┬──────────┘         │
             │                YES  │  NO                │
             │                ┌────┴────┐               │
             │                ▼         ▼               │
             │          ┌──────────┐    │               │
             │          │  RESUME  │    │               │
             │          └────┬─────┘    │               │
             │               │          │               │
             │               ▼          ▼               │
             │          ┌─────────────────────────┐     │
             │          │    Back to STREAMING    │─────┘
             │          └─────────────────────────┘
             │
             ▼
      ┌─────────────┐
      │  COMPLETE   │
      └─────────────┘
```

### Callback Reference

| Callback | Signature | When Called |
| -------- | --------- | ----------- |
| `on_event` | `(event: ObservabilityEvent) -> None` | Any runtime event emitted |

### ObservabilityEventType Reference

| Event Type | Description |
| ---------- | ----------- |
| `SESSION_START` | New execution session begins |
| `SESSION_END` | Session completed |
| `STREAM_INIT` | Stream initialized |
| `STREAM_READY` | Stream ready for tokens |
| `RETRY_START` | Retry sequence starting |
| `RETRY_ATTEMPT` | Individual retry attempt |
| `RETRY_END` | Retry sequence completed |
| `RETRY_GIVE_UP` | All retries exhausted |
| `FALLBACK_START` | Switching to fallback model |
| `FALLBACK_END` | Fallback sequence completed |
| `GUARDRAIL_PHASE_START` | Guardrail check starting |
| `GUARDRAIL_RULE_RESULT` | Individual rule result |
| `GUARDRAIL_PHASE_END` | Guardrail check completed |
| `DRIFT_CHECK_RESULT` | Drift detection result |
| `NETWORK_ERROR` | Network error occurred |
| `NETWORK_RECOVERY` | Recovered from network error |
| `CHECKPOINT_SAVED` | Checkpoint saved |
| `COMPLETE` | Stream completed successfully |
| `ERROR` | Error occurred |

### Usage Example

```python
import l0

def handle_event(event: l0.ObservabilityEvent):
    match event.type:
        case l0.ObservabilityEventType.SESSION_START:
            print(f"Session started: {event.stream_id}")
        case l0.ObservabilityEventType.RETRY_ATTEMPT:
            print(f"Retrying (attempt {event.meta.get('attempt', '?')})")
        case l0.ObservabilityEventType.FALLBACK_START:
            print(f"Switching to fallback {event.meta.get('index', '?')}")
        case l0.ObservabilityEventType.CHECKPOINT_SAVED:
            print(f"Checkpoint saved ({event.meta.get('token_count', 0)} tokens)")
        case l0.ObservabilityEventType.NETWORK_ERROR:
            print(f"Network error: {event.meta.get('error', 'unknown')}")
        case l0.ObservabilityEventType.COMPLETE:
            print(f"Complete! Duration: {event.meta.get('duration', 0)}s")
        case l0.ObservabilityEventType.ERROR:
            print(f"Error: {event.meta.get('error', 'unknown')}")

result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ),
    on_event=handle_event,
    meta={"user_id": "123", "request_id": "abc"},
)
```

---

## Streaming Runtime

L0 wraps LLM streams with deterministic behavior and unified event types.

### Unified Event Format

All streams are normalized to `Event` objects:

```python
@dataclass
class Event:
    type: EventType                           # Event type
    text: str | None = None                   # Token content
    data: dict[str, Any] | None = None        # Tool call / misc data
    error: Exception | None = None            # Error (for error events)
    usage: dict[str, int] | None = None       # Token usage
    timestamp: float | None = None            # Event timestamp

    # Pythonic type check properties
    @property
    def is_token(self) -> bool: ...
    @property
    def is_message(self) -> bool: ...
    @property
    def is_data(self) -> bool: ...
    @property
    def is_progress(self) -> bool: ...
    @property
    def is_tool_call(self) -> bool: ...
    @property
    def is_error(self) -> bool: ...
    @property
    def is_complete(self) -> bool: ...
```

### Event Types

```python
class EventType(str, Enum):
    TOKEN = "token"           # Text token
    MESSAGE = "message"       # Full message
    DATA = "data"             # Structured data
    PROGRESS = "progress"     # Progress update
    TOOL_CALL = "tool_call"   # Tool/function call
    ERROR = "error"           # Error occurred
    COMPLETE = "complete"     # Stream complete
```

### Tool Call Handling

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }],
        stream=True,
    ),
)

async for event in result:
    if event.is_tool_call:
        print(f"Tool: {event.data['name']}")
        print(f"Args: {event.data['arguments']}")
        print(f"ID: {event.data['id']}")
```

### State Tracking

```python
# Access state at any point
state = result.state

state.content           # Accumulated content
state.checkpoint        # Last validated checkpoint
state.token_count       # Total tokens received
state.model_retry_count # Model error retries
state.network_retry_count # Network error retries
state.fallback_index    # Current model (0=primary)
state.violations        # Guardrail violations
state.drift_detected    # Whether drift was detected
state.completed         # Stream completed successfully
state.aborted           # Stream was aborted
state.first_token_at    # Timestamp of first token
state.last_token_at     # Timestamp of last token
state.duration          # Total duration (seconds)
state.resumed           # Resumed from checkpoint
```

---

## Retry Configuration

### Retry

All delays are in **seconds** (float), matching Python conventions like `asyncio.sleep()`.

```python
@dataclass
class Retry:
    attempts: int = 3                 # Model errors only
    max_retries: int = 6              # Absolute cap (all errors)
    base_delay: float = 1.0           # Starting delay (seconds)
    max_delay: float = 10.0           # Maximum delay (seconds)
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER
```

### BackoffStrategy

```python
class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"    # delay * 2^attempt
    LINEAR = "linear"              # delay * (attempt + 1)
    FIXED = "fixed"                # constant delay
    FULL_JITTER = "full-jitter"    # random(0, exponential)
    FIXED_JITTER = "fixed-jitter"  # base/2 + random(base/2) - DEFAULT
```

### Backoff Calculation

| Strategy | Formula | Example (base=1.0s, attempt=2) |
| -------- | ------- | -------------------------------- |
| `EXPONENTIAL` | `min(base * 2^attempt, max)` | 4.0s |
| `LINEAR` | `min(base * (attempt + 1), max)` | 3.0s |
| `FIXED` | `base` | 1.0s |
| `FULL_JITTER` | `random(0, min(base * 2^attempt, max))` | 0-4.0s |
| `FIXED_JITTER` | `temp/2 + random(temp/2)` | 2.0-4.0s |

### Retry Behavior by Error Type

| Error Type | Retries | Counts Toward `attempts` | Counts Toward `max_retries` |
| ---------- | ------- | ------------------------ | --------------------------- |
| Network disconnect | Yes | No | Yes |
| Zero output | Yes | No | Yes |
| Timeout | Yes | No | Yes |
| 429 rate limit | Yes | No | Yes |
| 503 server error | Yes | No | Yes |
| Guardrail violation | Yes | **Yes** | Yes |
| Drift detected | Yes | **Yes** | Yes |
| Auth error (401/403) | No | - | - |

### RetryManager

```python
from l0.retry import RetryManager
from l0.types import Retry, BackoffStrategy

manager = RetryManager(Retry(
    attempts=3,
    strategy=BackoffStrategy.EXPONENTIAL,
))

# Check if should retry
if manager.should_retry(error):
    manager.record_attempt(error)
    delay = manager.get_delay(error)  # Returns seconds (float)
    await manager.wait(error)

# Get state
state = manager.get_state()
# {"model_retry_count": 1, "network_retry_count": 0, "total_retries": 1}

# Reset
manager.reset()
```

---

## Network Protection

### Error Categorization

```python
from l0.errors import categorize_error
from l0.types import ErrorCategory

category = categorize_error(error)

match category:
    case ErrorCategory.NETWORK:
        print("Network error - retry forever")
    case ErrorCategory.TRANSIENT:
        print("Transient (429/503) - retry forever")
    case ErrorCategory.MODEL:
        print("Model error - counts toward limit")
    case ErrorCategory.CONTENT:
        print("Content error - counts toward limit")
    case ErrorCategory.PROVIDER:
        print("Provider error - may retry")
    case ErrorCategory.FATAL:
        print("Fatal - no retry (401/403)")
    case ErrorCategory.INTERNAL:
        print("Internal - no retry (bug)")
```

### Network Error Patterns

L0 automatically detects these patterns in error messages:

| Pattern | Description |
| ------- | ----------- |
| `connection.*reset` | Connection reset by peer |
| `connection.*refused` | Connection refused |
| `connection.*timeout` | Connection timeout |
| `timed?\s*out` | Request timed out |
| `dns.*failed` | DNS resolution failed |
| `name.*resolution` | Name resolution error |
| `socket.*error` | Socket error |
| `ssl.*error` | SSL/TLS error |
| `eof.*occurred` | Unexpected EOF |
| `broken.*pipe` | Broken pipe |
| `network.*unreachable` | Network unreachable |
| `host.*unreachable` | Host unreachable |

### HTTP Status Code Handling

| Status | Category | Behavior |
| ------ | -------- | -------- |
| 429 | `TRANSIENT` | Retry forever |
| 500-599 | `TRANSIENT` | Retry forever |
| 401 | `FATAL` | No retry |
| 403 | `FATAL` | No retry |

---

## Structured Output

### structured(schema, stream, *, fallbacks, auto_correct, retry, on_validation_error, on_auto_correct, on_event, adapter)

Guaranteed valid JSON matching a Pydantic schema.

```python
from pydantic import BaseModel
import l0

class UserProfile(BaseModel):
    name: str
    age: int
    email: str
    tags: list[str] = []

result = await l0.structured(
    schema=UserProfile,
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Generate user data as JSON"}],
        stream=True,
    ),
    auto_correct=True,  # Fix common JSON errors
)

# Type-safe access
print(result.data.name)    # str
print(result.data.age)     # int
print(result.data.email)   # str
print(result.data.tags)    # list[str]
```

**Parameters:**

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `schema` | `type[BaseModel]` | required | Pydantic model class |
| `stream` | `AsyncIterator \| Callable[[], AsyncIterator]` | required | Async LLM stream or factory returning one |
| `fallbacks` | `list[AsyncIterator \| Callable]` | `None` | Fallback streams to try if primary fails |
| `auto_correct` | `bool` | `True` | Auto-fix common JSON errors |
| `retry` | `Retry` | `None` | Retry configuration for validation failures |
| `on_validation_error` | `Callable[[ValidationError, int], None]` | `None` | Callback when validation fails (error, attempt) |
| `on_auto_correct` | `Callable[[AutoCorrectInfo], None]` | `None` | Callback when auto-correction is applied |
| `on_event` | `Callable[[ObservabilityEvent], None]` | `None` | Callback for observability events |
| `adapter` | `Any \| str` | `None` | Adapter hint ("openai", "litellm", or instance) |

### JSON Auto-Correction

```python
from l0._utils import auto_correct_json, extract_json_from_markdown

# Remove trailing commas
auto_correct_json('{"a": 1,}')  # '{"a": 1}'

# Balance braces
auto_correct_json('{"a": {"b": 1}')  # '{"a": {"b": 1}}'

# Balance brackets
auto_correct_json('[1, 2, 3')  # '[1, 2, 3]'

# Strip whitespace
auto_correct_json('  {"a": 1}  ')  # '{"a": 1}'

# Extract from markdown fences
extract_json_from_markdown('''
Here's the data:
```json
{"key": "value"}
```
''')  # '{"key": "value"}'
```

---

## Fallback Models

Sequential fallback when primary model fails:

```python
result = await l0.run(
    stream=lambda: openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ),
    fallbacks=[
        # Fallback 1: Cheaper OpenAI model
        lambda: openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
        # Fallback 2: Different provider via LiteLLM
        lambda: litellm.acompletion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
    ],
)

# Check which model succeeded
if result.state.fallback_index == 0:
    print("Primary model (gpt-4o) succeeded")
elif result.state.fallback_index == 1:
    print("Fallback 1 (gpt-4o-mini) succeeded")
else:
    print(f"Fallback {result.state.fallback_index} succeeded")
```

### Fallback Behavior

1. Primary stream fails (error, timeout, guardrail violation)
2. L0 exhausts retries for primary stream
3. Moves to first fallback, resets retry counter
4. Repeats until success or all fallbacks exhausted
5. Raises last error if all fail

---

## Guardrails

### Built-in Rules

```python
import l0

# Individual rules
l0.json_rule()           # Validates JSON structure (balanced braces)
l0.strict_json_rule()    # Validates complete JSON (on completion only)
l0.pattern_rule()        # Detects "As an AI..." patterns
l0.zero_output_rule()    # Detects empty output
l0.stall_rule()          # Detects token stalls
l0.repetition_rule()     # Detects model looping
```

### Presets (Recommended)

```python
import l0

# Recommended: json + pattern + zero_output
guardrails = l0.Guardrails.recommended()

# Strict: All rules including drift detection
guardrails = l0.Guardrails.strict()

# JSON only
guardrails = l0.Guardrails.json_only()

# None (empty list)
guardrails = l0.Guardrails.none()
```

### Rule Details

| Rule | Streaming | Default Severity | Description |
| ---- | --------- | ---------------- | ----------- |
| `json_rule()` | Yes | error | Checks balanced `{}[]` brackets |
| `strict_json_rule()` | No | error | Validates JSON via `json.loads()` on complete |
| `pattern_rule(patterns)` | Yes | warning | Regex patterns (default: AI slop) |
| `zero_output_rule()` | No | error | Empty output on complete |
| `stall_rule(max_gap)` | Yes | warning | No tokens for `max_gap` seconds |
| `repetition_rule(window, threshold)` | Yes | error | Repeated content detection |

### Custom Guardrails

```python
from l0 import GuardrailRule, GuardrailViolation
from l0.types import State

def max_length_rule(limit: int = 1000) -> GuardrailRule:
    """Detect output exceeding length limit."""
    
    def check(state: State) -> list[GuardrailViolation]:
        if len(state.content) > limit:
            return [GuardrailViolation(
                rule="max_length",
                message=f"Output exceeds {limit} chars",
                severity="error",
                recoverable=True,
            )]
        return []
    
    return GuardrailRule(
        name="max_length",
        check=check,
        description="Detects output exceeding length limit",
        streaming=True,
        severity="error",
        recoverable=True,
    )

# Usage
result = await l0.run(
    stream=my_stream,
    guardrails=[max_length_rule(500)],
)
```

### GuardrailRule

```python
@dataclass
class GuardrailRule:
    name: str                                    # Unique name
    check: Callable[[State], list[GuardrailViolation]]
    description: str | None = None               # Human description
    streaming: bool = True                       # Check during streaming
    severity: Severity = "error"                 # Default severity
    recoverable: bool = True                     # Can retry on violation
```

### GuardrailViolation

```python
@dataclass
class GuardrailViolation:
    rule: str                         # Rule name that triggered
    message: str                      # Human-readable message
    severity: Severity                # "warning" | "error" | "fatal"
    recoverable: bool = True          # Can retry/fallback
    position: int | None = None       # Position in content
    timestamp: float | None = None    # When detected
    context: dict[str, Any] | None = None   # Extra context
    suggestion: str | None = None     # Suggested fix
```

### Violation Handling

```python
# Access violations from result
for violation in result.state.violations:
    print(f"[{violation.severity}] {violation.rule}: {violation.message}")
    
    if not violation.recoverable:
        print("  Fatal - cannot retry")
```

---

## Consensus

### consensus(tasks, strategy)

Multi-generation consensus for high-confidence results.

```python
import l0

result = await l0.consensus(
    tasks=[
        lambda: generate_answer_model_a(),
        lambda: generate_answer_model_b(),
        lambda: generate_answer_model_c(),
    ],
    strategy="majority",  # "unanimous" | "majority" | "best"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `tasks` | `list[Callable[[], Awaitable[T]]]` | required | Async callables |
| `strategy` | `Strategy` | `"majority"` | Consensus strategy |

### Strategies

| Strategy | Description | Raises |
| -------- | ----------- | ------ |
| `unanimous` | All results must be identical | `ValueError` if any differ |
| `majority` | Most common result wins (>50%) | `ValueError` if no majority |
| `best` | Return first result | Never (unless all fail) |

### Example: Multi-Model Validation

```python
async def get_answer(model: str) -> str:
    result = await l0.run(
        stream=lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            stream=True,
        ),
    )
    return await result.read()

# Require agreement from multiple models
try:
    answer = await l0.consensus(
        tasks=[
            lambda: get_answer("gpt-4o"),
            lambda: get_answer("gpt-4o-mini"),
            lambda: get_answer("gpt-4-turbo"),
        ],
        strategy="majority",
    )
    print(f"Consensus answer: {answer}")
except ValueError as e:
    print(f"No consensus: {e}")
```

---

## Parallel Operations

### parallel(tasks, concurrency)

Run tasks with concurrency limit.

```python
import l0

async def process_document(doc: str) -> str:
    result = await l0.run(stream=lambda: summarize(doc))
    return await result.read()

# Process 10 documents, max 3 concurrent
results = await l0.parallel(
    tasks=[lambda d=doc: process_document(d) for doc in documents],
    concurrency=3,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `tasks` | `list[Callable[[], Awaitable[T]]]` | required | Async callables |
| `concurrency` | `int` | `5` | Max concurrent tasks |

**Returns:** `list[T]` - Results in same order as tasks

### race(tasks)

Return first successful result, cancel remaining.

```python
import l0

# First model to respond wins
result = await l0.race([
    lambda: fast_but_expensive_model(),
    lambda: slow_but_cheap_model(),
    lambda: backup_model(),
])
```

**Behavior:**
1. All tasks start immediately
2. First to complete successfully is returned
3. All other tasks are cancelled
4. If first fails, does NOT wait for others

### batched(items, handler, batch_size)

Process items in batches.

```python
import l0

async def embed(text: str) -> list[float]:
    # Get embedding for single text
    return embedding

# Process 1000 texts in batches of 50
embeddings = await l0.batched(
    items=texts,  # 1000 texts
    handler=embed,
    batch_size=50,
)
# Result: 1000 embeddings in order
```

**Parameters:**

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `items` | `list[T]` | required | Items to process |
| `handler` | `Callable[[T], Awaitable[R]]` | required | Async handler |
| `batch_size` | `int` | `10` | Batch size |

### Pattern Comparison

| Pattern | Execution | Cost | Best For |
| ------- | --------- | ---- | -------- |
| `run()` with fallbacks | Sequential on failure | Low | High availability |
| `race()` | Parallel, first wins | High | Low latency |
| `parallel()` | Parallel with limit | Medium | Batch processing |
| `batched()` | Sequential batches | Low | Large datasets |
| `consensus()` | Parallel, vote | High | High reliability |

---

## Custom Adapters

### Adapter Protocol

```python
from typing import Protocol, Any
from collections.abc import AsyncIterator

class Adapter(Protocol):
    name: str
    
    def detect(self, stream: Any) -> bool:
        """Return True if this adapter can handle the stream."""
        ...
    
    def wrap(self, stream: Any) -> AsyncIterator[Event]:
        """Wrap raw stream into Event stream."""
        ...
```

### Built-in Adapters

| Adapter | Auto-Detected | Description |
| ------- | ------------- | ----------- |
| `OpenAIAdapter` | Yes | OpenAI SDK streams |
| `LiteLLMAdapter` | Yes | LiteLLM streams (alias for OpenAI) |

### Creating Custom Adapters

```python
from collections.abc import AsyncIterator
from typing import Any
import l0
from l0 import Event, EventType, Adapters

class AnthropicAdapter:
    """Adapter for direct Anthropic SDK (if not using LiteLLM)."""
    name = "anthropic"
    
    def detect(self, stream: Any) -> bool:
        return "anthropic" in type(stream).__module__
    
    async def wrap(self, stream: Any) -> AsyncIterator[Event]:
        usage = None
        
        async for event in stream:
            event_type = getattr(event, "type", None)
            
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and hasattr(delta, "text"):
                    yield Event(type=EventType.TOKEN, text=delta.text)
            
            elif event_type == "content_block_start":
                block = getattr(event, "content_block", None)
                if block and getattr(block, "type", None) == "tool_use":
                    yield Event(
                        type=EventType.TOOL_CALL,
                        data={
                            "id": getattr(block, "id", None),
                            "name": getattr(block, "name", None),
                        }
                    )
            
            elif event_type == "message_delta":
                msg_usage = getattr(event, "usage", None)
                if msg_usage:
                    usage = {
                        "input_tokens": getattr(msg_usage, "input_tokens", 0),
                        "output_tokens": getattr(msg_usage, "output_tokens", 0),
                    }
            
            elif event_type == "message_stop":
                yield Event(type=EventType.COMPLETE, usage=usage)

# Register for auto-detection
Adapters.register(AnthropicAdapter())
```

### Adapter Functions

```python
from l0 import Adapters

# Register custom adapter (takes priority over built-ins)
Adapters.register(MyAdapter())

# Explicitly detect adapter
adapter = Adapters.detect(stream)
print(adapter.name)

# Use specific adapter by name
result = await l0.run(
    stream=my_stream,
    adapter="openai",  # Force OpenAI adapter
)

# Use adapter instance directly
result = await l0.run(
    stream=my_stream,
    adapter=MyCustomAdapter(),
)
```

### Adapter Invariants

Adapters **MUST**:
- Preserve text exactly (no trimming, modification)
- Convert errors to error events (never throw from wrap)
- Emit `COMPLETE` event exactly once at end
- Handle empty/null content gracefully

---

## Observability

### EventBus

Central event bus for all L0 observability.

```python
from l0 import EventBus, ObservabilityEvent, ObservabilityEventType

def my_handler(event: ObservabilityEvent):
    print(f"[{event.type}] stream={event.stream_id}")
    print(f"  ts={event.ts}ms")
    print(f"  meta={event.meta}")

# Create event bus
bus = EventBus(
    handler=my_handler,
    meta={"service": "my-app"},
)

# Access stream ID (UUIDv7)
print(bus.stream_id)

# Emit custom events
bus.emit(ObservabilityEventType.CHECKPOINT_SAVED, checkpoint="...", token_count=100)
```

### Using with run()

```python
result = await l0.run(
    stream=my_stream,
    on_event=lambda e: print(f"[{e.type}] {e.meta}"),
    meta={"user_id": "123", "request_id": "abc"},
)
```

### ObservabilityEvent

```python
@dataclass
class ObservabilityEvent:
    type: ObservabilityEventType     # Event type
    ts: float                        # Unix epoch MILLISECONDS
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

## Error Handling

### Error Categories

```python
class ErrorCategory(str, Enum):
    NETWORK = "network"      # Connection drops, DNS, SSL
    TRANSIENT = "transient"  # 429, 503 - temporary
    MODEL = "model"          # Model refused, malformed
    CONTENT = "content"      # Guardrail, drift
    PROVIDER = "provider"    # API errors
    FATAL = "fatal"          # Auth errors (401/403)
    INTERNAL = "internal"    # Bugs, internal errors
```

### categorize_error(error)

```python
from l0.errors import categorize_error
from l0.types import ErrorCategory

try:
    result = await l0.run(stream=my_stream)
except Exception as error:
    category = categorize_error(error)
    
    if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
        print("Transient error - would have retried")
    elif category == ErrorCategory.FATAL:
        print("Fatal error - check credentials")
    elif category == ErrorCategory.INTERNAL:
        print("Bug - please report")
```

### Error Category Behavior

| Category | Retries | Counts Toward Limit | Example |
| -------- | ------- | ------------------- | ------- |
| `NETWORK` | Forever | No | Connection reset |
| `TRANSIENT` | Forever | No | 429 rate limit |
| `MODEL` | Limited | Yes | Model refused |
| `CONTENT` | Limited | Yes | Guardrail violation |
| `PROVIDER` | Depends | Depends | API error |
| `FATAL` | Never | - | 401 unauthorized |
| `INTERNAL` | Never | - | Bug |

### TimeoutError

L0's timeout error with details:

```python
from l0 import TimeoutError

try:
    result = await l0.run(
        stream=my_stream,
        timeout=l0.Timeout(initial_token=1.0),
    )
    async for event in result:
        pass
except TimeoutError as e:
    print(e.timeout_type)     # "initial_token" or "inter_token"
    print(e.timeout_seconds)  # The timeout value that was exceeded
```

---

## Stream Utilities

### consume_stream(stream)

Consume stream and return full text.

```python
import l0

result = await l0.run(stream=my_stream)
text = await l0.consume_stream(result)
print(text)
```

### get_text(result)

Helper to get text from Stream result.

```python
import l0

result = await l0.run(stream=my_stream)
text = await l0.get_text(result)
print(text)
```

### Aborting Streams

```python
result = await l0.run(stream=my_stream)

# Start consuming
async for event in result:
    if should_stop(event):
        result.abort()
        break
    process(event)

# Check if aborted
print(result.state.aborted)  # True
```

---

## Utility Functions

### JSON Utilities

```python
from l0._utils import auto_correct_json, extract_json_from_markdown

# Fix common JSON errors
fixed = auto_correct_json('{"a": 1,}')  # '{"a": 1}'
fixed = auto_correct_json('{"a": {"b": 1}')  # '{"a": {"b": 1}}'
fixed = auto_correct_json('[1, 2')  # '[1, 2]'

# Extract JSON from markdown
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
# Outputs: [l0] DEBUG: Starting L0 stream: ...
```

---

## Types

### Stream

```python
class Stream:
    """Async iterator result with state and abort attached."""
    
    state: State                              # Runtime state
    abort: Callable[[], None]                 # Abort the stream
    errors: list[Exception]                   # Errors encountered
    
    def __aiter__(self) -> Stream: ...
    async def __anext__(self) -> Event: ...
    async def __aenter__(self) -> Stream: ...
    async def __aexit__(...) -> bool: ...
    async def read(self) -> str:
        """Consume the stream and return the full text content."""
        ...
```

### LazyStream

```python
class LazyStream:
    """Lazy stream wrapper - no await needed on creation.
    
    Like httpx.AsyncClient() or aiohttp.ClientSession(), this returns
    immediately and only does async work when you iterate or read.
    """
    
    state: State                              # Runtime state (after started)
    errors: list[Exception]                   # Errors encountered
    
    def abort(self) -> None: ...
    def __aiter__(self) -> LazyStream: ...
    async def __anext__(self) -> Event: ...
    async def __aenter__(self) -> LazyStream: ...
    async def __aexit__(...) -> bool: ...
    async def read(self) -> str:
        """Consume the stream and return the full text content."""
        ...
```

### State

```python
@dataclass
class State:
    content: str = ""
    checkpoint: str = ""
    token_count: int = 0
    model_retry_count: int = 0
    network_retry_count: int = 0
    fallback_index: int = 0
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

### Event

```python
@dataclass
class Event:
    type: EventType
    text: str | None = None                   # Token content
    data: dict[str, Any] | None = None        # Tool call / misc data
    error: Exception | None = None            # Error (for error events)
    usage: dict[str, int] | None = None       # Token usage
    timestamp: float | None = None            # Event timestamp

    # Type check properties
    @property
    def is_token(self) -> bool: ...
    @property
    def is_message(self) -> bool: ...
    @property
    def is_data(self) -> bool: ...
    @property
    def is_progress(self) -> bool: ...
    @property
    def is_tool_call(self) -> bool: ...
    @property
    def is_error(self) -> bool: ...
    @property
    def is_complete(self) -> bool: ...
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

### Retry

```python
@dataclass
class Retry:
    attempts: int = 3                 # Model errors only
    max_retries: int = 6              # Absolute cap
    base_delay: float = 1.0           # Seconds
    max_delay: float = 10.0           # Seconds
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER
```

### Timeout

```python
@dataclass
class Timeout:
    initial_token: float = 5.0        # Seconds to first token
    inter_token: float = 10.0         # Seconds between tokens
```

### GuardrailRule

```python
@dataclass
class GuardrailRule:
    name: str
    check: Callable[[State], list[GuardrailViolation]]
    description: str | None = None
    streaming: bool = True
    severity: Severity = "error"
    recoverable: bool = True
```

### GuardrailViolation

```python
@dataclass
class GuardrailViolation:
    rule: str
    message: str
    severity: Severity
    recoverable: bool = True
    position: int | None = None
    timestamp: float | None = None
    context: dict[str, Any] | None = None
    suggestion: str | None = None
```

### Severity

```python
Severity = Literal["warning", "error", "fatal"]
```

### BackoffStrategy

```python
class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"    # delay * 2^attempt
    LINEAR = "linear"              # delay * (attempt + 1)
    FIXED = "fixed"                # constant delay
    FULL_JITTER = "full-jitter"    # random(0, exponential)
    FIXED_JITTER = "fixed-jitter"  # base/2 + random(base/2)
```

### ErrorCategory

```python
class ErrorCategory(str, Enum):
    NETWORK = "network"
    TRANSIENT = "transient"
    MODEL = "model"
    CONTENT = "content"
    PROVIDER = "provider"
    FATAL = "fatal"
    INTERNAL = "internal"
```

---

## Imports

### Main Import (Recommended)

```python
import l0

# Simple wrapping (no await needed!)
result = l0.wrap(stream, guardrails=l0.Guardrails.recommended())
text = await result.read()

# Or with retries/fallbacks
result = await l0.run(
    stream=lambda: create_stream(),
    fallbacks=[lambda: backup_stream()],
    guardrails=l0.Guardrails.recommended(),
)

async for event in result:
    if event.is_token:
        print(event.text, end="")
```

### Direct Imports

```python
from l0 import (
    # Core
    wrap,
    run,
    l0,  # Alias to run()
    Stream,
    LazyStream,
    State,
    Event,
    EventType,
    
    # Retry
    Retry,
    Timeout,
    TimeoutError,
    BackoffStrategy,
    
    # Guardrails
    Guardrails,  # Class with .recommended(), .strict(), etc.
    GuardrailRule,
    GuardrailViolation,
    json_rule,
    strict_json_rule,
    pattern_rule,
    zero_output_rule,
    stall_rule,
    repetition_rule,
    
    # Structured output
    structured,
    
    # Parallel operations
    parallel,
    race,
    batched,
    consensus,
    
    # Adapters
    Adapters,
    Adapter,
    OpenAIAdapter,
    
    # Observability
    EventBus,
    ObservabilityEvent,
    ObservabilityEventType,
    
    # Errors
    ErrorCategory,
    categorize_error,
    
    # Utilities
    consume_stream,
    get_text,
    enable_debug,
)
```

### Public Exports

| Category | Exports |
| -------- | ------- |
| Core | `wrap`, `run`, `l0` (alias), `Stream`, `LazyStream`, `State`, `Event`, `EventType` |
| Retry | `Retry`, `Timeout`, `TimeoutError`, `BackoffStrategy` |
| Guardrails | `Guardrails`, `GuardrailRule`, `GuardrailViolation`, `json_rule`, `strict_json_rule`, `pattern_rule`, `zero_output_rule`, `stall_rule`, `repetition_rule` |
| Structured | `structured` |
| Parallel | `parallel`, `race`, `batched`, `consensus` |
| Adapters | `Adapters`, `Adapter`, `OpenAIAdapter` |
| Observability | `EventBus`, `ObservabilityEvent`, `ObservabilityEventType` |
| Errors | `ErrorCategory`, `categorize_error` |
| Utilities | `consume_stream`, `get_text`, `enable_debug` |
| Version | `__version__` |

---

## See Also

- [README.md](./README.md) - Quick start guide
- [PLAN.md](./PLAN.md) - Implementation plan and design decisions
