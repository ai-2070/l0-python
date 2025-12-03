# L0 - Deterministic Streaming Execution Substrate (DSES) for AI

### The missing reliability and observability layer for all AI streams.

<p align="center">
  <a href="https://pypi.org/project/l0/">
    <img src="https://img.shields.io/pypi/v/l0?color=brightgreen&label=pypi" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/l0/">
    <img src="https://img.shields.io/pypi/pyversions/l0" alt="Python versions">
  </a>
  <img src="https://img.shields.io/badge/types-included-blue?logo=python&logoColor=white" alt="Types Included">
  <img src="https://img.shields.io/badge/asyncio-native-blue" alt="Asyncio Native">
  <img src="https://img.shields.io/badge/license-Apache_2.0-green" alt="Apache 2.0 License">
</p>

> LLMs produce high-value reasoning over a low-integrity transport layer.
> Streams stall, drop tokens, reorder events, violate timing guarantees, and expose no deterministic contract.
>
> This breaks retries. It breaks supervision. It breaks reproducibility.
> It makes reliable AI systems impossible to build on top of raw provider streams.
>
> **L0 is the deterministic execution substrate that fixes the transport -
> with guardrails designed for the streaming layer itself: stream-neutral, pattern-based, loop-safe, and timing-aware.**

L0 adds deterministic execution, fallbacks, retries, network protection, guardrails, drift detection, and tool tracking to any LLM stream - turning raw model output into production-grade behavior.

It works with **OpenAI** and **LiteLLM** (100+ providers including Anthropic, Cohere, Bedrock, Vertex, Gemini). Supports **tool calls** and provides full observability.

_Production-grade reliability. Just pass your stream. L0'll take it from here._

```
   Any AI Stream                    L0 Layer                         Your App
 ─────────────────    ┌──────────────────────────────────────┐    ─────────────
                      │                                      │
   OpenAI / LiteLLM   │   Retry · Fallback · Resume          │      Reliable
   Custom Streams  ──▶│   Guardrails · Timeouts · Consensus  │─────▶ Output
                      │   Full Observability                 │
                      │                                      │
                      └──────────────────────────────────────┘
 ─────────────────                                                ─────────────
                           L0 = Token-Level Reliability
```

## Features

| Feature                                  | Description                                                                                                                                                                                           |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Smart Retries**                        | Model-aware retries with fixed-jitter backoff. Automatic retries for zero-token output, network stalls, and provider overloads.                                                                      |
| **Network Protection**                   | Automatic recovery from dropped streams, slow responses, 429/503 load shedding, DNS errors, and partial chunks.                                                                                       |
| **Model Fallbacks**                      | Automatically fallback to secondary models (e.g., GPT-4o → GPT-4o-mini → Claude) with full retry logic.                                                                                               |
| **Zero-Token/Stall Protection**          | Detects when model produces nothing or stalls mid-stream. Automatically retries or switches to fallbacks.                                                                                             |
| **Drift Detection**                      | Detects repetition, stalls, and format drift before corruption propagates.                                                                                                                            |
| **Structured Output**                    | Guaranteed-valid JSON with Pydantic. Auto-corrects missing braces, commas, and markdown fences.                                                                                                       |
| **JSON Auto-Healing**                    | Automatic correction of truncated or malformed JSON (missing braces, brackets, quotes), and repair of broken Markdown code fences.                                                                   |
| **Guardrails**                           | JSON and pattern validation with fast streaming checks. Delta-only checks run sync; full-content scans defer to async.                                                                                |
| **Race: Fastest-Model Wins**             | Run multiple models or providers in parallel and return the fastest valid stream. Ideal for ultra-low-latency chat.                                                                                  |
| **Parallel: Fan-Out / Fan-In**           | Start multiple streams simultaneously and collect structured or summarized results. Perfect for agent-style multi-model workflows.                                                                    |
| **Consensus: Agreement Across Models**   | Combine multiple model outputs using unanimous, majority, or best-match consensus. Guarantees high-confidence generation.                                                                             |
| **Central Event Bus**                    | Full observability into every stream phase via `on_event` callback with structured event types.                                                                                                       |
| **Pure asyncio**                         | No compatibility layers (no anyio/trio). Native Python async for full determinism and performance.                                                                                                    |
| **Own Retry Logic**                      | No external dependencies (no tenacity). L0 controls all retry behavior for predictable execution.                                                                                                     |
| **Type-Safe**                            | Full type hints with `py.typed` marker. Passes mypy strict mode.                                                                                                                                      |
| **Minimal Dependencies**                 | Only httpx, pydantic, orjson, typing-extensions, uuid6. No heavy abstractions.                                                                                                                           |

## Quick Start

### With OpenAI SDK

```python
import asyncio
from openai import AsyncOpenAI
import l0

client = AsyncOpenAI()

async def main():
    result = await l0.run(
        stream=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        ),
        guardrails=l0.Guardrails.recommended(),
    )

    async for event in result:
        if event.is_token:
            print(event.text, end="", flush=True)

asyncio.run(main())
```

### With LiteLLM (100+ Providers)

```python
import asyncio
import litellm
import l0

async def main():
    result = await l0.run(
        stream=lambda: litellm.acompletion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        ),
        guardrails=l0.Guardrails.recommended(),
    )

    # Or just get full text
    text = await result.read()
    print(text)

asyncio.run(main())
```

### Simple Wrapping (No Lambda)

```python
import l0
import litellm

async def main():
    # Create your stream
    stream = litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

    # l0.wrap() returns immediately - no await needed!
    result = l0.wrap(stream, guardrails=l0.Guardrails.recommended())
    
    # Read full text
    text = await result.read()
    print(text)

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

### Expanded Configuration

```python
import asyncio
import l0
from openai import AsyncOpenAI

client = AsyncOpenAI()
prompt = "Write a haiku about coding"

async def main():
    result = await l0.run(
        # Primary model stream
        stream=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),

        # Optional: Fallback models
        fallbacks=[
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ),
        ],

        # Optional: Guardrails
        guardrails=l0.Guardrails.recommended(),
        # Or strict:
        # guardrails=l0.Guardrails.strict(),
        # Or custom:
        # guardrails=[l0.json_rule(), l0.pattern_rule()],

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

        # Optional: Event callback for observability
        on_event=lambda event: print(f"[{event.type}]"),

        # Optional: Metadata attached to all events
        meta={"user_id": "123", "session": "abc"},
    )

    # Stream events with Pythonic properties
    async for event in result:
        if event.is_token:
            print(event.text, end="")
        elif event.is_tool_call:
            print(f"Tool: {event.data}")
        elif event.is_complete:
            print(f"\nUsage: {event.usage}")

    # Access state anytime
    print(f"\nTokens: {result.state.token_count}")
    print(f"Duration: {result.state.duration}s")

asyncio.run(main())
```

**See Also: [API.md](./API.md) - Complete API reference**

## Core Features

| Feature                                           | Description                                                     |
| ------------------------------------------------- | --------------------------------------------------------------- |
| [Streaming Runtime](#streaming-runtime)           | Token-by-token normalization, checkpoints, unified events       |
| [Retry Logic](#retry-logic)                       | Smart retries with backoff, network vs model error distinction  |
| [Network Protection](#network-protection)         | Auto-recovery from 12+ network failure types                    |
| [Structured Output](#structured-output)           | Guaranteed valid JSON with Pydantic                             |
| [Fallback Models](#fallback-models)               | Sequential fallback when primary model fails                    |
| [Guardrails](#guardrails)                         | JSON validation, pattern detection, drift detection             |
| [Consensus](#consensus)                           | Multi-model agreement with voting strategies                    |
| [Parallel Operations](#parallel-operations)       | Race, batch, pool patterns for concurrent LLM calls             |
| [Custom Adapters](#custom-adapters)               | Bring your own adapter for any LLM provider                     |
| [Observability](#observability)                   | Central event bus with 25+ event types                          |
| [Error Handling](#error-handling)                 | Typed errors with categorization and recovery hints             |

---

## Streaming Runtime

L0 wraps LLM streams with deterministic behavior:

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ),
    timeout=l0.Timeout(
        initial_token=5.0,   # Seconds to first token
        inter_token=10.0,    # Seconds between tokens
    ),
)

# Unified event format with Pythonic properties
async for event in result:
    if event.is_token:
        print(event.text, end="")
    elif event.is_tool_call:
        print(f"Tool: {event.data}")
    elif event.is_complete:
        print("\nComplete")
    elif event.is_error:
        print(f"Error: {event.error}")

# Access state anytime
print(result.state.content)       # Full accumulated content
print(result.state.token_count)   # Total tokens received
print(result.state.checkpoint)    # Last stable checkpoint
```

> **Note:** Free and low-priority models may take **3–7 seconds** before emitting the first token and **10 seconds** between tokens.

---

## Retry Logic

Smart retry system that distinguishes network errors from model errors:

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(..., stream=True),
    retry=l0.Retry(
        attempts=3,                              # Model errors only (default: 3)
        max_retries=6,                           # Absolute cap (default: 6)
        base_delay=1.0,                          # Seconds (default: 1.0)
        max_delay=10.0,                          # Seconds (default: 10.0)
        strategy=l0.BackoffStrategy.FIXED_JITTER,
    ),
)
```

### Backoff Strategies

| Strategy | Formula | Description |
| -------- | ------- | ----------- |
| `EXPONENTIAL` | `delay * 2^attempt` | Classic exponential backoff |
| `LINEAR` | `delay * (attempt + 1)` | Linear increase |
| `FIXED` | `delay` | Constant delay |
| `FULL_JITTER` | `random(0, exponential)` | Random between 0 and exponential |
| `FIXED_JITTER` | `base/2 + random(base/2)` | AWS-style fixed jitter (default) |

### Retry Behavior

| Error Type           | Retries | Counts Toward `attempts` | Counts Toward `max_retries` |
| -------------------- | ------- | ------------------------ | --------------------------- |
| Network disconnect   | Yes     | No                       | Yes                         |
| Zero output          | Yes     | No                       | Yes                         |
| Timeout              | Yes     | No                       | Yes                         |
| 429 rate limit       | Yes     | No                       | Yes                         |
| 503 server error     | Yes     | No                       | Yes                         |
| Guardrail violation  | Yes     | **Yes**                  | Yes                         |
| Drift detected       | Yes     | **Yes**                  | Yes                         |
| Auth error (401/403) | No      | -                        | -                           |

---

## Network Protection

Automatic detection and recovery from network failures:

```python
from l0.errors import categorize_error
from l0.types import ErrorCategory

try:
    result = await l0.run(stream=my_stream)
except Exception as error:
    category = categorize_error(error)
    
    if category == ErrorCategory.NETWORK:
        print("Network error - will auto-retry")
    elif category == ErrorCategory.TRANSIENT:
        print("Transient error (429/503) - will auto-retry")
    elif category == ErrorCategory.FATAL:
        print("Fatal error - cannot retry")
```

### Detected Network Error Patterns

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

---

## Structured Output

Guaranteed valid JSON matching your Pydantic schema:

```python
from pydantic import BaseModel
import l0

class UserProfile(BaseModel):
    name: str
    age: int
    email: str

result = await l0.structured(
    schema=UserProfile,
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Generate user data as JSON"}],
        stream=True,
    ),
    auto_correct=True,  # Fix trailing commas, missing braces, etc.
)

# Type-safe access
print(result.name)   # str
print(result.age)    # int
print(result.email)  # str
```

### JSON Auto-Correction

L0 automatically fixes common JSON errors:

```python
from l0._utils import auto_correct_json, extract_json_from_markdown

# Fix trailing commas
auto_correct_json('{"a": 1,}')  # '{"a": 1}'

# Balance braces
auto_correct_json('{"a": {"b": 1}')  # '{"a": {"b": 1}}'

# Extract from markdown
extract_json_from_markdown('''
Here's the JSON:
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
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ),
    fallbacks=[
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
        lambda: litellm.acompletion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
    ],
)

# Check which model succeeded
print(result.state.fallback_index)  # 0 = primary, 1+ = fallback
```

### Fall-Through vs Race

| Pattern      | Execution                   | Cost               | Best For                          |
| ------------ | --------------------------- | ------------------ | --------------------------------- |
| Fall-through | Sequential, next on failure | Low (pay for 1)    | High availability, cost-sensitive |
| Race         | Parallel, first wins        | High (pay for all) | Low latency, speed-critical       |

```python
# Fall-through: Try models sequentially
result = await l0.run(
    stream=lambda: openai_stream(),
    fallbacks=[
        lambda: anthropic_stream(),
        lambda: local_model_stream(),
    ],
)

# Race: All models simultaneously, first wins
result = await l0.race([
    lambda: openai_stream(),
    lambda: anthropic_stream(),
])
```

---

## Guardrails

Pure functions that validate streaming output without rewriting it:

```python
import l0

result = await l0.run(
    stream=my_stream,
    guardrails=[
        l0.json_rule(),           # Validates JSON structure
        l0.pattern_rule(),        # Detects "As an AI..." patterns
        l0.zero_output_rule(),    # Detects empty output
    ],
)
```

### Presets

```python
import l0

# Recommended: JSON + pattern + zero_output
guardrails = l0.Guardrails.recommended()

# Strict: All rules including drift detection
guardrails = l0.Guardrails.strict()

# JSON only
guardrails = l0.Guardrails.json_only()

# None
guardrails = l0.Guardrails.none()
```

### Available Rules

| Rule | Description |
| ---- | ----------- |
| `json_rule()` | Validates JSON structure (balanced braces) |
| `strict_json_rule()` | Validates complete, parseable JSON |
| `pattern_rule()` | Detects "As an AI..." and similar patterns |
| `zero_output_rule()` | Detects empty output |
| `stall_rule()` | Detects token stalls |
| `repetition_rule()` | Detects model looping |

### Custom Guardrails

```python
from l0 import GuardrailRule, GuardrailViolation
from l0.types import State

def max_length_rule(limit: int = 1000) -> GuardrailRule:
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
        streaming=True,
        severity="error",
    )

# Use custom rule
result = await l0.run(
    stream=my_stream,
    guardrails=[max_length_rule(500)],
)
```

### Guardrail Violations

```python
# Access violations from state
for violation in result.state.violations:
    print(f"Rule: {violation.rule}")
    print(f"Message: {violation.message}")
    print(f"Severity: {violation.severity}")
    print(f"Recoverable: {violation.recoverable}")
```

---

## Consensus

Multi-generation consensus for high-confidence results:

```python
import l0

result = await l0.consensus(
    tasks=[
        lambda: model_a(),
        lambda: model_b(),
        lambda: model_c(),
    ],
    strategy="majority",  # or "unanimous", "best"
)
```

### Strategies

| Strategy | Description | Use Case |
| -------- | ----------- | -------- |
| `unanimous` | All results must match exactly | Safety-critical, requires agreement |
| `majority` | Most common result wins (>50%) | Balanced reliability |
| `best` | Return first result | Speed-focused |

---

## Parallel Operations

Run multiple LLM calls concurrently with different patterns:

### Race - First Response Wins

```python
import l0

result = await l0.race([
    lambda: fast_model(),
    lambda: slow_model(),
    lambda: backup_model(),
])
# Returns first successful response, cancels others
```

### Parallel with Concurrency Control

```python
import l0

results = await l0.parallel(
    tasks=[
        lambda: process("Task 1"),
        lambda: process("Task 2"),
        lambda: process("Task 3"),
    ],
    concurrency=2,  # Max 2 concurrent
)
```

### Batch Processing

```python
import l0

async def process(item: str) -> str:
    # Process single item
    return result

results = await l0.batched(
    items=["a", "b", "c", "d", "e"],
    handler=process,
    batch_size=2,
)
```

---

## Custom Adapters

L0 supports custom adapters for integrating any LLM provider:

### Built-in Adapters

| Adapter | Providers | Auto-Detected |
| ------- | --------- | ------------- |
| **OpenAI** | OpenAI SDK | Yes |
| **LiteLLM** | 100+ providers | Yes |

### Building Custom Adapters

```python
from collections.abc import AsyncIterator
from typing import Any
import l0
from l0 import Event, EventType, Adapters

class MyProviderAdapter:
    name = "my_provider"
    
    def detect(self, stream: Any) -> bool:
        """Check if this adapter can handle the given stream."""
        return "my_provider" in type(stream).__module__
    
    async def wrap(self, stream: Any) -> AsyncIterator[Event]:
        """Convert provider stream to L0 events."""
        usage = None
        
        async for chunk in stream:
            # Emit text tokens
            if chunk.text:
                yield Event(type=EventType.TOKEN, text=chunk.text)
            
            # Emit tool calls
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    yield Event(
                        type=EventType.TOOL_CALL,
                        data={
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
            
            # Track usage
            if chunk.usage:
                usage = {
                    "input_tokens": chunk.usage.input,
                    "output_tokens": chunk.usage.output,
                }
        
        # Emit completion
        yield Event(type=EventType.COMPLETE, usage=usage)

# Register for auto-detection
Adapters.register(MyProviderAdapter())
```

### Adapter Protocol

Adapters MUST:
- Preserve text exactly (no trimming, no modification)
- Convert errors to error events (never throw)
- Emit complete event exactly once at end

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

---

## Observability

Central event bus for all L0 observability:

```python
from l0 import ObservabilityEvent, ObservabilityEventType

def on_event(event: ObservabilityEvent):
    print(f"[{event.type}] stream={event.stream_id}")
    print(f"  ts={event.ts}, meta={event.meta}")

result = await l0.run(
    stream=my_stream,
    on_event=on_event,
    meta={"user_id": "123", "session": "abc"},
)
```

### Event Types

| Phase | Events | Purpose |
| ----- | ------ | ------- |
| Session | `SESSION_START` → `SESSION_END` | Session lifecycle |
| Stream | `STREAM_INIT` → `STREAM_READY` | Stream initialization |
| Retry | `RETRY_START` → `RETRY_ATTEMPT` → `RETRY_END` / `RETRY_GIVE_UP` | Retry loop |
| Fallback | `FALLBACK_START` → `FALLBACK_END` | Model switching |
| Guardrail | `GUARDRAIL_PHASE_START` → `GUARDRAIL_RULE_RESULT` → `GUARDRAIL_PHASE_END` | Validation |
| Network | `NETWORK_ERROR` → `NETWORK_RECOVERY` | Connection lifecycle |
| Completion | `COMPLETE` / `ERROR` | Final status |

---

## Error Handling

L0 provides detailed error context for debugging and recovery:

```python
from l0.errors import categorize_error
from l0.types import ErrorCategory

try:
    result = await l0.run(stream=my_stream)
except Exception as error:
    category = categorize_error(error)
    
    match category:
        case ErrorCategory.NETWORK:
            print("Network error - transient, will retry")
        case ErrorCategory.TRANSIENT:
            print("Transient error (429/503) - will retry")
        case ErrorCategory.MODEL:
            print("Model error - counts toward retry limit")
        case ErrorCategory.CONTENT:
            print("Content error - guardrail/drift violation")
        case ErrorCategory.FATAL:
            print("Fatal error - cannot retry")
        case ErrorCategory.INTERNAL:
            print("Internal error - bug, don't retry")
```

### Error Categories

| Category | Description | Retry Behavior |
| -------- | ----------- | -------------- |
| `NETWORK` | Connection drops, DNS, SSL errors | Retries until `max_retries` is reached (doesn't consume `attempts`) |
| `TRANSIENT` | 429 rate limit, 503 server error | Retries until `max_retries` is reached (doesn't consume `attempts`) |
| `MODEL` | Model refused, malformed response | Counts toward retry limit |
| `CONTENT` | Guardrail violation, drift | Counts toward retry limit |
| `PROVIDER` | API errors (may be retryable) | Depends on status |
| `FATAL` | Auth errors (401/403) | No retry |
| `INTERNAL` | Bugs, internal errors | No retry |

---

## Installation

```bash
# Basic installation
pip install l0

# With OpenAI support
pip install l0[openai]

# With LiteLLM (100+ providers)
pip install l0[litellm]

# With observability
pip install l0[observability]

# Development
pip install l0[dev]
```

Or with uv:

```bash
uv add l0
uv add l0 --extra openai
uv add l0 --extra litellm
```

### Dependencies

| Package | Purpose |
| ------- | ------- |
| `httpx` | HTTP client |
| `pydantic` | Schema validation |
| `orjson` | Fast JSON |
| `uuid6` | UUIDv7 for stream IDs |
| `typing-extensions` | Type hints |

### Optional Dependencies

| Extra | Packages |
| ----- | -------- |
| `openai` | `openai>=1.30` |
| `litellm` | `litellm>=1.40` |
| `observability` | `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-httpx`, `sentry-sdk` |
| `speed` | `uvloop` (Unix only) |
| `dev` | `pytest`, `pytest-asyncio`, `pytest-cov`, `mypy`, `ruff` |

---

## Philosophy

- **No magic** - Everything is explicit and predictable
- **Streaming-first** - Built for real-time token delivery
- **Signals, not rewrites** - Guardrails detect issues, don't modify output
- **Model-agnostic** - Works with any provider via adapters
- **Pure asyncio** - No compatibility layers, native Python async
- **Own retry logic** - No tenacity, full control over behavior

---

## Documentation

| Guide | Description |
| ----- | ----------- |
| [API.md](./API.md) | Complete API reference |
| [PLAN.md](./PLAN.md) | Implementation plan and design decisions |

---

## License

Apache-2.0
