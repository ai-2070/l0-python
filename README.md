# L0 - Deterministic Streaming Execution Substrate (DSES) for AI

### The missing reliability and observability layer for all AI streams.

![L0: The Missing AI Reliability Substrate](img/l0-banner.jpg)

<p align="center">
  <a href="https://pypi.org/project/ai2070-l0/">
    <img src="https://img.shields.io/pypi/v/ai2070-l0?color=brightgreen&label=pypi" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/ai2070-l0/">
    <img src="https://img.shields.io/pypi/pyversions/ai2070-l0" alt="Python versions">
  </a>
  <img src="https://img.shields.io/badge/types-included-blue?logo=python&logoColor=white" alt="Types Included">
  <img src="https://img.shields.io/badge/asyncio-native-blue" alt="Asyncio Native">
  <img src="https://img.shields.io/badge/tests-1800+-blue" alt="1800+ Tests">
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

```bash
pip install ai2070-l0
```

_Production-grade reliability. Just pass your stream. L0'll take it from here._

L0 includes 1,800+ tests covering all major reliability features.

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

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [Streaming Runtime](#streaming-runtime)
- [Retry Logic](#retry-logic)
- [Network Protection](#network-protection)
- [Structured Output](#structured-output)
- [Fallback Models](#fallback-models)
- [Last-Known-Good Token Resumption](#last-known-good-token-resumption)
- [Guardrails](#guardrails)
- [Consensus](#consensus)
- [Parallel Operations](#parallel-operations)
- [Custom Adapters](#custom-adapters)
- [Lifecycle Callbacks](#lifecycle-callbacks)
- [Observability Events](#observability-events)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Installation](#installation)
- [Philosophy](#philosophy)
- [Documentation](#documentation)
- [License](#license)

## Features

| Feature                                        | Description                                                                                                                                                                                           |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🔁 Smart Retries**                           | Model-aware retries with fixed-jitter backoff. Automatic retries for zero-token output, network stalls, and provider overloads.                                                                      |
| **🌐 Network Protection**                      | Automatic recovery from dropped streams, slow responses, 429/503 load shedding, DNS errors, and partial chunks.                                                                                       |
| **🔀 Model Fallbacks**                         | Automatically fallback to secondary models (e.g., GPT-4o → GPT-4o-mini → Claude) with full retry logic.                                                                                               |
| **💥 Zero-Token/Stall Protection**             | Detects when model produces nothing or stalls mid-stream. Automatically retries or switches to fallbacks.                                                                                             |
| **📍 Last-Known-Good Token Resumption**        | `continue_from_last_good_token` resumes from the last checkpoint on timeout or failure. No lost tokens.                                                                                              |
| **🧠 Drift Detection**                         | Detects repetition, stalls, and format drift before corruption propagates.                                                                                                                            |
| **🧱 Structured Output**                       | Guaranteed-valid JSON with Pydantic. Auto-corrects missing braces, commas, and markdown fences.                                                                                                       |
| **🩹 JSON Auto-Healing**                       | Automatic correction of truncated or malformed JSON (missing braces, brackets, quotes), and repair of broken Markdown code fences.                                                                   |
| **🛡️ Guardrails**                              | JSON, Markdown, and pattern validation with fast streaming checks. Delta-only checks run sync; full-content scans defer to async.                                                                    |
| **⚡ Race: Fastest-Model Wins**                | Run multiple models or providers in parallel and return the fastest valid stream. Ideal for ultra-low-latency chat.                                                                                  |
| **🌿 Parallel: Fan-Out / Fan-In**              | Start multiple streams simultaneously and collect structured or summarized results. Perfect for agent-style multi-model workflows.                                                                    |
| **🧩 Consensus: Agreement Across Models**      | Combine multiple model outputs using unanimous, majority, or best-match consensus. Guarantees high-confidence generation.                                                                             |
| **🔔 Lifecycle Callbacks**                     | `on_start`, `on_complete`, `on_error`, `on_event`, `on_violation`, `on_retry`, `on_fallback`, `on_tool_call` - full observability into every stream phase.                                           |
| **📡 Streaming-First Runtime**                 | Thin, deterministic wrapper with unified event types (`token`, `error`, `complete`) for easy UIs.                                                                                                     |
| **📼 Central Event Bus**                       | Full observability into every stream phase via `on_event` callback with 25+ structured event types.                                                                                                   |
| **🔌 Custom Adapters (BYOA)**                  | Bring your own adapter for any LLM provider. Built-in adapters for OpenAI and LiteLLM.                                                                                                                |
| **📦 Raw Chunk Access**                        | Access original provider chunks (e.g., OpenAI's `ChatCompletionChunk`) via `stream.raw()` for provider-specific processing.                                                                          |
| **⚡ Pure asyncio**                            | No compatibility layers (no anyio/trio). Native Python async for full determinism and performance.                                                                                                    |
| **🔧 Own Retry Logic**                         | No external dependencies (no tenacity). L0 controls all retry behavior for predictable execution.                                                                                                     |
| **📝 Type-Safe**                               | Full type hints with `py.typed` marker. Passes mypy strict mode.                                                                                                                                      |
| **📦 Minimal Dependencies**                    | Only httpx, pydantic, orjson, typing-extensions, uuid6. No heavy abstractions.                                                                                                                        |
| **🧪 Battle-Tested**                           | 1,800+ unit tests and 100+ integration tests validating real streaming, retries, and advanced behavior.                                                                                               |

## Quick Start

### Wrap Your Client (Recommended)

```python
import asyncio
from openai import AsyncOpenAI
import l0

async def main():
    # Wrap the client once - L0 reliability is automatic
    client = l0.wrap(AsyncOpenAI())

    # Use normally - no lambdas needed!
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

    # Stream with L0 events
    async for event in response:
        if event.is_token:
            print(event.text, end="", flush=True)

    # Or read all at once
    text = await response.read()

asyncio.run(main())
```

### With Configuration

```python
import l0
from openai import AsyncOpenAI

# Configure once, use everywhere
client = l0.wrap(
    AsyncOpenAI(),
    guardrails=l0.Guardrails.recommended(),
    retry=l0.Retry(max_attempts=5),
    timeout=l0.Timeout(initial_token=10.0, inter_token=30.0),
    continue_from_last_good_token=True,  # Resume from checkpoint on failure
)

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
```

### With LiteLLM (100+ Providers)

```python
import asyncio
import litellm
import l0

async def main():
    # For LiteLLM, use l0.run() with a factory function
    result = await l0.run(
        stream=lambda: litellm.acompletion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        ),
        guardrails=l0.Guardrails.recommended(),
    )

    # Read full text
    text = await result.read()
    print(text)

asyncio.run(main())
```

### Wrap a Raw Stream (Simple Cases)

```python
import l0

async def main():
    # For one-off streams without retry support
    raw_stream = await client.chat.completions.create(..., stream=True)
    
    result = l0.wrap(raw_stream)
    text = await result.read()
```

### Expanded Configuration

```python
import asyncio
import l0
from openai import AsyncOpenAI

prompt = "Write a haiku about coding"

async def main():
    # Wrap client with full configuration
    client = l0.wrap(
        AsyncOpenAI(),
        
        # Guardrails
        guardrails=l0.Guardrails.recommended(),
        
        # Retry configuration
        retry=l0.Retry(
            attempts=3,                              # LLM errors only
            max_retries=6,                           # Total (LLM + network)
            base_delay=1.0,                          # Seconds
            max_delay=10.0,                          # Seconds
            strategy=l0.BackoffStrategy.FIXED_JITTER,
        ),
        
        # Timeout configuration
        timeout=l0.Timeout(
            initial_token=5.0,   # Seconds to first token
            inter_token=10.0,    # Seconds between tokens
        ),
        
        # Checkpoint resumption (resume from last good token on failure)
        continue_from_last_good_token=True,
        
        # Event callback for observability
        on_event=lambda event: print(f"[{event.type}]"),
        
        # Metadata attached to all events
        meta={"user_id": "123", "session": "abc"},
    )

    # Use the wrapped client normally
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    # Stream events with Pythonic properties
    async for event in response:
        if event.is_token:
            print(event.text, end="")
        elif event.is_tool_call:
            print(f"Tool: {event.data}")
        elif event.is_complete:
            print(f"\nUsage: {event.usage}")

    # Access state anytime
    print(f"\nTokens: {response.state.token_count}")
    print(f"Duration: {response.state.duration}s")

asyncio.run(main())
```

### Using l0.run() with Fallbacks

For fallback models, use `l0.run()` with factory functions:

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
    ],
    guardrails=l0.Guardrails.recommended(),
    continue_from_last_good_token=True,
)
```

**See Also: [API.md](./API.md) - Complete API reference**

## Core Features

| Feature                                                               | Description                                                     |
| --------------------------------------------------------------------- | --------------------------------------------------------------- |
| [Streaming Runtime](#streaming-runtime)                               | Token-by-token normalization, checkpoints, resumable generation |
| [Retry Logic](#retry-logic)                                           | Smart retries with backoff, network vs model error distinction  |
| [Network Protection](#network-protection)                             | Auto-recovery from 12+ network failure types                    |
| [Structured Output](#structured-output)                               | Guaranteed valid JSON with Pydantic                             |
| [Fallback Models](#fallback-models)                                   | Sequential fallback when primary model fails                    |
| [Last-Known-Good Token Resumption](#last-known-good-token-resumption) | Resume from last checkpoint on retry/fallback (opt-in)          |
| [Guardrails](#guardrails)                                             | JSON validation, pattern detection, drift detection             |
| [Consensus](#consensus)                                               | Multi-model agreement with voting strategies                    |
| [Parallel Operations](#parallel-operations)                           | Race, batch, pool patterns for concurrent LLM calls             |
| [Custom Adapters](#custom-adapters)                                   | Bring your own adapter for any LLM provider                     |
| [Lifecycle Callbacks](#lifecycle-callbacks)                           | Full observability into every stream phase                      |
| [Observability Events](#observability-events)                         | Central event bus with 25+ structured event types               |
| [Error Handling](#error-handling)                                     | Typed errors with categorization and recovery hints             |
| [Testing](#testing)                                                   | 1,800+ tests covering all features                              |

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

# Extract from markdown (handles ```json ... ``` fences)
markdown_text = 'Here is the JSON:\n```json\n{"key": "value"}\n```'
extract_json_from_markdown(markdown_text)  # '{"key": "value"}'
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

## Last-Known-Good Token Resumption

When a stream fails mid-generation, L0 can resume from the last known good checkpoint instead of starting over. This preserves already-generated content and reduces latency on retries.

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ),
    retry=l0.Retry(attempts=3),

    # Enable continuation from last checkpoint (opt-in)
    continue_from_last_good_token=True,
)

# Check if continuation was used
print(result.state.resumed)        # True if resumed from checkpoint
print(result.state.resume_point)   # The checkpoint content
print(result.state.resume_from)    # Character offset where resume occurred
```

### How It Works

1. L0 maintains a checkpoint of successfully received tokens (every N tokens, configurable via `check_intervals`)
2. When a retry or fallback is triggered, the checkpoint is validated against guardrails and drift detection
3. If validation passes, the checkpoint content is emitted first to the consumer
4. The `build_continuation_prompt` callback (if provided) is called to allow updating the prompt for continuation
5. Telemetry tracks whether continuation was enabled, used, and the checkpoint details

### Using build_continuation_prompt

To have the LLM actually continue from where it left off (rather than just replaying tokens locally), use `build_continuation_prompt` to modify the prompt:

```python
continuation_prompt = ""
original_prompt = "Write a detailed analysis of..."

def build_prompt(checkpoint: str) -> str:
    global continuation_prompt
    continuation_prompt = f"{original_prompt}\n\nContinue from where you left off:\n{checkpoint}"
    return continuation_prompt

result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": continuation_prompt or original_prompt}],
        stream=True,
    ),
    continue_from_last_good_token=True,
    build_continuation_prompt=build_prompt,
    retry=l0.Retry(attempts=3),
)
```

When LLMs continue from a checkpoint, they often repeat words from the end. L0 automatically detects and removes this overlap (enabled by default).

### Example: Resuming After Network Error

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a detailed analysis of..."}],
        stream=True,
    ),
    fallbacks=[
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
    ],
    retry=l0.Retry(attempts=3),
    continue_from_last_good_token=True,
    check_intervals={"checkpoint": 10},  # Save checkpoint every 10 tokens
    on_event=lambda e: print(f"[{e.type}]"),
)

async for event in result:
    if event.is_token:
        print(event.text, end="", flush=True)

# Check telemetry for continuation usage
if result.state.resumed:
    print(f"\nResumed from checkpoint of length: {len(result.state.resume_point)}")
```

### Checkpoint Validation

Before using a checkpoint for continuation, L0 validates it:

- **Guardrails**: All configured guardrails are run against the checkpoint content
- **Drift Detection**: If enabled, checks for format drift in the checkpoint
- **Fatal Violations**: If any guardrail returns a fatal violation, the checkpoint is discarded and retry starts fresh

### Important Limitations

> ⚠️ **Do NOT use `continue_from_last_good_token` with structured output.**
>
> Continuation works by prepending checkpoint content to the next generation. For JSON/structured output, this can corrupt the data structure because:
>
> - The model may not properly continue the JSON syntax
> - Partial objects could result in invalid JSON
> - Schema validation may fail on malformed output
>
> For structured output, let L0 retry from scratch to ensure valid JSON.

```python
# ✅ GOOD - Text generation with continuation
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write an essay..."}],
        stream=True,
    ),
    continue_from_last_good_token=True,
)

# ❌ BAD - Do NOT use with structured output
result = await l0.structured(
    schema=MySchema,
    stream=lambda: client.chat.completions.create(..., stream=True),
    continue_from_last_good_token=True,  # DON'T DO THIS
)
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
| `Guardrails.json()` | Validates JSON structure (balanced braces) |
| `Guardrails.strict_json()` | Validates complete, parseable JSON |
| `Guardrails.pattern()` | Detects "As an AI..." and similar patterns |
| `Guardrails.zero_output()` | Detects empty output |
| `Guardrails.stall()` | Detects token stalls |
| `Guardrails.repetition()` | Detects model looping |

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

### Async Guardrail Checks

L0 uses a two-path strategy to avoid blocking the streaming loop:

#### Fast Path (Synchronous)

Runs immediately on each chunk for quick checks:

- **Delta-only checks**: Only examines the latest chunk (`context.delta`)
- **Small content**: Full check if total content < 5KB
- **Instant violations**: Blocked words, obvious patterns

```python
# Fast path triggers for:
# - Delta < 1KB
# - Total content < 5KB
# - Any violation found in delta
```

#### Slow Path (Asynchronous)

Deferred to `call_soon()` to avoid blocking:

- **Large content**: Full content scan for content > 5KB
- **Complex rules**: Pattern matching, structure analysis
- **Non-blocking**: Results delivered via callback

```python
from l0.guardrails import (
    run_async_guardrail_check,
    run_guardrail_check_async,
    create_guardrail_engine,
    json_rule,
    GuardrailContext,
)

engine = create_guardrail_engine([json_rule()])
context = GuardrailContext(content="...", completed=False, delta="...")

# Fast/slow path with immediate result if possible
def handle_result(result):
    if result.should_halt:
        print("Halting due to violation!")

result = run_async_guardrail_check(engine, context, handle_result)

if result is not None:
    # Fast path returned immediately
    print(f"Fast path: passed={result.passed}")
else:
    # Deferred to async callback
    print("Waiting for slow path...")

# Always async version (for async/await contexts)
result = await run_guardrail_check_async(engine, context)
print(f"Async result: passed={result.passed}")
```

#### Rule Complexity

| Rule | Complexity | When Checked |
| ---- | ---------- | ------------ |
| `zero_output_rule` | O(1) | Fast path |
| `json_rule` | O(n) | Scans full content |
| `markdown_rule` | O(n) | Scans full content |
| `pattern_rule` | O(n × p) | Scans full content × patterns |

For long outputs, increase `check_intervals["guardrails"]` to reduce frequency:

```python
result = await l0.run(
    stream=my_stream,
    guardrails=l0.Guardrails.recommended(),
    check_intervals={"guardrails": 50},  # Check every 50 tokens instead of default
)
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

## Lifecycle Callbacks

L0 provides callbacks for every phase of stream execution, giving you full observability into the streaming lifecycle:

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ),
    fallbacks=[lambda: client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )],
    guardrails=l0.Guardrails.recommended(),
    continue_from_last_good_token=True,
    retry=l0.Retry(attempts=3),

    # Called when a new execution attempt begins
    on_start=lambda attempt, is_retry, is_fallback: print(
        f"Starting attempt {attempt}" + (" (retry)" if is_retry else "") + (" (fallback)" if is_fallback else "")
    ),

    # Called when stream completes successfully
    on_complete=lambda state: print(f"Completed with {state.token_count} tokens"),

    # Called when an error occurs (before retry/fallback decision)
    on_error=lambda error, will_retry, will_fallback: print(
        f"Error: {error}" + (" Will retry..." if will_retry else "") + (" Will try fallback..." if will_fallback else "")
    ),

    # Called for every L0 event
    on_event=lambda event: print(event.text, end="") if event.is_token else None,

    # Called when a guardrail violation is detected
    on_violation=lambda violation: print(f"Violation: {violation.rule} - {violation.message}"),

    # Called when a retry is triggered
    on_retry=lambda attempt, reason: print(f"Retrying (attempt {attempt}): {reason}"),

    # Called when switching to a fallback model
    on_fallback=lambda index, reason: print(f"Switching to fallback {index}: {reason}"),

    # Called when resuming from checkpoint
    on_resume=lambda checkpoint, token_count: print(f"Resuming from checkpoint ({token_count} tokens)"),

    # Called when a checkpoint is saved
    on_checkpoint=lambda checkpoint, token_count: print(f"Checkpoint saved ({token_count} tokens)"),

    # Called when a timeout occurs
    on_timeout=lambda timeout_type, elapsed_ms: print(f"Timeout: {timeout_type} after {elapsed_ms}ms"),

    # Called when the stream is aborted
    on_abort=lambda token_count, content_length: print(f"Aborted after {token_count} tokens"),

    # Called when drift is detected
    on_drift=lambda types, confidence: print(f"Drift detected: {types} (confidence: {confidence})"),

    # Called when a tool call is detected
    on_tool_call=lambda tool_name, tool_call_id, args: print(f"Tool call: {tool_name} ({tool_call_id})"),
)
```

### Deterministic Lifecycle Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            L0 LIFECYCLE FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

                                ┌──────────┐
                                │  START   │
                                └────┬─────┘
                                     │
                                     ▼
                      ┌───────────────────────────────────┐
                      │ on_start(attempt, False, False)   │
                      └──────────────┬────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              STREAMING PHASE                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         on_event(event)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  During streaming, these callbacks fire as conditions occur:               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │on_checkpoint │  │ on_tool_call │  │   on_drift   │  │  on_timeout  │   │
│  │ (checkpoint, │  │ (tool_name,  │  │ (types,      │  │ (type,       │   │
│  │  token_count)│  │  id, args)   │  │  confidence) │  │  elapsed_ms) │   │
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
             │                     │                 ▼                 ▼
             │                     │          ┌─────────────┐   ┌───────────┐
             │                     │          │on_violation │   │ on_abort  │
             │                     │          └──────┬──────┘   │(token_cnt,│
             │                     │                 │          │content_len│
             │                     ▼                 ▼          └───────────┘
             │              ┌────────────────────────────────┐
             │              │ on_error(error, will_retry,    │
             │              │          will_fallback)        │
             │              └──────────────┬─────────────────┘
             │                             │
             │                 ┌───────────┼───────────┐
             │                 │           │           │
             │                 ▼           ▼           ▼
             │           ┌──────────┐ ┌──────────┐ ┌──────────┐
             │           │  RETRY   │ │ FALLBACK │ │  FATAL   │
             │           └────┬─────┘ └────┬─────┘ └────┬─────┘
             │                │            │            │
             │                ▼            ▼            │
             │          ┌───────────┐ ┌───────────┐     │
             │          │ on_retry  │ │on_fallback│     │
             │          └─────┬─────┘ └─────┬─────┘     │
             │                │             │           │
             │                │    ┌────────┘           │
             │                │    │                    │
             │                ▼    ▼                    │
             │          ┌─────────────────────┐         │
             │          │  Has checkpoint?    │         │
             │          └──────────┬──────────┘         │
             │                YES  │  NO                │
             │                ┌────┴────┐               │
             │                ▼         ▼               │
             │          ┌──────────┐    │               │
             │          │on_resume │    │               │
             │          └────┬─────┘    │               │
             │               │          │               │
             │               ▼          ▼               │
             │          ┌──────────────────────────┐    │
             │          │on_start(attempt, is_retry│    │
             │          │         is_fallback)     │────┼──► Back to STREAMING
             │          └──────────────────────────┘    │
             │                                          │
             ▼                                          ▼
      ┌─────────────┐                            ┌──────────┐
      │ on_complete │                            │  THROW   │
      │   (state)   │                            │  ERROR   │
      └─────────────┘                            └──────────┘
```

### Callback Reference

| Callback        | When Called                            | Signature                                                              |
| --------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| `on_start`      | New execution attempt begins           | `(attempt: int, is_retry: bool, is_fallback: bool) -> None`            |
| `on_complete`   | Stream finished successfully           | `(state: State) -> None`                                               |
| `on_error`      | Error occurred (before retry decision) | `(error: Exception, will_retry: bool, will_fallback: bool) -> None`    |
| `on_event`      | Any streaming event emitted            | `(event: Event) -> None`                                               |
| `on_violation`  | Guardrail violation detected           | `(violation: GuardrailViolation) -> None`                              |
| `on_retry`      | Retry triggered (same model)           | `(attempt: int, reason: str) -> None`                                  |
| `on_fallback`   | Switching to fallback model            | `(index: int, reason: str) -> None`                                    |
| `on_resume`     | Continuing from checkpoint             | `(checkpoint: str, token_count: int) -> None`                          |
| `on_checkpoint` | Checkpoint saved                       | `(checkpoint: str, token_count: int) -> None`                          |
| `on_timeout`    | Timeout occurred                       | `(timeout_type: str, elapsed_ms: int) -> None`                         |
| `on_abort`      | Stream aborted                         | `(token_count: int, content_length: int) -> None`                      |
| `on_drift`      | Drift detected                         | `(types: list[str], confidence: float | None) -> None`                 |
| `on_tool_call`  | Tool call detected                     | `(tool_name: str, tool_call_id: str, args: dict[str, Any]) -> None`    |

> **Note:** All callbacks are fire-and-forget. They execute asynchronously and never block the stream. Errors in callbacks are silently caught and do not affect stream processing.

### Use Cases

```python
import logging

logger = logging.getLogger(__name__)

# Logging and debugging
def on_start(attempt, is_retry, is_fallback):
    logger.info("stream.start", extra={"attempt": attempt, "is_retry": is_retry})

def on_complete(state):
    logger.info("stream.complete", extra={"tokens": state.token_count})

def on_error(err, will_retry, will_fallback):
    logger.error("stream.failed", extra={"error": str(err)})

# Use callbacks with l0
result = await l0.run(
    stream=lambda: client.chat.completions.create(..., stream=True),
    on_start=on_start,
    on_complete=on_complete,
    on_error=on_error,
    on_retry=lambda attempt, reason: show_retrying_indicator(),
    on_fallback=lambda index, reason: show_fallback_notice(),
    on_violation=lambda v: metrics.increment("violations", tags={"rule": v.rule}),
    on_timeout=lambda t, ms: metrics.increment("timeouts", tags={"type": t}),
)

# Stream events
async for event in result:
    if event.is_token:
        append_to_chat(event.text)
```

---

## Observability Events

L0 emits structured lifecycle events for every phase of execution. These events enable replay, profiling, debugging, and supervision.

Central event bus for all L0 observability:

```python
from l0 import ObservabilityEvent, ObservabilityEventType

def on_event(event: ObservabilityEvent):
    print(f"[{event.type}] stream={event.stream_id}")
    print(f"  ts={event.ts}, context={event.context}, meta={event.meta}")

result = await l0.run(
    stream=my_stream,
    on_event=on_event,
    context={"request_id": "req-123", "user_id": "user-456"},
)
```

### Event Types Overview

| Phase | Events | Purpose |
| ----- | ------ | ------- |
| Session | `SESSION_START` → `SESSION_END` | Session lifecycle |
| Stream | `STREAM_INIT` → `STREAM_READY` | Stream initialization |
| Adapter | `ADAPTER_WRAP_START` → `ADAPTER_DETECTED` → `ADAPTER_WRAP_END` | Adapter lifecycle |
| Timeout | `TIMEOUT_START` → `TIMEOUT_RESET` → `TIMEOUT_TRIGGERED` | Timeout tracking |
| Retry | `RETRY_START` → `RETRY_ATTEMPT` → `RETRY_END` / `RETRY_GIVE_UP` | Retry loop |
| Fallback | `FALLBACK_START` → `FALLBACK_MODEL_SELECTED` → `FALLBACK_END` | Model switching |
| Guardrail | `GUARDRAIL_PHASE_START` → `GUARDRAIL_RULE_RESULT` → `GUARDRAIL_PHASE_END` | Validation |
| Network | `NETWORK_ERROR` → `NETWORK_RECOVERY` | Connection lifecycle |
| Checkpoint | `CHECKPOINT_SAVED` | Checkpoint tracking |
| Resume | `RESUME_START` → `RESUME_END` | Resume from checkpoint |
| Tool | `TOOL_REQUESTED` → `TOOL_START` → `TOOL_RESULT` / `TOOL_ERROR` | Tool execution |
| Completion | `SESSION_SUMMARY` → `SESSION_END` | Final status |

### Stream Initialization Events

```python
{"type": "SESSION_START", "ts": ..., "session_id": ...}       # Anchor for entire session
{"type": "STREAM_INIT", "ts": ..., "model": ..., "provider": ...}  # Before contacting provider
{"type": "STREAM_READY", "ts": ...}                            # Connection established
```

### Adapter Events

```python
{"type": "ADAPTER_WRAP_START", "ts": ..., "stream_type": ..., "adapter_id": ...}
{"type": "ADAPTER_DETECTED", "ts": ..., "adapter_id": ...}
{"type": "ADAPTER_WRAP_END", "ts": ..., "adapter_id": ...}
```

### Timeout Events

```python
{"type": "TIMEOUT_START", "ts": ..., "timeout_type": "initial|inter", "configured_ms": ...}
{"type": "TIMEOUT_RESET", "ts": ..., "timeout_type": ..., "configured_ms": ..., "token_index": ...}
{"type": "TIMEOUT_TRIGGERED", "ts": ..., "timeout_type": ..., "elapsed_ms": ..., "configured_ms": ...}
```

### Retry Events

```python
{"type": "RETRY_START", "ts": ..., "attempt": ..., "max_attempts": ...}
{"type": "RETRY_ATTEMPT", "ts": ..., "index": ..., "reason": ..., "counts_toward_limit": ..., "is_network": ..., "is_model_issue": ...}
{"type": "RETRY_END", "ts": ..., "attempt": ..., "success": ..., "duration_ms": ...}
{"type": "RETRY_GIVE_UP", "ts": ..., "attempts": ..., "last_error": ...}  # Exhausted
```

### Fallback Events

```python
{"type": "FALLBACK_START", "ts": ..., "from_model": ..., "to_model": ..., "reason": ...}
{"type": "FALLBACK_MODEL_SELECTED", "ts": ..., "index": ..., "model": ...}
{"type": "FALLBACK_END", "ts": ..., "index": ..., "duration_ms": ...}
```

### Guardrail Events

```python
# Phase boundary events
{"type": "GUARDRAIL_PHASE_START", "ts": ..., "phase": "pre|post", "rule_count": ...}
{"type": "GUARDRAIL_PHASE_END", "ts": ..., "phase": ..., "passed": ..., "violations": ..., "duration_ms": ...}

# Per-rule lifecycle
{"type": "GUARDRAIL_RULE_START", "ts": ..., "index": ..., "rule_id": ..., "callback_id": ...}
{"type": "GUARDRAIL_RULE_RESULT", "ts": ..., "index": ..., "rule_id": ..., "passed": ..., "violation": ...}
{"type": "GUARDRAIL_RULE_END", "ts": ..., "index": ..., "rule_id": ..., "passed": ..., "callback_id": ..., "duration_ms": ...}

# Callback lifecycle (for async/external guardrails)
{"type": "GUARDRAIL_CALLBACK_START", "ts": ..., "callback_id": ..., "index": ..., "rule_id": ...}
{"type": "GUARDRAIL_CALLBACK_END", "ts": ..., "callback_id": ..., "index": ..., "rule_id": ..., "duration_ms": ..., "success": ..., "error": ...}
```

### Network Events

```python
{"type": "NETWORK_ERROR", "ts": ..., "error": ..., "code": ..., "will_retry": ...}
{"type": "NETWORK_RECOVERY", "ts": ..., "attempt_count": ..., "duration_ms": ...}
{"type": "CONNECTION_DROPPED", "ts": ..., "reason": ...}
{"type": "CONNECTION_RESTORED", "ts": ..., "duration_ms": ...}
```

### Checkpoint and Resume Events

```python
{"type": "CHECKPOINT_SAVED", "ts": ..., "checkpoint": ..., "token_count": ...}
{"type": "RESUME_START", "ts": ..., "checkpoint": ..., "state_hash": ..., "token_count": ...}
{"type": "RESUME_END", "ts": ..., "checkpoint": ..., "duration_ms": ..., "success": ...}
```

### Tool Events

```python
{"type": "TOOL_REQUESTED", "ts": ..., "tool_name": ..., "arguments": ..., "tool_call_id": ..., "context": ...}
{"type": "TOOL_START", "ts": ..., "tool_call_id": ..., "tool_name": ...}
{"type": "TOOL_RESULT", "ts": ..., "tool_call_id": ..., "result": ..., "duration_ms": ..., "context": ...}
{"type": "TOOL_ERROR", "ts": ..., "tool_call_id": ..., "error": ..., "duration_ms": ..., "context": ...}
{"type": "TOOL_COMPLETED", "ts": ..., "tool_call_id": ..., "status": "success|error"}
```

### Drift Events

```python
{"type": "DRIFT_CHECK_START", "ts": ..., "checkpoint": ..., "token_count": ..., "strategy": ...}
{"type": "DRIFT_CHECK_RESULT", "ts": ..., "detected": ..., "score": ..., "metrics": ..., "threshold": ...}
{"type": "DRIFT_CHECK_END", "ts": ..., "duration_ms": ...}
{"type": "DRIFT_CHECK_SKIPPED", "ts": ..., "reason": ...}  # When drift disabled
```

### Completion Events

```python
{"type": "FINALIZATION_START", "ts": ...}  # Tokens done, closing session
{"type": "FINALIZATION_END", "ts": ..., "duration_ms": ...}  # All workers closed

# Final session summary for replay
{"type": "SESSION_SUMMARY", "ts": ..., "token_count": ..., "start_ts": ..., "end_ts": ...,
 "drift_detected": ..., "guardrail_violations": ..., "fallback_depth": ..., 
 "retry_count": ..., "checkpoints_created": ...}

{"type": "SESSION_END", "ts": ...}  # Hard end-of-stream marker
```

### Abort Events

```python
{"type": "ABORT_REQUESTED", "ts": ..., "source": "user|timeout|error"}
{"type": "ABORT_COMPLETED", "ts": ..., "resources_freed": ...}
```

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

## Testing

L0 ships with **comprehensive test coverage** across all core reliability systems - including streaming, guardrails, structured output, retry logic, fallbacks, consensus, and observability.

### Test Coverage

| Category          | Tests  | Description                      |
| ----------------- | ------ | -------------------------------- |
| Unit Tests        | 1,800+ | Fast, mocked, no API calls       |
| Integration Tests | 100+   | Real API calls, OpenAI + LiteLLM |

```bash
# Run unit tests (fast, no API keys needed)
uv run pytest tests/ -v --ignore=tests/integration

# Run integration tests (requires API keys)
OPENAI_API_KEY=sk-... uv run pytest tests/integration -v
```

### SDK Adapter Matrix

L0 supports all major provider SDKs with full end-to-end testing:

| Adapter      | Integration | Version Range   |
| ------------ | ----------- | --------------- |
| **OpenAI**   | ✓           | `>=1.30`        |
| **LiteLLM**  | ✓           | `>=1.40`        |

### Feature Test Matrix

Every major reliability feature in L0 has dedicated test suites:

| Feature               | Unit | Integration | Notes                                    |
| --------------------- | ---- | ----------- | ---------------------------------------- |
| **Streaming**         | ✓    | ✓           | Token events, completion                 |
| **Guardrails**        | ✓    | ✓           | JSON/Markdown, patterns, drift           |
| **Structured Output** | ✓    | ✓           | Pydantic schemas, auto-correction        |
| **Retry Logic**       | ✓    | ✓           | Backoff, error classification            |
| **Network Errors**    | ✓    | –           | 12+ simulated error types                |
| **Fallback Models**   | ✓    | ✓           | Sequential fallthrough                   |
| **Parallel / Race**   | ✓    | ✓           | Concurrency, cancellation                |
| **Consensus**         | ✓    | ✓           | Unanimous, majority, best-match          |
| **Continuation**      | ✓    | ✓           | Last-known-good token resumption         |
| **Observability**     | ✓    | ✓           | Event bus, callbacks, context            |
| **Drift Detection**   | ✓    | –           | Pattern detection, entropy, format drift |
| **Custom Adapters**   | ✓    | ✓           | OpenAI, LiteLLM adapters                 |

---

## Installation

```bash
# Basic installation
pip install ai2070-l0

# With OpenAI support
pip install ai2070-l0[openai]

# With LiteLLM (100+ providers)
pip install ai2070-l0[litellm]

# With OpenTelemetry
pip install ai2070-l0[otel]

# With Sentry
pip install ai2070-l0[sentry]

# With full observability (both)
pip install ai2070-l0[observability]

# Development
pip install ai2070-l0[dev]
```

Or with uv:

```bash
uv add ai2070-l0
uv add ai2070-l0 --extra openai
uv add ai2070-l0 --extra litellm
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
| `otel` | `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-httpx` |
| `sentry` | `sentry-sdk` |
| `observability` | All of the above (convenience) |
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
