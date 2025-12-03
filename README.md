# L0 - Reliability Layer for AI/LLM Streaming

### The missing reliability and observability layer for all AI streams.

> LLMs produce high-value reasoning over a low-integrity transport layer.
> Streams stall, drop tokens, reorder events, violate timing guarantees, and expose no deterministic contract.
>
> This breaks retries. It breaks supervision. It breaks reproducibility.
> It makes reliable AI systems impossible to build on top of raw provider streams.
>
> **L0 is the deterministic execution substrate that fixes the transport -
> with guardrails designed for the streaming layer itself: stream-neutral, pattern-based, loop-safe, and timing-aware.**

L0 adds deterministic execution, fallbacks, retries, network protection, guardrails, drift detection, and tool tracking to any LLM stream - turning raw model output into production-grade behavior.

Works with **OpenAI**, **LiteLLM** (100+ providers including Anthropic, Cohere, Bedrock, Vertex), and **custom adapters**.

```bash
pip install l0
# or with uv
uv add l0
```

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

| Feature                              | Description                                                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **Smart Retries**                    | Model-aware retries with fixed-jitter backoff. Automatic retries for zero-token output, network stalls, and provider overloads. |
| **Network Protection**               | Automatic recovery from dropped streams, slow responses, 429/503 load shedding, DNS errors, and partial chunks. |
| **Model Fallbacks**                  | Automatically fallback to secondary models with full retry logic.                                                |
| **Zero-Token/Stall Protection**      | Detects when model produces nothing or stalls mid-stream. Automatically retries or switches to fallbacks.        |
| **Drift Detection**                  | Detects repetition, stalls, and format drift before corruption.                                                  |
| **Structured Output**                | Guaranteed-valid JSON with Pydantic. Auto-corrects missing braces, commas, and markdown fences.                  |
| **Guardrails**                       | JSON, pattern validation with streaming checks.                                                                  |
| **Race: Fastest-Model Wins**         | Run multiple models in parallel and return the fastest valid stream.                                             |
| **Parallel: Fan-Out / Fan-In**       | Start multiple streams simultaneously and collect results.                                                       |
| **Consensus: Agreement Across Models** | Combine multiple model outputs using unanimous, majority, or best-match consensus.                              |
| **Central Event Bus**                | Full observability into every stream phase via `on_event` callback.                                              |
| **Pure asyncio**                     | No compatibility layers. Native Python async.                                                                    |
| **Type-Safe**                        | Full type hints, passes mypy strict mode.                                                                        |

## Quick Start

### With OpenAI

```python
import asyncio
from openai import AsyncOpenAI
import l0

client = AsyncOpenAI()

async def main():
    result = await l0.l0(l0.L0Options(
        stream=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        ),
        retry=l0.RetryConfig(attempts=3),
        guardrails=l0.recommended_guardrails(),
    ))

    async for event in result.stream:
        if event.type == l0.EventType.TOKEN:
            print(event.value, end="", flush=True)

asyncio.run(main())
```

### With LiteLLM (100+ Providers)

```python
import asyncio
import litellm
import l0

async def main():
    result = await l0.l0(l0.L0Options(
        stream=lambda: litellm.acompletion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        ),
        retry=l0.RetryConfig(attempts=3),
    ))

    async for event in result.stream:
        if event.type == l0.EventType.TOKEN:
            print(event.value, end="", flush=True)

asyncio.run(main())
```

### Full Configuration

```python
import l0

result = await l0.l0(l0.L0Options(
    # Required: Stream factory
    stream=lambda: client.chat.completions.create(..., stream=True),

    # Optional: Fallback streams
    fallbacks=[
        lambda: fallback_client.chat.completions.create(..., stream=True),
    ],

    # Optional: Guardrails (default: none)
    guardrails=l0.recommended_guardrails(),
    # Other presets:
    # l0.strict_guardrails()  # All rules including drift detection

    # Optional: Retry configuration
    retry=l0.RetryConfig(
        attempts=3,           # LLM errors only
        max_retries=6,        # Total (LLM + network)
        base_delay_ms=1000,
        max_delay_ms=10000,
        strategy=l0.BackoffStrategy.FIXED_JITTER,
    ),

    # Optional: Timeout configuration
    timeout=l0.TimeoutConfig(
        initial_token_ms=5000,   # 5s to first token
        inter_token_ms=10000,    # 10s between tokens
    ),

    # Optional: Event callback for observability
    on_event=lambda event: print(f"[{event.type}] {event.meta}"),

    # Optional: Metadata attached to all events
    meta={"user_id": "123", "session": "abc"},
))

# Read the stream
async for event in result.stream:
    if event.type == l0.EventType.TOKEN:
        print(event.value, end="")

# Access state
print(result.state.token_count)
print(result.state.content)
```

## Core Features

### Retry with Backoff

Automatic retry with configurable backoff strategies:

```python
result = await l0.l0(l0.L0Options(
    stream=my_stream,
    retry=l0.RetryConfig(
        attempts=3,                              # Model errors retry limit
        max_retries=6,                           # Absolute maximum
        base_delay_ms=1000,                      # Starting delay
        max_delay_ms=10000,                      # Cap delay
        strategy=l0.BackoffStrategy.FIXED_JITTER,  # Backoff strategy
    ),
))
```

**Backoff Strategies:**
- `EXPONENTIAL` - Classic exponential backoff (delay * 2^attempt)
- `LINEAR` - Linear increase (delay * attempt)
- `FIXED` - Constant delay
- `FULL_JITTER` - Random delay between 0 and exponential value
- `FIXED_JITTER` - Base delay + random jitter (default)

**Error Categories:**

| Error Type           | Retries | Counts Toward `attempts` |
| -------------------- | ------- | ------------------------ |
| Network disconnect   | Yes     | No                       |
| Zero output          | Yes     | No                       |
| Timeout              | Yes     | No                       |
| 429 rate limit       | Yes     | No                       |
| 503 server error     | Yes     | No                       |
| Model error          | Yes     | **Yes**                  |
| Auth error (401/403) | No      | -                        |

### Fallbacks

Automatic fallback to alternative streams:

```python
result = await l0.l0(l0.L0Options(
    stream=lambda: openai_stream(),
    fallbacks=[
        lambda: anthropic_via_litellm(),
        lambda: local_model_stream(),
    ],
))

# Check which model succeeded
print(result.state.fallback_index)  # 0 = primary, 1+ = fallback
```

### Guardrails

Built-in content validation:

```python
from l0 import (
    json_rule,           # Balanced JSON brackets
    strict_json_rule,    # Valid JSON on completion
    pattern_rule,        # Detect unwanted patterns
    zero_output_rule,    # Detect empty output
    stall_rule,          # Detect token stalls
    repetition_rule,     # Detect model looping
)

# Presets
guardrails = l0.recommended_guardrails()  # json, pattern, zero_output
guardrails = l0.strict_guardrails()       # All rules including drift detection
```

Custom guardrails:

```python
from l0 import GuardrailRule, GuardrailViolation

def max_length_rule(limit: int = 1000) -> GuardrailRule:
    def check(state):
        if len(state.content) > limit:
            return [GuardrailViolation(
                rule="max_length",
                message=f"Output exceeds {limit} chars",
                severity="error",
            )]
        return []
    return GuardrailRule(name="max_length", check=check)
```

### Observability

Central event bus for monitoring:

```python
def on_event(event: l0.ObservabilityEvent):
    print(f"[{event.type}] stream={event.stream_id} {event.meta}")

result = await l0.l0(l0.L0Options(
    stream=my_stream,
    on_event=on_event,
    meta={"user_id": "123"},  # Attached to all events
))
```

**Event Types:**
- `STREAM_INIT`, `STREAM_READY`
- `RETRY_ATTEMPT`, `RETRY_GIVE_UP`
- `FALLBACK_START`, `FALLBACK_END`
- `GUARDRAIL_PHASE_START`, `GUARDRAIL_RULE_RESULT`
- `NETWORK_ERROR`, `NETWORK_RECOVERY`
- `COMPLETE`, `ERROR`

### Structured Output

Validate against Pydantic schemas:

```python
from pydantic import BaseModel
import l0

class Response(BaseModel):
    answer: str
    confidence: float

result = await l0.structured(
    schema=Response,
    options=l0.L0Options(stream=my_stream),
)
print(result.answer, result.confidence)
```

### Parallel Execution

```python
# Run with concurrency limit
results = await l0.parallel(
    tasks=[task1, task2, task3],
    concurrency=2,
)

# Race - first result wins
result = await l0.race([fast_model, slow_model])

# Batch processing
results = await l0.batched(
    items=documents,
    handler=process_doc,
    batch_size=10,
)
```

### Consensus

Multi-model agreement:

```python
result = await l0.consensus(
    tasks=[model_a, model_b, model_c],
    strategy="majority",  # or "unanimous", "best"
)
```

## Stream Utilities

```python
# Consume stream to string
text = await l0.consume_stream(result.stream)

# Or use helper
text = await l0.get_text(result)

# Abort stream
result.abort()

# Access state
print(result.state.token_count)
print(result.state.content)
print(result.state.violations)
```

## Adapters

L0 auto-detects OpenAI and LiteLLM streams. For custom providers:

```python
from l0 import Adapter, register_adapter, L0Event, EventType

class MyAdapter:
    name = "my_provider"
    
    def detect(self, stream):
        return "my_provider" in type(stream).__module__
    
    async def wrap(self, stream):
        async for chunk in stream:
            yield L0Event(type=EventType.TOKEN, value=chunk.text)
        yield L0Event(type=EventType.COMPLETE)

register_adapter(MyAdapter())
```

## Debug Logging

```python
import l0
l0.enable_debug()  # Enables debug output
```

## Installation

```bash
# Basic installation
pip install l0

# With OpenAI support
pip install l0[openai]

# With LiteLLM (100+ providers)
pip install l0[litellm]

# Development
pip install l0[dev]
```

Or with uv:

```bash
uv add l0
uv add l0 --extra openai
uv add l0 --extra litellm
```

## API Reference

See [API.md](./API.md) for complete API reference.

## Philosophy

- **No magic** - Everything is explicit and predictable
- **Streaming-first** - Built for real-time token delivery
- **Signals, not rewrites** - Guardrails detect issues, don't modify output
- **Model-agnostic** - Works with any provider via adapters
- **Pure asyncio** - No compatibility layers, native Python async

## License

Apache-2.0
