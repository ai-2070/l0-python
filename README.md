# L0

Reliability layer for AI/LLM streaming with retry logic, guardrails, fallbacks, and observability.

## Installation

```bash
# Using uv
uv add l0

# With OpenAI support
uv add l0[openai]

# With LiteLLM (100+ providers)
uv add l0[litellm]
```

## Quick Start

```python
import asyncio
from openai import AsyncOpenAI
import l0

client = AsyncOpenAI()

async def main():
    result = await l0.l0(l0.L0Options(
        stream=lambda: client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        ),
        retry=l0.RetryConfig(attempts=3),
        guardrails=[l0.json_rule(), l0.pattern_rule()],
    ))

    async for event in result.stream:
        if event.type == l0.EventType.TOKEN:
            print(event.value, end="", flush=True)

asyncio.run(main())
```

## Features

### Retry with Backoff

Automatic retry with configurable backoff strategies:

```python
result = await l0.l0(l0.L0Options(
    stream=my_stream,
    retry=l0.RetryConfig(
        attempts=3,              # Model errors retry limit
        max_retries=6,           # Absolute maximum
        base_delay_ms=1000,      # Starting delay
        max_delay_ms=10000,      # Cap delay
        strategy=l0.BackoffStrategy.FIXED_JITTER,
    ),
))
```

**Error Categories:**
- `NETWORK` / `TRANSIENT` - Retry forever (doesn't count toward limit)
- `MODEL` / `CONTENT` - Counts toward retry limit
- `FATAL` / `INTERNAL` - No retry

### Fallbacks

Automatic fallback to alternative streams:

```python
result = await l0.l0(l0.L0Options(
    stream=lambda: openai_stream(),
    fallbacks=[
        lambda: anthropic_stream(),
        lambda: local_model_stream(),
    ],
))
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
    print(f"[{event.type}] {event.meta}")

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

## License

Apache 2.0
