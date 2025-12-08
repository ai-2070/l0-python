# L0 Quick Start Guide

Get started with L0 in 5 minutes.

## Installation

```bash
pip install ai2070-l0
```

**With provider support:**

```bash
# With OpenAI
pip install ai2070-l0[openai]

# With LiteLLM (100+ providers)
pip install ai2070-l0[litellm]

# With all extras
pip install ai2070-l0[all]
```

## Basic Usage

### Wrap a Client (Recommended)

The simplest way to use L0 - wrap your OpenAI client once and use it normally:

```python
import l0
from openai import AsyncOpenAI

# Wrap the client once
client = l0.wrap(AsyncOpenAI())

# Use normally - L0 reliability is automatic
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    stream=True,
)

# Iterate with L0 events
async for event in response:
    if event.is_token:
        print(event.text, end="")

# Or read all at once
text = await response.read()

# Access state after completion
print(f"\n\nTokens: {response.state.token_count}")
```

You now have:

- Automatic retry on network failures (doesn't count toward retry limit)
- Guardrails detecting malformed output
- Zero-token detection
- Unified event format

### Using l0.run() for Advanced Control

For more control, use `l0.run()` with a stream factory:

```python
import l0
from openai import AsyncOpenAI

client = AsyncOpenAI()

result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a haiku about coding"}],
        stream=True,
    ),
    guardrails=l0.Guardrails.recommended(),
    retry=l0.Retry.recommended(),
)

async for event in result:
    if event.is_token:
        print(event.text, end="")

print(f"\n\nTokens: {result.state.token_count}")
```

---

## Common Patterns

### Structured Output (Guaranteed JSON)

```python
from l0 import structured
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    email: str

result = await structured(
    schema=UserProfile,
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Generate a user profile as JSON"}],
        stream=True,
    ),
)

# Type-safe access
print(result.data.name)   # str
print(result.data.age)    # int
print(result.data.email)  # str
```

Also supports JSON Schema - see [STRUCTURED_OUTPUT.md](./STRUCTURED_OUTPUT.md).

### Timeout Protection

```python
from l0 import l0, Timeout, Guardrails

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    
    # Optional timeout configuration
    timeout=Timeout(
        initial_token=5000,  # 5s to first token (default: 5000ms)
        inter_token=10000,   # 10s between tokens (default: 10000ms)
    ),
    
    # Optional guardrails
    guardrails=Guardrails.recommended(),
)
```

**Note:** Free and low-priority models may take **3-7 seconds** before emitting the first token and **10+ seconds** between tokens.

### Fallback Models

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    ),
    fallbacks=[
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        ),
        lambda: anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            stream=True,
        ),
    ],
)

if result.state.fallback_index > 0:
    print("Used fallback model")
```

### Custom Guardrails

```python
from l0 import l0, Guardrails

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    guardrails=[
        Guardrails.zero_output_rule(),
        Guardrails.pattern_rule(
            [r"forbidden"],
            "Contains forbidden word",
            severity="error",
        ),
    ],
)
```

### Document Processing

```python
from l0 import Window

window = Window.create(
    long_document,
    size=2000,
    overlap=200,
    strategy="paragraph",
)

results = await window.process_all(
    lambda chunk: {
        "stream": lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Summarize: {chunk.content}"}],
            stream=True,
        ),
    },
)
```

### Error Handling

```python
from l0 import l0, Error, NetworkError, is_error

try:
    result = await l0.run(
        stream=stream,
        guardrails=guardrails,
    )
    async for event in result:
        # Process events
        pass
except Exception as e:
    if is_error(e):
        print(f"Error code: {e.code}")
        print(f"Checkpoint: {e.get_checkpoint()}")
    elif NetworkError.check(e):
        print("Network issue - will auto-retry")
```

---

## Presets

### Guardrails

```python
from l0 import Guardrails

# Via class methods (recommended)
guardrails = Guardrails.minimal()       # JSON + zero output
guardrails = Guardrails.recommended()   # + Markdown, patterns
guardrails = Guardrails.strict()        # + LaTeX
guardrails = Guardrails.json_only()     # JSON + zero output
guardrails = Guardrails.markdown_only() # Markdown + zero output
guardrails = Guardrails.latex_only()    # LaTeX + zero output

# Via module constants
from l0 import (
    MINIMAL_GUARDRAILS,
    RECOMMENDED_GUARDRAILS,
    STRICT_GUARDRAILS,
    JSON_ONLY_GUARDRAILS,
    MARKDOWN_ONLY_GUARDRAILS,
    LATEX_ONLY_GUARDRAILS,
)
```

**Preset contents:**

| Preset       | Rules                                                          |
| ------------ | -------------------------------------------------------------- |
| minimal      | json_rule, zero_output_rule                                    |
| recommended  | json_rule, markdown_rule, zero_output_rule, pattern_rule       |
| strict       | json_rule, markdown_rule, latex_rule, pattern_rule, zero_output_rule |
| json_only    | json_rule, zero_output_rule                                    |
| markdown_only| markdown_rule, zero_output_rule                                |
| latex_only   | latex_rule, zero_output_rule                                   |

### Retry

```python
from l0 import Retry

# Via class methods (recommended)
retry = Retry.minimal()       # 2 attempts, linear backoff
retry = Retry.recommended()   # 3 attempts, fixed-jitter backoff
retry = Retry.strict()        # 3 attempts, full-jitter backoff
retry = Retry.exponential()   # 4 attempts, exponential backoff
retry = Retry.mobile()        # Optimized for mobile
retry = Retry.edge()          # Optimized for edge runtimes

# Via module constants
from l0 import (
    MINIMAL_RETRY,
    RECOMMENDED_RETRY,
    STRICT_RETRY,
    EXPONENTIAL_RETRY,
)
```

**Preset details:**

| Preset      | attempts | max_retries | strategy     |
| ----------- | -------- | ----------- | ------------ |
| minimal     | 2        | 4           | linear       |
| recommended | 3        | 6           | fixed-jitter |
| strict      | 3        | 6           | full-jitter  |
| exponential | 4        | 8           | exponential  |

---

## Result State

After consuming the stream:

```python
print({
    "content": result.state.content,           # Full output
    "token_count": result.state.token_count,   # Token count
    "completed": result.state.completed,       # Stream finished
    "model_retry_count": result.state.model_retry_count,     # Model retries (counts toward limit)
    "network_retry_count": result.state.network_retry_count, # Network retries (doesn't count)
    "fallback_index": result.state.fallback_index,           # Which stream was used (0 = primary)
    "violations": result.state.violations,     # Guardrail violations
})
```

---

## Event Types

L0 provides a unified event format:

```python
async for event in result:
    if event.is_token:
        print(event.text, end="")       # Text token
    elif event.is_message:
        print(event.data)               # Structured message
    elif event.is_tool_call:
        print(event.data)               # Tool call
    elif event.is_data:
        print(event.payload)            # Multimodal data
    elif event.is_progress:
        print(event.progress.percent)   # Progress update
    elif event.is_complete:
        print("Done!", event.usage)     # Stream complete
    elif event.is_error:
        print("Error:", event.error)    # Error occurred
```

---

## Monitoring

Enable built-in observability:

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_event=lambda event: print(f"Event: {event.type}"),
)

# Lifecycle callbacks
result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_token=lambda text: print(text, end=""),
    on_complete=lambda state: print(f"\nDone! {state.token_count} tokens"),
    on_error=lambda e, will_retry, will_fallback: print(f"Error: {e}"),
    on_retry=lambda attempt, reason: print(f"Retry {attempt}: {reason}"),
    on_violation=lambda v: print(f"Violation: {v.message}"),
)
```

For OpenTelemetry and Sentry integration, see [MONITORING.md](./MONITORING.md).

---

## Next Steps

| Guide                                          | Description                  |
| ---------------------------------------------- | ---------------------------- |
| [API.md](./API.md)                             | Complete API reference       |
| [STRUCTURED_OUTPUT.md](./STRUCTURED_OUTPUT.md) | Guaranteed JSON with schemas |
| [DOCUMENT_WINDOWS.md](./DOCUMENT_WINDOWS.md)   | Processing long documents    |
| [NETWORK_ERRORS.md](./NETWORK_ERRORS.md)       | Network error handling       |
| [PERFORMANCE.md](./PERFORMANCE.md)             | Performance tuning           |
| [ERROR_HANDLING.md](./ERROR_HANDLING.md)       | Error codes and recovery     |
| [MONITORING.md](./MONITORING.md)               | Telemetry and observability  |
| [GUARDRAILS.md](./GUARDRAILS.md)               | Guardrails and validation    |
| [CUSTOM_ADAPTERS.md](./CUSTOM_ADAPTERS.md)     | Build your own adapters      |
