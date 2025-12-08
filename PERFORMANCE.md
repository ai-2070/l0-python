# Performance Tuning Guide

This guide covers performance optimization for L0 in production environments.

## Table of Contents

- [Timeout Configuration](#timeout-configuration)
- [Retry Optimization](#retry-optimization)
- [Guardrail Performance](#guardrail-performance)
- [Memory Management](#memory-management)
- [Streaming Best Practices](#streaming-best-practices)
- [Document Window Tuning](#document-window-tuning)
- [Consensus Optimization](#consensus-optimization)

---

## Timeout Configuration

### Initial Token Timeout

The time to wait for the first token. Default is 5000ms. Set based on your model and network conditions:

```python
from l0 import l0, Timeout

result = await l0(
    stream=lambda: client.chat.completions.create(...),
    timeout=Timeout(
        initial_token=3000,  # 3 seconds for first token
    ),
)
```

**Recommendations:**

- **Fast models (GPT-4o-mini, Claude Haiku):** 1500-2000ms
- **Standard models (GPT-4o, Claude Sonnet):** 2000-3000ms
- **Large models (GPT-4, Claude Opus):** 3000-5000ms
- **Edge/mobile networks:** Add 1000-2000ms buffer

### Inter-Token Timeout

Maximum gap between tokens during streaming. Default is 10000ms (10 seconds):

```python
result = await l0(
    stream=lambda: client.chat.completions.create(...),
    timeout=Timeout(
        inter_token=1000,  # 1 second max gap
    ),
)
```

**Recommendations:**

- **Most use cases:** 1000ms
- **Long-form generation:** 2000ms (models may pause to "think")
- **Code generation:** 1500ms (complex reasoning)

### Combined Timeout Configuration

```python
from l0 import l0, Timeout

result = await l0(
    stream=lambda: client.chat.completions.create(...),
    timeout=Timeout(
        initial_token=5000,  # 5 seconds for first token (default)
        inter_token=10000,   # 10 seconds between tokens (default)
    ),
)
```

---

## Retry Optimization

### Backoff Strategies

Choose based on your use case:

```python
from l0 import Retry
from l0.types import BackoffStrategy

# Fixed jitter (default) - AWS-style fixed base + random jitter
# Good for: Most production workloads (prevents thundering herd)
retry = Retry(strategy=BackoffStrategy.FIXED_JITTER, base_delay=1.0, max_delay=10.0)

# Exponential - doubles delay each retry
# Good for: Gradual backpressure on overloaded services
retry = Retry(strategy=BackoffStrategy.EXPONENTIAL, base_delay=1.0, max_delay=10.0)

# Full jitter - random delay up to exponential max
# Good for: High-concurrency systems
retry = Retry(strategy=BackoffStrategy.FULL_JITTER, base_delay=1.0, max_delay=10.0)

# Linear - adds base_delay each retry
# Good for: Predictable delay requirements
retry = Retry(strategy=BackoffStrategy.LINEAR, base_delay=0.5, max_delay=5.0)

# Fixed - same delay every time
# Good for: Simple retry logic, testing
retry = Retry(strategy=BackoffStrategy.FIXED, base_delay=1.0)
```

### Retry Limits

L0 has two retry limits:

- **`attempts`**: Maximum retry attempts for model failures (default: 3). Network and transient errors do not count toward this limit.
- **`max_retries`**: Absolute maximum retries across ALL error types (default: 6). This is a hard cap including network errors.

```python
from l0 import Retry

# Conservative (fast failure)
retry = Retry(attempts=1)

# Balanced
retry = Retry(attempts=2)

# Default (recommended)
retry = Retry(attempts=3)

# With custom absolute cap
retry = Retry(attempts=3, max_retries=10)
```

### Selective Retry Reasons

Only retry on specific error types:

```python
from l0 import Retry
from l0.types import RetryableErrorType

# Defaults - all recoverable errors
retry = Retry(
    retry_on=[
        RetryableErrorType.ZERO_OUTPUT,
        RetryableErrorType.GUARDRAIL_VIOLATION,
        RetryableErrorType.DRIFT,
        RetryableErrorType.INCOMPLETE,
        RetryableErrorType.NETWORK_ERROR,
        RetryableErrorType.TIMEOUT,
        RetryableErrorType.RATE_LIMIT,
        RetryableErrorType.SERVER_ERROR,
    ],
)

# Minimal - only retry network issues
retry = Retry(
    retry_on=[
        RetryableErrorType.NETWORK_ERROR,
        RetryableErrorType.TIMEOUT,
    ]
)
```

Available retry reasons:

- `ZERO_OUTPUT` - No tokens received
- `GUARDRAIL_VIOLATION` - Guardrail check failed
- `DRIFT` - Content drift detected
- `INCOMPLETE` - Stream ended unexpectedly
- `NETWORK_ERROR` - Network connectivity issues
- `TIMEOUT` - Request timed out
- `RATE_LIMIT` - Rate limit (429) response
- `SERVER_ERROR` - Server error (5xx) response

### Error-Type-Specific Delays

Configure custom delays for specific network error types:

```python
from l0 import Retry, ErrorTypeDelays

retry = Retry(
    attempts=3,
    base_delay=1.0,
    error_type_delays=ErrorTypeDelays(
        connection_dropped=1.0,    # Default: 1.0s
        fetch_error=0.5,           # Default: 0.5s
        econnreset=1.0,            # Default: 1.0s
        econnrefused=2.0,          # Default: 2.0s
        sse_aborted=0.5,           # Default: 0.5s
        no_bytes=0.5,              # Default: 0.5s
        partial_chunks=0.5,        # Default: 0.5s
        runtime_killed=2.0,        # Default: 2.0s
        background_throttle=5.0,   # Default: 5.0s
        dns_error=3.0,             # Default: 3.0s
        timeout=1.0,               # Default: 1.0s
        unknown=1.0,               # Default: 1.0s
    ),
)
```

### Error Categories

L0 categorizes errors for retry decision-making:

| Category    | Description                            | Counts Toward Limit               |
| ----------- | -------------------------------------- | --------------------------------- |
| `NETWORK`   | Network/connection failures            | No (retries forever with backoff) |
| `TRANSIENT` | Rate limits (429), 503, timeouts       | No (retries forever with backoff) |
| `MODEL`     | Model-side errors (bad response)       | Yes                               |
| `CONTENT`   | Guardrails, drift                      | Yes                               |
| `PROVIDER`  | API errors (may retry based on status) | Depends                           |
| `FATAL`     | Auth failures, invalid config          | No retry                          |
| `INTERNAL`  | Internal bugs                          | No retry                          |

---

## Guardrail Performance

### Check Intervals

Control how often guardrails run during streaming:

```python
from l0 import l0, CheckIntervals, Guardrails

result = await l0(
    stream=lambda: client.chat.completions.create(...),
    guardrails=Guardrails.recommended(),
    check_intervals=CheckIntervals(
        guardrails=10,   # Check every 10 tokens (default: 5)
        drift=20,        # Check drift every 20 tokens (default: 10)
        checkpoint=15,   # Save checkpoint every 15 tokens (default: 10)
    ),
)
```

**Performance Warning:** Both guardrails and drift detection scan the accumulated content at each check interval. For very long outputs (multi-MB), this becomes O(n) per check. Consider:

- Increasing intervals for long-form content
- Using streaming-optimized guardrail rules that only check the delta
- Setting a maximum content length before disabling checks

**Trade-offs:**

- Lower intervals = faster detection, higher CPU
- Higher intervals = lower CPU, delayed detection

**Recommendations:**

- For simple delta-only rules: 1-5 tokens
- For rules that scan full content: 10-20 tokens
- For very long outputs: 50+ tokens

### Guardrail Selection

Only include guardrails you need:

```python
from l0 import Guardrails

# Minimal overhead
guardrails = [Guardrails.zero_output_rule()]

# Balanced
guardrails = [Guardrails.json_rule(), Guardrails.zero_output_rule()]

# Full validation (higher overhead)
guardrails = Guardrails.recommended()
```

### Pattern Matching

For custom patterns, pre-compile regexes:

```python
import re

# Pre-compile patterns at module level
FORBIDDEN_PATTERNS = [
    re.compile(r"sensitive_keyword", re.IGNORECASE),
    re.compile(r"another_pattern"),
]

# Reuse in guardrails
guardrails = [Guardrails.pattern_rule(FORBIDDEN_PATTERNS, "Forbidden content")]
```

---

## Memory Management

### Stream Consumption

Always consume streams to prevent memory buildup:

```python
# Good - fully consume stream
async for event in result:
    # Process events
    pass

# Or use read() to get full content
content = await result.read()

# Bad - abandoned stream may leak
result = await l0(stream=stream)
# Never consuming result
```

### Checkpoint Pruning

Checkpoints grow with content. For long generations:

```python
# Access checkpoint for recovery
checkpoint = result.state.checkpoint

# Clear after use if not needed
result.state.checkpoint = ""
```

### Async Context Manager

Use the context manager pattern for automatic cleanup:

```python
from l0 import l0

# Automatic cleanup with context manager
async with l0.wrap(stream) as result:
    async for event in result:
        if event.is_token:
            print(event.text, end="")
# Resources cleaned up automatically
```

---

## Streaming Best Practices

### Token Accumulation

L0 uses efficient token accumulation internally. For custom processing:

```python
# Good - efficient accumulation
tokens: list[str] = []
async for event in result:
    if event.is_token:
        tokens.append(event.text)
content = "".join(tokens)

# Or simply use read()
content = await result.read()

# Avoid - O(n^2) string concatenation
content = ""
async for event in result:
    if event.is_token:
        content += event.text  # Slow for large outputs
```

### Concurrent Streams

Use asyncio for concurrent stream processing:

```python
import asyncio
from l0 import l0

async def process_prompt(prompt: str) -> str:
    result = await l0(
        stream=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
    )
    return await result.read()

# Process multiple prompts concurrently
prompts = ["Question 1", "Question 2", "Question 3"]
results = await asyncio.gather(*[process_prompt(p) for p in prompts])
```

### Cancellation

Use asyncio task cancellation for cleanup:

```python
import asyncio
from l0 import l0

async def stream_with_timeout():
    try:
        result = await asyncio.wait_for(
            l0(stream=lambda: client.chat.completions.create(...)),
            timeout=30.0,
        )
        return await result.read()
    except asyncio.TimeoutError:
        print("Stream timed out")
        return None
```

---

## Document Window Tuning

### Chunk Size

Balance context vs. token limits:

```python
from l0 import Window

# Small chunks - more API calls, better context per chunk
window = Window.create(doc, size=1000, overlap=100)

# Large chunks - fewer calls, may exceed limits
window = Window.create(doc, size=4000, overlap=400)
```

**Recommendations by model:**

- **GPT-4o (128K context):** 4000-8000 tokens/chunk
- **GPT-4o-mini (128K context):** 4000-8000 tokens/chunk
- **Claude 3.5 (200K context):** 8000-16000 tokens/chunk
- **Gemini 1.5 (1M context):** 16000+ tokens/chunk

### Overlap Strategy

Maintain context between chunks:

```python
from l0 import Window

# 10% overlap (standard)
window = Window.create(doc, size=2000, overlap=200)

# 20% overlap (better continuity)
window = Window.create(doc, size=2000, overlap=400)

# No overlap (independent chunks)
window = Window.create(doc, size=2000, overlap=0)
```

### Chunking Strategy

Choose the chunking strategy that fits your content:

```python
from l0 import Window

# Token-based (default) - chunks by estimated token count
window = Window.create(doc, size=2000, strategy="token")

# Character-based - chunks by character count
window = Window.create(doc, size=8000, strategy="char")

# Paragraph-based - preserves paragraph boundaries
window = Window.create(doc, size=2000, strategy="paragraph")

# Sentence-based - preserves sentence boundaries
window = Window.create(doc, size=2000, strategy="sentence")
```

### Parallel Processing

Process chunks concurrently:

```python
from l0 import Window

window = Window.create(document, size=2000, overlap=200)

# Process all chunks with concurrency limit
results = await window.process_all(
    lambda chunk: {
        "stream": lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Summarize: {chunk.content}"}],
            stream=True,
        )
    },
    concurrency=3,  # Process 3 chunks at a time
)

# Check processing stats
print(f"Success rate: {results.stats.success_rate}%")
print(f"Total duration: {results.stats.total_duration}ms")
```

---

## Consensus Optimization

### Stream Count

Balance confidence vs. cost:

```python
from l0 import Consensus

# Minimum (low confidence) - 2 tasks required
result = await Consensus.run([task1, task2])

# Recommended (good confidence)
result = await Consensus.run([task1, task2, task3])

# High confidence (expensive)
result = await Consensus.run([task1, task2, task3, task4, task5])
```

### Strategy Selection

Choose based on requirements:

```python
from l0 import Consensus

# Majority - fastest, good for most cases
result = await Consensus.run(tasks, strategy="majority", threshold=0.6)

# Unanimous - strict, may fail more often
result = await Consensus.run(tasks, strategy="unanimous", threshold=1.0)

# Weighted - when some sources are more reliable
result = await Consensus.run(
    tasks,
    strategy="weighted",
    weights=[1.0, 0.8, 0.6],
)

# Best - choose highest quality output
result = await Consensus.run(tasks, strategy="best")
```

### Presets

Use presets for common configurations:

```python
from l0 import Consensus

# Strict - all must agree
result = await Consensus.strict(tasks)

# Standard - majority rules (default)
result = await Consensus.standard(tasks)

# Lenient - flexible matching
result = await Consensus.lenient(tasks)

# Best - choose best single output
result = await Consensus.best(tasks)

# Or use preset configurations
Consensus.STRICT    # ConsensusPreset for unanimous agreement
Consensus.STANDARD  # ConsensusPreset for majority rules
Consensus.LENIENT   # ConsensusPreset for flexible matching
Consensus.BEST      # ConsensusPreset for best output
```

### Early Termination

For structured output comparison, L0 uses early termination in deep equality checks. This means consensus returns faster when outputs obviously differ.

---

## Benchmarks

Typical performance characteristics (measured on Python 3.11+):

| Operation                  | Latency | Notes                    |
| -------------------------- | ------- | ------------------------ |
| Guardrail check (JSON)     | <0.1ms  | Per check interval       |
| Guardrail check (Markdown) | <0.2ms  | Per check interval       |
| Pattern detection          | <0.5ms  | Depends on pattern count |
| Deep equality check        | <1ms    | With early termination   |
| Structural similarity      | 1-5ms   | Depends on object depth  |
| Token accumulation         | O(n)    | Linear with token count  |

---

## RETRY_DEFAULTS Reference

L0 exports default retry configuration values:

```python
from l0 import RETRY_DEFAULTS, ERROR_TYPE_DELAY_DEFAULTS

# RETRY_DEFAULTS
RETRY_DEFAULTS.attempts          # 3 - Maximum model failure retries
RETRY_DEFAULTS.max_retries       # 6 - Absolute maximum across all error types
RETRY_DEFAULTS.base_delay        # 1.0 - Base delay in seconds
RETRY_DEFAULTS.max_delay         # 10.0 - Maximum delay cap in seconds
RETRY_DEFAULTS.network_max_delay # 30.0 - Max delay for network error suggestions
RETRY_DEFAULTS.backoff           # BackoffStrategy.FIXED_JITTER
RETRY_DEFAULTS.retry_on          # Tuple of RetryableErrorType values

# ERROR_TYPE_DELAY_DEFAULTS (all in seconds)
ERROR_TYPE_DELAY_DEFAULTS.connection_dropped   # 1.0
ERROR_TYPE_DELAY_DEFAULTS.fetch_error          # 0.5
ERROR_TYPE_DELAY_DEFAULTS.econnreset           # 1.0
ERROR_TYPE_DELAY_DEFAULTS.econnrefused         # 2.0
ERROR_TYPE_DELAY_DEFAULTS.sse_aborted          # 0.5
ERROR_TYPE_DELAY_DEFAULTS.no_bytes             # 0.5
ERROR_TYPE_DELAY_DEFAULTS.partial_chunks       # 0.5
ERROR_TYPE_DELAY_DEFAULTS.runtime_killed       # 2.0
ERROR_TYPE_DELAY_DEFAULTS.background_throttle  # 5.0
ERROR_TYPE_DELAY_DEFAULTS.dns_error            # 3.0
ERROR_TYPE_DELAY_DEFAULTS.ssl_error            # 0.0 (SSL errors are not retried)
ERROR_TYPE_DELAY_DEFAULTS.timeout              # 1.0
ERROR_TYPE_DELAY_DEFAULTS.unknown              # 1.0
```

---

## Production Checklist

- [ ] Set appropriate timeouts for your model (`Timeout.initial_token`, `Timeout.inter_token`)
- [ ] Configure retry limits to balance reliability vs. latency (`attempts`, `max_retries`)
- [ ] Select only needed guardrails
- [ ] Use async context manager for automatic cleanup
- [ ] Use appropriate chunk sizes for document windows
- [ ] Pre-compile regex patterns for custom guardrails
- [ ] Consume all streams to prevent memory leaks
- [ ] Consider error-type-specific delays for network errors
- [ ] Increase check intervals for long-form content generation
- [ ] Use `asyncio.gather()` for concurrent stream processing
