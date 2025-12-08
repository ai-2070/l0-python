# Network Error Handling Guide

L0 provides comprehensive network error detection and automatic recovery.

## Quick Start

```python
from l0 import l0, Retry

result = await l0(
    stream=lambda: client.chat.completions.create(model=model, messages=messages, stream=True),
    retry=Retry.recommended(),  # Handles all network errors automatically
)

print(f"Network retries: {result.state.network_retry_count}")
print(f"Model retries: {result.state.model_retry_count}")
```

---

## Supported Error Types

| Error Type          | Description                | Retries | Base Delay |
| ------------------- | -------------------------- | ------- | ---------- |
| Connection Dropped  | Connection lost mid-stream | Yes     | 1.0s       |
| fetch() TypeError   | Fetch API failure          | Yes     | 0.5s       |
| ECONNRESET          | Connection reset by peer   | Yes     | 1.0s       |
| ECONNREFUSED        | Server refused connection  | Yes     | 2.0s       |
| SSE Aborted         | Server-sent events aborted | Yes     | 0.5s       |
| No Bytes            | Server sent no data        | Yes     | 0.5s       |
| Partial Chunks      | Incomplete data received   | Yes     | 0.5s       |
| Runtime Killed      | Lambda/Edge timeout        | Yes     | 2.0s       |
| Background Throttle | Mobile tab backgrounded    | Yes     | 5.0s       |
| DNS Error           | Host not found             | Yes     | 3.0s       |
| SSL Error           | Certificate/TLS error      | **No**  | -          |
| Timeout             | Request timed out          | Yes     | 1.0s       |
| Unknown             | Unknown network error      | Yes     | 1.0s       |

**Key:** Network errors do NOT count toward the model retry limit.

---

## Error Categories

L0 classifies errors into categories that determine retry behavior:

```python
from l0.types import ErrorCategory

class ErrorCategory(str, Enum):
    NETWORK = "network"      # Retry forever, doesn't count toward limit
    TRANSIENT = "transient"  # Retry forever (429, 503, timeouts), doesn't count
    MODEL = "model"          # Model errors, counts toward retry limit
    CONTENT = "content"      # Guardrails/drift, counts toward limit
    PROVIDER = "provider"    # Provider/API errors
    FATAL = "fatal"          # Don't retry (auth, SSL, config)
    INTERNAL = "internal"    # Internal bugs, don't retry
```

---

## Error Detection

### NetworkError Class

The `NetworkError` class provides a scoped API for network error detection and analysis:

```python
from l0 import NetworkError

try:
    result = await l0(stream=stream, retry=Retry.recommended())
except Exception as error:
    if NetworkError.check(error):
        analysis = NetworkError.analyze(error)
        print(f"Type: {analysis.type}")              # NetworkErrorType
        print(f"Retryable: {analysis.retryable}")
        print(f"Counts toward limit: {analysis.counts_toward_limit}")
        print(f"Suggestion: {analysis.suggestion}")
        print(f"Context: {analysis.context}")
```

### NetworkErrorType Enum

```python
from l0.errors import NetworkErrorType

class NetworkErrorType(str, Enum):
    CONNECTION_DROPPED = "connection_dropped"
    FETCH_ERROR = "fetch_error"
    ECONNRESET = "econnreset"
    ECONNREFUSED = "econnrefused"
    SSE_ABORTED = "sse_aborted"
    NO_BYTES = "no_bytes"
    PARTIAL_CHUNKS = "partial_chunks"
    RUNTIME_KILLED = "runtime_killed"
    BACKGROUND_THROTTLE = "background_throttle"
    DNS_ERROR = "dns_error"
    SSL_ERROR = "ssl_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
```

### NetworkErrorAnalysis Dataclass

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class NetworkErrorAnalysis:
    type: NetworkErrorType
    retryable: bool
    counts_toward_limit: bool
    suggestion: str
    context: dict[str, Any]  # Additional context about the error
```

### Specific Error Checks

```python
from l0 import NetworkError

# Via scoped class (recommended)
if NetworkError.is_connection_dropped(error):
    # Connection was dropped mid-stream
    pass

if NetworkError.is_timeout(error):
    # Request timed out
    pass

if NetworkError.is_ssl(error):
    # SSL/TLS error - NOT retryable
    pass

# All available detection methods:
NetworkError.is_connection_dropped(error)
NetworkError.is_fetch_error(error)
NetworkError.is_econnreset(error)
NetworkError.is_econnrefused(error)
NetworkError.is_sse_aborted(error)
NetworkError.is_no_bytes(error)
NetworkError.is_partial_chunks(error)
NetworkError.is_runtime_killed(error)
NetworkError.is_background_throttle(error)
NetworkError.is_dns(error)
NetworkError.is_ssl(error)
NetworkError.is_timeout(error)
NetworkError.check(error)  # Any network error
```

### Module-Level Functions

For convenience, standalone functions are also available:

```python
from l0.errors import (
    is_network_error,
    analyze_network_error,
    is_connection_dropped,
    is_econnreset,
    is_econnrefused,
    is_sse_aborted,
    is_timeout_error,
    is_dns_error,
    is_ssl_error,
)

if is_network_error(error):
    analysis = analyze_network_error(error)
    print(analysis.type)
```

---

## Retry Configuration

### Retry Presets

```python
from l0 import Retry

# Class method presets (recommended)
retry = Retry.minimal()       # 2 attempts, 4 max, linear backoff
retry = Retry.recommended()   # 3 attempts, 6 max, fixed-jitter backoff (default)
retry = Retry.strict()        # 3 attempts, 6 max, full-jitter backoff
retry = Retry.exponential()   # 4 attempts, 8 max, exponential backoff
retry = Retry.mobile()        # Optimized for mobile environments
retry = Retry.edge()          # Optimized for edge runtimes

# Module-level constants (TypeScript API parity)
from l0 import MINIMAL_RETRY, RECOMMENDED_RETRY, STRICT_RETRY, EXPONENTIAL_RETRY
```

### Retry Defaults

```python
from l0 import RETRY_DEFAULTS, ERROR_TYPE_DELAY_DEFAULTS

# RETRY_DEFAULTS
RETRY_DEFAULTS.attempts        # 3 - Max model failure retries
RETRY_DEFAULTS.max_retries     # 6 - Absolute max across ALL error types
RETRY_DEFAULTS.base_delay      # 1.0 - Base delay in seconds
RETRY_DEFAULTS.max_delay       # 10.0 - Max delay cap in seconds
RETRY_DEFAULTS.network_max_delay  # 30.0 - Max delay for network errors
RETRY_DEFAULTS.backoff         # BackoffStrategy.FIXED_JITTER
RETRY_DEFAULTS.retry_on        # Tuple of RetryableErrorType values

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

### Custom Delay Configuration

Configure different delays for each error type:

```python
from l0 import l0, Retry, ErrorTypeDelays

result = await l0(
    stream=lambda: client.chat.completions.create(...),
    retry=Retry(
        attempts=3,
        max_retries=6,  # Absolute cap across ALL error types
        strategy=BackoffStrategy.FIXED_JITTER,
        error_type_delays=ErrorTypeDelays(
            connection_dropped=2.0,   # 2s for connection drops
            fetch_error=0.5,          # 0.5s for fetch errors
            econnreset=1.5,           # 1.5s for ECONNRESET
            econnrefused=3.0,         # 3s for ECONNREFUSED
            sse_aborted=1.0,          # 1s for SSE aborted
            no_bytes=0.5,             # 0.5s for no bytes
            partial_chunks=0.75,      # 0.75s for partial chunks
            runtime_killed=5.0,       # 5s for runtime kills
            background_throttle=10.0, # 10s for background throttle
            dns_error=4.0,            # 4s for DNS errors
            timeout=2.0,              # 2s for timeouts
            unknown=1.0,              # 1s for unknown errors
        ),
    ),
)
```

### Backoff Strategies

```python
from l0.types import BackoffStrategy

BackoffStrategy.EXPONENTIAL   # 2^n * base_delay
BackoffStrategy.LINEAR        # n * base_delay
BackoffStrategy.FIXED         # base_delay (constant)
BackoffStrategy.FULL_JITTER   # random(0, 2^n * base_delay)
BackoffStrategy.FIXED_JITTER  # random(base_delay/2, base_delay * 1.5) - AWS-style
```

### Retryable Error Types

Control which error types trigger retries:

```python
from l0 import Retry
from l0.types import RetryableErrorType

# Only retry specific error types
retry = Retry(
    attempts=3,
    retry_on=[
        RetryableErrorType.NETWORK_ERROR,
        RetryableErrorType.TIMEOUT,
        RetryableErrorType.RATE_LIMIT,
    ],
)

# All available types:
RetryableErrorType.ZERO_OUTPUT           # Empty response from model
RetryableErrorType.GUARDRAIL_VIOLATION   # Content failed validation
RetryableErrorType.DRIFT                 # Output drift detected
RetryableErrorType.INCOMPLETE            # Incomplete response
RetryableErrorType.NETWORK_ERROR         # Network connectivity issues
RetryableErrorType.TIMEOUT               # Request timed out
RetryableErrorType.RATE_LIMIT            # 429 rate limit errors
RetryableErrorType.SERVER_ERROR          # 5xx server errors
```

### Custom Retry Callbacks

```python
from l0 import Retry
from l0.types import ErrorCategory

# Custom should_retry callback (sync or async)
async def should_retry(error, state, attempt, category):
    # Return False to skip retry
    if category == ErrorCategory.FATAL:
        return False
    if attempt >= 3:
        return False
    return True

# Custom delay calculation
def custom_delay(context):
    # context is a RetryContext with:
    #   attempt, error, category, is_network,
    #   model_retry_count, network_retry_count, total_retries,
    #   base_delay, max_delay
    return min(context.attempt * 2.0, 30.0)

retry = Retry(
    attempts=3,
    should_retry=should_retry,
    calculate_delay=custom_delay,
)
```

---

## Environment-Specific Configuration

### Mobile

```python
result = await l0(
    stream=lambda: client.chat.completions.create(...),
    retry=Retry(
        attempts=3,
        max_retries=8,  # Allow more retries on mobile
        strategy=BackoffStrategy.FULL_JITTER,
        error_type_delays=ErrorTypeDelays(
            background_throttle=15.0,  # Wait longer for mobile
            timeout=3.0,               # More lenient timeouts
            connection_dropped=2.5,    # Mobile networks unstable
        ),
    ),
    # Timeouts in milliseconds
    timeout=Timeout(
        initial_token=5000,  # 5s to first token
        inter_token=10000,   # 10s between tokens
    ),
)

# Or use the preset
result = await l0(
    stream=lambda: client.chat.completions.create(...),
    retry=Retry.mobile(),
)
```

### Edge Runtime

```python
result = await l0(
    stream=lambda: client.chat.completions.create(...),
    retry=Retry(
        attempts=3,
        max_retries=4,  # Keep total retries low
        strategy=BackoffStrategy.FIXED_JITTER,
        max_delay=5.0,  # Keep delays short
        error_type_delays=ErrorTypeDelays(
            runtime_killed=2.0,  # Quick retry on timeout
            timeout=1.5,
        ),
    ),
    timeout=Timeout(
        initial_token=5000,
        inter_token=10000,
    ),
)

# Or use the preset
result = await l0(
    stream=lambda: client.chat.completions.create(...),
    retry=Retry.edge(),
)
```

---

## Retry Manager

For advanced use cases, use the `RetryManager` directly:

```python
from l0.retry import RetryManager
from l0.types import ErrorCategory, Retry

manager = RetryManager(Retry(
    attempts=3,
    max_retries=6,
    strategy=BackoffStrategy.FIXED_JITTER,
))

# Check if should retry (sync)
should = manager.should_retry(error)

# Check if should retry (async - supports async should_retry callback)
should = await manager.should_retry_async(error, state)

# Record a retry attempt
manager.record_attempt(error)

# Get delay for error
delay = manager.get_delay(error)  # Returns seconds

# Wait before retry
await manager.wait(error)

# Get current state
state = manager.get_state()
print(f"Model retries: {state['model_retry_count']}")
print(f"Network retries: {state['network_retry_count']}")
print(f"Total retries: {state['total_retries']}")

# Get error history
history = manager.get_error_history()

# Reset state
manager.reset()
```

### Error Categorization

```python
from l0 import Error, ErrorCategory

# Via Error class (recommended)
category = Error.categorize(error)
if category == ErrorCategory.NETWORK:
    # Network error - retry forever with backoff
    pass

# Quick check if error is retryable
if Error.is_retryable(error):
    # Can retry this error
    pass

# Get category for error code
from l0.errors import ErrorCode
category = Error.get_category(ErrorCode.NETWORK_ERROR)
```

---

## Utility Functions

```python
from l0 import NetworkError
from l0.errors import NetworkErrorType

# Get suggested delay with exponential backoff
delay = NetworkError.suggest_delay(error, attempt=0)

# With custom delays (in seconds)
custom_delays = {
    NetworkErrorType.CONNECTION_DROPPED: 2.0,
    NetworkErrorType.TIMEOUT: 1.5,
}
delay = NetworkError.suggest_delay(
    error,
    attempt=0,
    custom_delays=custom_delays,
    max_delay=30.0,
)

# Get human-readable description
description = NetworkError.describe(error)
# "Network error: econnreset (Connection was reset by peer)"

# Check if stream was interrupted mid-flight
if NetworkError.is_stream_interrupted(error, token_count=50):
    print("Partial content in checkpoint")

# Create enhanced error with analysis attached
enhanced = NetworkError.create(error)
print(enhanced.analysis.type)
print(enhanced.analysis.suggestion)
```

---

## L0 Error Class

L0 provides an enhanced error class with recovery context:

```python
from l0 import Error, is_error
from l0.errors import ErrorCode

try:
    result = await l0(stream=stream, retry=Retry.recommended())
except Exception as e:
    if is_error(e):
        print(f"Code: {e.code}")              # ErrorCode.ZERO_OUTPUT
        print(f"Category: {e.category}")       # ErrorCategory.CONTENT
        print(f"Has checkpoint: {e.has_checkpoint}")
        print(f"Checkpoint: {e.get_checkpoint()}")
        print(f"Timestamp: {e.timestamp}")
        print(f"Details: {e.to_detailed_string()}")
        print(f"JSON: {e.to_json()}")
```

### Error Codes

```python
from l0.errors import ErrorCode

ErrorCode.STREAM_ABORTED              # Stream was aborted
ErrorCode.INITIAL_TOKEN_TIMEOUT       # Timeout waiting for first token
ErrorCode.INTER_TOKEN_TIMEOUT         # Timeout between tokens
ErrorCode.ZERO_OUTPUT                 # Model produced no output
ErrorCode.GUARDRAIL_VIOLATION         # Content failed guardrails (retryable)
ErrorCode.FATAL_GUARDRAIL_VIOLATION   # Fatal guardrail violation (not retryable)
ErrorCode.INVALID_STREAM              # Invalid stream configuration
ErrorCode.ALL_STREAMS_EXHAUSTED       # All fallback streams exhausted
ErrorCode.NETWORK_ERROR               # Network connectivity error
ErrorCode.DRIFT_DETECTED              # Output drift detected
ErrorCode.ADAPTER_NOT_FOUND           # No adapter found for stream
ErrorCode.FEATURE_NOT_ENABLED         # Feature not enabled
```

### ErrorContext Dataclass

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ErrorContext:
    code: ErrorCode
    checkpoint: str | None = None         # Last good content for continuation
    token_count: int | None = None        # Tokens before failure
    content_length: int | None = None     # Content length before failure
    model_retry_count: int | None = None  # Retry attempts made
    network_retry_count: int | None = None  # Network retries made
    fallback_index: int | None = None     # Which fallback was tried
    metadata: dict[str, Any] | None = None  # Internal metadata
    context: dict[str, Any] | None = None   # User-provided context
```

---

## Monitoring

```python
from l0 import l0, Retry, NetworkError

result = await l0(
    stream=lambda: client.chat.completions.create(...),
    retry=Retry.recommended(),
    on_event=lambda event: handle_event(event),
)

# After completion
print(f"Network retries: {result.state.network_retry_count}")
print(f"Model retries: {result.state.model_retry_count}")

# Check network errors that occurred
for error in result.state.network_errors:
    analysis = NetworkError.analyze(error)
    print(f"Error type: {analysis.type}")
```

---

## State Tracking

L0 tracks retry information in the result state:

```python
from dataclasses import dataclass

@dataclass
class State:
    # ... other fields ...
    model_retry_count: int = 0      # Number of model retries
    network_retry_count: int = 0    # Number of network retries
    network_errors: list = []       # List of network errors encountered
    checkpoint: str = ""            # Last known good content
    resumed: bool = False           # Whether stream was resumed
```

---

## Best Practices

1. **Use `Retry.recommended()`** - Handles all network errors automatically with sensible defaults
2. **Set `max_retries`** - Prevent infinite loops with an absolute cap across all error types
3. **Set appropriate timeouts** - Higher for mobile/edge, lower for fast models
4. **Customize delays per error type** - Tune for your infrastructure
5. **Monitor network retries** - Alert if consistently high
6. **Handle checkpoints** - Partial content preserved in `result.state.checkpoint`

```python
from l0 import l0, Retry, Timeout, ErrorTypeDelays

# Production configuration
result = await l0(
    stream=lambda: client.chat.completions.create(...),
    retry=Retry(
        attempts=3,
        max_retries=6,  # Absolute cap
        strategy=BackoffStrategy.FULL_JITTER,
        max_delay=10.0,
        error_type_delays=ErrorTypeDelays(
            connection_dropped=1.0,
            runtime_killed=3.0,
            background_throttle=10.0,
        ),
    ),
    timeout=Timeout(
        initial_token=5000,  # 5s to first token
        inter_token=10000,   # 10s between tokens
    ),
)

# Check results
if result.state.network_retry_count > 0:
    print(f"Experienced {result.state.network_retry_count} network retries")
```

---

## API Reference

### NetworkError Class Methods

| Method | Description |
| ------ | ----------- |
| `NetworkError.check(error)` | Check if error is any network error |
| `NetworkError.analyze(error)` | Get detailed analysis of network error |
| `NetworkError.describe(error)` | Get human-readable description |
| `NetworkError.suggest_delay(error, attempt, ...)` | Get suggested retry delay |
| `NetworkError.is_stream_interrupted(error, token_count)` | Check if stream was interrupted |
| `NetworkError.create(error, analysis)` | Create enhanced error with analysis |
| `NetworkError.is_connection_dropped(error)` | Check for connection drop |
| `NetworkError.is_fetch_error(error)` | Check for fetch error |
| `NetworkError.is_econnreset(error)` | Check for ECONNRESET |
| `NetworkError.is_econnrefused(error)` | Check for ECONNREFUSED |
| `NetworkError.is_sse_aborted(error)` | Check for SSE abortion |
| `NetworkError.is_no_bytes(error)` | Check for no bytes error |
| `NetworkError.is_partial_chunks(error)` | Check for partial chunks |
| `NetworkError.is_runtime_killed(error)` | Check for runtime killed |
| `NetworkError.is_background_throttle(error)` | Check for background throttle |
| `NetworkError.is_dns(error)` | Check for DNS error |
| `NetworkError.is_ssl(error)` | Check for SSL error |
| `NetworkError.is_timeout(error)` | Check for timeout |

### Error Class Methods

| Method | Description |
| ------ | ----------- |
| `Error.categorize(error)` | Get error category for any exception |
| `Error.is_retryable(error)` | Check if error should be retried |
| `Error.is_error(error)` | Type guard for L0 Error |
| `Error.get_category(code)` | Get category for error code |

### Retry Presets

| Preset | Attempts | Max Retries | Strategy |
| ------ | -------- | ----------- | -------- |
| `Retry.minimal()` | 2 | 4 | Linear |
| `Retry.recommended()` | 3 | 6 | Fixed-Jitter |
| `Retry.strict()` | 3 | 6 | Full-Jitter |
| `Retry.exponential()` | 4 | 8 | Exponential |
| `Retry.mobile()` | 3 | 6 | Full-Jitter (tuned) |
| `Retry.edge()` | 3 | 6 | Fixed-Jitter (short) |
