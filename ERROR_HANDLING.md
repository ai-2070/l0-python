# Error Handling Guide

This guide covers error handling patterns and error codes in L0.

## Table of Contents

- [Error Types](#error-types)
- [Error Class](#error-class)
- [Error Events](#error-events)
- [Error Codes](#error-codes)
- [Error Categories](#error-categories)
- [Network Error Detection](#network-error-detection)
- [Recovery Patterns](#recovery-patterns)
- [Best Practices](#best-practices)

---

## Error Types

L0 distinguishes between different error types for appropriate handling:

### L0 Errors

Errors raised by L0 itself, with rich context for debugging and recovery:

```python
from l0 import Error, is_error, ErrorCode

try:
    result = await l0.run(stream, guardrails=my_guardrails)
except Error as e:
    # L0-specific error with context
    print(e.code)           # ErrorCode.ZERO_OUTPUT
    print(e.category)       # ErrorCategory.CONTENT
    print(e.context)        # ErrorContext with details
    print(e.has_checkpoint) # True if checkpoint available
    print(e.timestamp)      # When error occurred

# Or use type guard
except Exception as e:
    if is_error(e):
        print(e.code)
        print(e.category)
```

### Network Errors

Transient failures from network issues:

```python
from l0 import NetworkError, is_network_error

try:
    result = await l0.run(stream)
except Exception as e:
    if is_network_error(e):
        analysis = NetworkError.analyze(e)
        print(analysis.type)                # NetworkErrorType.TIMEOUT
        print(analysis.retryable)           # True
        print(analysis.counts_toward_limit) # False (network errors don't count)
        print(analysis.suggestion)          # "Retry with longer timeout..."
        print(analysis.context)             # Additional context dict
```

### Standard Errors

Regular Python errors from invalid configuration or usage:

```python
try:
    await l0.run(stream=None)  # Invalid
except TypeError as e:
    # Standard Python error
    print(e)
```

---

## Error Class

The `Error` class provides structured error information:

```python
from l0 import Error, ErrorCode, ErrorContext, ErrorCategory

class Error(Exception):
    code: ErrorCode              # Error code enum
    context: ErrorContext        # Rich context
    timestamp: float             # Unix timestamp
    
    @property
    def category(self) -> ErrorCategory: ...
    
    @property
    def has_checkpoint(self) -> bool: ...
    
    def get_checkpoint(self) -> str | None: ...
    def to_detailed_string(self) -> str: ...
    def to_json(self) -> dict[str, Any]: ...
    
    # Static methods
    @staticmethod
    def is_error(e: Any) -> bool: ...
    
    @staticmethod
    def categorize(e: Exception) -> ErrorCategory: ...
    
    @staticmethod
    def is_retryable(e: Exception) -> bool: ...
    
    @staticmethod
    def get_category(code: ErrorCode) -> ErrorCategory: ...
```

### ErrorContext

```python
from l0 import ErrorContext

@dataclass
class ErrorContext:
    code: ErrorCode
    checkpoint: str | None = None        # Last good content for continuation
    token_count: int | None = None       # Tokens before failure
    content_length: int | None = None    # Content length before failure
    model_retry_count: int | None = None # Model retry attempts made
    network_retry_count: int | None = None  # Network retries made
    fallback_index: int | None = None    # Which fallback was tried (0 = primary)
    metadata: dict[str, Any] | None = None  # Internal metadata
    context: dict[str, Any] | None = None   # User-provided context
```

### Usage Example

```python
from l0 import Error, is_error

try:
    result = await l0.run(
        stream=my_stream,
        guardrails=strict_guardrails,
    )
except Error as e:
    # Log detailed error info
    print(e.to_detailed_string())
    # "Message | Tokens: 42 | Retries: 2 | Fallback: 1 | Checkpoint: 150 chars"

    # Access JSON representation
    print(e.to_json())
    # {
    #     "name": "Error",
    #     "code": "GUARDRAIL_VIOLATION",
    #     "category": "content",
    #     "message": "...",
    #     "timestamp": 1699000000.0,
    #     "hasCheckpoint": True,
    #     "checkpoint": "...",
    #     "tokenCount": 42,
    #     "modelRetryCount": 2,
    #     "networkRetryCount": 0,
    #     "fallbackIndex": 1
    # }

    # Check if we have a checkpoint for continuation
    if e.has_checkpoint:
        checkpoint = e.get_checkpoint()
        # Retry with checkpoint context
        ...

    # Access specific context
    print(f"Failed after {e.context.token_count} tokens")
    print(f"Model retry attempts: {e.context.model_retry_count}")
```

---

## Error Events

When errors occur, L0 emits `ERROR` events with detailed failure and recovery information.

### FailureType

What actually went wrong - the root cause of the failure:

```python
from l0 import FailureType

class FailureType(str, Enum):
    NETWORK = "network"        # Connection drops, DNS, SSL, fetch errors
    MODEL = "model"            # Model refused, content filter, guardrail violation
    TOOL = "tool"              # Tool execution failed
    TIMEOUT = "timeout"        # Initial token or inter-token timeout
    ABORT = "abort"            # User or signal abort
    ZERO_OUTPUT = "zero_output"  # Empty response from model
    UNKNOWN = "unknown"        # Unclassified error
```

### RecoveryStrategy

What L0 decided to do next:

```python
from l0 import RecoveryStrategy

class RecoveryStrategy(str, Enum):
    RETRY = "retry"        # Will retry the same stream
    FALLBACK = "fallback"  # Will try next fallback stream
    CONTINUE = "continue"  # Will continue despite error (non-fatal)
    HALT = "halt"          # Will stop, no recovery possible
```

### RecoveryPolicy

Why L0 chose that recovery strategy:

```python
from l0 import RecoveryPolicy

@dataclass
class RecoveryPolicy:
    retry_enabled: bool = True
    fallback_enabled: bool = False
    max_retries: int = 3
    max_fallbacks: int = 0
    attempt: int = 1            # Current retry attempt (1-based)
    fallback_index: int | None = None  # Current fallback index (0 = primary)
```

### Handling Error Events

```python
from l0.events import ObservabilityEventType

result = await l0.run(
    stream=my_stream,
    on_event=lambda event: handle_event(event),
)

def handle_event(event):
    if event.type == ObservabilityEventType.ERROR:
        print("Error:", event.meta.get("error"))
        print("Error code:", event.meta.get("code"))
        print("Recovery:", event.meta.get("recoveryStrategy"))
        
        # Track failure types
        metrics.increment(f"l0.failure.{event.meta.get('failureType')}")
        metrics.increment(f"l0.recovery.{event.meta.get('recoveryStrategy')}")
        
        # Alert on exhausted retries
        if event.meta.get("recoveryStrategy") == "halt":
            alerting.send(f"L0 halted after {event.meta.get('attempt')} attempts")
```

---

## Error Codes

L0 uses specific error codes for programmatic handling:

| Code                        | Description                                       | Category  |
| --------------------------- | ------------------------------------------------- | --------- |
| `STREAM_ABORTED`            | Stream was aborted (user cancellation or timeout) | PROVIDER  |
| `INITIAL_TOKEN_TIMEOUT`     | First token didn't arrive in time                 | TRANSIENT |
| `INTER_TOKEN_TIMEOUT`       | Gap between tokens exceeded limit                 | TRANSIENT |
| `ZERO_OUTPUT`               | Stream produced no meaningful output              | CONTENT   |
| `GUARDRAIL_VIOLATION`       | Content violated a guardrail rule                 | CONTENT   |
| `FATAL_GUARDRAIL_VIOLATION` | Content violated a fatal guardrail                | CONTENT   |
| `INVALID_STREAM`            | Stream factory returned invalid stream            | INTERNAL  |
| `ALL_STREAMS_EXHAUSTED`     | All streams (primary + fallbacks) failed          | PROVIDER  |
| `NETWORK_ERROR`             | Network-level failure                             | NETWORK   |
| `DRIFT_DETECTED`            | Output drifted from expected behavior             | CONTENT   |
| `ADAPTER_NOT_FOUND`         | Named adapter not found in registry               | INTERNAL  |
| `FEATURE_NOT_ENABLED`       | Feature requires explicit enablement              | INTERNAL  |

### ErrorCode Enum

```python
from l0 import ErrorCode

class ErrorCode(str, Enum):
    STREAM_ABORTED = "STREAM_ABORTED"
    INITIAL_TOKEN_TIMEOUT = "INITIAL_TOKEN_TIMEOUT"
    INTER_TOKEN_TIMEOUT = "INTER_TOKEN_TIMEOUT"
    ZERO_OUTPUT = "ZERO_OUTPUT"
    GUARDRAIL_VIOLATION = "GUARDRAIL_VIOLATION"
    FATAL_GUARDRAIL_VIOLATION = "FATAL_GUARDRAIL_VIOLATION"
    INVALID_STREAM = "INVALID_STREAM"
    ALL_STREAMS_EXHAUSTED = "ALL_STREAMS_EXHAUSTED"
    NETWORK_ERROR = "NETWORK_ERROR"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    ADAPTER_NOT_FOUND = "ADAPTER_NOT_FOUND"
    FEATURE_NOT_ENABLED = "FEATURE_NOT_ENABLED"
```

### Handling Specific Codes

```python
from l0 import Error, ErrorCode, is_error

try:
    result = await l0.run(stream, guardrails=my_guardrails)
except Error as e:
    match e.code:
        case ErrorCode.ZERO_OUTPUT:
            # Model produced nothing - maybe adjust prompt
            print("Empty response, adjusting prompt...")
        
        case ErrorCode.GUARDRAIL_VIOLATION:
            # Content failed validation - log for review
            print("Content violated:", e.context.metadata)
        
        case ErrorCode.INITIAL_TOKEN_TIMEOUT:
            # First token slow - network or model overloaded
            print("Model slow to respond")
        
        case ErrorCode.ALL_STREAMS_EXHAUSTED:
            # All models failed - critical failure
            print("All models unavailable")
        
        case ErrorCode.ADAPTER_NOT_FOUND:
            # Named adapter not registered
            print("Register the adapter first")
        
        case ErrorCode.FEATURE_NOT_ENABLED:
            # Feature needs to be enabled
            print("Call the enable function first")
        
        case _:
            raise
```

---

## Error Categories

L0's retry system categorizes errors for appropriate handling:

```python
from l0 import Error, ErrorCategory

# Categorize any exception
category = Error.categorize(error)

# Or from L0 Error
if is_error(error):
    print(error.category)

match category:
    case ErrorCategory.NETWORK:
        # Retry forever with backoff, doesn't count toward limit
        pass
    
    case ErrorCategory.TRANSIENT:
        # Rate limits, server errors, timeouts - retry forever
        pass
    
    case ErrorCategory.CONTENT:
        # Guardrails, drift, zero output - counts toward retry limit
        pass
    
    case ErrorCategory.MODEL:
        # Model-side errors - counts toward retry limit
        pass
    
    case ErrorCategory.PROVIDER:
        # Provider/API errors - may retry depending on status code
        pass
    
    case ErrorCategory.FATAL:
        # Don't retry (auth errors, SSL, invalid requests)
        pass
    
    case ErrorCategory.INTERNAL:
        # Internal bugs, invalid config - don't retry
        pass
```

### ErrorCategory Enum

```python
from l0 import ErrorCategory

class ErrorCategory(str, Enum):
    NETWORK = "network"      # Retry forever, doesn't count toward limit
    TRANSIENT = "transient"  # Retry forever (429, 503), doesn't count
    MODEL = "model"          # Model errors, counts toward retry limit
    CONTENT = "content"      # Guardrails, drift, counts toward limit
    PROVIDER = "provider"    # Provider/API errors
    FATAL = "fatal"          # Don't retry (auth, SSL, config)
    INTERNAL = "internal"    # Internal bugs, don't retry
```

### Category Breakdown

**NETWORK (retry forever, no count)**

- Connection dropped
- fetch() TypeError
- ECONNRESET / ECONNREFUSED
- SSE aborted
- DNS errors

**TRANSIENT (retry forever, no count)**

- 429 rate limit
- 503 server overload
- Timeouts (initial, inter-token)

**CONTENT (retry with limit)**

- Guardrail violations
- Drift detected
- Zero output

**MODEL (retry with limit)**

- Model-caused errors
- Bad response format
- Server error (model-side)

**PROVIDER (may retry)**

- Stream aborted
- All streams exhausted

**FATAL (no retry)**

- 401/403 auth errors
- Invalid request
- SSL errors
- Fatal guardrail violations

**INTERNAL (no retry)**

- Invalid stream
- Adapter not found
- Feature not enabled

---

## Network Error Detection

L0 provides detailed network error analysis via the `NetworkError` class:

```python
from l0 import NetworkError, NetworkErrorType, is_network_error

if is_network_error(error):
    analysis = NetworkError.analyze(error)
    
    print(analysis.type)                # NetworkErrorType.TIMEOUT
    print(analysis.retryable)           # True
    print(analysis.counts_toward_limit) # False (always for network)
    print(analysis.suggestion)          # Human-readable suggestion
    print(analysis.context)             # Additional context dict
```

### NetworkError Class API

```python
from l0 import NetworkError

# Check specific error types
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

# Check if any network error
NetworkError.check(error)

# Analyze error
analysis = NetworkError.analyze(error)

# Get human-readable description
desc = NetworkError.describe(error)

# Suggest retry delay
delay = NetworkError.suggest_delay(error, attempt=2)

# Check if stream was interrupted mid-flight
if NetworkError.is_stream_interrupted(error, token_count=50):
    print("Partial content in checkpoint")

# Create enhanced error with analysis attached
enhanced = NetworkError.create(original_error)
```

### Network Error Types

| Type                  | Description                        | Retryable |
| --------------------- | ---------------------------------- | --------- |
| `CONNECTION_DROPPED`  | Connection closed unexpectedly     | Yes       |
| `FETCH_ERROR`         | fetch() failed                     | Yes       |
| `ECONNRESET`          | Connection reset by peer           | Yes       |
| `ECONNREFUSED`        | Connection refused                 | Yes       |
| `SSE_ABORTED`         | Server-sent events aborted         | Yes       |
| `NO_BYTES`            | No data received                   | Yes       |
| `PARTIAL_CHUNKS`      | Incomplete data received           | Yes       |
| `RUNTIME_KILLED`      | Runtime terminated (Lambda/etc.)   | Yes       |
| `BACKGROUND_THROTTLE` | Mobile tab backgrounded            | Yes       |
| `DNS_ERROR`           | DNS resolution failed              | Yes       |
| `SSL_ERROR`           | SSL/TLS error                      | No        |
| `TIMEOUT`             | Request timed out                  | Yes       |
| `UNKNOWN`             | Unknown network error              | Yes       |

### Custom Delay by Error Type

```python
from l0 import Retry, ErrorTypeDelays

result = await l0.run(
    stream=my_stream,
    retry=Retry(
        attempts=3,
        error_type_delays=ErrorTypeDelays(
            connection_dropped=2.0,  # Wait longer for connection issues
            timeout=0.5,             # Retry faster on timeouts
            dns_error=5.0,           # DNS needs more time
        ),
    ),
)
```

### Standalone Network Detection Functions

For convenience, L0 also exports standalone functions:

```python
from l0 import (
    is_network_error,
    is_connection_dropped,
    is_fetch_error,
    is_econnreset,
    is_econnrefused,
    is_sse_aborted,
    is_no_bytes,
    is_partial_chunks,
    is_runtime_killed,
    is_background_throttle,
    is_dns_error,
    is_ssl_error,
    is_timeout_error,
    analyze_network_error,
    describe_network_error,
    suggest_retry_delay,
    is_stream_interrupted,
)
```

---

## Recovery Patterns

### Checkpoint Recovery

Use checkpoints to resume from last good state:

```python
checkpoint = ""

try:
    result = await l0.run(stream, guardrails=my_guardrails)
    async for event in result:
        # Process events
        pass
except Error as e:
    if e.has_checkpoint:
        checkpoint = e.get_checkpoint()

        # Retry with checkpoint context
        result = await l0.run(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"Continue from: {checkpoint}\n\nOriginal prompt: {prompt}"
                }],
                stream=True,
            )
        )
```

### Fallback Models

Automatically try cheaper models on failure:

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
        lambda: litellm.acompletion(
            model="anthropic/claude-3-haiku-20240307",
            messages=messages,
            stream=True,
        ),
    ],
)

# Check which model succeeded
if result.state.fallback_index > 0:
    print(f"Used fallback model {result.state.fallback_index}")
```

### Graceful Degradation

Handle errors at the application level:

```python
from l0 import Error, ErrorCode

async def generate_with_fallback(prompt: str) -> str:
    try:
        # Try L0 with full guardrails
        result = await l0.run(
            stream=my_stream,
            guardrails=strict_guardrails,
            retry=Retry.recommended(),
        )
        return await result.read()
    except Error as e:
        if e.code == ErrorCode.ALL_STREAMS_EXHAUSTED:
            # All models failed - return cached/default response
            return get_cached_response(prompt)
        raise
```

---

## Best Practices

### 1. Always Check Error Type

```python
from l0 import Error, is_error, is_network_error

try:
    result = await l0.run(stream, guardrails=my_guardrails)
except Exception as e:
    if is_error(e):
        # Handle L0-specific errors
        print(f"L0 error: {e.code}")
    elif is_network_error(e):
        # Handle network errors
        analysis = NetworkError.analyze(e)
        print(f"Network error: {analysis.type}")
    else:
        # Handle other errors
        raise
```

### 2. Log Error Context

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = await l0.run(stream)
except Error as e:
    logger.error({
        "code": e.code.value,
        "category": e.category.value,
        "token_count": e.context.token_count,
        "model_retry_count": e.context.model_retry_count,
        "network_retry_count": e.context.network_retry_count,
        "checkpoint": e.get_checkpoint()[:100] if e.has_checkpoint else None,
        "timestamp": e.timestamp,
    })
```

### 3. Set Appropriate Retry Limits

```python
from l0 import Retry

# Production: balance reliability vs latency
retry = Retry(
    attempts=3,        # Model errors (default: 3)
    max_retries=6,     # Absolute cap (all errors, default: 6)
)

result = await l0.run(stream, retry=retry)
```

### 4. Use Error Codes for Metrics

```python
from l0 import Error, is_error

try:
    result = await l0.run(stream)
except Error as e:
    metrics.increment(f"l0.error.{e.code.value}")
    metrics.increment(f"l0.error.category.{e.category.value}")
    metrics.increment(f"l0.error.has_checkpoint.{e.has_checkpoint}")
```

### 5. Handle Abort

```python
result = await l0.run(stream)

# Abort when needed
result.abort()

# Check if aborted
if result.state.aborted:
    print("Stream was aborted")
```

### 6. Test Error Scenarios

```python
import pytest
from l0 import Error, ErrorCode

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handles_zero_output(self):
        async def mock_stream():
            # Emit nothing
            return
            yield  # Make it a generator
        
        with pytest.raises(Error) as exc_info:
            await l0.run(stream=lambda: mock_stream())
        
        assert exc_info.value.code == ErrorCode.ZERO_OUTPUT
    
    @pytest.mark.asyncio
    async def test_handles_network_errors(self):
        async def mock_stream():
            raise ConnectionError("Network error")
            yield  # Make it a generator
        
        # Should retry automatically
        with pytest.raises(Error):
            await l0.run(
                stream=lambda: mock_stream(),
                retry=Retry(max_retries=1),
            )
```

---

## Error Reference

### Complete Error Flow

```
Stream starts
    |
    v
[First token received?]--No--> INITIAL_TOKEN_TIMEOUT (TRANSIENT, retry)
    |
    Yes
    v
[Token gap OK?]--No--> INTER_TOKEN_TIMEOUT (TRANSIENT, retry)
    |
    Yes
    v
[Guardrail check]--Fail--> GUARDRAIL_VIOLATION (CONTENT, retry if not fatal)
    |                 |
    Pass        [Fatal?]--Yes--> FATAL_GUARDRAIL_VIOLATION (halt)
    v
[Content accumulates...]
    |
    v
[Stream complete?]--Error--> Check error type
    |                              |
    Yes                    [Network?]--Yes--> NETWORK (retry, no count)
    |                              |
    v                      [Model?]--Yes--> MODEL (retry, counts)
[Final validation]                 |
    |                      [Fatal?]--Yes--> FATAL (halt)
    v                              |
[Zero output?]--Yes--> ZERO_OUTPUT [Internal?]--Yes--> INTERNAL (halt)
    |              (CONTENT, retry)
    No
    v
Success!
```

### Error Code to Category Mapping

```python
from l0 import ErrorCode, ErrorCategory, Error

# Use Error.get_category() or get_error_category()

NETWORK_ERROR         -> NETWORK
INITIAL_TOKEN_TIMEOUT -> TRANSIENT
INTER_TOKEN_TIMEOUT   -> TRANSIENT
GUARDRAIL_VIOLATION   -> CONTENT
FATAL_GUARDRAIL_VIOLATION -> CONTENT
DRIFT_DETECTED        -> CONTENT
ZERO_OUTPUT           -> CONTENT
INVALID_STREAM        -> INTERNAL
ADAPTER_NOT_FOUND     -> INTERNAL
FEATURE_NOT_ENABLED   -> INTERNAL
STREAM_ABORTED        -> PROVIDER
ALL_STREAMS_EXHAUSTED -> PROVIDER
```

---

## Types Summary

| Type                    | Description                           |
| ----------------------- | ------------------------------------- |
| `Error`                 | L0 error with rich context            |
| `ErrorCode`             | Error code enum                       |
| `ErrorContext`          | Rich context for errors               |
| `ErrorCategory`         | Category for retry decisions          |
| `FailureType`           | Root cause of failure                 |
| `RecoveryStrategy`      | What L0 decided to do                 |
| `RecoveryPolicy`        | Why L0 chose that strategy            |
| `NetworkError`          | Network error detection class         |
| `NetworkErrorType`      | Network error type enum               |
| `NetworkErrorAnalysis`  | Detailed network error analysis       |
