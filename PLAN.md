# L0 Python Port Plan

A clean Python implementation of L0 - a reliability layer for AI/LLM streaming.

## Overview

**L0** wraps LLM streams with retry logic, guardrails, fallbacks, and observability.

**Source:** `ts/` directory (reference only - not a line-by-line port)
**Target:** Python 3.10+ with pure `asyncio`

---

## Design Principles

1. **Pure asyncio** - No anyio/trio. Full determinism and performance.
2. **Own retry logic** - No tenacity. L0 controls all retry behavior.
3. **Centralized events** - One event bus for all observability.
4. **Clean slate** - Simplified structure, not a direct TS port.
5. **Minimal adapters** - Python SDK streams are simpler than TS.

---

## Package Structure

```
l0/
├── __init__.py           # Public API exports
├── py.typed              # PEP 561 marker
│
├── types.py              # All type definitions
├── events.py             # Central event bus + event types
├── errors.py             # Error categorization
│
├── core.py               # Main l0() function
├── retry.py              # RetryManager (own implementation)
├── state.py              # L0State management
├── stream.py             # Stream utilities
│
├── adapters.py           # All adapters in one file (simple)
├── guardrails.py         # Engine + built-in rules
│
├── structured.py         # Structured output with Pydantic
├── parallel.py           # parallel(), race(), batched()
├── consensus.py          # Multi-model consensus
│
└── _utils.py             # Internal utilities (backoff, json repair)
```

**Note:** Flat structure. No deep nesting. One file per concept.

---

## Phase 1: Foundation

### 1.1 Types (`l0/types.py`)

Core types using dataclasses and enums:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Callable, AsyncIterator, Any

class EventType(str, Enum):
    TOKEN = "token"
    MESSAGE = "message"
    DATA = "data"
    ERROR = "error"
    COMPLETE = "complete"

class ErrorCategory(str, Enum):
    NETWORK = "network"      # Infinite retry
    TRANSIENT = "transient"  # Infinite retry (429, 503)
    MODEL = "model"          # Counts toward limit
    CONTENT = "content"      # Counts toward limit
    FATAL = "fatal"          # No retry

class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FIXED_JITTER = "fixed_jitter"
    FULL_JITTER = "full_jitter"

@dataclass
class L0Event:
    type: EventType
    value: str | None = None
    error: Exception | None = None
    usage: dict[str, int] | None = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class L0State:
    content: str = ""
    checkpoint: str = ""
    token_count: int = 0
    model_retries: int = 0
    network_retries: int = 0
    fallback_index: int = 0
    violations: list["GuardrailViolation"] = field(default_factory=list)
    completed: bool = False
    aborted: bool = False
    first_token_at: float | None = None
    last_token_at: float | None = None

@dataclass
class L0Options:
    stream: Callable[[], AsyncIterator[Any]]
    fallbacks: list[Callable[[], AsyncIterator[Any]]] = field(default_factory=list)
    guardrails: list["GuardrailRule"] = field(default_factory=list)
    retry: "RetryConfig | None" = None
    timeout: "TimeoutConfig | None" = None
    adapter: "Adapter | str | None" = None
    on_event: Callable[["ObservabilityEvent"], None] | None = None

@dataclass
class L0Result:
    stream: AsyncIterator[L0Event]
    state: L0State
    abort: Callable[[], None]
```

### 1.2 Central Event Bus (`l0/events.py`)

Single event system for all observability:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable

class ObservabilityEventType(str, Enum):
    # Session
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    
    # Stream
    STREAM_START = "stream_start"
    STREAM_TOKEN = "stream_token"
    STREAM_COMPLETE = "stream_complete"
    
    # Retry
    RETRY_ATTEMPT = "retry_attempt"
    RETRY_EXHAUSTED = "retry_exhausted"
    
    # Fallback
    FALLBACK_START = "fallback_start"
    FALLBACK_SUCCESS = "fallback_success"
    
    # Guardrails
    GUARDRAIL_CHECK = "guardrail_check"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    
    # Errors
    ERROR_NETWORK = "error_network"
    ERROR_MODEL = "error_model"
    ERROR_FATAL = "error_fatal"

@dataclass
class ObservabilityEvent:
    type: ObservabilityEventType
    timestamp: float
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)

class EventBus:
    """Central event bus for all L0 observability."""
    
    def __init__(self, handler: Callable[[ObservabilityEvent], None] | None = None):
        self._handler = handler
        self._session_id = str(uuid.uuid4())
    
    def emit(self, event_type: ObservabilityEventType, **data: Any) -> None:
        if self._handler:
            event = ObservabilityEvent(
                type=event_type,
                timestamp=time.time(),
                session_id=self._session_id,
                data=data
            )
            self._handler(event)
```

### 1.3 Error Categorization (`l0/errors.py`)

```python
import re
from .types import ErrorCategory

# Network error patterns
NETWORK_PATTERNS = [
    r"connection.*reset",
    r"connection.*refused", 
    r"connection.*timeout",
    r"timed?\s*out",
    r"dns.*failed",
    r"name.*resolution",
    r"socket.*error",
    r"ssl.*error",
    r"eof.*occurred",
    r"broken.*pipe",
    r"network.*unreachable",
    r"host.*unreachable",
]

def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize error for retry decisions."""
    msg = str(error).lower()
    
    # Check network patterns
    for pattern in NETWORK_PATTERNS:
        if re.search(pattern, msg):
            return ErrorCategory.NETWORK
    
    # Check HTTP status if available
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status:
        if status == 429:
            return ErrorCategory.TRANSIENT
        if status in (401, 403):
            return ErrorCategory.FATAL
        if 500 <= status < 600:
            return ErrorCategory.TRANSIENT
    
    # Check for rate limit in message
    if "rate" in msg and "limit" in msg:
        return ErrorCategory.TRANSIENT
    
    return ErrorCategory.MODEL
```

---

## Phase 2: Runtime

### 2.1 Retry Manager (`l0/retry.py`)

Own implementation - no tenacity:

```python
import asyncio
import random
from dataclasses import dataclass
from .types import ErrorCategory, BackoffStrategy
from .errors import categorize_error

@dataclass
class RetryConfig:
    max_attempts: int = 3           # Model errors only
    base_delay: float = 1.0         # seconds
    max_delay: float = 30.0         # seconds
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER

class RetryManager:
    """Manages retry logic with error-aware backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.model_attempts = 0
        self.network_attempts = 0
    
    def should_retry(self, error: Exception) -> bool:
        category = categorize_error(error)
        
        if category == ErrorCategory.FATAL:
            return False
        if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
            return True  # Always retry, doesn't count
        
        # MODEL or CONTENT - counts toward limit
        return self.model_attempts < self.config.max_attempts
    
    def record_attempt(self, error: Exception) -> None:
        category = categorize_error(error)
        if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
            self.network_attempts += 1
        else:
            self.model_attempts += 1
    
    def get_delay(self, error: Exception) -> float:
        category = categorize_error(error)
        attempt = self.network_attempts if category == ErrorCategory.NETWORK else self.model_attempts
        
        base = self.config.base_delay
        cap = self.config.max_delay
        
        match self.config.strategy:
            case BackoffStrategy.EXPONENTIAL:
                delay = min(base * (2 ** attempt), cap)
            case BackoffStrategy.LINEAR:
                delay = min(base * (attempt + 1), cap)
            case BackoffStrategy.FIXED:
                delay = base
            case BackoffStrategy.FIXED_JITTER:
                temp = min(base * (2 ** attempt), cap)
                delay = temp / 2 + random.random() * (temp / 2)
            case BackoffStrategy.FULL_JITTER:
                delay = random.random() * min(base * (2 ** attempt), cap)
        
        return delay
    
    async def wait(self, error: Exception) -> None:
        delay = self.get_delay(error)
        await asyncio.sleep(delay)
```

### 2.2 State (`l0/state.py`)

Simple state container - no state machine needed:

```python
from dataclasses import dataclass, field
import time
from .types import L0State

def create_state() -> L0State:
    """Create fresh L0 state."""
    return L0State()

def update_checkpoint(state: L0State) -> None:
    """Save current content as checkpoint."""
    state.checkpoint = state.content

def append_token(state: L0State, token: str) -> None:
    """Append token to content and update timing."""
    now = time.time()
    if state.first_token_at is None:
        state.first_token_at = now
    state.last_token_at = now
    state.content += token
    state.token_count += 1
```

### 2.3 Stream Utilities (`l0/stream.py`)

```python
from typing import AsyncIterator
from .types import L0Event, EventType

async def consume_stream(stream: AsyncIterator[L0Event]) -> str:
    """Consume stream and return full text."""
    content = ""
    async for event in stream:
        if event.type == EventType.TOKEN and event.value:
            content += event.value
    return content

async def get_text(result: "L0Result") -> str:
    """Helper to get text from L0Result."""
    return await consume_stream(result.stream)
```

---

## Phase 3: Core Function

### 3.1 Main l0() Function (`l0/core.py`)

```python
import asyncio
from typing import AsyncIterator
from .types import L0Options, L0Result, L0State, L0Event, EventType
from .retry import RetryManager, RetryConfig
from .state import create_state, append_token, update_checkpoint
from .events import EventBus, ObservabilityEventType
from .adapters import detect_adapter, wrap_stream
from .guardrails import check_guardrails

async def l0(options: L0Options) -> L0Result:
    """Main L0 wrapper function."""
    
    state = create_state()
    retry_mgr = RetryManager(options.retry or RetryConfig())
    event_bus = EventBus(options.on_event)
    aborted = False
    
    def abort():
        nonlocal aborted
        aborted = True
    
    async def run_stream() -> AsyncIterator[L0Event]:
        nonlocal state
        
        streams = [options.stream] + options.fallbacks
        
        for fallback_idx, stream_fn in enumerate(streams):
            state.fallback_index = fallback_idx
            
            if fallback_idx > 0:
                event_bus.emit(ObservabilityEventType.FALLBACK_START, index=fallback_idx)
            
            while True:
                try:
                    event_bus.emit(ObservabilityEventType.STREAM_START)
                    raw_stream = await stream_fn()
                    adapter = detect_adapter(raw_stream, options.adapter)
                    
                    async for event in wrap_stream(raw_stream, adapter):
                        if aborted:
                            state.aborted = True
                            return
                        
                        if event.type == EventType.TOKEN and event.value:
                            append_token(state, event.value)
                            
                            # Check guardrails periodically
                            if state.token_count % 10 == 0:
                                violations = check_guardrails(state, options.guardrails)
                                if violations:
                                    state.violations.extend(violations)
                                    event_bus.emit(ObservabilityEventType.GUARDRAIL_VIOLATION, 
                                                 violations=violations)
                        
                        yield event
                    
                    # Success
                    state.completed = True
                    event_bus.emit(ObservabilityEventType.STREAM_COMPLETE)
                    return
                    
                except Exception as e:
                    event_bus.emit(ObservabilityEventType.ERROR_NETWORK, error=str(e))
                    
                    if retry_mgr.should_retry(e):
                        retry_mgr.record_attempt(e)
                        event_bus.emit(ObservabilityEventType.RETRY_ATTEMPT,
                                     attempt=retry_mgr.model_attempts)
                        await retry_mgr.wait(e)
                        update_checkpoint(state)
                        continue
                    else:
                        # Try next fallback
                        break
        
        # All fallbacks exhausted
        event_bus.emit(ObservabilityEventType.RETRY_EXHAUSTED)
        raise RuntimeError("All streams and fallbacks exhausted")
    
    return L0Result(
        stream=run_stream(),
        state=state,
        abort=abort
    )
```

---

## Phase 4: Adapters

### 4.1 Simple Adapters (`l0/adapters.py`)

Unified in one file - Python SDKs are simpler:

```python
from typing import AsyncIterator, Any, Protocol
from .types import L0Event, EventType

class Adapter(Protocol):
    name: str
    def detect(self, stream: Any) -> bool: ...
    async def wrap(self, stream: Any) -> AsyncIterator[L0Event]: ...

class OpenAIAdapter:
    name = "openai"
    
    def detect(self, stream: Any) -> bool:
        return hasattr(stream, "__aiter__") and "openai" in type(stream).__module__
    
    async def wrap(self, stream: Any) -> AsyncIterator[L0Event]:
        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield L0Event(type=EventType.TOKEN, value=delta.content)
        yield L0Event(type=EventType.COMPLETE)

class AnthropicAdapter:
    name = "anthropic"
    
    def detect(self, stream: Any) -> bool:
        return hasattr(stream, "__aiter__") and "anthropic" in type(stream).__module__
    
    async def wrap(self, stream: Any) -> AsyncIterator[L0Event]:
        async for event in stream:
            if hasattr(event, "type"):
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield L0Event(type=EventType.TOKEN, value=event.delta.text)
                elif event.type == "message_stop":
                    yield L0Event(type=EventType.COMPLETE)

# Registry
_adapters: list[Adapter] = [OpenAIAdapter(), AnthropicAdapter()]

def register_adapter(adapter: Adapter) -> None:
    _adapters.insert(0, adapter)

def detect_adapter(stream: Any, hint: Adapter | str | None = None) -> Adapter:
    if isinstance(hint, Adapter):
        return hint
    if isinstance(hint, str):
        for a in _adapters:
            if a.name == hint:
                return a
    for a in _adapters:
        if a.detect(stream):
            return a
    raise ValueError("No adapter found for stream")

async def wrap_stream(stream: Any, adapter: Adapter) -> AsyncIterator[L0Event]:
    async for event in adapter.wrap(stream):
        yield event
```

---

## Phase 5: Guardrails

### 5.1 Guardrails (`l0/guardrails.py`)

Engine and rules in one file:

```python
import re
import json
from dataclasses import dataclass
from typing import Callable, Literal
from .types import L0State

Severity = Literal["warning", "error", "fatal"]

@dataclass
class GuardrailViolation:
    rule: str
    message: str
    severity: Severity

@dataclass
class GuardrailRule:
    name: str
    check: Callable[[L0State], list[GuardrailViolation]]
    severity: Severity = "error"

def check_guardrails(state: L0State, rules: list[GuardrailRule]) -> list[GuardrailViolation]:
    """Run all guardrail rules against current state."""
    violations = []
    for rule in rules:
        violations.extend(rule.check(state))
    return violations

# Built-in rules

def json_rule() -> GuardrailRule:
    """Check for balanced JSON braces."""
    def check(state: L0State) -> list[GuardrailViolation]:
        content = state.content
        opens = content.count("{") + content.count("[")
        closes = content.count("}") + content.count("]")
        if opens < closes:
            return [GuardrailViolation("json", "Unbalanced JSON brackets", "error")]
        return []
    return GuardrailRule(name="json", check=check)

def strict_json_rule() -> GuardrailRule:
    """Validate complete JSON on completion."""
    def check(state: L0State) -> list[GuardrailViolation]:
        if not state.completed:
            return []
        try:
            json.loads(state.content)
            return []
        except json.JSONDecodeError as e:
            return [GuardrailViolation("strict_json", f"Invalid JSON: {e}", "error")]
    return GuardrailRule(name="strict_json", check=check)

def pattern_rule(patterns: list[str] | None = None) -> GuardrailRule:
    """Detect unwanted patterns (e.g., AI slop)."""
    default_patterns = [
        r"\bas an ai\b",
        r"\bi cannot\b",
        r"\bi don'?t have\b",
        r"\bunfortunately\b",
        r"\bi apologize\b",
    ]
    patterns = patterns or default_patterns
    
    def check(state: L0State) -> list[GuardrailViolation]:
        violations = []
        for pattern in patterns:
            if re.search(pattern, state.content, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    "pattern", f"Matched unwanted pattern: {pattern}", "warning"
                ))
        return violations
    return GuardrailRule(name="pattern", check=check, severity="warning")

def zero_output_rule() -> GuardrailRule:
    """Detect empty or whitespace-only output."""
    def check(state: L0State) -> list[GuardrailViolation]:
        if state.completed and not state.content.strip():
            return [GuardrailViolation("zero_output", "Empty output", "error")]
        return []
    return GuardrailRule(name="zero_output", check=check)

# Preset collections
def recommended_guardrails() -> list[GuardrailRule]:
    return [json_rule(), pattern_rule(), zero_output_rule()]
```

---

## Phase 6: Structured Output

### 6.1 Structured (`l0/structured.py`)

```python
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
from .core import l0
from .types import L0Options
from .stream import consume_stream
from ._utils import auto_correct_json

T = TypeVar("T", bound=BaseModel)

async def structured(
    schema: Type[T],
    options: L0Options,
    auto_correct: bool = True
) -> T:
    """Get structured output validated against Pydantic schema."""
    
    result = await l0(options)
    text = await consume_stream(result.stream)
    
    if auto_correct:
        text = auto_correct_json(text)
    
    try:
        return schema.model_validate_json(text)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")
```

---

## Phase 7: Advanced Features

### 7.1 Parallel Operations (`l0/parallel.py`)

```python
import asyncio
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")

async def parallel(
    tasks: list[Callable[[], Awaitable[T]]],
    concurrency: int = 5
) -> list[T]:
    """Run tasks with concurrency limit."""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def limited(task: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await task()
    
    return await asyncio.gather(*[limited(t) for t in tasks])

async def race(tasks: list[Callable[[], Awaitable[T]]]) -> T:
    """Return first successful result."""
    done, pending = await asyncio.wait(
        [asyncio.create_task(t()) for t in tasks],
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    return done.pop().result()
```

### 7.2 Consensus (`l0/consensus.py`)

```python
from typing import TypeVar, Callable, Awaitable, Literal
from collections import Counter

T = TypeVar("T")
Strategy = Literal["unanimous", "majority", "best"]

async def consensus(
    tasks: list[Callable[[], Awaitable[T]]],
    strategy: Strategy = "majority"
) -> T:
    """Run multiple tasks and resolve consensus."""
    results = await asyncio.gather(*[t() for t in tasks])
    
    match strategy:
        case "unanimous":
            if len(set(str(r) for r in results)) == 1:
                return results[0]
            raise ValueError("No unanimous consensus")
        
        case "majority":
            counter = Counter(str(r) for r in results)
            winner = counter.most_common(1)[0][0]
            for r in results:
                if str(r) == winner:
                    return r
            raise ValueError("No majority")
        
        case "best":
            return results[0]  # First result as default
```

---

## Phase 8: Internal Utilities

### 8.1 Utils (`l0/_utils.py`)

```python
import re

def auto_correct_json(text: str) -> str:
    """Auto-correct common JSON errors."""
    # Remove markdown fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    
    # Remove trailing commas
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    
    # Balance braces
    opens = text.count("{") - text.count("}")
    if opens > 0:
        text += "}" * opens
    
    brackets = text.count("[") - text.count("]")
    if brackets > 0:
        text += "]" * brackets
    
    return text.strip()
```

---

## Dependencies

Update `pyproject.toml`:

```toml
[project]
dependencies = [
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "orjson>=3.9.0",
    "typing-extensions>=4.9.0",
]

[project.optional-dependencies]
openai = ["openai>=1.30"]
anthropic = ["anthropic>=0.25"]
observability = [
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "sentry-sdk>=1.40",
]
dev = [
    "pytest",
    "pytest-asyncio",
    "pytest-mock",
    "coverage",
    "ruff",
    "mypy",
]
```

**Removed:**
- `anyio` - Using pure asyncio
- `tenacity` - Own retry implementation
- `regex` - Standard `re` is sufficient
- `uuid6` - Standard `uuid` is fine

---

## Implementation Order

### Sprint 1: Core
1. `l0/types.py` - All types
2. `l0/errors.py` - Error categorization
3. `l0/events.py` - Central event bus
4. `l0/_utils.py` - Internal utilities
5. Unit tests

### Sprint 2: Runtime
1. `l0/retry.py` - RetryManager
2. `l0/state.py` - State management
3. `l0/stream.py` - Stream utilities
4. Unit tests

### Sprint 3: Core + Adapters
1. `l0/core.py` - Main l0() function
2. `l0/adapters.py` - All adapters
3. Integration tests

### Sprint 4: Features
1. `l0/guardrails.py` - Engine + rules
2. `l0/structured.py` - Pydantic integration
3. `l0/parallel.py` - Concurrency utilities
4. `l0/consensus.py` - Multi-model consensus
5. Feature tests

### Sprint 5: Polish
1. `l0/__init__.py` - Public exports
2. Documentation
3. Examples
4. Final test coverage

---

## Success Criteria

1. Pure asyncio - no compatibility layers
2. Own retry logic - fully deterministic
3. Central event bus - single observability point
4. Type-safe with mypy
5. Test coverage >80%
6. Clean, flat structure
