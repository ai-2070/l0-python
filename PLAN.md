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
├── version.py            # Semantic version string
├── logging.py            # Internal debug logs (disabled by default)
│
├── types.py              # All type definitions
├── events.py             # Central event bus + event types
├── errors.py             # Error categorization
│
├── runtime.py            # Main l0() function (execution engine)
├── retry.py              # RetryManager (own implementation)
├── state.py              # L0State management
├── stream.py             # Stream utilities
│
├── adapters.py           # All adapters in one file (simple)
├── guardrails.py         # Engine + built-in rules + drift detection
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

### 1.1 Version (`l0/version.py`)

```python
__version__ = "0.1.0"
```

### 1.2 Logging (`l0/logging.py`)

```python
import logging

logger = logging.getLogger("l0")
logger.addHandler(logging.NullHandler())  # Disabled by default

def enable_debug() -> None:
    """Enable debug logging for L0."""
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[l0] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
```

### 1.3 Types (`l0/types.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, AsyncIterator, Any
from enum import Enum
import time


# ─────────────────────────────────────────────────────────────────────────────
# Event Types (matches TS: token | message | data | progress | error | complete)
# ─────────────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    TOKEN = "token"
    MESSAGE = "message"
    DATA = "data"
    PROGRESS = "progress"      # TS has this
    TOOL_CALL = "tool_call"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class L0Event:
    """Unified event from adapter-normalized LLM stream."""
    type: EventType
    value: str | None = None               # Matches TS field name
    data: dict[str, Any] | None = None     # Tool call payload, progress, or misc
    error: Exception | None = None
    usage: dict[str, int] | None = None
    timestamp: float | None = None         # Matches TS field name (optional in TS)


# ─────────────────────────────────────────────────────────────────────────────
# Error Categories (matches TS ErrorCategory enum)
# ─────────────────────────────────────────────────────────────────────────────

class ErrorCategory(str, Enum):
    NETWORK = "network"        # Retry forever, doesn't count
    TRANSIENT = "transient"    # Retry forever (429, 503), doesn't count
    MODEL = "model"            # Counts toward limit
    CONTENT = "content"        # Counts toward limit
    PROVIDER = "provider"      # May retry depending on status
    FATAL = "fatal"            # Don't retry
    INTERNAL = "internal"      # Don't retry (bugs)


class BackoffStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FULL_JITTER = "full-jitter"      # TS uses hyphen
    FIXED_JITTER = "fixed-jitter"    # TS uses hyphen


# ─────────────────────────────────────────────────────────────────────────────
# State Object (matches TS L0State)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class L0State:
    content: str = ""
    checkpoint: str = ""
    token_count: int = 0
    model_retry_count: int = 0
    network_retry_count: int = 0
    fallback_index: int = 0
    violations: list[Any] = field(default_factory=list)
    drift_detected: bool = False           # TS has this
    completed: bool = False
    aborted: bool = False
    first_token_at: float | None = None
    last_token_at: float | None = None
    duration: float | None = None          # TS has this
    resumed: bool = False                  # TS has this
    network_errors: list[Any] = field(default_factory=list)  # TS has this


# ─────────────────────────────────────────────────────────────────────────────
# Retry + Timeout Configs (matches TS defaults)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    attempts: int = 3                      # TS default: 3
    max_retries: int = 6                   # TS default: 6 (absolute max)
    base_delay_ms: int = 1000              # TS default: 1000ms
    max_delay_ms: int = 10000              # TS default: 10000ms
    strategy: BackoffStrategy = BackoffStrategy.FIXED_JITTER


@dataclass
class TimeoutConfig:
    initial_token_ms: int = 5000           # TS default: 5000ms
    inter_token_ms: int = 10000            # TS default: 10000ms


# ─────────────────────────────────────────────────────────────────────────────
# Options + Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class L0Options:
    stream: Callable[[], AsyncIterator[Any]]
    fallbacks: list[Callable[[], AsyncIterator[Any]]] = field(default_factory=list)
    guardrails: list[Any] = field(default_factory=list)
    retry: RetryConfig | None = None
    timeout: TimeoutConfig | None = None
    adapter: Any | str | None = None
    on_event: Callable[[Any], None] | None = None
    meta: dict[str, Any] | None = None     # TS has this for observability


@dataclass
class L0Result:
    stream: AsyncIterator[L0Event]
    state: L0State
    abort: Callable[[], None]
    errors: list[Exception] = field(default_factory=list)  # TS has this
```

### 1.4 Central Event Bus (`l0/events.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
from uuid_extensions import uuid7str
import time


# ─────────────────────────────────────────────────────────────────────────────
# Event Types (matches TS EventType - UPPER_CASE values)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Observability Event (matches TS L0ObservabilityEvent)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObservabilityEvent:
    type: ObservabilityEventType
    ts: float                                    # Unix epoch milliseconds
    stream_id: str                               # UUID v7 (matches TS streamId)
    meta: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Central event bus for all L0 observability."""

    def __init__(
        self,
        handler: Callable[[ObservabilityEvent], None] | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self._handler = handler
        self._stream_id = uuid7str()
        self._meta = meta or {}

    @property
    def stream_id(self) -> str:
        return self._stream_id

    def emit(self, event_type: ObservabilityEventType, **event_meta: Any) -> None:
        if not self._handler:
            return

        event = ObservabilityEvent(
            type=event_type,
            ts=time.time() * 1000,               # TS uses milliseconds
            stream_id=self._stream_id,
            meta={**self._meta, **event_meta},
        )
        self._handler(event)
```

### 1.5 Error Categorization (`l0/errors.py`)

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
from .types import ErrorCategory, BackoffStrategy, RetryConfig
from .errors import categorize_error
from .logging import logger


class RetryManager:
    """Manages retry logic with error-aware backoff. Matches TS RetryManager."""
    
    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self.model_retry_count = 0
        self.network_retry_count = 0
        self.total_retries = 0
    
    def should_retry(self, error: Exception) -> bool:
        category = categorize_error(error)
        logger.debug(f"Error category: {category}, model_retries: {self.model_retry_count}")
        
        # Check absolute max
        if self.total_retries >= self.config.max_retries:
            return False
        
        if category in (ErrorCategory.FATAL, ErrorCategory.INTERNAL):
            return False
        if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
            return True  # Always retry, doesn't count toward model limit
        
        # MODEL or CONTENT - counts toward limit
        return self.model_retry_count < self.config.attempts
    
    def record_attempt(self, error: Exception) -> None:
        category = categorize_error(error)
        self.total_retries += 1
        if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
            self.network_retry_count += 1
        else:
            self.model_retry_count += 1
    
    def get_delay_ms(self, error: Exception) -> int:
        """Get delay in milliseconds (matches TS)."""
        category = categorize_error(error)
        attempt = self.network_retry_count if category == ErrorCategory.NETWORK else self.model_retry_count
        
        base = self.config.base_delay_ms
        cap = self.config.max_delay_ms
        
        match self.config.strategy:
            case BackoffStrategy.EXPONENTIAL:
                delay = min(base * (2 ** attempt), cap)
            case BackoffStrategy.LINEAR:
                delay = min(base * (attempt + 1), cap)
            case BackoffStrategy.FIXED:
                delay = base
            case BackoffStrategy.FIXED_JITTER:
                temp = min(base * (2 ** attempt), cap)
                delay = temp // 2 + int(random.random() * (temp // 2))
            case BackoffStrategy.FULL_JITTER:
                delay = int(random.random() * min(base * (2 ** attempt), cap))
        
        logger.debug(f"Retry delay: {delay}ms (strategy: {self.config.strategy})")
        return delay
    
    async def wait(self, error: Exception) -> None:
        delay_ms = self.get_delay_ms(error)
        await asyncio.sleep(delay_ms / 1000)
    
    def get_state(self) -> dict:
        return {
            "model_retry_count": self.model_retry_count,
            "network_retry_count": self.network_retry_count,
            "total_retries": self.total_retries,
        }
    
    def reset(self) -> None:
        self.model_retry_count = 0
        self.network_retry_count = 0
        self.total_retries = 0
```

### 2.2 State (`l0/state.py`)

```python
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


def mark_completed(state: L0State) -> None:
    """Mark stream as completed and calculate duration."""
    state.completed = True
    if state.first_token_at is not None:
        state.duration = (state.last_token_at or time.time()) - state.first_token_at
```

### 2.3 Stream Utilities (`l0/stream.py`)

```python
from typing import AsyncIterator, TYPE_CHECKING
from .types import L0Event, EventType

if TYPE_CHECKING:
    from .types import L0Result


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

## Phase 3: Runtime Engine

### 3.1 Main l0() Function (`l0/runtime.py`)

```python
import asyncio
from typing import AsyncIterator
from .types import L0Options, L0Result, L0State, L0Event, EventType, RetryConfig
from .retry import RetryManager
from .state import create_state, append_token, update_checkpoint
from .events import EventBus, ObservabilityEventType
from .adapters import detect_adapter
from .guardrails import check_guardrails
from .logging import logger

async def l0(options: L0Options) -> L0Result:
    """Main L0 wrapper function."""
    
    state = create_state()
    retry_mgr = RetryManager(options.retry)
    event_bus = EventBus(options.on_event, meta=options.meta)
    errors: list[Exception] = []
    aborted = False
    
    logger.debug(f"Starting L0 stream: {event_bus.stream_id}")
    
    def abort() -> None:
        nonlocal aborted
        aborted = True
        logger.debug("Abort requested")
    
    async def run_stream() -> AsyncIterator[L0Event]:
        nonlocal state
        
        streams = [options.stream] + options.fallbacks
        
        for fallback_idx, stream_fn in enumerate(streams):
            state.fallback_index = fallback_idx
            
            if fallback_idx > 0:
                logger.debug(f"Trying fallback {fallback_idx}")
                event_bus.emit(ObservabilityEventType.FALLBACK_START, index=fallback_idx)
            
            while True:
                try:
                    event_bus.emit(ObservabilityEventType.STREAM_INIT)
                    raw_stream = await stream_fn()
                    adapter = detect_adapter(raw_stream, options.adapter)
                    event_bus.emit(ObservabilityEventType.STREAM_READY)
                    
                    async for event in adapter.wrap(raw_stream):
                        if aborted:
                            state.aborted = True
                            return
                        
                        if event.type == EventType.TOKEN and event.value:
                            append_token(state, event.value)
                            
                            # Check guardrails periodically
                            if state.token_count % 5 == 0 and options.guardrails:
                                event_bus.emit(ObservabilityEventType.GUARDRAIL_PHASE_START)
                                violations = check_guardrails(state, options.guardrails)
                                if violations:
                                    state.violations.extend(violations)
                                    event_bus.emit(
                                        ObservabilityEventType.GUARDRAIL_RULE_RESULT,
                                        violations=[v.__dict__ for v in violations]
                                    )
                                event_bus.emit(ObservabilityEventType.GUARDRAIL_PHASE_END)
                        
                        yield event
                    
                    # Success
                    mark_completed(state)
                    event_bus.emit(ObservabilityEventType.COMPLETE, token_count=state.token_count)
                    logger.debug(f"Stream complete: {state.token_count} tokens")
                    return
                    
                except Exception as e:
                    errors.append(e)
                    logger.debug(f"Stream error: {e}")
                    event_bus.emit(ObservabilityEventType.NETWORK_ERROR, error=str(e))
                    
                    if retry_mgr.should_retry(e):
                        retry_mgr.record_attempt(e)
                        state.model_retry_count = retry_mgr.model_retry_count
                        state.network_retry_count = retry_mgr.network_retry_count
                        event_bus.emit(
                            ObservabilityEventType.RETRY_ATTEMPT,
                            attempt=retry_mgr.total_retries
                        )
                        await retry_mgr.wait(e)
                        update_checkpoint(state)
                        continue
                    else:
                        # Try next fallback
                        break
        
        # All fallbacks exhausted
        event_bus.emit(ObservabilityEventType.RETRY_GIVE_UP)
        raise RuntimeError("All streams and fallbacks exhausted")
    
    return L0Result(
        stream=run_stream(),
        state=state,
        abort=abort,
        errors=errors,
    )
```

---

## Phase 4: Adapters

### 4.1 Adapters (`l0/adapters.py`)

Adapters handle their own metadata extraction (usage, tool calls):

```python
from typing import AsyncIterator, Any, Protocol, runtime_checkable
from .types import L0Event, EventType
from .logging import logger

@runtime_checkable
class Adapter(Protocol):
    name: str
    def detect(self, stream: Any) -> bool: ...
    async def wrap(self, stream: Any) -> AsyncIterator[L0Event]: ...

class OpenAIAdapter:
    name = "openai"
    
    def detect(self, stream: Any) -> bool:
        type_name = type(stream).__module__
        return "openai" in type_name
    
    async def wrap(self, stream: Any) -> AsyncIterator[L0Event]:
        usage = None
        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                
                if delta:
                    # Text content
                    if hasattr(delta, "content") and delta.content:
                        yield L0Event(type=EventType.TOKEN, value=delta.content)
                    
                    # Tool calls
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            yield L0Event(
                                type=EventType.TOOL_CALL,
                                data={
                                    "id": getattr(tc, "id", None),
                                    "name": getattr(tc.function, "name", None) if hasattr(tc, "function") else None,
                                    "arguments": getattr(tc.function, "arguments", None) if hasattr(tc, "function") else None,
                                }
                            )
            
            # Extract usage if present (typically on last chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "input_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(chunk.usage, "completion_tokens", 0),
                }
        
        yield L0Event(type=EventType.COMPLETE, usage=usage)


class AnthropicAdapter:
    name = "anthropic"
    
    def detect(self, stream: Any) -> bool:
        type_name = type(stream).__module__
        return "anthropic" in type_name
    
    async def wrap(self, stream: Any) -> AsyncIterator[L0Event]:
        usage = None
        async for event in stream:
            event_type = getattr(event, "type", None)
            
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and hasattr(delta, "text"):
                    yield L0Event(type=EventType.TOKEN, value=delta.text)
            
            elif event_type == "content_block_start":
                block = getattr(event, "content_block", None)
                if block and getattr(block, "type", None) == "tool_use":
                    yield L0Event(
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
                yield L0Event(type=EventType.COMPLETE, usage=usage)

# Registry
_adapters: list[Adapter] = [OpenAIAdapter(), AnthropicAdapter()]

def register_adapter(adapter: Adapter) -> None:
    """Register a custom adapter (takes priority)."""
    _adapters.insert(0, adapter)

def detect_adapter(stream: Any, hint: "Adapter | str | None" = None) -> Adapter:
    """Detect or lookup adapter for stream."""
    if hint is not None and not isinstance(hint, str):
        return hint
    
    if isinstance(hint, str):
        for a in _adapters:
            if a.name == hint:
                return a
        raise ValueError(f"Unknown adapter: {hint}")
    
    for a in _adapters:
        if a.detect(stream):
            logger.debug(f"Detected adapter: {a.name}")
            return a
    
    raise ValueError("No adapter found for stream")
```

---

## Phase 5: Guardrails

### 5.1 Guardrails (`l0/guardrails.py`)

Engine, built-in rules, and drift detection:

```python
import re
import json
from dataclasses import dataclass
from typing import Callable, Literal, TYPE_CHECKING
from .logging import logger

if TYPE_CHECKING:
    from .types import L0State

Severity = Literal["warning", "error", "fatal"]


@dataclass
class GuardrailViolation:
    """Matches TS GuardrailViolation interface."""
    rule: str                              # Name of the rule that was violated
    message: str                           # Human-readable message
    severity: Severity                     # Severity of the violation
    recoverable: bool = True               # Whether this violation is recoverable via retry
    position: int | None = None            # Position in content where violation occurred
    timestamp: float | None = None         # Timestamp when violation was detected
    context: dict[str, Any] | None = None  # Additional context about the violation
    suggestion: str | None = None          # Suggested fix or action


@dataclass
class GuardrailRule:
    """Matches TS GuardrailRule interface."""
    name: str                              # Unique name of the rule
    check: Callable[["L0State"], list[GuardrailViolation]]  # Check function
    description: str | None = None         # Description of what the rule checks
    streaming: bool = True                 # Whether to run on every token or only at completion
    severity: Severity = "error"           # Default severity for violations from this rule
    recoverable: bool = True               # Whether violations are recoverable via retry

def check_guardrails(state: "L0State", rules: list[GuardrailRule]) -> list[GuardrailViolation]:
    """Run all guardrail rules against current state."""
    violations = []
    for rule in rules:
        result = rule.check(state)
        if result:
            logger.debug(f"Guardrail '{rule.name}' triggered: {len(result)} violations")
        violations.extend(result)
    return violations

# ─────────────────────────────────────────────────────────────────────────────
# Built-in Rules
# ─────────────────────────────────────────────────────────────────────────────

def json_rule() -> GuardrailRule:
    """Check for balanced JSON braces."""
    def check(state: "L0State") -> list[GuardrailViolation]:
        content = state.content
        opens = content.count("{") + content.count("[")
        closes = content.count("}") + content.count("]")
        if opens < closes:
            return [GuardrailViolation("json", "Unbalanced JSON brackets", "error")]
        return []
    return GuardrailRule(name="json", check=check)

def strict_json_rule() -> GuardrailRule:
    """Validate complete JSON on completion."""
    def check(state: "L0State") -> list[GuardrailViolation]:
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
    
    def check(state: "L0State") -> list[GuardrailViolation]:
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
    def check(state: "L0State") -> list[GuardrailViolation]:
        if state.completed and not state.content.strip():
            return [GuardrailViolation("zero_output", "Empty output", "error")]
        return []
    return GuardrailRule(name="zero_output", check=check)

# ─────────────────────────────────────────────────────────────────────────────
# Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

def stall_rule(max_gap: float = 5.0) -> GuardrailRule:
    """Detect token stalls (no tokens for too long)."""
    import time
    
    def check(state: "L0State") -> list[GuardrailViolation]:
        if state.last_token_at is None:
            return []
        gap = time.time() - state.last_token_at
        if gap > max_gap:
            return [GuardrailViolation("stall", f"Token stall: {gap:.1f}s", "warning")]
        return []
    return GuardrailRule(name="stall", check=check, severity="warning")

def repetition_rule(window: int = 100, threshold: float = 0.5) -> GuardrailRule:
    """Detect repetitive output (model looping)."""
    def check(state: "L0State") -> list[GuardrailViolation]:
        content = state.content
        if len(content) < window * 2:
            return []
        
        recent = content[-window:]
        previous = content[-window*2:-window]
        
        # Simple similarity: count matching characters
        matches = sum(1 for a, b in zip(recent, previous) if a == b)
        similarity = matches / window
        
        if similarity > threshold:
            return [GuardrailViolation(
                "repetition",
                f"Repetitive output detected ({similarity:.0%} similar)",
                "error"
            )]
        return []
    return GuardrailRule(name="repetition", check=check)

# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────

def recommended_guardrails() -> list[GuardrailRule]:
    """Recommended set of guardrails."""
    return [json_rule(), pattern_rule(), zero_output_rule()]

def strict_guardrails() -> list[GuardrailRule]:
    """Strict guardrails including drift detection."""
    return [
        json_rule(),
        strict_json_rule(),
        pattern_rule(),
        zero_output_rule(),
        stall_rule(),
        repetition_rule(),
    ]
```

---

## Phase 6: Structured Output

### 6.1 Structured (`l0/structured.py`)

```python
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
from .runtime import l0
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

async def batched(
    items: list[T],
    handler: Callable[[T], Awaitable[T]],
    batch_size: int = 10
) -> list[T]:
    """Process items in batches."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[handler(item) for item in batch])
        results.extend(batch_results)
    return results
```

### 7.2 Consensus (`l0/consensus.py`)

```python
import asyncio
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

## Phase 9: Public API

### 9.1 Package Init (`l0/__init__.py`)

```python
"""L0 - Reliability layer for AI/LLM streaming."""

from .version import __version__
from .types import (
    L0Event,
    L0State,
    L0Options,
    L0Result,
    RetryConfig,
    TimeoutConfig,
    EventType,
    ErrorCategory,
    BackoffStrategy,
)
from .events import ObservabilityEvent, ObservabilityEventType, EventBus
from .runtime import l0
from .stream import consume_stream, get_text
from .adapters import Adapter, register_adapter, detect_adapter
from .guardrails import (
    GuardrailRule,
    GuardrailViolation,
    check_guardrails,
    json_rule,
    strict_json_rule,
    pattern_rule,
    zero_output_rule,
    stall_rule,
    repetition_rule,
    recommended_guardrails,
    strict_guardrails,
)
from .structured import structured
from .parallel import parallel, race, batched
from .consensus import consensus
from .logging import enable_debug

__all__ = [
    # Version
    "__version__",
    # Core
    "l0",
    "L0Event",
    "L0State",
    "L0Options",
    "L0Result",
    "RetryConfig",
    "TimeoutConfig",
    "EventType",
    "ErrorCategory",
    "BackoffStrategy",
    # Events
    "ObservabilityEvent",
    "ObservabilityEventType",
    "EventBus",
    # Stream
    "consume_stream",
    "get_text",
    # Adapters
    "Adapter",
    "register_adapter",
    "detect_adapter",
    # Guardrails
    "GuardrailRule",
    "GuardrailViolation",
    "check_guardrails",
    "json_rule",
    "strict_json_rule",
    "pattern_rule",
    "zero_output_rule",
    "stall_rule",
    "repetition_rule",
    "recommended_guardrails",
    "strict_guardrails",
    # Structured
    "structured",
    # Parallel
    "parallel",
    "race",
    "batched",
    # Consensus
    "consensus",
    # Debug
    "enable_debug",
]
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
    "uuid-extensions>=1.0.0",    # UUIDv7 support
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

---

## Implementation Order

### Sprint 1: Core
1. `l0/version.py` - Version string
2. `l0/logging.py` - Debug logging
3. `l0/types.py` - All types
4. `l0/errors.py` - Error categorization
5. `l0/events.py` - Central event bus
6. `l0/_utils.py` - Internal utilities
7. Unit tests

### Sprint 2: Runtime
1. `l0/retry.py` - RetryManager
2. `l0/state.py` - State management
3. `l0/stream.py` - Stream utilities
4. Unit tests

### Sprint 3: Runtime + Adapters
1. `l0/runtime.py` - Main l0() function
2. `l0/adapters.py` - All adapters
3. Integration tests

### Sprint 4: Features
1. `l0/guardrails.py` - Engine + rules + drift
2. `l0/structured.py` - Pydantic integration
3. `l0/parallel.py` - Concurrency utilities
4. `l0/consensus.py` - Multi-model consensus
5. Feature tests

### Sprint 5: Polish
1. `l0/__init__.py` - Public exports
2. `l0/py.typed` - PEP 561 marker
3. Documentation
4. Examples
5. Final test coverage

---

## Success Criteria

1. Pure asyncio - no compatibility layers
2. Own retry logic - fully deterministic
3. Central event bus - single observability point
4. Type-safe with mypy
5. Test coverage >80%
6. Clean, flat structure
7. Drift detection included
