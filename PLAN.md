# L0 Python Port Plan

This document outlines the plan to port the L0 TypeScript library to Python.

## Overview

**L0** is a production-grade reliability and observability layer for AI/LLM streaming applications. It wraps unreliable LLM streams with retry logic, guardrails, fallbacks, drift detection, and comprehensive monitoring.

**Source:** `ts/` directory (~27,300 lines of TypeScript)
**Target:** Python 3.10+ with async/await

---

## Phase 1: Core Foundation

### 1.1 Project Structure Setup

Create the package structure:

```
l0/
├── __init__.py              # Main exports
├── py.typed                 # PEP 561 marker
├── types.py                 # Core type definitions
├── core.py                  # Main l0() function
├── adapters/
│   ├── __init__.py
│   ├── base.py              # Adapter Protocol
│   ├── openai.py            # OpenAI adapter
│   ├── anthropic.py         # Anthropic adapter
│   └── registry.py          # Auto-detection registry
├── guardrails/
│   ├── __init__.py
│   ├── engine.py            # Guardrail execution engine
│   ├── rules.py             # Built-in rules (JSON, markdown, patterns)
│   └── types.py             # Guardrail types
├── runtime/
│   ├── __init__.py
│   ├── retry.py             # RetryManager with backoff strategies
│   ├── stream.py            # Stream processing utilities
│   ├── state.py             # L0State management
│   ├── state_machine.py     # Runtime state machine
│   ├── events.py            # Event normalization
│   └── callbacks.py         # Lifecycle callback management
└── utils/
    ├── __init__.py
    ├── errors.py            # Error categorization (12+ network patterns)
    ├── timers.py            # Backoff calculation, sleep utilities
    ├── auto_correct.py      # JSON auto-correction
    └── tokens.py            # Token analysis utilities
```

### 1.2 Core Types (`l0/types.py`)

Port the following from `ts/src/types/`:

- `L0Event` - Unified event format (token, message, data, progress, error, complete)
- `L0State` - Runtime state tracking
- `L0Options` - Configuration options
- `L0Result` - Result container
- `L0Telemetry` - Performance metrics
- `ErrorCategory` - Error classification enum
- `RetryReason` - Retry reason enum
- `BackoffStrategy` - Backoff strategy enum

Use Python equivalents:
- `TypedDict` for data-only structures
- `@dataclass` for structures with methods
- `Enum` for error categories and strategies
- `Protocol` for adapter interface
- `Literal` for string unions

### 1.3 Error Utilities (`l0/utils/errors.py`)

Port from `ts/src/utils/errors.ts`:

- `categorize_error()` - Classify errors into categories
- `is_network_error()` - Detect 12+ network error patterns
- `is_retryable()` - Determine if error should trigger retry
- Network error patterns:
  - Connection reset, refused, timeout
  - DNS failures, socket errors
  - SSL/TLS errors
  - HTTP 429, 500-599 status codes

### 1.4 Timer Utilities (`l0/utils/timers.py`)

Port from `ts/src/utils/timers.ts`:

- `calculate_backoff()` - Compute delay with strategy
- Backoff strategies:
  - `exponential` - 2^attempt * base
  - `linear` - attempt * base
  - `fixed` - constant delay
  - `fixed_jitter` - AWS-style (half + random half)
  - `full_jitter` - random up to cap
- `sleep_with_jitter()` - Async sleep with optional jitter

---

## Phase 2: Runtime Core

### 2.1 Retry Manager (`l0/runtime/retry.py`)

Port from `ts/src/runtime/retry.ts`:

- `RetryManager` class with:
  - Smart error categorization
  - Separate counters for network vs model errors
  - Network errors: infinite retry, no count
  - Transient errors (429, 503): infinite retry, no count
  - Model errors: count toward limit
  - Content errors: count toward limit
  - Fatal errors: no retry
- Error-type-specific delays:
  - Connection dropped: 1000ms
  - DNS error: 3000ms
  - Rate limit: respect Retry-After header
- `should_retry()` - Decision logic
- `get_delay()` - Calculate next delay
- `record_attempt()` - Track attempt

### 2.2 State Management (`l0/runtime/state.py`)

Port from `ts/src/runtime/state.ts`:

- `L0State` dataclass with:
  - `content: str` - Accumulated text
  - `checkpoint: str` - Last known good state
  - `token_count: int` - Token counter
  - `model_retry_count: int` - Model error retries
  - `network_retry_count: int` - Network retries
  - `fallback_index: int` - Current fallback
  - `violations: list[GuardrailViolation]`
  - `drift_detected: bool`
  - `completed: bool`
  - Timing fields (`first_token_at`, `last_token_at`, `duration`)
- State update methods
- Checkpoint management

### 2.3 State Machine (`l0/runtime/state_machine.py`)

Port from `ts/src/runtime/state-machine.ts`:

- Runtime states: `PENDING`, `ACTIVE`, `COMPLETING`, `COMPLETED`, `ERRORED`, `ABORTED`
- Valid transitions enforcement
- Event emission on state change

### 2.4 Stream Processing (`l0/runtime/stream.py`)

Port from `ts/src/runtime/stream.ts`:

- `consume_stream()` - Async generator consumer
- `get_text()` - Extract text from result
- Stream transformation utilities
- Event batching for callbacks

### 2.5 Core l0 Function (`l0/core.py`)

Port from `ts/src/runtime/l0.ts`:

Main `l0()` async function:
1. Validate options
2. Initialize state and retry manager
3. Create stream wrapper
4. Process events with:
   - Guardrail checks at intervals
   - Checkpoint saving
   - Timeout detection
   - Abort handling
5. Handle retries and fallbacks
6. Return `L0Result`

---

## Phase 3: Adapters

### 3.1 Adapter Protocol (`l0/adapters/base.py`)

Port from `ts/src/adapters/`:

```python
@runtime_checkable
class L0Adapter(Protocol[StreamT]):
    name: str
    
    def detect(self, stream: Any) -> TypeGuard[StreamT]:
        """Optional auto-detection."""
        ...
    
    async def wrap(self, stream: StreamT) -> AsyncGenerator[L0Event, None]:
        """Wrap SDK stream into L0Events."""
        ...
```

### 3.2 OpenAI Adapter (`l0/adapters/openai.py`)

Port from `ts/src/adapters/openai.ts`:

- Detect OpenAI stream objects
- Convert chunks to L0Events
- Handle tool calls
- Extract usage information

### 3.3 Anthropic Adapter (`l0/adapters/anthropic.py`)

Port from `ts/src/adapters/anthropic.ts`:

- Detect Anthropic stream objects
- Convert message events to L0Events
- Handle tool use blocks
- Extract usage information

### 3.4 Adapter Registry (`l0/adapters/registry.py`)

Port from `ts/src/adapters/registry.ts`:

- `register_adapter()` - Add adapter to registry
- `detect_adapter()` - Auto-detect from stream
- `get_adapter()` - Lookup by name

---

## Phase 4: Guardrails

### 4.1 Guardrail Types (`l0/guardrails/types.py`)

Port from `ts/src/types/guardrails.ts`:

- `GuardrailRule` Protocol
- `GuardrailContext` dataclass
- `GuardrailViolation` dataclass
- Severity levels: `warning`, `error`, `fatal`

### 4.2 Guardrail Engine (`l0/guardrails/engine.py`)

Port from `ts/src/guardrails/engine.ts`:

- `GuardrailEngine` class
- Fast path (sync) for small content (<1KB delta, <5KB total)
- Slow path (async) for large content
- Batch violation collection
- Early termination on fatal

### 4.3 Built-in Rules (`l0/guardrails/rules.py`)

Port from `ts/src/guardrails/`:

- `json_rule()` - JSON structure validation
- `strict_json_rule()` - Strict JSON parsing
- `markdown_rule()` - Markdown fence validation
- `latex_rule()` - LaTeX environment balancing
- `pattern_rule()` - Anti-AI phrase detection
- `custom_pattern_rule()` - Custom regex patterns
- `zero_output_rule()` - Empty output detection

---

## Phase 5: Structured Output

### 5.1 Structured Output (`l0/structured.py`)

Port from `ts/src/structured.ts`:

- `structured()` async function
- Pydantic schema validation (instead of Zod)
- Auto-correction of JSON errors:
  - Missing braces/brackets
  - Trailing commas
  - Markdown fence removal
- Retry on validation failure
- Type-safe output with generics

### 5.2 Auto-Correction (`l0/utils/auto_correct.py`)

Port from `ts/src/utils/autoCorrect.ts`:

- `auto_correct_json()` - Fix common JSON errors
- `extract_json_from_markdown()` - Remove code fences
- `balance_braces()` - Fix missing closers
- `remove_trailing_commas()` - Clean syntax

---

## Phase 6: Advanced Features

### 6.1 Parallel Operations (`l0/parallel.py`)

Port from `ts/src/runtime/parallel.ts`:

- `parallel()` - Run N operations with concurrency limit
- `race()` - First successful result wins
- `batched()` - Process in batches
- Use `asyncio.gather()` and `asyncio.Semaphore`

### 6.2 Consensus (`l0/consensus.py`)

Port from `ts/src/consensus.ts`:

- `consensus()` async function
- Strategies: `unanimous`, `majority`, `weighted`, `best`
- Conflict resolution: `vote`, `merge`, `best`
- Similarity matrix calculation
- Field-level consensus tracking

### 6.3 Pipeline (`l0/pipeline.py`)

Port from `ts/src/pipeline.ts`:

- `pipe()` - Chain operations
- `create_pipeline()` - Build reusable pipeline
- Stage execution with error handling

### 6.4 Document Windows (`l0/window.py`)

Port from `ts/src/window.ts`:

- `create_window()` - Chunking interface
- Chunking strategies:
  - `token` - Fixed token count
  - `char` - Fixed character count
  - `paragraph` - Paragraph boundaries
  - `sentence` - Sentence boundaries
- Configurable overlap
- Result merging

---

## Phase 7: Observability

### 7.1 Event Types (`l0/observability/events.py`)

Port from `ts/src/types/observability.ts`:

- 80+ event types organized by category
- Session, Stream, Adapter events
- Retry, Fallback events
- Guardrail, Drift events
- Structured output events

### 7.2 Monitoring (`l0/observability/monitoring.py`)

Port from `ts/src/runtime/monitoring.ts`:

- `create_monitor()` - Factory function
- Event emission
- Sampling support
- Callback management

### 7.3 OpenTelemetry Integration (`l0/observability/opentelemetry.py`)

Port from `ts/src/runtime/opentelemetry.ts`:

- Span creation for operations
- Attribute recording
- Error tracking
- Metrics export

### 7.4 Sentry Integration (`l0/observability/sentry.py`)

Port from `ts/src/runtime/sentry.ts`:

- Error capture
- Breadcrumb trails
- Context enrichment

---

## Phase 8: Testing

### 8.1 Test Structure

```
tests/
├── conftest.py              # Fixtures and mocks
├── unit/
│   ├── test_types.py
│   ├── test_errors.py
│   ├── test_timers.py
│   ├── test_retry.py
│   ├── test_state.py
│   ├── test_guardrails.py
│   ├── test_structured.py
│   └── test_adapters.py
├── integration/
│   ├── test_core.py
│   ├── test_consensus.py
│   └── test_parallel.py
└── fixtures/
    └── streams.py           # Mock stream generators
```

### 8.2 Testing Approach

- Use `pytest` + `pytest-asyncio`
- Mock LLM responses with async generators
- Property-based testing with `hypothesis` for edge cases
- Port key test cases from TypeScript suite

---

## Implementation Order

### Sprint 1: Foundation (Core types, errors, timers)
1. Create package structure
2. Port `l0/types.py`
3. Port `l0/utils/errors.py`
4. Port `l0/utils/timers.py`
5. Add unit tests

### Sprint 2: Runtime (Retry, state, stream)
1. Port `l0/runtime/retry.py`
2. Port `l0/runtime/state.py`
3. Port `l0/runtime/state_machine.py`
4. Port `l0/runtime/stream.py`
5. Add unit tests

### Sprint 3: Core Function
1. Port `l0/core.py` (main l0 function)
2. Port `l0/runtime/callbacks.py`
3. Integration tests

### Sprint 4: Adapters
1. Port `l0/adapters/base.py`
2. Port `l0/adapters/openai.py`
3. Port `l0/adapters/anthropic.py`
4. Port `l0/adapters/registry.py`
5. Add adapter tests

### Sprint 5: Guardrails
1. Port `l0/guardrails/types.py`
2. Port `l0/guardrails/engine.py`
3. Port `l0/guardrails/rules.py`
4. Add guardrail tests

### Sprint 6: Structured Output
1. Port `l0/structured.py`
2. Port `l0/utils/auto_correct.py`
3. Add structured output tests

### Sprint 7: Advanced Features
1. Port `l0/parallel.py`
2. Port `l0/consensus.py`
3. Port `l0/pipeline.py`
4. Port `l0/window.py`
5. Add integration tests

### Sprint 8: Observability
1. Port observability events
2. Port monitoring
3. Port OpenTelemetry integration
4. Port Sentry integration
5. Add observability tests

---

## Key Design Decisions

### 1. Async-First
All I/O operations use `async`/`await` with `asyncio`.

### 2. Type Safety
Full type hints using `typing` module, with `py.typed` marker for PEP 561 compliance.

### 3. Pydantic for Validation
Use Pydantic v2 instead of Zod for structured output validation.

### 4. Protocol-Based Adapters
Use `typing.Protocol` for adapter interface (structural subtyping).

### 5. Zero Core Dependencies
Core module has minimal dependencies; optional features use extras.

### 6. Performance Preservation
- Fast/slow path for guardrails
- Lazy loading for optional features
- Use `orjson` for fast JSON parsing

### 7. Pythonic Naming
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants

---

## Dependencies (Already in pyproject.toml)

**Core:**
- `httpx` - HTTP client (for error detection patterns)
- `anyio` - Async compatibility
- `pydantic` - Validation and serialization
- `orjson` - Fast JSON parsing
- `regex` - Advanced regex support
- `tenacity` - Retry utilities (reference, may use custom)
- `typing-extensions` - Backport typing features

**Optional:**
- `openai` - OpenAI SDK adapter
- `litellm` - Multi-provider adapter
- `opentelemetry-*` - Observability
- `sentry-sdk` - Error tracking
- `jsonschema` - JSON Schema validation
- `json-repair` - JSON auto-correction
- `uvloop` - Fast event loop (non-Windows)

---

## Success Criteria

1. All core functionality ported and working
2. Type safety with mypy passing
3. Test coverage >80%
4. Performance comparable to TypeScript version
5. Clean, idiomatic Python code
6. Comprehensive documentation
7. Working examples for common use cases
