# L0 Python Implementation Plan

## Overview

This document outlines the complete implementation plan for achieving full parity between the L0 Python library and the TypeScript reference implementation. The plan is organized into phases, with each phase building on the previous one.

---

## Phase 1: Core Runtime Enhancements

### 1.1 Retry System Enhancements

**Priority: High**

#### Missing Features

1. **`retryOn` Configuration**
   - Allow specifying which error types to retry on
   - Types: `zero_output`, `guardrail_violation`, `drift`, `incomplete`, `network_error`, `timeout`, `rate_limit`, `server_error`
   
   ```python
   @dataclass
   class Retry:
       attempts: int = 3
       max_retries: int = 6
       base_delay: float = 1.0
       max_delay: float = 10.0
       backoff: BackoffStrategy = "fixed-jitter"
       retry_on: list[str] | None = None  # NEW
       error_type_delays: ErrorTypeDelays | None = None
       max_error_history: int | None = None
       should_retry: Callable | None = None  # NEW - veto callback
       calculate_delay: Callable | None = None  # NEW - custom delay
   ```

2. **`shouldRetry` Callback**
   - Async callback for custom retry veto logic
   - Signature: `async (error, state, attempt, category) -> bool`

3. **`calculateDelay` Callback**
   - Custom delay calculation function
   - Signature: `(context: RetryContext) -> float`

#### Implementation Tasks

- [ ] Add `retry_on` field to `Retry` dataclass
- [ ] Add `should_retry` callback support
- [ ] Add `calculate_delay` callback support
- [ ] Update `RetryManager` to respect `retry_on` filter
- [ ] Add tests for all new retry options

### 1.2 Check Intervals Configuration

**Priority: High**

#### Missing Features

Add `CheckIntervals` configuration for tuning guardrail/drift/checkpoint frequency:

```python
@dataclass
class CheckIntervals:
    guardrails: int = 5   # Check guardrails every N tokens
    drift: int = 10       # Check drift every N tokens
    checkpoint: int = 10  # Save checkpoint every N tokens
```

#### Implementation Tasks

- [ ] Create `CheckIntervals` dataclass in `types.py`
- [ ] Add `check_intervals` parameter to `l0.run()` and `l0.wrap()`
- [ ] Implement token counting in runtime
- [ ] Apply check intervals in guardrail execution
- [ ] Apply check intervals in drift detection
- [ ] Apply check intervals in checkpoint saving
- [ ] Add tests for check interval behavior

### 1.3 User Metadata

**Priority: Medium**

#### Missing Features

Add `meta` parameter for attaching user metadata to all observability events:

```python
result = await l0.run(
    stream=...,
    meta={"request_id": "req_123", "user_id": "user_456"}
)
```

#### Implementation Tasks

- [ ] Add `meta` parameter to `l0.run()` options
- [ ] Propagate meta to all `ObservabilityEvent` instances
- [ ] Include meta in telemetry export
- [ ] Add tests for metadata propagation

### 1.4 Continuation Enhancements

**Priority: High**

#### Missing Features

1. **`buildContinuationPrompt` Callback**
   - Allow users to modify the prompt when resuming from checkpoint
   - Signature: `(checkpoint: str) -> str`

2. **State Properties**
   - `resumed: bool` - Whether continuation was used
   - `resume_point: str` - The checkpoint content
   - `resume_from: int` - Character offset where resume occurred

```python
result = await l0.run(
    stream=lambda: create_stream(prompt=continuation_prompt or original),
    continue_from_last_known_good_token=True,
    build_continuation_prompt=lambda checkpoint: f"{original}\n\nContinue:\n{checkpoint}",
    retry=Retry(attempts=3),
)
```

#### Implementation Tasks

- [ ] Add `build_continuation_prompt` callback parameter
- [ ] Add `resumed`, `resume_point`, `resume_from` to `State`
- [ ] Implement checkpoint validation against guardrails before resume
- [ ] Emit `RESUME_START` / `RESUME_END` observability events
- [ ] Add tests for continuation with custom prompts

---

## Phase 2: Lifecycle Callbacks

### 2.1 Complete Callback Set

**Priority: High**

#### Current State

Python has some callbacks but needs verification and completion.

#### Required Callbacks

| Callback | Signature | Status |
|----------|-----------|--------|
| `on_start` | `(attempt: int, is_retry: bool, is_fallback: bool) -> None` | Verify |
| `on_complete` | `(state: State) -> None` | Verify |
| `on_error` | `(error: Error, will_retry: bool, will_fallback: bool) -> None` | Verify |
| `on_event` | `(event: Event) -> None` | Add |
| `on_violation` | `(violation: GuardrailViolation) -> None` | Verify |
| `on_retry` | `(attempt: int, reason: str) -> None` | Verify |
| `on_fallback` | `(index: int, reason: str) -> None` | Verify |
| `on_resume` | `(checkpoint: str, token_count: int) -> None` | Add |
| `on_checkpoint` | `(checkpoint: str, token_count: int) -> None` | Add |
| `on_timeout` | `(type: str, elapsed_ms: float) -> None` | Verify |
| `on_abort` | `(token_count: int, content_length: int) -> None` | Add |
| `on_drift` | `(types: list[str], confidence: float) -> None` | Add |
| `on_tool_call` | `(tool_name: str, tool_call_id: str, args: dict) -> None` | Verify |

#### Implementation Tasks

- [ ] Audit existing callbacks for signature parity
- [ ] Add missing callbacks: `on_event`, `on_resume`, `on_checkpoint`, `on_abort`, `on_drift`
- [ ] Ensure all callbacks are fire-and-forget (non-blocking)
- [ ] Add callback wrapper to catch and log errors silently
- [ ] Add comprehensive callback tests

---

## Phase 3: Guardrails Enhancements

### 3.1 LaTeX Rule

**Priority: Low**

#### Implementation Tasks

- [ ] Verify `latex_rule()` exists in Python
- [ ] If missing, implement LaTeX environment validation
- [ ] Add tests for LaTeX guardrail

### 3.2 Custom Pattern Rule Enhancement

**Priority: Medium**

#### Implementation Tasks

- [ ] Verify `custom_pattern_rule()` API matches TS
- [ ] Ensure pattern matching uses proper regex
- [ ] Add severity parameter support

---

## Phase 4: Parallel Operations

### 4.1 Operation Pool

**Priority: Medium**

#### Missing Features

`OperationPool` for dynamic workload management:

```python
pool = l0.create_pool(3)  # Max 3 concurrent operations

result1 = pool.execute(stream=lambda: stream1)
result2 = pool.execute(stream=lambda: stream2)

await pool.drain()

print(pool.get_queue_length())
print(pool.get_active_workers())
```

#### Implementation Tasks

- [ ] Create `OperationPool` class
- [ ] Implement `execute()` method with queueing
- [ ] Implement `drain()` to wait for all operations
- [ ] Implement `get_queue_length()` and `get_active_workers()`
- [ ] Add `create_pool()` factory function
- [ ] Add tests for pool behavior

### 4.2 Race Result Enhancement

**Priority: Low**

#### Implementation Tasks

- [ ] Verify `race()` returns `winner_index`
- [ ] Ensure proper cancellation of losing streams

### 4.3 Parallel Result Enhancement

**Priority: Low**

#### Implementation Tasks

- [ ] Verify `ParallelResult` has all fields: `success_count`, `failure_count`, `duration`, `all_succeeded`
- [ ] Add `aggregated_telemetry` field

---

## Phase 5: Structured Output

### 5.1 Helper Functions

**Priority: Medium**

#### Missing Features

```python
# Quick object schema
result = await l0.structured_object(
    {"name": str, "age": int},
    stream=...
)

# Quick array schema
result = await l0.structured_array(
    ItemModel,
    stream=...
)
```

#### Implementation Tasks

- [ ] Add `structured_object()` helper function
- [ ] Add `structured_array()` helper function
- [ ] Add tests for helpers

### 5.2 Presets

**Priority: Medium**

#### Missing Features

```python
# Presets
minimal_structured = StructuredConfig(auto_correct=False, attempts=1)
recommended_structured = StructuredConfig(auto_correct=True, attempts=2)
strict_structured = StructuredConfig(auto_correct=True, strict_mode=True, attempts=3)
```

#### Implementation Tasks

- [ ] Create `StructuredConfig` dataclass
- [ ] Add preset constants
- [ ] Add `strict_mode` option (reject unknown fields)
- [ ] Add tests for presets

### 5.3 JSON Schema Adapter

**Priority: Low**

#### Missing Features

Support for raw JSON Schema (beyond Pydantic):

```python
from l0 import register_json_schema_adapter, wrap_json_schema

register_json_schema_adapter({
    "validate": lambda schema, data: ...,
    "format_errors": lambda errors: ...
})

schema = wrap_json_schema({
    "type": "object",
    "properties": {"name": {"type": "string"}}
})
```

#### Implementation Tasks

- [ ] Create JSON Schema adapter interface
- [ ] Implement `register_json_schema_adapter()`
- [ ] Implement `wrap_json_schema()`
- [ ] Add tests for JSON Schema validation

---

## Phase 6: Pipeline/Pipe Operations

### 6.1 Streaming Pipelines

**Priority: High**

#### Missing Features

Multi-step streaming workflows:

```python
result = await l0.pipe(
    steps=[
        {"stream": lambda: summarize_stream, "guardrails": summary_rules},
        {"stream": lambda prev: refine_stream(prev), "guardrails": refine_rules},
        {"stream": lambda prev: translate_stream(prev)}
    ],
    pass_state=True
)
```

#### Implementation Tasks

- [ ] Create `PipeStep` dataclass
- [ ] Create `PipeResult` dataclass
- [ ] Implement `pipe()` function
- [ ] Support guardrails between stages
- [ ] Support state passing between stages
- [ ] Add tests for pipeline execution

---

## Phase 7: Observability Events

### 7.1 Complete Event Types

**Priority: Medium**

#### Required Event Types

Audit and implement all event types from TS:

**Session Events:**
- `SESSION_START`, `SESSION_END`, `SESSION_SUMMARY`

**Stream Events:**
- `STREAM_INIT`, `STREAM_READY`

**Adapter Events:**
- `ADAPTER_DETECTED`, `ADAPTER_WRAP_START`, `ADAPTER_WRAP_END`

**Timeout Events:**
- `TIMEOUT_START`, `TIMEOUT_RESET`, `TIMEOUT_TRIGGERED`

**Network Events:**
- `NETWORK_ERROR`, `NETWORK_RECOVERY`, `CONNECTION_DROPPED`, `CONNECTION_RESTORED`

**Abort Events:**
- `ABORT_REQUESTED`, `ABORT_COMPLETED`

**Tool Events:**
- `TOOL_REQUESTED`, `TOOL_START`, `TOOL_RESULT`, `TOOL_ERROR`, `TOOL_COMPLETED`

**Guardrail Events:**
- `GUARDRAIL_PHASE_START`, `GUARDRAIL_PHASE_END`
- `GUARDRAIL_RULE_START`, `GUARDRAIL_RULE_RESULT`, `GUARDRAIL_RULE_END`
- `GUARDRAIL_CALLBACK_START`, `GUARDRAIL_CALLBACK_END`

**Drift Events:**
- `DRIFT_CHECK_START`, `DRIFT_CHECK_RESULT`, `DRIFT_CHECK_END`, `DRIFT_CHECK_SKIPPED`

**Checkpoint Events:**
- `CHECKPOINT_SAVED`

**Resume Events:**
- `RESUME_START`, `RESUME_END`

**Retry Events:**
- `RETRY_START`, `RETRY_ATTEMPT`, `RETRY_END`, `RETRY_GIVE_UP`

**Fallback Events:**
- `FALLBACK_START`, `FALLBACK_MODEL_SELECTED`, `FALLBACK_END`

**Completion Events:**
- `FINALIZATION_START`, `FINALIZATION_END`

**Consensus Events:**
- `CONSENSUS_START`, `CONSENSUS_STREAM_START`, `CONSENSUS_STREAM_END`
- `CONSENSUS_OUTPUT_COLLECTED`, `CONSENSUS_ANALYSIS`, `CONSENSUS_RESOLUTION`, `CONSENSUS_END`

**Structured Output Events:**
- `PARSE_START`, `PARSE_END`, `PARSE_ERROR`
- `SCHEMA_VALIDATION_START`, `SCHEMA_VALIDATION_END`
- `AUTO_CORRECT_START`, `AUTO_CORRECT_END`

**Continuation Events:**
- `CONTINUATION_START`, `CONTINUATION_END`
- `DEDUPLICATION_START`, `DEDUPLICATION_END`

#### Implementation Tasks

- [ ] Audit existing `ObservabilityEventType` enum
- [ ] Add missing event types
- [ ] Ensure all events include `ts` (timestamp) and `stream_id`
- [ ] Emit events at appropriate points in runtime
- [ ] Add tests for event emission

---

## Phase 8: Monitoring Enhancements

### 8.1 Event Handler Utilities

**Priority: Medium**

#### Missing Features

```python
from l0.monitoring import combine_events, filter_events, exclude_events

# Combine multiple handlers
on_event = combine_events(otel_handler, sentry_handler, custom_handler)

# Filter to specific events
on_event = filter_events(
    [ObservabilityEventType.ERROR, ObservabilityEventType.RETRY_ATTEMPT],
    error_handler
)

# Exclude noisy events
on_event = exclude_events(
    [ObservabilityEventType.TOKEN],
    log_handler
)
```

#### Implementation Tasks

- [ ] Implement `combine_events()` utility
- [ ] Implement `filter_events()` utility
- [ ] Implement `exclude_events()` utility
- [ ] Add `debounce_events()` for rate limiting
- [ ] Add `batch_events()` for batching
- [ ] Add tests for all utilities

### 8.2 OpenTelemetry GenAI Semantic Conventions

**Priority: Low**

#### Implementation Tasks

- [ ] Add `gen_ai.*` attributes to OTel spans
- [ ] Follow [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [ ] Add `l0.*` custom attributes

---

## Phase 9: Drift Detection Enhancements

### 9.1 Complete Drift Detection

**Priority: Medium**

#### Missing Features

1. **Entropy spike detection**
2. **Markdown collapse detection**
3. **Confidence scores**

#### Implementation Tasks

- [ ] Implement entropy calculation
- [ ] Add entropy spike detection
- [ ] Add markdown collapse detection (format drift)
- [ ] Return confidence scores from drift checks
- [ ] Add tests for all drift types

---

## Phase 10: Adapters

### 10.1 Helper Functions

**Priority: Medium**

#### Missing Features

```python
from l0.adapters import to_l0_events, to_multimodal_l0_events

# Simple text extraction
async for event in to_l0_events(stream, lambda chunk: chunk.text):
    ...

# Multimodal extraction
async for event in to_multimodal_l0_events(stream, {
    "extract_text": lambda c: c.text,
    "extract_data": lambda c: c.image,
    "extract_progress": lambda c: c.progress
}):
    ...
```

#### Implementation Tasks

- [ ] Verify `to_l0_events()` exists
- [ ] Verify `to_multimodal_l0_events()` exists
- [ ] Add all `create_adapter_*_event()` helpers
- [ ] Add tests for helper functions

---

## Phase 11: Error Handling Enhancements

### 11.1 Error Events

**Priority: Medium**

#### Missing Features

Error events should include:
- `failure_type`: "network" | "model" | "timeout" | "abort" | "zero_output" | "tool" | "unknown"
- `recovery_strategy`: "retry" | "fallback" | "halt"
- `policy`: Recovery policy details

#### Implementation Tasks

- [ ] Add `failure_type` to error events
- [ ] Add `recovery_strategy` to error events
- [ ] Add `policy` object to error events
- [ ] Add tests for error event contents

### 11.2 Error Utilities

**Priority: Low**

#### Implementation Tasks

- [ ] Verify `is_l0_error()` function exists
- [ ] Verify `error.get_checkpoint()` method exists
- [ ] Verify `error.has_checkpoint` property exists
- [ ] Add `error.to_detailed_string()` method
- [ ] Add `error.to_json()` method

---

## Phase 12: State Enhancements

### 12.1 Complete State Properties

**Priority: Medium**

#### Required Properties

```python
@dataclass
class State:
    content: str = ""
    token_count: int = 0
    checkpoint: str | None = None
    duration: float | None = None
    completed: bool = False
    violations: list[GuardrailViolation] = field(default_factory=list)
    retry_attempts: int = 0
    network_retry_count: int = 0
    fallback_index: int = -1              # -1 if primary
    drift_detected: bool = False
    data_outputs: list[DataPayload] = field(default_factory=list)
    last_progress: Progress | None = None
    
    # Continuation properties
    resumed: bool = False                  # NEW
    resume_point: str | None = None        # NEW
    resume_from: int | None = None         # NEW
```

#### Implementation Tasks

- [ ] Audit existing `State` class
- [ ] Add missing properties
- [ ] Ensure all properties are populated during runtime
- [ ] Add tests for state tracking

---

## Phase 13: Documentation & Testing

### 13.1 Documentation

**Priority: Medium**

#### Implementation Tasks

- [ ] Update README.md with all new features
- [ ] Create API.md with complete API reference
- [ ] Create MIGRATION.md for TypeScript users
- [ ] Add docstrings to all public functions/classes
- [ ] Add type hints to all parameters

### 13.2 Test Coverage

**Priority: High**

#### Implementation Tasks

- [ ] Achieve feature parity in tests
- [ ] Add integration tests for all SDK adapters
- [ ] Add lifecycle tests matching TS (44+ tests)
- [ ] Add performance tests
- [ ] Add stress tests for edge cases

---

## Implementation Order

### Sprint 1: Core Foundations (Week 1-2)
1. Phase 1: Core Runtime Enhancements
2. Phase 2: Lifecycle Callbacks
3. Phase 13: State Enhancements

### Sprint 2: Safety & Reliability (Week 3-4)
4. Phase 3: Guardrails Enhancements
5. Phase 9: Drift Detection Enhancements
6. Phase 12: Error Handling Enhancements

### Sprint 3: Concurrency (Week 5)
7. Phase 4: Parallel Operations
8. Phase 6: Pipeline/Pipe Operations

### Sprint 4: Structured Data (Week 6)
9. Phase 5: Structured Output
10. Phase 11: Adapters

### Sprint 5: Observability (Week 7)
11. Phase 7: Observability Events
12. Phase 8: Monitoring Enhancements

### Sprint 6: Advanced Features (Week 8)
13. Phase 10: Interceptors
14. Phase 14: Documentation & Testing

---

## File Structure

```
l0-python/src/l0/
├── __init__.py              # Main exports
├── types.py                 # Core types (State, Event, Retry, Timeout, CheckIntervals)
├── runtime.py               # Main l0() function
├── client.py                # Client wrapper
├── errors.py                # Error types
├── retry.py                 # RetryManager
├── continuation.py          # Checkpoint resumption
├── state.py                 # State management
├── stream.py                # Stream utilities
├── 
├── guardrails.py            # Guardrail system
├── structured.py            # Structured output
├── consensus.py             # Multi-model consensus
├── parallel.py              # Parallel operations
├── pipe.py                  # Pipeline operations (NEW)
├── pool.py                  # Operation pool (NEW)
├── window.py                # Document windows
├── 
├── interceptors/            # Interceptor system (NEW)
│   ├── __init__.py
│   ├── types.py
│   ├── manager.py
│   └── builtin.py
├── 
├── adapters.py              # Adapter system
├── multimodal.py            # Multimodal support
├── events.py                # Observability events
├── 
├── formatting/              # Formatting utilities
├── monitoring/              # Monitoring & telemetry
├── eventsourcing/           # Event sourcing
└── 
└── _utils.py                # Internal utilities
```

---

## Success Criteria

1. **API Parity**: All TypeScript functions/options available in Python
2. **Type Safety**: Full type hints with Pydantic validation
3. **Test Coverage**: Match TS test count (~2600+ unit, ~250+ integration)
4. **Documentation**: Complete API docs with examples
5. **Performance**: Comparable to TS implementation
6. **Pythonic**: Follow Python idioms (snake_case, context managers, etc.)

---

## Notes

- Python uses `snake_case` for all names (vs `camelCase` in TS)
- Timeouts are in **seconds** (not milliseconds) to be Pythonic
- Use `asyncio` for all async operations
- Use Pydantic v2 for validation (vs Zod in TS)
- Use `dataclasses` with `@dataclass` decorator for simple types
- Use `typing.Protocol` for adapter interfaces
