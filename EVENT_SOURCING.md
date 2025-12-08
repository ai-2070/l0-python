# Event Sourcing

Record, replay, and audit L0 streams with event sourcing. Every token, retry, checkpoint, and error is captured as a replayable event.

## Quick Start

```python
from l0 import EventSourcing

# Create an event store
store = EventSourcing.memory()

# Record events
recorder = EventSourcing.recorder(store)
await recorder.record_start({"model": "gpt-4"})
await recorder.record_token("Hello", 0)
await recorder.record_token(" world", 1)
await recorder.record_complete("Hello world", 2)

# Replay the stream
result = await EventSourcing.replay(recorder.stream_id, store)
async for event in result:
    if event.event.type == "TOKEN":
        print(event.event.value, end="")
# Output: Hello world

# Get final state
state = result.state
print(f"\nTokens: {state.token_count}")  # Tokens: 2
```

## Event Types

L0 captures 10 event types covering the full lifecycle:

```python
from l0.eventsourcing import RecordedEventType

# Available event types
RecordedEventType.START        # Stream started with options
RecordedEventType.TOKEN        # Token emitted
RecordedEventType.CHECKPOINT   # Checkpoint saved
RecordedEventType.GUARDRAIL    # Guardrail check result
RecordedEventType.DRIFT        # Drift detection result
RecordedEventType.RETRY        # Retry attempted
RecordedEventType.FALLBACK     # Fallback model switch
RecordedEventType.CONTINUATION # Resumed from checkpoint
RecordedEventType.COMPLETE     # Stream completed
RecordedEventType.ERROR        # Stream errored
```

### Event Dataclasses

Each event type has a specific dataclass:

```python
from l0.eventsourcing import (
    StartEvent,       # type, ts, options
    TokenEvent,       # type, ts, value, index
    CheckpointEvent,  # type, ts, content, at
    GuardrailEvent,   # type, ts, result
    DriftEvent,       # type, ts, result
    RetryEvent,       # type, ts, attempt, reason, delay, counts_toward_limit
    FallbackEvent,    # type, ts, from_, to, reason
    ContinuationEvent,  # type, ts, checkpoint, at
    CompleteEvent,    # type, ts, content, token_count
    ErrorEvent,       # type, ts, error
)
```

### Event Envelope

Events are wrapped in envelopes with metadata:

```python
from l0.eventsourcing import EventEnvelope

# EventEnvelope fields
envelope.stream_id  # Unique stream identifier
envelope.seq        # Sequence number (0, 1, 2, ...)
envelope.event      # The actual event (TokenEvent, etc.)
```

## Event Stores

### In-Memory Store

Fast, ephemeral storage for testing and development:

```python
from l0 import EventSourcing

# Basic usage
store = EventSourcing.memory()

# With prefix and TTL
store = EventSourcing.memory(prefix="myapp", ttl=60000)  # 60 second TTL
```

### File Store

Persistent storage to disk:

```python
from l0 import EventSourcing

# Basic usage
store = EventSourcing.file("./events")

# With prefix and TTL
store = EventSourcing.file(
    base_path="./l0-events",
    prefix="prod",
    ttl=7 * 24 * 60 * 60 * 1000,  # 7 days
)
```

### Composite Store

Write to multiple backends simultaneously:

```python
from l0 import EventSourcing

# Write to both memory and file, read from memory (index 0)
memory_store = EventSourcing.memory()
file_store = EventSourcing.file("./events")

composite = EventSourcing.composite(
    stores=[memory_store, file_store],
    primary_index=0,  # Read from memory
)
```

### TTL Store

Wrap any store with TTL expiration:

```python
from l0 import EventSourcing

base_store = EventSourcing.memory()
ttl_store = EventSourcing.with_ttl(base_store, ttl_ms=3600000)  # 1 hour
```

### EventStore Protocol

Implement custom stores by following the protocol:

```python
from l0.eventsourcing import EventStore, EventEnvelope
from typing import Protocol

class EventStore(Protocol):
    """Event store protocol for custom implementations."""

    async def append(self, stream_id: str, event: EventEnvelope) -> None:
        """Append an event to a stream."""
        ...

    async def get_events(self, stream_id: str) -> list[EventEnvelope]:
        """Get all events for a stream."""
        ...

    async def exists(self, stream_id: str) -> bool:
        """Check if a stream exists."""
        ...

    async def delete(self, stream_id: str) -> bool:
        """Delete a stream and its events."""
        ...

    async def list_streams(self, prefix: str | None = None) -> list[str]:
        """List all stream IDs, optionally filtered by prefix."""
        ...
```

## Recording Events

### EventRecorder

The recorder provides methods for each event type:

```python
from l0 import EventSourcing

store = EventSourcing.memory()
recorder = EventSourcing.recorder(store)

# Or with a custom stream ID
recorder = EventSourcing.recorder(store, stream_id="my-stream-123")

# Access the stream ID
print(recorder.stream_id)
```

### Recording Methods

```python
# Start event (called first)
await recorder.record_start(options={"model": "gpt-4", "temperature": 0.7})

# Token events
await recorder.record_token("Hello", index=0)
await recorder.record_token(" world", index=1)

# Checkpoint (for resumption)
await recorder.record_checkpoint(content="Hello", at=1)

# Guardrail check
await recorder.record_guardrail(
    result={"passed": True, "violations": []},
    check_type="content",
)

# Drift detection
await recorder.record_drift(result={"detected": False, "type": None})

# Retry
await recorder.record_retry(
    attempt=1,
    reason="network_error",
    delay=1000,
    counts_toward_limit=False,  # Network retries don't count
)

# Fallback
await recorder.record_fallback(
    from_index=0,
    to_index=1,
    reason="max_retries_exceeded",
)

# Continuation (resumed from checkpoint)
await recorder.record_continuation(checkpoint="Hello", at=1)

# Completion
await recorder.record_complete(content="Hello world", token_count=2)

# Error
await recorder.record_error(
    error={"name": "TimeoutError", "message": "Stream timed out"},
)
```

## Replaying Events

### Basic Replay

```python
from l0 import EventSourcing

store = EventSourcing.memory()
# ... record some events ...

# Replay the stream
result = await EventSourcing.replay(stream_id, store)

async for envelope in result:
    event = envelope.event
    print(f"[{envelope.seq}] {event.type}: {event}")
```

### Replay to State

Get the final reconstructed state without iterating:

```python
from l0 import EventSourcing

replayer = EventSourcing.replayer(store)
state = await replayer.replay_to_state(stream_id)

print(state.content)        # Final content
print(state.token_count)    # Total tokens
print(state.checkpoint)     # Last checkpoint
print(state.completed)      # Whether stream completed
print(state.error)          # Error if any
print(state.violations)     # Guardrail violations
print(state.drift_detected) # Whether drift was detected
print(state.retry_attempts) # Retries that counted toward limit
print(state.network_retry_count)  # Network retries
print(state.fallback_index) # Current fallback index
print(state.start_ts)       # Start timestamp
print(state.end_ts)         # End timestamp
```

### Replay Tokens Only

Stream just the tokens:

```python
replayer = EventSourcing.replayer(store)

async for token in replayer.replay_tokens(stream_id, speed=0):
    print(token, end="")
```

### Replay with Timing

Replay at real-time speed or faster:

```python
# Instant replay (default)
result = await EventSourcing.replay(stream_id, store, speed=0)

# Real-time playback
result = await EventSourcing.replay(stream_id, store, speed=1)

# 2x speed
result = await EventSourcing.replay(stream_id, store, speed=2)
```

### Partial Replay

Replay a range of events:

```python
result = await EventSourcing.replay(
    stream_id,
    store,
    from_seq=5,    # Start from sequence 5
    to_seq=20,     # Stop at sequence 20
)
```

### Replay with Callbacks

Fire callbacks during replay for testing:

```python
result = await EventSourcing.replay(
    stream_id,
    store,
    fire_callbacks=True,
)

result.set_callbacks(
    on_token=lambda token: print(f"Token: {token}"),
    on_violation=lambda v: print(f"Violation: {v}"),
    on_retry=lambda attempt, reason: print(f"Retry {attempt}: {reason}"),
    on_event=lambda envelope: print(f"Event: {envelope.event.type}"),
)

async for envelope in result:
    pass  # Callbacks fire automatically

# Access final state
print(result.state)
```

## Stream Metadata

Get metadata about a recorded stream without replaying:

```python
from l0 import EventSourcing

meta = await EventSourcing.metadata(stream_id, store)

if meta:
    print(meta.stream_id)     # Stream identifier
    print(meta.event_count)   # Total events
    print(meta.token_count)   # Total tokens
    print(meta.start_ts)      # Start timestamp
    print(meta.end_ts)        # End timestamp
    print(meta.completed)     # Whether completed
    print(meta.has_error)     # Whether errored
    print(meta.options)       # Original options
```

## Comparing Replays

Compare two replay states to detect differences:

```python
from l0 import EventSourcing

# Replay two streams
replayer = EventSourcing.replayer(store)
state1 = await replayer.replay_to_state(stream_id_1)
state2 = await replayer.replay_to_state(stream_id_2)

# Compare
comparison = EventSourcing.compare(state1, state2)

print(comparison.identical)    # True if states match
print(comparison.differences)  # List of differences

# Example differences:
# ["content: 'Hello...' vs 'Hi...'", "token_count: 10 vs 12"]
```

## Storage Adapters

### Using Adapters

Create stores using registered adapters:

```python
from l0 import EventSourcing
from l0.eventsourcing import StorageAdapterConfig

# Memory adapter
store = await EventSourcing.create(StorageAdapterConfig(type="memory"))

# File adapter
store = await EventSourcing.create(StorageAdapterConfig(
    type="file",
    connection="./events",
    prefix="l0_events",
    ttl=7 * 24 * 60 * 60 * 1000,  # 7 days
))
```

### Built-in Adapters

L0 includes two adapters by default:

- `memory` - In-memory storage
- `file` - File-based persistence

### Custom Adapters

Register your own storage backends:

```python
from l0 import EventSourcing
from l0.eventsourcing import StorageAdapterConfig

# Synchronous factory
def create_redis_store(config: StorageAdapterConfig):
    return RedisEventStore(
        connection=config.connection,
        prefix=config.prefix,
        ttl=config.ttl,
        **config.options,
    )

EventSourcing.register_adapter("redis", create_redis_store)

# Async factory
async def create_postgres_store(config: StorageAdapterConfig):
    pool = await asyncpg.create_pool(config.connection)
    return PostgresEventStore(pool, config.prefix)

EventSourcing.register_adapter("postgres", create_postgres_store)

# Use custom adapter
store = await EventSourcing.create(StorageAdapterConfig(
    type="redis",
    connection="redis://localhost:6379",
    prefix="l0",
    options={"db": 0},
))
```

### Managing Adapters

```python
from l0 import EventSourcing

# List registered adapters
adapters = EventSourcing.list_adapters()
print(adapters)  # ['memory', 'file', 'redis', ...]

# Unregister an adapter
removed = EventSourcing.unregister_adapter("redis")
print(removed)  # True
```

## EventSourcing Scoped API

The `EventSourcing` class provides a unified, scoped API:

```python
from l0 import EventSourcing

# Store factories
EventSourcing.memory(prefix="l0", ttl=0)
EventSourcing.file(base_path="./events", prefix="l0", ttl=0)
EventSourcing.composite(stores=[...], primary_index=0)
EventSourcing.with_ttl(store, ttl_ms)
await EventSourcing.create(config)

# Recorder & Replayer
EventSourcing.recorder(store, stream_id=None)
EventSourcing.replayer(store)

# Replay functions
await EventSourcing.replay(stream_id, store, speed=0, fire_callbacks=False, from_seq=0, to_seq=None)
await EventSourcing.metadata(stream_id, store)
EventSourcing.compare(state1, state2)

# Utilities
EventSourcing.generate_id()

# Adapter registry
EventSourcing.register_adapter(adapter_type, factory)
EventSourcing.unregister_adapter(adapter_type)
EventSourcing.list_adapters()

# Type aliases
EventSourcing.Event       # RecordedEvent
EventSourcing.EventType   # RecordedEventType
EventSourcing.Envelope    # EventEnvelope
EventSourcing.State       # ReplayedState
EventSourcing.Snapshot    # Snapshot
EventSourcing.Metadata    # StreamMetadata
EventSourcing.Comparison  # ReplayComparison
EventSourcing.Config      # StorageAdapterConfig
```

## Use Cases

### Testing and Debugging

Record production streams and replay locally:

```python
# In production: record events
store = EventSourcing.file("./debug-events")
recorder = EventSourcing.recorder(store, stream_id=request_id)

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    on_event=lambda e: record_event_to_store(recorder, e),
)

# Later: replay for debugging
store = EventSourcing.file("./debug-events")
result = await EventSourcing.replay(request_id, store)

async for envelope in result:
    print(f"{envelope.seq}: {envelope.event}")
```

### Audit Trails

Capture complete audit history:

```python
# Create persistent store with 90-day retention
store = EventSourcing.file(
    base_path="./audit-logs",
    prefix="audit",
    ttl=90 * 24 * 60 * 60 * 1000,  # 90 days
)

recorder = EventSourcing.recorder(store)

# Record user context
await recorder.record_start({
    "model": "gpt-4",
    "user_id": user.id,
    "request_id": request.id,
    "timestamp": datetime.now().isoformat(),
})

# ... record stream events ...

# List all streams for a user
streams = await store.list_streams(prefix=f"audit-{user.id}")
```

### Regression Testing

Compare model outputs across versions:

```python
async def test_model_regression():
    store = EventSourcing.memory()

    # Run with model A
    recorder_a = EventSourcing.recorder(store, stream_id="model-a")
    await run_and_record(recorder_a, model="gpt-4-0613")

    # Run with model B
    recorder_b = EventSourcing.recorder(store, stream_id="model-b")
    await run_and_record(recorder_b, model="gpt-4-turbo")

    # Compare outputs
    replayer = EventSourcing.replayer(store)
    state_a = await replayer.replay_to_state("model-a")
    state_b = await replayer.replay_to_state("model-b")

    comparison = EventSourcing.compare(state_a, state_b)

    if not comparison.identical:
        print("Differences found:")
        for diff in comparison.differences:
            print(f"  - {diff}")
```

### Time-Travel Debugging

Replay to a specific point in the stream:

```python
# Get metadata to find total events
meta = await EventSourcing.metadata(stream_id, store)
print(f"Total events: {meta.event_count}")

# Replay first half only
result = await EventSourcing.replay(
    stream_id,
    store,
    to_seq=meta.event_count // 2,
)

async for envelope in result:
    print(f"{envelope.seq}: {envelope.event.type}")

print(f"State at midpoint: {result.state.content}")
```

## Best Practices

### Stream IDs

Use meaningful stream IDs for easier debugging:

```python
import uuid

# Good: includes context
stream_id = f"chat-{user_id}-{uuid.uuid4().hex[:8]}"
stream_id = f"completion-{request_id}"

# Also good: use L0's generator
stream_id = EventSourcing.generate_id()  # UUIDv7-based
```

### Store Selection

Choose the right store for your use case:

| Use Case | Recommended Store |
|----------|-------------------|
| Unit tests | `EventSourcing.memory()` |
| Integration tests | `EventSourcing.file("./test-events")` |
| Local development | `EventSourcing.file("./dev-events")` |
| Production audit | Custom adapter (Redis, PostgreSQL, etc.) |
| Multi-region | `EventSourcing.composite([local, remote])` |

### TTL Configuration

Set appropriate TTLs to manage storage:

```python
# Development: short TTL
dev_store = EventSourcing.memory(ttl=300000)  # 5 minutes

# Production: longer TTL
prod_store = EventSourcing.file(
    base_path="./events",
    ttl=7 * 24 * 60 * 60 * 1000,  # 7 days
)

# Audit: extended retention
audit_store = EventSourcing.file(
    base_path="./audit",
    ttl=365 * 24 * 60 * 60 * 1000,  # 1 year
)
```

### Error Handling

Always handle replay errors gracefully:

```python
meta = await EventSourcing.metadata(stream_id, store)

if meta is None:
    print(f"Stream {stream_id} not found")
    return

if meta.has_error:
    print(f"Stream {stream_id} ended with error")
    state = await EventSourcing.replayer(store).replay_to_state(stream_id)
    print(f"Error: {state.error}")
    return

# Safe to replay
result = await EventSourcing.replay(stream_id, store)
```

## Snapshots

For long streams, snapshots provide efficient state recovery:

```python
from l0.eventsourcing import Snapshot

# Snapshot structure
snapshot = Snapshot(
    stream_id="my-stream",
    seq=100,                    # Snapshot at sequence 100
    content="...",              # Content up to this point
    token_count=100,
    checkpoint="...",
    violations=[],
    drift_detected=False,
    retry_attempts=0,
    network_retry_count=0,
    fallback_index=0,
    ts=1234567890.0,
)

# Stores implementing EventStoreWithSnapshots support:
# - save_snapshot(snapshot)
# - get_snapshot(stream_id)
# - Replay from snapshot instead of beginning
```

## Type Aliases

The `EventSourcing` class provides convenient type aliases:

```python
from l0 import EventSourcing

# Use type aliases for cleaner code
event: EventSourcing.Event = ...
event_type: EventSourcing.EventType = ...
envelope: EventSourcing.Envelope = ...
state: EventSourcing.State = ...
snapshot: EventSourcing.Snapshot = ...
metadata: EventSourcing.Metadata = ...
comparison: EventSourcing.Comparison = ...
config: EventSourcing.Config = ...
```
