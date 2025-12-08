# Parallel Operations

L0 provides utilities for running async tasks with concurrency control, racing tasks, sequential execution, batch processing, and operation pools.

## Basic Usage

```python
import l0
from l0 import Parallel

results = await Parallel.run(
    [
        lambda: l0.run(stream=lambda: client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Translate to Spanish: Hello"}], stream=True
        )),
        lambda: l0.run(stream=lambda: client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Translate to French: Hello"}], stream=True
        )),
        lambda: l0.run(stream=lambda: client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Translate to German: Hello"}], stream=True
        )),
    ],
    concurrency=2,
    fail_fast=False,
)

print(f"Success: {results.success_count}")
print(f"Spanish: {results.results[0].state.content if results.results[0] else 'Failed'}")
```

Or use the standalone functions:

```python
from l0.parallel import parallel, race, sequential, batched

results = await parallel(tasks, concurrency=3)
```

---

## Parallel Execution

### parallel()

Run tasks with a concurrency limit:

```python
from l0 import Parallel

result = await Parallel.run(
    tasks=[
        lambda: fetch(url1),
        lambda: fetch(url2),
        lambda: fetch(url3),
    ],
    concurrency=2,           # Max concurrent tasks (default: 5)
    fail_fast=False,         # Stop on first error (default: False)
    on_progress=lambda done, total: print(f"{done}/{total}"),
    on_complete=lambda result, index: print(f"Task {index} done"),
    on_error=lambda error, index: print(f"Task {index} failed: {error}"),
)

print(f"Success: {result.success_count}/{len(result.results)}")
print(f"Failed: {result.failure_count}")
print(f"Duration: {result.duration:.2f}s")
print(f"All succeeded: {result.all_succeeded}")

# Get only successful results
for r in result.successful_results():
    print(r)
```

### ParallelOptions

```python
from l0.parallel import ParallelOptions

options = ParallelOptions(
    concurrency=5,           # Maximum concurrent tasks (default: 5)
    fail_fast=False,         # Stop on first error (default: False)
    on_progress=None,        # Callback for progress updates (completed, total)
    on_complete=None,        # Callback when a task completes (result, index)
    on_error=None,           # Callback when a task fails (error, index)
)

result = await Parallel.run(tasks, options)
```

### ParallelResult

```python
from l0.parallel import ParallelResult

# ParallelResult fields
result.results          # list[T | None] - Results (None for failed tasks)
result.errors           # list[Exception | None] - Errors (None for successful tasks)
result.success_count    # Number of successful tasks
result.failure_count    # Number of failed tasks
result.duration         # Total execution time in seconds
result.all_succeeded    # Whether all tasks succeeded (property)

# Get only successful results
successful = result.successful_results()  # list[T]
```

---

## Race - First Wins

Return the first successful result and cancel remaining tasks:

```python
from l0 import Parallel

result = await Parallel.race([
    lambda: call_openai(prompt),
    lambda: call_anthropic(prompt),
    lambda: call_google(prompt),
])

print(f"Winner index: {result.winner_index}")  # 0-based
print(f"Response: {result.value}")
```

### Race with Error Handling

```python
result = await Parallel.race(
    tasks=[
        lambda: call_openai(prompt),
        lambda: call_anthropic(prompt),
    ],
    on_error=lambda error, index: print(f"Provider {index} failed: {error}"),
)
```

### RaceResult

```python
from l0.parallel import RaceResult

# RaceResult fields
result.value         # The winning result value
result.winner_index  # Index of the winning operation (0-based)
```

### Race Behavior

- Uses `asyncio.wait(return_when=FIRST_COMPLETED)` internally
- Cancels remaining tasks when first success is found
- If all tasks fail, raises the last exception
- Raises `RuntimeError` if no tasks provided

---

## Sequential Execution

Run tasks one at a time, in order:

```python
from l0 import Parallel

results = await Parallel.sequential([
    lambda: process(item1),
    lambda: process(item2),
    lambda: process(item3),
])

# results is list[T] in the same order as input
```

This is equivalent to `parallel(tasks, concurrency=1)` but simpler.

---

## Batch Processing

Process items in batches with a handler function:

```python
from l0 import Parallel

async def process_url(url: str) -> dict:
    # Process a single URL
    ...

results = await Parallel.batched(
    items=urls,              # List of items to process
    handler=process_url,     # Async function to apply to each item
    batch_size=5,            # Items per batch (default: 10)
    on_progress=lambda done, total: print(f"{done}/{total}"),
)

# results is list[T] in the same order as items
```

### Batched vs Parallel

| Aspect | `batched()` | `parallel()` |
|--------|-------------|--------------|
| Input | Items + handler function | List of task factories |
| Execution | Waits for each batch to complete before next | Rolling window of concurrent tasks |
| Use case | Process a list of items | Run independent async tasks |

---

## Operation Pool

For dynamic workload management, use `OperationPool`:

```python
import l0

# Create a pool with 3 concurrent workers
pool = l0.create_pool(3)

# Submit operations dynamically
result1 = pool.execute(stream=lambda: client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Task 1"}], stream=True
))
result2 = pool.execute(stream=lambda: client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Task 2"}], stream=True
))
result3 = pool.execute(stream=lambda: client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Task 3"}], stream=True
))

# Wait for all operations to complete
await pool.drain()

# Get results (returns State with accumulated content)
state1 = await result1
state2 = await result2
state3 = await result3

print(state1.content)
print(state2.content)
```

### Pool with Shared Configuration

```python
import l0

pool = l0.create_pool(
    worker_count=3,
    shared_retry=l0.Retry.recommended(),
    shared_timeout=l0.Timeout(initial_token=10.0, inter_token=30.0),
    shared_guardrails=l0.Guardrails.recommended(),
    on_event=lambda event: print(f"Event: {event.type}"),
    context={"user_id": "user_123"},
)

# Operations inherit shared config (can be overridden per-operation)
result = pool.execute(
    stream=lambda: client.chat.completions.create(...),
    fallbacks=[lambda: fallback_client.chat.completions.create(...)],
    guardrails=l0.Guardrails.strict(),  # Override shared guardrails
)
```

### Pool Methods

| Method | Description |
|--------|-------------|
| `execute(stream, ...)` | Submit operation, returns `Future[State]` |
| `drain()` | Wait for all queued operations to complete |
| `shutdown()` | Drain and stop all workers |
| `get_queue_length()` | Number of operations waiting |
| `get_active_workers()` | Number of workers currently executing |
| `get_stats()` | Get pool statistics |
| `worker_count` | Property: configured number of workers |

### Pool Statistics

```python
stats = pool.get_stats()

print(stats.total_executed)  # Total operations executed
print(stats.total_succeeded) # Operations that completed successfully
print(stats.total_failed)    # Operations that failed
print(stats.total_duration)  # Total execution time in seconds
```

### PoolOptions

```python
from l0.pool import PoolOptions

options = PoolOptions(
    shared_retry=None,       # Retry config applied to all operations
    shared_timeout=None,     # Timeout config applied to all operations
    shared_guardrails=None,  # Guardrails applied to all operations
    on_event=None,           # Callback for observability events
    context=None,            # User context attached to all events
)
```

---

## Fall-Through vs Race

### Fall-Through (Sequential Fallback)

Try models one at a time, moving to next only if current exhausts retries:

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(
        model="gpt-4o", messages=messages, stream=True
    ),
    fallbacks=[
        lambda: client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, stream=True
        ),
        lambda: anthropic_client.messages.create(
            model="claude-3-haiku-20240307", messages=messages, stream=True
        ),
    ],
    retry=l0.Retry.recommended(),
)
# 1. GPT-4o (retries) → 2. GPT-4o-mini (retries) → 3. Claude Haiku (retries)
```

**Use when:** Cost matters, latency acceptable, high availability required.

### Race (Parallel)

Call all models simultaneously, use fastest response:

```python
from l0 import Parallel

result = await Parallel.race([
    lambda: l0.run(stream=lambda: openai_stream),
    lambda: l0.run(stream=lambda: anthropic_stream),
    lambda: l0.run(stream=lambda: google_stream),
])
# All called at once, first to complete wins
# Others are automatically cancelled
```

**Use when:** Latency critical, cost not a constraint.

### Comparison

| Aspect | Fall-Through | Race |
|--------|--------------|------|
| Execution | Sequential | Parallel |
| Latency | Higher | Lower |
| Cost | Low | High (pay for all) |
| Best For | High availability | Low latency |

---

## AggregatedTelemetry

When using parallel operations with L0 streams, you can aggregate telemetry:

```python
from l0.parallel import AggregatedTelemetry

# AggregatedTelemetry fields
telemetry.total_tokens           # Total tokens used across all operations
telemetry.total_duration         # Total duration in seconds
telemetry.total_retries          # Total retry attempts
telemetry.total_network_errors   # Total network errors
telemetry.total_violations       # Total guardrail violations
telemetry.avg_tokens_per_second  # Average tokens per second
telemetry.avg_time_to_first_token  # Average time to first token in seconds
```

---

## Type Aliases

The `Parallel` class provides convenient type aliases:

```python
from l0 import Parallel

# Type aliases
result: Parallel.Result = ...       # ParallelResult
race_result: Parallel.RaceResult = ...  # RaceResult
options: Parallel.Options = ...     # ParallelOptions
telemetry: Parallel.Telemetry = ... # AggregatedTelemetry
```

---

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `parallel(tasks, options, ...)` | Run tasks with concurrency limit |
| `race(tasks, on_error)` | First successful result wins |
| `sequential(tasks)` | Run tasks one at a time |
| `batched(items, handler, batch_size, ...)` | Process items in batches |
| `create_pool(worker_count, ...)` | Create operation pool |

### Parallel Class

| Method | Description |
|--------|-------------|
| `Parallel.run(tasks, ...)` | Run tasks with concurrency limit |
| `Parallel.race(tasks, ...)` | First successful result wins |
| `Parallel.sequential(tasks)` | Run tasks one at a time |
| `Parallel.batched(items, handler, ...)` | Process items in batches |

### OperationPool

| Method | Description |
|--------|-------------|
| `pool.execute(stream, ...)` | Submit operation to pool |
| `pool.drain()` | Wait for all operations |
| `pool.shutdown()` | Drain and stop workers |
| `pool.get_queue_length()` | Get queue size |
| `pool.get_active_workers()` | Get active worker count |
| `pool.get_stats()` | Get pool statistics |

### Result Types

| Type | Description |
|------|-------------|
| `ParallelResult` | Result from parallel execution |
| `RaceResult` | Result from race operation |
| `ParallelOptions` | Options for parallel execution |
| `AggregatedTelemetry` | Aggregated telemetry from operations |
| `PoolOptions` | Options for operation pool |
| `PoolStats` | Statistics from operation pool |
