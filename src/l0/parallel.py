"""Parallel execution utilities for L0."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AggregatedTelemetry:
    """Aggregated telemetry from parallel operations.

    Attributes:
        total_tokens: Total tokens used across all operations
        total_duration: Total duration in seconds
        total_retries: Total retry attempts
        total_network_errors: Total network errors
        total_violations: Total guardrail violations
        avg_tokens_per_second: Average tokens per second
        avg_time_to_first_token: Average time to first token in seconds
    """

    total_tokens: int = 0
    total_duration: float = 0.0
    total_retries: int = 0
    total_network_errors: int = 0
    total_violations: int = 0
    avg_tokens_per_second: float = 0.0
    avg_time_to_first_token: float = 0.0


@dataclass
class RaceResult(Generic[T]):
    """Result from race operation.

    Attributes:
        value: The winning result value
        winner_index: Index of the winning operation (0-based)
    """

    value: T
    winner_index: int


@dataclass
class ParallelResult(Generic[T]):
    """Result of parallel execution.

    Attributes:
        results: List of successful results (None for failed tasks)
        errors: List of errors (None for successful tasks)
        success_count: Number of successful tasks
        failure_count: Number of failed tasks
        duration: Total execution time in seconds
        all_succeeded: Whether all tasks succeeded
    """

    results: list[T | None]
    errors: list[Exception | None]
    success_count: int = 0
    failure_count: int = 0
    duration: float = 0.0
    aggregated_telemetry: AggregatedTelemetry | None = None

    @property
    def all_succeeded(self) -> bool:
        """Check if all tasks succeeded."""
        return self.failure_count == 0

    def successful_results(self) -> list[T]:
        """Get only successful results (where no error occurred)."""
        return [cast(T, r) for r, e in zip(self.results, self.errors) if e is None]


@dataclass
class ParallelOptions:
    """Options for parallel execution.

    Attributes:
        concurrency: Maximum concurrent tasks (default: 5)
        fail_fast: Stop on first error (default: False)
        on_progress: Callback for progress updates (completed, total)
        on_complete: Callback when a task completes (result, index)
        on_error: Callback when a task fails (error, index)
    """

    concurrency: int = 5
    fail_fast: bool = False
    on_progress: Callable[[int, int], None] | None = None
    on_complete: Callable[[Any, int], None] | None = None
    on_error: Callable[[Exception, int], None] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────────────────


async def parallel(
    tasks: list[Callable[[], Awaitable[T]]],
    options: ParallelOptions | None = None,
    *,
    concurrency: int | None = None,
    fail_fast: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    on_complete: Callable[[T, int], None] | None = None,
    on_error: Callable[[Exception, int], None] | None = None,
) -> ParallelResult[T]:
    """Run tasks with concurrency limit.

    Args:
        tasks: List of async callables to execute
        options: ParallelOptions instance (alternative to kwargs)
        concurrency: Maximum concurrent tasks (default: 5)
        fail_fast: Stop on first error (default: False)
        on_progress: Callback for progress updates
        on_complete: Callback when a task completes
        on_error: Callback when a task fails

    Returns:
        ParallelResult with results, errors, and statistics

    Example:
        ```python
        async def fetch(url):
            ...

        result = await parallel(
            [lambda u=url: fetch(u) for url in urls],
            concurrency=3,
            fail_fast=False,
            on_progress=lambda done, total: print(f"{done}/{total}"),
        )

        print(f"Success: {result.success_count}/{len(urls)}")
        for r in result.successful_results():
            print(r)
        ```
    """
    # Use options object or kwargs
    if options:
        concurrency = options.concurrency
        fail_fast = options.fail_fast
        on_progress = options.on_progress
        on_complete = options.on_complete
        on_error = options.on_error
    else:
        concurrency = concurrency or 5

    if not tasks:
        return ParallelResult[T](
            results=[], errors=[], success_count=0, failure_count=0
        )

    start_time = time.time()
    semaphore = asyncio.Semaphore(concurrency)
    results: list[T | None] = [None] * len(tasks)
    errors: list[Exception | None] = [None] * len(tasks)
    completed = 0
    success_count = 0
    failure_count = 0
    cancel_event = asyncio.Event() if fail_fast else None

    async def run_task(task: Callable[[], Awaitable[T]], index: int) -> None:
        nonlocal completed, success_count, failure_count

        if cancel_event and cancel_event.is_set():
            return

        async with semaphore:
            if cancel_event and cancel_event.is_set():
                return

            try:
                result = await task()
                results[index] = result
                success_count += 1
                if on_complete:
                    on_complete(result, index)
            except Exception as e:
                errors[index] = e
                failure_count += 1
                if on_error:
                    on_error(e, index)
                if cancel_event:
                    cancel_event.set()
            finally:
                completed += 1
                if on_progress:
                    on_progress(completed, len(tasks))

    # Run all tasks
    await asyncio.gather(
        *[run_task(t, i) for i, t in enumerate(tasks)], return_exceptions=True
    )

    duration = time.time() - start_time

    return ParallelResult[T](
        results=results,
        errors=errors,
        success_count=success_count,
        failure_count=failure_count,
        duration=duration,
    )


async def race(
    tasks: list[Callable[[], Awaitable[T]]],
    *,
    on_error: Callable[[Exception, int], None] | None = None,
) -> RaceResult[T]:
    """Return first successful result, cancel remaining tasks.

    Args:
        tasks: List of async callables to race
        on_error: Callback when a task fails

    Returns:
        RaceResult containing the value and winner_index (0-based)

    Raises:
        RuntimeError: If no tasks provided
        Exception: If all tasks fail, raises the last exception

    Example:
        ```python
        # Race multiple providers
        result = await race([
            lambda: call_openai(prompt),
            lambda: call_anthropic(prompt),
            lambda: call_google(prompt),
        ])
        print(f"Winner: provider {result.winner_index}")
        print(f"Response: {result.value}")
        ```
    """
    if not tasks:
        raise RuntimeError("No tasks provided")

    pending_tasks: list[asyncio.Task[T]] = [
        asyncio.create_task(cast(Coroutine[Any, Any, T], t())) for t in tasks
    ]
    # Map task to original index for error reporting
    task_to_index: dict[asyncio.Task[T], int] = {
        task: i for i, task in enumerate(pending_tasks)
    }
    last_error: Exception | None = None
    winner_idx: int = -1

    try:
        while pending_tasks:
            done, pending_set = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            pending_tasks = list(pending_set)

            # Check all completed tasks for a success
            success_result: T | None = None
            found_success = False

            for task in done:
                try:
                    result = task.result()
                    # Found a successful result
                    if not found_success:
                        success_result = result
                        winner_idx = task_to_index.get(task, -1)
                        found_success = True
                except Exception as e:
                    last_error = e
                    if on_error:
                        task_index = task_to_index.get(task, -1)
                        on_error(e, task_index)

            # Only cancel and return after checking all done tasks
            if found_success:
                for p in pending_tasks:
                    p.cancel()
                return RaceResult(
                    value=cast(T, success_result), winner_index=winner_idx
                )

        # All tasks failed
        if last_error:
            raise last_error
        raise RuntimeError("All tasks failed")

    except Exception:
        # Cancel all on error
        for task in pending_tasks:
            task.cancel()
        raise


async def sequential(tasks: list[Callable[[], Awaitable[T]]]) -> list[T]:
    """Run tasks one at a time, in order.

    Args:
        tasks: List of async callables to execute

    Returns:
        List of results in the same order as input

    Example:
        ```python
        results = await sequential([
            lambda: process(item1),
            lambda: process(item2),
            lambda: process(item3),
        ])
        ```
    """
    results: list[T] = []
    for task in tasks:
        result = await task()
        results.append(result)
    return results


async def batched(
    items: list[T],
    handler: Callable[[T], Awaitable[T]],
    batch_size: int = 10,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[T]:
    """Process items in batches.

    Args:
        items: List of items to process
        handler: Async function to apply to each item
        batch_size: Number of items to process concurrently
        on_progress: Callback for progress updates

    Returns:
        List of processed results in the same order as input

    Example:
        ```python
        async def process(url: str) -> dict:
            ...

        results = await batched(
            urls,
            process,
            batch_size=5,
            on_progress=lambda done, total: print(f"{done}/{total}"),
        )
        ```
    """
    results: list[T] = []
    total = len(items)
    completed = 0

    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(*[handler(item) for item in batch])
        results.extend(list(batch_results))
        completed += len(batch)
        if on_progress:
            on_progress(completed, total)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class Parallel:
    """Scoped API for parallel execution utilities.

    Provides utilities for running async tasks with concurrency control,
    racing tasks, sequential execution, and batch processing.

    Usage:
        ```python
        from l0 import Parallel

        # Run tasks with concurrency limit
        result = await Parallel.run(
            [lambda: fetch(url) for url in urls],
            concurrency=3,
        )

        # Race multiple providers
        winner = await Parallel.race([
            lambda: call_openai(prompt),
            lambda: call_anthropic(prompt),
        ])

        # Process items in batches
        results = await Parallel.batched(items, handler, batch_size=10)

        # Run tasks sequentially
        results = await Parallel.sequential(tasks)
        ```
    """

    # Re-export types for convenience
    Result = ParallelResult
    RaceResult = RaceResult
    Options = ParallelOptions
    Telemetry = AggregatedTelemetry

    @staticmethod
    async def run(
        tasks: list[Callable[[], Awaitable[T]]],
        options: ParallelOptions | None = None,
        *,
        concurrency: int | None = None,
        fail_fast: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        on_complete: Callable[[T, int], None] | None = None,
        on_error: Callable[[Exception, int], None] | None = None,
    ) -> ParallelResult[T]:
        """Run tasks with concurrency limit.

        Args:
            tasks: List of async callables to execute
            options: ParallelOptions instance (alternative to kwargs)
            concurrency: Maximum concurrent tasks (default: 5)
            fail_fast: Stop on first error (default: False)
            on_progress: Callback for progress updates
            on_complete: Callback when a task completes
            on_error: Callback when a task fails

        Returns:
            ParallelResult with results, errors, and statistics
        """
        return await parallel(
            tasks,
            options,
            concurrency=concurrency,
            fail_fast=fail_fast,
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
        )

    @staticmethod
    async def race(
        tasks: list[Callable[[], Awaitable[T]]],
        *,
        on_error: Callable[[Exception, int], None] | None = None,
    ) -> RaceResult[T]:
        """Return first successful result, cancel remaining tasks.

        Args:
            tasks: List of async callables to race
            on_error: Callback when a task fails

        Returns:
            RaceResult containing the value and winner_index (0-based)
        """
        return await race(tasks, on_error=on_error)

    @staticmethod
    async def sequential(tasks: list[Callable[[], Awaitable[T]]]) -> list[T]:
        """Run tasks one at a time, in order.

        Args:
            tasks: List of async callables to execute

        Returns:
            List of results in the same order as input
        """
        return await sequential(tasks)

    @staticmethod
    async def batched(
        items: list[T],
        handler: Callable[[T], Awaitable[T]],
        batch_size: int = 10,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[T]:
        """Process items in batches.

        Args:
            items: List of items to process
            handler: Async function to apply to each item
            batch_size: Number of items to process concurrently
            on_progress: Callback for progress updates

        Returns:
            List of processed results in the same order as input
        """
        return await batched(items, handler, batch_size, on_progress=on_progress)
