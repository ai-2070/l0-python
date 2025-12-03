"""Parallel execution utilities for L0."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar, cast

T = TypeVar("T")


async def parallel(
    tasks: list[Callable[[], Awaitable[T]]],
    concurrency: int = 5,
) -> list[T]:
    """Run tasks with concurrency limit.

    Args:
        tasks: List of async callables to execute
        concurrency: Maximum number of concurrent tasks

    Returns:
        List of results in the same order as input tasks
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(task: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await task()

    results = await asyncio.gather(*[limited(t) for t in tasks])
    return list(results)


async def race(tasks: list[Callable[[], Awaitable[T]]]) -> T:
    """Return first successful result, cancel remaining tasks.

    Args:
        tasks: List of async callables to race

    Returns:
        Result from the first task to complete successfully

    Raises:
        RuntimeError: If no tasks provided
        Exception: If all tasks fail, raises the last exception
    """
    if not tasks:
        raise RuntimeError("No tasks provided")

    # Create tasks - cast Awaitable to Coroutine for create_task
    pending_tasks: list[asyncio.Task[T]] = [
        asyncio.create_task(cast(Coroutine[Any, Any, T], t())) for t in tasks
    ]

    try:
        done, pending_set = await asyncio.wait(
            pending_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending_set:
            task.cancel()

        # Return the first completed result
        return done.pop().result()
    except Exception:
        # Cancel all on error
        for task in pending_tasks:
            task.cancel()
        raise


async def batched(
    items: list[T],
    handler: Callable[[T], Awaitable[T]],
    batch_size: int = 10,
) -> list[T]:
    """Process items in batches.

    Args:
        items: List of items to process
        handler: Async function to apply to each item
        batch_size: Number of items to process concurrently

    Returns:
        List of processed results in the same order as input
    """
    results: list[T] = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(*[handler(item) for item in batch])
        results.extend(list(batch_results))
    return results
