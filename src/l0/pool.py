"""Operation pool for dynamic workload management with shared concurrency."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .logging import logger
from .types import AwaitableStreamFactory, Retry, State, Timeout

if TYPE_CHECKING:
    from .events import ObservabilityEvent
    from .guardrails import GuardrailRule

T = TypeVar("T")


@dataclass
class PoolOptions:
    """Configuration options for OperationPool.

    Attributes:
        shared_retry: Retry configuration applied to all operations
        shared_timeout: Timeout configuration applied to all operations
        shared_guardrails: Guardrails applied to all operations
        on_event: Callback for observability events
        context: User context attached to all events
    """

    shared_retry: Retry | None = None
    shared_timeout: Timeout | None = None
    shared_guardrails: list["GuardrailRule"] | None = None
    on_event: Callable[["ObservabilityEvent"], None] | None = None
    context: dict[str, Any] | None = None


@dataclass
class PoolStats:
    """Statistics for the operation pool.

    Attributes:
        total_executed: Total operations executed
        total_succeeded: Operations that completed successfully
        total_failed: Operations that failed
        total_duration: Total execution time in seconds
    """

    total_executed: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_duration: float = 0.0


@dataclass
class PooledOperation(Generic[T]):
    """An operation queued in the pool."""

    stream: AwaitableStreamFactory
    fallbacks: list[AwaitableStreamFactory] | None = None
    guardrails: list["GuardrailRule"] | None = None
    retry: Retry | None = None
    timeout: Timeout | None = None
    future: asyncio.Future[State] = field(default_factory=asyncio.Future)


class OperationPool(Generic[T]):
    """Pool for executing streaming operations with shared concurrency.

    Provides dynamic workload management where operations are queued and
    executed with a configurable number of concurrent workers.

    Usage:
        ```python
        import l0

        # Create a pool with 3 concurrent workers
        pool = l0.create_pool(3)

        # Submit operations dynamically
        result1 = pool.execute(stream=lambda: stream1)
        result2 = pool.execute(stream=lambda: stream2)
        result3 = pool.execute(stream=lambda: stream3)

        # Wait for all operations
        await pool.drain()

        # Get results (returns State with accumulated content)
        state1 = await result1
        state2 = await result2
        print(state1.content)
        print(state2.content)

        # Check stats
        print(f"Queue length: {pool.get_queue_length()}")
        print(f"Active workers: {pool.get_active_workers()}")
        ```

    Args:
        worker_count: Maximum number of concurrent operations
        options: Shared configuration for all operations
    """

    def __init__(
        self,
        worker_count: int = 3,
        options: PoolOptions | None = None,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")

        self._worker_count = worker_count
        self._options = options or PoolOptions()
        self._queue: asyncio.Queue[PooledOperation[T] | None] = asyncio.Queue()
        self._active_workers = 0
        self._workers: list[asyncio.Task[None]] = []
        self._started = False
        self._draining = False
        self._stats = PoolStats()
        self._lock = asyncio.Lock()

    def _ensure_started(self) -> None:
        """Start worker tasks if not already started."""
        if self._started:
            return

        self._started = True
        for i in range(self._worker_count):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
        logger.debug(f"Pool started with {self._worker_count} workers")

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop that processes operations from the queue."""
        logger.debug(f"Worker {worker_id} started")

        while True:
            try:
                operation = await self._queue.get()

                # None is the shutdown signal
                if operation is None:
                    self._queue.task_done()
                    break

                async with self._lock:
                    self._active_workers += 1

                try:
                    import time

                    start_time = time.time()

                    # Import here to avoid circular imports
                    from .runtime import _internal_run

                    # Merge shared options with operation-specific options
                    guardrails = (
                        operation.guardrails or self._options.shared_guardrails or []
                    )
                    retry = operation.retry or self._options.shared_retry
                    timeout = operation.timeout or self._options.shared_timeout

                    result = await _internal_run(
                        stream=operation.stream,
                        fallbacks=operation.fallbacks,
                        guardrails=guardrails,
                        retry=retry,
                        timeout=timeout,
                        on_event=self._options.on_event,
                        context=self._options.context,
                    )

                    # Consume the stream to completion and collect state
                    async for _ in result:
                        pass

                    duration = time.time() - start_time

                    async with self._lock:
                        self._stats.total_executed += 1
                        self._stats.total_succeeded += 1
                        self._stats.total_duration += duration

                    # Return the accumulated State, not the exhausted stream
                    operation.future.set_result(result.state)

                except Exception as e:
                    async with self._lock:
                        self._stats.total_executed += 1
                        self._stats.total_failed += 1

                    operation.future.set_exception(e)

                finally:
                    async with self._lock:
                        self._active_workers -= 1
                    self._queue.task_done()

            except asyncio.CancelledError:
                break

        logger.debug(f"Worker {worker_id} stopped")

    def execute(
        self,
        stream: AwaitableStreamFactory,
        *,
        fallbacks: list[AwaitableStreamFactory] | None = None,
        guardrails: list["GuardrailRule"] | None = None,
        retry: Retry | None = None,
        timeout: Timeout | None = None,
    ) -> asyncio.Future[State]:
        """Submit an operation to the pool for execution.

        Returns immediately with a Future that resolves to the State result
        when the operation completes. The stream is fully consumed internally
        and the accumulated state is returned.

        Args:
            stream: Factory function that returns an async LLM stream
            fallbacks: Optional list of fallback stream factories
            guardrails: Optional guardrail rules (overrides shared)
            retry: Optional retry config (overrides shared)
            timeout: Optional timeout config (overrides shared)

        Returns:
            Future that resolves to State with accumulated content

        Example:
            ```python
            pool = l0.create_pool(3)

            # Submit and get future immediately
            future = pool.execute(stream=lambda: my_stream)

            # Do other work...

            # Wait for result when needed
            state = await future
            print(state.content)
            ```
        """
        self._ensure_started()

        if self._draining:
            raise RuntimeError("Cannot execute operations while pool is draining")

        operation: PooledOperation[T] = PooledOperation(
            stream=stream,
            fallbacks=fallbacks,
            guardrails=guardrails,
            retry=retry,
            timeout=timeout,
        )

        self._queue.put_nowait(operation)
        logger.debug(f"Operation queued, queue size: {self._queue.qsize()}")

        return operation.future

    async def drain(self) -> None:
        """Wait for all queued operations to complete.

        After drain completes, the pool can still accept new operations.

        Example:
            ```python
            pool = l0.create_pool(3)

            pool.execute(stream=lambda: stream1)
            pool.execute(stream=lambda: stream2)

            # Wait for all to complete
            await pool.drain()

            print("All operations complete!")
            ```
        """
        self._draining = True
        logger.debug("Draining pool...")

        # Wait for queue to be empty
        await self._queue.join()

        self._draining = False
        logger.debug("Pool drained")

    async def shutdown(self) -> None:
        """Shutdown the pool and all workers.

        Waits for all pending operations to complete, then stops workers.
        The pool cannot be used after shutdown.

        Example:
            ```python
            pool = l0.create_pool(3)

            # ... use pool ...

            # Clean shutdown
            await pool.shutdown()
            ```
        """
        if not self._started:
            return

        logger.debug("Shutting down pool...")

        # Wait for pending work
        await self.drain()

        # Send shutdown signal to all workers
        for _ in range(self._worker_count):
            await self._queue.put(None)

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        self._started = False
        logger.debug("Pool shutdown complete")

    def get_queue_length(self) -> int:
        """Get the number of operations waiting in the queue."""
        return self._queue.qsize()

    def get_active_workers(self) -> int:
        """Get the number of workers currently executing operations."""
        return self._active_workers

    def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        return PoolStats(
            total_executed=self._stats.total_executed,
            total_succeeded=self._stats.total_succeeded,
            total_failed=self._stats.total_failed,
            total_duration=self._stats.total_duration,
        )

    @property
    def worker_count(self) -> int:
        """Get the configured number of workers."""
        return self._worker_count


def create_pool(
    worker_count: int = 3,
    *,
    shared_retry: Retry | None = None,
    shared_timeout: Timeout | None = None,
    shared_guardrails: list["GuardrailRule"] | None = None,
    on_event: Callable[["ObservabilityEvent"], None] | None = None,
    context: dict[str, Any] | None = None,
) -> OperationPool[Any]:
    """Create an operation pool for dynamic workload management.

    The pool executes streaming operations with a shared concurrency limit,
    allowing you to submit operations dynamically and process them as
    workers become available.

    Args:
        worker_count: Maximum concurrent operations (default: 3)
        shared_retry: Retry config applied to all operations
        shared_timeout: Timeout config applied to all operations
        shared_guardrails: Guardrails applied to all operations
        on_event: Callback for observability events
        context: User context attached to all events

    Returns:
        OperationPool instance

    Example:
        ```python
        import l0

        pool = l0.create_pool(
            worker_count=3,
            shared_retry=l0.Retry.recommended(),
            shared_guardrails=l0.Guardrails.recommended(),
        )

        # Submit operations
        result1 = pool.execute(stream=lambda: client.chat.completions.create(...))
        result2 = pool.execute(stream=lambda: client.chat.completions.create(...))

        # Wait for all
        await pool.drain()

        # Get results (State with accumulated content)
        state1 = await result1
        state2 = await result2

        print(state1.content)
        print(state2.content)
        ```
    """
    options = PoolOptions(
        shared_retry=shared_retry,
        shared_timeout=shared_timeout,
        shared_guardrails=shared_guardrails,
        on_event=on_event,
        context=context,
    )
    return OperationPool(worker_count=worker_count, options=options)
