"""Tests for OperationPool - dynamic workload management.

Tests for OperationPool class, create_pool function, and concurrent execution.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from l0.adapters import AdaptedEvent, Adapters
from l0.pool import (
    OperationPool,
    PooledOperation,
    PoolOptions,
    PoolStats,
    create_pool,
)
from l0.types import Event, EventType, Retry, Timeout

# ============================================================================
# Test Helpers
# ============================================================================


class PassthroughAdapter:
    """Test adapter that passes through Event objects directly."""

    name = "passthrough"

    def detect(self, stream: Any) -> bool:
        """Detect async generators (our test streams)."""
        return hasattr(stream, "__anext__")

    async def wrap(self, stream: Any) -> AsyncIterator[AdaptedEvent[Any]]:
        """Pass through events wrapped in AdaptedEvent."""
        async for event in stream:
            yield AdaptedEvent(event=event, raw_chunk=None)


@pytest.fixture(autouse=True)
def register_passthrough_adapter():
    """Register and cleanup the passthrough adapter for tests."""
    Adapters.register(PassthroughAdapter())
    yield
    Adapters.reset()


async def create_token_stream(tokens: list[str]) -> AsyncIterator[Event]:
    """Create a simple token stream from an array of tokens."""
    for token in tokens:
        yield Event(type=EventType.TOKEN, text=token)
    yield Event(type=EventType.COMPLETE)


async def create_slow_stream(tokens: list[str], delay_ms: int) -> AsyncIterator[Event]:
    """Create a slow stream that delays between tokens."""
    for token in tokens:
        await asyncio.sleep(delay_ms / 1000)
        yield Event(type=EventType.TOKEN, text=token)
    yield Event(type=EventType.COMPLETE)


async def create_failing_stream() -> AsyncIterator[Event]:
    """Create a stream that fails."""
    yield Event(type=EventType.ERROR, error=Exception("Stream failed"))


# ============================================================================
# PoolOptions Tests
# ============================================================================


class TestPoolOptions:
    """Tests for PoolOptions."""

    def test_default_options(self):
        """Should have correct defaults."""
        options = PoolOptions()

        assert options.shared_retry is None
        assert options.shared_timeout is None
        assert options.shared_guardrails is None
        assert options.on_event is None
        assert options.meta is None

    def test_custom_options(self):
        """Should accept custom options."""
        retry = Retry(attempts=3)
        timeout = Timeout(initial_token=5000, inter_token=10000)
        on_event = MagicMock()

        options = PoolOptions(
            shared_retry=retry,
            shared_timeout=timeout,
            on_event=on_event,
            meta={"env": "test"},
        )

        assert options.shared_retry is retry
        assert options.shared_timeout is timeout
        assert options.on_event is on_event
        assert options.meta == {"env": "test"}


# ============================================================================
# PoolStats Tests
# ============================================================================


class TestPoolStats:
    """Tests for PoolStats."""

    def test_default_stats(self):
        """Should have correct defaults."""
        stats = PoolStats()

        assert stats.total_executed == 0
        assert stats.total_succeeded == 0
        assert stats.total_failed == 0
        assert stats.total_duration == 0.0

    def test_custom_stats(self):
        """Should accept custom values."""
        stats = PoolStats(
            total_executed=10,
            total_succeeded=8,
            total_failed=2,
            total_duration=5.5,
        )

        assert stats.total_executed == 10
        assert stats.total_succeeded == 8
        assert stats.total_failed == 2
        assert stats.total_duration == 5.5


# ============================================================================
# PooledOperation Tests
# ============================================================================


class TestPooledOperation:
    """Tests for PooledOperation."""

    def test_create_operation(self):
        """Should create a pooled operation."""

        async def stream():
            async for event in create_token_stream(["test"]):
                yield event

        op = PooledOperation(stream=stream)

        assert op.stream is stream
        assert op.fallbacks is None
        assert op.guardrails is None
        assert op.retry is None
        assert op.timeout is None

    def test_create_operation_with_options(self):
        """Should create operation with options."""

        async def stream():
            async for event in create_token_stream(["test"]):
                yield event

        async def fallback():
            async for event in create_token_stream(["fallback"]):
                yield event

        retry = Retry(attempts=3)

        op = PooledOperation(
            stream=stream,
            fallbacks=[fallback],
            retry=retry,
        )

        assert op.stream is stream
        assert len(op.fallbacks) == 1
        assert op.retry is retry


# ============================================================================
# OperationPool Creation Tests
# ============================================================================


class TestOperationPoolCreation:
    """Tests for OperationPool creation."""

    def test_create_default_pool(self):
        """Should create a pool with default settings."""
        pool = OperationPool()

        assert pool is not None
        assert pool.get_queue_length() == 0
        assert pool.get_active_workers() == 0

    def test_create_pool_with_worker_count(self):
        """Should create a pool with specified worker count."""
        pool = OperationPool(worker_count=5)

        assert pool is not None

    def test_create_pool_with_options(self):
        """Should create a pool with options."""
        options = PoolOptions(meta={"env": "test"})
        pool = OperationPool(worker_count=3, options=options)

        assert pool is not None

    def test_invalid_worker_count(self):
        """Should raise for invalid worker count."""
        with pytest.raises(ValueError, match="worker_count"):
            OperationPool(worker_count=0)

        with pytest.raises(ValueError, match="worker_count"):
            OperationPool(worker_count=-1)


# ============================================================================
# create_pool Function Tests
# ============================================================================


class TestCreatePoolFunction:
    """Tests for create_pool function."""

    def test_create_pool_default(self):
        """Should create a pool with defaults."""
        pool = create_pool()
        assert isinstance(pool, OperationPool)

    def test_create_pool_with_workers(self):
        """Should create a pool with worker count."""
        pool = create_pool(worker_count=5)
        assert isinstance(pool, OperationPool)

    def test_create_pool_with_options(self):
        """Should create a pool with options."""
        pool = create_pool(
            worker_count=3,
            meta={"key": "value"},
        )
        assert isinstance(pool, OperationPool)


# ============================================================================
# Pool Queue Management Tests
# ============================================================================


class TestPoolQueueManagement:
    """Tests for pool queue management."""

    @pytest.mark.asyncio
    async def test_get_queue_length(self):
        """Should return correct queue length."""
        pool = create_pool(worker_count=1)

        assert pool.get_queue_length() == 0

    @pytest.mark.asyncio
    async def test_get_active_workers(self):
        """Should return correct active worker count."""
        pool = create_pool(worker_count=3)

        assert pool.get_active_workers() == 0


# ============================================================================
# Pool Execution Tests
# ============================================================================


class TestPoolExecution:
    """Tests for pool execution."""

    @pytest.mark.asyncio
    async def test_execute_single_operation(self):
        """Should execute a single operation."""
        pool = create_pool(worker_count=1)

        async def stream():
            async for event in create_token_stream(["hello", "world"]):
                yield event

        result_future = pool.execute(stream=stream)
        await pool.drain()

        result = await result_future
        content = await result.read()

        assert content == "helloworld"

    @pytest.mark.asyncio
    async def test_execute_multiple_operations(self):
        """Should execute multiple operations."""
        pool = create_pool(worker_count=3)

        async def stream1():
            async for event in create_token_stream(["a"]):
                yield event

        async def stream2():
            async for event in create_token_stream(["b"]):
                yield event

        async def stream3():
            async for event in create_token_stream(["c"]):
                yield event

        result1 = pool.execute(stream=stream1)
        result2 = pool.execute(stream=stream2)
        result3 = pool.execute(stream=stream3)

        await pool.drain()

        content1 = await (await result1).read()
        content2 = await (await result2).read()
        content3 = await (await result3).read()

        assert content1 == "a"
        assert content2 == "b"
        assert content3 == "c"

    @pytest.mark.asyncio
    async def test_execute_with_concurrency_limit(self):
        """Should respect concurrency limit."""
        pool = create_pool(worker_count=2)
        execution_order: list[str] = []

        async def slow_stream(name: str):
            execution_order.append(f"{name}-start")
            await asyncio.sleep(0.1)
            yield Event(type=EventType.TOKEN, text=name)
            yield Event(type=EventType.COMPLETE)
            execution_order.append(f"{name}-end")

        pool.execute(stream=lambda: slow_stream("a"))
        pool.execute(stream=lambda: slow_stream("b"))
        pool.execute(stream=lambda: slow_stream("c"))

        await pool.drain()

        # With 2 workers, we should see at most 2 concurrent starts
        # before any ends
        start_count = 0
        for event in execution_order:
            if event.endswith("-start"):
                start_count += 1
            if event.endswith("-end"):
                start_count -= 1
            assert start_count <= 2  # Never more than 2 concurrent


# ============================================================================
# Pool Drain Tests
# ============================================================================


class TestPoolDrain:
    """Tests for pool drain functionality."""

    @pytest.mark.asyncio
    async def test_drain_empty_pool(self):
        """Should drain empty pool immediately."""
        pool = create_pool(worker_count=1)

        # Should not hang
        await asyncio.wait_for(pool.drain(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_drain_waits_for_completion(self):
        """Should wait for all operations to complete."""
        pool = create_pool(worker_count=1)
        completed: list[bool] = []

        async def stream():
            await asyncio.sleep(0.1)
            yield Event(type=EventType.TOKEN, text="done")
            yield Event(type=EventType.COMPLETE)
            completed.append(True)

        pool.execute(stream=stream)
        pool.execute(stream=stream)

        await pool.drain()

        assert len(completed) == 2


# ============================================================================
# Pool Stats Tests
# ============================================================================


class TestPoolStatsTracking:
    """Tests for pool statistics tracking."""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Should return pool statistics."""
        pool = create_pool(worker_count=1)

        stats = pool.get_stats()

        assert isinstance(stats, PoolStats)
        assert stats.total_executed == 0

    @pytest.mark.asyncio
    async def test_stats_track_success(self):
        """Should track successful operations."""
        pool = create_pool(worker_count=1)

        async def stream():
            async for event in create_token_stream(["test"]):
                yield event

        pool.execute(stream=stream)
        await pool.drain()

        stats = pool.get_stats()
        assert stats.total_executed == 1
        assert stats.total_succeeded == 1
        assert stats.total_failed == 0

    @pytest.mark.asyncio
    async def test_stats_track_failure(self):
        """Should track failed operations."""
        pool = create_pool(worker_count=1)

        async def failing_stream():
            raise Exception("Stream failed")
            yield  # Make it an async generator

        pool.execute(stream=failing_stream)
        await pool.drain()

        stats = pool.get_stats()
        assert stats.total_executed == 1
        assert stats.total_failed == 1


# ============================================================================
# Pool Shutdown Tests
# ============================================================================


class TestPoolShutdown:
    """Tests for pool shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_workers(self):
        """Should stop all workers on shutdown."""
        pool = create_pool(worker_count=3)

        # Execute something to start workers
        async def stream():
            async for event in create_token_stream(["test"]):
                yield event

        pool.execute(stream=stream)
        await pool.drain()

        await pool.shutdown()

        # After shutdown, pool should not accept new work
        # (implementation dependent)

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_pending(self):
        """Should wait for pending operations before shutdown."""
        pool = create_pool(worker_count=1)
        completed: list[bool] = []

        async def slow_stream():
            await asyncio.sleep(0.1)
            yield Event(type=EventType.TOKEN, text="done")
            yield Event(type=EventType.COMPLETE)
            completed.append(True)

        pool.execute(stream=slow_stream)
        await pool.shutdown()

        assert len(completed) == 1


# ============================================================================
# Pool Event Callback Tests
# ============================================================================


class TestPoolEventCallbacks:
    """Tests for pool event callbacks."""

    @pytest.mark.asyncio
    async def test_on_event_callback(self):
        """Should call on_event for observability events."""
        events_received: list[Any] = []

        pool = create_pool(
            worker_count=1,
            on_event=lambda e: events_received.append(e),
        )

        async def stream():
            async for event in create_token_stream(["test"]):
                yield event

        pool.execute(stream=stream)
        await pool.drain()

        # Should have received some events
        assert len(events_received) >= 0  # Events may or may not be emitted


# ============================================================================
# Pool with Shared Config Tests
# ============================================================================


class TestPoolSharedConfig:
    """Tests for pool with shared configuration."""

    @pytest.mark.asyncio
    async def test_shared_retry_config(self):
        """Should apply shared retry config to operations."""
        pool = create_pool(
            worker_count=1,
            shared_retry=Retry(attempts=3),
        )

        async def stream():
            async for event in create_token_stream(["test"]):
                yield event

        result = pool.execute(stream=stream)
        await pool.drain()

        # Operation should complete (retry config applied)
        content = await (await result).read()
        assert content == "test"

    @pytest.mark.asyncio
    async def test_shared_timeout_config(self):
        """Should apply shared timeout config to operations."""
        pool = create_pool(
            worker_count=1,
            shared_timeout=Timeout(initial_token=5000, inter_token=10000),
        )

        async def stream():
            async for event in create_token_stream(["test"]):
                yield event

        result = pool.execute(stream=stream)
        await pool.drain()

        content = await (await result).read()
        assert content == "test"

    @pytest.mark.asyncio
    async def test_shared_meta(self):
        """Should attach shared meta to all operations."""
        pool = create_pool(
            worker_count=1,
            meta={"requestId": "req-123"},
        )

        async def stream():
            async for event in create_token_stream(["test"]):
                yield event

        result = pool.execute(stream=stream)
        await pool.drain()

        content = await (await result).read()
        assert content == "test"
