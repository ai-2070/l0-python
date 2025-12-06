"""Tests for l0.pool module."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from l0.adapters import AdaptedEvent, Adapters
from l0.pool import OperationPool, create_pool
from l0.types import Event, EventType, State


class PassthroughAdapter:
    """Test adapter that passes through Event objects directly."""

    name = "passthrough"

    def detect(self, stream: Any) -> bool:
        """Detect async generators (our test streams)."""
        return hasattr(stream, "__anext__")

    async def wrap(
        self, stream: Any, options: Any = None
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        """Pass through events wrapped in AdaptedEvent."""
        async for event in stream:
            yield AdaptedEvent(event=event, raw_chunk=None)


@pytest.fixture(autouse=True)
def register_passthrough_adapter() -> Any:
    """Register and cleanup the passthrough adapter for tests."""
    Adapters.register(PassthroughAdapter())
    yield
    Adapters.reset()


def make_stream(content: str):
    """Create a stream factory that yields the given content."""

    async def stream() -> AsyncIterator[Event]:
        for char in content:
            yield Event(type=EventType.TOKEN, text=char)
        yield Event(type=EventType.COMPLETE)

    return stream


class TestPoolCreation:
    """Tests for pool creation and configuration."""

    def test_create_pool_default_workers(self) -> None:
        """Test that create_pool defaults to 3 workers."""
        pool: OperationPool[Any] = create_pool()
        assert pool.worker_count == 3

    def test_create_pool_custom_workers(self) -> None:
        """Test that create_pool accepts custom worker count."""
        pool: OperationPool[Any] = create_pool(5)
        assert pool.worker_count == 5

    def test_create_pool_invalid_workers(self) -> None:
        """Test that create_pool rejects invalid worker counts."""
        with pytest.raises(ValueError, match="worker_count must be at least 1"):
            create_pool(0)

        with pytest.raises(ValueError, match="worker_count must be at least 1"):
            create_pool(-1)


class TestPoolExecution:
    """Tests for pool operation execution."""

    @pytest.mark.asyncio
    async def test_execute_returns_future(self) -> None:
        """Test that execute returns a Future immediately."""
        pool: OperationPool[Any] = create_pool(1)

        future = pool.execute(stream=make_stream("hello"))
        assert isinstance(future, asyncio.Future)

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_execute_returns_state_with_content(self) -> None:
        """Test that awaiting the future returns State with accumulated content."""
        pool: OperationPool[Any] = create_pool(1)

        future = pool.execute(stream=make_stream("hello"))
        await pool.drain()

        state = await future
        assert isinstance(state, State)
        assert state.content == "hello"

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_execute_multiple_operations(self) -> None:
        """Test executing multiple operations concurrently."""
        pool: OperationPool[Any] = create_pool(3)

        future1 = pool.execute(stream=make_stream("one"))
        future2 = pool.execute(stream=make_stream("two"))
        future3 = pool.execute(stream=make_stream("three"))

        await pool.drain()

        state1 = await future1
        state2 = await future2
        state3 = await future3

        assert state1.content == "one"
        assert state2.content == "two"
        assert state3.content == "three"

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_state_has_token_count(self) -> None:
        """Test that State includes token count after execution."""
        pool: OperationPool[Any] = create_pool(1)

        future = pool.execute(stream=make_stream("hello"))
        await pool.drain()

        state = await future
        # "hello" = 5 characters, each yielded as a token
        assert state.token_count == 5

        await pool.shutdown()


class TestPoolStats:
    """Tests for pool statistics."""

    @pytest.mark.asyncio
    async def test_stats_track_success(self) -> None:
        """Test that stats track successful operations."""
        pool: OperationPool[Any] = create_pool(1)

        pool.execute(stream=make_stream("one"))
        pool.execute(stream=make_stream("two"))
        await pool.drain()

        stats = pool.get_stats()
        assert stats.total_executed == 2
        assert stats.total_succeeded == 2
        assert stats.total_failed == 0
        assert stats.total_duration > 0

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_stats_track_failure(self) -> None:
        """Test that stats track failed operations."""
        pool: OperationPool[Any] = create_pool(1)

        async def failing_stream() -> AsyncIterator[Event]:
            raise RuntimeError("Stream failed")
            yield  # Make it a generator

        future = pool.execute(stream=failing_stream)
        await pool.drain()

        # The future should have the exception
        with pytest.raises(RuntimeError, match="Stream failed"):
            await future

        stats = pool.get_stats()
        assert stats.total_executed == 1
        assert stats.total_succeeded == 0
        assert stats.total_failed == 1

        await pool.shutdown()


class TestPoolDrain:
    """Tests for pool draining."""

    @pytest.mark.asyncio
    async def test_drain_waits_for_completion(self) -> None:
        """Test that drain waits for all operations to complete."""
        pool: OperationPool[Any] = create_pool(1)
        completed: list[str] = []

        async def slow_stream(name: str) -> AsyncIterator[Event]:
            await asyncio.sleep(0.01)
            completed.append(name)
            yield Event(type=EventType.TOKEN, text=name)
            yield Event(type=EventType.COMPLETE)

        pool.execute(stream=lambda: slow_stream("first"))
        pool.execute(stream=lambda: slow_stream("second"))

        # Before drain, operations may not be complete
        await pool.drain()

        # After drain, all operations should be complete
        assert len(completed) == 2
        assert "first" in completed
        assert "second" in completed

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_cannot_execute_while_draining(self) -> None:
        """Test that execute raises error while pool is draining."""
        pool: OperationPool[Any] = create_pool(1)

        async def slow_stream() -> AsyncIterator[Event]:
            await asyncio.sleep(0.1)
            yield Event(type=EventType.TOKEN, text="slow")
            yield Event(type=EventType.COMPLETE)

        pool.execute(stream=slow_stream)

        # Start draining in background
        drain_task = asyncio.create_task(pool.drain())

        # Give drain time to start
        await asyncio.sleep(0.01)

        # Should raise while draining
        with pytest.raises(
            RuntimeError, match="Cannot execute operations while pool is draining"
        ):
            pool.execute(stream=make_stream("new"))

        await drain_task
        await pool.shutdown()


class TestPoolShutdown:
    """Tests for pool shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_completes_pending(self) -> None:
        """Test that shutdown waits for pending operations."""
        pool: OperationPool[Any] = create_pool(1)

        future = pool.execute(stream=make_stream("test"))

        await pool.shutdown()

        # Future should be resolved after shutdown
        state = await future
        assert state.content == "test"

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self) -> None:
        """Test that shutdown can be called multiple times."""
        pool: OperationPool[Any] = create_pool(1)

        pool.execute(stream=make_stream("test"))

        await pool.shutdown()
        await pool.shutdown()  # Should not raise


class TestPoolQueueAndWorkers:
    """Tests for queue and worker management."""

    @pytest.mark.asyncio
    async def test_get_queue_length(self) -> None:
        """Test that get_queue_length returns correct count."""
        pool: OperationPool[Any] = create_pool(1)

        # Don't start yet - manually control
        assert pool.get_queue_length() == 0

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_get_active_workers(self) -> None:
        """Test that get_active_workers returns correct count."""
        pool: OperationPool[Any] = create_pool(3)

        # Initially no active workers
        assert pool.get_active_workers() == 0

        await pool.shutdown()
