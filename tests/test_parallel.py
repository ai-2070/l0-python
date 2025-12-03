"""Tests for l0.parallel module."""

import asyncio

import pytest

from l0.parallel import batched, parallel, race


class TestParallel:
    @pytest.mark.asyncio
    async def test_runs_all_tasks(self):
        results = []

        async def task(n: int):
            results.append(n)
            return n * 2

        output = await parallel([lambda n=i: task(n) for i in range(5)])

        assert len(output) == 5
        assert output == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_respects_concurrency(self):
        running = 0
        max_running = 0

        async def task():
            nonlocal running, max_running
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0.01)
            running -= 1
            return True

        await parallel([task for _ in range(10)], concurrency=3)

        assert max_running <= 3

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        result = await parallel([])
        assert result == []


class TestRace:
    @pytest.mark.asyncio
    async def test_returns_first_result(self):
        async def fast():
            return "fast"

        async def slow():
            await asyncio.sleep(1)
            return "slow"

        result = await race([fast, slow])
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_cancels_remaining(self):
        cancelled = []

        async def task1():
            return "first"

        async def task2():
            try:
                await asyncio.sleep(10)
                return "second"
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        await race([task1, task2])
        await asyncio.sleep(0.01)  # Let cancellation propagate

        assert len(cancelled) == 1

    @pytest.mark.asyncio
    async def test_empty_tasks_raises(self):
        with pytest.raises(RuntimeError, match="No tasks provided"):
            await race([])


class TestBatched:
    @pytest.mark.asyncio
    async def test_processes_in_batches(self):
        processed = []

        async def handler(item: int) -> int:
            processed.append(item)
            return item * 2

        result = await batched([1, 2, 3, 4, 5], handler, batch_size=2)

        assert result == [2, 4, 6, 8, 10]
        assert processed == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_empty_items(self):
        async def handler(item: int) -> int:
            return item

        result = await batched([], handler)
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_size_larger_than_items(self):
        async def handler(item: int) -> int:
            return item

        result = await batched([1, 2], handler, batch_size=10)
        assert result == [1, 2]
