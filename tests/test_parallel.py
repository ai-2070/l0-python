"""Tests for l0.parallel module."""

import asyncio

import pytest

from l0.parallel import (
    ParallelOptions,
    ParallelResult,
    batched,
    parallel,
    race,
    sequential,
)


class TestParallel:
    @pytest.mark.asyncio
    async def test_runs_all_tasks(self):
        """Test that all tasks are executed."""
        results_collected = []

        async def task(n: int):
            results_collected.append(n)
            return n * 2

        result = await parallel([lambda n=i: task(n) for i in range(5)])

        assert isinstance(result, ParallelResult)
        assert result.success_count == 5
        assert result.failure_count == 0
        assert result.all_succeeded
        assert result.successful_results() == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_respects_concurrency(self):
        """Test that concurrency limit is respected."""
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
        """Test empty task list."""
        result = await parallel([])
        assert result.results == []
        assert result.success_count == 0
        assert result.all_succeeded

    @pytest.mark.asyncio
    async def test_fail_fast(self):
        """Test fail_fast stops on first error."""
        call_count = 0

        async def failing_task():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First task fails")
            await asyncio.sleep(0.1)
            return "success"

        result = await parallel(
            [failing_task for _ in range(5)],
            concurrency=1,
            fail_fast=True,
        )

        assert result.failure_count >= 1
        # With fail_fast, not all tasks should complete
        assert result.success_count + result.failure_count <= 5

    @pytest.mark.asyncio
    async def test_on_progress_callback(self):
        """Test on_progress callback is called."""
        progress_updates = []

        async def task():
            return True

        await parallel(
            [task for _ in range(3)],
            on_progress=lambda done, total: progress_updates.append((done, total)),
        )

        assert len(progress_updates) == 3
        assert progress_updates[-1] == (3, 3)

    @pytest.mark.asyncio
    async def test_on_complete_callback(self):
        """Test on_complete callback is called."""
        completions = []

        async def task(n: int):
            return n * 2

        await parallel(
            [lambda n=i: task(n) for i in range(3)],
            on_complete=lambda result, idx: completions.append((result, idx)),
        )

        assert len(completions) == 3

    @pytest.mark.asyncio
    async def test_on_error_callback(self):
        """Test on_error callback is called."""
        errors_received = []

        async def failing_task():
            raise ValueError("Task failed")

        await parallel(
            [failing_task for _ in range(2)],
            on_error=lambda err, idx: errors_received.append((str(err), idx)),
        )

        assert len(errors_received) == 2

    @pytest.mark.asyncio
    async def test_parallel_options_object(self):
        """Test using ParallelOptions object."""

        async def task():
            return True

        options = ParallelOptions(concurrency=2, fail_fast=False)
        result = await parallel([task for _ in range(3)], options=options)

        assert result.success_count == 3

    @pytest.mark.asyncio
    async def test_tracks_duration(self):
        """Test that duration is tracked."""

        async def task():
            await asyncio.sleep(0.01)
            return True

        result = await parallel([task for _ in range(2)])

        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_handles_mixed_success_failure(self):
        """Test handling mix of successes and failures."""

        async def task(n: int):
            if n % 2 == 0:
                raise ValueError(f"Even number: {n}")
            return n

        result = await parallel([lambda n=i: task(n) for i in range(4)])

        assert result.success_count == 2  # 1, 3
        assert result.failure_count == 2  # 0, 2
        assert not result.all_succeeded
        assert result.successful_results() == [1, 3]

    @pytest.mark.asyncio
    async def test_successful_results_includes_none_returns(self):
        """Test that successful_results includes tasks that return None."""

        async def task_returning_none():
            return None

        async def task_returning_value():
            return "value"

        result = await parallel([task_returning_none, task_returning_value])

        assert result.success_count == 2
        assert result.failure_count == 0
        # Both should be in successful_results, including the None
        successful = result.successful_results()
        assert len(successful) == 2
        assert None in successful
        assert "value" in successful


class TestRace:
    @pytest.mark.asyncio
    async def test_returns_first_result(self):
        """Test returns first successful result."""

        async def fast():
            return "fast"

        async def slow():
            await asyncio.sleep(1)
            return "slow"

        result = await race([fast, slow])
        assert result.value == "fast"
        assert result.winner_index == 0

    @pytest.mark.asyncio
    async def test_cancels_remaining(self):
        """Test remaining tasks are cancelled."""
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
        """Test empty task list raises."""
        with pytest.raises(RuntimeError, match="No tasks provided"):
            await race([])

    @pytest.mark.asyncio
    async def test_skips_failed_returns_success(self):
        """Test that failed tasks are skipped, returns first success."""

        async def failing():
            raise ValueError("I fail")

        async def succeeding():
            await asyncio.sleep(0.01)
            return "success"

        result = await race([failing, succeeding])
        assert result.value == "success"

    @pytest.mark.asyncio
    async def test_all_fail_raises_last(self):
        """Test all tasks failing raises last error."""

        async def fail1():
            raise ValueError("Error 1")

        async def fail2():
            await asyncio.sleep(0.01)
            raise ValueError("Error 2")

        with pytest.raises(ValueError):
            await race([fail1, fail2])

    @pytest.mark.asyncio
    async def test_fast_failure_does_not_abort_slow_success(self):
        """Test that a fast-failing task doesn't abort a slower successful task."""

        async def fast_fail():
            raise ValueError("I fail immediately")

        async def slow_success():
            await asyncio.sleep(0.05)
            return "success"

        result = await race([fast_fail, slow_success])
        assert result.value == "success"

    @pytest.mark.asyncio
    async def test_on_error_receives_correct_task_index(self):
        """Test that on_error callback receives the correct task index."""
        error_indices: list[int] = []

        async def task0_fail():
            raise ValueError("Task 0 failed")

        async def task1_fail():
            await asyncio.sleep(0.01)
            raise ValueError("Task 1 failed")

        async def task2_success():
            await asyncio.sleep(0.02)
            return "success"

        def on_error(e: Exception, index: int):
            error_indices.append(index)

        result = await race([task0_fail, task1_fail, task2_success], on_error=on_error)
        assert result.value == "success"
        # Both failing tasks should report their correct indices
        assert 0 in error_indices
        assert 1 in error_indices


class TestSequential:
    @pytest.mark.asyncio
    async def test_runs_in_order(self):
        """Test tasks run sequentially in order."""
        order = []

        async def task(n: int):
            order.append(n)
            return n

        results = await sequential([lambda n=i: task(n) for i in range(3)])

        assert results == [0, 1, 2]
        assert order == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        """Test empty task list."""
        results = await sequential([])
        assert results == []

    @pytest.mark.asyncio
    async def test_one_at_a_time(self):
        """Test only one task runs at a time."""
        running = 0
        max_running = 0

        async def task():
            nonlocal running, max_running
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0.01)
            running -= 1
            return True

        await sequential([task for _ in range(5)])

        assert max_running == 1


class TestBatched:
    @pytest.mark.asyncio
    async def test_processes_in_batches(self):
        """Test items are processed in batches."""
        processed = []

        async def handler(item: int) -> int:
            processed.append(item)
            return item * 2

        result = await batched([1, 2, 3, 4, 5], handler, batch_size=2)

        assert result == [2, 4, 6, 8, 10]
        assert processed == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_empty_items(self):
        """Test empty item list."""

        async def handler(item: int) -> int:
            return item

        result = await batched([], handler)
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_size_larger_than_items(self):
        """Test batch size larger than items."""

        async def handler(item: int) -> int:
            return item

        result = await batched([1, 2], handler, batch_size=10)
        assert result == [1, 2]

    @pytest.mark.asyncio
    async def test_on_progress_callback(self):
        """Test on_progress callback."""
        progress = []

        async def handler(item: int) -> int:
            return item

        await batched(
            [1, 2, 3, 4, 5],
            handler,
            batch_size=2,
            on_progress=lambda done, total: progress.append((done, total)),
        )

        assert progress == [(2, 5), (4, 5), (5, 5)]
