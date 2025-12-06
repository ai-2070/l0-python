"""Tests for race() and parallel() with enhanced result types.

Tests for RaceResult, AggregatedTelemetry, and ParallelResult enhancements.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from l0.parallel import (
    AggregatedTelemetry,
    ParallelOptions,
    ParallelResult,
    RaceResult,
    batched,
    parallel,
    race,
    sequential,
)

# ============================================================================
# AggregatedTelemetry Tests
# ============================================================================


class TestAggregatedTelemetry:
    """Tests for AggregatedTelemetry dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        telemetry = AggregatedTelemetry()

        assert telemetry.total_tokens == 0
        assert telemetry.total_duration == 0.0
        assert telemetry.total_retries == 0
        assert telemetry.total_network_errors == 0
        assert telemetry.total_violations == 0
        assert telemetry.avg_tokens_per_second == 0.0
        assert telemetry.avg_time_to_first_token == 0.0

    def test_custom_values(self):
        """Should accept custom values."""
        telemetry = AggregatedTelemetry(
            total_tokens=100,
            total_duration=5.5,
            total_retries=2,
            total_network_errors=1,
            total_violations=0,
            avg_tokens_per_second=18.2,
            avg_time_to_first_token=0.5,
        )

        assert telemetry.total_tokens == 100
        assert telemetry.total_duration == 5.5
        assert telemetry.total_retries == 2
        assert telemetry.total_network_errors == 1
        assert telemetry.total_violations == 0
        assert telemetry.avg_tokens_per_second == 18.2
        assert telemetry.avg_time_to_first_token == 0.5


# ============================================================================
# RaceResult Tests
# ============================================================================


class TestRaceResult:
    """Tests for RaceResult dataclass."""

    def test_create_race_result(self):
        """Should create a RaceResult."""
        result = RaceResult(value="winner", winner_index=1)

        assert result.value == "winner"
        assert result.winner_index == 1

    def test_race_result_with_any_type(self):
        """Should work with any value type."""
        # String
        result1 = RaceResult(value="string", winner_index=0)
        assert result1.value == "string"

        # Integer
        result2 = RaceResult(value=42, winner_index=1)
        assert result2.value == 42

        # Dict
        result3 = RaceResult(value={"key": "value"}, winner_index=2)
        assert result3.value == {"key": "value"}

        # List
        result4 = RaceResult(value=[1, 2, 3], winner_index=0)
        assert result4.value == [1, 2, 3]


# ============================================================================
# ParallelResult Tests
# ============================================================================


class TestParallelResult:
    """Tests for ParallelResult dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        result = ParallelResult(
            results=[],
            errors=[],
        )

        assert result.results == []
        assert result.errors == []
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.duration == 0.0
        assert result.aggregated_telemetry is None

    def test_all_succeeded_property(self):
        """Should correctly compute all_succeeded."""
        success_result = ParallelResult(
            results=["a", "b", "c"],
            errors=[None, None, None],
            success_count=3,
            failure_count=0,
        )
        assert success_result.all_succeeded is True

        partial_result = ParallelResult(
            results=["a", None, "c"],
            errors=[None, Exception("failed"), None],
            success_count=2,
            failure_count=1,
        )
        assert partial_result.all_succeeded is False

    def test_successful_results_method(self):
        """Should return only successful results."""
        result = ParallelResult(
            results=["a", None, "c", None],
            errors=[None, Exception("e1"), None, Exception("e2")],
            success_count=2,
            failure_count=2,
        )

        successful = result.successful_results()
        assert successful == ["a", "c"]

    def test_with_aggregated_telemetry(self):
        """Should include aggregated telemetry."""
        telemetry = AggregatedTelemetry(
            total_tokens=50,
            total_duration=2.0,
        )

        result = ParallelResult(
            results=["a"],
            errors=[None],
            success_count=1,
            failure_count=0,
            aggregated_telemetry=telemetry,
        )

        assert result.aggregated_telemetry is not None
        assert result.aggregated_telemetry.total_tokens == 50


# ============================================================================
# race() Function Tests
# ============================================================================


class TestRaceFunction:
    """Tests for race() function."""

    @pytest.mark.asyncio
    async def test_race_returns_first_success(self):
        """Should return first successful result."""

        async def fast():
            return "fast"

        async def slow():
            await asyncio.sleep(1.0)
            return "slow"

        result = await race([fast, slow])

        assert isinstance(result, RaceResult)
        assert result.value == "fast"

    @pytest.mark.asyncio
    async def test_race_winner_index_correct(self):
        """Should return correct winner_index."""

        async def slow():
            await asyncio.sleep(0.2)
            return "slow"

        async def fast():
            return "fast"

        # fast is at index 1
        result = await race([slow, fast])

        assert result.value == "fast"
        assert result.winner_index == 1

    @pytest.mark.asyncio
    async def test_race_cancels_losers(self):
        """Should cancel remaining tasks when winner found."""
        slow_started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fast():
            # Wait a tiny bit to ensure slow task starts
            await asyncio.sleep(0.01)
            return "fast"

        async def slow():
            try:
                slow_started.set()
                await asyncio.sleep(10.0)
                return "slow"
            except asyncio.CancelledError:
                cancelled.set()
                raise

        result = await race([fast, slow])

        # Give time for cancellation to propagate
        await asyncio.sleep(0.1)

        assert result.value == "fast"
        # The slow task should have started and been cancelled
        assert slow_started.is_set(), "slow task should have started"
        assert cancelled.is_set(), "slow task should have been cancelled"

    @pytest.mark.asyncio
    async def test_race_with_failing_task(self):
        """Should skip failing tasks and return first success."""

        async def failing():
            raise Exception("Failed")

        async def success():
            return "success"

        result = await race([failing, success])

        assert result.value == "success"
        assert result.winner_index == 1

    @pytest.mark.asyncio
    async def test_race_all_fail_raises(self):
        """Should raise if all tasks fail."""

        async def fail1():
            raise Exception("Fail 1")

        async def fail2():
            raise Exception("Fail 2")

        with pytest.raises(Exception):
            await race([fail1, fail2])

    @pytest.mark.asyncio
    async def test_race_empty_tasks_raises(self):
        """Should raise for empty task list."""
        with pytest.raises(RuntimeError, match="No tasks"):
            await race([])

    @pytest.mark.asyncio
    async def test_race_on_error_callback(self):
        """Should call on_error for failed tasks."""
        errors: list[tuple[Exception, int]] = []

        async def failing():
            raise Exception("Task failed")

        async def success():
            await asyncio.sleep(0.1)
            return "success"

        result = await race(
            [failing, success],
            on_error=lambda e, idx: errors.append((e, idx)),
        )

        assert result.value == "success"
        assert len(errors) == 1
        assert errors[0][1] == 0  # failing was at index 0


# ============================================================================
# parallel() Function Tests
# ============================================================================


class TestParallelFunction:
    """Tests for parallel() function."""

    @pytest.mark.asyncio
    async def test_parallel_executes_all(self):
        """Should execute all tasks."""
        results_list: list[str] = []

        async def task1():
            results_list.append("task1")
            return "result1"

        async def task2():
            results_list.append("task2")
            return "result2"

        result = await parallel([task1, task2])

        assert len(results_list) == 2
        assert result.success_count == 2
        assert result.failure_count == 0

    @pytest.mark.asyncio
    async def test_parallel_respects_concurrency(self):
        """Should respect concurrency limit."""
        concurrent_count = 0
        max_concurrent = 0

        async def task():
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return "done"

        await parallel(
            [task] * 10,
            concurrency=3,
        )

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_parallel_fail_fast(self):
        """Should stop on first error when fail_fast=True."""
        completed: list[str] = []

        async def task1():
            await asyncio.sleep(0.05)
            completed.append("task1")
            return "result1"

        async def failing():
            raise Exception("Failed")

        async def task2():
            await asyncio.sleep(0.1)
            completed.append("task2")
            return "result2"

        result = await parallel(
            [task1, failing, task2],
            fail_fast=True,
        )

        # Some tasks may have completed before failure was noticed
        assert result.failure_count >= 1

    @pytest.mark.asyncio
    async def test_parallel_continue_on_error(self):
        """Should continue on error when fail_fast=False."""

        async def task1():
            return "result1"

        async def failing():
            raise Exception("Failed")

        async def task2():
            return "result2"

        result = await parallel(
            [task1, failing, task2],
            fail_fast=False,
        )

        assert result.success_count == 2
        assert result.failure_count == 1

    @pytest.mark.asyncio
    async def test_parallel_on_progress_callback(self):
        """Should call on_progress callback."""
        progress_calls: list[tuple[int, int]] = []

        async def task():
            return "done"

        await parallel(
            [task, task, task],
            on_progress=lambda completed, total: progress_calls.append(
                (completed, total)
            ),
        )

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    @pytest.mark.asyncio
    async def test_parallel_on_complete_callback(self):
        """Should call on_complete for each task."""
        complete_calls: list[tuple[Any, int]] = []

        async def task1():
            return "result1"

        async def task2():
            return "result2"

        await parallel(
            [task1, task2],
            on_complete=lambda result, idx: complete_calls.append((result, idx)),
        )

        assert len(complete_calls) == 2

    @pytest.mark.asyncio
    async def test_parallel_on_error_callback(self):
        """Should call on_error for failed tasks."""
        error_calls: list[tuple[Exception, int]] = []

        async def failing():
            raise Exception("Failed")

        async def task():
            return "done"

        await parallel(
            [failing, task],
            fail_fast=False,
            on_error=lambda err, idx: error_calls.append((err, idx)),
        )

        assert len(error_calls) == 1
        assert error_calls[0][1] == 0  # failing was at index 0

    @pytest.mark.asyncio
    async def test_parallel_result_duration(self):
        """Should track execution duration."""

        async def slow_task():
            await asyncio.sleep(0.1)
            return "done"

        result = await parallel([slow_task])

        assert result.duration >= 0.1

    @pytest.mark.asyncio
    async def test_parallel_empty_tasks(self):
        """Should handle empty task list."""
        result = await parallel([])

        assert result.results == []
        assert result.errors == []
        assert result.success_count == 0
        assert result.failure_count == 0

    @pytest.mark.asyncio
    async def test_parallel_with_options(self):
        """Should accept ParallelOptions."""

        async def task():
            return "done"

        options = ParallelOptions(
            concurrency=2,
            fail_fast=False,
        )

        result = await parallel([task, task], options=options)

        assert result.success_count == 2


# ============================================================================
# sequential() Function Tests
# ============================================================================


class TestSequentialFunction:
    """Tests for sequential() function."""

    @pytest.mark.asyncio
    async def test_sequential_executes_in_order(self):
        """Should execute tasks in order."""
        order: list[int] = []

        async def task(n: int):
            order.append(n)
            return n

        results = await sequential(
            [
                lambda: task(1),
                lambda: task(2),
                lambda: task(3),
            ]
        )

        assert order == [1, 2, 3]
        assert results == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_sequential_stops_on_error(self):
        """Should stop on error."""
        order: list[int] = []

        async def task(n: int):
            order.append(n)
            if n == 2:
                raise Exception("Failed")
            return n

        with pytest.raises(Exception):
            await sequential(
                [
                    lambda: task(1),
                    lambda: task(2),
                    lambda: task(3),
                ]
            )

        assert order == [1, 2]

    @pytest.mark.asyncio
    async def test_sequential_empty_tasks(self):
        """Should handle empty task list."""
        results = await sequential([])
        assert results == []


# ============================================================================
# batched() Function Tests
# ============================================================================


class TestBatchedFunction:
    """Tests for batched() function."""

    @pytest.mark.asyncio
    async def test_batched_processes_all_items(self):
        """Should process all items."""

        async def handler(item: int) -> int:
            return item * 2

        results = await batched(
            [1, 2, 3, 4, 5],
            handler,
            batch_size=2,
        )

        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_batched_respects_batch_size(self):
        """Should process in batches of specified size."""
        batch_starts: list[int] = []
        items_in_progress = 0
        max_concurrent = 0

        async def handler(item: int) -> int:
            nonlocal items_in_progress, max_concurrent
            items_in_progress += 1
            max_concurrent = max(max_concurrent, items_in_progress)
            await asyncio.sleep(0.05)
            items_in_progress -= 1
            return item

        await batched(
            [1, 2, 3, 4, 5, 6],
            handler,
            batch_size=2,
        )

        # Max concurrent should be at most batch_size
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_batched_on_progress_callback(self):
        """Should call on_progress callback."""
        progress: list[tuple[int, int]] = []

        async def handler(item: int) -> int:
            return item

        await batched(
            [1, 2, 3, 4],
            handler,
            batch_size=2,
            on_progress=lambda done, total: progress.append((done, total)),
        )

        assert progress[-1] == (4, 4)

    @pytest.mark.asyncio
    async def test_batched_preserves_order(self):
        """Should preserve item order in results."""

        async def handler(item: int) -> int:
            await asyncio.sleep(0.01 * (5 - item))  # Inverse delay
            return item

        results = await batched(
            [1, 2, 3, 4, 5],
            handler,
            batch_size=5,
        )

        assert results == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_batched_empty_items(self):
        """Should handle empty item list."""

        async def handler(item: int) -> int:
            return item

        results = await batched([], handler, batch_size=2)

        assert results == []
