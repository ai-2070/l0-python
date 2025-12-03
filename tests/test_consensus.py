"""Tests for l0.consensus module."""

import pytest

from l0.consensus import consensus


class TestConsensus:
    @pytest.mark.asyncio
    async def test_unanimous_success(self):
        async def task():
            return "same"

        result = await consensus([task, task, task], strategy="unanimous")
        assert result == "same"

    @pytest.mark.asyncio
    async def test_unanimous_failure(self):
        async def task1():
            return "a"

        async def task2():
            return "b"

        with pytest.raises(ValueError, match="No unanimous consensus"):
            await consensus([task1, task2], strategy="unanimous")

    @pytest.mark.asyncio
    async def test_majority_success(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        result = await consensus([task_a, task_a, task_b], strategy="majority")
        assert result == "a"

    @pytest.mark.asyncio
    async def test_majority_failure_no_majority(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        # 1 vs 1 - no majority
        with pytest.raises(ValueError, match="No majority consensus"):
            await consensus([task_a, task_b], strategy="majority")

    @pytest.mark.asyncio
    async def test_best_returns_first(self):
        results = []

        async def task1():
            results.append(1)
            return "first"

        async def task2():
            results.append(2)
            return "second"

        result = await consensus([task1, task2], strategy="best")
        assert result == "first"

    @pytest.mark.asyncio
    async def test_empty_tasks_raises(self):
        with pytest.raises(RuntimeError, match="No tasks provided"):
            await consensus([])

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises(self):
        async def task():
            return "a"

        with pytest.raises(ValueError, match="Unknown strategy"):
            await consensus([task], strategy="invalid")  # type: ignore
