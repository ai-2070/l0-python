"""Tests for l0.consensus module."""

import pytest
from pydantic import BaseModel

from l0.consensus import (
    Agreement,
    Consensus,
    ConsensusResult,
)


class TestConsensus:
    @pytest.mark.asyncio
    async def test_unanimous_success(self):
        async def task():
            return "same"

        result = await Consensus.run([task, task, task], strategy="unanimous")
        assert result.consensus == "same"
        assert result.confidence >= 0.99
        assert result.status == "success"
        assert len(result.agreements) >= 1

    @pytest.mark.asyncio
    async def test_unanimous_failure(self):
        async def task1():
            return "a"

        async def task2():
            return "b"

        with pytest.raises(ValueError, match="No unanimous consensus"):
            await Consensus.run(
                [task1, task2], strategy="unanimous", resolve_conflicts="fail"
            )

    @pytest.mark.asyncio
    async def test_majority_success(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        result = await Consensus.run([task_a, task_a, task_b], strategy="majority")
        assert result.consensus == "a"
        assert result.confidence >= 0.6
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_majority_failure_no_majority(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        # 1 vs 1 - no majority at default threshold
        with pytest.raises(ValueError, match="No majority consensus"):
            await Consensus.run(
                [task_a, task_b],
                strategy="majority",
                resolve_conflicts="fail",
            )

    @pytest.mark.asyncio
    async def test_best_returns_first(self):
        results = []

        async def task1():
            results.append(1)
            return "first"

        async def task2():
            results.append(2)
            return "second"

        result = await Consensus.run([task1, task2], strategy="best")
        assert result.consensus == "first"
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_empty_tasks_raises(self):
        with pytest.raises(RuntimeError, match="No tasks provided"):
            await Consensus.run([])

    @pytest.mark.asyncio
    async def test_single_task_raises(self):
        async def task():
            return "a"

        with pytest.raises(RuntimeError, match="At least 2 tasks"):
            await Consensus.run([task])

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises(self):
        async def task():
            return "a"

        with pytest.raises(ValueError, match="Unknown strategy"):
            await Consensus.run([task, task], strategy="invalid")  # type: ignore


class TestWeightedConsensus:
    @pytest.mark.asyncio
    async def test_weighted_higher_weight_wins(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        # task_a has higher weight
        result = await Consensus.run(
            [task_a, task_b],
            strategy="weighted",
            weights=[2.0, 1.0],
        )
        assert result.consensus == "a"

    @pytest.mark.asyncio
    async def test_weighted_lower_count_higher_weight(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        # One "a" with weight 3 vs two "b" with weight 1 each
        result = await Consensus.run(
            [task_a, task_b, task_b],
            strategy="weighted",
            weights=[3.0, 1.0, 1.0],
        )
        assert result.consensus == "a"


class TestConflictResolution:
    @pytest.mark.asyncio
    async def test_resolve_vote(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        result = await Consensus.run(
            [task_a, task_a, task_b],
            strategy="majority",
            resolve_conflicts="vote",
        )
        assert result.consensus == "a"

    @pytest.mark.asyncio
    async def test_resolve_best_uses_weight(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        # Low weight "a" vs high weight "b"
        result = await Consensus.run(
            [task_a, task_b],
            strategy="majority",
            resolve_conflicts="best",
            weights=[1.0, 2.0],
            minimum_agreement=0.0,  # Allow any agreement
        )
        assert result.consensus == "b"


class TestConsensusResult:
    @pytest.mark.asyncio
    async def test_result_has_analysis(self):
        async def task():
            return "value"

        result = await Consensus.run([task, task, task], strategy="unanimous")
        assert result.analysis is not None
        assert result.analysis.total_outputs == 3
        assert result.analysis.successful_outputs == 3
        assert result.analysis.failed_outputs == 0
        assert result.analysis.strategy == "unanimous"

    @pytest.mark.asyncio
    async def test_result_has_outputs(self):
        async def task():
            return "value"

        result = await Consensus.run([task, task], strategy="best")
        assert len(result.outputs) == 2
        assert all(o.success for o in result.outputs)
        assert all(o.value == "value" for o in result.outputs)

    @pytest.mark.asyncio
    async def test_similarity_matrix_computed(self):
        async def task_a():
            return "hello world"

        async def task_b():
            return "hello there"

        result = await Consensus.run([task_a, task_b], strategy="best")
        matrix = result.analysis.similarity_matrix
        assert len(matrix) == 2
        assert matrix[0][0] == 1.0  # Same to itself
        assert matrix[1][1] == 1.0
        assert 0 < matrix[0][1] < 1.0  # Partially similar


class TestStructuredConsensus:
    @pytest.mark.asyncio
    async def test_structured_consensus_with_schema(self):
        class Person(BaseModel):
            name: str
            age: int

        async def task():
            return Person(name="Alice", age=30)

        result = await Consensus.run([task, task], strategy="unanimous", schema=Person)
        assert result.type == "structured"
        assert result.field_consensus is not None
        assert "name" in result.field_consensus.fields
        assert "age" in result.field_consensus.fields
        assert result.field_consensus.fields["name"].agreement == 1.0


class TestHelperFunctions:
    def test_quick_true(self):
        outputs = ["a", "a", "a", "b"]
        assert Consensus.quick(outputs, 0.7)  # 75% >= 70%

    def test_quick_false(self):
        outputs = ["a", "a", "b", "b"]
        assert not Consensus.quick(outputs, 0.8)  # 50% < 80%

    def test_quick_empty(self):
        assert not Consensus.quick([])

    def test_quick_with_dicts_different_order(self):
        """Test that dicts with same content but different insertion order are equal."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "a": 1}
        dict3 = {"a": 1, "b": 2}
        outputs = [dict1, dict2, dict3]
        assert Consensus.quick(outputs, 1.0)  # All should be considered equal

    def test_get_value(self):
        outputs = ["a", "a", "b"]
        assert Consensus.get_value(outputs) == "a"

    def test_get_value_empty(self):
        assert Consensus.get_value([]) is None

    def test_get_value_integers(self):
        outputs = [1, 2, 1, 1]
        assert Consensus.get_value(outputs) == 1

    def test_get_value_with_dicts_different_order(self):
        """Test that get_value handles dicts with different key ordering."""
        dict1 = {"x": 10, "y": 20}
        dict2 = {"y": 20, "x": 10}
        dict3 = {"x": 10, "y": 20}
        outputs = [dict1, dict2, dict3]
        result = Consensus.get_value(outputs)
        assert result == {"x": 10, "y": 20}

    @pytest.mark.asyncio
    async def test_validate_passes(self):
        async def task():
            return "value"

        result = await Consensus.run([task, task, task], strategy="unanimous")
        assert Consensus.validate(result, min_confidence=0.8)

    @pytest.mark.asyncio
    async def test_validate_fails_low_confidence(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        result = await Consensus.run(
            [task_a, task_a, task_b],
            strategy="majority",
            resolve_conflicts="vote",
        )
        # Confidence is ~0.67, so this should fail at 0.9 threshold
        assert not Consensus.validate(result, min_confidence=0.9)


class TestPresets:
    @pytest.mark.asyncio
    async def test_strict_requires_unanimous(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        # Strict should fail when outputs differ
        with pytest.raises(ValueError):
            await Consensus.strict([task_a, task_b])

    @pytest.mark.asyncio
    async def test_strict_succeeds_unanimous(self):
        async def task():
            return "same"

        result = await Consensus.strict([task, task, task])
        assert result.consensus == "same"
        assert result.confidence >= 0.99

    @pytest.mark.asyncio
    async def test_standard_majority_wins(self):
        async def task_a():
            return "a"

        async def task_b():
            return "b"

        result = await Consensus.standard([task_a, task_a, task_b])
        assert result.consensus == "a"


class TestObservabilityEvents:
    @pytest.mark.asyncio
    async def test_emits_events(self):
        events_received = []

        def on_event(event):
            events_received.append(event.type.value)

        async def task():
            return "value"

        await Consensus.run([task, task], strategy="unanimous", on_event=on_event)

        assert "CONSENSUS_START" in events_received
        assert "CONSENSUS_STREAM_START" in events_received
        assert "CONSENSUS_STREAM_END" in events_received
        assert "CONSENSUS_OUTPUT_COLLECTED" in events_received
        assert "CONSENSUS_ANALYSIS" in events_received
        assert "CONSENSUS_RESOLUTION" in events_received
        assert "CONSENSUS_END" in events_received
