"""Tests for l0.consensus module."""

from typing import Any

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
    async def test_unanimous_fails_when_task_errors(self):
        """Test that unanimous strategy doesn't report success when a task fails."""

        async def good_task():
            return "same"

        async def failing_task():
            raise RuntimeError("Task failed")

        with pytest.raises(ValueError, match="No unanimous consensus"):
            await Consensus.run(
                [good_task, good_task, failing_task],
                strategy="unanimous",
                resolve_conflicts="fail",
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
            await Consensus.run([task, task], strategy="invalid")  # type: ignore[arg-type]


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

    @pytest.mark.asyncio
    async def test_resolve_best_zero_weights_no_crash(self):
        """Test that best resolution doesn't crash with zero weights."""

        async def task_a():
            return "a"

        async def task_b():
            return "b"

        # All zero weights should not cause division by zero
        result = await Consensus.run(
            [task_a, task_b],
            strategy="majority",
            resolve_conflicts="best",
            weights=[0.0, 0.0],
            minimum_agreement=0.0,
        )
        # Should return a result without crashing
        assert result.consensus in ("a", "b")
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_resolve_merge_with_dicts(self):
        """Test that merge resolution properly merges dict values."""

        async def task_a():
            return {"name": "Alice", "age": 30}

        async def task_b():
            return {"name": "Bob", "city": "NYC"}

        # Use unanimous strategy to force conflict resolution when values differ
        result = await Consensus.run(
            [task_a, task_b],
            strategy="unanimous",
            resolve_conflicts="merge",
        )
        # Should merge dicts (later overrides earlier for same keys)
        assert isinstance(result.consensus, dict)
        assert result.consensus["city"] == "NYC"
        assert result.consensus["age"] == 30
        # "name" should be from task_b since it runs later
        assert result.consensus["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_resolve_merge_with_lists(self):
        """Test that merge resolution properly merges list values."""

        async def task_a():
            return [1, 2, 3]

        async def task_b():
            return [2, 3, 4]

        # Use unanimous strategy to force conflict resolution when values differ
        result = await Consensus.run(
            [task_a, task_b],
            strategy="unanimous",
            resolve_conflicts="merge",
        )
        # Should concatenate unique items
        assert isinstance(result.consensus, list)
        assert set(result.consensus) == {1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_resolve_merge_with_strings(self):
        """Test that merge resolution properly joins string values."""

        async def task_a():
            return "hello"

        async def task_b():
            return "world"

        # Use unanimous strategy to force conflict resolution when values differ
        result = await Consensus.run(
            [task_a, task_b],
            strategy="unanimous",
            resolve_conflicts="merge",
        )
        # Should join with " | "
        assert isinstance(result.consensus, str)
        assert "hello" in result.consensus
        assert "world" in result.consensus
        assert " | " in result.consensus

    @pytest.mark.asyncio
    async def test_resolve_merge_majority_with_dicts(self):
        """Test merge resolution with majority strategy when no majority exists."""

        async def task_a():
            return {"name": "Alice", "age": 30}

        async def task_b():
            return {"name": "Bob", "city": "NYC"}

        async def task_c():
            return {"name": "Charlie", "country": "USA"}

        # Three different values, no majority - should trigger merge
        result = await Consensus.run(
            [task_a, task_b, task_c],
            strategy="majority",
            resolve_conflicts="merge",
            minimum_agreement=0.5,  # No single value will reach 50%
        )
        assert isinstance(result.consensus, dict)
        # All keys should be merged
        assert "age" in result.consensus
        assert "city" in result.consensus
        assert "country" in result.consensus

    @pytest.mark.asyncio
    async def test_resolve_merge_majority_with_lists(self):
        """Test merge resolution with majority strategy for lists."""

        # Use very different lists that won't be grouped as similar
        async def task_a():
            return ["apple", "banana", "cherry"]

        async def task_b():
            return [100, 200, 300, 400, 500]

        async def task_c():
            return ["x"]

        result = await Consensus.run(
            [task_a, task_b, task_c],
            strategy="majority",
            resolve_conflicts="merge",
            minimum_agreement=0.5,
        )
        assert isinstance(result.consensus, list)
        # All unique items should be merged
        assert "apple" in result.consensus
        assert 100 in result.consensus
        assert "x" in result.consensus

    @pytest.mark.asyncio
    async def test_resolve_merge_majority_with_strings(self):
        """Test merge resolution with majority strategy for strings."""

        async def task_a():
            return "alpha"

        async def task_b():
            return "beta"

        async def task_c():
            return "gamma"

        result = await Consensus.run(
            [task_a, task_b, task_c],
            strategy="majority",
            resolve_conflicts="merge",
            minimum_agreement=0.5,
        )
        assert isinstance(result.consensus, str)
        assert "alpha" in result.consensus
        assert "beta" in result.consensus
        assert "gamma" in result.consensus

    @pytest.mark.asyncio
    async def test_resolve_merge_with_integers(self):
        """Test that merge resolution converts integers to strings and joins them."""

        async def task_a():
            return 42

        async def task_b():
            return 100

        # Use unanimous strategy to force conflict resolution
        result = await Consensus.run(
            [task_a, task_b],
            strategy="unanimous",
            resolve_conflicts="merge",
        )
        # Numbers fall through to string handling
        assert isinstance(result.consensus, str)
        assert "42" in result.consensus
        assert "100" in result.consensus
        assert " | " in result.consensus

    @pytest.mark.asyncio
    async def test_resolve_merge_with_floats(self):
        """Test that merge resolution converts floats to strings and joins them."""

        async def task_a():
            return 3.14

        async def task_b():
            return 2.71

        result = await Consensus.run(
            [task_a, task_b],
            strategy="unanimous",
            resolve_conflicts="merge",
        )
        # Numbers fall through to string handling
        assert isinstance(result.consensus, str)
        assert "3.14" in result.consensus
        assert "2.71" in result.consensus

    @pytest.mark.asyncio
    async def test_resolve_merge_majority_with_numbers(self):
        """Test merge resolution with majority strategy for numbers."""

        async def task_a():
            return 10

        async def task_b():
            return 20

        async def task_c():
            return 30

        result = await Consensus.run(
            [task_a, task_b, task_c],
            strategy="majority",
            resolve_conflicts="merge",
            minimum_agreement=0.5,
        )
        assert isinstance(result.consensus, str)
        assert "10" in result.consensus
        assert "20" in result.consensus
        assert "30" in result.consensus

    @pytest.mark.asyncio
    async def test_resolve_merge_with_small_integers(self):
        """Test merge with small integers like 1 and 2."""

        async def task_a():
            return 1

        async def task_b():
            return 2

        result = await Consensus.run(
            [task_a, task_b],
            strategy="unanimous",
            resolve_conflicts="merge",
        )
        assert isinstance(result.consensus, str)
        assert "1" in result.consensus
        assert "2" in result.consensus

    @pytest.mark.asyncio
    async def test_resolve_merge_majority_with_small_integers(self):
        """Test merge with small integers using majority strategy."""

        async def task_a():
            return 1

        async def task_b():
            return 2

        async def task_c():
            return 3

        result = await Consensus.run(
            [task_a, task_b, task_c],
            strategy="majority",
            resolve_conflicts="merge",
            minimum_agreement=0.5,
        )
        assert isinstance(result.consensus, str)
        assert "1" in result.consensus
        assert "2" in result.consensus
        assert "3" in result.consensus


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

    def test_quick_with_nested_dicts_different_order(self):
        """Test that nested dicts with same content but different key order are equal."""
        dict1 = {"outer": {"a": 1, "b": 2}, "other": 3}
        dict2 = {"other": 3, "outer": {"b": 2, "a": 1}}
        dict3 = {"outer": {"a": 1, "b": 2}, "other": 3}
        outputs = [dict1, dict2, dict3]
        assert Consensus.quick(outputs, 1.0)  # All should be considered equal

    def test_quick_with_deeply_nested_dicts(self):
        """Test that deeply nested dicts normalize correctly."""
        dict1 = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
        dict2 = {"l1": {"l2": {"l3": {"b": 2, "a": 1}}}}
        outputs = [dict1, dict2]
        assert Consensus.quick(outputs, 1.0)

    def test_quick_with_nested_dicts_in_lists(self):
        """Test that dicts nested in lists normalize correctly."""
        dict1 = {"items": [{"a": 1, "b": 2}, {"c": 3, "d": 4}]}
        dict2 = {"items": [{"b": 2, "a": 1}, {"d": 4, "c": 3}]}
        outputs = [dict1, dict2]
        assert Consensus.quick(outputs, 1.0)

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

    def test_dict_does_not_collide_with_string(self):
        """Test that a dict doesn't collide with a string that looks like it."""
        dict_val = {"key": "value"}
        string_val = "{'key': 'value'}"
        outputs = [dict_val, dict_val, string_val]
        # The two dicts should win, not be confused with the string
        result = Consensus.get_value(outputs)
        assert result == {"key": "value"}
        assert isinstance(result, dict)

    def test_nested_dict_does_not_collide_with_string(self):
        """Test that nested dicts don't collide with string representations."""
        nested = {"outer": {"inner": 1}}
        string_repr = str(nested)
        outputs = [nested, nested, string_repr]
        result = Consensus.get_value(outputs)
        assert result == {"outer": {"inner": 1}}
        assert isinstance(result, dict)

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

        def on_event(event: Any) -> None:
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
