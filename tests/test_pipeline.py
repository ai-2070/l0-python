"""Tests for l0.pipeline module."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from l0.adapters import AdaptedEvent, Adapters
from l0.pipeline import (
    PipelineStep,
    StepContext,
    StepResult,
    create_branch_step,
)
from l0.types import Event, EventType


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


def make_stream(content: str) -> Any:
    """Create a stream factory that yields the given content."""

    async def stream() -> AsyncIterator[Event]:
        for char in content:
            yield Event(type=EventType.TOKEN, text=char)
        yield Event(type=EventType.COMPLETE)

    return stream


class TestCreateBranchStep:
    """Tests for create_branch_step function."""

    @pytest.mark.asyncio
    async def test_branch_takes_true_path(self) -> None:
        """Test that branch takes if_true path when condition is true."""
        true_step = PipelineStep(
            name="true_step",
            fn=lambda input, ctx: make_stream("TRUE"),
        )
        false_step = PipelineStep(
            name="false_step",
            fn=lambda input, ctx: make_stream("FALSE"),
        )

        branch = create_branch_step(
            name="test_branch",
            condition=lambda input, ctx: True,
            if_true=true_step,
            if_false=false_step,
        )

        # Create a mock context
        context = StepContext(
            step_index=0,
            total_steps=1,
            previous_results=[],
            metadata={},
        )

        # Execute the branch function
        stream_factory = branch.fn("input", context)

        # Consume the stream
        content = ""
        async for event in stream_factory():
            if event.type == EventType.TOKEN:
                content += event.text

        assert content == "TRUE"

    @pytest.mark.asyncio
    async def test_branch_takes_false_path(self) -> None:
        """Test that branch takes if_false path when condition is false."""
        true_step = PipelineStep(
            name="true_step",
            fn=lambda input, ctx: make_stream("TRUE"),
        )
        false_step = PipelineStep(
            name="false_step",
            fn=lambda input, ctx: make_stream("FALSE"),
        )

        branch = create_branch_step(
            name="test_branch",
            condition=lambda input, ctx: False,
            if_true=true_step,
            if_false=false_step,
        )

        context = StepContext(
            step_index=0,
            total_steps=1,
            previous_results=[],
            metadata={},
        )

        stream_factory = branch.fn("input", context)

        content = ""
        async for event in stream_factory():
            if event.type == EventType.TOKEN:
                content += event.text

        assert content == "FALSE"

    def test_branch_transform_uses_correct_step(self) -> None:
        """Test that branch_transform uses the transform from the taken branch."""
        true_step = PipelineStep(
            name="true_step",
            fn=lambda input, ctx: make_stream("TRUE"),
            transform=lambda content, ctx: f"transformed_true:{content}",
        )
        false_step = PipelineStep(
            name="false_step",
            fn=lambda input, ctx: make_stream("FALSE"),
            transform=lambda content, ctx: f"transformed_false:{content}",
        )

        branch = create_branch_step(
            name="test_branch",
            condition=lambda input, ctx: True,
            if_true=true_step,
            if_false=false_step,
        )

        context = StepContext(
            step_index=0,
            total_steps=1,
            previous_results=[],
            metadata={},
        )

        # First call branch.fn to set which branch was taken
        branch.fn("input", context)

        # Then call transform - should use true_step's transform
        assert branch.transform is not None
        result = branch.transform("content", context)
        assert result == "transformed_true:content"

    def test_branch_transform_cleans_up_after_use(self) -> None:
        """Test that branch_taken dict is cleaned up after transform is called."""
        true_step = PipelineStep(
            name="true_step",
            fn=lambda input, ctx: make_stream("TRUE"),
            transform=lambda content, ctx: f"true:{content}",
        )
        false_step = PipelineStep(
            name="false_step",
            fn=lambda input, ctx: make_stream("FALSE"),
            transform=lambda content, ctx: f"false:{content}",
        )

        branch = create_branch_step(
            name="test_branch",
            condition=lambda input, ctx: True,
            if_true=true_step,
            if_false=false_step,
        )

        # Create multiple contexts and use them
        contexts = [
            StepContext(step_index=0, total_steps=1, previous_results=[], metadata={})
            for _ in range(100)
        ]

        # Process each context
        for ctx in contexts:
            branch.fn("input", ctx)
            assert branch.transform is not None
            branch.transform("content", ctx)

        # After processing, the internal dict should be empty
        # We can't directly access branch_taken, but we can verify
        # the transform still works (uses default if_true when not found)
        new_context = StepContext(
            step_index=0, total_steps=1, previous_results=[], metadata={}
        )
        # Don't call branch.fn first - should fall back to if_true's transform
        assert branch.transform is not None
        result = branch.transform("fallback", new_context)
        assert result == "true:fallback"  # Falls back to if_true

    def test_branch_transform_without_fn_call_uses_default(self) -> None:
        """Test that transform uses if_true when fn was never called for context."""
        true_step = PipelineStep(
            name="true_step",
            fn=lambda input, ctx: make_stream("TRUE"),
            transform=lambda content, ctx: "from_true",
        )
        false_step = PipelineStep(
            name="false_step",
            fn=lambda input, ctx: make_stream("FALSE"),
            transform=lambda content, ctx: "from_false",
        )

        branch = create_branch_step(
            name="test_branch",
            condition=lambda input, ctx: False,  # Would take false path
            if_true=true_step,
            if_false=false_step,
        )

        context = StepContext(
            step_index=0,
            total_steps=1,
            previous_results=[],
            metadata={},
        )

        # Call transform without calling fn first
        # Should fall back to if_true's transform
        assert branch.transform is not None
        result = branch.transform("content", context)
        assert result == "from_true"

    def test_branch_no_memory_leak_with_many_contexts(self) -> None:
        """Test that branch_taken dict doesn't leak memory with many contexts."""
        true_step = PipelineStep(
            name="true_step",
            fn=lambda input, ctx: make_stream("TRUE"),
            transform=lambda content, ctx: content.upper(),
        )
        false_step = PipelineStep(
            name="false_step",
            fn=lambda input, ctx: make_stream("FALSE"),
            transform=lambda content, ctx: content.lower(),
        )

        branch = create_branch_step(
            name="test_branch",
            condition=lambda input, ctx: input == "go_true",
            if_true=true_step,
            if_false=false_step,
        )

        # Process many contexts - alternating between true and false paths
        for i in range(1000):
            ctx = StepContext(
                step_index=0, total_steps=1, previous_results=[], metadata={}
            )
            input_val = "go_true" if i % 2 == 0 else "go_false"
            branch.fn(input_val, ctx)

            assert branch.transform is not None
            result = branch.transform("Test", ctx)

            if i % 2 == 0:
                assert result == "TEST"  # true path uppercases
            else:
                assert result == "test"  # false path lowercases

        # If there was a memory leak, we'd have 1000 entries in branch_taken
        # After calling transform, entries should be cleaned up
        # Verify by checking a new context still works correctly
        final_ctx = StepContext(
            step_index=0, total_steps=1, previous_results=[], metadata={}
        )
        branch.fn("go_false", final_ctx)
        assert branch.transform is not None
        result = branch.transform("Final", final_ctx)
        assert result == "final"


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_pipeline_step_creation(self) -> None:
        """Test creating a PipelineStep with required fields."""
        step: PipelineStep[str, str] = PipelineStep(
            name="test_step",
            fn=lambda input, ctx: make_stream(str(input)),
        )
        assert step.name == "test_step"
        assert step.transform is None
        assert step.condition is None

    def test_pipeline_step_with_transform(self) -> None:
        """Test PipelineStep with transform function."""
        step: PipelineStep[str, str] = PipelineStep(
            name="test_step",
            fn=lambda input, ctx: make_stream(str(input)),
            transform=lambda content, ctx: content.upper(),
        )

        ctx = StepContext(step_index=0, total_steps=1, previous_results=[], metadata={})
        assert step.transform is not None
        assert step.transform("hello", ctx) == "HELLO"

    def test_pipeline_step_with_condition(self) -> None:
        """Test PipelineStep with condition function."""
        step: PipelineStep[str, str] = PipelineStep(
            name="test_step",
            fn=lambda input, ctx: make_stream(str(input)),
            condition=lambda input, ctx: len(str(input)) > 5,
        )

        ctx = StepContext(step_index=0, total_steps=1, previous_results=[], metadata={})
        assert step.condition is not None
        assert step.condition("short", ctx) is False
        assert step.condition("longer_input", ctx) is True


class TestStepContext:
    """Tests for StepContext dataclass."""

    def test_step_context_creation(self) -> None:
        """Test creating a StepContext with required fields."""
        ctx = StepContext(
            step_index=2,
            total_steps=5,
            previous_results=[],
            metadata={"key": "value"},
        )
        assert ctx.step_index == 2
        assert ctx.total_steps == 5
        assert ctx.previous_results == []
        assert ctx.metadata == {"key": "value"}
        assert ctx.cancelled is False

    def test_step_context_cancelled_default(self) -> None:
        """Test that cancelled defaults to False."""
        ctx = StepContext(
            step_index=0,
            total_steps=1,
            previous_results=[],
            metadata={},
        )
        assert ctx.cancelled is False

    def test_step_context_with_previous_results(self) -> None:
        """Test StepContext with previous results."""
        prev_result = StepResult(
            step_name="prev_step",
            step_index=0,
            input="prev_input",
            output="prev_output",
            raw_content="raw",
            status="success",
        )

        ctx = StepContext(
            step_index=1,
            total_steps=2,
            previous_results=[prev_result],
            metadata={},
        )

        assert len(ctx.previous_results) == 1
        assert ctx.previous_results[0].step_name == "prev_step"
