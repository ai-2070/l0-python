"""Tests for l0.pipeline module."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from l0.adapters import AdaptedEvent, Adapters
from l0.pipeline import (
    FAST_PIPELINE,
    PRODUCTION_PIPELINE,
    RELIABLE_PIPELINE,
    Pipeline,
    PipelineOptions,
    PipelineResult,
    PipelineStep,
    StepContext,
    StepResult,
    chain_pipelines,
    create_branch_step,
    create_pipeline,
    create_step,
    parallel_pipelines,
    pipe,
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


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_creation(self) -> None:
        """Test creating StepResult with all fields."""
        result = StepResult(
            step_name="test_step",
            step_index=0,
            input="input_data",
            output="output_data",
            raw_content="raw content",
            status="success",
            error=None,
            duration=1500,
            startTime=1000000,
            endTime=1001500,
            token_count=10,
        )
        assert result.step_name == "test_step"
        assert result.step_index == 0
        assert result.input == "input_data"
        assert result.output == "output_data"
        assert result.raw_content == "raw content"
        assert result.status == "success"
        assert result.error is None
        assert result.duration == 1500
        assert result.token_count == 10

    def test_step_result_with_error(self) -> None:
        """Test StepResult with error state."""
        error = ValueError("test error")
        result = StepResult(
            step_name="failing_step",
            step_index=1,
            input="input",
            output=None,
            raw_content="",
            status="error",
            error=error,
        )
        assert result.status == "error"
        assert result.error == error
        assert result.output is None


class TestPipelineOptions:
    """Tests for PipelineOptions dataclass."""

    def test_default_options(self) -> None:
        """Test default PipelineOptions values."""
        opts = PipelineOptions()
        assert opts.name is None
        assert opts.stop_on_error is True
        assert opts.timeout is None
        assert opts.on_start is None
        assert opts.on_complete is None
        assert opts.on_error is None
        assert opts.on_progress is None
        assert opts.metadata == {}

    def test_options_with_callbacks(self) -> None:
        """Test PipelineOptions with callbacks."""
        on_start = MagicMock()
        on_complete = MagicMock()
        on_error = MagicMock()
        on_progress = MagicMock()

        opts = PipelineOptions(
            name="test_pipeline",
            stop_on_error=False,
            timeout=60.0,
            on_start=on_start,
            on_complete=on_complete,
            on_error=on_error,
            on_progress=on_progress,
            metadata={"key": "value"},
        )

        assert opts.name == "test_pipeline"
        assert opts.stop_on_error is False
        assert opts.timeout == 60.0
        assert opts.metadata == {"key": "value"}


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_success(self) -> None:
        """Test successful PipelineResult."""
        result = PipelineResult(
            name="test_pipeline",
            output="final output",
            steps=[],
            status="success",
            error=None,
            duration=2500,
            startTime=1000000,
            endTime=1002500,
            metadata={},
        )
        assert result.name == "test_pipeline"
        assert result.output == "final output"
        assert result.status == "success"
        assert result.error is None
        assert result.duration == 2500

    def test_pipeline_result_error(self) -> None:
        """Test PipelineResult with error."""
        error = RuntimeError("pipeline failed")
        result = PipelineResult(
            name="failing_pipeline",
            output=None,
            steps=[],
            status="error",
            error=error,
        )
        assert result.status == "error"
        assert result.error == error


class TestPipelineClass:
    """Tests for Pipeline class."""

    def test_pipeline_init_empty(self) -> None:
        """Test creating empty Pipeline."""
        pipeline = Pipeline()
        assert pipeline.steps == []
        assert pipeline.options is not None
        assert pipeline.name is None

    def test_pipeline_init_with_steps(self) -> None:
        """Test creating Pipeline with steps."""
        step = PipelineStep(
            name="step1",
            fn=lambda input, ctx: make_stream("output"),
        )
        pipeline = Pipeline(steps=[step])
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "step1"

    def test_pipeline_init_with_options(self) -> None:
        """Test creating Pipeline with options."""
        opts = PipelineOptions(name="my_pipeline", timeout=30.0)
        pipeline = Pipeline(options=opts)
        assert pipeline.name == "my_pipeline"
        assert pipeline.options.timeout == 30.0

    def test_add_step(self) -> None:
        """Test adding steps to Pipeline."""
        pipeline = Pipeline()
        step1 = PipelineStep(name="step1", fn=lambda i, c: make_stream("1"))
        step2 = PipelineStep(name="step2", fn=lambda i, c: make_stream("2"))

        result = pipeline.add_step(step1).add_step(step2)

        # Should return self for chaining
        assert result is pipeline
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == "step1"
        assert pipeline.steps[1].name == "step2"

    def test_remove_step(self) -> None:
        """Test removing steps from Pipeline."""
        step1 = PipelineStep(name="step1", fn=lambda i, c: make_stream("1"))
        step2 = PipelineStep(name="step2", fn=lambda i, c: make_stream("2"))
        pipeline = Pipeline(steps=[step1, step2])

        result = pipeline.remove_step("step1")

        assert result is pipeline
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "step2"

    def test_remove_nonexistent_step(self) -> None:
        """Test removing non-existent step is a no-op."""
        step = PipelineStep(name="step1", fn=lambda i, c: make_stream("1"))
        pipeline = Pipeline(steps=[step])

        pipeline.remove_step("nonexistent")

        assert len(pipeline.steps) == 1

    def test_get_step(self) -> None:
        """Test getting step by name."""
        step1 = PipelineStep(name="step1", fn=lambda i, c: make_stream("1"))
        step2 = PipelineStep(name="step2", fn=lambda i, c: make_stream("2"))
        pipeline = Pipeline(steps=[step1, step2])

        assert pipeline.get_step("step1") is step1
        assert pipeline.get_step("step2") is step2
        assert pipeline.get_step("nonexistent") is None

    def test_clone(self) -> None:
        """Test cloning Pipeline."""
        step = PipelineStep(
            name="step1",
            fn=lambda i, c: make_stream("1"),
            metadata={"key": "value"},
        )
        opts = PipelineOptions(
            name="original",
            timeout=30.0,
            metadata={"pipeline_key": "pipeline_value"},
        )
        original = Pipeline(steps=[step], options=opts)

        cloned = original.clone()

        # Should be different objects
        assert cloned is not original
        assert cloned.steps is not original.steps
        assert cloned.options is not original.options

        # But same values
        assert cloned.name == original.name
        assert len(cloned.steps) == len(original.steps)
        assert cloned.steps[0].name == original.steps[0].name
        assert cloned.options.timeout == original.options.timeout

        # Metadata should be copied
        assert cloned.steps[0].metadata == {"key": "value"}
        assert cloned.options.metadata == {"pipeline_key": "pipeline_value"}

        # Modifying clone shouldn't affect original
        cloned.steps[0].metadata["new"] = "data"
        assert "new" not in original.steps[0].metadata


class TestCreatePipeline:
    """Tests for create_pipeline factory function."""

    def test_create_pipeline_empty(self) -> None:
        """Test creating empty pipeline."""
        pipeline = create_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert pipeline.steps == []

    def test_create_pipeline_with_steps(self) -> None:
        """Test creating pipeline with steps."""
        step = PipelineStep(name="step1", fn=lambda i, c: make_stream("1"))
        pipeline = create_pipeline(steps=[step])
        assert len(pipeline.steps) == 1

    def test_create_pipeline_with_options(self) -> None:
        """Test creating pipeline with options."""
        opts = PipelineOptions(name="my_pipeline")
        pipeline = create_pipeline(options=opts)
        assert pipeline.name == "my_pipeline"


class TestCreateStep:
    """Tests for create_step helper function."""

    def test_create_step_basic(self) -> None:
        """Test creating a step from prompt function."""
        prompt_fn = lambda doc: f"Summarize: {doc}"
        stream_factory = lambda prompt: make_stream(prompt)()

        step = create_step("summarize", prompt_fn, stream_factory)

        assert step.name == "summarize"
        assert step.transform is None

    def test_create_step_with_transform(self) -> None:
        """Test creating step with transform function."""
        prompt_fn = lambda doc: f"Summarize: {doc}"
        stream_factory = lambda prompt: make_stream(prompt)()
        transform = lambda content, ctx: content.upper()

        step = create_step("summarize", prompt_fn, stream_factory, transform=transform)

        assert step.transform is not None
        ctx = StepContext(step_index=0, total_steps=1, previous_results=[], metadata={})
        assert step.transform("hello", ctx) == "HELLO"


class TestChainPipelines:
    """Tests for chain_pipelines function."""

    def test_chain_empty(self) -> None:
        """Test chaining no pipelines."""
        result = chain_pipelines()
        assert isinstance(result, Pipeline)
        assert result.steps == []
        assert result.name is None

    def test_chain_single_pipeline(self) -> None:
        """Test chaining single pipeline."""
        step = PipelineStep(name="step1", fn=lambda i, c: make_stream("1"))
        p1 = Pipeline(steps=[step], options=PipelineOptions(name="p1"))

        result = chain_pipelines(p1)

        assert len(result.steps) == 1
        assert result.name == "p1"

    def test_chain_multiple_pipelines(self) -> None:
        """Test chaining multiple pipelines."""
        step1 = PipelineStep(name="step1", fn=lambda i, c: make_stream("1"))
        step2 = PipelineStep(name="step2", fn=lambda i, c: make_stream("2"))
        step3 = PipelineStep(name="step3", fn=lambda i, c: make_stream("3"))

        p1 = Pipeline(steps=[step1], options=PipelineOptions(name="p1"))
        p2 = Pipeline(steps=[step2], options=PipelineOptions(name="p2"))
        p3 = Pipeline(steps=[step3], options=PipelineOptions(name="p3"))

        result = chain_pipelines(p1, p2, p3)

        assert len(result.steps) == 3
        assert result.steps[0].name == "step1"
        assert result.steps[1].name == "step2"
        assert result.steps[2].name == "step3"
        assert result.name == "p1 -> p2 -> p3"


class TestPipelinePresets:
    """Tests for pipeline preset configurations."""

    def test_fast_pipeline_preset(self) -> None:
        """Test FAST_PIPELINE preset."""
        assert FAST_PIPELINE.stop_on_error is True

    def test_reliable_pipeline_preset(self) -> None:
        """Test RELIABLE_PIPELINE preset."""
        assert RELIABLE_PIPELINE.stop_on_error is False

    def test_production_pipeline_preset(self) -> None:
        """Test PRODUCTION_PIPELINE preset."""
        assert PRODUCTION_PIPELINE.stop_on_error is False
        assert PRODUCTION_PIPELINE.timeout == 300.0


class TestPipeExecution:
    """Tests for pipe() function execution."""

    @pytest.mark.asyncio
    async def test_pipe_single_step(self) -> None:
        """Test pipe with single step."""
        step = PipelineStep(
            name="step1",
            fn=lambda input, ctx: make_stream("Hello"),
        )

        with patch("l0.pipeline._internal_run") as mock_run:
            # Mock the stream result
            mock_stream = MagicMock()
            mock_stream.__aiter__ = lambda self: iter(
                [
                    Event(type=EventType.TOKEN, text="Hello"),
                    Event(type=EventType.COMPLETE),
                ]
            ).__iter__()

            async def async_iter():
                yield Event(type=EventType.TOKEN, text="Hello")
                yield Event(type=EventType.COMPLETE)

            mock_stream.__aiter__ = lambda self: async_iter()
            mock_run.return_value = mock_stream

            result = await pipe([step], "input")

            assert result.status == "success"
            assert len(result.steps) == 1
            assert result.steps[0].step_name == "step1"

    @pytest.mark.asyncio
    async def test_pipe_with_skipped_step(self) -> None:
        """Test pipe with conditional step that gets skipped."""
        step = PipelineStep(
            name="conditional_step",
            fn=lambda input, ctx: make_stream("output"),
            condition=lambda input, ctx: False,  # Always skip
        )

        result = await pipe([step], "input")

        assert result.status == "success"
        assert len(result.steps) == 1
        assert result.steps[0].status == "skipped"
        assert result.steps[0].raw_content == ""

    @pytest.mark.asyncio
    async def test_pipe_with_callbacks(self) -> None:
        """Test pipe with lifecycle callbacks."""
        start_called = []
        complete_called = []
        progress_called = []

        opts = PipelineOptions(
            on_start=lambda input: start_called.append(input),
            on_complete=lambda result: complete_called.append(result),
            on_progress=lambda step, total: progress_called.append((step, total)),
        )

        step = PipelineStep(
            name="step1",
            fn=lambda input, ctx: make_stream("output"),
            condition=lambda input, ctx: False,  # Skip to avoid mocking _internal_run
        )

        await pipe([step], "test_input", opts)

        assert start_called == ["test_input"]
        assert len(complete_called) == 1
        assert progress_called == [(0, 1)]

    @pytest.mark.asyncio
    async def test_pipe_timeout(self) -> None:
        """Test pipe with timeout."""
        import asyncio

        async def slow_step(input: Any, ctx: StepContext) -> Any:
            await asyncio.sleep(10)
            return make_stream("output")

        step = PipelineStep(name="slow_step", fn=slow_step)
        opts = PipelineOptions(timeout=0.01)  # Very short timeout

        result = await pipe([step], "input", opts)

        assert result.status == "error"
        assert isinstance(result.error, TimeoutError)

    @pytest.mark.asyncio
    async def test_pipe_stop_on_error_false(self) -> None:
        """Test pipe continues on error when stop_on_error=False."""
        step1 = PipelineStep(
            name="failing_step",
            fn=lambda i, c: (_ for _ in ()).throw(ValueError("step1 failed")),
        )
        step2 = PipelineStep(
            name="step2",
            fn=lambda i, c: make_stream("step2 output"),
            condition=lambda i, c: False,  # Skip to avoid _internal_run
        )

        opts = PipelineOptions(stop_on_error=False)
        result = await pipe([step1, step2], "input", opts)

        assert result.status == "partial"
        assert len(result.steps) == 2
        assert result.steps[0].status == "error"
        assert result.steps[1].status == "skipped"


class TestParallelPipelines:
    """Tests for parallel_pipelines function."""

    @pytest.mark.asyncio
    async def test_parallel_pipelines_combiner(self) -> None:
        """Test running pipelines in parallel and combining results."""
        # Create mock pipelines that return immediately
        p1 = MagicMock(spec=Pipeline)
        p2 = MagicMock(spec=Pipeline)
        p3 = MagicMock(spec=Pipeline)

        # Mock run methods
        async def run1(input: Any) -> PipelineResult[str]:
            return PipelineResult(
                name="p1", output="result1", steps=[], status="success"
            )

        async def run2(input: Any) -> PipelineResult[str]:
            return PipelineResult(
                name="p2", output="result2", steps=[], status="success"
            )

        async def run3(input: Any) -> PipelineResult[str]:
            return PipelineResult(
                name="p3", output="result3", steps=[], status="success"
            )

        p1.run = run1
        p2.run = run2
        p3.run = run3

        def combiner(results: list[PipelineResult[str]]) -> dict[str, str | None]:
            return {f"p{i + 1}": r.output for i, r in enumerate(results)}

        result = await parallel_pipelines([p1, p2, p3], "input", combiner)

        assert result == {"p1": "result1", "p2": "result2", "p3": "result3"}
