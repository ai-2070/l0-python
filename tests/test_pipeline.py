"""Tests for L0 Pipeline API.

Tests for Pipeline class, pipe() function, and multi-step workflow execution.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.l0.adapters import AdaptedEvent, Adapters
from src.l0.pipeline import (
    FAST_PIPELINE,
    PRODUCTION_PIPELINE,
    RELIABLE_PIPELINE,
    Pipeline,
    PipelineOptions,
    PipelineResult,
    PipelineStep,
    StepContext,
    StepResult,
    pipe,
)
from src.l0.types import Event, EventType

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


def stream_factory(text: str):
    """Create a stream factory that returns a stream with the given text."""

    async def factory():
        yield Event(type=EventType.TOKEN, text=text)
        yield Event(type=EventType.COMPLETE)

    return factory


# ============================================================================
# StepContext Tests
# ============================================================================


class TestStepContext:
    """Tests for StepContext."""

    def test_create_context(self):
        """Should create a StepContext."""
        ctx = StepContext(
            step_index=0,
            total_steps=3,
            previous_results=[],
            metadata={"key": "value"},
        )

        assert ctx.step_index == 0
        assert ctx.total_steps == 3
        assert ctx.previous_results == []
        assert ctx.metadata == {"key": "value"}
        assert ctx.cancelled is False

    def test_context_cancelled_flag(self):
        """Should track cancelled state."""
        ctx = StepContext(
            step_index=0,
            total_steps=3,
            previous_results=[],
            metadata={},
            cancelled=True,
        )

        assert ctx.cancelled is True


# ============================================================================
# StepResult Tests
# ============================================================================


class TestStepResult:
    """Tests for StepResult."""

    def test_create_success_result(self):
        """Should create a success result."""
        result = StepResult(
            step_name="test",
            step_index=0,
            input="input",
            output="output",
            raw_content="raw",
            status="success",
            duration=1.5,
            token_count=10,
        )

        assert result.step_name == "test"
        assert result.step_index == 0
        assert result.input == "input"
        assert result.output == "output"
        assert result.raw_content == "raw"
        assert result.status == "success"
        assert result.error is None

    def test_create_error_result(self):
        """Should create an error result."""
        error = Exception("Test error")
        result = StepResult(
            step_name="test",
            step_index=0,
            input="input",
            output=None,
            raw_content="",
            status="error",
            error=error,
        )

        assert result.status == "error"
        assert result.error is error


# ============================================================================
# PipelineStep Tests
# ============================================================================


class TestPipelineStep:
    """Tests for PipelineStep."""

    def test_create_basic_step(self):
        """Should create a basic step."""
        step = PipelineStep(
            name="test-step",
            fn=lambda input, ctx: stream_factory(input),
        )

        assert step.name == "test-step"
        assert step.fn is not None
        assert step.transform is None
        assert step.condition is None

    def test_create_step_with_transform(self):
        """Should create a step with transform."""
        step = PipelineStep(
            name="transform-step",
            fn=lambda input, ctx: stream_factory(input),
            transform=lambda content, ctx: content.upper(),
        )

        assert step.transform is not None

    def test_create_step_with_condition(self):
        """Should create a step with condition."""
        step = PipelineStep(
            name="conditional-step",
            fn=lambda input, ctx: stream_factory(input),
            condition=lambda input, ctx: len(input) > 5,
        )

        assert step.condition is not None

    def test_create_step_with_callbacks(self):
        """Should create a step with callbacks."""
        on_error = MagicMock()
        on_complete = MagicMock()

        step = PipelineStep(
            name="callback-step",
            fn=lambda input, ctx: stream_factory(input),
            on_error=on_error,
            on_complete=on_complete,
        )

        assert step.on_error is on_error
        assert step.on_complete is on_complete


# ============================================================================
# PipelineOptions Tests
# ============================================================================


class TestPipelineOptions:
    """Tests for PipelineOptions."""

    def test_default_options(self):
        """Should have correct defaults."""
        options = PipelineOptions()

        assert options.name is None
        assert options.stop_on_error is True
        assert options.timeout is None

    def test_custom_options(self):
        """Should accept custom options."""
        options = PipelineOptions(
            name="test-pipeline",
            stop_on_error=False,
            timeout=30.0,
            metadata={"env": "test"},
        )

        assert options.name == "test-pipeline"
        assert options.stop_on_error is False
        assert options.timeout == 30.0
        assert options.metadata == {"env": "test"}


# ============================================================================
# PipelineResult Tests
# ============================================================================


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_success_result(self):
        """Should represent success result."""
        result = PipelineResult(
            name="test",
            output="final output",
            steps=[],
            status="success",
            duration=2.5,
            start_time=0,
            end_time=2.5,
        )

        assert result.status == "success"
        assert result.output == "final output"

    def test_error_result(self):
        """Should represent error result."""
        result = PipelineResult(
            name="test",
            output=None,
            steps=[],
            status="error",
            error=Exception("Failed"),
            duration=1.0,
            start_time=0,
            end_time=1.0,
        )

        assert result.status == "error"
        assert result.error is not None


# ============================================================================
# Pipeline Presets Tests
# ============================================================================


class TestPipelinePresets:
    """Tests for pipeline presets."""

    def test_fast_pipeline_preset(self):
        """Should have FAST_PIPELINE preset."""
        assert FAST_PIPELINE is not None
        assert FAST_PIPELINE.stop_on_error is True

    def test_production_pipeline_preset(self):
        """Should have PRODUCTION_PIPELINE preset."""
        assert PRODUCTION_PIPELINE is not None

    def test_reliable_pipeline_preset(self):
        """Should have RELIABLE_PIPELINE preset."""
        assert RELIABLE_PIPELINE is not None


# ============================================================================
# Pipeline Class Tests
# ============================================================================


class TestPipeline:
    """Tests for Pipeline class."""

    def test_create_empty_pipeline(self):
        """Should create an empty pipeline."""
        pipeline = Pipeline(steps=[])
        assert pipeline is not None

    def test_create_pipeline_with_steps(self):
        """Should create a pipeline with steps."""
        steps = [
            PipelineStep(name="step1", fn=lambda x, ctx: stream_factory(x)),
            PipelineStep(name="step2", fn=lambda x, ctx: stream_factory(x)),
        ]

        pipeline = Pipeline(steps=steps)
        assert len(pipeline.steps) == 2

    def test_create_pipeline_with_options(self):
        """Should create a pipeline with options."""
        options = PipelineOptions(name="test", stop_on_error=False)
        pipeline = Pipeline(steps=[], options=options)

        assert pipeline.options.name == "test"
        assert pipeline.options.stop_on_error is False

    def test_add_step(self):
        """Should add a step to pipeline."""
        pipeline = Pipeline(steps=[])

        step = PipelineStep(name="new-step", fn=lambda x, ctx: stream_factory(x))
        pipeline.add_step(step)

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "new-step"


# ============================================================================
# Pipe Function Tests
# ============================================================================


class TestPipeFunction:
    """Tests for pipe() function."""

    @pytest.mark.asyncio
    async def test_pipe_single_step(self):
        """Should execute single step pipeline."""

        async def step_fn(input_data: str, ctx: StepContext):
            async def stream():
                yield Event(type=EventType.TOKEN, text=f"processed: {input_data}")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [PipelineStep(name="process", fn=step_fn)]

        result = await pipe(
            input="hello",
            steps=steps,
        )

        assert result.status == "success"
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_pipe_multiple_steps(self):
        """Should execute multiple steps in order."""
        call_order: list[str] = []

        async def step1(input_data: str, ctx: StepContext):
            call_order.append("step1")

            async def stream():
                yield Event(type=EventType.TOKEN, text="step1-output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        async def step2(input_data: str, ctx: StepContext):
            call_order.append("step2")

            async def stream():
                yield Event(type=EventType.TOKEN, text="step2-output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(name="step1", fn=step1),
            PipelineStep(name="step2", fn=step2),
        ]

        result = await pipe(input="start", steps=steps)

        assert call_order == ["step1", "step2"]
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_pipe_with_transform(self):
        """Should apply transform to step output."""

        async def step_fn(input_data: str, ctx: StepContext):
            async def stream():
                yield Event(type=EventType.TOKEN, text="hello")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(
                name="uppercase",
                fn=step_fn,
                transform=lambda content, ctx: content.upper(),
            ),
        ]

        result = await pipe(input="test", steps=steps)

        assert result.status == "success"
        # The transform should have been applied
        assert result.steps[0].output == "HELLO"

    @pytest.mark.asyncio
    async def test_pipe_with_condition_true(self):
        """Should run step when condition is true."""
        ran = []

        async def step_fn(input_data: str, ctx: StepContext):
            ran.append(True)

            async def stream():
                yield Event(type=EventType.TOKEN, text="ran")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(
                name="conditional",
                fn=step_fn,
                condition=lambda input, ctx: True,
            ),
        ]

        result = await pipe(input="test", steps=steps)

        assert len(ran) == 1
        assert result.steps[0].status == "success"

    @pytest.mark.asyncio
    async def test_pipe_with_condition_false(self):
        """Should skip step when condition is false."""
        ran = []

        async def step_fn(input_data: str, ctx: StepContext):
            ran.append(True)

            async def stream():
                yield Event(type=EventType.TOKEN, text="ran")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(
                name="conditional",
                fn=step_fn,
                condition=lambda input, ctx: False,
            ),
        ]

        result = await pipe(input="test", steps=steps)

        assert len(ran) == 0
        assert result.steps[0].status == "skipped"

    @pytest.mark.asyncio
    async def test_pipe_passes_context(self):
        """Should pass correct context to steps."""
        received_ctx: list[StepContext] = []

        async def step_fn(input_data: str, ctx: StepContext):
            received_ctx.append(ctx)

            async def stream():
                yield Event(type=EventType.TOKEN, text="output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(name="step1", fn=step_fn),
            PipelineStep(name="step2", fn=step_fn),
        ]

        await pipe(input="test", steps=steps, metadata={"key": "value"})

        assert len(received_ctx) == 2

        # First step
        assert received_ctx[0].step_index == 0
        assert received_ctx[0].total_steps == 2
        assert len(received_ctx[0].previous_results) == 0
        assert received_ctx[0].metadata == {"key": "value"}

        # Second step
        assert received_ctx[1].step_index == 1
        assert len(received_ctx[1].previous_results) == 1

    @pytest.mark.asyncio
    async def test_pipe_on_progress_callback(self):
        """Should call on_progress callback."""
        progress_calls: list[tuple[int, int]] = []

        async def step_fn(input_data: str, ctx: StepContext):
            async def stream():
                yield Event(type=EventType.TOKEN, text="output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(name="step1", fn=step_fn),
            PipelineStep(name="step2", fn=step_fn),
        ]

        await pipe(
            input="test",
            steps=steps,
            on_progress=lambda idx, total: progress_calls.append((idx, total)),
        )

        assert (0, 2) in progress_calls
        assert (1, 2) in progress_calls

    @pytest.mark.asyncio
    async def test_pipe_on_complete_callback(self):
        """Should call on_complete callback."""
        on_complete = MagicMock()

        async def step_fn(input_data: str, ctx: StepContext):
            async def stream():
                yield Event(type=EventType.TOKEN, text="output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [PipelineStep(name="step1", fn=step_fn)]

        await pipe(input="test", steps=steps, on_complete=on_complete)

        on_complete.assert_called_once()
        result = on_complete.call_args[0][0]
        assert isinstance(result, PipelineResult)


# ============================================================================
# Pipeline Error Handling Tests
# ============================================================================


class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.mark.asyncio
    async def test_stop_on_error_default(self):
        """Should stop on error by default."""
        step2_ran = []

        async def failing_step(input_data: str, ctx: StepContext):
            async def stream():
                yield Event(type=EventType.ERROR, error=Exception("Step failed"))

            return stream()

        async def step2(input_data: str, ctx: StepContext):
            step2_ran.append(True)

            async def stream():
                yield Event(type=EventType.TOKEN, text="output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(name="failing", fn=failing_step),
            PipelineStep(name="step2", fn=step2),
        ]

        result = await pipe(input="test", steps=steps)

        assert result.status == "error"
        assert len(step2_ran) == 0

    @pytest.mark.asyncio
    async def test_continue_on_error(self):
        """Should continue on error when stop_on_error is False."""
        step2_ran = []

        async def failing_step(input_data: str, ctx: StepContext):
            async def stream():
                yield Event(type=EventType.ERROR, error=Exception("Step failed"))

            return stream()

        async def step2(input_data: str, ctx: StepContext):
            step2_ran.append(True)

            async def stream():
                yield Event(type=EventType.TOKEN, text="output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(name="failing", fn=failing_step),
            PipelineStep(name="step2", fn=step2),
        ]

        result = await pipe(input="test", steps=steps, stop_on_error=False)

        assert result.status == "partial"
        assert len(step2_ran) == 1

    @pytest.mark.asyncio
    async def test_step_on_error_callback(self):
        """Should call step on_error callback."""
        on_error = MagicMock()

        async def failing_step(input_data: str, ctx: StepContext):
            async def stream():
                yield Event(type=EventType.ERROR, error=Exception("Step failed"))

            return stream()

        steps = [
            PipelineStep(name="failing", fn=failing_step, on_error=on_error),
        ]

        await pipe(input="test", steps=steps)

        on_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_on_error_callback(self):
        """Should call pipeline on_error callback."""
        on_error = MagicMock()

        async def failing_step(input_data: str, ctx: StepContext):
            async def stream():
                yield Event(type=EventType.ERROR, error=Exception("Step failed"))

            return stream()

        steps = [PipelineStep(name="failing", fn=failing_step)]

        await pipe(input="test", steps=steps, on_error=on_error)

        on_error.assert_called_once()


# ============================================================================
# Pipeline Chaining Tests
# ============================================================================


class TestPipelineChaining:
    """Tests for pipeline output chaining."""

    @pytest.mark.asyncio
    async def test_chain_output_to_next_input(self):
        """Should chain step output to next step input."""
        received_inputs: list[str] = []

        async def step1(input_data: str, ctx: StepContext):
            received_inputs.append(input_data)

            async def stream():
                yield Event(type=EventType.TOKEN, text="step1-output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        async def step2(input_data: str, ctx: StepContext):
            received_inputs.append(input_data)

            async def stream():
                yield Event(type=EventType.TOKEN, text="step2-output")
                yield Event(type=EventType.COMPLETE)

            return stream()

        steps = [
            PipelineStep(name="step1", fn=step1),
            PipelineStep(name="step2", fn=step2),
        ]

        await pipe(input="initial", steps=steps)

        assert received_inputs[0] == "initial"
        assert received_inputs[1] == "step1-output"

    @pytest.mark.asyncio
    async def test_transform_affects_next_input(self):
        """Should pass transformed output to next step."""
        received_inputs: list[str] = []

        async def step1(input_data: str, ctx: StepContext):
            received_inputs.append(input_data)

            async def stream():
                yield Event(type=EventType.TOKEN, text="hello")
                yield Event(type=EventType.COMPLETE)

            return stream()

        async def step2(input_data: str, ctx: StepContext):
            received_inputs.append(input_data)

            async def stream():
                yield Event(type=EventType.TOKEN, text="done")
                yield Event(type=EventType.COMPLETE)

        steps = [
            PipelineStep(
                name="step1",
                fn=step1,
                transform=lambda content, ctx: content.upper(),
            ),
            PipelineStep(name="step2", fn=step2),
        ]

        await pipe(input="start", steps=steps)

        assert received_inputs[0] == "start"
        assert received_inputs[1] == "HELLO"
