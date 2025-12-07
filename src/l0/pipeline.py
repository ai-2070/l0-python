"""L0 Pipeline API - Multi-phase streaming workflows.

Provides a way to chain multiple LLM operations together, where each step
receives the output of the previous step and can transform it before
passing to the next step.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, cast

from .runtime import _internal_run
from .types import Stream, StreamFactory

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass
class StepContext:
    """Context passed to each pipeline step."""

    step_index: int
    """Current step index (0-based)."""

    total_steps: int
    """Total number of steps in the pipeline."""

    previous_results: list["StepResult"]
    """Results from all previous steps."""

    metadata: dict[str, Any]
    """Pipeline-wide metadata."""

    cancelled: bool = False
    """Whether the pipeline has been cancelled."""


@dataclass
class StepResult(Generic[TOutput]):
    """Result from a single pipeline step."""

    step_name: str
    """Name of the step."""

    step_index: int
    """Index of the step (0-based)."""

    input: Any
    """Input passed to this step."""

    output: TOutput | None
    """Output from this step (transformed or raw content)."""

    raw_content: str
    """Raw content from the L0 stream."""

    status: str  # "success" | "error" | "skipped"
    """Execution status."""

    error: Exception | None = None
    """Error if step failed."""

    duration: int = 0
    """Step duration in milliseconds."""

    startTime: int = 0
    """Timestamp when step started (milliseconds)."""

    endTime: int = 0
    """Timestamp when step ended (milliseconds)."""

    token_count: int = 0
    """Number of tokens generated."""


@dataclass
class PipelineStep(Generic[TInput, TOutput]):
    """Configuration for a single pipeline step."""

    name: str
    """Step name (for logging/debugging)."""

    fn: Callable[[TInput, StepContext], StreamFactory | Any]
    """Step function that takes input and returns a stream factory or L0 options."""

    transform: Callable[[str, StepContext], TOutput] | None = None
    """Optional transform function to process content before next step."""

    condition: Callable[[TInput, StepContext], bool] | None = None
    """Optional condition to determine if step should run."""

    on_error: Callable[[Exception, StepContext], None] | None = None
    """Optional error handler for this step."""

    on_complete: Callable[[StepResult[TOutput], StepContext], None] | None = None
    """Optional callback when step completes."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Step-specific metadata."""


@dataclass
class PipelineOptions:
    """Pipeline configuration options."""

    name: str | None = None
    """Pipeline name (for logging/debugging)."""

    stop_on_error: bool = True
    """Stop execution on first error (default: True)."""

    timeout: float | None = None
    """Maximum execution time for entire pipeline in seconds."""

    on_start: Callable[[Any], None] | None = None
    """Callback when pipeline starts."""

    on_complete: Callable[["PipelineResult"], None] | None = None
    """Callback when pipeline completes."""

    on_error: Callable[[Exception, int], None] | None = None
    """Callback when pipeline errors (error, step_index)."""

    on_progress: Callable[[int, int], None] | None = None
    """Callback for step progress (step_index, total_steps)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Pipeline-wide metadata."""


@dataclass
class PipelineResult(Generic[TOutput]):
    """Result from pipeline execution."""

    name: str | None
    """Pipeline name."""

    output: TOutput | None
    """Final output from last step."""

    steps: list[StepResult]
    """Results from all steps."""

    status: str  # "success" | "error" | "partial"
    """Overall execution status."""

    error: Exception | None = None
    """Error if pipeline failed."""

    duration: int = 0
    """Total duration in seconds."""

    startTime: int = 0
    """Timestamp when pipeline started."""

    endTime: int = 0
    """Timestamp when pipeline ended."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Pipeline metadata."""


async def pipe(
    steps: list[PipelineStep],
    input: Any,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    """Execute a pipeline of streaming steps.

    Each step receives the output of the previous step and can transform it
    before passing to the next step. Guardrails can be applied between steps.

    Args:
        steps: Array of pipeline steps
        input: Initial input to the first step
        options: Pipeline options

    Returns:
        Pipeline result with all step results

    Example:
        ```python
        from openai import AsyncOpenAI
        import l0

        client = AsyncOpenAI()

        # Step functions can return:
        # 1. A dict with "stream" key (TypeScript-style)
        # 2. A stream factory (callable)
        # 3. An async generator directly (most Pythonic)

        async def summarize_step(text: str, ctx: l0.StepContext):
            # Return dict with stream factory (TypeScript-style)
            return {
                "stream": lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"Summarize: {text}"}],
                    stream=True,
                )
            }

        async def refine_step(summary: str, ctx: l0.StepContext):
            # Or return a stream factory directly
            return lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Refine: {summary}"}],
                stream=True,
            )

        result = await l0.pipe(
            [
                l0.PipelineStep(name="summarize", fn=summarize_step),
                l0.PipelineStep(name="refine", fn=refine_step),
            ],
            long_document,
            l0.PipelineOptions(name="summarize-refine"),
        )

        print(result.output)  # Final refined summary
        ```
    """
    opts = options or PipelineOptions()
    start_time = time.time()
    step_results: list[StepResult] = []
    current_input: Any = input
    final_output: Any = input
    pipeline_error: Exception | None = None
    pipeline_status = "success"

    # Create cancellation flag
    cancelled = False

    async def run_pipeline() -> None:
        nonlocal current_input, final_output, pipeline_error, pipeline_status, cancelled

        # Call on_start callback
        if opts.on_start:
            _maybe_await(opts.on_start(input))

        # Execute each step
        for i, step in enumerate(steps):
            if cancelled:
                break

            step_start_time = time.time()

            # Build step context
            context = StepContext(
                step_index=i,
                total_steps=len(steps),
                previous_results=step_results.copy(),
                metadata=opts.metadata.copy(),
                cancelled=cancelled,
            )

            # Call progress callback
            if opts.on_progress:
                _maybe_await(opts.on_progress(i, len(steps)))

            # Check step condition
            if step.condition:
                should_run = step.condition(current_input, context)
                if asyncio.iscoroutine(should_run):
                    should_run = await should_run
                if not should_run:
                    step_results.append(
                        StepResult(
                            step_name=step.name,
                            step_index=i,
                            input=current_input,
                            output=current_input,
                            raw_content="",
                            status="skipped",
                            duration=int((time.time() - step_start_time) * 1000),
                            startTime=int(step_start_time * 1000),
                            endTime=int(time.time() * 1000),
                        )
                    )
                    continue

            try:
                # Get stream factory from step function
                step_result_or_factory = step.fn(current_input, context)
                if asyncio.iscoroutine(step_result_or_factory):
                    step_result_or_factory = await step_result_or_factory

                # Handle different return types:
                # 1. Dict with "stream" key (TypeScript-style): {"stream": factory}
                # 2. Direct stream factory (callable)
                # 3. Direct async generator (most Pythonic)
                stream_factory: StreamFactory
                if isinstance(step_result_or_factory, dict):
                    if "stream" in step_result_or_factory:
                        stream_factory = step_result_or_factory["stream"]
                    else:
                        raise ValueError(
                            f"Step {step.name} returned dict without 'stream' key"
                        )
                elif callable(step_result_or_factory):
                    stream_factory = cast(StreamFactory, step_result_or_factory)
                else:
                    raise TypeError(
                        f"Step {step.name} must return a stream factory or dict with 'stream' key"
                    )

                # Execute L0
                result: Stream = await _internal_run(stream=stream_factory)

                # Consume stream and get content
                content = ""
                token_count = 0
                async for event in result:
                    if event.is_token and event.text:
                        content += event.text
                        token_count += 1

                # Transform output if transform function provided
                step_output: Any
                if step.transform:
                    step_output = step.transform(content, context)
                    if asyncio.iscoroutine(step_output):
                        step_output = await step_output
                else:
                    step_output = content

                step_result: StepResult = StepResult(
                    step_name=step.name,
                    step_index=i,
                    input=current_input,
                    output=step_output,
                    raw_content=content,
                    status="success",
                    duration=int((time.time() - step_start_time) * 1000),
                    startTime=int(step_start_time * 1000),
                    endTime=int(time.time() * 1000),
                    token_count=token_count,
                )

                step_results.append(step_result)

                # Call step on_complete callback
                if step.on_complete:
                    _maybe_await(step.on_complete(step_result, context))

                # Update current input for next step
                current_input = step_output
                final_output = step_output

            except Exception as e:
                step_result = StepResult(
                    step_name=step.name,
                    step_index=i,
                    input=current_input,
                    output=None,
                    raw_content="",
                    status="error",
                    error=e,
                    duration=int((time.time() - step_start_time) * 1000),
                    startTime=int(step_start_time * 1000),
                    endTime=int(time.time() * 1000),
                )

                step_results.append(step_result)

                # Call step on_error callback
                if step.on_error:
                    _maybe_await(step.on_error(e, context))

                # Call pipeline on_error callback
                if opts.on_error:
                    _maybe_await(opts.on_error(e, i))

                if opts.stop_on_error:
                    pipeline_error = e
                    pipeline_status = "error"
                    break
                else:
                    pipeline_status = "partial"

    try:
        if opts.timeout:
            await asyncio.wait_for(run_pipeline(), timeout=opts.timeout)
        else:
            await run_pipeline()
    except asyncio.TimeoutError:
        pipeline_error = TimeoutError(f"Pipeline timeout after {opts.timeout}s")
        pipeline_status = "error"
    except Exception as e:
        pipeline_error = e
        pipeline_status = "error"

    result = PipelineResult(
        name=opts.name,
        output=final_output,
        steps=step_results,
        status=pipeline_status,
        error=pipeline_error,
        duration=int((time.time() - start_time) * 1000),
        startTime=int(start_time * 1000),
        endTime=int(time.time() * 1000),
        metadata=opts.metadata,
    )

    # Call on_complete callback
    if opts.on_complete:
        _maybe_await(opts.on_complete(result))

    return result


class Pipeline(Generic[TInput, TOutput]):
    """Reusable pipeline with chainable configuration."""

    def __init__(
        self,
        steps: list[PipelineStep] | None = None,
        options: PipelineOptions | None = None,
    ) -> None:
        """Create a new pipeline.

        Args:
            steps: Initial pipeline steps
            options: Default pipeline options
        """
        self._steps: list[PipelineStep] = list(steps) if steps else []
        self._options = options or PipelineOptions()

    @property
    def name(self) -> str | None:
        """Pipeline name."""
        return self._options.name

    @property
    def steps(self) -> list[PipelineStep]:
        """Pipeline steps."""
        return self._steps

    @property
    def options(self) -> PipelineOptions:
        """Pipeline options."""
        return self._options

    async def run(self, input: TInput) -> PipelineResult[TOutput]:
        """Execute pipeline with input.

        Args:
            input: Input to the first step

        Returns:
            Pipeline result with all step results
        """
        return await pipe(self._steps, input, self._options)

    def add_step(self, step: PipelineStep) -> "Pipeline[TInput, TOutput]":
        """Add a step to the pipeline.

        Args:
            step: Step to add

        Returns:
            Self for chaining
        """
        self._steps.append(step)
        return self

    def remove_step(self, name: str) -> "Pipeline[TInput, TOutput]":
        """Remove a step by name.

        Args:
            name: Name of step to remove

        Returns:
            Self for chaining
        """
        self._steps = [s for s in self._steps if s.name != name]
        return self

    def get_step(self, name: str) -> PipelineStep | None:
        """Get step by name.

        Args:
            name: Name of step to find

        Returns:
            Step if found, None otherwise
        """
        for step in self._steps:
            if step.name == name:
                return step
        return None

    def clone(self) -> "Pipeline[TInput, TOutput]":
        """Clone pipeline.

        Returns:
            New pipeline with same configuration
        """
        return Pipeline(
            steps=[
                PipelineStep(
                    name=s.name,
                    fn=s.fn,
                    transform=s.transform,
                    condition=s.condition,
                    on_error=s.on_error,
                    on_complete=s.on_complete,
                    metadata=s.metadata.copy(),
                )
                for s in self._steps
            ],
            options=PipelineOptions(
                name=self._options.name,
                stop_on_error=self._options.stop_on_error,
                timeout=self._options.timeout,
                on_start=self._options.on_start,
                on_complete=self._options.on_complete,
                on_error=self._options.on_error,
                on_progress=self._options.on_progress,
                metadata=self._options.metadata.copy(),
            ),
        )


def create_pipeline(
    steps: list[PipelineStep] | None = None,
    options: PipelineOptions | None = None,
) -> Pipeline:
    """Create a reusable pipeline.

    Args:
        steps: Pipeline steps
        options: Default pipeline options

    Returns:
        Pipeline object with run method

    Example:
        ```python
        summarize_pipeline = l0.create_pipeline(
            [
                l0.PipelineStep(name="extract", fn=extract_step),
                l0.PipelineStep(name="summarize", fn=summarize_step),
                l0.PipelineStep(name="format", fn=format_step),
            ],
            l0.PipelineOptions(name="document-summarizer"),
        )

        result = await summarize_pipeline.run(document)
        ```
    """
    return Pipeline(steps=steps, options=options)


def create_step(
    name: str,
    prompt_fn: Callable[[Any], str],
    stream_factory: Callable[[str], Any],
    transform: Callable[[str, StepContext], Any] | None = None,
) -> PipelineStep:
    """Create a simple step from a prompt template.

    Args:
        name: Step name
        prompt_fn: Function that generates prompt from input
        stream_factory: Function that creates the stream from prompt
        transform: Optional transform function

    Returns:
        Pipeline step

    Example:
        ```python
        summarize_step = l0.create_step(
            "summarize",
            lambda doc: f"Summarize: {doc}",
            lambda prompt: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ),
        )
        ```
    """
    return PipelineStep(
        name=name,
        fn=lambda input, ctx: lambda: stream_factory(prompt_fn(input)),
        transform=transform,
    )


def chain_pipelines(*pipelines: Pipeline) -> Pipeline:
    """Chain multiple pipelines together.

    Args:
        pipelines: Pipelines to chain

    Returns:
        Combined pipeline

    Example:
        ```python
        full_pipeline = l0.chain_pipelines(
            extract_pipeline,
            analyze_pipeline,
            format_pipeline,
        )
        ```
    """
    all_steps: list[PipelineStep] = []
    names: list[str] = []

    for p in pipelines:
        all_steps.extend(p.steps)
        if p.name:
            names.append(p.name)

    return Pipeline(
        steps=all_steps,
        options=PipelineOptions(name=" -> ".join(names) if names else None),
    )


async def parallel_pipelines(
    pipelines: list[Pipeline],
    input: Any,
    combiner: Callable[[list[PipelineResult]], Any],
) -> Any:
    """Run pipelines in parallel and combine results.

    Args:
        pipelines: Pipelines to run
        input: Input for all pipelines
        combiner: Function to combine results

    Returns:
        Combined output

    Example:
        ```python
        results = await l0.parallel_pipelines(
            [sentiment_pipeline, entity_pipeline, summary_pipeline],
            document,
            lambda results: {
                "sentiment": results[0].output,
                "entities": results[1].output,
                "summary": results[2].output,
            },
        )
        ```
    """
    results = await asyncio.gather(*[p.run(input) for p in pipelines])
    return combiner(list(results))


def create_branch_step(
    name: str,
    condition: Callable[[Any, StepContext], bool],
    if_true: PipelineStep,
    if_false: PipelineStep,
) -> PipelineStep:
    """Create a conditional branch step.

    Args:
        name: Step name
        condition: Condition function
        if_true: Step to run if condition is true
        if_false: Step to run if condition is false

    Returns:
        Pipeline step

    Example:
        ```python
        branch_step = l0.create_branch_step(
            "route",
            lambda input, ctx: len(input) > 1000,
            summarize_step,
            pass_through_step,
        )
        ```
    """
    # Track which branch was taken per context
    branch_taken: dict[int, PipelineStep] = {}

    def branch_fn(input: Any, context: StepContext) -> Any:
        result = condition(input, context)
        step = if_true if result else if_false
        branch_taken[id(context)] = step
        return step.fn(input, context)

    def branch_transform(content: str, context: StepContext) -> Any:
        context_id = id(context)
        step = branch_taken.pop(context_id, if_true)
        if step.transform:
            return step.transform(content, context)
        return content

    return PipelineStep(
        name=name,
        fn=branch_fn,
        transform=branch_transform,
    )


def _maybe_await(value: Any) -> Any:
    """Helper to handle both sync and async callbacks."""
    if asyncio.iscoroutine(value):
        # Schedule as task to avoid blocking
        asyncio.create_task(value)
    return value


# Preset pipeline configurations
FAST_PIPELINE = PipelineOptions(
    stop_on_error=True,
)
"""Fast pipeline - minimal config, fail fast."""

RELIABLE_PIPELINE = PipelineOptions(
    stop_on_error=False,
)
"""Reliable pipeline - graceful failures."""

PRODUCTION_PIPELINE = PipelineOptions(
    stop_on_error=False,
    timeout=300.0,  # 5 minutes
)
"""Production pipeline - timeouts, graceful failures."""
