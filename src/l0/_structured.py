"""Structured output with Pydantic validation."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ValidationError, create_model

from ._utils import (
    AutoCorrectResult,
    CorrectionType,
    auto_correct_json,
    extract_json,
    extract_json_from_markdown,
    is_valid_json,
)
from .adapters import Adapter
from .events import EventBus, ObservabilityEvent, ObservabilityEventType
from .runtime import _internal_run
from .types import (
    AwaitableStreamFactory,
    AwaitableStreamSource,
    Event,
    RawStream,
    Retry,
    State,
    StreamFactory,
    Timeout,
)

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)
ResultT = TypeVar("ResultT")  # For StructuredResult, not bound to BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StructuredState:
    """Extended state for structured output with validation metrics.

    Attributes:
        validation_failures: Number of validation failures
        auto_corrections: Number of auto-corrections applied
        validation_errors: List of validation errors encountered
        correction_types: Types of corrections applied
        validation_time_ms: Time spent on validation (milliseconds)
    """

    validation_failures: int = 0
    auto_corrections: int = 0
    validation_errors: list[ValidationError] = field(default_factory=list)
    correction_types: list[str] = field(default_factory=list)
    validation_time_ms: float | None = None


@dataclass
class StructuredTelemetry:
    """Telemetry data for structured output.

    Attributes:
        schema_name: Name of the schema used
        validation_attempts: Number of validation attempts
        validation_failures: Number of validation failures
        auto_corrections: Number of auto-corrections applied
        correction_types: Types of corrections applied
        validation_success: Whether validation ultimately succeeded
        validation_time_ms: Time spent on validation (milliseconds)
    """

    schema_name: str | None = None
    validation_attempts: int = 0
    validation_failures: int = 0
    auto_corrections: int = 0
    correction_types: list[str] = field(default_factory=list)
    validation_success: bool = False
    validation_time_ms: float | None = None


@dataclass
class StructuredResult(Generic[ResultT]):
    """Result of structured output extraction.

    Attributes:
        data: Validated Pydantic model instance (or list of models for array results)
        raw: Raw JSON string before parsing
        corrected: Whether auto-correction was applied
        corrections: List of corrections applied
        state: L0 runtime state (token counts, retries, etc.)
        structured_state: Structured-specific state with validation metrics
        telemetry: Telemetry data (if monitoring enabled)
        errors: List of errors encountered during retries
    """

    data: ResultT
    raw: str
    corrected: bool = False
    corrections: list[str] = field(default_factory=list)
    state: State | None = None
    _aborted: bool = False

    def abort(self) -> None:
        """Abort the structured stream.

        Signals that the stream should stop processing.
        Matches TypeScript structured result API.
        """
        self._aborted = True

    @property
    def is_aborted(self) -> bool:
        """Check if abort was requested."""
        return self._aborted

    structured_state: StructuredState | None = None
    telemetry: StructuredTelemetry | None = None
    errors: list[Exception] = field(default_factory=list)


@dataclass
class AutoCorrectInfo:
    """Information passed to on_auto_correct callback."""

    original: str
    corrected: str
    corrections: list[str]
    success: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────


async def structured(
    schema: type[T],
    stream: AwaitableStreamSource,
    *,
    fallbacks: list[AwaitableStreamSource] | None = None,
    auto_correct: bool = True,
    strict_mode: bool = False,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    detect_zero_tokens: bool = False,
    monitoring: bool = False,
    on_validation_error: Callable[[ValidationError, int], None] | None = None,
    on_auto_correct: Callable[[AutoCorrectInfo], None] | None = None,
    on_retry: Callable[[int, str], None] | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    adapter: Adapter | str | None = None,
) -> StructuredResult[T]:
    """Get structured output validated against Pydantic schema.

    Args:
        schema: Pydantic model class to validate against
        stream: Async LLM stream or factory function that returns one
        fallbacks: Optional fallback streams to try if primary fails
        auto_correct: Whether to attempt JSON auto-correction (default: True)
        strict_mode: Reject unknown fields in output (default: False)
        retry: Retry configuration for validation failures
        timeout: Timeout configuration (initial_token, inter_token)
        detect_zero_tokens: Detect zero-token outputs (default: False for structured)
        monitoring: Enable telemetry collection (default: False)
        on_validation_error: Callback when validation fails (error, attempt)
        on_auto_correct: Callback when auto-correction is applied
        on_retry: Callback when retry occurs (attempt, reason)
        on_event: Optional callback for observability events
        adapter: Optional adapter hint ("openai", "litellm", or Adapter instance)

    Returns:
        StructuredResult with validated data and metadata

    Raises:
        ValueError: If schema validation fails after all retries

    Example:
        ```python
        from pydantic import BaseModel
        import l0

        class User(BaseModel):
            name: str
            age: int

        result = await l0.structured(
            schema=User,
            stream=openai_stream,
            auto_correct=True,
            retry=l0.Retry(attempts=3),
        )

        print(result.data.name)  # Type-safe access
        print(result.corrected)  # Was auto-correction applied?
        ```
    """
    event_bus = EventBus(on_event)
    retry_config = retry or Retry(attempts=1)
    max_attempts = retry_config.attempts if retry_config.attempts is not None else 1

    # Track structured-specific state
    validation_attempts = 0
    validation_failures = 0
    auto_corrections = 0
    correction_types: list[str] = []
    validation_errors: list[ValidationError] = []
    errors: list[Exception] = []
    validation_start_time = 0.0
    validation_end_time = 0.0

    # Helper to check if something is a direct async iterator (not a factory)
    def _is_async_iterator(obj: Any) -> bool:
        return hasattr(obj, "__anext__") and not callable(obj)

    # Helper to wrap a direct async iterator in a buffering factory
    # This consumes the iterator once and replays from buffer on subsequent calls
    def _make_buffering_factory(
        iterator: RawStream,
    ) -> StreamFactory:
        buffer: list[Any] = []
        consumed = False

        async def buffering_iterator() -> RawStream:
            nonlocal consumed
            if consumed:
                # Replay from buffer
                for item in buffer:
                    yield item
            else:
                # First consumption - buffer and yield
                async for item in iterator:
                    buffer.append(item)
                    yield item
                consumed = True

        return buffering_iterator

    # Wrap direct async iterators in buffering factories for retry support
    if _is_async_iterator(stream):
        stream = _make_buffering_factory(cast(RawStream, stream))

    # Build list of streams to try
    all_streams: list[AwaitableStreamSource] = [stream]
    if fallbacks:
        wrapped_fallbacks: list[AwaitableStreamFactory] = []
        for fb in fallbacks:
            if _is_async_iterator(fb):
                wrapped_fallbacks.append(_make_buffering_factory(cast(RawStream, fb)))
            else:
                wrapped_fallbacks.append(cast(AwaitableStreamFactory, fb))
        all_streams.extend(wrapped_fallbacks)

    last_error: Exception | None = None
    fallback_index = 0

    for stream_source in all_streams:
        for attempt in range(max_attempts):
            try:
                # _internal_run expects a callable factory
                # Handle both direct async iterators and factory functions
                def make_stream_factory(
                    src: AwaitableStreamSource,
                ) -> AwaitableStreamFactory:
                    if callable(src) and not hasattr(src, "__anext__"):
                        # It's already a factory
                        return src
                    else:
                        # It's a direct async iterator - wrap in factory
                        # Note: This only works once per stream!
                        return lambda: cast(RawStream, src)

                stream_factory = make_stream_factory(stream_source)

                # Run through L0 runtime
                result = await _internal_run(
                    stream=stream_factory,
                    on_event=on_event,
                    adapter=adapter,
                    timeout=timeout,
                )
                text = await result.read()
                state = result.state

                # Check for zero-token output
                if detect_zero_tokens and (not text or text.strip() == ""):
                    raise ValueError("Zero-token output detected")

                # Extract and validate
                validation_start_time = time.time()
                validation_attempts += 1

                validated = _parse_and_validate(
                    text=text,
                    schema=schema,
                    auto_correct=auto_correct,
                    strict_mode=strict_mode,
                    on_auto_correct=on_auto_correct,
                    event_bus=event_bus,
                )

                validation_end_time = time.time()
                validation_time_ms = (
                    validation_end_time - validation_start_time
                ) * 1000

                if validated.corrected:
                    auto_corrections += 1
                    correction_types.extend(validated.corrections)

                # Build structured state
                structured_state = StructuredState(
                    validation_failures=validation_failures,
                    auto_corrections=auto_corrections,
                    validation_errors=validation_errors,
                    correction_types=list(set(correction_types)),
                    validation_time_ms=validation_time_ms,
                )

                # Build telemetry if monitoring enabled
                telemetry = None
                if monitoring:
                    telemetry = StructuredTelemetry(
                        schema_name=schema.__name__,
                        validation_attempts=validation_attempts,
                        validation_failures=validation_failures,
                        auto_corrections=auto_corrections,
                        correction_types=list(set(correction_types)),
                        validation_success=True,
                        validation_time_ms=validation_time_ms,
                    )

                return StructuredResult(
                    data=validated.data,
                    raw=validated.raw,
                    corrected=validated.corrected,
                    corrections=validated.corrections,
                    state=state,
                    structured_state=structured_state,
                    telemetry=telemetry,
                    errors=errors,
                )

            except ValidationError as e:
                last_error = e
                validation_failures += 1
                validation_errors.append(e)
                errors.append(e)

                if on_validation_error:
                    on_validation_error(e, attempt + 1)

                # Don't retry on last attempt of last stream
                is_last_stream = fallback_index == len(all_streams) - 1
                is_last_attempt = attempt == max_attempts - 1
                if is_last_stream and is_last_attempt:
                    break

                if on_retry:
                    on_retry(
                        attempt + 1, f"Validation failed: {e.error_count()} errors"
                    )

                continue

            except Exception as e:
                last_error = e
                errors.append(e)
                # Non-validation errors - try next fallback
                if on_retry and attempt < max_attempts - 1:
                    on_retry(attempt + 1, str(e))
                break

        fallback_index += 1

    # All attempts exhausted
    if isinstance(last_error, ValidationError):
        raise ValueError(
            f"Schema validation failed after all retries: {last_error}"
        ) from last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("All attempts exhausted with no error recorded")


@dataclass
class _ParseResult(Generic[T]):
    """Internal parse result."""

    data: T
    raw: str
    corrected: bool
    corrections: list[str]


def _parse_and_validate(
    text: str,
    schema: type[T],
    auto_correct: bool,
    strict_mode: bool,
    on_auto_correct: Callable[[AutoCorrectInfo], None] | None,
    event_bus: EventBus,
) -> _ParseResult[T]:
    """Parse and validate JSON text against schema."""
    event_bus.emit(
        ObservabilityEventType.PARSE_START,
        content_length=len(text),
    )
    parse_start = time.time()

    # Extract JSON from markdown if present
    original_text = text
    text = extract_json_from_markdown(text)

    # Auto-correct if enabled
    corrected = False
    corrections: list[str] = []

    if auto_correct:
        event_bus.emit(ObservabilityEventType.AUTO_CORRECT_START)
        result = auto_correct_json(text, track_corrections=True)
        text = result.text
        corrected = result.corrected
        corrections = result.corrections

        # If auto-correction failed, try extract_json as fallback
        if not result.success:
            extracted = extract_json(original_text)
            if extracted != original_text:
                result = auto_correct_json(extracted, track_corrections=True)
                if result.success:
                    text = result.text
                    corrected = True
                    if "extract_json" not in corrections:
                        corrections.insert(0, "extract_json")
                    corrections.extend(result.corrections)

        if corrected and on_auto_correct:
            on_auto_correct(
                AutoCorrectInfo(
                    original=original_text,
                    corrected=text,
                    corrections=corrections,
                    success=result.success,
                )
            )

        event_bus.emit(
            ObservabilityEventType.AUTO_CORRECT_END,
            corrected=corrected,
            corrections=corrections,
        )

    # Validate against schema
    event_bus.emit(
        ObservabilityEventType.SCHEMA_VALIDATION_START,
        schema_type="pydantic",
        schema_name=schema.__name__,
    )
    validation_start = time.time()

    try:
        # Use strict mode if requested (forbid extra fields)
        if strict_mode:
            # Parse JSON first, then validate and check for extra fields
            parsed_json = json.loads(text)

            # Check for extra fields not in the schema
            if isinstance(parsed_json, dict):
                schema_fields = set(schema.model_fields.keys())
                input_fields = set(parsed_json.keys())
                extra_fields = input_fields - schema_fields
                if extra_fields:
                    from pydantic_core import InitErrorDetails

                    line_errors: list[InitErrorDetails] = [
                        InitErrorDetails(
                            type="extra_forbidden",
                            loc=(field,),
                            input=parsed_json.get(field),
                        )
                        for field in extra_fields
                    ]
                    raise ValidationError.from_exception_data(
                        f"Extra fields not allowed: {extra_fields}",
                        line_errors,
                    )

            parsed = schema.model_validate(parsed_json)
        else:
            parsed = schema.model_validate_json(text)

        validation_duration = (time.time() - validation_start) * 1000
        event_bus.emit(
            ObservabilityEventType.SCHEMA_VALIDATION_END,
            valid=True,
            duration_ms=validation_duration,
        )
        parse_duration = (time.time() - parse_start) * 1000
        event_bus.emit(
            ObservabilityEventType.PARSE_END,
            success=True,
            duration_ms=parse_duration,
        )

        return _ParseResult(
            data=parsed,
            raw=text,
            corrected=corrected,
            corrections=corrections,
        )

    except ValidationError:
        validation_duration = (time.time() - validation_start) * 1000
        event_bus.emit(
            ObservabilityEventType.SCHEMA_VALIDATION_END,
            valid=False,
            duration_ms=validation_duration,
        )
        parse_duration = (time.time() - parse_start) * 1000
        event_bus.emit(
            ObservabilityEventType.PARSE_END,
            success=False,
            duration_ms=parse_duration,
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Variant
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StructuredStreamResult(Generic[T]):
    """Result holder for structured streaming.

    The stream yields events while collecting content.
    Call `await result.validate()` after consuming the stream.
    """

    _text: str = ""
    _schema: type[T] | None = None
    _auto_correct: bool = True
    _strict_mode: bool = False
    _on_auto_correct: Callable[[AutoCorrectInfo], None] | None = None
    _on_event: Callable[[ObservabilityEvent], None] | None = None
    _validated: StructuredResult[T] | None = None
    state: State | None = None
    _aborted: bool = False

    def abort(self) -> None:
        """Abort the structured stream.

        Signals that the stream should stop processing.
        Matches TypeScript structured result API.
        """
        self._aborted = True

    @property
    def is_aborted(self) -> bool:
        """Check if abort was requested."""
        return self._aborted

    async def validate(self) -> StructuredResult[T]:
        """Validate collected content against schema.

        Call this after consuming the stream.

        Returns:
            StructuredResult with validated data

        Raises:
            ValueError: If validation fails
        """
        if self._validated is not None:
            return self._validated

        if self._schema is None:
            raise ValueError("Schema not set")

        event_bus = EventBus(self._on_event)
        try:
            parsed = _parse_and_validate(
                text=self._text,
                schema=self._schema,
                auto_correct=self._auto_correct,
                strict_mode=self._strict_mode,
                on_auto_correct=self._on_auto_correct,
                event_bus=event_bus,
            )
        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}") from e

        self._validated = StructuredResult(
            data=parsed.data,
            raw=parsed.raw,
            corrected=parsed.corrected,
            corrections=parsed.corrections,
            state=self.state,
        )
        return self._validated


async def structured_stream(
    schema: type[T],
    stream: AwaitableStreamSource,
    *,
    auto_correct: bool = True,
    strict_mode: bool = False,
    timeout: Timeout | None = None,
    on_auto_correct: Callable[[AutoCorrectInfo], None] | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    adapter: Adapter | str | None = None,
) -> tuple["AsyncIterator[Event]", StructuredStreamResult[T]]:
    """Stream tokens with validation at the end.

    Args:
        schema: Pydantic model class to validate against
        stream: Async LLM stream or factory function
        auto_correct: Whether to attempt JSON auto-correction
        strict_mode: Reject unknown fields in output (default: False)
        timeout: Timeout configuration (initial_token, inter_token)
        on_auto_correct: Callback when auto-correction is applied
        on_event: Optional callback for observability events
        adapter: Optional adapter hint

    Returns:
        Tuple of (event stream, result holder)
        Consume the stream, then call `await result.validate()`

    Example:
        ```python
        stream, result = await l0.structured_stream(
            schema=User,
            stream=openai_stream,
        )

        async for event in stream:
            if event.is_token:
                print(event.text, end="")

        validated = await result.validate()
        print(validated.data)
        ```
    """

    # _internal_run expects a callable factory
    def make_stream_factory(
        src: AwaitableStreamSource,
    ) -> AwaitableStreamFactory:
        if callable(src) and not hasattr(src, "__anext__"):
            return src
        else:
            return lambda: cast(RawStream, src)

    stream_factory = make_stream_factory(stream)

    # Create result holder
    result_holder = StructuredStreamResult[T]()
    result_holder._schema = schema
    result_holder._auto_correct = auto_correct
    result_holder._strict_mode = strict_mode
    result_holder._on_auto_correct = on_auto_correct
    result_holder._on_event = on_event

    # Run through L0 runtime
    l0_result = await _internal_run(
        stream=stream_factory,
        on_event=on_event,
        adapter=adapter,
        timeout=timeout,
    )

    async def collecting_stream() -> AsyncIterator[Event]:
        """Wrap stream to collect content."""
        content_parts: list[str] = []
        async for event in l0_result:
            if event.is_token and event.text:
                content_parts.append(event.text)
            yield event
        result_holder._text = "".join(content_parts)
        result_holder.state = l0_result.state

    return collecting_stream(), result_holder


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


async def structured_object(
    shape: dict[str, type | tuple[type, Any]],
    stream: AwaitableStreamSource,
    *,
    fallbacks: list[AwaitableStreamSource] | None = None,
    auto_correct: bool = True,
    strict_mode: bool = False,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    detect_zero_tokens: bool = False,
    monitoring: bool = False,
    on_validation_error: Callable[[ValidationError, int], None] | None = None,
    on_auto_correct: Callable[[AutoCorrectInfo], None] | None = None,
    on_retry: Callable[[int, str], None] | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    adapter: Adapter | str | None = None,
) -> StructuredResult[Any]:
    """Helper: Create structured output with a simple object schema.

    This is a convenience wrapper around `structured()` that creates a
    Pydantic model from a dictionary shape specification.

    Args:
        shape: Dictionary mapping field names to types or (type, default) tuples
        stream: Async LLM stream or factory function
        fallbacks: Optional fallback streams
        auto_correct: Whether to attempt JSON auto-correction (default: True)
        strict_mode: Reject unknown fields in output (default: False)
        retry: Retry configuration for validation failures
        timeout: Timeout configuration (initial_token, inter_token)
        detect_zero_tokens: Detect zero-token outputs (default: False)
        monitoring: Enable telemetry collection (default: False)
        on_validation_error: Callback when validation fails
        on_auto_correct: Callback when auto-correction is applied
        on_retry: Callback when retry occurs (attempt, reason)
        on_event: Optional callback for observability events
        adapter: Optional adapter hint

    Returns:
        StructuredResult with validated data

    Example:
        ```python
        result = await l0.structured_object(
            {"name": str, "age": int, "active": (bool, True)},
            stream=openai_stream,
        )

        print(result.data.name)
        print(result.data.age)
        ```
    """
    # Build field definitions for create_model
    field_definitions: dict[str, Any] = {}
    for field_name, field_spec in shape.items():
        if isinstance(field_spec, tuple):
            # (type, default) format
            field_type, default = field_spec
            field_definitions[field_name] = (field_type, default)
        else:
            # Just a type, required field
            field_definitions[field_name] = (field_spec, ...)

    # Create dynamic Pydantic model
    DynamicModel = create_model("DynamicObject", **field_definitions)

    return await structured(
        schema=DynamicModel,
        stream=stream,
        fallbacks=fallbacks,
        auto_correct=auto_correct,
        strict_mode=strict_mode,
        retry=retry,
        timeout=timeout,
        detect_zero_tokens=detect_zero_tokens,
        monitoring=monitoring,
        on_validation_error=on_validation_error,
        on_auto_correct=on_auto_correct,
        on_retry=on_retry,
        on_event=on_event,
        adapter=adapter,
    )


async def structured_array(
    item_schema: type[T],
    stream: AwaitableStreamSource,
    *,
    fallbacks: list[AwaitableStreamSource] | None = None,
    auto_correct: bool = True,
    strict_mode: bool = False,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    detect_zero_tokens: bool = False,
    monitoring: bool = False,
    on_validation_error: Callable[[ValidationError, int], None] | None = None,
    on_auto_correct: Callable[[AutoCorrectInfo], None] | None = None,
    on_retry: Callable[[int, str], None] | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    adapter: Adapter | str | None = None,
) -> StructuredResult[list[T]]:
    """Helper: Create structured output with an array schema.

    This is a convenience wrapper around `structured()` that validates
    an array of items against a Pydantic model.

    Args:
        item_schema: Pydantic model class for array items
        stream: Async LLM stream or factory function
        fallbacks: Optional fallback streams
        auto_correct: Whether to attempt JSON auto-correction (default: True)
        strict_mode: Reject unknown fields in output (default: False)
        retry: Retry configuration for validation failures
        timeout: Timeout configuration (initial_token, inter_token)
        detect_zero_tokens: Detect zero-token outputs (default: False)
        monitoring: Enable telemetry collection (default: False)
        on_validation_error: Callback when validation fails
        on_auto_correct: Callback when auto-correction is applied
        on_retry: Callback when retry occurs (attempt, reason)
        on_event: Optional callback for observability events
        adapter: Optional adapter hint

    Returns:
        StructuredResult with validated list of items

    Example:
        ```python
        class User(BaseModel):
            name: str
            age: int

        result = await l0.structured_array(
            User,
            stream=openai_stream,
        )

        for user in result.data:
            print(user.name)
        ```
    """
    # Custom parsing to handle array validation
    event_bus = EventBus(on_event)
    retry_config = retry or Retry(attempts=1)
    max_attempts = retry_config.attempts if retry_config.attempts is not None else 1

    # Track structured-specific state
    validation_attempts = 0
    validation_failures = 0
    auto_corrections_count = 0
    correction_types: list[str] = []
    validation_errors: list[ValidationError] = []
    errors: list[Exception] = []
    validation_start_time = 0.0

    def _is_async_iterator(obj: Any) -> bool:
        return hasattr(obj, "__anext__") and not callable(obj)

    def _make_buffering_factory(iterator: RawStream) -> StreamFactory:
        buffer: list[Any] = []
        consumed = False

        async def buffering_iterator() -> RawStream:
            nonlocal consumed
            if consumed:
                for item in buffer:
                    yield item
            else:
                async for item in iterator:
                    buffer.append(item)
                    yield item
                consumed = True

        return buffering_iterator

    if _is_async_iterator(stream):
        stream = _make_buffering_factory(cast(RawStream, stream))

    all_streams: list[AwaitableStreamSource] = [stream]
    if fallbacks:
        wrapped_fallbacks: list[AwaitableStreamFactory] = []
        for fb in fallbacks:
            if _is_async_iterator(fb):
                wrapped_fallbacks.append(_make_buffering_factory(cast(RawStream, fb)))
            else:
                wrapped_fallbacks.append(cast(AwaitableStreamFactory, fb))
        all_streams.extend(wrapped_fallbacks)

    last_error: Exception | None = None
    fallback_index = 0

    for stream_source in all_streams:
        for attempt in range(max_attempts):
            try:

                def make_stream_factory(
                    src: AwaitableStreamSource,
                ) -> AwaitableStreamFactory:
                    if callable(src) and not hasattr(src, "__anext__"):
                        return src
                    else:
                        return lambda: cast(RawStream, src)

                stream_factory = make_stream_factory(stream_source)

                result = await _internal_run(
                    stream=stream_factory,
                    on_event=on_event,
                    adapter=adapter,
                    timeout=timeout,
                )
                text = await result.read()
                state = result.state

                # Check for zero-token output
                if detect_zero_tokens and (not text or text.strip() == ""):
                    raise ValueError("Zero-token output detected")

                # Parse and validate as array
                event_bus.emit(
                    ObservabilityEventType.PARSE_START,
                    content_length=len(text),
                )
                parse_start = time.time()
                validation_start_time = time.time()
                validation_attempts += 1

                original_text = text
                text = extract_json_from_markdown(text)

                corrected = False
                corrections: list[str] = []

                if auto_correct:
                    event_bus.emit(ObservabilityEventType.AUTO_CORRECT_START)
                    ac_result = auto_correct_json(text, track_corrections=True)
                    text = ac_result.text
                    corrected = ac_result.corrected
                    corrections = ac_result.corrections

                    # If auto-correction failed, try extract_json as fallback
                    if not ac_result.success:
                        extracted = extract_json(original_text)
                        if extracted != original_text:
                            ac_result = auto_correct_json(
                                extracted, track_corrections=True
                            )
                            if ac_result.success:
                                text = ac_result.text
                                corrected = True
                                if "extract_json" not in corrections:
                                    corrections.insert(0, "extract_json")
                                corrections.extend(ac_result.corrections)

                    if corrected and on_auto_correct:
                        on_auto_correct(
                            AutoCorrectInfo(
                                original=original_text,
                                corrected=text,
                                corrections=corrections,
                                success=ac_result.success,
                            )
                        )

                    event_bus.emit(
                        ObservabilityEventType.AUTO_CORRECT_END,
                        corrected=corrected,
                        corrections=corrections,
                    )

                # Validate as list of items
                event_bus.emit(
                    ObservabilityEventType.SCHEMA_VALIDATION_START,
                    schema_type="pydantic",
                    schema_name=f"list[{item_schema.__name__}]",
                )
                validation_start = time.time()

                parsed_json = json.loads(text)
                if not isinstance(parsed_json, list):
                    raise ValidationError.from_exception_data(
                        "Expected array",
                        [{"type": "list_type", "loc": (), "input": parsed_json}],
                    )

                # Validate each item
                validated_items: list[T] = []
                for i, item in enumerate(parsed_json):
                    if strict_mode:
                        validated_items.append(
                            item_schema.model_validate(item, strict=True)
                        )
                    else:
                        validated_items.append(item_schema.model_validate(item))

                validation_duration = (time.time() - validation_start) * 1000
                event_bus.emit(
                    ObservabilityEventType.SCHEMA_VALIDATION_END,
                    valid=True,
                    duration_ms=validation_duration,
                )
                parse_duration = (time.time() - parse_start) * 1000
                event_bus.emit(
                    ObservabilityEventType.PARSE_END,
                    success=True,
                    duration_ms=parse_duration,
                )

                if corrected:
                    auto_corrections_count += 1
                    correction_types.extend(corrections)

                # Build structured state
                structured_state = StructuredState(
                    validation_failures=validation_failures,
                    auto_corrections=auto_corrections_count,
                    validation_errors=validation_errors,
                    correction_types=list(set(correction_types)),
                    validation_time_ms=validation_duration,
                )

                # Build telemetry if monitoring enabled
                telemetry = None
                if monitoring:
                    telemetry = StructuredTelemetry(
                        schema_name=f"list[{item_schema.__name__}]",
                        validation_attempts=validation_attempts,
                        validation_failures=validation_failures,
                        auto_corrections=auto_corrections_count,
                        correction_types=list(set(correction_types)),
                        validation_success=True,
                        validation_time_ms=validation_duration,
                    )

                return StructuredResult(
                    data=validated_items,
                    raw=text,
                    corrected=corrected,
                    corrections=corrections,
                    state=state,
                    structured_state=structured_state,
                    telemetry=telemetry,
                    errors=errors,
                )

            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
                if isinstance(e, ValidationError):
                    validation_failures += 1
                    validation_errors.append(e)
                errors.append(e)

                if on_validation_error and isinstance(e, ValidationError):
                    on_validation_error(e, attempt + 1)

                is_last_stream = fallback_index == len(all_streams) - 1
                is_last_attempt = attempt == max_attempts - 1
                if is_last_stream and is_last_attempt:
                    break

                if on_retry:
                    on_retry(attempt + 1, str(e))

                continue

            except Exception as e:
                last_error = e
                errors.append(e)
                if on_retry and attempt < max_attempts - 1:
                    on_retry(attempt + 1, str(e))
                break

        fallback_index += 1

    if isinstance(last_error, (ValidationError, json.JSONDecodeError)):
        raise ValueError(
            f"Array validation failed after all retries: {last_error}"
        ) from last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("All attempts exhausted with no error recorded")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Presets
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StructuredConfig:
    """Configuration presets for structured output.

    Attributes:
        auto_correct: Whether to attempt JSON auto-correction
        attempts: Number of validation retry attempts
        strict_mode: Reject unknown fields in output
    """

    auto_correct: bool = True
    attempts: int = 1
    strict_mode: bool = False


# Preset configurations
MINIMAL_STRUCTURED = StructuredConfig(auto_correct=False, attempts=1)
"""Minimal config - no auto-correction, single attempt."""

RECOMMENDED_STRUCTURED = StructuredConfig(auto_correct=True, attempts=2)
"""Recommended config - auto-correction enabled, 2 attempts."""

STRICT_STRUCTURED = StructuredConfig(auto_correct=True, strict_mode=True, attempts=3)
"""Strict config - auto-correction, strict mode, 3 attempts."""
