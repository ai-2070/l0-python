"""Structured output with Pydantic validation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ._utils import AutoCorrectResult, auto_correct_json, extract_json_from_markdown
from .events import EventBus, ObservabilityEvent, ObservabilityEventType
from .runtime import _internal_run
from .types import Event, Retry, State

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StructuredResult(Generic[T]):
    """Result of structured output extraction.

    Attributes:
        data: Validated Pydantic model instance
        raw: Raw JSON string before parsing
        corrected: Whether auto-correction was applied
        corrections: List of corrections applied
        state: L0 runtime state (token counts, retries, etc.)
    """

    data: T
    raw: str
    corrected: bool = False
    corrections: list[str] = field(default_factory=list)
    state: State | None = None


@dataclass
class AutoCorrectInfo:
    """Information passed to on_auto_correct callback."""

    original: str
    corrected: str
    corrections: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────


async def structured(
    schema: type[T],
    stream: AsyncIterator[Any] | Callable[[], AsyncIterator[Any]],
    *,
    fallbacks: list[AsyncIterator[Any] | Callable[[], AsyncIterator[Any]]]
    | None = None,
    auto_correct: bool = True,
    retry: Retry | None = None,
    on_validation_error: Callable[[ValidationError, int], None] | None = None,
    on_auto_correct: Callable[[AutoCorrectInfo], None] | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    adapter: Any | str | None = None,
) -> StructuredResult[T]:
    """Get structured output validated against Pydantic schema.

    Args:
        schema: Pydantic model class to validate against
        stream: Async LLM stream or factory function that returns one
        fallbacks: Optional fallback streams to try if primary fails
        auto_correct: Whether to attempt JSON auto-correction (default: True)
        retry: Retry configuration for validation failures
        on_validation_error: Callback when validation fails (error, attempt)
        on_auto_correct: Callback when auto-correction is applied
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
    max_attempts = retry_config.attempts

    # Build list of streams to try
    all_streams: list[AsyncIterator[Any] | Callable[[], AsyncIterator[Any]]] = [stream]
    if fallbacks:
        all_streams.extend(fallbacks)

    last_error: Exception | None = None
    fallback_index = 0

    for stream_source in all_streams:
        for attempt in range(max_attempts):
            try:
                # _internal_run expects a callable factory
                # Handle both direct async iterators and factory functions
                def make_stream_factory(src: Any) -> Callable[[], AsyncIterator[Any]]:
                    if callable(src) and not hasattr(src, "__anext__"):
                        # It's already a factory
                        return src
                    else:
                        # It's a direct async iterator - wrap in factory
                        # Note: This only works once per stream!
                        return lambda: src

                stream_factory = make_stream_factory(stream_source)

                # Run through L0 runtime
                result = await _internal_run(
                    stream=stream_factory,
                    on_event=on_event,
                    adapter=adapter,
                )
                text = await result.read()
                state = result.state

                # Extract and validate
                validated = _parse_and_validate(
                    text=text,
                    schema=schema,
                    auto_correct=auto_correct,
                    on_auto_correct=on_auto_correct,
                    event_bus=event_bus,
                )

                return StructuredResult(
                    data=validated.data,
                    raw=validated.raw,
                    corrected=validated.corrected,
                    corrections=validated.corrections,
                    state=state,
                )

            except ValidationError as e:
                last_error = e
                if on_validation_error:
                    on_validation_error(e, attempt + 1)

                # Don't retry on last attempt of last stream
                is_last_stream = fallback_index == len(all_streams) - 1
                is_last_attempt = attempt == max_attempts - 1
                if is_last_stream and is_last_attempt:
                    break

                continue

            except Exception as e:
                last_error = e
                # Non-validation errors - try next fallback
                break

        fallback_index += 1

    # All attempts exhausted
    if isinstance(last_error, ValidationError):
        raise ValueError(
            f"Schema validation failed after all retries: {last_error}"
        ) from last_error
    raise last_error


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

        if corrected and on_auto_correct:
            on_auto_correct(
                AutoCorrectInfo(
                    original=original_text,
                    corrected=text,
                    corrections=corrections,
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
    _on_auto_correct: Callable[[AutoCorrectInfo], None] | None = None
    _validated: StructuredResult[T] | None = None
    state: State | None = None

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

        event_bus = EventBus(None)
        parsed = _parse_and_validate(
            text=self._text,
            schema=self._schema,
            auto_correct=self._auto_correct,
            on_auto_correct=self._on_auto_correct,
            event_bus=event_bus,
        )

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
    stream: AsyncIterator[Any] | Callable[[], AsyncIterator[Any]],
    *,
    auto_correct: bool = True,
    on_auto_correct: Callable[[AutoCorrectInfo], None] | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    adapter: Any | str | None = None,
) -> tuple[AsyncIterator[Event], StructuredStreamResult[T]]:
    """Stream tokens with validation at the end.

    Args:
        schema: Pydantic model class to validate against
        stream: Async LLM stream or factory function
        auto_correct: Whether to attempt JSON auto-correction
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
    def make_stream_factory(src: Any) -> Callable[[], AsyncIterator[Any]]:
        if callable(src) and not hasattr(src, "__anext__"):
            return src
        else:
            return lambda: src

    stream_factory = make_stream_factory(stream)

    # Create result holder
    result_holder = StructuredStreamResult[T]()
    result_holder._schema = schema
    result_holder._auto_correct = auto_correct
    result_holder._on_auto_correct = on_auto_correct

    # Run through L0 runtime
    l0_result = await _internal_run(
        stream=stream_factory,
        on_event=on_event,
        adapter=adapter,
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
