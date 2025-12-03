"""Structured output with Pydantic validation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from ._utils import auto_correct_json, extract_json_from_markdown
from .events import EventBus, ObservabilityEvent, ObservabilityEventType
from .runtime import _internal_run

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)


async def structured(
    schema: type[T],
    stream: Callable[[], AsyncIterator[Any]],
    *,
    auto_correct: bool = True,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    adapter: Any | str | None = None,
) -> T:
    """Get structured output validated against Pydantic schema.

    Args:
        schema: Pydantic model class to validate against
        stream: Factory function that returns an async LLM stream
        auto_correct: Whether to attempt JSON auto-correction
        on_event: Optional callback for observability events
        adapter: Optional adapter hint ("openai", "litellm", or Adapter instance)

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If schema validation fails
    """
    event_bus = EventBus(on_event)

    result = await _internal_run(stream=stream, on_event=on_event, adapter=adapter)
    text = await result.read()

    # Extract JSON from markdown if present
    event_bus.emit(
        ObservabilityEventType.PARSE_START,
        content_length=len(text),
    )
    parse_start = time.time()

    text = extract_json_from_markdown(text)

    if auto_correct:
        event_bus.emit(ObservabilityEventType.AUTO_CORRECT_START)
        original_text = text
        text = auto_correct_json(text)
        corrected = text != original_text
        event_bus.emit(
            ObservabilityEventType.AUTO_CORRECT_END,
            corrected=corrected,
        )

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
        return parsed
    except ValidationError as e:
        validation_duration = (time.time() - validation_start) * 1000
        event_bus.emit(
            ObservabilityEventType.SCHEMA_VALIDATION_END,
            valid=False,
            errors=str(e),
            duration_ms=validation_duration,
        )
        event_bus.emit(
            ObservabilityEventType.PARSE_ERROR,
            error=str(e),
        )
        raise ValueError(f"Schema validation failed: {e}") from e
