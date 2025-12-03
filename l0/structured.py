"""Structured output with Pydantic validation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ._utils import auto_correct_json, extract_json_from_markdown
from .runtime import _internal_run

T = TypeVar("T", bound=BaseModel)


async def structured(
    schema: type[T],
    stream: Callable[[], AsyncIterator[Any]],
    *,
    auto_correct: bool = True,
) -> T:
    """Get structured output validated against Pydantic schema.

    Args:
        schema: Pydantic model class to validate against
        stream: Factory function that returns an async LLM stream
        auto_correct: Whether to attempt JSON auto-correction

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If schema validation fails
    """
    result = await _internal_run(stream=stream)
    text = await result.read()

    # Extract JSON from markdown if present
    text = extract_json_from_markdown(text)

    if auto_correct:
        text = auto_correct_json(text)

    try:
        return schema.model_validate_json(text)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}") from e
