"""Structured output exports."""

from .._structured import (
    MINIMAL_STRUCTURED,
    RECOMMENDED_STRUCTURED,
    STRICT_STRUCTURED,
    AutoCorrectInfo,
    StructuredConfig,
    StructuredResult,
    StructuredState,
    StructuredStreamResult,
    StructuredTelemetry,
    structured,
    structured_array,
    structured_object,
    structured_stream,
)

# Alias to avoid module shadowing (l0.structured is a module file)
structured_output = structured

__all__ = [
    "MINIMAL_STRUCTURED",
    "RECOMMENDED_STRUCTURED",
    "STRICT_STRUCTURED",
    "AutoCorrectInfo",
    "StructuredConfig",
    "StructuredResult",
    "StructuredState",
    "StructuredStreamResult",
    "StructuredTelemetry",
    "structured",
    "structured_output",
    "structured_array",
    "structured_object",
    "structured_stream",
]
