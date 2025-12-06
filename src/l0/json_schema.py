"""JSON Schema Compatibility Layer for L0.

L0 supports JSON Schema validation via user-provided validators.
This module provides type-safe abstractions for working with JSON Schema.

JSON Schema is a standard for describing JSON data structures.
Popular validators: jsonschema, fastjsonschema, etc.

L0 uses JSON Schema for:
1. Schema validation (via user-provided validate function)
2. Error handling (via user-provided error formatting)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

# JSON Schema definition (simplified type - JSON Schema is complex)
JSONSchemaDefinition = dict[str, Any]


@dataclass
class JSONSchemaValidationError:
    """Validation error from JSON Schema validation."""

    path: str
    message: str
    keyword: str | None = None
    params: dict[str, Any] | None = None


@dataclass
class JSONSchemaValidationSuccess(Generic[T]):
    """Successful validation result."""

    valid: bool = True
    data: T | None = None


@dataclass
class JSONSchemaValidationFailure:
    """Failed validation result."""

    valid: bool = False
    errors: list[JSONSchemaValidationError] | None = None


JSONSchemaValidationResult = (
    JSONSchemaValidationSuccess[T] | JSONSchemaValidationFailure
)


class JSONSchemaAdapter(Protocol):
    """Adapter interface for JSON Schema validation.

    Users provide this to enable JSON Schema support in L0.
    """

    def validate(
        self, schema: JSONSchemaDefinition, data: Any
    ) -> JSONSchemaValidationSuccess[Any] | JSONSchemaValidationFailure:
        """Validate data against a JSON Schema.

        Args:
            schema: The JSON Schema definition
            data: The data to validate

        Returns:
            Validation result with typed data or errors
        """
        ...

    def format_errors(self, errors: list[JSONSchemaValidationError]) -> str:
        """Format validation errors into human-readable messages.

        Args:
            errors: Array of validation errors

        Returns:
            Formatted error message
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Global Adapter Storage
# ─────────────────────────────────────────────────────────────────────────────

_json_schema_adapter: JSONSchemaAdapter | None = None


def register_json_schema_adapter(adapter: JSONSchemaAdapter) -> None:
    """Register a JSON Schema adapter.

    Call this once at app startup to enable JSON Schema support.

    Example:
        ```python
        import jsonschema
        from l0 import register_json_schema_adapter, JSONSchemaValidationError

        class JsonSchemaAdapter:
            def validate(self, schema, data):
                validator = jsonschema.Draft7Validator(schema)
                errors = list(validator.iter_errors(data))
                if not errors:
                    return JSONSchemaValidationSuccess(valid=True, data=data)
                return JSONSchemaValidationFailure(
                    valid=False,
                    errors=[
                        JSONSchemaValidationError(
                            path="/".join(str(p) for p in e.absolute_path),
                            message=e.message,
                            keyword=e.validator,
                        )
                        for e in errors
                    ],
                )

            def format_errors(self, errors):
                return "; ".join(f"{e.path}: {e.message}" for e in errors)

        register_json_schema_adapter(JsonSchemaAdapter())
        ```
    """
    global _json_schema_adapter
    _json_schema_adapter = adapter


def unregister_json_schema_adapter() -> None:
    """Unregister the JSON Schema adapter."""
    global _json_schema_adapter
    _json_schema_adapter = None


def has_json_schema_adapter() -> bool:
    """Check if a JSON Schema adapter is registered."""
    return _json_schema_adapter is not None


def get_json_schema_adapter() -> JSONSchemaAdapter:
    """Get the registered JSON Schema adapter.

    Raises:
        RuntimeError: If no adapter is registered.
    """
    if _json_schema_adapter is None:
        raise RuntimeError(
            "JSON Schema adapter not registered. Call register_json_schema_adapter() first."
        )
    return _json_schema_adapter


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def is_json_schema(value: Any) -> bool:
    """Check if a value looks like a JSON Schema definition."""
    if not isinstance(value, dict):
        return False

    # JSON Schema typically has type, properties, or $schema
    return (
        "$schema" in value
        or "type" in value
        or "properties" in value
        or "$ref" in value
        or "allOf" in value
        or "anyOf" in value
        or "oneOf" in value
    )


def validate_json_schema(
    schema: JSONSchemaDefinition, data: Any
) -> tuple[bool, Any | None, Exception | None]:
    """Validate data against a JSON Schema.

    Returns a normalized result compatible with L0's error handling.

    Args:
        schema: The JSON Schema definition
        data: The data to validate

    Returns:
        Tuple of (success, data_or_none, error_or_none)
    """
    adapter = get_json_schema_adapter()
    result = adapter.validate(schema, data)

    if isinstance(result, JSONSchemaValidationSuccess) and result.valid:
        return (True, result.data, None)
    elif isinstance(result, JSONSchemaValidationFailure):
        errors = result.errors or []
        message = adapter.format_errors(errors)
        return (False, None, ValueError(message))
    else:
        return (False, None, ValueError("Unknown validation result"))


# ─────────────────────────────────────────────────────────────────────────────
# Unified Schema Wrapper
# ─────────────────────────────────────────────────────────────────────────────


class UnifiedSchema(Generic[T]):
    """Unified schema wrapper that works with Pydantic, JSON Schema, etc.

    Example:
        ```python
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
            "required": ["name", "age"],
        }

        unified = wrap_json_schema(schema)
        result = unified.safe_parse({"name": "Alice", "age": 30})
        if result[0]:
            print(result[1])  # {'name': 'Alice', 'age': 30}
        ```
    """

    def __init__(
        self,
        tag: str,
        parse_fn: Callable[[Any], T],
        safe_parse_fn: Callable[[Any], tuple[bool, T | None, Exception | None]],
    ):
        self._tag = tag
        self._parse_fn = parse_fn
        self._safe_parse_fn = safe_parse_fn

    @property
    def tag(self) -> str:
        """Schema type tag (e.g., 'pydantic', 'jsonschema')."""
        return self._tag

    def parse(self, data: Any) -> T:
        """Parse data, raising on validation failure."""
        return self._parse_fn(data)

    def safe_parse(self, data: Any) -> tuple[bool, T | None, Exception | None]:
        """Parse data without raising, returning (success, data, error)."""
        return self._safe_parse_fn(data)


def wrap_json_schema(schema: JSONSchemaDefinition) -> UnifiedSchema[Any]:
    """Wrap a JSON Schema in a unified interface for use with structured().

    Example:
        ```python
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
            "required": ["name", "age"],
        }

        result = await structured(
            schema=wrap_json_schema(schema),
            stream=stream,
        )
        ```
    """

    def parse(data: Any) -> Any:
        success, result, error = validate_json_schema(schema, data)
        if success:
            return result
        raise error  # type: ignore

    def safe_parse(data: Any) -> tuple[bool, Any | None, Exception | None]:
        return validate_json_schema(schema, data)

    return UnifiedSchema(
        tag="jsonschema",
        parse_fn=parse,
        safe_parse_fn=safe_parse,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simple In-Memory Adapter
# ─────────────────────────────────────────────────────────────────────────────


def _get_json_type(value: Any) -> str:
    """Get the JSON type of a value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


class SimpleJSONSchemaAdapter:
    """Simple in-memory JSON Schema adapter for basic validation.

    This is a minimal implementation for simple schemas without external dependencies.
    For production use, prefer jsonschema or fastjsonschema.
    """

    def validate(
        self, schema: JSONSchemaDefinition, data: Any
    ) -> JSONSchemaValidationSuccess[Any] | JSONSchemaValidationFailure:
        """Validate data against a JSON Schema."""
        errors: list[JSONSchemaValidationError] = []

        def validate_value(s: JSONSchemaDefinition, value: Any, path: str) -> None:
            # Type validation
            if "type" in s:
                types = s["type"] if isinstance(s["type"], list) else [s["type"]]
                actual_type = _get_json_type(value)

                type_matches = False
                for t in types:
                    if t == actual_type:
                        type_matches = True
                        break
                    # JSON Schema "number" includes integers
                    if t == "number" and actual_type == "integer":
                        type_matches = True
                        break
                    # JSON Schema "integer" is a subtype of number
                    if t == "integer" and actual_type == "number":
                        if isinstance(value, float) and value.is_integer():
                            type_matches = True
                            break

                if not type_matches:
                    errors.append(
                        JSONSchemaValidationError(
                            path=path,
                            message=f"Expected {' or '.join(types)}, got {actual_type}",
                            keyword="type",
                        )
                    )
                    return

            # Enum validation
            if "enum" in s and value not in s["enum"]:
                errors.append(
                    JSONSchemaValidationError(
                        path=path,
                        message=f"Value must be one of: {', '.join(str(v) for v in s['enum'])}",
                        keyword="enum",
                    )
                )

            # Const validation
            if "const" in s and value != s["const"]:
                errors.append(
                    JSONSchemaValidationError(
                        path=path,
                        message=f"Value must be {s['const']!r}",
                        keyword="const",
                    )
                )

            # Object validation
            if s.get("type") == "object" and isinstance(value, dict):
                # Required properties
                if "required" in s:
                    for prop in s["required"]:
                        if prop not in value:
                            errors.append(
                                JSONSchemaValidationError(
                                    path=f"{path}/{prop}",
                                    message=f"Missing required property: {prop}",
                                    keyword="required",
                                )
                            )

                # Property validation
                if "properties" in s:
                    for key, prop_schema in s["properties"].items():
                        if key in value:
                            validate_value(prop_schema, value[key], f"{path}/{key}")

            # Array validation
            if s.get("type") == "array" and isinstance(value, list):
                items_schema = s.get("items")
                if items_schema and isinstance(items_schema, dict):
                    for idx, item in enumerate(value):
                        validate_value(items_schema, item, f"{path}/{idx}")

            # String validation
            if s.get("type") == "string" and isinstance(value, str):
                if "minLength" in s and len(value) < s["minLength"]:
                    errors.append(
                        JSONSchemaValidationError(
                            path=path,
                            message=f"String must be at least {s['minLength']} characters",
                            keyword="minLength",
                        )
                    )
                if "maxLength" in s and len(value) > s["maxLength"]:
                    errors.append(
                        JSONSchemaValidationError(
                            path=path,
                            message=f"String must be at most {s['maxLength']} characters",
                            keyword="maxLength",
                        )
                    )
                if "pattern" in s:
                    import re

                    if not re.search(s["pattern"], value):
                        errors.append(
                            JSONSchemaValidationError(
                                path=path,
                                message=f"String must match pattern: {s['pattern']}",
                                keyword="pattern",
                            )
                        )

            # Number validation
            if s.get("type") in ("number", "integer") and isinstance(
                value, (int, float)
            ):
                if "minimum" in s and value < s["minimum"]:
                    errors.append(
                        JSONSchemaValidationError(
                            path=path,
                            message=f"Number must be >= {s['minimum']}",
                            keyword="minimum",
                        )
                    )
                if "maximum" in s and value > s["maximum"]:
                    errors.append(
                        JSONSchemaValidationError(
                            path=path,
                            message=f"Number must be <= {s['maximum']}",
                            keyword="maximum",
                        )
                    )

        validate_value(schema, data, "")

        if not errors:
            return JSONSchemaValidationSuccess(valid=True, data=data)
        return JSONSchemaValidationFailure(valid=False, errors=errors)

    def format_errors(self, errors: list[JSONSchemaValidationError]) -> str:
        """Format validation errors into human-readable messages."""
        return "; ".join(f"{e.path or '/'}: {e.message}" for e in errors)


def create_simple_json_schema_adapter() -> SimpleJSONSchemaAdapter:
    """Create a simple in-memory JSON Schema adapter for basic validation.

    This is a minimal implementation for simple schemas without external dependencies.
    For production use, prefer jsonschema or fastjsonschema.

    Example:
        ```python
        from l0 import register_json_schema_adapter, create_simple_json_schema_adapter

        register_json_schema_adapter(create_simple_json_schema_adapter())
        ```
    """
    return SimpleJSONSchemaAdapter()


# ─────────────────────────────────────────────────────────────────────────────
# Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class JSONSchema:
    """Scoped API for JSON Schema utilities.

    Provides utilities for validating data against JSON Schema definitions,
    registering validators, and wrapping schemas for use with structured().

    Usage:
        ```python
        from l0 import JSONSchema

        # Register a simple adapter (for basic validation)
        JSONSchema.register(JSONSchema.create_simple_adapter())

        # Check if value looks like a JSON Schema
        is_schema = JSONSchema.is_schema({"type": "object", "properties": {}})

        # Validate data against schema
        success, data, error = JSONSchema.validate(schema, {"name": "Alice"})

        # Wrap schema for use with structured()
        unified = JSONSchema.wrap(schema)
        result = unified.safe_parse(data)
        ```
    """

    # Re-export types for convenience
    Definition = JSONSchemaDefinition
    ValidationError = JSONSchemaValidationError
    ValidationSuccess = JSONSchemaValidationSuccess
    ValidationFailure = JSONSchemaValidationFailure
    Adapter = JSONSchemaAdapter
    SimpleAdapter = SimpleJSONSchemaAdapter
    Unified = UnifiedSchema

    @staticmethod
    def register(adapter: JSONSchemaAdapter) -> None:
        """Register a JSON Schema adapter.

        Call this once at app startup to enable JSON Schema support.

        Args:
            adapter: The JSON Schema adapter to register
        """
        register_json_schema_adapter(adapter)

    @staticmethod
    def unregister() -> None:
        """Unregister the JSON Schema adapter."""
        unregister_json_schema_adapter()

    @staticmethod
    def has_adapter() -> bool:
        """Check if a JSON Schema adapter is registered."""
        return has_json_schema_adapter()

    @staticmethod
    def get_adapter() -> JSONSchemaAdapter:
        """Get the registered JSON Schema adapter.

        Raises:
            RuntimeError: If no adapter is registered.
        """
        return get_json_schema_adapter()

    @staticmethod
    def is_schema(value: Any) -> bool:
        """Check if a value looks like a JSON Schema definition."""
        return is_json_schema(value)

    @staticmethod
    def validate(
        schema: JSONSchemaDefinition, data: Any
    ) -> tuple[bool, Any | None, Exception | None]:
        """Validate data against a JSON Schema.

        Returns a normalized result compatible with L0's error handling.

        Args:
            schema: The JSON Schema definition
            data: The data to validate

        Returns:
            Tuple of (success, data_or_none, error_or_none)
        """
        return validate_json_schema(schema, data)

    @staticmethod
    def wrap(schema: JSONSchemaDefinition) -> UnifiedSchema[Any]:
        """Wrap a JSON Schema in a unified interface for use with structured().

        Args:
            schema: The JSON Schema definition

        Returns:
            UnifiedSchema wrapper for the schema
        """
        return wrap_json_schema(schema)

    @staticmethod
    def create_simple_adapter() -> SimpleJSONSchemaAdapter:
        """Create a simple in-memory JSON Schema adapter for basic validation.

        This is a minimal implementation for simple schemas without external dependencies.
        For production use, prefer jsonschema or fastjsonschema.

        Returns:
            SimpleJSONSchemaAdapter instance
        """
        return create_simple_json_schema_adapter()
