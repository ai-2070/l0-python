"""Tests for JSON Schema compatibility layer.

Tests for JSONSchemaAdapter, validation functions, and UnifiedSchema wrapper.
"""

from __future__ import annotations

import pytest

from l0.json_schema import (
    JSONSchemaDefinition,
    JSONSchemaValidationError,
    JSONSchemaValidationFailure,
    JSONSchemaValidationSuccess,
    SimpleJSONSchemaAdapter,
    UnifiedSchema,
    create_simple_json_schema_adapter,
    get_json_schema_adapter,
    has_json_schema_adapter,
    is_json_schema,
    register_json_schema_adapter,
    unregister_json_schema_adapter,
    validate_json_schema,
    wrap_json_schema,
)


@pytest.fixture(autouse=True)
def cleanup_adapter():
    """Clean up adapter registration after each test."""
    yield
    unregister_json_schema_adapter()


class TestAdapterRegistration:
    """Tests for adapter registration."""

    def test_no_adapter_initially(self):
        """Should have no adapter registered initially."""
        assert has_json_schema_adapter() is False

    def test_register_adapter(self):
        """Should register an adapter."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)
        assert has_json_schema_adapter() is True

    def test_unregister_adapter(self):
        """Should unregister an adapter."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)
        unregister_json_schema_adapter()
        assert has_json_schema_adapter() is False

    def test_get_adapter_raises_without_registration(self):
        """Should raise when getting adapter without registration."""
        with pytest.raises(RuntimeError, match="not registered"):
            get_json_schema_adapter()

    def test_get_adapter_returns_registered(self):
        """Should return registered adapter."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)
        result = get_json_schema_adapter()
        assert result is adapter


class TestIsJsonSchema:
    """Tests for is_json_schema function."""

    def test_detect_schema_with_type(self):
        """Should detect schema with 'type' property."""
        schema = {"type": "object"}
        assert is_json_schema(schema) is True

    def test_detect_schema_with_properties(self):
        """Should detect schema with 'properties' property."""
        schema = {"properties": {"name": {"type": "string"}}}
        assert is_json_schema(schema) is True

    def test_detect_schema_with_dollar_schema(self):
        """Should detect schema with '$schema' property."""
        schema = {"$schema": "http://json-schema.org/draft-07/schema#"}
        assert is_json_schema(schema) is True

    def test_detect_schema_with_ref(self):
        """Should detect schema with '$ref' property."""
        schema = {"$ref": "#/definitions/Something"}
        assert is_json_schema(schema) is True

    def test_detect_schema_with_allOf(self):
        """Should detect schema with 'allOf' property."""
        schema = {"allOf": [{"type": "string"}]}
        assert is_json_schema(schema) is True

    def test_detect_schema_with_anyOf(self):
        """Should detect schema with 'anyOf' property."""
        schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
        assert is_json_schema(schema) is True

    def test_detect_schema_with_oneOf(self):
        """Should detect schema with 'oneOf' property."""
        schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
        assert is_json_schema(schema) is True

    def test_not_detect_empty_dict(self):
        """Should not detect empty dict as schema."""
        assert is_json_schema({}) is False

    def test_not_detect_non_dict(self):
        """Should not detect non-dict as schema."""
        assert is_json_schema("string") is False
        assert is_json_schema(123) is False
        assert is_json_schema(None) is False
        assert is_json_schema([]) is False


class TestSimpleJSONSchemaAdapter:
    """Tests for SimpleJSONSchemaAdapter."""

    def test_validate_string_type(self):
        """Should validate string type."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "string"}

        result = adapter.validate(schema, "hello")
        assert result.valid is True
        assert isinstance(result, JSONSchemaValidationSuccess)
        assert result.data == "hello"

        result = adapter.validate(schema, 123)
        assert result.valid is False

    def test_validate_number_type(self):
        """Should validate number type."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "number"}

        result = adapter.validate(schema, 42)
        assert result.valid is True

        result = adapter.validate(schema, 3.14)
        assert result.valid is True

        result = adapter.validate(schema, "not a number")
        assert result.valid is False

    def test_validate_integer_type(self):
        """Should validate integer type."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "integer"}

        result = adapter.validate(schema, 42)
        assert result.valid is True

        result = adapter.validate(schema, 3.14)
        assert result.valid is False

    def test_validate_boolean_type(self):
        """Should validate boolean type."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "boolean"}

        result = adapter.validate(schema, True)
        assert result.valid is True

        result = adapter.validate(schema, False)
        assert result.valid is True

        result = adapter.validate(schema, "true")
        assert result.valid is False

    def test_validate_array_type(self):
        """Should validate array type."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "array"}

        result = adapter.validate(schema, [1, 2, 3])
        assert result.valid is True

        result = adapter.validate(schema, "not an array")
        assert result.valid is False

    def test_validate_object_type(self):
        """Should validate object type."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "object"}

        result = adapter.validate(schema, {"key": "value"})
        assert result.valid is True

        result = adapter.validate(schema, "not an object")
        assert result.valid is False

    def test_validate_null_type(self):
        """Should validate null type."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "null"}

        result = adapter.validate(schema, None)
        assert result.valid is True

        result = adapter.validate(schema, "not null")
        assert result.valid is False

    def test_validate_required_properties(self):
        """Should validate required properties."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
            "required": ["name", "age"],
        }

        result = adapter.validate(schema, {"name": "Alice", "age": 30})
        assert result.valid is True

        result = adapter.validate(schema, {"name": "Alice"})
        assert result.valid is False
        assert isinstance(result, JSONSchemaValidationFailure)
        assert result.errors is not None
        assert any("age" in e.path for e in result.errors)

    def test_validate_nested_properties(self):
        """Should validate nested properties."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        }

        result = adapter.validate(schema, {"person": {"name": "Alice"}})
        assert result.valid is True

        result = adapter.validate(schema, {"person": {"name": 123}})
        assert result.valid is False

    def test_validate_array_items(self):
        """Should validate array items."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }

        result = adapter.validate(schema, ["a", "b", "c"])
        assert result.valid is True

        result = adapter.validate(schema, ["a", 123, "c"])
        assert result.valid is False

    def test_validate_enum(self):
        """Should validate enum values."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"enum": ["red", "green", "blue"]}

        result = adapter.validate(schema, "red")
        assert result.valid is True

        result = adapter.validate(schema, "yellow")
        assert result.valid is False

    def test_validate_const(self):
        """Should validate const values."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"const": "fixed"}

        result = adapter.validate(schema, "fixed")
        assert result.valid is True

        result = adapter.validate(schema, "other")
        assert result.valid is False

    def test_validate_min_length(self):
        """Should validate minLength."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "string", "minLength": 3}

        result = adapter.validate(schema, "hello")
        assert result.valid is True

        result = adapter.validate(schema, "hi")
        assert result.valid is False

    def test_validate_max_length(self):
        """Should validate maxLength."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "string", "maxLength": 5}

        result = adapter.validate(schema, "hello")
        assert result.valid is True

        result = adapter.validate(schema, "hello world")
        assert result.valid is False

    def test_validate_pattern(self):
        """Should validate pattern."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "string", "pattern": "^[a-z]+$"}

        result = adapter.validate(schema, "hello")
        assert result.valid is True

        result = adapter.validate(schema, "Hello123")
        assert result.valid is False

    def test_validate_minimum(self):
        """Should validate minimum."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "number", "minimum": 0}

        result = adapter.validate(schema, 5)
        assert result.valid is True

        result = adapter.validate(schema, -1)
        assert result.valid is False

    def test_validate_maximum(self):
        """Should validate maximum."""
        adapter = SimpleJSONSchemaAdapter()
        schema = {"type": "number", "maximum": 100}

        result = adapter.validate(schema, 50)
        assert result.valid is True

        result = adapter.validate(schema, 150)
        assert result.valid is False

    def test_format_errors(self):
        """Should format errors correctly."""
        adapter = SimpleJSONSchemaAdapter()
        errors = [
            JSONSchemaValidationError(path="/name", message="Required"),
            JSONSchemaValidationError(path="/age", message="Must be number"),
        ]

        result = adapter.format_errors(errors)
        assert "/name" in result
        assert "Required" in result
        assert "/age" in result
        assert "Must be number" in result


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_validate_success(self):
        """Should return success for valid data."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)

        schema = {"type": "string"}
        success, data, error = validate_json_schema(schema, "hello")

        assert success is True
        assert data == "hello"
        assert error is None

    def test_validate_failure(self):
        """Should return failure for invalid data."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)

        schema = {"type": "string"}
        success, data, error = validate_json_schema(schema, 123)

        assert success is False
        assert data is None
        assert error is not None


class TestWrapJsonSchema:
    """Tests for wrap_json_schema function."""

    def test_wrap_creates_unified_schema(self):
        """Should create a UnifiedSchema."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)

        schema = {"type": "string"}
        unified = wrap_json_schema(schema)

        assert isinstance(unified, UnifiedSchema)
        assert unified.tag == "jsonschema"

    def test_parse_valid_data(self):
        """Should parse valid data."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)

        schema = {"type": "string"}
        unified = wrap_json_schema(schema)

        result = unified.parse("hello")
        assert result == "hello"

    def test_parse_invalid_data_raises(self):
        """Should raise for invalid data."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)

        schema = {"type": "string"}
        unified = wrap_json_schema(schema)

        with pytest.raises(ValueError):
            unified.parse(123)

    def test_safe_parse_valid_data(self):
        """Should safe_parse valid data."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)

        schema = {"type": "string"}
        unified = wrap_json_schema(schema)

        success, data, error = unified.safe_parse("hello")
        assert success is True
        assert data == "hello"
        assert error is None

    def test_safe_parse_invalid_data(self):
        """Should safe_parse invalid data without raising."""
        adapter = create_simple_json_schema_adapter()
        register_json_schema_adapter(adapter)

        schema = {"type": "string"}
        unified = wrap_json_schema(schema)

        success, data, error = unified.safe_parse(123)
        assert success is False
        assert data is None
        assert error is not None


class TestCreateSimpleJsonSchemaAdapter:
    """Tests for create_simple_json_schema_adapter function."""

    def test_creates_adapter(self):
        """Should create a SimpleJSONSchemaAdapter."""
        adapter = create_simple_json_schema_adapter()
        assert isinstance(adapter, SimpleJSONSchemaAdapter)

    def test_adapter_is_functional(self):
        """Should create a functional adapter."""
        adapter = create_simple_json_schema_adapter()

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = adapter.validate(schema, {"name": "Alice"})

        assert result.valid is True


class TestComplexSchemas:
    """Tests for complex schema validation."""

    def test_validate_user_schema(self):
        """Should validate complex user schema."""
        adapter = create_simple_json_schema_adapter()

        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string", "minLength": 1},
                "email": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "roles": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["id", "name", "email"],
        }

        valid_user = {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "age": 30,
            "roles": ["admin", "user"],
        }

        result = adapter.validate(schema, valid_user)
        assert result.valid is True

        invalid_user = {
            "id": "not an int",
            "name": "",
            "email": "alice@example.com",
            "age": -5,
        }

        result = adapter.validate(schema, invalid_user)
        assert result.valid is False

    def test_validate_nested_array_schema(self):
        """Should validate nested array schema."""
        adapter = create_simple_json_schema_adapter()

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["name", "score"],
            },
        }

        valid_data = [
            {"name": "Alice", "score": 95},
            {"name": "Bob", "score": 87},
        ]

        result = adapter.validate(schema, valid_data)
        assert result.valid is True

        invalid_data = [
            {"name": "Alice", "score": 95},
            {"name": "Bob"},  # missing score
        ]

        result = adapter.validate(schema, invalid_data)
        assert result.valid is False
