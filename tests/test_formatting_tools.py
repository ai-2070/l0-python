"""Tests for l0.formatting.tools module."""

from __future__ import annotations

import json

import pytest

from l0.formatting.tools import (
    FunctionCall,
    Tool,
    ToolParameter,
    create_parameter,
    create_tool,
    format_function_arguments,
    format_tool,
    format_tools,
    parse_function_call,
    validate_tool,
)


class TestCreateParameter:
    """Tests for create_parameter function."""

    def test_create_basic_parameter(self):
        param = create_parameter("location", "string", "City name", True)
        assert param.name == "location"
        assert param.type == "string"
        assert param.description == "City name"
        assert param.required is True

    def test_create_optional_parameter(self):
        param = create_parameter("units", "string", "Temperature units", False)
        assert param.required is False

    def test_create_parameter_with_enum(self):
        param = create_parameter(
            "color", "string", "Color choice", True, enum=["red", "green", "blue"]
        )
        assert param.enum == ["red", "green", "blue"]

    def test_create_parameter_with_default(self):
        param = create_parameter("limit", "integer", "Max results", False, default=10)
        assert param.default == 10

    def test_create_array_parameter(self):
        param = create_parameter(
            "tags", "array", "List of tags", False, items={"type": "string"}
        )
        assert param.type == "array"
        assert param.items == {"type": "string"}


class TestCreateTool:
    """Tests for create_tool function."""

    def test_create_basic_tool(self):
        tool = create_tool("get_weather", "Get current weather")
        assert tool.name == "get_weather"
        assert tool.description == "Get current weather"
        assert tool.parameters == []

    def test_create_tool_with_parameters(self):
        params = [
            create_parameter("location", "string", "City", True),
            create_parameter("units", "string", "Units", False),
        ]
        tool = create_tool("get_weather", "Get weather", params)
        assert len(tool.parameters) == 2


class TestValidateTool:
    """Tests for validate_tool function."""

    def test_valid_tool(self):
        tool = create_tool(
            "get_weather",
            "Get weather",
            [
                create_parameter("location", "string", "City", True),
            ],
        )
        errors = validate_tool(tool)
        assert len(errors) == 0

    def test_empty_name(self):
        tool = Tool(name="", description="Test")
        errors = validate_tool(tool)
        assert "Tool name is required" in errors

    def test_invalid_name(self):
        tool = Tool(name="123invalid", description="Test")
        errors = validate_tool(tool)
        assert any("valid identifier" in e for e in errors)

    def test_missing_description(self):
        """Test that missing description is an error (not just a warning)."""
        tool = Tool(name="test_tool", description="")
        errors = validate_tool(tool)
        assert any("description is required" in e.lower() for e in errors)

    def test_duplicate_parameter_names(self):
        tool = Tool(
            name="test",
            description="Test",
            parameters=[
                ToolParameter(name="param", type="string"),
                ToolParameter(name="param", type="string"),
            ],
        )
        errors = validate_tool(tool)
        assert any("Duplicate parameter name" in e for e in errors)


class TestFormatToolJsonSchema:
    """Tests for format_tool with json-schema style."""

    def test_format_basic_tool(self):
        tool = create_tool(
            "get_weather",
            "Get weather",
            [
                create_parameter("location", "string", "City name", True),
            ],
        )
        result = format_tool(tool, {"style": "json-schema"})
        assert isinstance(result, dict)
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather"
        assert "parameters" in result
        assert result["parameters"]["type"] == "object"
        assert "location" in result["parameters"]["properties"]
        assert result["parameters"]["required"] == ["location"]

    def test_format_tool_with_enum(self):
        tool = create_tool(
            "set_color",
            "Set color",
            [
                create_parameter(
                    "color", "string", "Color", True, enum=["red", "blue"]
                ),
            ],
        )
        result = format_tool(tool, {"style": "json-schema"})
        assert isinstance(result, dict)
        assert result["parameters"]["properties"]["color"]["enum"] == ["red", "blue"]

    def test_format_without_description(self):
        tool = create_tool("test", "Description")
        result = format_tool(
            tool, {"style": "json-schema", "include_description": False}
        )
        assert isinstance(result, dict)
        assert "description" not in result


class TestFormatToolTypescript:
    """Tests for format_tool with typescript style."""

    def test_format_basic_tool(self):
        tool = create_tool(
            "get_weather",
            "Get weather",
            [
                create_parameter("location", "string", "City", True),
            ],
        )
        result = format_tool(tool, {"style": "typescript"})
        assert "function get_weather" in result
        assert "location: string" in result
        assert ": void;" in result

    def test_format_optional_parameter(self):
        tool = create_tool(
            "test",
            "Test",
            [
                create_parameter("optional", "string", "Optional param", False),
            ],
        )
        result = format_tool(tool, {"style": "typescript"})
        assert "optional?: string" in result

    def test_format_with_description(self):
        tool = create_tool("test", "Test description")
        result = format_tool(tool, {"style": "typescript"})
        assert "// Test description" in result

    def test_optional_params_reordered_after_required(self):
        """Optional parameters should be reordered after required ones for valid TypeScript."""
        tool = create_tool(
            "test",
            "Test",
            [
                create_parameter("optional1", "string", "Optional first", False),
                create_parameter("required1", "string", "Required", True),
                create_parameter("optional2", "number", "Optional second", False),
                create_parameter("required2", "boolean", "Required second", True),
            ],
        )
        result = format_tool(tool, {"style": "typescript"})
        # Required params should come before optional params
        # Within each group, params are sorted alphabetically
        assert (
            "required1: string, required2: boolean, optional1?: string, optional2?: number"
            in result
        )


class TestFormatToolNatural:
    """Tests for format_tool with natural style."""

    def test_format_basic_tool(self):
        tool = create_tool(
            "get_weather",
            "Get weather",
            [
                create_parameter("location", "string", "City", True),
            ],
        )
        result = format_tool(tool, {"style": "natural"})
        assert "Tool: get_weather" in result
        assert "Description: Get weather" in result
        assert "Parameters:" in result
        assert "location (required): string - City" in result

    def test_format_optional_parameter(self):
        tool = create_tool(
            "test",
            "Test",
            [
                create_parameter("opt", "number", "Optional", False),
            ],
        )
        result = format_tool(tool, {"style": "natural"})
        assert "opt (optional): number" in result


class TestFormatToolXml:
    """Tests for format_tool with xml style."""

    def test_format_basic_tool(self):
        tool = create_tool(
            "get_weather",
            "Get weather",
            [
                create_parameter("location", "string", "City", True),
            ],
        )
        result = format_tool(tool, {"style": "xml"})
        assert '<tool name="get_weather">' in result
        assert "<description>Get weather</description>" in result
        assert '<parameter name="location"' in result
        assert 'type="string"' in result
        assert 'required="true"' in result

    def test_format_xml_escapes_special_chars(self):
        """Test that XML special characters are escaped."""
        tool = create_tool(
            "test<tool>",
            'Description with "quotes" & <brackets>',
            [
                create_parameter("param<name>", "string", 'Has "quotes"', True),
            ],
        )
        result = format_tool(tool, {"style": "xml"})
        # Verify special chars are escaped
        assert 'name="test&lt;tool&gt;"' in result
        assert "&amp;" in result
        assert "&lt;brackets&gt;" in result
        assert "&quot;quotes&quot;" in result
        # Verify no raw special chars in attribute values
        assert 'name="test<tool>"' not in result


class TestFormatTools:
    """Tests for format_tools function."""

    def test_format_multiple_tools_json_schema(self):
        tools = [
            create_tool("tool1", "First"),
            create_tool("tool2", "Second"),
        ]
        result = format_tools(tools, {"style": "json-schema"})
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "tool1"
        assert result[1]["name"] == "tool2"

    def test_format_multiple_tools_natural(self):
        tools = [
            create_tool("tool1", "First"),
            create_tool("tool2", "Second"),
        ]
        result = format_tools(tools, {"style": "natural"})
        assert isinstance(result, str)
        assert "Tool: tool1" in result
        assert "Tool: tool2" in result


class TestParseFunctionCall:
    """Tests for parse_function_call function."""

    def test_parse_json_args(self):
        result = parse_function_call('get_weather({"location": "NYC"})')
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments == {"location": "NYC"}

    def test_parse_keyword_args(self):
        result = parse_function_call("get_weather(location=NYC, units=metric)")
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments["location"] == "NYC"
        assert result.arguments["units"] == "metric"

    def test_parse_empty_args(self):
        result = parse_function_call("do_something()")
        assert result is not None
        assert result.name == "do_something"
        assert result.arguments == {}

    def test_parse_with_surrounding_text(self):
        result = parse_function_call(
            'I will call get_weather({"location": "LA"}) for you'
        )
        assert result is not None
        assert result.name == "get_weather"
        assert result.arguments["location"] == "LA"

    def test_parse_no_function_call(self):
        result = parse_function_call("Just some text without a function call")
        assert result is None

    def test_parse_array_args(self):
        result = parse_function_call("process([1, 2, 3])")
        assert result is not None
        assert result.name == "process"
        assert "_args" in result.arguments
        assert result.arguments["_args"] == [1, 2, 3]

    def test_parse_with_multiple_json_blocks(self):
        """Test that greedy matching doesn't grab content across multiple JSON blocks."""
        # This simulates output with multiple function-like patterns
        result = parse_function_call(
            'first_func({"a": 1}) and then second_func({"b": 2})'
        )
        assert result is not None
        assert result.name == "first_func"
        assert result.arguments == {"a": 1}

    def test_parse_keyword_args_with_json_value(self):
        """Test that keyword args with JSON values containing commas are parsed correctly."""
        result = parse_function_call(
            'update_config(name=test, data={"key": "value", "count": 5})'
        )
        assert result is not None
        assert result.name == "update_config"
        assert result.arguments["name"] == "test"
        assert result.arguments["data"] == {"key": "value", "count": 5}

    def test_parse_keyword_args_with_array_value(self):
        """Test that keyword args with array values containing commas are parsed correctly."""
        result = parse_function_call("process(items=[1, 2, 3], mode=fast)")
        assert result is not None
        assert result.name == "process"
        assert result.arguments["items"] == [1, 2, 3]
        assert result.arguments["mode"] == "fast"

    def test_parse_nested_json(self):
        """Test parsing function call with nested JSON objects."""
        result = parse_function_call('create({"user": {"name": "John", "age": 30}})')
        assert result is not None
        assert result.name == "create"
        # Note: with non-greedy matching, nested JSON may not parse fully
        # This test documents the current behavior

    def test_parse_escaped_quotes_in_string(self):
        """Test that escaped quotes inside strings don't terminate the string early."""
        result = parse_function_call(r'save(text="He said \"hello\"", name=test)')
        assert result is not None
        assert result.name == "save"
        assert result.arguments["name"] == "test"
        # The text argument should contain the escaped quotes
        assert "hello" in str(result.arguments["text"])

    def test_parse_escaped_quotes_in_json(self):
        """Test parsing JSON with escaped quotes."""
        result = parse_function_call(r'create({"message": "Say \"hi\"", "count": 1})')
        assert result is not None
        assert result.name == "create"


class TestFormatFunctionArguments:
    """Tests for format_function_arguments function."""

    def test_format_compact(self):
        result = format_function_arguments({"location": "NYC"}, False)
        assert result == '{"location": "NYC"}'

    def test_format_pretty(self):
        result = format_function_arguments({"location": "NYC"}, True)
        assert "{\n" in result
        assert '"location": "NYC"' in result
        assert "\n}" in result

    def test_format_nested(self):
        args = {"user": {"name": "John", "age": 30}}
        result = format_function_arguments(args, True)
        parsed = json.loads(result)
        assert parsed["user"]["name"] == "John"


class TestFormatToolsIncludeTypes:
    """Tests for include_types option in tool formatting."""

    def test_json_schema_include_types_default(self):
        """Test that types are included by default in json-schema."""
        tool = create_tool(
            "test",
            "Test tool",
            [create_parameter("param", "string", "A param", True)],
        )
        result = format_tool(tool, {"style": "json-schema"})
        assert isinstance(result, dict)
        assert result["parameters"]["properties"]["param"]["type"] == "string"

    def test_json_schema_include_types_false(self):
        """Test that types can be excluded from json-schema."""
        tool = create_tool(
            "test",
            "Test tool",
            [create_parameter("param", "string", "A param", True)],
        )
        result = format_tool(tool, {"style": "json-schema", "include_types": False})
        assert isinstance(result, dict)
        assert "type" not in result["parameters"]["properties"]["param"]


class TestFormatToolsIncludeExamples:
    """Tests for include_examples option in tool formatting."""

    def test_natural_include_examples_with_enum(self):
        """Test that examples show enum values when include_examples=True."""
        tool = create_tool(
            "set_color",
            "Set a color",
            [
                create_parameter(
                    "color", "string", "The color", True, enum=["red", "green", "blue"]
                )
            ],
        )
        result = format_tool(tool, {"style": "natural", "include_examples": True})
        assert "red" in result
        assert "green" in result
        assert "blue" in result

    def test_natural_include_examples_with_default(self):
        """Test that examples show default value when include_examples=True."""
        tool = create_tool(
            "set_limit",
            "Set a limit",
            [create_parameter("limit", "integer", "Max items", False, default=10)],
        )
        result = format_tool(tool, {"style": "natural", "include_examples": True})
        assert "10" in result

    def test_natural_include_examples_false(self):
        """Test that examples are not shown when include_examples=False."""
        tool = create_tool(
            "set_color",
            "Set a color",
            [
                create_parameter(
                    "color", "string", "The color", True, enum=["red", "green", "blue"]
                )
            ],
        )
        result = format_tool(tool, {"style": "natural", "include_examples": False})
        # The "Example usage:" section should not appear when include_examples=False
        assert "Example usage:" not in result


class TestFormatToolsSeparator:
    """Tests for tool separator in format_tools."""

    def test_separator_is_50_equals(self):
        """Test that the separator between tools is 50 equals signs."""
        tools = [
            create_tool("tool1", "First tool"),
            create_tool("tool2", "Second tool"),
        ]
        result = format_tools(tools, {"style": "natural"})
        assert isinstance(result, str)
        # Should contain exactly 50 equals signs as separator
        assert "=" * 50 in result

    def test_separator_not_30_equals(self):
        """Test that the old 30 equals separator is not used."""
        tools = [
            create_tool("tool1", "First tool"),
            create_tool("tool2", "Second tool"),
        ]
        result = format_tools(tools, {"style": "natural"})
        # Should not have exactly 30 equals (the old value)
        # Check that 30 equals followed by newline is not present without more equals
        assert isinstance(result, str)
        lines = result.split("\n")
        for line in lines:
            if line.strip() and all(c == "=" for c in line.strip()):
                assert len(line.strip()) == 50
