"""Tool formatting utilities for L0.

This module provides functions for formatting tool/function definitions
for LLM consumption and parsing function calls from model output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import escape as html_escape
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

ToolFormatStyle = Literal["json-schema", "typescript", "natural", "xml"]
ParameterType = Literal["string", "number", "integer", "boolean", "array", "object"]


@dataclass
class ToolParameter:
    """A parameter definition for a tool."""

    name: str
    type: ParameterType
    description: str = ""
    required: bool = False
    enum: list[str] | None = None
    default: Any = None
    items: dict[str, Any] | None = None  # For array types
    properties: dict[str, Any] | None = None  # For object types


@dataclass
class Tool:
    """A tool/function definition."""

    name: str
    description: str = ""
    parameters: list[ToolParameter] = field(default_factory=list)


@dataclass
class ToolFormatOptions:
    """Options for formatting tools."""

    style: ToolFormatStyle = "json-schema"
    include_description: bool = True


@dataclass
class FunctionCall:
    """A parsed function call."""

    name: str
    arguments: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Creation Helpers
# ─────────────────────────────────────────────────────────────────────────────


def create_parameter(
    name: str,
    param_type: ParameterType,
    description: str = "",
    required: bool = False,
    *,
    enum: list[str] | None = None,
    default: Any = None,
    items: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
) -> ToolParameter:
    """Create a parameter definition.

    Args:
        name: The parameter name.
        param_type: The parameter type.
        description: Description of the parameter.
        required: Whether the parameter is required.
        enum: List of allowed values (for string types).
        default: Default value if not provided.
        items: Item schema for array types.
        properties: Property schema for object types.

    Returns:
        A ToolParameter object.

    Example:
        >>> param = create_parameter("location", "string", "City name", True)
        >>> param.name
        'location'
        >>> param.required
        True
    """
    return ToolParameter(
        name=name,
        type=param_type,
        description=description,
        required=required,
        enum=enum,
        default=default,
        items=items,
        properties=properties,
    )


def create_tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
) -> Tool:
    """Create a tool definition.

    Args:
        name: The tool name.
        description: Description of what the tool does.
        parameters: List of parameter definitions.

    Returns:
        A Tool object.

    Example:
        >>> tool = create_tool("get_weather", "Get current weather", [
        ...     create_parameter("location", "string", "City name", True),
        ... ])
        >>> tool.name
        'get_weather'
    """
    return Tool(
        name=name,
        description=description,
        parameters=parameters or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_tool(tool: Tool) -> list[str]:
    """Validate a tool definition.

    Args:
        tool: The tool to validate.

    Returns:
        A list of validation error messages. Empty if valid.

    Example:
        >>> tool = Tool(name="", description="Test")
        >>> errors = validate_tool(tool)
        >>> "Tool name is required" in errors
        True
    """
    errors = []

    if not tool.name:
        errors.append("Tool name is required")
    elif not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tool.name):
        errors.append(
            f"Tool name '{tool.name}' must be a valid identifier "
            "(start with letter or underscore, contain only alphanumeric and underscores)"
        )

    if not tool.description:
        errors.append("Tool description is recommended")

    seen_names = set()
    for param in tool.parameters:
        if not param.name:
            errors.append("Parameter name is required")
        elif param.name in seen_names:
            errors.append(f"Duplicate parameter name: {param.name}")
        else:
            seen_names.add(param.name)

        valid_types = {"string", "number", "integer", "boolean", "array", "object"}
        if param.type not in valid_types:
            errors.append(f"Invalid parameter type: {param.type}")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Tool Formatting
# ─────────────────────────────────────────────────────────────────────────────


def _format_json_schema(tool: Tool, include_description: bool) -> dict[str, Any]:
    """Format tool as JSON Schema (OpenAI function calling format)."""
    properties: dict[str, Any] = {}
    required_params = []

    for param in tool.parameters:
        prop: dict[str, Any] = {"type": param.type}

        if param.description:
            prop["description"] = param.description

        if param.enum:
            prop["enum"] = param.enum

        if param.default is not None:
            prop["default"] = param.default

        if param.type == "array" and param.items:
            prop["items"] = param.items

        if param.type == "object" and param.properties:
            prop["properties"] = param.properties

        properties[param.name] = prop

        if param.required:
            required_params.append(param.name)

    result: dict[str, Any] = {
        "name": tool.name,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }

    if include_description and tool.description:
        result["description"] = tool.description

    if required_params:
        result["parameters"]["required"] = required_params

    return result


def _format_typescript(tool: Tool) -> str:
    """Format tool as TypeScript function signature."""
    params = []
    for param in tool.parameters:
        type_map = {
            "string": "string",
            "number": "number",
            "integer": "number",
            "boolean": "boolean",
            "array": "any[]",
            "object": "object",
        }
        ts_type = type_map.get(param.type, "any")

        if param.required:
            params.append(f"{param.name}: {ts_type}")
        else:
            params.append(f"{param.name}?: {ts_type}")

    params_str = ", ".join(params)

    if tool.description:
        return f"// {tool.description}\nfunction {tool.name}({params_str}): void;"
    return f"function {tool.name}({params_str}): void;"


def _format_natural(tool: Tool) -> str:
    """Format tool in natural language."""
    lines = [f"Tool: {tool.name}"]

    if tool.description:
        lines.append(f"Description: {tool.description}")

    if tool.parameters:
        lines.append("Parameters:")
        for param in tool.parameters:
            req_str = "(required)" if param.required else "(optional)"
            if param.description:
                lines.append(
                    f"  - {param.name} {req_str}: {param.type} - {param.description}"
                )
            else:
                lines.append(f"  - {param.name} {req_str}: {param.type}")

    return "\n".join(lines)


def _escape_xml(value: str) -> str:
    """Escape a string for safe XML output."""
    return html_escape(value, quote=True)


def _format_xml(tool: Tool) -> str:
    """Format tool as XML."""
    lines = [f'<tool name="{_escape_xml(tool.name)}">']

    if tool.description:
        lines.append(f"  <description>{_escape_xml(tool.description)}</description>")

    if tool.parameters:
        lines.append("  <parameters>")
        for param in tool.parameters:
            req_str = "true" if param.required else "false"
            desc = (
                f' description="{_escape_xml(param.description)}"'
                if param.description
                else ""
            )
            lines.append(
                f'    <parameter name="{_escape_xml(param.name)}" '
                f'type="{param.type}" required="{req_str}"{desc}/>'
            )
        lines.append("  </parameters>")

    lines.append("</tool>")
    return "\n".join(lines)


def format_tool(
    tool: Tool,
    options: ToolFormatOptions | dict[str, Any] | None = None,
) -> str | dict[str, Any]:
    """Format a single tool definition.

    Args:
        tool: The tool to format.
        options: Formatting options (style, include_description).

    Returns:
        The formatted tool (string or dict depending on style).

    Example:
        >>> tool = create_tool("get_weather", "Get weather", [
        ...     create_parameter("location", "string", "City", True),
        ... ])
        >>> format_tool(tool, {"style": "natural"})
        'Tool: get_weather\\nDescription: Get weather\\nParameters:\\n  - location (required): string - City'
    """
    if options is None:
        opts = ToolFormatOptions()
    elif isinstance(options, dict):
        opts = ToolFormatOptions(
            style=options.get("style", "json-schema"),
            include_description=options.get("include_description", True),
        )
    else:
        opts = options

    if opts.style == "json-schema":
        return _format_json_schema(tool, opts.include_description)
    elif opts.style == "typescript":
        return _format_typescript(tool)
    elif opts.style == "natural":
        return _format_natural(tool)
    elif opts.style == "xml":
        return _format_xml(tool)

    return _format_natural(tool)


def format_tools(
    tools: list[Tool],
    options: ToolFormatOptions | dict[str, Any] | None = None,
) -> str | list[dict[str, Any]]:
    """Format multiple tool definitions.

    Args:
        tools: The tools to format.
        options: Formatting options.

    Returns:
        The formatted tools (list for json-schema, string for others).

    Example:
        >>> tools = [
        ...     create_tool("tool1", "First tool"),
        ...     create_tool("tool2", "Second tool"),
        ... ]
        >>> result = format_tools(tools, {"style": "natural"})
        >>> "Tool: tool1" in result
        True
    """
    if options is None:
        opts = ToolFormatOptions()
    elif isinstance(options, dict):
        opts = ToolFormatOptions(
            style=options.get("style", "json-schema"),
            include_description=options.get("include_description", True),
        )
    else:
        opts = options

    if opts.style == "json-schema":
        return [_format_json_schema(t, opts.include_description) for t in tools]

    formatted = [format_tool(t, opts) for t in tools]
    return "\n\n".join(str(f) for f in formatted)


# ─────────────────────────────────────────────────────────────────────────────
# Function Call Parsing
# ─────────────────────────────────────────────────────────────────────────────

# Pattern for function call: function_name({"arg": "value"}) or function_name(arg=value)
# Use non-greedy quantifiers to avoid matching across multiple JSON blocks
_FUNCTION_CALL_PATTERN = re.compile(
    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(\{.*?\}|\[.*?\]|.*?)\s*\)",
    re.DOTALL,
)


def parse_function_call(output: str) -> FunctionCall | None:
    """Parse a function call from model output.

    Args:
        output: The model output containing a function call.

    Returns:
        A FunctionCall object if found, None otherwise.

    Example:
        >>> result = parse_function_call('get_weather({"location": "NYC"})')
        >>> result.name
        'get_weather'
        >>> result.arguments
        {'location': 'NYC'}
    """
    match = _FUNCTION_CALL_PATTERN.search(output)
    if not match:
        return None

    name = match.group(1)
    args_str = match.group(2).strip()

    # Try to parse as JSON
    if args_str.startswith("{") or args_str.startswith("["):
        try:
            arguments = json.loads(args_str)
            if isinstance(arguments, dict):
                return FunctionCall(name=name, arguments=arguments)
            return FunctionCall(name=name, arguments={"_args": arguments})
        except json.JSONDecodeError:
            pass

    # Try to parse as keyword arguments (key=value, key=value)
    if args_str:
        arguments = {}
        # Simple key=value parsing
        pairs = args_str.split(",")
        for pair in pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                arguments[key] = value
        if arguments:
            return FunctionCall(name=name, arguments=arguments)

    return FunctionCall(name=name, arguments={})


def format_function_arguments(
    arguments: dict[str, Any],
    pretty: bool = False,
) -> str:
    """Format function arguments as JSON.

    Args:
        arguments: The arguments dictionary.
        pretty: Whether to pretty-print the JSON.

    Returns:
        The JSON string.

    Example:
        >>> format_function_arguments({"location": "NYC"}, True)
        '{\\n  "location": "NYC"\\n}'
    """
    if pretty:
        return json.dumps(arguments, indent=2)
    return json.dumps(arguments)
