"""Formatting helpers for L0.

This module provides utilities for formatting context, memory, output
instructions, tool definitions, and strings for LLM consumption.

Example:
    >>> from l0.formatting import format_context, format_memory, format_tool

    # Context formatting with XML delimiters (default)
    >>> format_context("User manual content", label="Documentation")
    '<documentation>\\nUser manual content\\n</documentation>'

    # Memory formatting
    >>> memory = [{"role": "user", "content": "Hello"}]
    >>> format_memory(memory)
    'User: Hello'

    # Tool formatting
    >>> tool = create_tool("get_weather", "Get weather", [
    ...     create_parameter("location", "string", "City", True),
    ... ])
    >>> format_tool(tool, {"style": "natural"})
    'Tool: get_weather\\nDescription: Get weather\\nParameters:\\n  - location (required): string - City'
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Context Formatting
# ─────────────────────────────────────────────────────────────────────────────
from .context import (
    ContextItem,
    ContextOptions,
    DelimiterType,
    DocumentMetadata,
    escape_delimiters,
    format_context,
    format_document,
    format_instructions,
    format_multiple_contexts,
    unescape_delimiters,
)

# ─────────────────────────────────────────────────────────────────────────────
# Memory Formatting
# ─────────────────────────────────────────────────────────────────────────────
from .memory import (
    MemoryEntry,
    MemoryFormatOptions,
    MemoryRole,
    MemoryStyle,
    calculate_memory_size,
    create_memory_entry,
    filter_memory_by_role,
    format_memory,
    get_last_n_entries,
    merge_memory,
    truncate_memory,
)

# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────
from .output import (
    JsonOutputOptions,
    OutputConstraints,
    OutputFormat,
    OutputFormatSectionOptions,
    StructuredOutputOptions,
    clean_output,
    create_output_format_section,
    extract_json_from_output,
    format_json_output,
    format_output_constraints,
    format_structured_output,
    validate_json_output,
    wrap_output_instruction,
)

# Re-export wrapOutputInstruction at top level for convenience
from .output import wrap_output_instruction as wrapOutputInstruction

# ─────────────────────────────────────────────────────────────────────────────
# String Utilities
# ─────────────────────────────────────────────────────────────────────────────
from .strings import (
    Alignment,
    escape,
    escape_html,
    escape_regex,
    pad,
    remove_ansi,
    sanitize,
    trim,
    truncate,
    truncate_words,
    unescape,
    unescape_html,
    wrap,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tool Formatting
# ─────────────────────────────────────────────────────────────────────────────
from .tools import (
    FunctionCall,
    ParameterType,
    Tool,
    ToolFormatOptions,
    ToolFormatStyle,
    ToolParameter,
    create_parameter,
    create_tool,
    format_function_arguments,
    format_tool,
    format_tools,
    parse_function_call,
    validate_tool,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Context
    "format_context",
    "format_multiple_contexts",
    "format_document",
    "format_instructions",
    "escape_delimiters",
    "unescape_delimiters",
    "ContextOptions",
    "ContextItem",
    "DocumentMetadata",
    "DelimiterType",
    # Memory
    "format_memory",
    "create_memory_entry",
    "merge_memory",
    "filter_memory_by_role",
    "get_last_n_entries",
    "calculate_memory_size",
    "truncate_memory",
    "MemoryEntry",
    "MemoryFormatOptions",
    "MemoryRole",
    "MemoryStyle",
    # Output
    "format_json_output",
    "format_structured_output",
    "format_output_constraints",
    "create_output_format_section",
    "extract_json_from_output",
    "clean_output",
    "validate_json_output",
    "wrap_output_instruction",
    "JsonOutputOptions",
    "StructuredOutputOptions",
    "OutputConstraints",
    "OutputFormatSectionOptions",
    "OutputFormat",
    # Tools
    "format_tool",
    "format_tools",
    "create_tool",
    "create_parameter",
    "validate_tool",
    "parse_function_call",
    "format_function_arguments",
    "Tool",
    "ToolParameter",
    "ToolFormatOptions",
    "ToolFormatStyle",
    "ParameterType",
    "FunctionCall",
    # Strings
    "trim",
    "escape",
    "unescape",
    "escape_html",
    "unescape_html",
    "escape_regex",
    "sanitize",
    "truncate",
    "truncate_words",
    "wrap",
    "pad",
    "remove_ansi",
    "Alignment",
]
