"""Output formatting utilities for L0.

This module provides functions for generating output format instructions
and extracting/cleaning model outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

OutputFormat = Literal["json", "yaml", "xml", "markdown", "plain"]


@dataclass
class JsonOutputOptions:
    """Options for JSON output formatting."""

    strict: bool = False
    schema: str | None = None
    example: str | None = None


@dataclass
class StructuredOutputOptions:
    """Options for structured output formatting."""

    strict: bool = False
    schema: str | None = None
    example: str | None = None


@dataclass
class OutputConstraints:
    """Constraints for output formatting."""

    max_length: int | None = None
    min_length: int | None = None
    no_code_blocks: bool = False
    no_markdown: bool = False
    language: str | None = None
    tone: str | None = None


@dataclass
class OutputFormatSectionOptions:
    """Options for creating a complete output format section."""

    strict: bool = False
    schema: str | None = None
    example: str | None = None
    constraints: OutputConstraints | dict[str, Any] | None = None
    wrap: bool = True  # Default to True to match TypeScript


# ─────────────────────────────────────────────────────────────────────────────
# Output Format Instructions
# ─────────────────────────────────────────────────────────────────────────────


def format_json_output(
    options: JsonOutputOptions | dict[str, Any] | None = None,
) -> str:
    """Generate instructions for requesting JSON output.

    Args:
        options: Options for JSON formatting (strict mode, schema, example).
            - strict: Require JSON-only response (default: False)
            - schema: JSON schema to include (optional)
            - example: Example output to include (optional)
            - include_instructions: Whether to include instructions (default: True)

    Returns:
        Instructions string for the model.

    Example:
        >>> format_json_output({"strict": True})
        'Respond with valid JSON only. Do not include any text before or after the JSON object.\\nDo not wrap the JSON in markdown code blocks or backticks.\\nStart your response with { and end with }.'

        >>> format_json_output({"strict": True, "schema": '{ "name": "string" }'})
        'Respond with valid JSON only...\\n\\nExpected JSON schema:\\n{ "name": "string" }'
    """
    include_instructions = True
    if options is None:
        opts = JsonOutputOptions()
    elif isinstance(options, dict):
        opts = JsonOutputOptions(
            strict=options.get("strict", False),
            schema=options.get("schema"),
            example=options.get("example"),
        )
        include_instructions = options.get("include_instructions", True)
    else:
        opts = options

    parts = []

    if include_instructions:
        if opts.strict:
            parts.append(
                "Respond with valid JSON only. Do not include any text before or after the JSON object."
            )
            parts.append("Do not wrap the JSON in markdown code blocks or backticks.")
            parts.append("Start your response with { and end with }.")
        else:
            parts.append("Respond with valid JSON.")

    if opts.schema:
        if parts:
            parts.append("")
        parts.append(f"Expected JSON schema:\n{opts.schema}")

    if opts.example:
        if parts:
            parts.append("")
        parts.append(f"Example output:\n{opts.example}")

    return "\n".join(parts)


def format_structured_output(
    format_type: OutputFormat,
    options: StructuredOutputOptions | dict[str, Any] | None = None,
) -> str:
    """Generate instructions for requesting specific output formats.

    Args:
        format_type: The output format - "json", "yaml", "xml", "markdown", "plain".
        options: Options for formatting (strict mode, schema, example).

    Returns:
        Instructions string for the model.

    Example:
        >>> format_structured_output("yaml", {"strict": True})
        'Respond with valid YAML only. Do not include any text before or after the YAML. Do not wrap in code blocks.'

        >>> format_structured_output("markdown")
        'Respond with Markdown.'
    """
    if options is None:
        opts = StructuredOutputOptions()
    elif isinstance(options, dict):
        opts = StructuredOutputOptions(
            strict=options.get("strict", False),
            schema=options.get("schema"),
            example=options.get("example"),
        )
    else:
        opts = options

    format_name = format_type.upper()
    parts = []

    if opts.strict:
        parts.append(
            f"Respond with valid {format_name} only. "
            f"Do not include any text before or after the {format_name}. "
            "Do not wrap in code blocks."
        )
    else:
        parts.append(f"Respond with {format_name}.")

    if opts.schema:
        parts.append(f"\nUse this schema:\n{opts.schema}")

    if opts.example:
        parts.append(f"\nExample:\n{opts.example}")

    return "\n".join(parts)


def format_output_constraints(
    constraints: OutputConstraints | dict[str, Any],
) -> str:
    """Generate instructions for output constraints.

    Args:
        constraints: The output constraints to apply.

    Returns:
        Instructions string describing the constraints.

    Example:
        >>> format_output_constraints({
        ...     "max_length": 500,
        ...     "min_length": 100,
        ...     "no_code_blocks": True,
        ...     "language": "Spanish",
        ...     "tone": "professional",
        ... })
        'Keep your response under 500 characters.\\nProvide at least 100 characters in your response.\\nDo not use code blocks or backticks.\\nRespond in Spanish.\\nUse a professional tone.'
    """
    if isinstance(constraints, dict):
        c = OutputConstraints(
            max_length=constraints.get("max_length") or constraints.get("maxLength"),
            min_length=constraints.get("min_length") or constraints.get("minLength"),
            no_code_blocks=constraints.get("no_code_blocks")
            or constraints.get("noCodeBlocks", False),
            no_markdown=constraints.get("no_markdown")
            or constraints.get("noMarkdown", False),
            language=constraints.get("language"),
            tone=constraints.get("tone"),
        )
    else:
        c = constraints

    lines = []

    if c.max_length is not None:
        lines.append(f"Keep your response under {c.max_length} characters.")

    if c.min_length is not None:
        lines.append(f"Provide at least {c.min_length} characters in your response.")

    if c.no_code_blocks:
        lines.append("Do not use code blocks or backticks.")

    if c.no_markdown:
        lines.append("Do not use Markdown formatting.")

    if c.language:
        lines.append(f"Respond in {c.language}.")

    if c.tone:
        lines.append(f"Use a {c.tone} tone.")

    return "\n".join(lines)


def wrap_output_instruction(instruction: str) -> str:
    """Wrap output instruction in clear delimiter.

    Args:
        instruction: Instruction text.

    Returns:
        Wrapped instruction with output_format tags.

    Example:
        >>> wrap_output_instruction("Respond with JSON only")
        '<output_format>\\nRespond with JSON only\\n</output_format>'
    """
    return f"<output_format>\n{instruction}\n</output_format>"


def create_output_format_section(
    format_type: OutputFormat,
    options: OutputFormatSectionOptions | dict[str, Any] | None = None,
) -> str:
    """Create a complete output format section.

    Args:
        format_type: The output format type.
        options: Section options including format options and constraints.

    Returns:
        A complete output format section string.

    Example:
        >>> create_output_format_section("json", {
        ...     "strict": True,
        ...     "schema": '{ "result": "string" }',
        ...     "constraints": {"max_length": 1000},
        ... })
        '<output_format>\\nRespond with valid JSON only...\\n</output_format>'
    """
    if options is None:
        opts = OutputFormatSectionOptions()
    elif isinstance(options, dict):
        opts = OutputFormatSectionOptions(
            strict=options.get("strict", False),
            schema=options.get("schema"),
            example=options.get("example"),
            constraints=options.get("constraints"),
            wrap=options.get("wrap", True),  # Default to True
        )
    else:
        opts = options

    parts = []

    # Add format instructions
    if format_type == "json":
        parts.append(
            format_json_output(
                JsonOutputOptions(
                    strict=opts.strict,
                    schema=opts.schema,
                    example=opts.example,
                )
            )
        )
    else:
        parts.append(
            format_structured_output(
                format_type,
                StructuredOutputOptions(
                    strict=opts.strict,
                    schema=opts.schema,
                    example=opts.example,
                ),
            )
        )

    # Add constraints if provided
    if opts.constraints:
        constraint_str = format_output_constraints(opts.constraints)
        if constraint_str:
            parts.append("")
            parts.append(constraint_str)

    content = "\n".join(parts)

    if opts.wrap:
        return wrap_output_instruction(content)

    return content


# ─────────────────────────────────────────────────────────────────────────────
# Output Extraction and Cleaning
# ─────────────────────────────────────────────────────────────────────────────

# Pattern for code blocks
_CODE_BLOCK_PATTERN = re.compile(r"```(?:[^\n`]*)?\s*\n?(.*?)\n?```", re.DOTALL)

# Common prefixes to remove
_COMMON_PREFIXES = [
    r"^(?:Sure,?\s*)?(?:here\s+is|here's)\s+(?:the\s+)?(?:result|output|response|answer|JSON|data)[:\s]*",
    r"^(?:The\s+)?(?:result|output|response|answer|JSON|data)\s+is[:\s]*",
    r"^(?:I\s+)?(?:will\s+)?(?:provide|give|return|output)[:\s]*",
    r"^(?:Based\s+on\s+.+?,?\s*)?(?:here\s+is|here's)[:\s]*",
]


def extract_json_from_output(output: str) -> str:
    """Extract JSON from model output that may contain extra text.

    Handles JSON in code blocks, with surrounding text, or raw JSON.

    Args:
        output: The raw model output.

    Returns:
        The extracted JSON string.

    Example:
        >>> extract_json_from_output('Here is the result: {"name": "John"}')
        '{"name": "John"}'

        >>> extract_json_from_output('```json\\n{"name": "John"}\\n```')
        '{"name": "John"}'
    """
    # First, try to extract from code blocks - find the first valid JSON block
    for block in _CODE_BLOCK_PATTERN.finditer(output):
        candidate = block.group(1).strip()
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    # Try to find JSON object or array
    # Look for { ... } or [ ... ], validating each candidate
    stripped = output.strip()

    # Find all potential JSON start positions
    search_start = 0
    while search_start < len(stripped):
        # Find the next start of JSON
        json_start = -1
        for i in range(search_start, len(stripped)):
            if stripped[i] in "{[":
                json_start = i
                break

        if json_start == -1:
            break

        # Find the matching end
        start_char = stripped[json_start]
        end_char = "}" if start_char == "{" else "]"

        depth = 0
        in_string = False
        escape_next = False
        json_end = -1

        for i in range(json_start, len(stripped)):
            char = stripped[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        if json_end != -1:
            candidate = stripped[json_start:json_end]
            # Validate that it's actually valid JSON
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # Not valid JSON, try the next brace pair
                search_start = json_start + 1
                continue
        else:
            # No matching end found, skip this start and keep searching
            search_start = json_start + 1
            continue

    return stripped


def clean_output(output: str) -> str:
    """Clean common prefixes and wrappers from model output.

    Removes common conversational prefixes and code block wrappers.

    Args:
        output: The raw model output.

    Returns:
        The cleaned output string.

    Example:
        >>> clean_output("Sure, here is the result:\\n```json\\n{}\\n```")
        '{}'
    """
    cleaned = output.strip()

    # Remove code blocks first
    match = _CODE_BLOCK_PATTERN.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Try to remove common prefixes
        for pattern in _COMMON_PREFIXES:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()

    return cleaned


def validate_json_output(output: str) -> tuple[bool, str | None]:
    """Validate that output is valid JSON.

    Args:
        output: The output to validate.

    Returns:
        A tuple of (is_valid, error_message).

    Example:
        >>> validate_json_output('{"name": "John"}')
        (True, None)

        >>> validate_json_output('{"name": }')
        (False, 'Expecting value: line 1 column 10 (char 9)')
    """
    try:
        json.loads(output)
        return (True, None)
    except json.JSONDecodeError as e:
        return (False, str(e))
