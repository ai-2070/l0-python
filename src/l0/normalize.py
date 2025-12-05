"""Text normalization utilities for L0.

Provides utilities for normalizing newlines, whitespace, and indentation
in text content. Useful for preparing text for model consumption or
comparing outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


def normalize_newlines(text: str) -> str:
    """Normalize newlines to Unix-style (\\n).

    Converts \\r\\n (Windows) and \\r (old Mac) to \\n.

    Args:
        text: Text to normalize

    Returns:
        Text with normalized newlines
    """
    if not text:
        return text

    # Replace \r\n with \n, then replace remaining \r with \n
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass
class WhitespaceOptions:
    """Options for whitespace normalization."""

    collapse_spaces: bool = False
    """Collapse multiple spaces into one."""

    trim_lines: bool = False
    """Trim whitespace from each line."""

    remove_empty_lines: bool = False
    """Remove empty lines."""


def normalize_whitespace(
    text: str,
    *,
    collapse_spaces: bool = False,
    trim_lines: bool = False,
    remove_empty_lines: bool = False,
) -> str:
    """Normalize whitespace (collapse multiple spaces, trim lines).

    Args:
        text: Text to normalize
        collapse_spaces: Collapse multiple spaces into one
        trim_lines: Trim whitespace from each line
        remove_empty_lines: Remove empty lines

    Returns:
        Text with normalized whitespace
    """
    if not text:
        return text

    result = text

    # Normalize newlines first
    result = normalize_newlines(result)

    # Collapse multiple spaces into one
    if collapse_spaces:
        result = re.sub(r" {2,}", " ", result)

    # Trim each line
    if trim_lines:
        result = "\n".join(line.strip() for line in result.split("\n"))

    # Remove empty lines
    if remove_empty_lines:
        result = "\n".join(line for line in result.split("\n") if line.strip())

    return result


def normalize_indentation(
    text: str,
    mode: Literal["spaces", "tabs"] = "spaces",
    spaces_per_tab: int = 2,
) -> str:
    """Normalize indentation (convert tabs to spaces or vice versa).

    Args:
        text: Text to normalize
        mode: Target indentation mode ("spaces" or "tabs")
        spaces_per_tab: Number of spaces per tab (default: 2)

    Returns:
        Text with normalized indentation
    """
    if not text:
        return text

    lines = normalize_newlines(text).split("\n")

    if mode == "spaces":
        # Convert tabs to spaces
        return "\n".join(line.replace("\t", " " * spaces_per_tab) for line in lines)
    else:
        # Convert leading spaces to tabs
        result_lines = []
        for line in lines:
            # Only convert leading spaces
            match = re.match(r"^ +", line)
            if match:
                spaces = len(match.group())
                tabs = spaces // spaces_per_tab
                remaining = spaces % spaces_per_tab
                converted = "\t" * tabs + " " * remaining + line[spaces:]
                result_lines.append(converted)
            else:
                result_lines.append(line)
        return "\n".join(result_lines)


def dedent(text: str) -> str:
    """Remove common leading indentation from all lines.

    Useful for normalizing code blocks.

    Args:
        text: Text to dedent

    Returns:
        Dedented text
    """
    if not text:
        return text

    lines = normalize_newlines(text).split("\n")

    # Find minimum indentation (excluding empty lines)
    min_indent = float("inf")
    for line in lines:
        if not line.strip():
            continue

        match = re.match(r"^[ \t]*", line)
        indent = len(match.group()) if match else 0
        min_indent = min(min_indent, indent)

    # If no indentation found, return as-is
    if min_indent == float("inf") or min_indent == 0:
        return text

    # Remove the common indentation
    return "\n".join(
        line[int(min_indent) :] if line.strip() else line for line in lines
    )


def indent(text: str, prefix: str | int = 2) -> str:
    """Add indentation to all non-empty lines.

    Args:
        text: Text to indent
        prefix: Indentation to add (string or number of spaces)

    Returns:
        Indented text
    """
    if not text:
        return text

    indent_str = " " * prefix if isinstance(prefix, int) else prefix
    lines = normalize_newlines(text).split("\n")

    return "\n".join(indent_str + line if line.strip() else line for line in lines)


def trim_text(text: str) -> str:
    """Trim whitespace from start and end of text.

    Also removes leading/trailing empty lines.

    Args:
        text: Text to trim

    Returns:
        Trimmed text
    """
    if not text:
        return text

    lines = normalize_newlines(text).split("\n")

    # Remove leading empty lines
    while lines and not lines[0].strip():
        lines.pop(0)

    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines).strip()


@dataclass
class NormalizeOptions:
    """Options for text normalization."""

    newlines: bool = True
    """Normalize newlines to \\n."""

    whitespace: bool = False
    """Collapse multiple spaces."""

    indentation: Literal["spaces", "tabs", False] = False
    """Normalize indentation mode."""

    spaces_per_tab: int = 2
    """Number of spaces per tab."""

    dedent: bool = False
    """Remove common leading indentation."""

    trim: bool = False
    """Trim leading/trailing whitespace."""


def normalize_text(
    text: str,
    *,
    newlines: bool = True,
    whitespace: bool = False,
    indentation: Literal["spaces", "tabs", False] = False,
    spaces_per_tab: int = 2,
    dedent_text: bool = False,
    trim: bool = False,
) -> str:
    """Normalize all whitespace aspects of text.

    Combines multiple normalization operations.

    Args:
        text: Text to normalize
        newlines: Normalize newlines to \\n (default: True)
        whitespace: Collapse multiple spaces (default: False)
        indentation: Normalize indentation mode (default: False)
        spaces_per_tab: Number of spaces per tab (default: 2)
        dedent_text: Remove common leading indentation (default: False)
        trim: Trim leading/trailing whitespace (default: False)

    Returns:
        Fully normalized text
    """
    if not text:
        return text

    result = text

    # Normalize newlines
    if newlines:
        result = normalize_newlines(result)

    # Normalize whitespace
    if whitespace:
        result = normalize_whitespace(
            result,
            collapse_spaces=True,
            trim_lines=False,
            remove_empty_lines=False,
        )

    # Normalize indentation
    if indentation:
        result = normalize_indentation(result, indentation, spaces_per_tab)

    # Dedent
    if dedent_text:
        result = dedent(result)

    # Trim
    if trim:
        result = trim_text(result)

    return result


def ensure_trailing_newline(text: str) -> str:
    """Ensure text ends with a single newline.

    Args:
        text: Text to normalize

    Returns:
        Text with single trailing newline
    """
    if not text:
        return text

    normalized = normalize_newlines(text)

    # Remove any trailing newlines
    trimmed = normalized.rstrip("\n")

    # Add single newline
    return trimmed + "\n"


def remove_trailing_whitespace(text: str) -> str:
    """Remove all trailing whitespace from each line.

    Args:
        text: Text to process

    Returns:
        Text with trailing whitespace removed
    """
    if not text:
        return text

    return "\n".join(line.rstrip() for line in normalize_newlines(text).split("\n"))


def normalize_for_model(text: str) -> str:
    """Normalize line endings and ensure consistent formatting.

    Good for preparing text for model consumption.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return text

    return normalize_text(
        text,
        newlines=True,
        whitespace=True,
        trim=True,
    )


def is_whitespace_only(text: str) -> bool:
    """Check if text contains only whitespace.

    Args:
        text: Text to check

    Returns:
        True if text is empty or only whitespace
    """
    if not text:
        return True
    return bool(re.match(r"^[\s\r\n\t]*$", text))


def count_lines(text: str) -> int:
    """Count lines in text.

    Args:
        text: Text to count lines in

    Returns:
        Number of lines
    """
    if not text:
        return 0
    return len(normalize_newlines(text).split("\n"))


def get_line(text: str, line_index: int) -> str | None:
    """Get line at specific index.

    Args:
        text: Text to extract from
        line_index: Zero-based line index

    Returns:
        Line content or None if out of bounds
    """
    if not text:
        return None

    lines = normalize_newlines(text).split("\n")
    if line_index < 0 or line_index >= len(lines):
        return None

    return lines[line_index]


def replace_line(text: str, line_index: int, new_line: str) -> str:
    """Replace line at specific index.

    Args:
        text: Text to modify
        line_index: Zero-based line index
        new_line: New line content

    Returns:
        Modified text
    """
    if not text:
        return text

    lines = normalize_newlines(text).split("\n")
    if line_index < 0 or line_index >= len(lines):
        return text

    lines[line_index] = new_line
    return "\n".join(lines)
