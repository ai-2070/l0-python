"""String manipulation utilities for L0.

This module provides common string manipulation functions for working with
LLM inputs and outputs.
"""

from __future__ import annotations

import re
import textwrap
from typing import Literal

# ─────────────────────────────────────────────────────────────────────────────
# Escape/Unescape Functions
# ─────────────────────────────────────────────────────────────────────────────


def escape(s: str) -> str:
    """Escape special characters in a string.

    Escapes newlines, tabs, carriage returns, and quotes.

    Args:
        s: The string to escape.

    Returns:
        The escaped string.

    Example:
        >>> escape('Hello\\n"World"')
        'Hello\\\\n\\\\"World\\\\"'
    """
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace('"', '\\"')
    )


def unescape(s: str) -> str:
    """Unescape special characters in a string.

    Unescapes newlines, tabs, carriage returns, and quotes.

    Args:
        s: The string to unescape.

    Returns:
        The unescaped string.

    Example:
        >>> unescape('Hello\\\\nWorld')
        'Hello\\nWorld'
    """
    result = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char == "n":
                result.append("\n")
                i += 2
            elif next_char == "r":
                result.append("\r")
                i += 2
            elif next_char == "t":
                result.append("\t")
                i += 2
            elif next_char == '"':
                result.append('"')
                i += 2
            elif next_char == "\\":
                result.append("\\")
                i += 2
            else:
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def escape_html(s: str) -> str:
    """Escape HTML entities in a string.

    Args:
        s: The string to escape.

    Returns:
        The string with HTML entities escaped.

    Example:
        >>> escape_html('<div>')
        '&lt;div&gt;'
    """
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def unescape_html(s: str) -> str:
    """Unescape HTML entities in a string.

    Args:
        s: The string to unescape.

    Returns:
        The string with HTML entities unescaped.

    Example:
        >>> unescape_html('&lt;div&gt;')
        '<div>'
        >>> unescape_html('&#x27;')  # Also handles hex variant
        "'"
    """
    # Replace &amp; last to prevent double-decoding (e.g., &amp;lt; -> &lt; -> <)
    return (
        s.replace("&#39;", "'")
        .replace("&#x27;", "'")  # Hex variant for apostrophe
        .replace("&quot;", '"')
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
    )


def escape_regex(s: str) -> str:
    """Escape regex special characters in a string.

    Args:
        s: The string to escape.

    Returns:
        The string with regex special characters escaped.

    Example:
        >>> escape_regex('file.txt')
        'file\\\\.txt'
    """
    return re.escape(s)


# ─────────────────────────────────────────────────────────────────────────────
# Sanitization Functions
# ─────────────────────────────────────────────────────────────────────────────


def sanitize(s: str) -> str:
    """Remove control characters from a string.

    Removes all control characters except newlines, tabs, and carriage returns.

    Args:
        s: The string to sanitize.

    Returns:
        The sanitized string.

    Example:
        >>> sanitize('Hello\\x00World')
        'HelloWorld'
    """
    # Remove control characters (0x00-0x1F) except \t (0x09), \n (0x0A), \r (0x0D)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)


def trim(s: str) -> str:
    """Trim whitespace from both ends of a string.

    Args:
        s: The string to trim.

    Returns:
        The trimmed string.

    Example:
        >>> trim('  Hello  ')
        'Hello'
    """
    return s.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Truncation Functions
# ─────────────────────────────────────────────────────────────────────────────


def truncate(s: str, max_length: int, suffix: str = "...") -> str:
    """Truncate a string to a maximum length with a suffix.

    Args:
        s: The string to truncate.
        max_length: The maximum length including the suffix.
        suffix: The suffix to append when truncating. Defaults to "...".

    Returns:
        The truncated string with suffix, or the original if shorter than max.

    Example:
        >>> truncate('Hello World', 8)
        'Hello...'
    """
    if len(s) <= max_length:
        return s
    if max_length <= len(suffix):
        return suffix[:max_length]
    return s[: max_length - len(suffix)] + suffix


def truncate_words(s: str, max_length: int, suffix: str = "...") -> str:
    """Truncate a string at word boundaries.

    Args:
        s: The string to truncate.
        max_length: The maximum length including the suffix.
        suffix: The suffix to append when truncating. Defaults to "...".

    Returns:
        The truncated string at word boundary with suffix.

    Example:
        >>> truncate_words('Hello World Test', 12)
        'Hello...'
    """
    if len(s) <= max_length:
        return s
    if max_length <= len(suffix):
        return suffix[:max_length]

    # Find the last space before the cutoff point
    cutoff = max_length - len(suffix)
    if cutoff <= 0:
        return suffix[:max_length]

    # If the cutoff is in the middle of a word, find the previous word boundary
    if cutoff < len(s) and s[cutoff] != " ":
        last_space = s.rfind(" ", 0, cutoff + 1)
        if last_space > 0:
            cutoff = last_space

    return s[:cutoff].rstrip() + suffix


# ─────────────────────────────────────────────────────────────────────────────
# Formatting Functions
# ─────────────────────────────────────────────────────────────────────────────


def wrap(s: str, width: int) -> str:
    """Wrap text to a specified width.

    Args:
        s: The string to wrap.
        width: The maximum line width.

    Returns:
        The wrapped string with line breaks.

    Example:
        >>> wrap('Hello World Test', 10)
        'Hello\\nWorld Test'
    """
    return textwrap.fill(s, width=width, break_long_words=True, break_on_hyphens=True)


Alignment = Literal["left", "right", "center"]


def pad(
    s: str,
    length: int,
    char: str = " ",
    align: Alignment = "left",
) -> str:
    """Pad a string to a specified length.

    Args:
        s: The string to pad.
        length: The target length.
        char: The character to use for padding. Defaults to space.
        align: The alignment - "left", "right", or "center". Defaults to "left".

    Returns:
        The padded string.

    Example:
        >>> pad('Hi', 10)
        'Hi        '
        >>> pad('Hi', 10, ' ', 'right')
        '        Hi'
        >>> pad('Hi', 10, ' ', 'center')
        '    Hi    '
    """
    if len(s) >= length:
        return s

    padding_needed = length - len(s)
    pad_char = char[0] if char else " "

    if align == "right":
        return pad_char * padding_needed + s
    elif align == "center":
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return pad_char * left_pad + s + pad_char * right_pad
    else:  # left
        return s + pad_char * padding_needed


# ─────────────────────────────────────────────────────────────────────────────
# ANSI Functions
# ─────────────────────────────────────────────────────────────────────────────

# Pattern to match ANSI escape sequences (CSI sequences per ECMA-48)
# Matches ESC [ followed by parameter bytes, intermediate bytes, and final byte
_ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def remove_ansi(s: str) -> str:
    """Remove ANSI escape codes from a string.

    Args:
        s: The string containing ANSI codes.

    Returns:
        The string with ANSI codes removed.

    Example:
        >>> remove_ansi('\\x1b[31mRed\\x1b[0m')
        'Red'
    """
    return _ANSI_PATTERN.sub("", s)
