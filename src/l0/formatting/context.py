"""Context formatting utilities for L0.

This module provides functions for formatting context, documents, and
instructions with proper delimiters for LLM consumption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape as html_escape
from typing import Any, Literal


def _escape_xml(value: str) -> str:
    """Escape a string for safe XML output."""
    return html_escape(value, quote=True)


def _sanitize_xml_tag(key: str) -> str:
    """Sanitize a string to be a valid XML tag name.

    Only allows alphanumeric characters, underscores, and hyphens.
    XML tag names must start with a letter or underscore.
    Returns 'extra' if the result would be empty.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", key)
    # XML tag names must start with a letter or underscore, not digit or hyphen
    if sanitized and not re.match(r"[A-Za-z_]", sanitized[0]):
        sanitized = f"extra{sanitized}"
    return sanitized or "extra"


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

DelimiterType = Literal["xml", "markdown", "brackets", "none"]


@dataclass
class ContextOptions:
    """Options for formatting context."""

    label: str = "Context"
    delimiter: DelimiterType = "xml"
    dedent: bool = True
    normalize: bool = True
    custom_delimiter_start: str | None = None
    custom_delimiter_end: str | None = None


@dataclass
class DocumentMetadata:
    """Metadata for a document."""

    title: str | None = None
    author: str | None = None
    date: str | None = None
    source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextItem:
    """An item for multiple context formatting."""

    content: str
    label: str = "context"


# ─────────────────────────────────────────────────────────────────────────────
# Delimiter Escaping
# ─────────────────────────────────────────────────────────────────────────────


def escape_delimiters(content: str, delimiter: DelimiterType = "xml") -> str:
    """Escape delimiters in content to prevent injection attacks.

    Args:
        content: The content to escape.
        delimiter: The delimiter type to escape for.

    Returns:
        The escaped content.

    Example:
        >>> escape_delimiters("<script>alert('xss')</script>", "xml")
        "&lt;script&gt;alert('xss')&lt;/script&gt;"
    """
    if delimiter == "xml":
        return content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    elif delimiter == "markdown":
        # Escape markdown heading markers and code fences
        lines = content.split("\n")
        escaped = []
        for line in lines:
            if line.startswith("#"):
                line = "\\" + line
            if line.startswith("```"):
                line = "\\" + line
            escaped.append(line)
        return "\n".join(escaped)
    elif delimiter == "brackets":
        # Escape bracket markers
        return content.replace("[", "\\[").replace("]", "\\]")
    return content


def unescape_delimiters(content: str, delimiter: DelimiterType = "xml") -> str:
    """Unescape delimiters in content.

    Args:
        content: The content to unescape.
        delimiter: The delimiter type to unescape.

    Returns:
        The unescaped content.

    Example:
        >>> unescape_delimiters("&lt;div&gt;", "xml")
        '<div>'
    """
    if delimiter == "xml":
        return content.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    elif delimiter == "markdown":
        lines = content.split("\n")
        unescaped = []
        for line in lines:
            if line.startswith("\\#"):
                line = line[1:]
            if line.startswith("\\```"):
                line = line[1:]
            unescaped.append(line)
        return "\n".join(unescaped)
    elif delimiter == "brackets":
        return content.replace("\\[", "[").replace("\\]", "]")
    return content


# ─────────────────────────────────────────────────────────────────────────────
# Context Formatting
# ─────────────────────────────────────────────────────────────────────────────


def _dedent_content(content: str) -> str:
    """Remove common leading whitespace from content."""
    import textwrap

    return textwrap.dedent(content)


def _normalize_content(content: str) -> str:
    """Normalize whitespace in content."""
    # Normalize line endings and collapse multiple blank lines
    import re

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def format_context(
    content: str,
    *,
    label: str = "Context",
    delimiter: DelimiterType = "xml",
    dedent: bool = True,
    normalize: bool = True,
    custom_delimiter_start: str | None = None,
    custom_delimiter_end: str | None = None,
) -> str:
    """Wrap content with proper delimiters.

    Args:
        content: The content to wrap.
        label: The label for the context section (default: "Context").
        delimiter: The delimiter type - "xml", "markdown", "brackets", or "none".
        dedent: Whether to remove common leading whitespace (default: True).
        normalize: Whether to normalize whitespace (default: True).
        custom_delimiter_start: Custom start delimiter (overrides delimiter style).
        custom_delimiter_end: Custom end delimiter (overrides delimiter style).

    Returns:
        The formatted context string.

    Example:
        >>> format_context("User manual content", label="Documentation")
        '<documentation>\\nUser manual content\\n</documentation>'

        >>> format_context("Content", label="Context", delimiter="markdown")
        '# Context\\n\\nContent'

        >>> format_context("Content", delimiter="brackets")
        '[CONTEXT]\\n==============================\\nContent\\n=============================='

        >>> format_context("Content", delimiter="none")
        'Content'

        >>> format_context("Content", custom_delimiter_start="<<<START>>>", custom_delimiter_end="<<<END>>>")
        '<<<START>>>\\nContent\\n<<<END>>>'
    """
    if not content or not content.strip():
        return ""

    # Process content
    processed = content
    if dedent:
        processed = _dedent_content(processed)
    if normalize:
        processed = _normalize_content(processed)

    # Custom delimiters override delimiter style
    if custom_delimiter_start and custom_delimiter_end:
        return f"{custom_delimiter_start}\n{processed}\n{custom_delimiter_end}"

    # Escape content to prevent injection attacks
    escaped = escape_delimiters(processed, delimiter)

    label_lower = label.lower().replace(" ", "_")
    label_upper = label.upper()

    if delimiter == "xml":
        safe_label = _sanitize_xml_tag(label_lower)
        return f"<{safe_label}>\n{escaped}\n</{safe_label}>"
    elif delimiter == "markdown":
        escaped_label = escape_delimiters(label, delimiter)
        return f"# {escaped_label}\n\n{escaped}"
    elif delimiter == "brackets":
        separator = "=" * max(20, len(label) + 10)
        return f"[{label_upper}]\n{separator}\n{escaped}\n{separator}"
    elif delimiter == "none":
        return processed  # No escaping for "none" delimiter

    return processed


def format_multiple_contexts(
    items: list[ContextItem] | list[dict[str, str]],
    *,
    delimiter: DelimiterType = "xml",
) -> str:
    """Format multiple contexts with the specified delimiter.

    Args:
        items: List of ContextItem objects or dicts with 'content' and 'label'.
        delimiter: The delimiter type for all contexts.

    Returns:
        The formatted contexts as a single string.

    Example:
        >>> items = [
        ...     {"content": "Document 1", "label": "Doc1"},
        ...     {"content": "Document 2", "label": "Doc2"},
        ... ]
        >>> format_multiple_contexts(items)
        '<doc1>\\nDocument 1\\n</doc1>\\n\\n<doc2>\\nDocument 2\\n</doc2>'
    """
    formatted = []
    for item in items:
        if isinstance(item, dict):
            content = item.get("content", "")
            label = item.get("label", "Context")
        else:
            content = item.content
            label = item.label

        # Filter empty items
        if not content or not content.strip():
            continue

        result = format_context(content, label=label, delimiter=delimiter)
        if result:
            formatted.append(result)

    return "\n\n".join(formatted)


def format_document(
    content: str,
    metadata: DocumentMetadata | dict[str, Any] | None = None,
    *,
    delimiter: DelimiterType = "xml",
) -> str:
    """Format a document with optional metadata.

    Args:
        content: The document content.
        metadata: Document metadata (title, author, date, source, etc.).
        delimiter: The delimiter type for formatting.

    Returns:
        The formatted document string.

    Example:
        >>> format_document("Report content", {"title": "Q4 Report", "author": "Team"})
        '<q4_report>\\ntitle: Q4 Report\\nauthor: Team\\n\\nReport content\\n</q4_report>'
    """
    if not content or not content.strip():
        return ""

    if metadata is None:
        return format_context(content, label="Document", delimiter=delimiter)

    if isinstance(metadata, dict):
        meta = DocumentMetadata(
            title=metadata.get("title"),
            author=metadata.get("author"),
            date=metadata.get("date"),
            source=metadata.get("source"),
            extra={
                k: v
                for k, v in metadata.items()
                if k not in ("title", "author", "date", "source")
            },
        )
    else:
        meta = metadata

    # Build metadata lines (filter empty values)
    meta_lines = []
    if meta.title and meta.title.strip():
        meta_lines.append(f"title: {meta.title}")
    if meta.author and meta.author.strip():
        meta_lines.append(f"author: {meta.author}")
    if meta.date and meta.date.strip():
        meta_lines.append(f"date: {meta.date}")
    if meta.source and meta.source.strip():
        meta_lines.append(f"source: {meta.source}")
    for key, value in meta.extra.items():
        if value and str(value).strip():
            meta_lines.append(f"{key}: {value}")

    # Use title as label if provided
    label = meta.title if meta.title else "Document"

    # Build combined content with metadata at top
    if meta_lines:
        combined_content = "\n".join(meta_lines) + "\n\n" + content
    else:
        combined_content = content

    return format_context(combined_content, label=label, delimiter=delimiter)


def format_instructions(
    instructions: str,
    *,
    delimiter: DelimiterType = "xml",
) -> str:
    """Format system instructions with proper delimiters.

    Args:
        instructions: The system instructions.
        delimiter: The delimiter type for formatting.

    Returns:
        The formatted instructions string.

    Example:
        >>> format_instructions("You are a helpful assistant.")
        '<instructions>\\nYou are a helpful assistant.\\n</instructions>'
    """
    return format_context(instructions, label="Instructions", delimiter=delimiter)
