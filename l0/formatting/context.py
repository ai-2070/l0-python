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

DelimiterType = Literal["xml", "markdown", "brackets"]


@dataclass
class ContextOptions:
    """Options for formatting context."""

    label: str = "context"
    delimiter: DelimiterType = "xml"


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


def format_context(
    content: str,
    *,
    label: str = "context",
    delimiter: DelimiterType = "xml",
) -> str:
    """Wrap content with proper delimiters.

    Args:
        content: The content to wrap.
        label: The label for the context section.
        delimiter: The delimiter type - "xml", "markdown", or "brackets".

    Returns:
        The formatted context string.

    Example:
        >>> format_context("User manual content", label="Documentation")
        '<documentation>\\nUser manual content\\n</documentation>'

        >>> format_context("Content", label="Context", delimiter="markdown")
        '# Context\\n\\nContent'

        >>> format_context("Content", delimiter="brackets")
        '[CONTEXT]\\n==============================\\nContent\\n=============================='
    """
    label_lower = label.lower()
    label_upper = label.upper()

    # Escape content to prevent delimiter injection
    escaped_content = escape_delimiters(content, delimiter)

    if delimiter == "xml":
        safe_label = _sanitize_xml_tag(label_lower)
        return f"<{safe_label}>\n{escaped_content}\n</{safe_label}>"
    elif delimiter == "markdown":
        return f"# {label}\n\n{escaped_content}"
    elif delimiter == "brackets":
        separator = "=" * 30
        return f"[{label_upper}]\n{separator}\n{escaped_content}\n{separator}"
    return content


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
            label = item.get("label", "context")
        else:
            content = item.content
            label = item.label
        formatted.append(format_context(content, label=label, delimiter=delimiter))
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
        '<document>\\n<metadata>\\n<title>Q4 Report</title>\\n<author>Team</author>\\n</metadata>\\n<content>\\nReport content\\n</content>\\n</document>'
    """
    if metadata is None:
        return format_context(content, label="document", delimiter=delimiter)

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

    if delimiter == "xml":
        meta_parts = []
        if meta.title:
            meta_parts.append(f"<title>{_escape_xml(meta.title)}</title>")
        if meta.author:
            meta_parts.append(f"<author>{_escape_xml(meta.author)}</author>")
        if meta.date:
            meta_parts.append(f"<date>{_escape_xml(meta.date)}</date>")
        if meta.source:
            meta_parts.append(f"<source>{_escape_xml(meta.source)}</source>")
        for key, value in meta.extra.items():
            safe_key = _sanitize_xml_tag(key)
            safe_value = _escape_xml(str(value))
            meta_parts.append(f"<{safe_key}>{safe_value}</{safe_key}>")

        if meta_parts:
            meta_section = "<metadata>\n" + "\n".join(meta_parts) + "\n</metadata>"
            safe_content = _escape_xml(content)
            return f"<document>\n{meta_section}\n<content>\n{safe_content}\n</content>\n</document>"
        safe_content = _escape_xml(content)
        return f"<document>\n<content>\n{safe_content}\n</content>\n</document>"

    elif delimiter == "markdown":
        meta_parts = []
        if meta.title:
            meta_parts.append(f"**Title:** {meta.title}")
        if meta.author:
            meta_parts.append(f"**Author:** {meta.author}")
        if meta.date:
            meta_parts.append(f"**Date:** {meta.date}")
        if meta.source:
            meta_parts.append(f"**Source:** {meta.source}")
        for key, value in meta.extra.items():
            meta_parts.append(f"**{key.title()}:** {value}")

        if meta_parts:
            return "# Document\n\n" + "\n".join(meta_parts) + "\n\n---\n\n" + content
        return "# Document\n\n" + content

    elif delimiter == "brackets":
        separator = "=" * 30
        meta_parts = []
        if meta.title:
            meta_parts.append(f"Title: {escape_delimiters(meta.title, 'brackets')}")
        if meta.author:
            meta_parts.append(f"Author: {escape_delimiters(meta.author, 'brackets')}")
        if meta.date:
            meta_parts.append(f"Date: {escape_delimiters(meta.date, 'brackets')}")
        if meta.source:
            meta_parts.append(f"Source: {escape_delimiters(meta.source, 'brackets')}")
        for key, value in meta.extra.items():
            meta_parts.append(
                f"{key.title()}: {escape_delimiters(str(value), 'brackets')}"
            )

        escaped_content = escape_delimiters(content, "brackets")
        if meta_parts:
            meta_section = "\n".join(meta_parts)
            return f"[DOCUMENT]\n{separator}\n{meta_section}\n{separator}\n{escaped_content}\n{separator}"
        return f"[DOCUMENT]\n{separator}\n{escaped_content}\n{separator}"

    return content


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
        '<system_instructions>\\nYou are a helpful assistant.\\n</system_instructions>'
    """
    # Escape instructions to prevent delimiter injection
    escaped = escape_delimiters(instructions, delimiter)

    if delimiter == "xml":
        return f"<system_instructions>\n{escaped}\n</system_instructions>"
    elif delimiter == "markdown":
        return f"## System Instructions\n\n{escaped}"
    elif delimiter == "brackets":
        separator = "=" * 30
        return f"[SYSTEM INSTRUCTIONS]\n{separator}\n{escaped}\n{separator}"
    return instructions
