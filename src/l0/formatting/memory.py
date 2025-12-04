"""Memory formatting utilities for L0.

This module provides functions for formatting conversation history and
managing memory entries for LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


def _escape_xml_attr(value: Any) -> str:
    """Escape a value for use in an XML attribute."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

MemoryRole = Literal["user", "assistant", "system"]
MemoryStyle = Literal["conversational", "structured", "compact"]


@dataclass
class MemoryEntry:
    """A single memory/conversation entry."""

    role: MemoryRole
    content: str
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryFormatOptions:
    """Options for formatting memory."""

    max_entries: int | None = None
    include_timestamps: bool = False
    include_metadata: bool = False
    style: MemoryStyle = "conversational"


# ─────────────────────────────────────────────────────────────────────────────
# Memory Entry Creation
# ─────────────────────────────────────────────────────────────────────────────


def create_memory_entry(
    role: MemoryRole,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> MemoryEntry:
    """Create a timestamped memory entry.

    Args:
        role: The role - "user", "assistant", or "system".
        content: The message content.
        metadata: Optional metadata for the entry.

    Returns:
        A MemoryEntry with the current timestamp.

    Example:
        >>> entry = create_memory_entry("user", "Hello", {"source": "chat"})
        >>> entry.role
        'user'
        >>> entry.content
        'Hello'
    """
    return MemoryEntry(
        role=role,
        content=content,
        timestamp=datetime.now(),
        metadata=metadata or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memory Formatting
# ─────────────────────────────────────────────────────────────────────────────


def _format_role(role: MemoryRole, style: MemoryStyle) -> str:
    """Format a role based on the style."""
    if style == "compact":
        role_map = {"user": "U", "assistant": "A", "system": "S"}
        return role_map.get(role, role[0].upper())
    elif style == "conversational":
        return role.title()
    return role


def format_memory(
    memory: list[MemoryEntry] | list[dict[str, Any]],
    options: MemoryFormatOptions | dict[str, Any] | None = None,
) -> str:
    """Format conversation history for model context.

    Args:
        memory: List of MemoryEntry objects or dicts with 'role' and 'content'.
        options: Formatting options (max_entries, style, etc.).

    Returns:
        The formatted memory string.

    Example:
        >>> memory = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ]
        >>> format_memory(memory)
        'User: Hello\\n\\nAssistant: Hi there!'

        >>> format_memory(memory, {"style": "structured"})
        '<conversation_history>\\n  <message role="user">Hello</message>\\n  <message role="assistant">Hi there!</message>\\n</conversation_history>'

        >>> format_memory(memory, {"style": "compact"})
        'U: Hello\\nA: Hi there!'
    """
    if options is None:
        opts = MemoryFormatOptions()
    elif isinstance(options, dict):
        opts = MemoryFormatOptions(
            max_entries=options.get("max_entries"),
            include_timestamps=options.get("include_timestamps", False),
            include_metadata=options.get("include_metadata", False),
            style=options.get("style", "conversational"),
        )
    else:
        opts = options

    # Convert dicts to MemoryEntry objects
    entries: list[MemoryEntry] = []
    for item in memory:
        if isinstance(item, dict):
            entries.append(
                MemoryEntry(
                    role=item.get("role", "user"),
                    content=item.get("content", ""),
                    timestamp=item.get("timestamp"),
                    metadata=item.get("metadata", {}),
                )
            )
        else:
            entries.append(item)

    # Limit entries if specified
    if opts.max_entries is not None:
        if opts.max_entries <= 0:
            entries = []
        elif len(entries) > opts.max_entries:
            entries = entries[-opts.max_entries :]

    if opts.style == "structured":
        return _format_structured(entries, opts)
    elif opts.style == "compact":
        return _format_compact(entries, opts)
    else:
        return _format_conversational(entries, opts)


def _format_conversational(
    entries: list[MemoryEntry],
    opts: MemoryFormatOptions,
) -> str:
    """Format memory in conversational style."""
    lines = []
    for entry in entries:
        parts = []
        role = _format_role(entry.role, "conversational")

        if opts.include_timestamps and entry.timestamp:
            parts.append(f"[{entry.timestamp.isoformat()}]")

        parts.append(f"{role}: {entry.content}")

        if opts.include_metadata and entry.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in entry.metadata.items())
            parts.append(f"({meta_str})")

        lines.append(" ".join(parts))

    return "\n\n".join(lines)


def _format_structured(
    entries: list[MemoryEntry],
    opts: MemoryFormatOptions,
) -> str:
    """Format memory in structured XML style."""
    messages = []
    for entry in entries:
        attrs = [f'role="{entry.role}"']

        if opts.include_timestamps and entry.timestamp:
            attrs.append(f'timestamp="{entry.timestamp.isoformat()}"')

        if opts.include_metadata and entry.metadata:
            for key, value in entry.metadata.items():
                safe_key = _escape_xml_attr(key)
                safe_value = _escape_xml_attr(value)
                attrs.append(f'{safe_key}="{safe_value}"')

        attr_str = " ".join(attrs)
        safe_content = (
            entry.content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        messages.append(f"  <message {attr_str}>{safe_content}</message>")

    return (
        "<conversation_history>\n" + "\n".join(messages) + "\n</conversation_history>"
    )


def _format_compact(
    entries: list[MemoryEntry],
    opts: MemoryFormatOptions,
) -> str:
    """Format memory in compact style."""
    lines = []
    for entry in entries:
        role = _format_role(entry.role, "compact")
        line = f"{role}: {entry.content}"

        if opts.include_timestamps and entry.timestamp:
            line = f"[{entry.timestamp.strftime('%H:%M')}] {line}"

        lines.append(line)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Utilities
# ─────────────────────────────────────────────────────────────────────────────


def merge_memory(
    *memories: list[MemoryEntry] | list[dict[str, Any]],
) -> list[MemoryEntry]:
    """Merge multiple memory lists and sort by timestamp.

    Args:
        *memories: Variable number of memory lists to merge.

    Returns:
        A single merged and sorted list of MemoryEntry objects.

    Example:
        >>> m1 = [create_memory_entry("user", "First")]
        >>> m2 = [create_memory_entry("assistant", "Second")]
        >>> merged = merge_memory(m1, m2)
        >>> len(merged)
        2
    """
    all_entries: list[MemoryEntry] = []

    for memory in memories:
        for item in memory:
            if isinstance(item, dict):
                all_entries.append(
                    MemoryEntry(
                        role=item.get("role", "user"),
                        content=item.get("content", ""),
                        timestamp=item.get("timestamp"),
                        metadata=item.get("metadata", {}),
                    )
                )
            else:
                all_entries.append(item)

    # Sort by timestamp (entries without timestamp go to the end)
    # Use float timestamp to avoid comparing timezone-aware and naive datetimes
    def sort_key(e: MemoryEntry) -> tuple[int, float]:
        if e.timestamp is None:
            return (1, 0.0)
        return (0, e.timestamp.timestamp())

    return sorted(all_entries, key=sort_key)


def filter_memory_by_role(
    memory: list[MemoryEntry] | list[dict[str, Any]],
    role: MemoryRole,
) -> list[MemoryEntry]:
    """Filter memory entries by role.

    Args:
        memory: The memory list to filter.
        role: The role to filter for.

    Returns:
        A list of MemoryEntry objects matching the role.

    Example:
        >>> memory = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi"},
        ... ]
        >>> user_msgs = filter_memory_by_role(memory, "user")
        >>> len(user_msgs)
        1
    """
    result: list[MemoryEntry] = []

    for item in memory:
        if isinstance(item, dict):
            if item.get("role") == role:
                result.append(
                    MemoryEntry(
                        role=item.get("role", "user"),
                        content=item.get("content", ""),
                        timestamp=item.get("timestamp"),
                        metadata=item.get("metadata", {}),
                    )
                )
        else:
            if item.role == role:
                result.append(item)

    return result


def get_last_n_entries(
    memory: list[MemoryEntry] | list[dict[str, Any]],
    n: int,
) -> list[MemoryEntry]:
    """Get the last N entries from memory.

    Args:
        memory: The memory list.
        n: The number of entries to return.

    Returns:
        The last N entries as MemoryEntry objects.

    Example:
        >>> memory = [{"role": "user", "content": str(i)} for i in range(10)]
        >>> recent = get_last_n_entries(memory, 3)
        >>> len(recent)
        3
    """
    entries: list[MemoryEntry] = []

    for item in memory:
        if isinstance(item, dict):
            entries.append(
                MemoryEntry(
                    role=item.get("role", "user"),
                    content=item.get("content", ""),
                    timestamp=item.get("timestamp"),
                    metadata=item.get("metadata", {}),
                )
            )
        else:
            entries.append(item)

    if n <= 0:
        return []
    if n >= len(entries):
        return entries
    return entries[-n:]


def calculate_memory_size(memory: list[MemoryEntry] | list[dict[str, Any]]) -> int:
    """Calculate the total character count of memory.

    Args:
        memory: The memory list.

    Returns:
        The total character count.

    Example:
        >>> memory = [{"role": "user", "content": "Hello"}]
        >>> calculate_memory_size(memory)
        5
    """
    total = 0
    for item in memory:
        if isinstance(item, dict):
            total += len(item.get("content", ""))
        else:
            total += len(item.content)
    return total


def truncate_memory(
    memory: list[MemoryEntry] | list[dict[str, Any]],
    max_size: int,
) -> list[MemoryEntry]:
    """Truncate memory to fit within a character limit.

    Removes entries from the beginning until the total size is under the limit.
    Preserves the most recent entries.

    Args:
        memory: The memory list.
        max_size: The maximum total character count.

    Returns:
        The truncated memory as MemoryEntry objects.

    Example:
        >>> memory = [{"role": "user", "content": "A" * 100} for _ in range(10)]
        >>> truncated = truncate_memory(memory, 500)
        >>> calculate_memory_size(truncated) <= 500
        True
    """
    entries: list[MemoryEntry] = []

    for item in memory:
        if isinstance(item, dict):
            entries.append(
                MemoryEntry(
                    role=item.get("role", "user"),
                    content=item.get("content", ""),
                    timestamp=item.get("timestamp"),
                    metadata=item.get("metadata", {}),
                )
            )
        else:
            entries.append(item)

    # Calculate total size
    total_size = sum(len(e.content) for e in entries)

    # Remove entries from the beginning until under the limit
    while total_size > max_size and len(entries) > 0:
        removed = entries.pop(0)
        total_size -= len(removed.content)

    return entries
