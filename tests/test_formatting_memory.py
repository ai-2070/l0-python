"""Tests for l0.formatting.memory module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from l0.formatting.memory import (
    MemoryEntry,
    MemoryFormatOptions,
    calculate_memory_size,
    create_memory_entry,
    filter_memory_by_role,
    format_memory,
    get_last_n_entries,
    merge_memory,
    truncate_memory,
)


class TestCreateMemoryEntry:
    """Tests for create_memory_entry function."""

    def test_create_basic_entry(self):
        entry = create_memory_entry("user", "Hello")
        assert entry.role == "user"
        assert entry.content == "Hello"
        assert entry.timestamp is not None

    def test_create_entry_with_metadata(self):
        entry = create_memory_entry("assistant", "Hi!", {"source": "chat"})
        assert entry.role == "assistant"
        assert entry.content == "Hi!"
        assert entry.metadata == {"source": "chat"}

    def test_entry_has_timestamp(self):
        before = datetime.now()
        entry = create_memory_entry("user", "Test")
        after = datetime.now()
        assert entry.timestamp is not None
        assert before <= entry.timestamp <= after


class TestFormatMemory:
    """Tests for format_memory function."""

    def test_conversational_style_default(self):
        memory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = format_memory(memory)
        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result

    def test_structured_style(self):
        memory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = format_memory(memory, {"style": "structured"})
        assert "<conversation_history>" in result
        assert '<message role="user">Hello</message>' in result
        assert '<message role="assistant">Hi!</message>' in result
        assert "</conversation_history>" in result

    def test_compact_style(self):
        memory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = format_memory(memory, {"style": "compact"})
        assert "U: Hello" in result
        assert "A: Hi!" in result

    def test_max_entries(self):
        memory = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
        result = format_memory(memory, {"max_entries": 3})
        assert "Message 7" in result
        assert "Message 8" in result
        assert "Message 9" in result
        assert "Message 0" not in result

    def test_max_entries_zero_returns_empty(self):
        memory = [{"role": "user", "content": f"Message {i}"} for i in range(5)]
        result = format_memory(memory, {"max_entries": 0})
        assert result == ""

    def test_max_entries_negative_returns_empty(self):
        memory = [{"role": "user", "content": f"Message {i}"} for i in range(5)]
        result = format_memory(memory, {"max_entries": -1})
        assert result == ""

    def test_with_timestamps(self):
        now = datetime.now()
        memory = [
            MemoryEntry(role="user", content="Hello", timestamp=now),
        ]
        result = format_memory(memory, {"include_timestamps": True})
        assert now.isoformat() in result

    def test_with_metadata(self):
        memory = [
            MemoryEntry(role="user", content="Hello", metadata={"source": "web"}),
        ]
        result = format_memory(memory, {"include_metadata": True})
        assert "source=web" in result

    def test_structured_metadata_escapes_special_chars(self):
        """Test that metadata with special XML characters is properly escaped."""
        memory = [
            MemoryEntry(
                role="user",
                content="Hello",
                metadata={"key<>&\"'": "value<>&\"'"},
            ),
        ]
        result = format_memory(
            memory, {"style": "structured", "include_metadata": True}
        )
        # Verify special characters are escaped
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result
        assert "&#39;" in result
        # Verify raw special characters are not present in attributes
        assert "key<>&\"'" not in result

    def test_structured_content_escapes_special_chars(self):
        """Test that message content with special XML characters is properly escaped."""
        memory = [
            MemoryEntry(
                role="user",
                content="Check if x < 10 && y > 5",
            ),
        ]
        result = format_memory(memory, {"style": "structured"})
        # Verify special characters in content are escaped
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        # Verify raw special characters are not in content
        assert "x < 10" not in result
        assert "y > 5" not in result

    def test_format_with_options_object(self):
        memory = [{"role": "user", "content": "Test"}]
        opts = MemoryFormatOptions(style="compact")
        result = format_memory(memory, opts)
        assert "U: Test" in result

    def test_system_role(self):
        memory = [{"role": "system", "content": "You are helpful"}]
        result = format_memory(memory)
        assert "System: You are helpful" in result

    def test_compact_system_role(self):
        memory = [{"role": "system", "content": "Instructions"}]
        result = format_memory(memory, {"style": "compact"})
        assert "S: Instructions" in result


class TestMergeMemory:
    """Tests for merge_memory function."""

    def test_merge_two_memories(self):
        now = datetime.now()
        m1 = [MemoryEntry(role="user", content="First", timestamp=now)]
        m2 = [
            MemoryEntry(
                role="assistant", content="Second", timestamp=now + timedelta(seconds=1)
            )
        ]
        merged = merge_memory(m1, m2)
        assert len(merged) == 2
        assert merged[0].content == "First"
        assert merged[1].content == "Second"

    def test_merge_sorts_by_timestamp(self):
        now = datetime.now()
        m1 = [
            MemoryEntry(
                role="user", content="Later", timestamp=now + timedelta(seconds=2)
            )
        ]
        m2 = [MemoryEntry(role="assistant", content="Earlier", timestamp=now)]
        merged = merge_memory(m1, m2)
        assert merged[0].content == "Earlier"
        assert merged[1].content == "Later"

    def test_merge_with_dicts(self):
        m1 = [{"role": "user", "content": "Hello"}]
        m2 = [{"role": "assistant", "content": "Hi"}]
        merged = merge_memory(m1, m2)
        assert len(merged) == 2

    def test_merge_empty_memories(self):
        merged = merge_memory([], [])
        assert merged == []

    def test_merge_mixed_timezone_aware_and_naive_datetimes(self):
        """Test that merging works with mixed timezone-aware and naive datetimes."""
        from datetime import timezone

        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        aware_dt = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        m1 = [MemoryEntry(role="user", content="Naive", timestamp=naive_dt)]
        m2 = [MemoryEntry(role="assistant", content="Aware", timestamp=aware_dt)]

        # This should not raise TypeError
        merged = merge_memory(m1, m2)
        assert len(merged) == 2
        # Both entries should be present (order depends on timezone interpretation)
        contents = {e.content for e in merged}
        assert contents == {"Naive", "Aware"}

    def test_merge_with_none_timestamps(self):
        """Test that entries without timestamps are sorted to the end."""
        now = datetime.now()
        m1 = [MemoryEntry(role="user", content="With timestamp", timestamp=now)]
        m2 = [MemoryEntry(role="assistant", content="No timestamp", timestamp=None)]

        merged = merge_memory(m1, m2)
        assert len(merged) == 2
        assert merged[0].content == "With timestamp"
        assert merged[1].content == "No timestamp"


class TestFilterMemoryByRole:
    """Tests for filter_memory_by_role function."""

    def test_filter_user_messages(self):
        memory = [
            {"role": "user", "content": "User 1"},
            {"role": "assistant", "content": "Assistant 1"},
            {"role": "user", "content": "User 2"},
        ]
        result = filter_memory_by_role(memory, "user")
        assert len(result) == 2
        assert all(e.role == "user" for e in result)

    def test_filter_assistant_messages(self):
        memory = [
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Assistant"},
        ]
        result = filter_memory_by_role(memory, "assistant")
        assert len(result) == 1
        assert result[0].content == "Assistant"

    def test_filter_no_matches(self):
        memory = [{"role": "user", "content": "Test"}]
        result = filter_memory_by_role(memory, "system")
        assert len(result) == 0


class TestGetLastNEntries:
    """Tests for get_last_n_entries function."""

    def test_get_last_3(self):
        memory = [{"role": "user", "content": str(i)} for i in range(10)]
        result = get_last_n_entries(memory, 3)
        assert len(result) == 3
        assert result[0].content == "7"
        assert result[2].content == "9"

    def test_get_more_than_available(self):
        memory = [{"role": "user", "content": "Only one"}]
        result = get_last_n_entries(memory, 5)
        assert len(result) == 1

    def test_get_from_empty(self):
        result = get_last_n_entries([], 5)
        assert len(result) == 0

    def test_get_zero_returns_empty(self):
        memory = [{"role": "user", "content": str(i)} for i in range(5)]
        result = get_last_n_entries(memory, 0)
        assert len(result) == 0

    def test_get_negative_returns_empty(self):
        memory = [{"role": "user", "content": str(i)} for i in range(5)]
        result = get_last_n_entries(memory, -3)
        assert len(result) == 0


class TestCalculateMemorySize:
    """Tests for calculate_memory_size function."""

    def test_calculate_size(self):
        memory = [
            {"role": "user", "content": "Hello"},  # 5 chars
            {"role": "assistant", "content": "Hi!"},  # 3 chars
        ]
        size = calculate_memory_size(memory)
        assert size == 8

    def test_calculate_empty_memory(self):
        size = calculate_memory_size([])
        assert size == 0

    def test_calculate_with_memory_entries(self):
        memory = [
            MemoryEntry(role="user", content="Test"),  # 4 chars
        ]
        size = calculate_memory_size(memory)
        assert size == 4


class TestTruncateMemory:
    """Tests for truncate_memory function."""

    def test_truncate_to_fit(self):
        memory = [
            {"role": "user", "content": "A" * 100} for _ in range(10)
        ]  # 1000 total chars
        result = truncate_memory(memory, 500)
        size = calculate_memory_size(result)
        assert size <= 500

    def test_truncate_preserves_recent(self):
        """Test that truncation keeps the most recent messages."""
        # Each message is ~20 chars, 10 messages = ~200 chars total
        memory = [
            {"role": "user", "content": f"This is message {i}"} for i in range(10)
        ]
        # Limit to 60 chars - should only keep the last few messages
        result = truncate_memory(memory, 60)
        # Should have truncated (removed older messages)
        assert len(result) < 10
        # Should keep the most recent messages
        assert result[-1].content == "This is message 9"
        # Earliest messages should be dropped
        contents = [r.content for r in result]
        assert "This is message 0" not in contents

    def test_truncate_under_limit(self):
        memory = [{"role": "user", "content": "Short"}]
        result = truncate_memory(memory, 1000)
        assert len(result) == 1

    def test_truncate_empty(self):
        result = truncate_memory([], 100)
        assert result == []
