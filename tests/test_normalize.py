"""Tests for text normalization utilities."""

import pytest

from l0 import (
    count_lines,
    dedent,
    ensure_trailing_newline,
    get_line,
    indent,
    is_whitespace_only,
    normalize_for_model,
    normalize_indentation,
    normalize_newlines,
    normalize_text,
    normalize_whitespace,
    remove_trailing_whitespace,
    replace_line,
    trim_text,
)


class TestNormalizeNewlines:
    """Tests for normalize_newlines."""

    def test_empty_string(self) -> None:
        assert normalize_newlines("") == ""

    def test_none_like_empty(self) -> None:
        # Empty string should return empty
        assert normalize_newlines("") == ""

    def test_unix_newlines_unchanged(self) -> None:
        text = "line1\nline2\nline3"
        assert normalize_newlines(text) == text

    def test_windows_newlines(self) -> None:
        text = "line1\r\nline2\r\nline3"
        assert normalize_newlines(text) == "line1\nline2\nline3"

    def test_old_mac_newlines(self) -> None:
        text = "line1\rline2\rline3"
        assert normalize_newlines(text) == "line1\nline2\nline3"

    def test_mixed_newlines(self) -> None:
        text = "line1\r\nline2\rline3\nline4"
        assert normalize_newlines(text) == "line1\nline2\nline3\nline4"


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace."""

    def test_empty_string(self) -> None:
        assert normalize_whitespace("") == ""

    def test_collapse_spaces(self) -> None:
        text = "hello   world    test"
        result = normalize_whitespace(text, collapse_spaces=True)
        assert result == "hello world test"

    def test_trim_lines(self) -> None:
        text = "  hello  \n  world  "
        result = normalize_whitespace(text, trim_lines=True)
        assert result == "hello\nworld"

    def test_remove_empty_lines(self) -> None:
        text = "hello\n\n\nworld\n\ntest"
        result = normalize_whitespace(text, remove_empty_lines=True)
        assert result == "hello\nworld\ntest"

    def test_all_options(self) -> None:
        text = "  hello   world  \n\n  test  "
        result = normalize_whitespace(
            text,
            collapse_spaces=True,
            trim_lines=True,
            remove_empty_lines=True,
        )
        assert result == "hello world\ntest"


class TestNormalizeIndentation:
    """Tests for normalize_indentation."""

    def test_empty_string(self) -> None:
        assert normalize_indentation("") == ""

    def test_tabs_to_spaces(self) -> None:
        text = "\thello\n\t\tworld"
        result = normalize_indentation(text, mode="spaces", spaces_per_tab=2)
        assert result == "  hello\n    world"

    def test_tabs_to_spaces_4(self) -> None:
        text = "\thello"
        result = normalize_indentation(text, mode="spaces", spaces_per_tab=4)
        assert result == "    hello"

    def test_spaces_to_tabs(self) -> None:
        text = "    hello\n        world"
        result = normalize_indentation(text, mode="tabs", spaces_per_tab=4)
        assert result == "\thello\n\t\tworld"

    def test_partial_indent_spaces_to_tabs(self) -> None:
        # 5 spaces with 4 spaces per tab = 1 tab + 1 space
        text = "     hello"
        result = normalize_indentation(text, mode="tabs", spaces_per_tab=4)
        assert result == "\t hello"


class TestDedent:
    """Tests for dedent."""

    def test_empty_string(self) -> None:
        assert dedent("") == ""

    def test_no_indent(self) -> None:
        text = "hello\nworld"
        assert dedent(text) == text

    def test_common_indent(self) -> None:
        text = "    hello\n    world\n    test"
        assert dedent(text) == "hello\nworld\ntest"

    def test_varying_indent(self) -> None:
        text = "    hello\n        world\n    test"
        assert dedent(text) == "hello\n    world\ntest"

    def test_preserves_empty_lines(self) -> None:
        text = "    hello\n\n    world"
        assert dedent(text) == "hello\n\nworld"


class TestIndent:
    """Tests for indent."""

    def test_empty_string(self) -> None:
        assert indent("") == ""

    def test_indent_with_spaces(self) -> None:
        text = "hello\nworld"
        result = indent(text, 2)
        assert result == "  hello\n  world"

    def test_indent_with_string(self) -> None:
        text = "hello\nworld"
        result = indent(text, ">>> ")
        assert result == ">>> hello\n>>> world"

    def test_preserves_empty_lines(self) -> None:
        text = "hello\n\nworld"
        result = indent(text, 2)
        assert result == "  hello\n\n  world"


class TestTrimText:
    """Tests for trim_text."""

    def test_empty_string(self) -> None:
        assert trim_text("") == ""

    def test_removes_leading_empty_lines(self) -> None:
        text = "\n\n\nhello\nworld"
        assert trim_text(text) == "hello\nworld"

    def test_removes_trailing_empty_lines(self) -> None:
        text = "hello\nworld\n\n\n"
        assert trim_text(text) == "hello\nworld"

    def test_trims_whitespace(self) -> None:
        text = "  \n  hello  \n  "
        assert trim_text(text) == "hello"


class TestNormalizeText:
    """Tests for normalize_text."""

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_newlines_only(self) -> None:
        text = "hello\r\nworld"
        result = normalize_text(text, newlines=True)
        assert result == "hello\nworld"

    def test_full_normalization(self) -> None:
        text = "\r\n  hello   world  \r\n\r\n  test  \r\n"
        result = normalize_text(
            text,
            newlines=True,
            whitespace=True,
            trim=True,
        )
        # whitespace=True collapses multiple spaces but preserves trailing space after "world"
        assert result == "hello world \n\n test"


class TestEnsureTrailingNewline:
    """Tests for ensure_trailing_newline."""

    def test_empty_string(self) -> None:
        assert ensure_trailing_newline("") == ""

    def test_no_trailing_newline(self) -> None:
        assert ensure_trailing_newline("hello") == "hello\n"

    def test_single_trailing_newline(self) -> None:
        assert ensure_trailing_newline("hello\n") == "hello\n"

    def test_multiple_trailing_newlines(self) -> None:
        assert ensure_trailing_newline("hello\n\n\n") == "hello\n"


class TestRemoveTrailingWhitespace:
    """Tests for remove_trailing_whitespace."""

    def test_empty_string(self) -> None:
        assert remove_trailing_whitespace("") == ""

    def test_removes_trailing_spaces(self) -> None:
        text = "hello   \nworld  \ntest"
        assert remove_trailing_whitespace(text) == "hello\nworld\ntest"

    def test_removes_trailing_tabs(self) -> None:
        text = "hello\t\t\nworld"
        assert remove_trailing_whitespace(text) == "hello\nworld"


class TestNormalizeForModel:
    """Tests for normalize_for_model."""

    def test_empty_string(self) -> None:
        assert normalize_for_model("") == ""

    def test_normalizes_for_model(self) -> None:
        text = "  hello   world  \r\n  test  "
        result = normalize_for_model(text)
        # Collapses multiple spaces but preserves single trailing space
        assert result == "hello world \n test"


class TestIsWhitespaceOnly:
    """Tests for is_whitespace_only."""

    def test_empty_string(self) -> None:
        assert is_whitespace_only("") is True

    def test_whitespace_only(self) -> None:
        assert is_whitespace_only("   \n\t  ") is True

    def test_has_content(self) -> None:
        assert is_whitespace_only("  hello  ") is False


class TestCountLines:
    """Tests for count_lines."""

    def test_empty_string(self) -> None:
        assert count_lines("") == 0

    def test_single_line(self) -> None:
        assert count_lines("hello") == 1

    def test_multiple_lines(self) -> None:
        assert count_lines("hello\nworld\ntest") == 3


class TestGetLine:
    """Tests for get_line."""

    def test_empty_string(self) -> None:
        assert get_line("", 0) is None

    def test_valid_index(self) -> None:
        text = "hello\nworld\ntest"
        assert get_line(text, 0) == "hello"
        assert get_line(text, 1) == "world"
        assert get_line(text, 2) == "test"

    def test_out_of_bounds(self) -> None:
        text = "hello\nworld"
        assert get_line(text, 5) is None
        assert get_line(text, -1) is None


class TestReplaceLine:
    """Tests for replace_line."""

    def test_empty_string(self) -> None:
        assert replace_line("", 0, "new") == ""

    def test_replace_first_line(self) -> None:
        text = "hello\nworld\ntest"
        result = replace_line(text, 0, "NEW")
        assert result == "NEW\nworld\ntest"

    def test_replace_middle_line(self) -> None:
        text = "hello\nworld\ntest"
        result = replace_line(text, 1, "NEW")
        assert result == "hello\nNEW\ntest"

    def test_out_of_bounds(self) -> None:
        text = "hello\nworld"
        assert replace_line(text, 5, "new") == text
