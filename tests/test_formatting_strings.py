"""Tests for l0.formatting.strings module."""

from __future__ import annotations

import pytest

from l0.formatting.strings import (
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


class TestEscapeUnescape:
    """Tests for escape and unescape functions."""

    def test_escape_newline(self):
        assert escape("Hello\nWorld") == "Hello\\nWorld"

    def test_escape_tab(self):
        assert escape("Hello\tWorld") == "Hello\\tWorld"

    def test_escape_carriage_return(self):
        assert escape("Hello\rWorld") == "Hello\\rWorld"

    def test_escape_quote(self):
        assert escape('Hello"World') == 'Hello\\"World'

    def test_escape_backslash(self):
        assert escape("Hello\\World") == "Hello\\\\World"

    def test_escape_multiple(self):
        assert escape('Hello\n"World"\t') == 'Hello\\n\\"World\\"\\t'

    def test_unescape_newline(self):
        assert unescape("Hello\\nWorld") == "Hello\nWorld"

    def test_unescape_tab(self):
        assert unescape("Hello\\tWorld") == "Hello\tWorld"

    def test_unescape_carriage_return(self):
        assert unescape("Hello\\rWorld") == "Hello\rWorld"

    def test_unescape_quote(self):
        assert unescape('Hello\\"World') == 'Hello"World'

    def test_unescape_backslash(self):
        assert unescape("Hello\\\\World") == "Hello\\World"

    def test_roundtrip(self):
        original = 'Hello\n"World"\t\\test'
        assert unescape(escape(original)) == original


class TestHtmlEscaping:
    """Tests for HTML escaping functions."""

    def test_escape_html_angle_brackets(self):
        assert escape_html("<div>") == "&lt;div&gt;"

    def test_escape_html_ampersand(self):
        assert escape_html("A & B") == "A &amp; B"

    def test_escape_html_quotes(self):
        assert escape_html('"test"') == "&quot;test&quot;"

    def test_escape_html_single_quote(self):
        assert escape_html("it's") == "it&#39;s"

    def test_escape_html_full(self):
        assert (
            escape_html("<a href='#'>Link</a>")
            == "&lt;a href=&#39;#&#39;&gt;Link&lt;/a&gt;"
        )

    def test_unescape_html_angle_brackets(self):
        assert unescape_html("&lt;div&gt;") == "<div>"

    def test_unescape_html_ampersand(self):
        assert unescape_html("A &amp; B") == "A & B"

    def test_unescape_html_quotes(self):
        assert unescape_html("&quot;test&quot;") == '"test"'

    def test_unescape_html_double_encoded_entities(self):
        """Test that double-encoded entities are only decoded once.

        By replacing &amp; last, we prevent injection attacks where
        &amp;lt;script&amp;gt; could become <script> if decoded incorrectly.
        """
        # &amp;lt; should become &lt; (not <) - only one level of decoding
        assert unescape_html("&amp;lt;") == "&lt;"
        # &amp;gt; should become &gt; (not >)
        assert unescape_html("&amp;gt;") == "&gt;"
        # &amp;amp; becomes &amp; (only one level decoded)
        assert unescape_html("&amp;amp;") == "&amp;"
        # Verify injection attack is prevented
        assert unescape_html("&amp;lt;script&amp;gt;") == "&lt;script&gt;"

    def test_unescape_html_hex_apostrophe(self):
        """Test that hex variant &#x27; is also unescaped to apostrophe."""
        assert unescape_html("it&#x27;s") == "it's"

    def test_unescape_html_both_apostrophe_variants(self):
        """Test that both &#39; and &#x27; are unescaped."""
        assert unescape_html("&#39;hello&#x27;") == "'hello'"

    def test_roundtrip_html(self):
        original = '<script>alert("xss")</script>'
        assert unescape_html(escape_html(original)) == original


class TestEscapeRegex:
    """Tests for regex escaping."""

    def test_escape_dot(self):
        assert escape_regex("file.txt") == "file\\.txt"

    def test_escape_asterisk(self):
        assert escape_regex("*.py") == "\\*\\.py"

    def test_escape_brackets(self):
        assert escape_regex("[test]") == "\\[test\\]"

    def test_escape_parentheses(self):
        assert escape_regex("(group)") == "\\(group\\)"

    def test_escape_special_chars(self):
        result = escape_regex("^$.|?*+()[]{}")
        assert "\\^" in result
        assert "\\$" in result


class TestSanitize:
    """Tests for sanitize function."""

    def test_remove_null_byte(self):
        assert sanitize("Hello\x00World") == "HelloWorld"

    def test_preserve_newline(self):
        assert sanitize("Hello\nWorld") == "Hello\nWorld"

    def test_preserve_tab(self):
        assert sanitize("Hello\tWorld") == "Hello\tWorld"

    def test_preserve_carriage_return(self):
        assert sanitize("Hello\rWorld") == "Hello\rWorld"

    def test_remove_control_chars(self):
        assert sanitize("Hello\x01\x02\x03World") == "HelloWorld"

    def test_remove_delete_char(self):
        assert sanitize("Hello\x7fWorld") == "HelloWorld"


class TestTrim:
    """Tests for trim function."""

    def test_trim_spaces(self):
        assert trim("  Hello  ") == "Hello"

    def test_trim_tabs(self):
        assert trim("\tHello\t") == "Hello"

    def test_trim_newlines(self):
        assert trim("\nHello\n") == "Hello"

    def test_trim_mixed(self):
        assert trim("  \t\n Hello \n\t  ") == "Hello"

    def test_trim_no_whitespace(self):
        assert trim("Hello") == "Hello"


class TestTruncate:
    """Tests for truncate function."""

    def test_truncate_short_string(self):
        assert truncate("Hello", 10) == "Hello"

    def test_truncate_exact_length(self):
        assert truncate("Hello", 5) == "Hello"

    def test_truncate_with_suffix(self):
        assert truncate("Hello World", 8) == "Hello..."

    def test_truncate_custom_suffix(self):
        assert truncate("Hello World", 8, "…") == "Hello W…"

    def test_truncate_very_short(self):
        assert truncate("Hello", 3) == "..."

    def test_truncate_suffix_longer_than_max(self):
        assert truncate("Hello", 2) == ".."


class TestTruncateWords:
    """Tests for truncate_words function."""

    def test_truncate_words_short_string(self):
        assert truncate_words("Hello", 10) == "Hello"

    def test_truncate_words_at_boundary(self):
        assert truncate_words("Hello World Test", 12) == "Hello..."

    def test_truncate_words_custom_suffix(self):
        assert truncate_words("Hello World Test", 12, "…") == "Hello World…"

    def test_truncate_words_exact_fit(self):
        assert truncate_words("Hello World", 20) == "Hello World"


class TestWrap:
    """Tests for wrap function."""

    def test_wrap_short_text(self):
        result = wrap("Hello", 10)
        assert result == "Hello"

    def test_wrap_long_text(self):
        result = wrap("Hello World Test", 10)
        assert "\n" in result

    def test_wrap_single_long_word(self):
        result = wrap("Supercalifragilisticexpialidocious", 10)
        assert "\n" in result


class TestPad:
    """Tests for pad function."""

    def test_pad_left(self):
        assert pad("Hi", 10) == "Hi        "

    def test_pad_right(self):
        assert pad("Hi", 10, " ", "right") == "        Hi"

    def test_pad_center(self):
        assert pad("Hi", 10, " ", "center") == "    Hi    "

    def test_pad_custom_char(self):
        assert pad("Hi", 10, "-") == "Hi--------"

    def test_pad_already_long(self):
        assert pad("Hello World", 5) == "Hello World"


class TestRemoveAnsi:
    """Tests for remove_ansi function."""

    def test_remove_color_code(self):
        assert remove_ansi("\x1b[31mRed\x1b[0m") == "Red"

    def test_remove_bold_code(self):
        assert remove_ansi("\x1b[1mBold\x1b[0m") == "Bold"

    def test_remove_multiple_codes(self):
        assert remove_ansi("\x1b[31m\x1b[1mBold Red\x1b[0m") == "Bold Red"

    def test_no_ansi_codes(self):
        assert remove_ansi("Plain text") == "Plain text"

    def test_complex_sequence(self):
        assert remove_ansi("\x1b[38;5;208mOrange\x1b[0m") == "Orange"

    def test_remove_cursor_sequence_with_tilde(self):
        """Test removal of sequences ending in ~ (e.g., cursor key codes)."""
        # Delete key sequence
        assert remove_ansi("\x1b[3~") == ""
        # Page up
        assert remove_ansi("\x1b[5~") == ""
        # Text with cursor sequences
        assert remove_ansi("Hello\x1b[3~World") == "HelloWorld"

    def test_remove_private_mode_sequences(self):
        """Test removal of sequences containing ? (private modes)."""
        # Show cursor
        assert remove_ansi("\x1b[?25h") == ""
        # Hide cursor
        assert remove_ansi("\x1b[?25l") == ""
        # Text with private mode sequences
        assert remove_ansi("\x1b[?25lHidden\x1b[?25h") == "Hidden"
