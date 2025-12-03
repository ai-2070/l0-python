"""Tests for l0._utils module."""

import pytest

from l0._utils import AutoCorrectResult, auto_correct_json, extract_json_from_markdown


class TestAutoCorrectJson:
    """Test auto_correct_json function."""

    def test_removes_trailing_commas(self):
        """Test trailing comma removal."""
        result = auto_correct_json('{"a": 1,}')
        assert result.text == '{"a": 1}'
        assert result.corrected is True

    def test_balances_braces(self):
        """Test missing brace balancing."""
        result = auto_correct_json('{"a": {"b": 1}')
        assert result.text == '{"a": {"b": 1}}'
        assert result.corrected is True

    def test_balances_brackets(self):
        """Test missing bracket balancing."""
        result = auto_correct_json("[1, 2, 3")
        assert result.text == "[1, 2, 3]"
        assert result.corrected is True

    def test_strips_whitespace(self):
        """Test whitespace stripping."""
        result = auto_correct_json('  {"a": 1}  ')
        assert result.text == '{"a": 1}'

    def test_removes_text_prefix(self):
        """Test text prefix removal."""
        result = auto_correct_json(
            'Sure! Here is the JSON: {"a": 1}', track_corrections=True
        )
        assert result.text == '{"a": 1}'
        assert result.corrected is True
        assert any("prefix" in c.lower() for c in result.corrections)

    def test_removes_text_suffix(self):
        """Test text suffix removal."""
        result = auto_correct_json(
            '{"a": 1} Let me know if you need anything!', track_corrections=True
        )
        assert result.text == '{"a": 1}'
        assert result.corrected is True
        assert any("suffix" in c.lower() for c in result.corrections)

    def test_converts_single_quotes(self):
        """Test single quote to double quote conversion."""
        result = auto_correct_json("{'name': 'Alice'}", track_corrections=True)
        assert '"name"' in result.text
        assert '"Alice"' in result.text
        assert result.corrected is True
        assert any("quote" in c.lower() for c in result.corrections)

    def test_converts_single_quotes_with_apostrophe(self):
        """Test that apostrophes inside single-quoted strings are preserved."""
        result = auto_correct_json(
            "{'message': 'Don\\'t panic'}", track_corrections=True
        )
        assert '"message"' in result.text
        assert "Don't panic" in result.text or "Don\\'t panic" in result.text
        assert result.corrected is True

    def test_converts_single_quotes_multiple_values(self):
        """Test single quote conversion with multiple values containing apostrophes."""
        result = auto_correct_json(
            "{'a': 'it\\'s fine', 'b': 'that\\'s ok'}", track_corrections=True
        )
        assert '"a"' in result.text
        assert '"b"' in result.text
        assert result.corrected is True

    def test_removes_markdown_fences(self):
        """Test markdown fence removal."""
        result = auto_correct_json('```json\n{"a": 1}\n```', track_corrections=True)
        assert result.text == '{"a": 1}'
        assert result.corrected is True
        assert any("markdown" in c.lower() for c in result.corrections)

    def test_preserves_backticks_inside_json_strings(self):
        """Test that triple backticks inside JSON strings are not corrupted."""
        # JSON with backticks in a string value should be preserved
        json_with_backticks = '{"code": "Use ```python\\nprint()\\n``` for code"}'
        result = auto_correct_json(json_with_backticks)
        assert result.text == json_with_backticks
        assert result.corrected is False

    def test_complex_correction(self):
        """Test multiple corrections at once."""
        text = """Sure! Here's the data:
```json
{"name": "Bob", "age": 30,}
```
Hope this helps!"""
        result = auto_correct_json(text, track_corrections=True)
        assert '"name"' in result.text
        assert '"Bob"' in result.text
        assert ",}" not in result.text
        assert result.corrected is True

    def test_no_correction_needed(self):
        """Test valid JSON doesn't get marked as corrected."""
        result = auto_correct_json('{"a": 1}')
        assert result.text == '{"a": 1}'
        assert result.corrected is False

    def test_braces_inside_strings_not_counted(self):
        """Test that braces/brackets inside strings are ignored for balancing."""
        # Valid JSON with literal { and [ inside string values
        result = auto_correct_json('{"key": "{value}"}')
        assert result.text == '{"key": "{value}"}'
        assert result.corrected is False

        result = auto_correct_json('{"key": "[1, 2, 3]"}')
        assert result.text == '{"key": "[1, 2, 3]"}'
        assert result.corrected is False

        # More complex: nested braces in strings
        result = auto_correct_json('{"msg": "use {brackets} and [arrays]"}')
        assert result.text == '{"msg": "use {brackets} and [arrays]"}'
        assert result.corrected is False

    def test_track_corrections_flag(self):
        """Test that corrections list is populated when tracking."""
        result = auto_correct_json('{"a": 1,}', track_corrections=True)
        assert len(result.corrections) > 0
        assert any("comma" in c.lower() for c in result.corrections)

        # Without tracking
        result2 = auto_correct_json('{"a": 1,}', track_corrections=False)
        assert len(result2.corrections) == 0


class TestExtractJsonFromMarkdown:
    """Test extract_json_from_markdown function."""

    def test_extracts_json_block(self):
        """Test extraction from json code block."""
        text = """Here is the response:
```json
{"key": "value"}
```
Done."""
        result = extract_json_from_markdown(text)
        assert result == '{"key": "value"}'

    def test_extracts_plain_code_block(self):
        """Test extraction from plain code block."""
        text = """```
{"key": "value"}
```"""
        result = extract_json_from_markdown(text)
        assert result == '{"key": "value"}'

    def test_returns_original_if_no_block(self):
        """Test returns original when no code block."""
        text = '{"key": "value"}'
        result = extract_json_from_markdown(text)
        assert result == '{"key": "value"}'

    def test_handles_multiline_json(self):
        """Test multiline JSON extraction."""
        text = """```json
{
  "key": "value",
  "nested": {
    "a": 1
  }
}
```"""
        result = extract_json_from_markdown(text)
        assert '"key": "value"' in result
        assert '"nested"' in result
