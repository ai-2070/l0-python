"""Tests for l0._utils module."""

import pytest

from l0._utils import (
    AutoCorrectResult,
    auto_correct_json,
    extract_json,
    extract_json_from_markdown,
    is_valid_json,
    safe_json_parse,
)


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

    def test_converts_multiline_single_quoted_values(self):
        """Test that multiline single-quoted values are converted."""
        text = "{'msg': 'line1\nline2\nline3'}"
        result = auto_correct_json(text)
        assert '"msg"' in result.text
        assert '"line1\nline2\nline3"' in result.text
        assert result.corrected is True

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


class TestExtractJson:
    """Test extract_json function (string-aware JSON extraction)."""

    def test_extracts_json_object_from_text(self):
        """Test extraction of JSON object from surrounding text."""
        text = 'Here is the result: {"name": "Alice", "age": 30} Hope this helps!'
        result = extract_json(text)
        assert result == '{"name": "Alice", "age": 30}'

    def test_extracts_json_array_from_text(self):
        """Test extraction of JSON array from surrounding text."""
        text = "The items are: [1, 2, 3] as you can see."
        result = extract_json(text)
        assert result == "[1, 2, 3]"

    def test_returns_original_if_no_json(self):
        """Test returns original text when no JSON found."""
        text = "This is just plain text without JSON."
        result = extract_json(text)
        assert result == text

    def test_ignores_braces_in_quoted_strings(self):
        """Test that braces inside quoted strings in prose are ignored."""
        text = (
            'The format uses "{key}" syntax. Here is the actual JSON: {"name": "Bob"}'
        )
        result = extract_json(text)
        assert result == '{"name": "Bob"}'

    def test_ignores_brackets_in_quoted_strings(self):
        """Test that brackets inside quoted strings in prose are ignored."""
        text = 'Use "[index]" notation. The data: [1, 2, 3]'
        result = extract_json(text)
        assert result == "[1, 2, 3]"

    def test_handles_nested_objects(self):
        """Test extraction of deeply nested JSON."""
        text = 'Result: {"a": {"b": {"c": 1}}} end'
        result = extract_json(text)
        assert result == '{"a": {"b": {"c": 1}}}'

    def test_handles_nested_arrays(self):
        """Test extraction of nested arrays."""
        text = "Data: [[1, 2], [3, 4]] done"
        result = extract_json(text)
        assert result == "[[1, 2], [3, 4]]"

    def test_handles_mixed_nested_structures(self):
        """Test extraction of mixed nested objects and arrays."""
        text = 'Output: {"items": [{"id": 1}, {"id": 2}]} finished'
        result = extract_json(text)
        assert result == '{"items": [{"id": 1}, {"id": 2}]}'

    def test_handles_escaped_quotes_in_json_strings(self):
        """Test handling of escaped quotes inside JSON strings."""
        text = 'Message: {"text": "He said \\"hello\\""} end'
        result = extract_json(text)
        assert result == '{"text": "He said \\"hello\\""}'

    def test_prefers_first_json_structure(self):
        """Test that the first JSON structure is extracted."""
        text = 'First: {"a": 1} Second: {"b": 2}'
        result = extract_json(text)
        assert result == '{"a": 1}'

    def test_handles_empty_object(self):
        """Test extraction of empty objects."""
        text = "Empty: {} done"
        result = extract_json(text)
        assert result == "{}"

    def test_handles_empty_array(self):
        """Test extraction of empty arrays."""
        text = "Empty: [] done"
        result = extract_json(text)
        assert result == "[]"


class TestIsValidJson:
    """Test is_valid_json function."""

    def test_valid_object(self):
        """Test valid JSON object."""
        assert is_valid_json('{"key": "value"}') is True

    def test_valid_array(self):
        """Test valid JSON array."""
        assert is_valid_json("[1, 2, 3]") is True

    def test_valid_primitives(self):
        """Test valid JSON primitives."""
        assert is_valid_json('"string"') is True
        assert is_valid_json("123") is True
        assert is_valid_json("true") is True
        assert is_valid_json("false") is True
        assert is_valid_json("null") is True

    def test_invalid_json(self):
        """Test invalid JSON."""
        assert is_valid_json("{invalid}") is False
        assert is_valid_json('{"key": }') is False
        assert is_valid_json("[1, 2,]") is False

    def test_empty_string(self):
        """Test empty string."""
        assert is_valid_json("") is False

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        assert is_valid_json("   ") is False


class TestSafeJsonParse:
    """Test safe_json_parse function."""

    def test_parses_valid_json(self):
        """Test parsing valid JSON without correction."""
        result = safe_json_parse('{"name": "Alice"}')
        assert result["data"] == {"name": "Alice"}
        assert result["corrected"] is False
        assert result["corrections"] == []

    def test_parses_and_corrects_invalid_json(self):
        """Test parsing and correcting invalid JSON."""
        result = safe_json_parse('{"name": "Alice",}')
        assert result["data"] == {"name": "Alice"}
        assert result["corrected"] is True
        assert len(result["corrections"]) > 0

    def test_extracts_json_from_text(self):
        """Test extraction of JSON from surrounding text."""
        result = safe_json_parse('Here is the JSON: {"value": 42} done')
        assert result["data"] == {"value": 42}
        assert result["corrected"] is True

    def test_raises_on_unparseable_json(self):
        """Test raising error for unparseable JSON."""
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            safe_json_parse("this is not json at all")

    def test_raises_without_auto_correct(self):
        """Test raising error when auto_correct is disabled."""
        with pytest.raises(ValueError):
            safe_json_parse('{"key": "value",}', auto_correct=False)

    def test_corrects_missing_brace(self):
        """Test correcting missing closing brace."""
        result = safe_json_parse('{"key": "value"')
        assert result["data"] == {"key": "value"}
        assert result["corrected"] is True
