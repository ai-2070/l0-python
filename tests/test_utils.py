"""Tests for l0._utils module."""

import pytest

from l0._utils import auto_correct_json, extract_json_from_markdown


class TestAutoCorrectJson:
    def test_removes_trailing_commas(self):
        result = auto_correct_json('{"a": 1,}')
        assert result == '{"a": 1}'

    def test_balances_braces(self):
        result = auto_correct_json('{"a": {"b": 1}')
        assert result == '{"a": {"b": 1}}'

    def test_balances_brackets(self):
        result = auto_correct_json("[1, 2, 3")
        assert result == "[1, 2, 3]"

    def test_strips_whitespace(self):
        result = auto_correct_json('  {"a": 1}  ')
        assert result == '{"a": 1}'


class TestExtractJsonFromMarkdown:
    def test_extracts_json_block(self):
        text = """Here is the response:
```json
{"key": "value"}
```
Done."""
        result = extract_json_from_markdown(text)
        assert result == '{"key": "value"}'

    def test_extracts_plain_code_block(self):
        text = """```
{"key": "value"}
```"""
        result = extract_json_from_markdown(text)
        assert result == '{"key": "value"}'

    def test_returns_original_if_no_block(self):
        text = '{"key": "value"}'
        result = extract_json_from_markdown(text)
        assert result == '{"key": "value"}'

    def test_handles_multiline_json(self):
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
