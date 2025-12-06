"""Tests for l0.formatting.output module."""

from __future__ import annotations

import pytest

from l0.formatting.output import (
    JsonOutputOptions,
    OutputConstraints,
    clean_output,
    create_output_format_section,
    extract_json_from_output,
    format_json_output,
    format_output_constraints,
    format_structured_output,
    validate_json_output,
    wrap_output_instruction,
)


class TestFormatJsonOutput:
    """Tests for format_json_output function."""

    def test_strict_json_output(self):
        result = format_json_output({"strict": True})
        assert "valid JSON only" in result
        assert "Do not include any text" in result
        assert "Do not wrap" in result

    def test_non_strict_json_output(self):
        result = format_json_output({"strict": False})
        assert "JSON" in result

    def test_json_output_with_schema(self):
        result = format_json_output(
            {
                "strict": True,
                "schema": '{ "name": "string", "age": "number" }',
            }
        )
        assert "valid JSON only" in result
        assert "schema" in result.lower()
        assert '"name": "string"' in result

    def test_json_output_with_example(self):
        result = format_json_output(
            {
                "strict": True,
                "example": '{"name": "John", "age": 30}',
            }
        )
        assert "xample" in result  # Example or example
        assert '{"name": "John"' in result

    def test_json_output_with_options_object(self):
        opts = JsonOutputOptions(strict=True, schema='{"test": "schema"}')
        result = format_json_output(opts)
        assert "valid JSON only" in result
        assert '{"test": "schema"}' in result


class TestFormatStructuredOutput:
    """Tests for format_structured_output function."""

    def test_yaml_strict(self):
        result = format_structured_output("yaml", {"strict": True})
        assert "valid YAML only" in result
        assert "Do not include any text" in result

    def test_xml_strict(self):
        result = format_structured_output("xml", {"strict": True})
        assert "valid XML only" in result

    def test_markdown_non_strict(self):
        result = format_structured_output("markdown")
        assert result == "Respond with MARKDOWN."

    def test_plain_format(self):
        result = format_structured_output("plain")
        assert result == "Respond with PLAIN."

    def test_with_schema(self):
        result = format_structured_output("yaml", {"schema": "name: string"})
        assert "Use this schema" in result
        assert "name: string" in result


class TestFormatOutputConstraints:
    """Tests for format_output_constraints function."""

    def test_max_length_constraint(self):
        result = format_output_constraints({"max_length": 500})
        # TS uses "Keep your response under X characters" format
        assert "500" in result
        assert "character" in result.lower()

    def test_min_length_constraint(self):
        result = format_output_constraints({"min_length": 100})
        # TS uses "Provide at least X characters" format
        assert "100" in result
        assert "character" in result.lower()

    def test_no_code_blocks(self):
        result = format_output_constraints({"no_code_blocks": True})
        assert "Do not use code blocks" in result

    def test_no_markdown(self):
        result = format_output_constraints({"no_markdown": True})
        assert "markdown" in result.lower() or "Markdown" in result

    def test_language_constraint(self):
        result = format_output_constraints({"language": "Spanish"})
        assert "Respond in Spanish" in result

    def test_tone_constraint(self):
        result = format_output_constraints({"tone": "professional"})
        assert "Use a professional tone" in result

    def test_multiple_constraints(self):
        result = format_output_constraints(
            {
                "max_length": 500,
                "min_length": 100,
                "language": "French",
                "tone": "casual",
            }
        )
        assert "500" in result
        assert "100" in result
        assert "French" in result
        assert "casual" in result

    def test_empty_constraints(self):
        result = format_output_constraints({})
        assert result == ""

    def test_with_constraints_object(self):
        constraints = OutputConstraints(max_length=1000, language="German")
        result = format_output_constraints(constraints)
        assert "1000" in result
        assert "German" in result


class TestCreateOutputFormatSection:
    """Tests for create_output_format_section function."""

    def test_json_format_section(self):
        result = create_output_format_section("json", {"strict": True})
        assert "valid JSON only" in result

    def test_format_section_with_wrap(self):
        result = create_output_format_section(
            "json",
            {
                "strict": True,
                "wrap": True,
            },
        )
        assert "<output_format>" in result
        assert "</output_format>" in result

    def test_format_section_with_constraints(self):
        result = create_output_format_section(
            "json",
            {
                "strict": True,
                "constraints": {"max_length": 1000},
            },
        )
        assert "valid JSON only" in result
        assert "1000" in result

    def test_format_section_with_schema(self):
        result = create_output_format_section(
            "json",
            {
                "strict": True,
                "schema": '{ "result": "string" }',
            },
        )
        assert '{ "result": "string" }' in result


class TestExtractJsonFromOutput:
    """Tests for extract_json_from_output function."""

    def test_extract_from_plain_json(self):
        result = extract_json_from_output('{"name": "John"}')
        assert result == '{"name": "John"}'

    def test_extract_from_text_with_json(self):
        result = extract_json_from_output('Here is the result: {"name": "John"}')
        assert result == '{"name": "John"}'

    def test_extract_from_json_code_block(self):
        result = extract_json_from_output('```json\n{"name": "John"}\n```')
        assert result == '{"name": "John"}'

    def test_extract_from_code_block_no_lang(self):
        result = extract_json_from_output('```\n{"name": "John"}\n```')
        assert result == '{"name": "John"}'

    def test_extract_array(self):
        result = extract_json_from_output("Result: [1, 2, 3]")
        assert result == "[1, 2, 3]"

    def test_extract_nested_json(self):
        json_str = '{"outer": {"inner": "value"}}'
        result = extract_json_from_output(f"Here: {json_str}")
        assert result == json_str

    def test_extract_with_text_after(self):
        result = extract_json_from_output('Result: {"key": "value"} That is all.')
        assert result == '{"key": "value"}'

    def test_extract_skips_non_json_code_blocks(self):
        """Test that non-JSON code blocks are skipped."""
        output = """Here's some code:
```python
def hello():
    print("Hello")
```

And here's the JSON:
```json
{"name": "John"}
```
"""
        result = extract_json_from_output(output)
        assert result == '{"name": "John"}'

    def test_extract_first_valid_json_block(self):
        """Test that the first valid JSON code block is returned."""
        output = """
```
not valid json
```

```json
{"first": true}
```

```json
{"second": true}
```
"""
        result = extract_json_from_output(output)
        assert result == '{"first": true}'

    def test_extract_skips_invalid_brace_pairs(self):
        """Test that invalid brace pairs are skipped to find valid JSON."""
        # Earlier {placeholder} should be skipped in favor of valid JSON
        output = 'Use {placeholder} format. Result: {"name": "John", "age": 30}'
        result = extract_json_from_output(output)
        assert result == '{"name": "John", "age": 30}'

    def test_extract_skips_multiple_invalid_pairs(self):
        """Test skipping multiple invalid brace pairs."""
        output = 'Values {x} and {y} are set. JSON: {"valid": true}'
        result = extract_json_from_output(output)
        assert result == '{"valid": true}'

    def test_extract_skips_unmatched_opening_brace(self):
        """Test that unmatched opening brace doesn't stop the search."""
        output = 'Some text with { unmatched brace. JSON: {"valid": true}'
        result = extract_json_from_output(output)
        assert result == '{"valid": true}'

    def test_extract_skips_unmatched_opening_bracket(self):
        """Test that unmatched opening bracket doesn't stop the search."""
        output = "Array [ without end. Real array: [1, 2, 3]"
        result = extract_json_from_output(output)
        assert result == "[1, 2, 3]"


class TestCleanOutput:
    """Tests for clean_output function."""

    def test_clean_code_block(self):
        result = clean_output("```json\n{}\n```")
        assert result == "{}"

    def test_clean_sure_prefix(self):
        result = clean_output("Sure, here is the result:\n{}")
        assert result.strip() == "{}"

    def test_clean_here_is_prefix(self):
        result = clean_output("Here is the JSON:\n{}")
        assert result.strip() == "{}"

    def test_clean_already_clean(self):
        result = clean_output('{"clean": true}')
        assert result == '{"clean": true}'

    def test_clean_complex_prefix(self):
        result = clean_output("Sure, here is the result:\n```json\n{}\n```")
        assert result == "{}"

    def test_clean_yaml_code_block(self):
        """Test that non-json language tags are also stripped."""
        result = clean_output("```yaml\nkey: value\n```")
        assert result == "key: value"

    def test_clean_python_code_block(self):
        """Test that python code blocks are stripped."""
        result = clean_output("```python\nprint('hello')\n```")
        assert result == "print('hello')"

    def test_clean_unlabeled_code_block(self):
        """Test that unlabeled code blocks are stripped."""
        result = clean_output("```\n{}\n```")
        assert result == "{}"


class TestValidateJsonOutput:
    """Tests for validate_json_output function."""

    def test_valid_json(self):
        is_valid, error = validate_json_output('{"name": "John"}')
        assert is_valid is True
        assert error is None

    def test_valid_array(self):
        is_valid, error = validate_json_output("[1, 2, 3]")
        assert is_valid is True
        assert error is None

    def test_invalid_json(self):
        is_valid, error = validate_json_output('{"name": }')
        assert is_valid is False
        assert error is not None
        assert "Expecting value" in error

    def test_empty_string(self):
        is_valid, error = validate_json_output("")
        assert is_valid is False
        assert error is not None

    def test_not_json(self):
        is_valid, error = validate_json_output("Just some text")
        assert is_valid is False
        assert error is not None


class TestWrapOutputInstruction:
    """Tests for wrap_output_instruction function."""

    def test_wrap_basic_instruction(self):
        result = wrap_output_instruction("Respond with valid JSON only.")
        assert (
            result == "<output_format>\nRespond with valid JSON only.\n</output_format>"
        )

    def test_wrap_multiline_instruction(self):
        instruction = "Line 1\nLine 2\nLine 3"
        result = wrap_output_instruction(instruction)
        assert "<output_format>" in result
        assert "</output_format>" in result
        assert "Line 1\nLine 2\nLine 3" in result

    def test_wrap_empty_instruction(self):
        result = wrap_output_instruction("")
        assert result == "<output_format>\n\n</output_format>"


class TestOutputFormatSectionDefaultWrap:
    """Tests for default wrap behavior in create_output_format_section."""

    def test_default_wrap_is_true(self):
        """Test that wrap defaults to True."""
        result = create_output_format_section("json", {"strict": True})
        assert "<output_format>" in result
        assert "</output_format>" in result

    def test_wrap_false_no_tags(self):
        """Test that wrap=False excludes output_format tags."""
        result = create_output_format_section("json", {"strict": True, "wrap": False})
        assert "<output_format>" not in result
        assert "</output_format>" not in result
