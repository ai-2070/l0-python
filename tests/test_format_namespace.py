"""Tests for l0.Format namespace."""

from __future__ import annotations

from l0 import Format


class TestFormatNamespace:
    """Tests for the Format namespace class."""

    def test_context(self):
        result = Format.context("Hello", label="test")
        assert "<test>" in result
        assert "Hello" in result

    def test_contexts(self):
        items = [{"content": "A", "label": "a"}, {"content": "B", "label": "b"}]
        result = Format.contexts(items)
        assert "<a>" in result
        assert "<b>" in result

    def test_document(self):
        result = Format.document("Content", {"title": "Test"})
        # Document uses title as XML tag label and includes metadata as key-value
        assert "<test>" in result
        assert "title: Test" in result
        assert "Content" in result

    def test_instructions(self):
        result = Format.instructions("Be helpful")
        assert "<instructions>" in result
        assert "Be helpful" in result

    def test_memory(self):
        mem = [{"role": "user", "content": "Hi"}]
        result = Format.memory(mem)
        assert "User: Hi" in result

    def test_memory_entry(self):
        entry = Format.memory_entry("user", "Hello")
        assert entry.role == "user"
        assert entry.content == "Hello"
        assert entry.timestamp is not None

    def test_json_output(self):
        result = Format.json_output({"strict": True})
        assert "valid JSON only" in result

    def test_structured_output(self):
        result = Format.structured_output("yaml", {"strict": True})
        assert "YAML" in result

    def test_create_tool(self):
        tool = Format.create_tool("test", "Test tool")
        assert tool.name == "test"
        assert tool.description == "Test tool"

    def test_parameter(self):
        param = Format.parameter("loc", "string", "Location", True)
        assert param.name == "loc"
        assert param.required is True

    def test_tool(self):
        tool = Format.create_tool("test", "Test")
        result = Format.tool(tool, {"style": "natural"})
        assert "Tool: test" in result

    def test_escape_html(self):
        assert Format.escape_html("<div>") == "&lt;div&gt;"

    def test_truncate(self):
        assert Format.truncate("Hello World", 8) == "Hello..."

    def test_pad(self):
        assert Format.pad("Hi", 5) == "Hi   "

    def test_wrap(self):
        result = Format.wrap("Hello World Test", 10)
        assert "\n" in result

    def test_types_accessible(self):
        # Verify types are accessible on the namespace
        assert Format.MemoryEntry is not None
        assert Format.Tool is not None
        assert Format.ToolParameter is not None
        assert Format.FunctionCall is not None

    def test_extract_json(self):
        result = Format.extract_json('Result: {"key": "value"}')
        assert result == '{"key": "value"}'

    def test_validate_json(self):
        is_valid, error = Format.validate_json('{"valid": true}')
        assert is_valid is True
        assert error is None

    def test_parse_function_call(self):
        result = Format.parse_function_call('test({"a": 1})')
        assert result is not None
        assert result.name == "test"
        assert result.arguments == {"a": 1}
