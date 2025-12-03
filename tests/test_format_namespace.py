"""Tests for l0.Format namespace."""

from __future__ import annotations

import l0


class TestFormatNamespace:
    """Tests for the Format namespace class."""

    def test_context(self):
        result = l0.Format.context("Hello", label="test")
        assert "<test>" in result
        assert "Hello" in result

    def test_contexts(self):
        items = [{"content": "A", "label": "a"}, {"content": "B", "label": "b"}]
        result = l0.Format.contexts(items)
        assert "<a>" in result
        assert "<b>" in result

    def test_document(self):
        result = l0.Format.document("Content", {"title": "Test"})
        assert "<title>Test</title>" in result

    def test_instructions(self):
        result = l0.Format.instructions("Be helpful")
        assert "<system_instructions>" in result

    def test_memory(self):
        mem = [{"role": "user", "content": "Hi"}]
        result = l0.Format.memory(mem)
        assert "User: Hi" in result

    def test_memory_entry(self):
        entry = l0.Format.memory_entry("user", "Hello")
        assert entry.role == "user"
        assert entry.content == "Hello"
        assert entry.timestamp is not None

    def test_json_output(self):
        result = l0.Format.json_output({"strict": True})
        assert "valid JSON only" in result

    def test_structured_output(self):
        result = l0.Format.structured_output("yaml", {"strict": True})
        assert "YAML" in result

    def test_create_tool(self):
        tool = l0.Format.create_tool("test", "Test tool")
        assert tool.name == "test"
        assert tool.description == "Test tool"

    def test_parameter(self):
        param = l0.Format.parameter("loc", "string", "Location", True)
        assert param.name == "loc"
        assert param.required is True

    def test_tool(self):
        tool = l0.Format.create_tool("test", "Test")
        result = l0.Format.tool(tool, {"style": "natural"})
        assert "Tool: test" in result

    def test_escape_html(self):
        assert l0.Format.escape_html("<div>") == "&lt;div&gt;"

    def test_truncate(self):
        assert l0.Format.truncate("Hello World", 8) == "Hello..."

    def test_pad(self):
        assert l0.Format.pad("Hi", 5) == "Hi   "

    def test_wrap(self):
        result = l0.Format.wrap("Hello World Test", 10)
        assert "\n" in result

    def test_types_accessible(self):
        # Verify types are accessible on the namespace
        assert l0.Format.MemoryEntry is not None
        assert l0.Format.Tool is not None
        assert l0.Format.ToolParameter is not None
        assert l0.Format.FunctionCall is not None

    def test_extract_json(self):
        result = l0.Format.extract_json('Result: {"key": "value"}')
        assert result == '{"key": "value"}'

    def test_validate_json(self):
        is_valid, error = l0.Format.validate_json('{"valid": true}')
        assert is_valid is True
        assert error is None

    def test_parse_function_call(self):
        result = l0.Format.parse_function_call('test({"a": 1})')
        assert result is not None
        assert result.name == "test"
        assert result.arguments == {"a": 1}
