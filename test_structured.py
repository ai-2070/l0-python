"""Test structured output implementation."""

from l0._utils import auto_correct_json


def test_auto_correct():
    """Test auto_correct_json function."""

    # Test 1: Text prefix removal
    result = auto_correct_json(
        'Sure! Here is the JSON: {"name": "Alice"}', track_corrections=True
    )
    assert '{"name": "Alice"}' in result.text, f"Got: {result.text}"
    print("Test 1 (text prefix): PASS")

    # Test 2: Trailing comma
    result = auto_correct_json('{"a": 1,}', track_corrections=True)
    assert result.text == '{"a": 1}', f"Got: {result.text}"
    assert result.corrected
    print("Test 2 (trailing comma): PASS")

    # Test 3: Missing brace
    result = auto_correct_json('{"name": "Alice"', track_corrections=True)
    assert result.text == '{"name": "Alice"}', f"Got: {result.text}"
    assert result.corrected
    assert any("brace" in c.lower() for c in result.corrections)
    print("Test 3 (missing brace): PASS")

    # Test 4: Single quotes
    result = auto_correct_json("{'name': 'Alice'}", track_corrections=True)
    assert '"name"' in result.text, f"Got: {result.text}"
    assert '"Alice"' in result.text, f"Got: {result.text}"
    print("Test 4 (single quotes): PASS")

    # Test 5: Markdown fence
    result = auto_correct_json('```json\n{"a": 1}\n```', track_corrections=True)
    assert result.text == '{"a": 1}', f"Got: {result.text}"
    print("Test 5 (markdown fence): PASS")

    # Test 6: Text suffix removal
    result = auto_correct_json(
        '{"a": 1} Let me know if you need anything else!', track_corrections=True
    )
    assert result.text == '{"a": 1}', f"Got: {result.text}"
    print("Test 6 (text suffix): PASS")

    # Test 7: Complex case
    text = """Sure! Here's the user data:
```json
{"name": "Bob", "age": 30,}
```
Hope this helps!"""
    result = auto_correct_json(text, track_corrections=True)
    assert '"name"' in result.text
    assert '"Bob"' in result.text
    assert ",}" not in result.text  # Trailing comma removed
    print("Test 7 (complex case): PASS")

    print("\nAll auto_correct tests passed!")


def test_imports():
    """Test all structured imports."""
    from l0 import (
        structured,
        structured_stream,
        StructuredResult,
        StructuredStreamResult,
        AutoCorrectInfo,
    )

    print("All imports successful!")


if __name__ == "__main__":
    test_imports()
    test_auto_correct()
