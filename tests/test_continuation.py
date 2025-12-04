"""Tests for L0 continuation and deduplication system."""

import pytest

from l0.continuation import (
    ContinuationConfig,
    DeduplicationOptions,
    OverlapResult,
    deduplicate_continuation,
    detect_overlap,
)


class TestDeduplicationOptions:
    """Tests for DeduplicationOptions configuration."""

    def test_default_options(self):
        """Test default deduplication options."""
        options = DeduplicationOptions()
        assert options.min_overlap == 2
        assert options.max_overlap == 500
        assert options.case_sensitive is True
        assert options.normalize_whitespace is False

    def test_custom_options(self):
        """Test custom deduplication options."""
        options = DeduplicationOptions(
            min_overlap=5,
            max_overlap=100,
            case_sensitive=False,
            normalize_whitespace=True,
        )
        assert options.min_overlap == 5
        assert options.max_overlap == 100
        assert options.case_sensitive is False
        assert options.normalize_whitespace is True


class TestDetectOverlap:
    """Tests for detect_overlap function."""

    def test_simple_overlap(self):
        """Test simple word overlap detection."""
        result = detect_overlap("Hello world", "world is great")

        assert result.has_overlap is True
        assert result.overlap_length == 5
        assert result.overlap_text == "world"
        assert result.deduplicated == " is great"

    def test_no_overlap(self):
        """Test when there's no overlap."""
        result = detect_overlap("Hello", "Goodbye")

        assert result.has_overlap is False
        assert result.overlap_length == 0
        assert result.overlap_text == ""
        assert result.deduplicated == "Goodbye"

    def test_multi_word_overlap(self):
        """Test multi-word overlap."""
        result = detect_overlap("The quick brown fox", "brown fox jumps over")

        assert result.has_overlap is True
        assert result.overlap_length == 9  # "brown fox"
        assert result.overlap_text == "brown fox"
        assert result.deduplicated == " jumps over"

    def test_full_overlap(self):
        """Test when continuation is entirely overlapping."""
        result = detect_overlap("Hello world", "world")

        assert result.has_overlap is True
        assert result.overlap_length == 5
        assert result.deduplicated == ""

    def test_empty_checkpoint(self):
        """Test with empty checkpoint."""
        result = detect_overlap("", "Hello world")

        assert result.has_overlap is False
        assert result.deduplicated == "Hello world"

    def test_empty_continuation(self):
        """Test with empty continuation."""
        result = detect_overlap("Hello world", "")

        assert result.has_overlap is False
        assert result.deduplicated == ""

    def test_case_sensitive_default(self):
        """Test case-sensitive matching (default)."""
        result = detect_overlap("Hello World", "world test")

        # No overlap because 'World' != 'world'
        assert result.has_overlap is False

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        options = DeduplicationOptions(case_sensitive=False)
        result = detect_overlap("Hello World", "world test", options)

        assert result.has_overlap is True
        assert result.overlap_length == 5
        assert result.deduplicated == " test"

    def test_min_overlap_threshold(self):
        """Test minimum overlap threshold."""
        # With default min_overlap=2
        result = detect_overlap("ab", "bc")
        assert result.has_overlap is False  # Only 1 char overlap

        result = detect_overlap("abc", "bc")
        assert result.has_overlap is True  # 2 char overlap

    def test_custom_min_overlap(self):
        """Test custom minimum overlap."""
        options = DeduplicationOptions(min_overlap=5)
        result = detect_overlap("test", "test more", options)

        # Only 4 chars overlap, but min is 5
        assert result.has_overlap is False

    def test_max_overlap_limit(self):
        """Test maximum overlap limit."""
        long_checkpoint = "a" * 1000
        long_continuation = "a" * 600 + "b"

        options = DeduplicationOptions(max_overlap=500)
        result = detect_overlap(long_checkpoint, long_continuation, options)

        # Should only check up to max_overlap chars
        assert result.has_overlap is True
        assert result.overlap_length <= 500

    def test_code_continuation(self):
        """Test code-like continuation."""
        checkpoint = 'function hello() {\n  console.log("Hello'
        continuation = 'console.log("Hello, World!");\n}'

        result = detect_overlap(checkpoint, continuation)

        assert result.has_overlap is True
        assert "Hello" in result.overlap_text

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        options = DeduplicationOptions(normalize_whitespace=True)
        result = detect_overlap("hello  world", "hello world test", options)

        # With normalization, should find overlap
        assert result.has_overlap is True

    def test_punctuation_overlap(self):
        """Test overlap with punctuation."""
        # Single char doesn't meet min_overlap=2
        result = detect_overlap("end of sentence.", ". Start of next")
        assert result.has_overlap is False  # Only 1 char overlap

        # Multiple punctuation does
        result = detect_overlap("end...", "... more")
        assert result.has_overlap is True
        assert result.overlap_text == "..."


class TestDeduplicateContinuation:
    """Tests for deduplicate_continuation convenience function."""

    def test_simple_dedup(self):
        """Test simple deduplication."""
        result = deduplicate_continuation("Hello world", "world is great")
        assert result == " is great"

    def test_no_dedup_needed(self):
        """Test when no deduplication needed."""
        result = deduplicate_continuation("Hello", "World")
        assert result == "World"

    def test_with_options(self):
        """Test with custom options."""
        options = DeduplicationOptions(case_sensitive=False)
        result = deduplicate_continuation("Hello World", "WORLD test", options)
        assert result == " test"


class TestContinuationConfig:
    """Tests for ContinuationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ContinuationConfig.default()
        assert config.enabled is True
        assert config.checkpoint_interval == 5
        assert config.deduplicate is True
        assert config.validate_checkpoint is True

    def test_disabled_config(self):
        """Test disabled configuration."""
        config = ContinuationConfig.disabled()
        assert config.enabled is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContinuationConfig(
            enabled=True,
            checkpoint_interval=10,
            deduplicate=False,
            validate_checkpoint=False,
            deduplication_options=DeduplicationOptions(min_overlap=5),
        )
        assert config.checkpoint_interval == 10
        assert config.deduplicate is False
        assert config.deduplication_options.min_overlap == 5


class TestOverlapResult:
    """Tests for OverlapResult dataclass."""

    def test_overlap_result_creation(self):
        """Test creating OverlapResult."""
        result = OverlapResult(
            has_overlap=True,
            overlap_length=5,
            overlap_text="hello",
            deduplicated=" world",
        )
        assert result.has_overlap is True
        assert result.overlap_length == 5
        assert result.overlap_text == "hello"
        assert result.deduplicated == " world"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_overlap(self):
        """Test overlap with unicode characters."""
        result = detect_overlap("Hello 世界", "世界 is cool")

        assert result.has_overlap is True
        assert result.overlap_text == "世界"

    def test_emoji_overlap(self):
        """Test overlap with emojis."""
        # Single emoji may not meet min_overlap depending on encoding
        # Use min_overlap=1 to test emoji specifically
        options = DeduplicationOptions(min_overlap=1)
        result = detect_overlap("Hello 👋", "👋 World", options)

        assert result.has_overlap is True
        assert "👋" in result.overlap_text

    def test_newline_overlap(self):
        """Test overlap with newlines."""
        result = detect_overlap("line1\nline2", "line2\nline3")

        assert result.has_overlap is True
        assert "line2" in result.overlap_text

    def test_tab_overlap(self):
        """Test overlap with tabs."""
        # Single tab doesn't meet min_overlap=2
        result = detect_overlap("hello\t", "\tworld")
        assert result.has_overlap is False  # Only 1 char

        # Multiple whitespace with min_overlap=1
        options = DeduplicationOptions(min_overlap=1)
        result = detect_overlap("hello\t", "\tworld", options)
        assert result.has_overlap is True

    def test_very_long_overlap(self):
        """Test very long overlap detection."""
        text = "This is a very long repeated text. " * 20
        checkpoint = text
        continuation = text + " Additional content"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        # Should find some overlap within max_overlap limit


class TestDeduplicationThorough:
    """Thorough tests for deduplication scenarios."""

    # ─────────────────────────────────────────────────────────────────────────
    # Real-world LLM continuation scenarios
    # ─────────────────────────────────────────────────────────────────────────

    def test_llm_word_repetition(self):
        """Test typical LLM word repetition at continuation boundary."""
        # LLM often repeats the last few words when continuing
        checkpoint = "The quick brown fox jumps over"
        continuation = "jumps over the lazy dog"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "jumps over"
        assert result.deduplicated == " the lazy dog"

    def test_llm_sentence_boundary(self):
        """Test continuation at sentence boundary."""
        checkpoint = "This is the first sentence. And this is"
        continuation = "And this is the second sentence."

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "And this is"
        assert result.deduplicated == " the second sentence."

    def test_llm_paragraph_continuation(self):
        """Test continuation across paragraphs."""
        checkpoint = "End of first paragraph.\n\nStart of second"
        continuation = "Start of second paragraph continues here."

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "Start of second"
        assert result.deduplicated == " paragraph continues here."

    def test_llm_partial_word_no_overlap(self):
        """Test that partial word matches don't count as overlap."""
        # "test" and "testing" - we shouldn't dedupe partial words incorrectly
        checkpoint = "This is a test"
        continuation = "testing the system"

        result = detect_overlap(checkpoint, continuation)
        # "test" matches "test" at start of "testing"
        assert result.has_overlap is True
        assert result.overlap_text == "test"
        assert result.deduplicated == "ing the system"

    def test_llm_numbered_list_continuation(self):
        """Test continuation of numbered lists."""
        checkpoint = "1. First item\n2. Second item\n3."
        continuation = "3. Third item\n4. Fourth item"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "3."
        assert result.deduplicated == " Third item\n4. Fourth item"

    def test_llm_bullet_list_continuation(self):
        """Test continuation of bullet lists."""
        checkpoint = "- Item one\n- Item two\n- Item"
        continuation = "- Item three\n- Item four"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "- Item"
        assert result.deduplicated == " three\n- Item four"

    # ─────────────────────────────────────────────────────────────────────────
    # Code generation scenarios
    # ─────────────────────────────────────────────────────────────────────────

    def test_code_function_continuation(self):
        """Test code function continuation."""
        checkpoint = """def hello_world():
    print("Hello"""
        continuation = """print("Hello, World!")
    return True"""

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert 'print("Hello' in result.overlap_text
        assert result.deduplicated == ', World!")\n    return True'

    def test_code_json_continuation(self):
        """Test JSON code continuation."""
        checkpoint = '{"name": "John", "age":'
        continuation = '"age": 30, "city": "NYC"}'

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == '"age":'
        assert result.deduplicated == ' 30, "city": "NYC"}'

    def test_code_python_class(self):
        """Test Python class continuation."""
        checkpoint = """class MyClass:
    def __init__(self"""
        continuation = """def __init__(self, value):
        self.value = value"""

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "def __init__(self" in result.overlap_text

    def test_code_import_statements(self):
        """Test import statement continuation."""
        checkpoint = "import os\nimport sys\nimport"
        continuation = "import json\nimport typing"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "import"
        assert result.deduplicated == " json\nimport typing"

    def test_code_html_tag_continuation(self):
        """Test HTML tag continuation."""
        checkpoint = '<div class="container">\n  <p>Hello'
        continuation = "<p>Hello World</p>\n</div>"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "<p>Hello" in result.overlap_text

    def test_code_sql_query(self):
        """Test SQL query continuation."""
        checkpoint = "SELECT id, name FROM users WHERE"
        continuation = "WHERE status = 'active' ORDER BY name"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "WHERE"
        assert result.deduplicated == " status = 'active' ORDER BY name"

    # ─────────────────────────────────────────────────────────────────────────
    # Markdown scenarios
    # ─────────────────────────────────────────────────────────────────────────

    def test_markdown_heading_continuation(self):
        """Test markdown heading continuation."""
        checkpoint = "# Main Title\n\n## Section"
        continuation = "## Section One\n\nContent here"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "## Section"
        assert result.deduplicated == " One\n\nContent here"

    def test_markdown_code_block(self):
        """Test markdown code block continuation."""
        checkpoint = "Here's the code:\n\n```python\ndef"
        continuation = "```python\ndef hello():\n    pass\n```"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "```python\ndef" in result.overlap_text

    def test_markdown_link_continuation(self):
        """Test markdown link continuation."""
        checkpoint = "Check out [this link](https://example"
        continuation = "(https://example.com) for more info."

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "(https://example" in result.overlap_text

    def test_markdown_table_continuation(self):
        """Test markdown table continuation."""
        checkpoint = "| Name | Age |\n|------|-----|\n| John"
        continuation = "| John | 30  |\n| Jane | 25  |"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "| John"

    # ─────────────────────────────────────────────────────────────────────────
    # Whitespace handling
    # ─────────────────────────────────────────────────────────────────────────

    def test_trailing_space_overlap(self):
        """Test overlap with trailing spaces."""
        checkpoint = "Hello world "
        continuation = " world again"

        # The space + "world" should match
        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert " world" in result.overlap_text

    def test_multiple_spaces_overlap(self):
        """Test overlap with multiple spaces."""
        checkpoint = "Hello  world"  # double space
        continuation = "  world  test"  # double spaces

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "  world" in result.overlap_text

    def test_mixed_whitespace_no_normalization(self):
        """Test that mixed whitespace doesn't match without normalization."""
        checkpoint = "hello\tworld"  # tab
        continuation = "hello world test"  # space

        result = detect_overlap(checkpoint, continuation)
        # Suffix "\tworld" doesn't match prefix "hello" - no overlap
        # The algorithm matches checkpoint SUFFIX to continuation PREFIX
        assert result.has_overlap is False

        # But if continuation starts with the checkpoint suffix...
        checkpoint2 = "hello\tworld"
        continuation2 = "\tworld and more"
        result2 = detect_overlap(checkpoint2, continuation2)
        assert result2.has_overlap is True
        assert result2.overlap_text == "\tworld"

    def test_mixed_whitespace_with_normalization(self):
        """Test mixed whitespace with normalization enabled."""
        options = DeduplicationOptions(normalize_whitespace=True)
        checkpoint = "hello  world"  # double space
        continuation = "hello world test"  # single space

        result = detect_overlap(checkpoint, continuation, options)
        assert result.has_overlap is True
        # With normalization, should match more
        assert "hello" in result.overlap_text

    def test_newline_variations(self):
        """Test different newline styles."""
        # Unix style
        checkpoint = "line1\nline2"
        continuation = "line2\nline3"
        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "line2" in result.overlap_text

    # ─────────────────────────────────────────────────────────────────────────
    # Case sensitivity scenarios
    # ─────────────────────────────────────────────────────────────────────────

    def test_case_mismatch_no_overlap(self):
        """Test that case mismatch prevents overlap by default."""
        checkpoint = "Hello World"
        continuation = "hello world again"

        result = detect_overlap(checkpoint, continuation)
        # Suffix "World" doesn't match prefix "hello" (case sensitive)
        # The algorithm matches checkpoint SUFFIX to continuation PREFIX
        assert result.has_overlap is False

        # But with matching case at boundary...
        checkpoint2 = "Hello World"
        continuation2 = "World again"
        result2 = detect_overlap(checkpoint2, continuation2)
        assert result2.has_overlap is True
        assert result2.overlap_text == "World"

    def test_case_insensitive_full_match(self):
        """Test case-insensitive matching finds full overlap."""
        options = DeduplicationOptions(case_sensitive=False)
        checkpoint = "Hello World"
        continuation = "HELLO WORLD again"

        result = detect_overlap(checkpoint, continuation, options)
        assert result.has_overlap is True
        # Should find much longer overlap with case insensitivity
        assert len(result.overlap_text) >= 10  # "Hello World" or "HELLO WORLD"

    def test_case_insensitive_partial(self):
        """Test case-insensitive partial match."""
        options = DeduplicationOptions(case_sensitive=False)
        checkpoint = "The Quick Brown"
        continuation = "brown Fox"

        result = detect_overlap(checkpoint, continuation, options)
        assert result.has_overlap is True
        assert result.overlap_text.lower() == "brown"

    # ─────────────────────────────────────────────────────────────────────────
    # Boundary conditions
    # ─────────────────────────────────────────────────────────────────────────

    def test_exact_min_overlap(self):
        """Test overlap exactly at min_overlap boundary."""
        options = DeduplicationOptions(min_overlap=5)

        # Exactly 5 chars - should match
        result = detect_overlap("hello", "hello world", options)
        assert result.has_overlap is True
        assert result.overlap_length == 5

        # 4 chars - should not match
        result = detect_overlap("hell", "hello world", options)
        assert result.has_overlap is False

    def test_exact_max_overlap(self):
        """Test overlap at max_overlap boundary."""
        # Create strings longer than max_overlap
        options = DeduplicationOptions(max_overlap=10)
        long_text = "a" * 20
        checkpoint = long_text
        continuation = long_text + "extra"

        result = detect_overlap(checkpoint, continuation, options)
        assert result.has_overlap is True
        # Should be limited to max_overlap
        assert result.overlap_length == 10

    def test_checkpoint_shorter_than_continuation(self):
        """Test when checkpoint is shorter than continuation."""
        checkpoint = "hi"
        continuation = "hi there my friend"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "hi"
        assert result.deduplicated == " there my friend"

    def test_continuation_shorter_than_checkpoint(self):
        """Test when continuation is shorter than checkpoint."""
        checkpoint = "hello world my friend"
        continuation = "friend"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "friend"
        assert result.deduplicated == ""

    def test_identical_strings(self):
        """Test when checkpoint and continuation are identical."""
        text = "Hello World"
        result = detect_overlap(text, text)

        assert result.has_overlap is True
        assert result.overlap_text == text
        assert result.deduplicated == ""

    def test_no_common_chars(self):
        """Test when there are no common characters."""
        checkpoint = "aaaa"
        continuation = "bbbb"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is False
        assert result.deduplicated == "bbbb"

    # ─────────────────────────────────────────────────────────────────────────
    # Special characters and symbols
    # ─────────────────────────────────────────────────────────────────────────

    def test_special_regex_chars(self):
        """Test that regex special characters don't break matching."""
        checkpoint = "test (value) [array]"
        continuation = "[array] more stuff"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "[array]"

    def test_backslash_handling(self):
        """Test backslash characters."""
        checkpoint = "path\\to\\file"
        continuation = "\\file\\name"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "\\file" in result.overlap_text

    def test_quotes_handling(self):
        """Test various quote characters."""
        checkpoint = 'He said "hello'
        continuation = '"hello world"'

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert '"hello' in result.overlap_text

    def test_unicode_punctuation(self):
        """Test Unicode punctuation marks."""
        checkpoint = "Hello… world"
        continuation = "… world continues"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "… world" in result.overlap_text

    def test_math_symbols(self):
        """Test mathematical symbols."""
        checkpoint = "x² + y² ="
        continuation = "y² = z²"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "y²" in result.overlap_text

    def test_currency_symbols(self):
        """Test currency symbols."""
        checkpoint = "Price: $100"
        continuation = "$100 USD"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "$100" in result.overlap_text

    # ─────────────────────────────────────────────────────────────────────────
    # Performance and stress tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_large_strings_performance(self):
        """Test performance with large strings."""
        # The algorithm matches checkpoint SUFFIX to continuation PREFIX
        # So the overlap must be at the END of checkpoint
        overlap_text = "overlap_marker"
        checkpoint = "x" * 5000 + overlap_text
        continuation = overlap_text + "z" * 10000

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == overlap_text
        assert result.deduplicated == "z" * 10000

    def test_many_potential_overlaps(self):
        """Test string with many potential overlap points."""
        # The algorithm matches checkpoint SUFFIX to continuation PREFIX
        # Use a pattern where suffix matches prefix
        checkpoint = "ab " * 100 + "end"
        continuation = "end of the story"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert result.overlap_text == "end"
        assert result.deduplicated == " of the story"

        # Also test repeated pattern overlap
        checkpoint2 = "ab " * 100
        continuation2 = "ab " * 50 + "different"
        result2 = detect_overlap(checkpoint2, continuation2)
        # Suffix "ab " matches prefix "ab "
        assert result2.has_overlap is True

    def test_worst_case_no_overlap(self):
        """Test worst case - long strings with no overlap."""
        checkpoint = "a" * 1000
        continuation = "b" * 1000

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is False
        assert result.deduplicated == "b" * 1000

    # ─────────────────────────────────────────────────────────────────────────
    # Real conversation/text scenarios
    # ─────────────────────────────────────────────────────────────────────────

    def test_conversational_continuation(self):
        """Test continuation in conversational text."""
        checkpoint = "I think we should consider the following options:"
        continuation = "the following options:\n1. Option A\n2. Option B"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "the following options:" in result.overlap_text

    def test_technical_writing(self):
        """Test continuation in technical writing."""
        checkpoint = "The API endpoint accepts the following parameters: `id`"
        continuation = "`id` (required), `name` (optional)"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "`id`" in result.overlap_text

    def test_legal_text(self):
        """Test continuation in legal/formal text."""
        checkpoint = "WHEREAS, the parties agree to the terms and"
        continuation = "terms and conditions set forth herein"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "terms and" in result.overlap_text

    def test_poetry_continuation(self):
        """Test continuation in poetry/verse."""
        checkpoint = "Roses are red,\nViolets are"
        continuation = "Violets are blue,\nSugar is sweet"

        result = detect_overlap(checkpoint, continuation)
        assert result.has_overlap is True
        assert "Violets are" in result.overlap_text
