"""Tests for l0.guardrails module."""

import time
from typing import Any

import pytest

from l0.guardrails import (
    BAD_PATTERNS,
    GuardrailRule,
    Guardrails,
    GuardrailViolation,
    JsonAnalysis,
    LatexAnalysis,
    MarkdownAnalysis,
    analyze_json_structure,
    analyze_latex_structure,
    analyze_markdown_structure,
    check_guardrails,
    custom_pattern_rule,
    find_bad_patterns,
    is_noise_only,
    is_zero_output,
    json_rule,
    latex_rule,
    looks_like_json,
    looks_like_latex,
    looks_like_markdown,
    markdown_rule,
    pattern_rule,
    recommended_guardrails,
    repetition_rule,
    stall_rule,
    strict_guardrails,
    strict_json_rule,
    zero_output_rule,
)
from l0.types import State

# ─────────────────────────────────────────────────────────────────────────────
# Core Types
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailViolation:
    def test_create_violation(self):
        v = GuardrailViolation(
            rule="test",
            message="Test message",
            severity="error",
        )
        assert v.rule == "test"
        assert v.message == "Test message"
        assert v.severity == "error"
        assert v.recoverable is True

    def test_violation_with_all_fields(self):
        v = GuardrailViolation(
            rule="test",
            message="Test",
            severity="fatal",
            recoverable=False,
            position=10,
            timestamp=123.456,
            context={"key": "value"},
            suggestion="Fix it",
        )
        assert v.position == 10
        assert v.timestamp == 123.456
        assert v.context == {"key": "value"}
        assert v.suggestion == "Fix it"


# ─────────────────────────────────────────────────────────────────────────────
# JSON Analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeJsonStructure:
    def test_balanced_json(self):
        result = analyze_json_structure('{"key": "value"}')
        assert result.is_balanced is True
        assert result.open_braces == 1
        assert result.close_braces == 1
        assert len(result.issues) == 0

    def test_unbalanced_braces(self):
        result = analyze_json_structure('{"key": "value"')
        assert result.is_balanced is False
        assert result.open_braces == 1
        assert result.close_braces == 0
        assert any("Unbalanced braces" in issue for issue in result.issues)

    def test_too_many_closes(self):
        result = analyze_json_structure('{"key": "value"}}')
        assert result.is_balanced is False
        assert any("Too many closing braces" in issue for issue in result.issues)

    def test_unbalanced_brackets(self):
        result = analyze_json_structure("[1, 2, 3")
        assert result.is_balanced is False
        assert result.open_brackets == 1
        assert result.close_brackets == 0

    def test_unclosed_string(self):
        result = analyze_json_structure('{"key": "unclosed')
        assert result.unclosed_string is True
        assert result.in_string is True
        assert any("Unclosed string" in issue for issue in result.issues)

    def test_consecutive_commas(self):
        result = analyze_json_structure('{"a": 1,, "b": 2}')
        assert any("consecutive commas" in issue for issue in result.issues)

    def test_malformed_pattern_brace(self):
        result = analyze_json_structure("[,{]")
        assert any("Malformed pattern" in issue for issue in result.issues)

    def test_nested_json(self):
        result = analyze_json_structure('{"a": {"b": [1, 2, {"c": 3}]}}')
        assert result.is_balanced is True
        assert result.open_braces == 3
        assert result.close_braces == 3

    def test_string_with_braces(self):
        # Braces inside strings should not be counted
        result = analyze_json_structure('{"key": "{not a brace}"}')
        assert result.is_balanced is True


class TestLooksLikeJson:
    def test_object_start(self):
        assert looks_like_json('{"key": "value"}') is True

    def test_array_start(self):
        assert looks_like_json("[1, 2, 3]") is True

    def test_not_json(self):
        assert looks_like_json("Hello world") is False

    def test_empty(self):
        assert looks_like_json("") is False

    def test_whitespace_then_json(self):
        assert looks_like_json('  {"key": 1}') is True


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeMarkdownStructure:
    def test_balanced_fences(self):
        result = analyze_markdown_structure("```js\ncode\n```")
        assert result.is_balanced is True
        assert result.in_fence is False
        assert result.open_fences == 1
        assert result.close_fences == 1

    def test_unclosed_fence(self):
        result = analyze_markdown_structure("```js\ncode")
        assert result.is_balanced is False
        assert result.in_fence is True
        assert any("Unclosed code fence" in issue for issue in result.issues)

    def test_fence_language_detected(self):
        result = analyze_markdown_structure("```python\ncode\n```")
        assert "python" in result.fence_languages

    def test_multiple_fences(self):
        content = "```js\ncode1\n```\n\n```py\ncode2\n```"
        result = analyze_markdown_structure(content)
        assert result.is_balanced is True
        assert result.open_fences == 2


class TestLooksLikeMarkdown:
    def test_header(self):
        assert looks_like_markdown("# Header") is True

    def test_code_fence(self):
        assert looks_like_markdown("```code```") is True

    def test_list(self):
        assert looks_like_markdown("- item") is True

    def test_link(self):
        assert looks_like_markdown("[text](url)") is True

    def test_bold(self):
        assert looks_like_markdown("**bold**") is True

    def test_plain_text(self):
        assert looks_like_markdown("Just plain text") is False


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeLatexStructure:
    def test_balanced_environment(self):
        result = analyze_latex_structure(r"\begin{equation}x=1\end{equation}")
        assert result.is_balanced is True
        assert len(result.open_environments) == 0

    def test_unclosed_environment(self):
        result = analyze_latex_structure(r"\begin{equation}x=1")
        assert result.is_balanced is False
        assert "equation" in result.open_environments

    def test_mismatched_environment(self):
        result = analyze_latex_structure(r"\begin{equation}x=1\end{align}")
        assert any("Mismatched" in issue for issue in result.issues)

    def test_starred_environment_balanced(self):
        """Starred environments like align* should be detected."""
        result = analyze_latex_structure(r"\begin{align*}x=1\end{align*}")
        assert result.is_balanced is True
        assert len(result.open_environments) == 0

    def test_starred_environment_unbalanced(self):
        """Unclosed starred environments should be detected."""
        result = analyze_latex_structure(r"\begin{align*}x=1")
        assert result.is_balanced is False
        assert "align*" in result.open_environments

    def test_starred_environment_mismatched(self):
        """Mismatched starred environments should be detected."""
        result = analyze_latex_structure(r"\begin{align*}x=1\end{equation*}")
        assert any("Mismatched" in issue for issue in result.issues)

    def test_display_math_balanced(self):
        result = analyze_latex_structure("$$x=1$$")
        assert result.display_math_balanced is True

    def test_display_math_unbalanced(self):
        result = analyze_latex_structure("$$x=1")
        assert result.display_math_balanced is False

    def test_bracket_math_balanced(self):
        result = analyze_latex_structure(r"\[x=1\]")
        assert result.bracket_math_balanced is True

    def test_bracket_math_unbalanced(self):
        result = analyze_latex_structure(r"\[x=1")
        assert result.bracket_math_balanced is False

    def test_inline_math_balanced(self):
        result = analyze_latex_structure("$x=1$")
        assert result.inline_math_balanced is True

    def test_inline_math_unbalanced(self):
        result = analyze_latex_structure("$x=1")
        assert result.inline_math_balanced is False

    def test_escaped_dollar_not_counted_as_math(self):
        """Escaped dollar signs should not be treated as math delimiters."""
        result = analyze_latex_structure(r"The price is \$100")
        assert result.inline_math_balanced is True

    def test_escaped_dollar_with_actual_math(self):
        """Escaped dollars mixed with real math should work correctly."""
        result = analyze_latex_structure(r"Cost is \$50 and $x=1$")
        assert result.inline_math_balanced is True

    def test_multiple_escaped_dollars(self):
        """Multiple escaped dollar signs should all be ignored."""
        result = analyze_latex_structure(r"\$10, \$20, and \$30")
        assert result.inline_math_balanced is True

    def test_unbalanced_with_escaped_dollar(self):
        """Unbalanced math should still be detected with escaped dollars present."""
        result = analyze_latex_structure(r"\$100 and $x=1")
        assert result.inline_math_balanced is False


class TestLooksLikeLatex:
    def test_begin_end(self):
        assert looks_like_latex(r"\begin{document}") is True

    def test_display_math(self):
        assert looks_like_latex("$$x=1$$") is True

    def test_frac(self):
        assert looks_like_latex(r"\frac{1}{2}") is True

    def test_plain_text(self):
        assert looks_like_latex("Just text") is False


# ─────────────────────────────────────────────────────────────────────────────
# Zero Output / Noise Detection
# ─────────────────────────────────────────────────────────────────────────────


class TestIsZeroOutput:
    def test_empty_string(self):
        assert is_zero_output("") is True

    def test_whitespace_only(self):
        assert is_zero_output("   \n\t  ") is True

    def test_content(self):
        assert is_zero_output("Hello") is False


class TestIsNoiseOnly:
    def test_punctuation_only(self):
        assert is_noise_only("...") is True
        assert is_noise_only("!?!?") is True

    def test_repeated_char(self):
        assert is_noise_only("aaaaaaaaaaaaa") is True

    def test_real_content(self):
        assert is_noise_only("Hello world") is False

    def test_empty(self):
        assert is_noise_only("") is True


# ─────────────────────────────────────────────────────────────────────────────
# Bad Patterns
# ─────────────────────────────────────────────────────────────────────────────


class TestBadPatterns:
    def test_has_categories(self):
        assert len(BAD_PATTERNS.META_COMMENTARY) > 0
        assert len(BAD_PATTERNS.HEDGING) > 0
        assert len(BAD_PATTERNS.REFUSAL) > 0
        assert len(BAD_PATTERNS.INSTRUCTION_LEAK) > 0
        assert len(BAD_PATTERNS.PLACEHOLDERS) > 0
        assert len(BAD_PATTERNS.FORMAT_COLLAPSE) > 0

    def test_all_patterns(self):
        all_patterns = BAD_PATTERNS.all_patterns()
        assert len(all_patterns) > 50  # Should have many patterns


class TestFindBadPatterns:
    def test_find_meta_commentary(self):
        matches = find_bad_patterns(
            "As an AI, I cannot help", BAD_PATTERNS.META_COMMENTARY
        )
        assert len(matches) > 0

    def test_find_hedging(self):
        matches = find_bad_patterns("Sure! I'd be happy to help", BAD_PATTERNS.HEDGING)
        assert len(matches) > 0

    def test_no_matches(self):
        matches = find_bad_patterns(
            "Normal response text", BAD_PATTERNS.META_COMMENTARY
        )
        assert len(matches) == 0


# ─────────────────────────────────────────────────────────────────────────────
# JSON Rule
# ─────────────────────────────────────────────────────────────────────────────


class TestJsonRule:
    def test_balanced_json_passes(self):
        state = State(content='{"key": "value"}')
        rule = json_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_unbalanced_json_fails(self):
        state = State(content='{"key": "value"}}')
        rule = json_rule()
        violations = rule.check(state)
        assert len(violations) >= 1
        assert violations[0].rule == "json"

    def test_non_json_skipped(self):
        state = State(content="Just plain text")
        rule = json_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_malformed_pattern_detected(self):
        # Test unbalanced braces which is detected as malformed
        state = State(content='{"a": 1}}}', completed=True)
        rule = json_rule()
        violations = rule.check(state)
        assert len(violations) >= 1
        assert any("brace" in v.message.lower() for v in violations)

    def test_missing_closing_brace_on_completion(self):
        """Test that missing closing braces are detected on completion."""
        state = State(content='{"key": "value"', completed=True)
        rule = json_rule()
        violations = rule.check(state)
        assert len(violations) >= 1
        assert any("missing" in v.message.lower() for v in violations)

    def test_missing_closing_bracket_on_completion(self):
        """Test that missing closing brackets are detected on completion."""
        state = State(content="[1, 2, 3", completed=True)
        rule = json_rule()
        violations = rule.check(state)
        assert len(violations) >= 1
        assert any("missing" in v.message.lower() for v in violations)


class TestStrictJsonRule:
    def test_valid_json_passes(self):
        state = State(content='{"key": "value"}', completed=True)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_invalid_json_fails(self):
        state = State(content='{"key": value}', completed=True)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 1
        assert "Invalid JSON" in violations[0].message

    def test_incomplete_stream_skipped(self):
        state = State(content='{"key":', completed=False)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_non_json_content_fails(self):
        """Test that non-JSON content is flagged as a violation."""
        state = State(content="This is plain text", completed=True)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 1
        assert "does not appear to be JSON" in violations[0].message

    def test_empty_content_fails(self):
        """Test that empty content is flagged as a violation."""
        state = State(content="", completed=True)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 1
        assert "Empty output" in violations[0].message

    def test_whitespace_only_content_fails(self):
        """Test that whitespace-only content is flagged as a violation."""
        state = State(content="   \n\t  ", completed=True)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 1
        assert "Empty output" in violations[0].message


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Rule
# ─────────────────────────────────────────────────────────────────────────────


class TestMarkdownRule:
    def test_balanced_markdown_passes(self):
        state = State(content="```js\ncode\n```", completed=True)
        rule = markdown_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_unclosed_fence_fails(self):
        state = State(content="```js\ncode", completed=True)
        rule = markdown_rule()
        violations = rule.check(state)
        assert len(violations) >= 1
        assert any("Unclosed" in v.message for v in violations)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Rule
# ─────────────────────────────────────────────────────────────────────────────


class TestLatexRule:
    def test_balanced_latex_passes(self):
        state = State(content=r"\begin{equation}x\end{equation}", completed=True)
        rule = latex_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_unclosed_environment_fails(self):
        state = State(content=r"\begin{equation}x", completed=True)
        rule = latex_rule()
        violations = rule.check(state)
        assert len(violations) >= 1

    def test_non_latex_skipped(self):
        state = State(content="Just plain text", completed=True)
        rule = latex_rule()
        violations = rule.check(state)
        assert len(violations) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Rule
# ─────────────────────────────────────────────────────────────────────────────


class TestPatternRule:
    def test_default_patterns(self):
        rule = pattern_rule()
        state = State(content="As an AI, I cannot do that")
        violations = rule.check(state)
        assert len(violations) >= 1

    def test_custom_patterns(self):
        rule = pattern_rule(patterns=[r"\bfoo\b"])
        state = State(content="This has foo in it")
        violations = rule.check(state)
        assert len(violations) == 1

    def test_no_match_passes(self):
        rule = pattern_rule()
        state = State(content="Normal response without triggers")
        violations = rule.check(state)
        assert len(violations) == 0

    def test_include_categories(self):
        rule = pattern_rule(include_categories=["META_COMMENTARY"])
        state = State(content="As an AI assistant, I can help")
        violations = rule.check(state)
        assert len(violations) >= 1

    def test_exclude_categories(self):
        rule = pattern_rule(
            exclude_categories=["META_COMMENTARY", "HEDGING", "REFUSAL"]
        )
        state = State(content="As an AI, I cannot help")
        violations = rule.check(state)
        # Should not match META_COMMENTARY or REFUSAL patterns
        assert len(violations) == 0


class TestCustomPatternRule:
    def test_custom_rule(self):
        rule = custom_pattern_rule(
            [r"forbidden", r"blocked"], "Custom violation", "error"
        )
        state = State(content="This is forbidden content")
        violations = rule.check(state)
        assert len(violations) == 1
        assert "Custom violation" in violations[0].message
        assert violations[0].severity == "error"

    def test_no_match(self):
        rule = custom_pattern_rule([r"forbidden"], "Custom", "error")
        state = State(content="Normal content")
        violations = rule.check(state)
        assert len(violations) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Zero Output Rule
# ─────────────────────────────────────────────────────────────────────────────


class TestZeroOutputRule:
    def test_empty_completed_fails(self):
        state = State(content="", completed=True)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 1

    def test_whitespace_only_fails(self):
        state = State(content="   \n\t  ", completed=True)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 1

    def test_noise_only_fails(self):
        state = State(content="...", completed=True)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 1
        assert "noise" in violations[0].message.lower()

    def test_content_passes(self):
        state = State(content="Hello world", completed=True)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_incomplete_stream_skipped(self):
        state = State(content="", completed=False)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Drift Detection
# ─────────────────────────────────────────────────────────────────────────────


class TestStallRule:
    def test_no_stall(self):
        state = State(last_token_at=time.time())
        rule = stall_rule(max_gap=5.0)
        violations = rule.check(state)
        assert len(violations) == 0

    def test_stall_detected(self):
        state = State(last_token_at=time.time() - 10)
        rule = stall_rule(max_gap=5.0)
        violations = rule.check(state)
        assert len(violations) == 1
        assert state.drift_detected is True


class TestRepetitionRule:
    def test_no_repetition(self):
        state = State(content="a" * 100 + "b" * 100)
        rule = repetition_rule(window=100, threshold=0.5)
        violations = rule.check(state)
        # No character-based repetition violation
        char_violations = [v for v in violations if "character" in v.message.lower()]
        assert len(char_violations) == 0

    def test_repetition_detected(self):
        repeated_block = "x" * 100
        state = State(content=repeated_block + repeated_block)
        rule = repetition_rule(window=100, threshold=0.5)
        violations = rule.check(state)
        assert len(violations) >= 1
        assert state.drift_detected is True

    def test_short_content_skipped(self):
        state = State(content="short")
        rule = repetition_rule(window=100)
        violations = rule.check(state)
        assert len(violations) == 0

    def test_sentence_repetition(self):
        sentence = "This is a repeated sentence. "
        state = State(content=sentence * 5)
        rule = repetition_rule(sentence_check=True, sentence_repeat_count=3)
        violations = rule.check(state)
        assert any("repeated" in v.message.lower() for v in violations)


# ─────────────────────────────────────────────────────────────────────────────
# Check Guardrails
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckGuardrails:
    def test_runs_all_rules(self):
        state = State(content="As an AI, I cannot help", completed=True)
        rules = [pattern_rule(), zero_output_rule()]
        violations = check_guardrails(state, rules)
        assert len(violations) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────


class TestPresets:
    def test_minimal_guardrails(self):
        rules = Guardrails.minimal()
        assert len(rules) == 2
        names = [r.name for r in rules]
        assert "json" in names
        assert "zero_output" in names

    def test_recommended_guardrails(self):
        rules = Guardrails.recommended()
        assert len(rules) == 4
        names = [r.name for r in rules]
        assert "json" in names
        assert "markdown" in names
        assert "pattern" in names
        assert "zero_output" in names

    def test_strict_guardrails(self):
        rules = Guardrails.strict()
        assert len(rules) == 8
        names = [r.name for r in rules]
        assert "strict_json" in names
        assert "latex" in names
        assert "stall" in names
        assert "repetition" in names

    def test_json_only(self):
        rules = Guardrails.json_only()
        assert len(rules) == 3
        names = [r.name for r in rules]
        assert "json" in names
        assert "strict_json" in names

    def test_markdown_only(self):
        rules = Guardrails.markdown_only()
        assert len(rules) == 2
        names = [r.name for r in rules]
        assert "markdown" in names

    def test_latex_only(self):
        rules = Guardrails.latex_only()
        assert len(rules) == 2
        names = [r.name for r in rules]
        assert "latex" in names

    def test_none(self):
        rules = Guardrails.none()
        assert len(rules) == 0

    def test_legacy_recommended(self):
        rules = recommended_guardrails()
        assert len(rules) == 4

    def test_legacy_strict(self):
        rules = strict_guardrails()
        assert len(rules) == 8


# ─────────────────────────────────────────────────────────────────────────────
# Guardrails Namespace
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailsNamespace:
    """Tests for the Guardrails namespace class."""

    def test_rules_json(self):
        rule = Guardrails.json()
        assert rule.name == "json"

    def test_rules_strict_json(self):
        rule = Guardrails.strict_json()
        assert rule.name == "strict_json"

    def test_rules_markdown(self):
        rule = Guardrails.markdown()
        assert rule.name == "markdown"

    def test_rules_latex(self):
        rule = Guardrails.latex()
        assert rule.name == "latex"

    def test_rules_pattern(self):
        rule = Guardrails.pattern()
        assert rule.name == "pattern"

    def test_rules_custom_pattern(self):
        rule = Guardrails.custom_pattern([r"test"], "Test msg", "warning")
        assert rule.name == "custom_pattern"

    def test_rules_zero_output(self):
        rule = Guardrails.zero_output()
        assert rule.name == "zero_output"

    def test_rules_stall(self):
        rule = Guardrails.stall(max_gap=10.0)
        assert rule.name == "stall"

    def test_rules_repetition(self):
        rule = Guardrails.repetition(window=50)
        assert rule.name == "repetition"

    def test_analyze_json(self):
        result = Guardrails.analyze_json('{"key": 1}')
        assert result.is_balanced is True

    def test_analyze_markdown(self):
        result = Guardrails.analyze_markdown("```\ncode\n```")
        assert result.is_balanced is True

    def test_analyze_latex(self):
        result = Guardrails.analyze_latex(r"\begin{doc}\end{doc}")
        assert result.is_balanced is True

    def test_looks_like_json(self):
        assert Guardrails.looks_like_json("{}") is True
        assert Guardrails.looks_like_json("hello") is False

    def test_looks_like_markdown(self):
        assert Guardrails.looks_like_markdown("# Header") is True

    def test_looks_like_latex(self):
        assert Guardrails.looks_like_latex(r"\begin{doc}") is True

    def test_is_zero_output(self):
        assert Guardrails.is_zero_output("") is True
        assert Guardrails.is_zero_output("hello") is False

    def test_is_noise_only(self):
        assert Guardrails.is_noise_only("...") is True
        assert Guardrails.is_noise_only("hello") is False

    def test_find_patterns(self):
        matches = Guardrails.find_patterns(
            "As an AI", Guardrails.BAD_PATTERNS.META_COMMENTARY
        )
        assert len(matches) > 0

    def test_bad_patterns(self):
        assert len(Guardrails.BAD_PATTERNS.META_COMMENTARY) > 0
        assert len(Guardrails.BAD_PATTERNS.HEDGING) > 0

    def test_types_accessible(self):
        assert Guardrails.Rule is not None
        assert Guardrails.Violation is not None
        assert Guardrails.JsonAnalysis is not None

    def test_check(self):
        state = State(content="As an AI", completed=True)
        rules = [Guardrails.pattern()]
        violations = Guardrails.check(state, rules)
        assert len(violations) >= 1

    def test_new_types_accessible(self):
        """Test new types added for TypeScript parity."""
        assert Guardrails.Context is not None
        assert Guardrails.Result is not None
        assert Guardrails.ResultSummary is not None
        assert Guardrails.State is not None
        assert Guardrails.Config is not None
        assert Guardrails.Engine is not None


# ─────────────────────────────────────────────────────────────────────────────
# GuardrailEngine Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailEngine:
    """Tests for the GuardrailEngine class."""

    def test_create_engine(self):
        from l0.guardrails import GuardrailConfig, GuardrailEngine

        engine = GuardrailEngine(GuardrailConfig(rules=[json_rule()]))
        assert engine is not None

    def test_create_guardrail_engine_factory(self):
        from l0.guardrails import create_guardrail_engine

        engine = create_guardrail_engine([json_rule(), markdown_rule()])
        assert engine is not None

    def test_engine_check_passes(self):
        from l0.guardrails import GuardrailContext, create_guardrail_engine

        engine = create_guardrail_engine([json_rule()])
        result = engine.check(GuardrailContext(content='{"key": 1}', completed=True))
        assert result.passed is True
        assert len(result.violations) == 0
        assert result.should_retry is False
        assert result.should_halt is False

    def test_engine_check_fails(self):
        from l0.guardrails import GuardrailContext, create_guardrail_engine

        engine = create_guardrail_engine([json_rule()])
        result = engine.check(GuardrailContext(content='{"key": 1}}', completed=True))
        assert result.passed is False
        assert len(result.violations) >= 1

    def test_engine_result_summary(self):
        from l0.guardrails import GuardrailContext, create_guardrail_engine

        engine = create_guardrail_engine([json_rule()])
        result = engine.check(GuardrailContext(content='{"key": 1}}', completed=True))
        assert result.summary.total >= 1
        assert result.summary.errors >= 1

    def test_engine_add_rule(self):
        from l0.guardrails import create_guardrail_engine

        engine = create_guardrail_engine([json_rule()])
        assert len(engine._rules) == 1
        engine.add_rule(markdown_rule())
        # Check that engine now has both rules
        assert len(engine._rules) == 2

    def test_engine_remove_rule(self):
        from l0.guardrails import create_guardrail_engine

        engine = create_guardrail_engine([json_rule(), markdown_rule()])
        removed = engine.remove_rule("json")
        assert removed is True
        removed_again = engine.remove_rule("json")
        assert removed_again is False

    def test_engine_get_state(self):
        from l0.guardrails import GuardrailContext, create_guardrail_engine

        engine = create_guardrail_engine([json_rule()])
        engine.check(GuardrailContext(content='{"key": 1}}', completed=True))
        state = engine.get_state()
        assert state.violation_count >= 1

    def test_engine_reset(self):
        from l0.guardrails import GuardrailContext, create_guardrail_engine

        engine = create_guardrail_engine([json_rule()])
        engine.check(GuardrailContext(content='{"key": 1}}', completed=True))
        engine.reset()
        state = engine.get_state()
        assert state.violation_count == 0

    def test_engine_has_violations(self):
        from l0.guardrails import GuardrailContext, create_guardrail_engine

        engine = create_guardrail_engine([json_rule()])
        engine.check(GuardrailContext(content='{"key": 1}}', completed=True))
        assert engine.has_violations() is True

    def test_engine_get_violations_by_rule(self):
        from l0.guardrails import GuardrailContext, create_guardrail_engine

        engine = create_guardrail_engine([json_rule()])
        engine.check(GuardrailContext(content='{"key": 1}}', completed=True))
        violations = engine.get_violations_by_rule("json")
        assert len(violations) >= 1

    def test_engine_on_violation_callback(self):
        from l0.guardrails import GuardrailContext, create_guardrail_engine

        violations_received = []

        def on_violation(v: GuardrailViolation) -> None:
            violations_received.append(v)

        engine = create_guardrail_engine([json_rule()], on_violation=on_violation)
        engine.check(GuardrailContext(content='{"key": 1}}', completed=True))
        assert len(violations_received) >= 1

    def test_guardrails_create_engine(self):
        """Test Guardrails.create_engine() method."""
        engine = Guardrails.create_engine([json_rule()])
        assert engine is not None


# ─────────────────────────────────────────────────────────────────────────────
# GuardrailContext Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailContext:
    """Tests for the GuardrailContext dataclass."""

    def test_create_context(self):
        from l0.guardrails import GuardrailContext

        ctx = GuardrailContext(content="test content", completed=True)
        assert ctx.content == "test content"
        assert ctx.completed is True

    def test_context_with_all_fields(self):
        from l0.guardrails import GuardrailContext, GuardrailViolation

        ctx = GuardrailContext(
            content="test content",
            completed=True,
            checkpoint="previous content",
            delta="new content",
            token_count=100,
            metadata={"key": "value"},
            previous_violations=[
                GuardrailViolation(rule="test", message="test", severity="warning")
            ],
        )
        assert ctx.checkpoint == "previous content"
        assert ctx.delta == "new content"
        assert ctx.token_count == 100
        assert ctx.metadata == {"key": "value"}
        assert ctx.previous_violations is not None
        assert len(ctx.previous_violations) == 1


# ─────────────────────────────────────────────────────────────────────────────
# GuardrailResult Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuardrailResult:
    """Tests for the GuardrailResult dataclass."""

    def test_create_result(self):
        from l0.guardrails import (
            GuardrailResult,
            GuardrailResultSummary,
        )

        result = GuardrailResult(
            passed=True,
            violations=[],
            should_retry=False,
            should_halt=False,
            summary=GuardrailResultSummary(total=0, fatal=0, errors=0, warnings=0),
        )
        assert result.passed is True
        assert result.should_retry is False

    def test_result_with_violations(self):
        from l0.guardrails import (
            GuardrailResult,
            GuardrailResultSummary,
            GuardrailViolation,
        )

        result = GuardrailResult(
            passed=False,
            violations=[
                GuardrailViolation(rule="test", message="error", severity="error")
            ],
            should_retry=True,
            should_halt=False,
            summary=GuardrailResultSummary(total=1, fatal=0, errors=1, warnings=0),
        )
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.summary.errors == 1


# ─────────────────────────────────────────────────────────────────────────────
# check_guardrails_full Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckGuardrailsFull:
    """Tests for check_guardrails_full function."""

    def test_check_guardrails_full_passes(self):
        from l0.guardrails import GuardrailContext, check_guardrails_full

        result = check_guardrails_full(
            GuardrailContext(content='{"key": 1}', completed=True), [json_rule()]
        )
        assert result.passed is True
        assert result.should_retry is False
        assert result.should_halt is False

    def test_check_guardrails_full_fails(self):
        from l0.guardrails import GuardrailContext, check_guardrails_full

        result = check_guardrails_full(
            GuardrailContext(content='{"key": 1}}', completed=True), [json_rule()]
        )
        assert result.passed is False
        assert len(result.violations) >= 1

    def test_guardrails_check_full(self):
        """Test Guardrails.check_full() method."""
        from l0.guardrails import GuardrailContext

        result = Guardrails.check_full(
            GuardrailContext(content='{"key": 1}', completed=True), [json_rule()]
        )
        assert result.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Detection Functions Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPatternDetectionFunctions:
    """Tests for standalone pattern detection functions."""

    def test_detect_meta_commentary(self):
        from l0.guardrails import GuardrailContext, detect_meta_commentary

        ctx = GuardrailContext(content="As an AI, I cannot help", completed=True)
        violations = detect_meta_commentary(ctx)
        assert len(violations) >= 1
        assert violations[0].rule == "pattern-meta-commentary"

    def test_detect_meta_commentary_no_match(self):
        from l0.guardrails import GuardrailContext, detect_meta_commentary

        ctx = GuardrailContext(content="Normal content here", completed=True)
        violations = detect_meta_commentary(ctx)
        assert len(violations) == 0

    def test_detect_excessive_hedging(self):
        from l0.guardrails import GuardrailContext, detect_excessive_hedging

        ctx = GuardrailContext(content="Sure! Here is the answer", completed=True)
        violations = detect_excessive_hedging(ctx)
        assert len(violations) >= 1
        assert violations[0].rule == "pattern-hedging"

    def test_detect_refusal(self):
        from l0.guardrails import GuardrailContext, detect_refusal

        ctx = GuardrailContext(
            content="I cannot provide that information", completed=True
        )
        violations = detect_refusal(ctx)
        assert len(violations) >= 1
        assert violations[0].rule == "pattern-refusal"
        assert violations[0].recoverable is False

    def test_detect_instruction_leakage(self):
        from l0.guardrails import GuardrailContext, detect_instruction_leakage

        ctx = GuardrailContext(
            content="[SYSTEM] You are a helpful assistant", completed=True
        )
        violations = detect_instruction_leakage(ctx)
        assert len(violations) >= 1
        assert violations[0].rule == "pattern-instruction-leak"

    def test_detect_placeholders(self):
        from l0.guardrails import GuardrailContext, detect_placeholders

        ctx = GuardrailContext(content="Hello [INSERT NAME HERE]", completed=True)
        violations = detect_placeholders(ctx)
        assert len(violations) >= 1
        assert violations[0].rule == "pattern-placeholders"

    def test_detect_placeholders_incomplete_skipped(self):
        from l0.guardrails import GuardrailContext, detect_placeholders

        ctx = GuardrailContext(content="Hello [INSERT NAME HERE]", completed=False)
        violations = detect_placeholders(ctx)
        assert len(violations) == 0

    def test_detect_format_collapse(self):
        from l0.guardrails import GuardrailContext, detect_format_collapse

        ctx = GuardrailContext(content="Here is the code:", completed=True)
        violations = detect_format_collapse(ctx)
        assert len(violations) >= 1
        assert violations[0].rule == "pattern-format-collapse"

    def test_detect_repetition(self):
        from l0.guardrails import GuardrailContext, detect_repetition

        ctx = GuardrailContext(
            content="This is a test sentence. This is a test sentence. This is a test sentence.",
            completed=True,
        )
        violations = detect_repetition(ctx, threshold=2)
        assert len(violations) >= 1
        assert violations[0].rule == "pattern-repetition"

    def test_detect_repetition_incomplete_skipped(self):
        from l0.guardrails import GuardrailContext, detect_repetition

        ctx = GuardrailContext(
            content="This is a test. This is a test. This is a test.",
            completed=False,
        )
        violations = detect_repetition(ctx)
        assert len(violations) == 0

    def test_detect_first_last_duplicate(self):
        from l0.guardrails import GuardrailContext, detect_first_last_duplicate

        ctx = GuardrailContext(
            content="Hello world and this is the start. Some content in the middle that is long enough. Hello world and this is the start.",
            completed=True,
        )
        violations = detect_first_last_duplicate(ctx)
        assert len(violations) >= 1
        assert violations[0].rule == "pattern-first-last-duplicate"

    def test_detect_first_last_duplicate_short_content_skipped(self):
        from l0.guardrails import GuardrailContext, detect_first_last_duplicate

        ctx = GuardrailContext(content="Short. Short.", completed=True)
        violations = detect_first_last_duplicate(ctx)
        assert len(violations) == 0

    def test_detect_first_last_duplicate_incomplete_skipped(self):
        from l0.guardrails import GuardrailContext, detect_first_last_duplicate

        ctx = GuardrailContext(
            content="Hello world. Some content. Hello world.",
            completed=False,
        )
        violations = detect_first_last_duplicate(ctx)
        assert len(violations) == 0

    def test_guardrails_detect_methods(self):
        """Test Guardrails namespace detection methods."""
        from l0.guardrails import GuardrailContext

        ctx = GuardrailContext(content="As an AI", completed=True)
        violations = Guardrails.detect_meta_commentary(ctx)
        assert len(violations) >= 1

        ctx2 = GuardrailContext(
            content="Test. Content. Test.",
            completed=True,
        )
        violations2 = Guardrails.detect_repetition(ctx2, threshold=1)
        # May or may not find repetition depending on sentence length


# ─────────────────────────────────────────────────────────────────────────────
# Async Guardrail Check Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAsyncGuardrailCheck:
    """Tests for async guardrail check functions."""

    def test_run_async_guardrail_check_fast_path(self):
        """Test that fast path returns result immediately for small content."""
        from l0.guardrails import (
            GuardrailContext,
            create_guardrail_engine,
            run_async_guardrail_check,
        )

        engine = create_guardrail_engine([json_rule()])
        results = []

        def on_complete(result: Any) -> None:
            results.append(result)

        ctx = GuardrailContext(
            content='{"key": 1}',
            delta='{"key": 1}',
            completed=True,
        )
        result = run_async_guardrail_check(engine, ctx, on_complete)

        # For small content with delta, should return immediately
        assert result is not None or len(results) > 0

    def test_run_async_guardrail_check_with_violation(self):
        """Test that violations are detected in fast path."""
        from l0.guardrails import (
            GuardrailContext,
            create_guardrail_engine,
            run_async_guardrail_check,
        )

        engine = create_guardrail_engine([json_rule()])
        results = []

        def on_complete(result: Any) -> None:
            results.append(result)

        ctx = GuardrailContext(
            content='{"key": 1}}',
            delta='{"key": 1}}',
            completed=True,
        )
        result = run_async_guardrail_check(engine, ctx, on_complete)

        # Should find violation
        if result is not None:
            assert result.passed is False or len(result.violations) >= 1

    @pytest.mark.asyncio
    async def test_run_guardrail_check_async(self):
        """Test async version of guardrail check."""
        from l0.guardrails import (
            GuardrailContext,
            create_guardrail_engine,
            run_guardrail_check_async,
        )

        engine = create_guardrail_engine([json_rule()])
        ctx = GuardrailContext(content='{"key": 1}', completed=True)
        result = await run_guardrail_check_async(engine, ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_guardrail_check_async_with_violation(self):
        """Test async version detects violations."""
        from l0.guardrails import (
            GuardrailContext,
            create_guardrail_engine,
            run_guardrail_check_async,
        )

        engine = create_guardrail_engine([json_rule()])
        ctx = GuardrailContext(content='{"key": 1}}', completed=True)
        result = await run_guardrail_check_async(engine, ctx)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_guardrails_run_check_async(self):
        """Test Guardrails.run_check_async() method."""
        from l0.guardrails import GuardrailContext

        engine = Guardrails.create_engine([json_rule()])
        ctx = GuardrailContext(content='{"key": 1}', completed=True)
        result = await Guardrails.run_check_async(engine, ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_guardrail_check_async_fails_safe_on_exception(self):
        """Test that async guardrail check fails safe when engine.check() throws."""
        from unittest.mock import MagicMock

        from l0.guardrails import (
            GuardrailContext,
            run_guardrail_check_async,
        )

        # Create a mock engine that raises an exception on check()
        mock_engine = MagicMock()
        mock_engine.check.side_effect = RuntimeError("Simulated engine failure")

        ctx = GuardrailContext(content="test content", completed=True)

        # Should fail safe (passed=False) instead of raising or passing through
        result = await run_guardrail_check_async(mock_engine, ctx)

        assert result.passed is False
        assert result.should_retry is True
        assert len(result.violations) == 1
        assert result.violations[0].rule == "internal_error"
        assert "Simulated engine failure" in result.violations[0].message

    def test_engine_handles_rule_exception_gracefully(self):
        """Test that engine handles individual rule exceptions as warnings."""
        from l0.guardrails import (
            GuardrailConfig,
            GuardrailContext,
            GuardrailEngine,
            GuardrailRule,
        )

        # Create a rule that raises an exception
        def failing_check(state: State) -> list[GuardrailViolation]:
            raise RuntimeError("Simulated rule failure")

        failing_rule = GuardrailRule(
            name="failing_rule",
            check=failing_check,
            severity="error",
        )

        engine = GuardrailEngine(GuardrailConfig(rules=[failing_rule]))
        ctx = GuardrailContext(content="test content", completed=True)

        # Engine should handle this gracefully and return a result
        result = engine.check(ctx)

        # Rule failure is treated as a warning violation, not a pass
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].rule == "failing_rule"
        assert "Rule execution failed" in result.violations[0].message
        assert result.violations[0].severity == "warning"
