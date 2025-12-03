"""Tests for l0.guardrails module."""

import pytest

from l0.guardrails import (
    GuardrailRule,
    GuardrailViolation,
    check_guardrails,
    json_rule,
    pattern_rule,
    recommended_guardrails,
    repetition_rule,
    stall_rule,
    strict_guardrails,
    strict_json_rule,
    zero_output_rule,
)
from l0.types import L0State


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


class TestJsonRule:
    def test_balanced_json_passes(self):
        state = L0State(content='{"key": "value"}')
        rule = json_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_unbalanced_json_fails(self):
        state = L0State(content='{"key": "value"}}')
        rule = json_rule()
        violations = rule.check(state)
        assert len(violations) == 1
        assert violations[0].rule == "json"


class TestStrictJsonRule:
    def test_valid_json_passes(self):
        state = L0State(content='{"key": "value"}', completed=True)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_invalid_json_fails(self):
        state = L0State(content='{"key": value}', completed=True)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 1
        assert "Invalid JSON" in violations[0].message

    def test_incomplete_stream_skipped(self):
        """Should not check incomplete streams."""
        state = L0State(content='{"key":', completed=False)
        rule = strict_json_rule()
        violations = rule.check(state)
        assert len(violations) == 0


class TestPatternRule:
    def test_default_patterns(self):
        rule = pattern_rule()

        state = L0State(content="As an AI, I cannot do that")
        violations = rule.check(state)
        assert len(violations) >= 1

    def test_custom_patterns(self):
        rule = pattern_rule(patterns=[r"\bfoo\b"])

        state = L0State(content="This has foo in it")
        violations = rule.check(state)
        assert len(violations) == 1

    def test_no_match_passes(self):
        rule = pattern_rule()
        state = L0State(content="This is a normal response")
        violations = rule.check(state)
        assert len(violations) == 0


class TestZeroOutputRule:
    def test_empty_completed_fails(self):
        state = L0State(content="", completed=True)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 1

    def test_whitespace_only_fails(self):
        state = L0State(content="   \n\t  ", completed=True)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 1

    def test_content_passes(self):
        state = L0State(content="Hello", completed=True)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 0

    def test_incomplete_stream_skipped(self):
        state = L0State(content="", completed=False)
        rule = zero_output_rule()
        violations = rule.check(state)
        assert len(violations) == 0


class TestStallRule:
    def test_no_stall(self):
        import time

        state = L0State(last_token_at=time.time())
        rule = stall_rule(max_gap=5.0)
        violations = rule.check(state)
        assert len(violations) == 0

    def test_stall_detected(self):
        import time

        state = L0State(last_token_at=time.time() - 10)
        rule = stall_rule(max_gap=5.0)
        violations = rule.check(state)
        assert len(violations) == 1
        assert state.drift_detected is True


class TestRepetitionRule:
    def test_no_repetition(self):
        state = L0State(content="a" * 100 + "b" * 100)
        rule = repetition_rule(window=100, threshold=0.5)
        violations = rule.check(state)
        assert len(violations) == 0

    def test_repetition_detected(self):
        # Create content where the last 100 chars match the previous 100 chars
        repeated_block = "x" * 100
        state = L0State(content=repeated_block + repeated_block)
        rule = repetition_rule(window=100, threshold=0.5)
        violations = rule.check(state)
        assert len(violations) == 1
        assert state.drift_detected is True

    def test_short_content_skipped(self):
        state = L0State(content="short")
        rule = repetition_rule(window=100)
        violations = rule.check(state)
        assert len(violations) == 0


class TestCheckGuardrails:
    def test_runs_all_rules(self):
        state = L0State(content="As an AI, I cannot help", completed=True)
        rules = [pattern_rule(), zero_output_rule()]
        violations = check_guardrails(state, rules)
        assert len(violations) >= 1


class TestPresets:
    def test_recommended_guardrails(self):
        rules = recommended_guardrails()
        assert len(rules) == 3

    def test_strict_guardrails(self):
        rules = strict_guardrails()
        assert len(rules) == 6
