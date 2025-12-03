"""L0 guardrails engine with built-in rules and drift detection."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from .logging import logger

if TYPE_CHECKING:
    from .types import State

Severity = Literal["warning", "error", "fatal"]


@dataclass
class GuardrailViolation:
    """Guardrail violation details."""

    rule: str  # Name of the rule that was violated
    message: str  # Human-readable message
    severity: Severity  # Severity of the violation
    recoverable: bool = True  # Whether this violation is recoverable via retry
    position: int | None = None  # Position in content where violation occurred
    timestamp: float | None = None  # Timestamp when violation was detected
    context: dict[str, Any] | None = None  # Additional context about the violation
    suggestion: str | None = None  # Suggested fix or action


@dataclass
class GuardrailRule:
    """Guardrail rule definition."""

    name: str  # Unique name of the rule
    check: Callable[[State], list[GuardrailViolation]]  # Check function
    description: str | None = None  # Description of what the rule checks
    streaming: bool = True  # Whether to run on every token or only at completion
    severity: Severity = "error"  # Default severity for violations from this rule
    recoverable: bool = True  # Whether violations are recoverable via retry


def check_guardrails(
    state: State, rules: list[GuardrailRule]
) -> list[GuardrailViolation]:
    """Run all guardrail rules against current state."""
    violations = []
    for rule in rules:
        result = rule.check(state)
        if result:
            logger.debug(f"Guardrail '{rule.name}' triggered: {len(result)} violations")
        violations.extend(result)
    return violations


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Rules
# ─────────────────────────────────────────────────────────────────────────────


def json_rule() -> GuardrailRule:
    """Check for balanced JSON braces."""

    def check(state: State) -> list[GuardrailViolation]:
        content = state.content
        opens = content.count("{") + content.count("[")
        closes = content.count("}") + content.count("]")
        if opens < closes:
            return [
                GuardrailViolation(
                    rule="json",
                    message="Unbalanced JSON brackets",
                    severity="error",
                )
            ]
        return []

    return GuardrailRule(name="json", check=check)


def strict_json_rule() -> GuardrailRule:
    """Validate complete JSON on completion."""

    def check(state: State) -> list[GuardrailViolation]:
        if not state.completed:
            return []
        try:
            json.loads(state.content)
            return []
        except json.JSONDecodeError as e:
            return [
                GuardrailViolation(
                    rule="strict_json",
                    message=f"Invalid JSON: {e}",
                    severity="error",
                )
            ]

    return GuardrailRule(name="strict_json", check=check, streaming=False)


def pattern_rule(patterns: list[str] | None = None) -> GuardrailRule:
    """Detect unwanted patterns (e.g., AI slop)."""
    default_patterns = [
        r"\bas an ai\b",
        r"\bi cannot\b",
        r"\bi don'?t have\b",
        r"\bunfortunately\b",
        r"\bi apologize\b",
    ]
    patterns = patterns or default_patterns

    def check(state: State) -> list[GuardrailViolation]:
        violations = []
        for pattern in patterns:
            if re.search(pattern, state.content, re.IGNORECASE):
                violations.append(
                    GuardrailViolation(
                        rule="pattern",
                        message=f"Matched unwanted pattern: {pattern}",
                        severity="warning",
                    )
                )
        return violations

    return GuardrailRule(name="pattern", check=check, severity="warning")


def zero_output_rule() -> GuardrailRule:
    """Detect empty or whitespace-only output."""

    def check(state: State) -> list[GuardrailViolation]:
        if state.completed and not state.content.strip():
            return [
                GuardrailViolation(
                    rule="zero_output",
                    message="Empty output",
                    severity="error",
                )
            ]
        return []

    return GuardrailRule(name="zero_output", check=check, streaming=False)


# ─────────────────────────────────────────────────────────────────────────────
# Drift Detection
# ─────────────────────────────────────────────────────────────────────────────


def stall_rule(max_gap: float = 5.0) -> GuardrailRule:
    """Detect token stalls (no tokens for too long)."""

    def check(state: State) -> list[GuardrailViolation]:
        if state.last_token_at is None:
            return []
        gap = time.time() - state.last_token_at
        if gap > max_gap:
            state.drift_detected = True
            return [
                GuardrailViolation(
                    rule="stall",
                    message=f"Token stall: {gap:.1f}s",
                    severity="warning",
                )
            ]
        return []

    return GuardrailRule(name="stall", check=check, severity="warning")


def repetition_rule(window: int = 100, threshold: float = 0.5) -> GuardrailRule:
    """Detect repetitive output (model looping)."""

    def check(state: State) -> list[GuardrailViolation]:
        content = state.content
        if len(content) < window * 2:
            return []

        recent = content[-window:]
        previous = content[-window * 2 : -window]

        # Simple similarity: count matching characters
        matches = sum(1 for a, b in zip(recent, previous) if a == b)
        similarity = matches / window

        if similarity > threshold:
            state.drift_detected = True
            return [
                GuardrailViolation(
                    rule="repetition",
                    message=f"Repetitive output detected ({similarity:.0%} similar)",
                    severity="error",
                )
            ]
        return []

    return GuardrailRule(name="repetition", check=check)


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────


def recommended_guardrails() -> list[GuardrailRule]:
    """Recommended set of guardrails."""
    return [json_rule(), pattern_rule(), zero_output_rule()]


def strict_guardrails() -> list[GuardrailRule]:
    """Strict guardrails including drift detection."""
    return [
        json_rule(),
        strict_json_rule(),
        pattern_rule(),
        zero_output_rule(),
        stall_rule(),
        repetition_rule(),
    ]
