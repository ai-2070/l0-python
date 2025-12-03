"""L0 guardrails engine with built-in rules and drift detection."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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
# Bad Patterns - Categorized
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BadPatterns:
    """Categories of bad patterns to detect in LLM output."""

    # Meta commentary about being an AI
    META_COMMENTARY: list[str] = field(
        default_factory=lambda: [
            r"\bas an ai\b",
            r"\bas an artificial intelligence\b",
            r"\bi'?m an ai\b",
            r"\bi am an ai\b",
            r"\bas a language model\b",
            r"\bas an llm\b",
            r"\bi'?m a language model\b",
            r"\bi am a language model\b",
            r"\bas an ai assistant\b",
            r"\bi'?m an ai assistant\b",
        ]
    )

    # Hedging and filler phrases
    HEDGING: list[str] = field(
        default_factory=lambda: [
            r"^sure[,!]?\s",
            r"^certainly[,!]?\s",
            r"^of course[,!]?\s",
            r"^absolutely[,!]?\s",
            r"^definitely[,!]?\s",
            r"^great question[,!]?\s",
            r"^good question[,!]?\s",
            r"^that'?s a great question\b",
            r"^that'?s a good question\b",
            r"^i'?d be happy to\b",
            r"^i would be happy to\b",
        ]
    )

    # Refusal patterns
    REFUSAL: list[str] = field(
        default_factory=lambda: [
            r"\bi cannot provide\b",
            r"\bi can'?t provide\b",
            r"\bi'?m not able to\b",
            r"\bi am not able to\b",
            r"\bi cannot assist\b",
            r"\bi can'?t assist\b",
            r"\bi'?m unable to\b",
            r"\bi am unable to\b",
            r"\bi cannot help with\b",
            r"\bi can'?t help with\b",
            r"\bi must decline\b",
            r"\bi have to decline\b",
        ]
    )

    # Instruction/prompt leakage
    INSTRUCTION_LEAK: list[str] = field(
        default_factory=lambda: [
            r"\[SYSTEM\]",
            r"\[INST\]",
            r"\[/INST\]",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"<\|system\|>",
            r"<\|user\|>",
            r"<\|assistant\|>",
            r"<<SYS>>",
            r"<</SYS>>",
            r"\[AVAILABLE_TOOLS\]",
            r"\[/AVAILABLE_TOOLS\]",
        ]
    )

    # Placeholder patterns
    PLACEHOLDERS: list[str] = field(
        default_factory=lambda: [
            r"\[INSERT[^\]]*\]",
            r"\[YOUR[^\]]*\]",
            r"\[PLACEHOLDER[^\]]*\]",
            r"\[TODO[^\]]*\]",
            r"\[FILL[^\]]*\]",
            r"\{\{[^}]+\}\}",
            r"<PLACEHOLDER>",
            r"<INSERT>",
            r"<YOUR[^>]*>",
        ]
    )

    # Format collapse patterns
    FORMAT_COLLAPSE: list[str] = field(
        default_factory=lambda: [
            r"^here is the\b",
            r"^here'?s the\b",
            r"^let me\b",
            r"^i will now\b",
            r"^i'?ll now\b",
            r"^below is\b",
            r"^the following is\b",
            r"^please find\b",
        ]
    )

    def all_patterns(self) -> list[str]:
        """Get all patterns from all categories."""
        return (
            self.META_COMMENTARY
            + self.HEDGING
            + self.REFUSAL
            + self.INSTRUCTION_LEAK
            + self.PLACEHOLDERS
            + self.FORMAT_COLLAPSE
        )


# Singleton instance
BAD_PATTERNS = BadPatterns()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class JsonAnalysis:
    """Result of JSON structure analysis."""

    is_balanced: bool
    open_braces: int
    close_braces: int
    open_brackets: int
    close_brackets: int
    in_string: bool
    unclosed_string: bool
    issues: list[str] = field(default_factory=list)


def analyze_json_structure(content: str) -> JsonAnalysis:
    """Analyze JSON structure for balance and issues.

    Args:
        content: The content to analyze.

    Returns:
        JsonAnalysis with detailed structure information.

    Example:
        >>> result = analyze_json_structure('{"a": 1')
        >>> result.is_balanced
        False
        >>> result.open_braces
        1
        >>> result.close_braces
        0
    """
    open_braces = 0
    close_braces = 0
    open_brackets = 0
    close_brackets = 0
    in_string = False
    escape_next = False
    issues: list[str] = []
    last_char = ""
    consecutive_commas = 0

    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            last_char = char
            continue

        if char == "\\" and in_string:
            escape_next = True
            last_char = char
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            last_char = char
            continue

        if in_string:
            last_char = char
            continue

        # Track braces and brackets
        if char == "{":
            open_braces += 1
            if last_char == ",":
                issues.append(f"Malformed pattern ',{{' at position {i}")
        elif char == "}":
            close_braces += 1
        elif char == "[":
            open_brackets += 1
            if last_char == ",":
                issues.append(f"Malformed pattern ',[' at position {i}")
        elif char == "]":
            close_brackets += 1
        elif char == ",":
            consecutive_commas += 1
            if consecutive_commas > 1:
                issues.append(f"Multiple consecutive commas at position {i}")
        else:
            if not char.isspace():
                consecutive_commas = 0

        last_char = char

    # Check for unclosed string
    unclosed_string = in_string

    if unclosed_string:
        issues.append("Unclosed string detected")

    # Check balance
    is_balanced = (
        open_braces == close_braces
        and open_brackets == close_brackets
        and not unclosed_string
    )

    if open_braces > close_braces:
        issues.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
    elif open_braces < close_braces:
        issues.append(
            f"Too many closing braces: {close_braces} close vs {open_braces} open"
        )

    if open_brackets > close_brackets:
        issues.append(
            f"Unbalanced brackets: {open_brackets} open, {close_brackets} close"
        )
    elif open_brackets < close_brackets:
        issues.append(
            f"Too many closing brackets: {close_brackets} close vs {open_brackets} open"
        )

    return JsonAnalysis(
        is_balanced=is_balanced,
        open_braces=open_braces,
        close_braces=close_braces,
        open_brackets=open_brackets,
        close_brackets=close_brackets,
        in_string=in_string,
        unclosed_string=unclosed_string,
        issues=issues,
    )


def looks_like_json(content: str) -> bool:
    """Check if content looks like it's trying to be JSON.

    Args:
        content: The content to check.

    Returns:
        True if content appears to be JSON.
    """
    stripped = content.strip()
    if not stripped:
        return False
    return stripped.startswith(("{", "[")) or stripped.endswith(("}", "]"))


@dataclass
class MarkdownAnalysis:
    """Result of Markdown structure analysis."""

    is_balanced: bool
    in_fence: bool
    open_fences: int
    close_fences: int
    fence_languages: list[str]
    table_rows: int
    inconsistent_columns: bool
    issues: list[str] = field(default_factory=list)


def analyze_markdown_structure(content: str) -> MarkdownAnalysis:
    """Analyze Markdown structure for issues.

    Args:
        content: The content to analyze.

    Returns:
        MarkdownAnalysis with detailed structure information.

    Example:
        >>> result = analyze_markdown_structure("```js\\ncode")
        >>> result.in_fence
        True
        >>> result.open_fences
        1
    """
    lines = content.split("\n")
    open_fences = 0
    close_fences = 0
    in_fence = False
    fence_languages: list[str] = []
    issues: list[str] = []

    # Table analysis
    table_rows = 0
    table_columns: list[int] = []
    in_table = False
    inconsistent_columns = False

    fence_pattern = re.compile(r"^(`{3,}|~{3,})(\w*)")

    for i, line in enumerate(lines):
        # Check for code fences
        fence_match = fence_pattern.match(line.strip())
        if fence_match:
            fence_marker = fence_match.group(1)
            lang = fence_match.group(2)

            if not in_fence:
                in_fence = True
                open_fences += 1
                if lang:
                    fence_languages.append(lang)
            else:
                in_fence = False
                close_fences += 1

        # Skip content inside fences for other checks
        if in_fence:
            continue

        # Check for tables
        if "|" in line and line.strip().startswith("|"):
            cols = line.count("|") - 1  # Subtract 1 for trailing pipe
            if not in_table:
                in_table = True
                table_columns.append(cols)
            else:
                table_rows += 1
                if table_columns and cols != table_columns[-1]:
                    # Allow separator rows with different structure
                    if not re.match(r"^\|[\s:-]+\|", line.strip()):
                        inconsistent_columns = True
                        issues.append(f"Inconsistent table columns at line {i + 1}")
        else:
            if in_table:
                in_table = False
                table_columns = []

    # Check for unclosed fences
    if in_fence:
        issues.append("Unclosed code fence")

    is_balanced = open_fences == close_fences

    return MarkdownAnalysis(
        is_balanced=is_balanced,
        in_fence=in_fence,
        open_fences=open_fences,
        close_fences=close_fences,
        fence_languages=fence_languages,
        table_rows=table_rows,
        inconsistent_columns=inconsistent_columns,
        issues=issues,
    )


def looks_like_markdown(content: str) -> bool:
    """Check if content looks like Markdown.

    Args:
        content: The content to check.

    Returns:
        True if content appears to be Markdown.
    """
    patterns = [
        r"^#{1,6}\s",  # Headers
        r"```",  # Code fences
        r"^\s*[-*+]\s",  # Lists
        r"^\s*\d+\.\s",  # Numbered lists
        r"\[.*\]\(.*\)",  # Links
        r"^\|.*\|",  # Tables
        r"\*\*.*\*\*",  # Bold
        r"__.*__",  # Bold
        r"\*[^*]+\*",  # Italic
        r"_[^_]+_",  # Italic
    ]
    for pattern in patterns:
        if re.search(pattern, content, re.MULTILINE):
            return True
    return False


@dataclass
class LatexAnalysis:
    """Result of LaTeX structure analysis."""

    is_balanced: bool
    open_environments: list[str]
    display_math_balanced: bool
    inline_math_balanced: bool
    bracket_math_balanced: bool
    issues: list[str] = field(default_factory=list)


def analyze_latex_structure(content: str) -> LatexAnalysis:
    """Analyze LaTeX structure for balance and issues.

    Args:
        content: The content to analyze.

    Returns:
        LatexAnalysis with detailed structure information.

    Example:
        >>> result = analyze_latex_structure("\\\\begin{equation}")
        >>> result.open_environments
        ['equation']
        >>> result.is_balanced
        False
    """
    issues: list[str] = []
    open_environments: list[str] = []

    # Find all \begin{env} and \end{env}
    begin_pattern = re.compile(r"\\begin\{(\w+)\}")
    end_pattern = re.compile(r"\\end\{(\w+)\}")

    begins = [(m.group(1), m.start()) for m in begin_pattern.finditer(content)]
    ends = [(m.group(1), m.start()) for m in end_pattern.finditer(content)]

    # Track environment stack
    env_stack: list[str] = []
    all_events = sorted(
        [(pos, "begin", env) for env, pos in begins]
        + [(pos, "end", env) for env, pos in ends],
        key=lambda x: x[0],
    )

    for pos, event_type, env in all_events:
        if event_type == "begin":
            env_stack.append(env)
        else:  # end
            if not env_stack:
                issues.append(f"Unexpected \\end{{{env}}} without matching \\begin")
            elif env_stack[-1] != env:
                issues.append(
                    f"Mismatched environment: expected \\end{{{env_stack[-1]}}}, got \\end{{{env}}}"
                )
                env_stack.pop()
            else:
                env_stack.pop()

    open_environments = env_stack.copy()
    if open_environments:
        issues.append(f"Unclosed environments: {', '.join(open_environments)}")

    # Check display math $$...$$
    dollar_count = len(re.findall(r"(?<!\$)\$\$(?!\$)", content))
    display_math_balanced = dollar_count % 2 == 0
    if not display_math_balanced:
        issues.append("Unbalanced display math ($$)")

    # Check bracket math \[...\]
    open_bracket = len(re.findall(r"\\\[", content))
    close_bracket = len(re.findall(r"\\\]", content))
    bracket_math_balanced = open_bracket == close_bracket
    if not bracket_math_balanced:
        issues.append(
            f"Unbalanced bracket math: {open_bracket} \\[ vs {close_bracket} \\]"
        )

    # Check inline math $...$ (excluding $$)
    # This is tricky because $ is used for both open and close
    # Count single $ not preceded or followed by another $
    singles = re.findall(r"(?<!\$)\$(?!\$)", content)
    inline_math_balanced = len(singles) % 2 == 0
    if not inline_math_balanced:
        issues.append("Unbalanced inline math ($)")

    is_balanced = (
        len(open_environments) == 0
        and display_math_balanced
        and inline_math_balanced
        and bracket_math_balanced
    )

    return LatexAnalysis(
        is_balanced=is_balanced,
        open_environments=open_environments,
        display_math_balanced=display_math_balanced,
        inline_math_balanced=inline_math_balanced,
        bracket_math_balanced=bracket_math_balanced,
        issues=issues,
    )


def looks_like_latex(content: str) -> bool:
    """Check if content looks like LaTeX.

    Args:
        content: The content to check.

    Returns:
        True if content appears to be LaTeX.
    """
    patterns = [
        r"\\begin\{",
        r"\\end\{",
        r"\$\$",
        r"\\\[",
        r"\\\]",
        r"\\frac\{",
        r"\\sum",
        r"\\int",
        r"\\alpha",
        r"\\beta",
        r"\\gamma",
        r"\\documentclass",
        r"\\usepackage",
    ]
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    return False


def is_zero_output(content: str) -> bool:
    """Check if content is effectively empty.

    Args:
        content: The content to check.

    Returns:
        True if content is empty, whitespace-only, or meaningless.
    """
    if not content:
        return True
    if not content.strip():
        return True
    return False


def is_noise_only(content: str) -> bool:
    """Check if content is just noise (punctuation, repeated chars).

    Args:
        content: The content to check.

    Returns:
        True if content appears to be noise.
    """
    stripped = content.strip()
    if not stripped:
        return True

    # Check if only punctuation and whitespace
    if re.match(r"^[\s\.,!?;:\-_=+*#@&%$^(){}[\]<>/\\|`~\"\']+$", stripped):
        return True

    # Check for repeated single character
    if len(set(stripped.replace(" ", ""))) <= 2 and len(stripped) > 10:
        return True

    # Check for repeated short pattern
    if len(stripped) > 20:
        chunk = stripped[:10]
        if stripped.count(chunk) > len(stripped) / 15:
            return True

    return False


def find_bad_patterns(
    content: str,
    patterns: list[str],
) -> list[tuple[str, re.Match[str]]]:
    """Find all matches of bad patterns in content.

    Args:
        content: The content to search.
        patterns: List of regex patterns to search for.

    Returns:
        List of (pattern, match) tuples for all matches found.
    """
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            matches.append((pattern, match))
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Rules
# ─────────────────────────────────────────────────────────────────────────────


def json_rule() -> GuardrailRule:
    """Check for balanced JSON structure during streaming.

    Detects:
    - Unbalanced {} and []
    - Unclosed strings
    - Multiple consecutive commas
    - Malformed patterns like {, or [,
    """

    def check(state: State) -> list[GuardrailViolation]:
        content = state.content
        if not content.strip():
            return []

        # Only check if it looks like JSON
        if not looks_like_json(content):
            return []

        analysis = analyze_json_structure(content)
        violations = []

        # During streaming, only report critical issues
        if not state.completed:
            # Too many closes is always bad
            if analysis.close_braces > analysis.open_braces:
                violations.append(
                    GuardrailViolation(
                        rule="json",
                        message="Too many closing braces",
                        severity="error",
                        suggestion="Check JSON structure",
                    )
                )
            if analysis.close_brackets > analysis.open_brackets:
                violations.append(
                    GuardrailViolation(
                        rule="json",
                        message="Too many closing brackets",
                        severity="error",
                        suggestion="Check JSON structure",
                    )
                )
            # Report malformed patterns immediately
            for issue in analysis.issues:
                if "Malformed pattern" in issue or "consecutive commas" in issue:
                    violations.append(
                        GuardrailViolation(
                            rule="json",
                            message=issue,
                            severity="error",
                        )
                    )
        else:
            # On completion, check for both extra closes AND missing closes
            if analysis.close_braces > analysis.open_braces:
                violations.append(
                    GuardrailViolation(
                        rule="json",
                        message=f"Too many closing braces: {analysis.close_braces} close vs {analysis.open_braces} open",
                        severity="error",
                    )
                )
            elif analysis.open_braces > analysis.close_braces:
                violations.append(
                    GuardrailViolation(
                        rule="json",
                        message=f"Missing closing braces: {analysis.open_braces} open vs {analysis.close_braces} close",
                        severity="error",
                    )
                )
            if analysis.close_brackets > analysis.open_brackets:
                violations.append(
                    GuardrailViolation(
                        rule="json",
                        message=f"Too many closing brackets: {analysis.close_brackets} close vs {analysis.open_brackets} open",
                        severity="error",
                    )
                )
            elif analysis.open_brackets > analysis.close_brackets:
                violations.append(
                    GuardrailViolation(
                        rule="json",
                        message=f"Missing closing brackets: {analysis.open_brackets} open vs {analysis.close_brackets} close",
                        severity="error",
                    )
                )
            # Report other issues (unclosed strings, etc.)
            for issue in analysis.issues:
                if "Unbalanced" not in issue and "closing" not in issue.lower():
                    violations.append(
                        GuardrailViolation(
                            rule="json",
                            message=issue,
                            severity="error",
                        )
                    )

        return violations

    return GuardrailRule(
        name="json",
        check=check,
        description="Validates JSON structure during streaming",
    )


def strict_json_rule() -> GuardrailRule:
    """Validate complete JSON on completion.

    Requires:
    - Valid parseable JSON
    - Root must be object or array
    """

    def check(state: State) -> list[GuardrailViolation]:
        if not state.completed:
            return []

        content = state.content.strip()
        if not content:
            return []

        # Only check if it looks like JSON
        if not looks_like_json(content):
            return []

        try:
            parsed = json.loads(content)
            # Check root is object or array
            if not isinstance(parsed, (dict, list)):
                return [
                    GuardrailViolation(
                        rule="strict_json",
                        message=f"JSON root must be object or array, got {type(parsed).__name__}",
                        severity="error",
                    )
                ]
            return []
        except json.JSONDecodeError as e:
            return [
                GuardrailViolation(
                    rule="strict_json",
                    message=f"Invalid JSON: {e}",
                    severity="error",
                    position=e.pos,
                )
            ]

    return GuardrailRule(
        name="strict_json",
        check=check,
        streaming=False,
        description="Validates complete JSON is parseable",
    )


def markdown_rule() -> GuardrailRule:
    """Validate Markdown structure.

    Detects:
    - Unclosed code fences (```)
    - Inconsistent table columns
    """

    def check(state: State) -> list[GuardrailViolation]:
        content = state.content
        if not content.strip():
            return []

        analysis = analyze_markdown_structure(content)
        violations = []

        # During streaming, only warn about unclosed fences
        if not state.completed:
            # This is expected during streaming, don't report
            pass
        else:
            # On completion, report issues
            for issue in analysis.issues:
                severity: Severity = "warning"
                if "Unclosed" in issue:
                    severity = "error"
                violations.append(
                    GuardrailViolation(
                        rule="markdown",
                        message=issue,
                        severity=severity,
                    )
                )

        return violations

    return GuardrailRule(
        name="markdown",
        check=check,
        description="Validates Markdown structure",
    )


def latex_rule() -> GuardrailRule:
    """Validate LaTeX environments and math.

    Detects:
    - Unclosed \\begin{env} environments
    - Mismatched environment names
    - Unbalanced \\[...\\] and $$...$$
    - Unbalanced inline math $...$
    """

    def check(state: State) -> list[GuardrailViolation]:
        content = state.content
        if not content.strip():
            return []

        # Only check if it looks like LaTeX
        if not looks_like_latex(content):
            return []

        analysis = analyze_latex_structure(content)
        violations = []

        # During streaming, only report mismatches (not unclosed)
        if not state.completed:
            for issue in analysis.issues:
                if "Mismatched" in issue or "Unexpected" in issue:
                    violations.append(
                        GuardrailViolation(
                            rule="latex",
                            message=issue,
                            severity="error",
                        )
                    )
        else:
            # On completion, report all issues
            for issue in analysis.issues:
                violations.append(
                    GuardrailViolation(
                        rule="latex",
                        message=issue,
                        severity="error",
                    )
                )

        return violations

    return GuardrailRule(
        name="latex",
        check=check,
        description="Validates LaTeX environments and math",
    )


def pattern_rule(
    patterns: list[str] | None = None,
    *,
    include_categories: list[str] | None = None,
    exclude_categories: list[str] | None = None,
) -> GuardrailRule:
    """Detect unwanted patterns in output.

    Args:
        patterns: Custom patterns to use. If None, uses BAD_PATTERNS based on categories.
        include_categories: Categories to include (META_COMMENTARY, HEDGING, REFUSAL,
                          INSTRUCTION_LEAK, PLACEHOLDERS, FORMAT_COLLAPSE). If None, uses all.
        exclude_categories: Categories to exclude.

    Built-in pattern categories:
    - META_COMMENTARY: "As an AI...", "I'm an AI assistant"
    - HEDGING: "Sure!", "Certainly!", "Of course!"
    - REFUSAL: "I cannot provide...", "I'm not able to..."
    - INSTRUCTION_LEAK: [SYSTEM], <|im_start|>
    - PLACEHOLDERS: [INSERT ...], {{placeholder}}
    - FORMAT_COLLAPSE: "Here is the...", "Let me..."
    """
    if patterns is None:
        # Build patterns from categories
        all_categories = {
            "META_COMMENTARY": BAD_PATTERNS.META_COMMENTARY,
            "HEDGING": BAD_PATTERNS.HEDGING,
            "REFUSAL": BAD_PATTERNS.REFUSAL,
            "INSTRUCTION_LEAK": BAD_PATTERNS.INSTRUCTION_LEAK,
            "PLACEHOLDERS": BAD_PATTERNS.PLACEHOLDERS,
            "FORMAT_COLLAPSE": BAD_PATTERNS.FORMAT_COLLAPSE,
        }

        if include_categories:
            categories = {
                k: v for k, v in all_categories.items() if k in include_categories
            }
        else:
            categories = all_categories

        if exclude_categories:
            categories = {
                k: v for k, v in categories.items() if k not in exclude_categories
            }

        patterns = []
        for cat_patterns in categories.values():
            patterns.extend(cat_patterns)

    def check(state: State) -> list[GuardrailViolation]:
        violations = []
        matches = find_bad_patterns(state.content, patterns)
        for pattern, match in matches:
            violations.append(
                GuardrailViolation(
                    rule="pattern",
                    message=f"Matched unwanted pattern: {match.group()}",
                    severity="warning",
                    position=match.start(),
                    context={"pattern": pattern, "matched": match.group()},
                )
            )
        return violations

    return GuardrailRule(
        name="pattern",
        check=check,
        severity="warning",
        description="Detects unwanted patterns in output",
    )


def custom_pattern_rule(
    patterns: list[str],
    message: str = "Custom pattern violation",
    severity: Severity = "error",
) -> GuardrailRule:
    """Create a custom pattern rule.

    Args:
        patterns: List of regex patterns to detect.
        message: Message to show when pattern is matched.
        severity: Severity level for violations.

    Example:
        >>> rule = custom_pattern_rule([r"forbidden", r"blocked"], "Custom violation", "error")
    """

    def check(state: State) -> list[GuardrailViolation]:
        violations = []
        matches = find_bad_patterns(state.content, patterns)
        for pattern, match in matches:
            violations.append(
                GuardrailViolation(
                    rule="custom_pattern",
                    message=f"{message}: {match.group()}",
                    severity=severity,
                    position=match.start(),
                    context={"pattern": pattern, "matched": match.group()},
                )
            )
        return violations

    return GuardrailRule(
        name="custom_pattern",
        check=check,
        severity=severity,
        description=f"Custom pattern rule: {message}",
    )


def zero_output_rule(
    *,
    min_completion_time: float | None = 0.5,
) -> GuardrailRule:
    """Detect empty or meaningless output.

    Detects:
    - Empty output
    - Whitespace-only output
    - Punctuation-only output
    - Repeated character noise
    - Suspiciously instant completion (if min_completion_time set)

    Args:
        min_completion_time: Minimum expected completion time in seconds.
                           If completion is faster, it's suspicious. Set to None to disable.
    """

    def check(state: State) -> list[GuardrailViolation]:
        if not state.completed:
            return []

        violations = []

        # Check for zero/empty output
        if is_zero_output(state.content):
            violations.append(
                GuardrailViolation(
                    rule="zero_output",
                    message="Empty or whitespace-only output",
                    severity="error",
                )
            )
            return violations

        # Check for noise-only output
        if is_noise_only(state.content):
            violations.append(
                GuardrailViolation(
                    rule="zero_output",
                    message="Output appears to be noise (punctuation or repeated characters)",
                    severity="error",
                )
            )

        # Check for suspiciously fast completion
        # Use duration if available, otherwise skip this check
        if min_completion_time and state.duration is not None:
            if state.duration < min_completion_time and len(state.content.strip()) < 20:
                violations.append(
                    GuardrailViolation(
                        rule="zero_output",
                        message=f"Suspiciously instant completion ({state.duration:.2f}s)",
                        severity="warning",
                        context={
                            "duration": state.duration,
                            "content_length": len(state.content),
                        },
                    )
                )

        return violations

    return GuardrailRule(
        name="zero_output",
        check=check,
        streaming=False,
        description="Detects empty or meaningless output",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Drift Detection
# ─────────────────────────────────────────────────────────────────────────────


def stall_rule(max_gap: float = 5.0) -> GuardrailRule:
    """Detect token stalls (no tokens for too long).

    Args:
        max_gap: Maximum seconds between tokens before triggering.
    """

    def check(state: State) -> list[GuardrailViolation]:
        if state.last_token_at is None:
            return []
        gap = time.time() - state.last_token_at
        if gap > max_gap:
            state.drift_detected = True
            return [
                GuardrailViolation(
                    rule="stall",
                    message=f"Token stall: {gap:.1f}s since last token",
                    severity="warning",
                    context={"gap_seconds": gap},
                )
            ]
        return []

    return GuardrailRule(
        name="stall",
        check=check,
        severity="warning",
        description="Detects token stalls",
    )


def repetition_rule(
    window: int = 100,
    threshold: float = 0.5,
    *,
    sentence_check: bool = True,
    sentence_repeat_count: int = 3,
) -> GuardrailRule:
    """Detect repetitive output (model looping).

    Args:
        window: Character window size for similarity check.
        threshold: Similarity threshold (0-1) to trigger.
        sentence_check: Also check for repeated sentences.
        sentence_repeat_count: Number of sentence repeats to trigger.
    """

    def check(state: State) -> list[GuardrailViolation]:
        content = state.content
        violations = []

        # Character-based similarity check
        if len(content) >= window * 2:
            recent = content[-window:]
            previous = content[-window * 2 : -window]

            matches = sum(1 for a, b in zip(recent, previous, strict=False) if a == b)
            similarity = matches / window

            if similarity > threshold:
                state.drift_detected = True
                violations.append(
                    GuardrailViolation(
                        rule="repetition",
                        message=f"Repetitive output detected ({similarity:.0%} character similarity)",
                        severity="error",
                        context={"similarity": similarity, "window": window},
                    )
                )

        # Sentence repetition check
        if sentence_check and len(content) > 50:
            # Split into sentences
            sentences = re.split(r"[.!?]+\s+", content)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) >= sentence_repeat_count:
                # Count sentence occurrences
                from collections import Counter

                counts = Counter(sentences)
                for sentence, count in counts.items():
                    if count >= sentence_repeat_count and len(sentence) > 20:
                        state.drift_detected = True
                        violations.append(
                            GuardrailViolation(
                                rule="repetition",
                                message=f"Sentence repeated {count} times",
                                severity="error",
                                context={
                                    "sentence": sentence[:50] + "...",
                                    "count": count,
                                },
                            )
                        )
                        break  # Only report once

        return violations

    return GuardrailRule(
        name="repetition",
        check=check,
        description="Detects repetitive output",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Presets (legacy functions - use Guardrails class instead)
# ─────────────────────────────────────────────────────────────────────────────


def recommended_guardrails() -> list[GuardrailRule]:
    """Recommended set of guardrails."""
    return [json_rule(), markdown_rule(), pattern_rule(), zero_output_rule()]


def strict_guardrails() -> list[GuardrailRule]:
    """Strict guardrails including drift detection."""
    return [
        json_rule(),
        strict_json_rule(),
        markdown_rule(),
        latex_rule(),
        pattern_rule(),
        zero_output_rule(),
        stall_rule(),
        repetition_rule(),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Guardrails class - Clean API
# ─────────────────────────────────────────────────────────────────────────────


class Guardrails:
    """Guardrails namespace for presets, rules, and analysis.

    Usage:
        # Presets
        guardrails = l0.Guardrails.recommended()
        guardrails = l0.Guardrails.strict()

        # Individual rules
        rules = [l0.Guardrails.json(), l0.Guardrails.pattern()]

        # Analysis
        result = l0.Guardrails.analyze_json('{"key": "value"}')

        # Patterns
        patterns = l0.Guardrails.BAD_PATTERNS.META_COMMENTARY
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Patterns
    # ─────────────────────────────────────────────────────────────────────────

    BAD_PATTERNS = BAD_PATTERNS

    # ─────────────────────────────────────────────────────────────────────────
    # Types (for type hints)
    # ─────────────────────────────────────────────────────────────────────────

    Rule = GuardrailRule
    Violation = GuardrailViolation
    JsonAnalysis = JsonAnalysis
    MarkdownAnalysis = MarkdownAnalysis
    LatexAnalysis = LatexAnalysis

    # ─────────────────────────────────────────────────────────────────────────
    # Presets
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def minimal() -> list[GuardrailRule]:
        """Minimal guardrails - JSON + zero output only.

        Includes:
        - json: Check balanced JSON brackets
        - zero_output: Detect empty output
        """
        return [json_rule(), zero_output_rule()]

    @staticmethod
    def recommended() -> list[GuardrailRule]:
        """Recommended guardrails for most use cases.

        Includes:
        - json: Check balanced JSON brackets
        - markdown: Check Markdown structure
        - pattern: Detect AI slop patterns
        - zero_output: Detect empty output
        """
        return [json_rule(), markdown_rule(), pattern_rule(), zero_output_rule()]

    @staticmethod
    def strict() -> list[GuardrailRule]:
        """Strict guardrails including drift detection.

        Includes everything in recommended(), plus:
        - strict_json: Validate complete JSON
        - latex: Validate LaTeX structure
        - stall: Detect token stalls
        - repetition: Detect model looping
        """
        return [
            json_rule(),
            strict_json_rule(),
            markdown_rule(),
            latex_rule(),
            pattern_rule(),
            zero_output_rule(),
            stall_rule(),
            repetition_rule(),
        ]

    @staticmethod
    def json_only() -> list[GuardrailRule]:
        """JSON validation + zero output."""
        return [json_rule(), strict_json_rule(), zero_output_rule()]

    @staticmethod
    def markdown_only() -> list[GuardrailRule]:
        """Markdown validation + zero output."""
        return [markdown_rule(), zero_output_rule()]

    @staticmethod
    def latex_only() -> list[GuardrailRule]:
        """LaTeX validation + zero output."""
        return [latex_rule(), zero_output_rule()]

    @staticmethod
    def none() -> list[GuardrailRule]:
        """No guardrails (explicit opt-out)."""
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Rules
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def json() -> GuardrailRule:
        """Check for balanced JSON structure during streaming."""
        return json_rule()

    @staticmethod
    def strict_json() -> GuardrailRule:
        """Validate complete JSON on completion."""
        return strict_json_rule()

    @staticmethod
    def markdown() -> GuardrailRule:
        """Validate Markdown structure."""
        return markdown_rule()

    @staticmethod
    def latex() -> GuardrailRule:
        """Validate LaTeX environments and math."""
        return latex_rule()

    @staticmethod
    def pattern(
        patterns: list[str] | None = None,
        *,
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
    ) -> GuardrailRule:
        """Detect unwanted patterns in output.

        Args:
            patterns: Custom patterns. If None, uses BAD_PATTERNS.
            include_categories: Categories to include.
            exclude_categories: Categories to exclude.
        """
        return pattern_rule(
            patterns,
            include_categories=include_categories,
            exclude_categories=exclude_categories,
        )

    @staticmethod
    def custom_pattern(
        patterns: list[str],
        message: str = "Custom pattern violation",
        severity: Severity = "error",
    ) -> GuardrailRule:
        """Create a custom pattern rule.

        Args:
            patterns: List of regex patterns to detect.
            message: Message to show when pattern is matched.
            severity: Severity level for violations.
        """
        return custom_pattern_rule(patterns, message, severity)

    @staticmethod
    def zero_output(*, min_completion_time: float | None = 0.5) -> GuardrailRule:
        """Detect empty or meaningless output.

        Args:
            min_completion_time: Minimum expected completion time.
        """
        return zero_output_rule(min_completion_time=min_completion_time)

    @staticmethod
    def stall(max_gap: float = 5.0) -> GuardrailRule:
        """Detect token stalls.

        Args:
            max_gap: Maximum seconds between tokens.
        """
        return stall_rule(max_gap)

    @staticmethod
    def repetition(
        window: int = 100,
        threshold: float = 0.5,
        *,
        sentence_check: bool = True,
        sentence_repeat_count: int = 3,
    ) -> GuardrailRule:
        """Detect repetitive output.

        Args:
            window: Character window size.
            threshold: Similarity threshold (0-1).
            sentence_check: Check for repeated sentences.
            sentence_repeat_count: Number of repeats to trigger.
        """
        return repetition_rule(
            window,
            threshold,
            sentence_check=sentence_check,
            sentence_repeat_count=sentence_repeat_count,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis Functions
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def analyze_json(content: str) -> JsonAnalysis:
        """Analyze JSON structure for balance and issues."""
        return analyze_json_structure(content)

    @staticmethod
    def analyze_markdown(content: str) -> MarkdownAnalysis:
        """Analyze Markdown structure for issues."""
        return analyze_markdown_structure(content)

    @staticmethod
    def analyze_latex(content: str) -> LatexAnalysis:
        """Analyze LaTeX structure for balance and issues."""
        return analyze_latex_structure(content)

    @staticmethod
    def looks_like_json(content: str) -> bool:
        """Check if content looks like JSON."""
        return looks_like_json(content)

    @staticmethod
    def looks_like_markdown(content: str) -> bool:
        """Check if content looks like Markdown."""
        return looks_like_markdown(content)

    @staticmethod
    def looks_like_latex(content: str) -> bool:
        """Check if content looks like LaTeX."""
        return looks_like_latex(content)

    @staticmethod
    def is_zero_output(content: str) -> bool:
        """Check if content is effectively empty."""
        return is_zero_output(content)

    @staticmethod
    def is_noise_only(content: str) -> bool:
        """Check if content is just noise."""
        return is_noise_only(content)

    @staticmethod
    def find_patterns(
        content: str, patterns: list[str]
    ) -> list[tuple[str, re.Match[str]]]:
        """Find all matches of patterns in content."""
        return find_bad_patterns(content, patterns)

    # ─────────────────────────────────────────────────────────────────────────
    # Check Function
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def check(state: State, rules: list[GuardrailRule]) -> list[GuardrailViolation]:
        """Run all guardrail rules against current state."""
        return check_guardrails(state, rules)
