"""Pydantic models for L0 guardrails types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["warning", "error", "fatal"]


class GuardrailViolationModel(BaseModel):
    """Pydantic model for guardrail violation."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    message: str
    severity: Severity
    recoverable: bool = True
    position: int | None = None
    timestamp: float | None = None
    context: dict[str, Any] | None = None
    suggestion: str | None = None


class GuardrailContextModel(BaseModel):
    """Pydantic model for guardrail context."""

    model_config = ConfigDict(extra="forbid")

    content: str
    completed: bool = False
    checkpoint: str | None = None
    delta: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] | None = None
    previous_violations: list[GuardrailViolationModel] | None = None


class GuardrailResultSummaryModel(BaseModel):
    """Pydantic model for guardrail result summary."""

    model_config = ConfigDict(extra="forbid")

    total: int
    fatal: int
    errors: int
    warnings: int


class GuardrailResultModel(BaseModel):
    """Pydantic model for guardrail result."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    violations: list[GuardrailViolationModel]
    should_retry: bool
    should_halt: bool
    summary: GuardrailResultSummaryModel


class GuardrailStateModel(BaseModel):
    """Pydantic model for guardrail engine state."""

    model_config = ConfigDict(extra="forbid")

    violations: list[GuardrailViolationModel] = Field(default_factory=list)
    violations_by_rule: dict[str, list[GuardrailViolationModel]] = Field(
        default_factory=dict
    )
    has_fatal_violations: bool = False
    has_error_violations: bool = False
    violation_count: int = 0
    last_check_time: float | None = None


class GuardrailConfigModel(BaseModel):
    """Pydantic model for guardrail configuration."""

    model_config = ConfigDict(extra="forbid")

    rules: list[Any] = Field(default_factory=list)  # GuardrailRule (has callable)
    stop_on_fatal: bool = True
    enable_streaming: bool = True
    check_interval: int = 100


class JsonAnalysisModel(BaseModel):
    """Pydantic model for JSON structure analysis."""

    model_config = ConfigDict(extra="forbid")

    is_balanced: bool
    open_braces: int
    close_braces: int
    open_brackets: int
    close_brackets: int
    in_string: bool
    unclosed_string: bool
    issues: list[str] = Field(default_factory=list)


class MarkdownAnalysisModel(BaseModel):
    """Pydantic model for Markdown structure analysis."""

    model_config = ConfigDict(extra="forbid")

    is_balanced: bool
    in_fence: bool
    open_fences: int
    close_fences: int
    fence_languages: list[str] = Field(default_factory=list)
    table_rows: int
    inconsistent_columns: bool
    issues: list[str] = Field(default_factory=list)


class LatexAnalysisModel(BaseModel):
    """Pydantic model for LaTeX structure analysis."""

    model_config = ConfigDict(extra="forbid")

    is_balanced: bool
    open_environments: list[str] = Field(default_factory=list)
    display_math_balanced: bool
    inline_math_balanced: bool
    bracket_math_balanced: bool
    issues: list[str] = Field(default_factory=list)
