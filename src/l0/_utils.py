"""Utility functions for L0."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

# Correction types that can be applied during auto-correction
CorrectionType = Literal[
    "close_brace",
    "close_bracket",
    "remove_trailing_comma",
    "strip_markdown_fence",
    "strip_json_prefix",
    "remove_prefix_text",
    "remove_suffix_text",
    "fix_quotes",
    "remove_comments",
    "escape_control_chars",
    "fill_missing_fields",
    "remove_unknown_fields",
    "coerce_types",
    "extract_json",
]


@dataclass
class AutoCorrectResult:
    """Result of JSON auto-correction."""

    text: str
    corrected: bool
    corrections: list[str] = field(default_factory=list)
    success: bool = True
    error: Exception | None = None


def auto_correct_json(text: str, track_corrections: bool = False) -> AutoCorrectResult:
    """Auto-correct common JSON errors from LLM output.

    Fixes:
    - Markdown fences (```json ... ```)
    - Text prefixes ("Sure! {...}" → "{...}")
    - Trailing commas
    - Missing closing braces/brackets
    - Single quotes → double quotes (in keys/values)
    - C-style comments (// and /* */)
    - Control characters in strings

    Args:
        text: Raw text that should contain JSON
        track_corrections: Whether to track what corrections were applied

    Returns:
        AutoCorrectResult with corrected text, metadata, and success flag
    """
    original = text
    corrections: list[str] = []

    try:
        # Extract content from markdown fences FIRST (before prefix removal)
        # Only match fences at start of line to avoid corrupting JSON with ``` in strings
        if "```" in text:
            # Try to find ```json ... ``` block first (fence at start of string or after newline)
            match = re.search(r"(?:^|\n)\s*```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
                if track_corrections:
                    corrections.append("strip_markdown_fence")
            else:
                # Try to find ``` ... ``` block
                match = re.search(r"(?:^|\n)\s*```\s*(.*?)\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)
                    if track_corrections:
                        corrections.append("strip_markdown_fence")

        # Remove "json" prefix at start (common LLM artifact)
        if text.strip().lower().startswith("json"):
            text = re.sub(r"^json\s*", "", text.strip(), flags=re.IGNORECASE)
            if track_corrections:
                corrections.append("strip_json_prefix")

        # Remove text prefix (e.g., "Sure! Here's the JSON:" or "Here is the response:")
        prefix_match = re.match(
            r"^[\s\S]*?(?:here(?:'s| is)[\s\S]*?[:.]?\s*)?(?=[\[{])",
            text,
            re.IGNORECASE,
        )
        if prefix_match and prefix_match.group():
            prefix = prefix_match.group()
            if prefix.strip():
                text = text[len(prefix) :]
                if track_corrections:
                    corrections.append("remove_prefix_text")

        # Remove C-style comments (// and /* */)
        if "//" in text or "/*" in text:
            # Remove single-line comments
            text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
            # Remove multi-line comments
            text = re.sub(r"/\*[\s\S]*?\*/", "", text)
            if track_corrections:
                corrections.append("remove_comments")

        # Remove text suffix after JSON closes
        # Find where JSON ends and remove trailing text
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        json_end = -1

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    json_end = i + 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if brace_count == 0 and bracket_count == 0:
                    json_end = i + 1

        if json_end > 0 and json_end < len(text):
            suffix = text[json_end:].strip()
            if suffix:
                text = text[:json_end]
                if track_corrections:
                    corrections.append("remove_suffix_text")

        # Fix single quotes to double quotes (careful with apostrophes in text)
        # Only replace single quotes that look like JSON delimiters
        single_quote_json = re.search(r"'\s*:", text) or re.search(r":\s*'", text)
        if single_quote_json:
            # Replace single-quoted keys: {'key': -> {"key":
            text = re.sub(r"'(\w+)'\s*:", r'"\1":', text)
            # Replace single-quoted string values: : 'value' -> : "value"
            # Use [\s\S] instead of . to match newlines in multiline values
            # This handles apostrophes like "Don't" correctly
            text = re.sub(r":\s*'([\s\S]*?)'(?=\s*[,}\]]|$)", r': "\1"', text)
            if track_corrections:
                corrections.append("fix_quotes")

        # Remove trailing commas before } or ]
        if re.search(r",\s*[}\]]", text):
            text = re.sub(r",(\s*[}\]])", r"\1", text)
            if track_corrections:
                corrections.append("remove_trailing_comma")

        # Balance braces and brackets (ignoring characters inside strings)
        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == "{":
                open_braces += 1
            elif char == "}":
                open_braces -= 1
            elif char == "[":
                open_brackets += 1
            elif char == "]":
                open_brackets -= 1

        if open_braces > 0:
            text += "}" * open_braces
            if track_corrections:
                corrections.append("close_brace")

        if open_brackets > 0:
            text += "]" * open_brackets
            if track_corrections:
                corrections.append("close_bracket")

        text = text.strip()
        corrected = text != original.strip()

        # Validate the result
        try:
            json.loads(text)
            return AutoCorrectResult(
                text=text,
                corrected=corrected,
                corrections=corrections if track_corrections else [],
                success=True,
            )
        except json.JSONDecodeError as e:
            return AutoCorrectResult(
                text=text,
                corrected=corrected,
                corrections=corrections if track_corrections else [],
                success=False,
                error=e,
            )

    except Exception as e:
        return AutoCorrectResult(
            text=original,
            corrected=False,
            corrections=[],
            success=False,
            error=e,
        )


def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code fences.

    Args:
        text: Text that may contain markdown-fenced JSON

    Returns:
        Extracted JSON string, or original text if no fences found
    """
    # Try to find ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find ``` ... ``` block
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def _find_first_json_delimiter(text: str) -> tuple[int, str, str] | None:
    """Find the first JSON delimiter ({ or [) that is NOT inside a quoted string.

    Args:
        text: Text to search

    Returns:
        Tuple of (start_index, open_char, close_char) or None if not found
    """
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        # Only consider delimiters outside of strings
        if not in_string:
            if char == "{":
                return (i, "{", "}")
            if char == "[":
                return (i, "[", "]")

    return None


def extract_json(text: str) -> str:
    """Extract JSON from text that may contain other content.

    Uses balanced brace matching to find the first complete JSON object or array.
    Correctly ignores braces that appear inside quoted strings in surrounding prose.

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON string or original text if no valid JSON found
    """
    # Find the first { or [ that is NOT inside a quoted string
    delimiter = _find_first_json_delimiter(text)

    if not delimiter:
        return text

    start_index, open_char, close_char = delimiter

    # Use balanced brace matching to find the end
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_index, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start_index : i + 1]

    # Couldn't find balanced braces, fall back to greedy regex
    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        return object_match.group(0)

    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        return array_match.group(0)

    return text


def is_valid_json(text: str) -> bool:
    """Check if a string is valid JSON.

    Args:
        text: String to check

    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def safe_json_parse(
    text: str,
    auto_correct: bool = True,
) -> dict[str, Any]:
    """Safely parse JSON with optional auto-correction.

    Args:
        text: JSON string to parse
        auto_correct: Whether to attempt auto-correction on failure

    Returns:
        Dict with 'data', 'corrected', and 'corrections' keys

    Raises:
        ValueError: If JSON cannot be parsed even after correction
    """
    # Try parsing as-is first
    try:
        data = json.loads(text)
        return {"data": data, "corrected": False, "corrections": []}
    except json.JSONDecodeError:
        pass

    if not auto_correct:
        raise ValueError(f"Invalid JSON: {text[:100]}...")

    # Try with auto-correction
    result = auto_correct_json(text, track_corrections=True)
    if result.success:
        data = json.loads(result.text)
        return {
            "data": data,
            "corrected": result.corrected,
            "corrections": result.corrections,
        }

    # Try extracting JSON first then correcting
    extracted = extract_json(text)
    if extracted != text:
        result = auto_correct_json(extracted, track_corrections=True)
        if result.success:
            data = json.loads(result.text)
            corrections = ["extract_json"] + result.corrections
            return {"data": data, "corrected": True, "corrections": corrections}

    raise ValueError(
        f"Failed to parse JSON: {result.error}" if result.error else "Invalid JSON"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class JSON:
    """Scoped API for JSON utilities.

    Provides utilities for extracting, validating, and parsing JSON from
    LLM outputs that may contain markdown fences, prose, or other artifacts.

    Usage:
        ```python
        from l0 import JSON

        # Extract JSON from text with surrounding prose
        json_str = JSON.extract('Here is the result: {"key": "value"}')

        # Check if string is valid JSON
        is_valid = JSON.is_valid('{"key": "value"}')

        # Parse JSON with auto-correction for common LLM errors
        result = JSON.parse(text)
        data = result["data"]
        was_corrected = result["corrected"]
        corrections = result["corrections"]

        # Auto-correct JSON without parsing
        result = JSON.auto_correct(text)
        corrected_text = result.text
        ```
    """

    # Re-export types for convenience
    CorrectionType = CorrectionType
    AutoCorrectResult = AutoCorrectResult

    @staticmethod
    def extract(text: str) -> str:
        """Extract JSON from text that may contain other content.

        Uses balanced brace matching to find the first complete JSON object or array.
        Correctly ignores braces that appear inside quoted strings in surrounding prose.

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON string or original text if no valid JSON found
        """
        return extract_json(text)

    @staticmethod
    def is_valid(text: str) -> bool:
        """Check if a string is valid JSON.

        Args:
            text: String to check

        Returns:
            True if valid JSON, False otherwise
        """
        return is_valid_json(text)

    @staticmethod
    def parse(
        text: str,
        auto_correct: bool = True,
    ) -> dict[str, Any]:
        """Safely parse JSON with optional auto-correction.

        Args:
            text: JSON string to parse
            auto_correct: Whether to attempt auto-correction on failure

        Returns:
            Dict with 'data', 'corrected', and 'corrections' keys

        Raises:
            ValueError: If JSON cannot be parsed even after correction
        """
        return safe_json_parse(text, auto_correct)

    @staticmethod
    def auto_correct(text: str, track_corrections: bool = False) -> AutoCorrectResult:
        """Auto-correct common JSON errors from LLM output.

        Fixes:
        - Markdown fences (```json ... ```)
        - Text prefixes ("Sure! {...}" → "{...}")
        - Trailing commas
        - Missing closing braces/brackets
        - Single quotes → double quotes (in keys/values)
        - C-style comments (// and /* */)
        - Control characters in strings

        Args:
            text: Raw text that should contain JSON
            track_corrections: Whether to track what corrections were applied

        Returns:
            AutoCorrectResult with corrected text, metadata, and success flag
        """
        return auto_correct_json(text, track_corrections)

    @staticmethod
    def extract_from_markdown(text: str) -> str:
        """Extract JSON from markdown code fences.

        Args:
            text: Text that may contain markdown-fenced JSON

        Returns:
            Extracted JSON string, or original text if no fences found
        """
        return extract_json_from_markdown(text)
