"""Utility functions for L0."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AutoCorrectResult:
    """Result of JSON auto-correction."""

    text: str
    corrected: bool
    corrections: list[str] = field(default_factory=list)


def auto_correct_json(text: str, track_corrections: bool = False) -> AutoCorrectResult:
    """Auto-correct common JSON errors from LLM output.

    Fixes:
    - Markdown fences (```json ... ```)
    - Text prefixes ("Sure! {...}" → "{...}")
    - Trailing commas
    - Missing closing braces/brackets
    - Single quotes → double quotes (in keys/values)

    Args:
        text: Raw text that should contain JSON
        track_corrections: Whether to track what corrections were applied

    Returns:
        AutoCorrectResult with corrected text and metadata
    """
    original = text
    corrections: list[str] = []

    # Extract content from markdown fences FIRST (before prefix removal)
    # Only match fences at start of line to avoid corrupting JSON with ``` in strings
    if "```" in text:
        # Try to find ```json ... ``` block first (fence at start of string or after newline)
        match = re.search(r"(?:^|\n)\s*```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
            if track_corrections:
                corrections.append("Removed markdown fences")
        else:
            # Try to find ``` ... ``` block
            match = re.search(r"(?:^|\n)\s*```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
                if track_corrections:
                    corrections.append("Removed markdown fences")

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
                corrections.append(f"Removed text prefix: {prefix.strip()[:50]}...")

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
                corrections.append(f"Removed text suffix: {suffix[:50]}...")

    # Fix single quotes to double quotes (careful with apostrophes in text)
    # Only replace single quotes that look like JSON delimiters
    single_quote_json = re.search(r"'\s*:", text) or re.search(r":\s*'", text)
    if single_quote_json:
        # Replace single-quoted keys: {'key': -> {"key":
        text = re.sub(r"'(\w+)'\s*:", r'"\1":', text)
        # Replace single-quoted string values: : 'value' -> : "value"
        # Use a greedy match up to a quote followed by , or } or ] or end
        # This handles apostrophes like "Don't" correctly
        text = re.sub(r":\s*'(.*?)'(?=\s*[,}\]]|$)", r': "\1"', text)
        if track_corrections:
            corrections.append("Converted single quotes to double quotes")

    # Remove trailing commas before } or ]
    if re.search(r",\s*[}\]]", text):
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        if track_corrections:
            corrections.append("Removed trailing commas")

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
            corrections.append(f"Added {open_braces} missing closing brace(s)")

    if open_brackets > 0:
        text += "]" * open_brackets
        if track_corrections:
            corrections.append(f"Added {open_brackets} missing closing bracket(s)")

    text = text.strip()
    corrected = text != original.strip()

    return AutoCorrectResult(
        text=text,
        corrected=corrected,
        corrections=corrections if track_corrections else [],
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
