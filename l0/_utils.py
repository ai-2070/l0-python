from __future__ import annotations

import re


def auto_correct_json(text: str) -> str:
    """Auto-correct common JSON errors."""
    # Remove markdown fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # Remove trailing commas
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # Balance braces
    opens = text.count("{") - text.count("}")
    if opens > 0:
        text += "}" * opens

    brackets = text.count("[") - text.count("]")
    if brackets > 0:
        text += "]" * brackets

    return text.strip()


def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code fences."""
    # Try to find ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find ``` ... ``` block
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()
