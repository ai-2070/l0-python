"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed


def has_openai() -> bool:
    """Check if OpenAI is available (module installed and API key set)."""
    try:
        import openai  # noqa: F401

        return bool(os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        return False


# Marker for integration tests requiring OpenAI
requires_openai = pytest.mark.skipif(
    not has_openai(),
    reason="OpenAI not available (module not installed or OPENAI_API_KEY not set)",
)
