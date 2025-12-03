"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest

# Load .env file if it exists
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed


def has_openai() -> bool:
    """Check if OpenAI is available (module installed and API key set)."""
    try:
        import openai  # noqa: F401  # type: ignore[import-not-found]

        return bool(os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        return False


def has_litellm() -> bool:
    """Check if LiteLLM is available (module installed and API key set)."""
    try:
        import litellm  # noqa: F401  # type: ignore[import-not-found]

        # LiteLLM can use various API keys - check for common ones
        return bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("COHERE_API_KEY")
        )
    except ImportError:
        return False


# Marker for integration tests requiring OpenAI
requires_openai = pytest.mark.skipif(
    not has_openai(),
    reason="OpenAI not available (module not installed or OPENAI_API_KEY not set)",
)

# Marker for integration tests requiring LiteLLM
requires_litellm = pytest.mark.skipif(
    not has_litellm(),
    reason="LiteLLM not available (module not installed or no API key set)",
)
