"""Pytest configuration for l0 tests."""

import pytest


@pytest.fixture
def sample_state():
    """Create a sample L0State for testing."""
    from l0.types import L0State

    return L0State(
        content="Hello world",
        token_count=2,
    )
