"""Pytest configuration for l0 tests."""

import pytest


@pytest.fixture
def sample_state():
    """Create a sample State for testing."""
    from l0.types import State

    return State(
        content="Hello world",
        token_count=2,
    )
