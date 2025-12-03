"""L0 state management."""

from __future__ import annotations

import time

from .types import State


def create_state() -> State:
    """Create fresh state."""
    return State()


def update_checkpoint(state: State) -> None:
    """Save current content as checkpoint."""
    state.checkpoint = state.content


def append_token(state: State, token: str) -> None:
    """Append token to content and update timing."""
    now = time.time()
    if state.first_token_at is None:
        state.first_token_at = now
    state.last_token_at = now
    state.content += token
    state.token_count += 1


def mark_completed(state: State) -> None:
    """Mark stream as completed and calculate duration."""
    state.completed = True
    if state.first_token_at is not None:
        state.duration = (state.last_token_at or time.time()) - state.first_token_at
