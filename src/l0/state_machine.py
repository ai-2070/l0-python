"""Lightweight state machine for L0 runtime.

Provides a simple state machine for tracking runtime state with
subscription support for state change notifications.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class RuntimeState(str, Enum):
    """Runtime state constants.

    Use these instead of string literals to prevent typos
    and get better editor autocomplete.
    """

    INIT = "init"
    WAITING_FOR_TOKEN = "waiting_for_token"
    STREAMING = "streaming"
    TOOL_CALL_DETECTED = "tool_call_detected"
    CONTINUATION_MATCHING = "continuation_matching"
    CHECKPOINT_VERIFYING = "checkpoint_verifying"
    RETRYING = "retrying"
    FALLBACK = "fallback"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    ERROR = "error"


# Convenience alias matching TypeScript's RuntimeStates object
RuntimeStates = RuntimeState


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: RuntimeState
    to_state: RuntimeState
    timestamp: float


class StateMachine:
    """Simple state machine for tracking runtime state.

    No transition validation - just a state holder with helpers.

    Usage:
        from l0 import StateMachine, RuntimeState

        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)
        print(sm.get())  # RuntimeState.STREAMING

        # Check state
        if sm.is_(RuntimeState.STREAMING, RuntimeState.CONTINUATION_MATCHING):
            print("Processing...")

        # Subscribe to changes
        unsubscribe = sm.subscribe(lambda state: print(f"State: {state}"))

        # Check if terminal
        if sm.is_terminal():
            print("Done!")

        # Get history
        for t in sm.get_history():
            print(f"{t.from_state} -> {t.to_state}")

        # Reset
        sm.reset()
    """

    __slots__ = ("_state", "_history", "_listeners")

    def __init__(self) -> None:
        self._state: RuntimeState = RuntimeState.INIT
        self._history: list[StateTransition] = []
        self._listeners: set[Callable[[RuntimeState], None]] = set()

    def transition(self, next_state: RuntimeState) -> None:
        """Transition to a new state.

        Args:
            next_state: The state to transition to.
        """
        if self._state != next_state:
            self._history.append(
                StateTransition(
                    from_state=self._state,
                    to_state=next_state,
                    timestamp=time.time(),
                )
            )
            self._state = next_state
            self._notify()

    def get(self) -> RuntimeState:
        """Get current state.

        Returns:
            The current runtime state.
        """
        return self._state

    def is_(self, *states: RuntimeState) -> bool:
        """Check if current state matches any of the provided states.

        Args:
            *states: States to check against.

        Returns:
            True if current state matches any of the provided states.
        """
        return self._state in states

    # Alias without underscore for more natural usage
    def is_state(self, *states: RuntimeState) -> bool:
        """Check if current state matches any of the provided states.

        Alias for is_() with a more explicit name.

        Args:
            *states: States to check against.

        Returns:
            True if current state matches any of the provided states.
        """
        return self._state in states

    def is_terminal(self) -> bool:
        """Check if state is terminal (complete or error).

        Returns:
            True if state is COMPLETE or ERROR.
        """
        return self._state in (RuntimeState.COMPLETE, RuntimeState.ERROR)

    def reset(self) -> None:
        """Reset to initial state and clear history."""
        previous_state = self._state
        self._state = RuntimeState.INIT
        self._history = []
        # Notify subscribers if state changed
        if previous_state != RuntimeState.INIT:
            self._notify()

    def get_history(self) -> list[StateTransition]:
        """Get state transition history.

        Returns:
            List of state transitions (copy to prevent mutation).
        """
        return list(self._history)

    def subscribe(self, listener: Callable[[RuntimeState], None]) -> Callable[[], None]:
        """Subscribe to state changes.

        Args:
            listener: Callback function that receives the new state.

        Returns:
            Unsubscribe function - call it to remove the listener.
        """
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    def _notify(self) -> None:
        """Notify all listeners of state change."""
        for listener in list(self._listeners):
            try:
                listener(self._state)
            except Exception:
                # Ignore listener errors
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Scoped API (class methods)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def create(cls) -> "StateMachine":
        """Create a new state machine instance.

        Returns:
            A new StateMachine instance.
        """
        return cls()


# Legacy standalone function (for backwards compatibility)
def create_state_machine() -> StateMachine:
    """Create a new state machine instance.

    Deprecated: Use StateMachine.create() instead.
    """
    return StateMachine.create()
