"""Tests for StateMachine."""

import pytest

from l0 import RuntimeState, StateMachine, StateTransition, create_state_machine


class TestRuntimeState:
    """Tests for RuntimeState enum."""

    def test_all_states_exist(self) -> None:
        assert RuntimeState.INIT.value == "init"
        assert RuntimeState.WAITING_FOR_TOKEN.value == "waiting_for_token"
        assert RuntimeState.STREAMING.value == "streaming"
        assert RuntimeState.TOOL_CALL_DETECTED.value == "tool_call_detected"
        assert RuntimeState.CONTINUATION_MATCHING.value == "continuation_matching"
        assert RuntimeState.CHECKPOINT_VERIFYING.value == "checkpoint_verifying"
        assert RuntimeState.RETRYING.value == "retrying"
        assert RuntimeState.FALLBACK.value == "fallback"
        assert RuntimeState.FINALIZING.value == "finalizing"
        assert RuntimeState.COMPLETE.value == "complete"
        assert RuntimeState.ERROR.value == "error"


class TestStateMachine:
    """Tests for StateMachine class."""

    def test_initial_state(self) -> None:
        sm = StateMachine()
        assert sm.get() == RuntimeState.INIT

    def test_transition(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)
        assert sm.get() == RuntimeState.STREAMING

    def test_transition_same_state_no_op(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)
        sm.transition(RuntimeState.STREAMING)  # Same state
        assert len(sm.get_history()) == 1  # Only one transition recorded

    def test_is_single_state(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)
        assert sm.is_(RuntimeState.STREAMING) is True
        assert sm.is_(RuntimeState.INIT) is False

    def test_is_multiple_states(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)
        assert sm.is_(RuntimeState.STREAMING, RuntimeState.RETRYING) is True
        assert sm.is_(RuntimeState.INIT, RuntimeState.COMPLETE) is False

    def test_is_state_alias(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)
        assert sm.is_state(RuntimeState.STREAMING) is True

    def test_is_terminal_false(self) -> None:
        sm = StateMachine()
        assert sm.is_terminal() is False
        sm.transition(RuntimeState.STREAMING)
        assert sm.is_terminal() is False

    def test_is_terminal_complete(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.COMPLETE)
        assert sm.is_terminal() is True

    def test_is_terminal_error(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.ERROR)
        assert sm.is_terminal() is True

    def test_reset(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)
        sm.transition(RuntimeState.COMPLETE)
        sm.reset()
        assert sm.get() == RuntimeState.INIT
        assert len(sm.get_history()) == 0

    def test_reset_from_init_no_notify(self) -> None:
        sm = StateMachine()
        notifications: list[RuntimeState] = []
        sm.subscribe(lambda s: notifications.append(s))
        sm.reset()  # Already at INIT, should not notify
        assert len(notifications) == 0

    def test_get_history(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)
        sm.transition(RuntimeState.COMPLETE)

        history = sm.get_history()
        assert len(history) == 2

        assert history[0].from_state == RuntimeState.INIT
        assert history[0].to_state == RuntimeState.STREAMING

        assert history[1].from_state == RuntimeState.STREAMING
        assert history[1].to_state == RuntimeState.COMPLETE

    def test_get_history_returns_copy(self) -> None:
        sm = StateMachine()
        sm.transition(RuntimeState.STREAMING)

        history1 = sm.get_history()
        history2 = sm.get_history()

        assert history1 is not history2
        assert history1 == history2

    def test_subscribe(self) -> None:
        sm = StateMachine()
        notifications: list[RuntimeState] = []

        sm.subscribe(lambda s: notifications.append(s))

        sm.transition(RuntimeState.STREAMING)
        sm.transition(RuntimeState.COMPLETE)

        assert notifications == [RuntimeState.STREAMING, RuntimeState.COMPLETE]

    def test_unsubscribe(self) -> None:
        sm = StateMachine()
        notifications: list[RuntimeState] = []

        unsubscribe = sm.subscribe(lambda s: notifications.append(s))
        sm.transition(RuntimeState.STREAMING)

        unsubscribe()
        sm.transition(RuntimeState.COMPLETE)

        assert notifications == [RuntimeState.STREAMING]

    def test_subscribe_error_ignored(self) -> None:
        sm = StateMachine()

        def bad_listener(s: RuntimeState) -> None:
            raise ValueError("Listener error")

        sm.subscribe(bad_listener)

        # Should not raise
        sm.transition(RuntimeState.STREAMING)
        assert sm.get() == RuntimeState.STREAMING

    def test_multiple_subscribers(self) -> None:
        sm = StateMachine()
        notifications1: list[RuntimeState] = []
        notifications2: list[RuntimeState] = []

        sm.subscribe(lambda s: notifications1.append(s))
        sm.subscribe(lambda s: notifications2.append(s))

        sm.transition(RuntimeState.STREAMING)

        assert notifications1 == [RuntimeState.STREAMING]
        assert notifications2 == [RuntimeState.STREAMING]

    def test_listener_unsubscribes_during_notification(self) -> None:
        """Test that unsubscribing during notification does not raise RuntimeError."""
        sm = StateMachine()
        notifications: list[RuntimeState] = []
        unsubscribe_fn: list = []

        def self_unsubscribing_listener(s: RuntimeState) -> None:
            notifications.append(s)
            if unsubscribe_fn:
                unsubscribe_fn[0]()  # Unsubscribe during notification

        unsubscribe = sm.subscribe(self_unsubscribing_listener)
        unsubscribe_fn.append(unsubscribe)

        # This should not raise RuntimeError: Set changed size during iteration
        sm.transition(RuntimeState.STREAMING)

        assert notifications == [RuntimeState.STREAMING]

    def test_listener_subscribes_during_notification(self) -> None:
        """Test that subscribing during notification does not raise RuntimeError."""
        sm = StateMachine()
        notifications: list[tuple[str, RuntimeState]] = []

        def subscribing_listener(s: RuntimeState) -> None:
            notifications.append(("first", s))
            # Subscribe a new listener during notification
            sm.subscribe(lambda state: notifications.append(("second", state)))

        sm.subscribe(subscribing_listener)

        # This should not raise RuntimeError: Set changed size during iteration
        sm.transition(RuntimeState.STREAMING)

        # First listener was notified
        assert ("first", RuntimeState.STREAMING) in notifications
        # Second listener was added but not notified for this transition
        # (because we iterate over a copy of the set)

    def test_create_class_method(self) -> None:
        sm = StateMachine.create()
        assert isinstance(sm, StateMachine)
        assert sm.get() == RuntimeState.INIT

    def test_create_state_machine_function(self) -> None:
        sm = create_state_machine()
        assert isinstance(sm, StateMachine)
        assert sm.get() == RuntimeState.INIT


class TestStateTransition:
    """Tests for StateTransition dataclass."""

    def test_fields(self) -> None:
        t = StateTransition(
            from_state=RuntimeState.INIT,
            to_state=RuntimeState.STREAMING,
            timestamp=1234567890.0,
        )
        assert t.from_state == RuntimeState.INIT
        assert t.to_state == RuntimeState.STREAMING
        assert t.timestamp == 1234567890.0
