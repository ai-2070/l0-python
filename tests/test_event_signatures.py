"""Tests for observability event signature validation.

These tests validate that emitted events have the correct field names
and types as defined in the canonical specification.

The canonical spec uses camelCase for all event fields to ensure
cross-language consistency between TypeScript and Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from l0.events import EventBus, ObservabilityEvent, ObservabilityEventType
from l0.runtime import _internal_run
from l0.types import Event, EventType


def load_spec() -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / "canonical-spec.json"
    with open(fixture_path) as f:
        return json.load(f)


SPEC = load_spec()
SPEC_EVENTS = SPEC["monitoring"]["observabilityEvents"]["events"]


class EventCapture:
    """Captures emitted observability events for testing."""

    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def __call__(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def get_by_type(
        self, event_type: ObservabilityEventType
    ) -> list[ObservabilityEvent]:
        return [e for e in self.events if e.type == event_type]

    def get_first(
        self, event_type: ObservabilityEventType
    ) -> ObservabilityEvent | None:
        events = self.get_by_type(event_type)
        return events[0] if events else None


def validate_event_fields(
    event: ObservabilityEvent,
    expected_fields: list[dict[str, Any]],
    event_name: str,
) -> list[str]:
    """Validate event meta fields against spec.

    Returns list of validation errors.
    """
    errors = []
    meta = event.meta

    # Check all required fields are present
    for field_spec in expected_fields:
        field_name = field_spec["name"]
        is_required = field_spec.get("required", True)

        if is_required and field_name not in meta:
            errors.append(f"{event_name}: Missing required field '{field_name}'")

    # Check field names use camelCase (no snake_case)
    for key in meta.keys():
        if "_" in key:
            # Find the expected camelCase name
            camel_case = "".join(
                word.capitalize() if i > 0 else word
                for i, word in enumerate(key.split("_"))
            )
            errors.append(
                f"{event_name}: Field '{key}' uses snake_case, "
                f"should be camelCase '{camel_case}'"
            )

    # Check for extra fields not in spec
    expected_names = {f["name"] for f in expected_fields}
    for key in meta.keys():
        # Convert to camelCase for comparison
        camel_key = "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(key.split("_"))
        )
        if camel_key not in expected_names and key not in expected_names:
            errors.append(f"{event_name}: Unexpected field '{key}' not in spec")

    return errors


# ============================================================================
# Session Event Signature Tests
# ============================================================================


class TestSessionStartSignature:
    """Tests for SESSION_START event signature."""

    @pytest.fixture
    def expected_fields(self) -> list[dict[str, Any]]:
        return SPEC_EVENTS["SESSION_START"]["fields"]

    @pytest.mark.asyncio
    async def test_session_start_has_correct_fields(
        self, expected_fields: list[dict[str, Any]]
    ) -> None:
        """SESSION_START should emit attempt, isRetry, isFallback."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.SESSION_START)
        assert event is not None, "SESSION_START event not emitted"

        errors = validate_event_fields(event, expected_fields, "SESSION_START")
        assert not errors, "\n".join(errors)

    @pytest.mark.asyncio
    async def test_session_start_field_types(self) -> None:
        """SESSION_START fields should have correct types."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.SESSION_START)
        assert event is not None

        meta = event.meta
        # Check field types
        assert "attempt" in meta, "Missing attempt field"
        if "attempt" in meta:
            assert isinstance(meta["attempt"], int), "attempt should be int"
        if "isRetry" in meta:
            assert isinstance(meta["isRetry"], bool), "isRetry should be bool"
        if "isFallback" in meta:
            assert isinstance(meta["isFallback"], bool), "isFallback should be bool"


# ============================================================================
# Timeout Event Signature Tests
# ============================================================================


class TestTimeoutEventSignatures:
    """Tests for TIMEOUT_* event signatures."""

    @pytest.mark.asyncio
    async def test_timeout_start_uses_camel_case(self) -> None:
        """TIMEOUT_START should use camelCase field names."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        from l0.types import Timeout

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
            timeout=Timeout(initial_token=30000, inter_token=10000),
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.TIMEOUT_START)
        if event:
            meta = event.meta
            # Should NOT have snake_case fields
            assert "timeout_type" not in meta, (
                "TIMEOUT_START uses snake_case 'timeout_type', should be 'timeoutType'"
            )
            assert "duration_seconds" not in meta, (
                "TIMEOUT_START uses 'duration_seconds', should be 'configuredMs'"
            )
            # Should have camelCase fields
            if "timeoutType" in meta:
                assert meta["timeoutType"] in ("initial", "inter"), (
                    f"timeoutType should be 'initial' or 'inter', got '{meta['timeoutType']}'"
                )

    @pytest.mark.asyncio
    async def test_timeout_reset_has_correct_fields(self) -> None:
        """TIMEOUT_RESET should have configuredMs field."""
        capture = EventCapture()

        async def multi_token_stream():
            yield Event(type=EventType.TOKEN, text="Hello ")
            yield Event(type=EventType.TOKEN, text="world")
            yield Event(type=EventType.COMPLETE)

        from l0.types import Timeout

        result = await _internal_run(
            stream=multi_token_stream,
            on_event=capture,
            timeout=Timeout(initial_token=30000, inter_token=10000),
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.TIMEOUT_RESET)
        if event:
            expected_fields = SPEC_EVENTS["TIMEOUT_RESET"]["fields"]
            errors = validate_event_fields(event, expected_fields, "TIMEOUT_RESET")
            assert not errors, "\n".join(errors)


# ============================================================================
# Adapter Event Signature Tests
# ============================================================================


class TestAdapterEventSignatures:
    """Tests for ADAPTER_* event signatures."""

    @pytest.mark.asyncio
    async def test_adapter_detected_uses_adapter_id(self) -> None:
        """ADAPTER_DETECTED should use 'adapterId' not 'adapter'."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.ADAPTER_DETECTED)
        if event:
            meta = event.meta
            assert "adapter" not in meta, (
                "ADAPTER_DETECTED uses 'adapter', should be 'adapterId'"
            )
            expected_fields = SPEC_EVENTS["ADAPTER_DETECTED"]["fields"]
            errors = validate_event_fields(event, expected_fields, "ADAPTER_DETECTED")
            assert not errors, "\n".join(errors)


# ============================================================================
# Network Event Signature Tests
# ============================================================================


class TestNetworkEventSignatures:
    """Tests for NETWORK_* event signatures."""

    @pytest.mark.asyncio
    async def test_network_error_has_correct_fields(self) -> None:
        """NETWORK_ERROR should have error, code, retryable fields."""
        from l0.types import Retry

        capture = EventCapture()

        # Create a stream that raises a network-like error
        async def stream_with_network_error():
            yield Event(type=EventType.TOKEN, text="Hello")
            raise ConnectionError("Connection reset by peer")

        try:
            result = await _internal_run(
                stream=stream_with_network_error,
                on_event=capture,
                retry=Retry(max_retries=0),  # Disable retries
            )
            async for _ in result:
                pass
        except Exception:
            pass  # Expected to fail

        event = capture.get_first(ObservabilityEventType.NETWORK_ERROR)
        assert event is not None, (
            "NETWORK_ERROR event should be emitted on network error"
        )

        expected_fields = SPEC_EVENTS["NETWORK_ERROR"]["fields"]
        errors = validate_event_fields(event, expected_fields, "NETWORK_ERROR")
        assert not errors, "\n".join(errors)

        meta = event.meta
        assert "error" in meta, "NETWORK_ERROR should have 'error' field"
        assert "retryable" in meta, "NETWORK_ERROR should have 'retryable' field"
        assert isinstance(meta["retryable"], bool), "retryable should be boolean"


# ============================================================================
# Tool Event Signature Tests
# ============================================================================


class TestToolEventSignatures:
    """Tests for TOOL_* event signatures."""

    @pytest.mark.asyncio
    async def test_tool_requested_arguments_is_object(self) -> None:
        """TOOL_REQUESTED arguments should be object, not string."""
        capture = EventCapture()

        async def stream_with_tool_call():
            yield Event(type=EventType.TOKEN, text="Let me check. ")
            yield Event(
                type=EventType.TOOL_CALL,
                data={
                    "id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "Seattle"}',
                },
            )
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=stream_with_tool_call,
            on_event=capture,
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.TOOL_REQUESTED)
        if event:
            meta = event.meta
            expected_fields = SPEC_EVENTS["TOOL_REQUESTED"]["fields"]
            errors = validate_event_fields(event, expected_fields, "TOOL_REQUESTED")
            assert not errors, "\n".join(errors)

            # arguments should be a dict, not a string
            if "arguments" in meta:
                assert isinstance(meta["arguments"], dict), (
                    f"TOOL_REQUESTED arguments should be dict, got {type(meta['arguments']).__name__}"
                )


# ============================================================================
# Drift Event Signature Tests
# ============================================================================


class TestDriftEventSignatures:
    """Tests for DRIFT_CHECK_* event signatures."""

    @pytest.mark.asyncio
    async def test_drift_check_result_has_correct_fields(self) -> None:
        """DRIFT_CHECK_RESULT should have detected, score, metrics, threshold."""
        capture = EventCapture()

        async def stream_for_drift():
            yield Event(type=EventType.TOKEN, text="Hello world")
            yield Event(type=EventType.COMPLETE)

        from l0.drift import DriftDetector
        from l0.types import CheckIntervals

        result = await _internal_run(
            stream=stream_for_drift,
            on_event=capture,
            drift_detector=DriftDetector(),
            check_intervals=CheckIntervals(drift=1),  # Check on every token
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.DRIFT_CHECK_RESULT)
        if event:
            expected_fields = SPEC_EVENTS["DRIFT_CHECK_RESULT"]["fields"]
            errors = validate_event_fields(event, expected_fields, "DRIFT_CHECK_RESULT")
            assert not errors, "\n".join(errors)

            # Specific field checks
            meta = event.meta
            assert "types" not in meta, (
                "DRIFT_CHECK_RESULT uses 'types', should use 'score' and 'metrics'"
            )
            assert "confidence" not in meta, (
                "DRIFT_CHECK_RESULT uses 'confidence', should use 'score'"
            )


# ============================================================================
# Guardrail Event Signature Tests
# ============================================================================


class TestGuardrailEventSignatures:
    """Tests for GUARDRAIL_* event signatures."""

    @pytest.mark.asyncio
    async def test_guardrail_phase_start_has_phase_field(self) -> None:
        """GUARDRAIL_PHASE_START should have phase field."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        from l0.guardrails import GuardrailRule, GuardrailViolation
        from l0.types import State

        def no_hello(state: State) -> list:
            if "hello" in state.content.lower():
                return [
                    GuardrailViolation(
                        rule="no-hello",
                        message="Content contains 'hello'",
                        severity="warning",
                    )
                ]
            return []

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
            guardrails=[GuardrailRule(name="no-hello", check=no_hello)],
        )
        try:
            async for _ in result:
                pass
        except Exception:
            pass  # Guardrail may fail

        event = capture.get_first(ObservabilityEventType.GUARDRAIL_PHASE_START)
        if event:
            expected_fields = SPEC_EVENTS["GUARDRAIL_PHASE_START"]["fields"]
            errors = validate_event_fields(
                event, expected_fields, "GUARDRAIL_PHASE_START"
            )
            assert not errors, "\n".join(errors)

            meta = event.meta
            assert "contextSize" not in meta, (
                "GUARDRAIL_PHASE_START has 'contextSize' which is not in spec"
            )
            if "phase" in meta:
                assert meta["phase"] in ("pre", "post"), (
                    f"phase should be 'pre' or 'post', got '{meta['phase']}'"
                )

    @pytest.mark.asyncio
    async def test_guardrail_rule_result_has_passed_field(self) -> None:
        """GUARDRAIL_RULE_RESULT should have passed boolean, not violations array."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        from l0.guardrails import GuardrailRule
        from l0.types import State

        def always_pass(state: State) -> list:
            return []  # No violations = pass

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
            guardrails=[GuardrailRule(name="always-pass", check=always_pass)],
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.GUARDRAIL_RULE_RESULT)
        if event:
            expected_fields = SPEC_EVENTS["GUARDRAIL_RULE_RESULT"]["fields"]
            errors = validate_event_fields(
                event, expected_fields, "GUARDRAIL_RULE_RESULT"
            )
            assert not errors, "\n".join(errors)

            meta = event.meta
            # Should have 'passed' boolean
            if "passed" not in meta:
                assert False, "GUARDRAIL_RULE_RESULT missing 'passed' field"
            # Should NOT have 'violations' array (single 'violation' object instead)
            assert "violations" not in meta, (
                "GUARDRAIL_RULE_RESULT has 'violations' array, "
                "should have single optional 'violation' object"
            )


# ============================================================================
# Checkpoint Event Signature Tests
# ============================================================================


class TestCheckpointEventSignatures:
    """Tests for CHECKPOINT_SAVED event signatures."""

    @pytest.mark.asyncio
    async def test_checkpoint_saved_uses_camel_case(self) -> None:
        """CHECKPOINT_SAVED should use camelCase field names."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello world")
            yield Event(type=EventType.COMPLETE)

        from l0.types import CheckIntervals

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
            check_intervals=CheckIntervals(checkpoint=1),  # Checkpoint every token
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.CHECKPOINT_SAVED)
        if event:
            expected_fields = SPEC_EVENTS["CHECKPOINT_SAVED"]["fields"]
            errors = validate_event_fields(event, expected_fields, "CHECKPOINT_SAVED")
            assert not errors, "\n".join(errors)

            meta = event.meta
            assert "token_count" not in meta, (
                "CHECKPOINT_SAVED uses 'token_count', should be 'tokenCount'"
            )


# ============================================================================
# Complete Event Signature Tests
# ============================================================================


class TestCompleteEventSignatures:
    """Tests for COMPLETE event signatures."""

    @pytest.mark.asyncio
    async def test_complete_has_correct_fields(self) -> None:
        """COMPLETE should have tokenCount, contentLength, state."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello world")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
        )
        async for _ in result:
            pass

        event = capture.get_first(ObservabilityEventType.COMPLETE)
        if event:
            expected_fields = SPEC_EVENTS["COMPLETE"]["fields"]
            errors = validate_event_fields(event, expected_fields, "COMPLETE")
            assert not errors, "\n".join(errors)

            meta = event.meta
            assert "token_count" not in meta, (
                "COMPLETE uses 'token_count', should be 'tokenCount'"
            )
            assert "content_length" not in meta, (
                "COMPLETE uses 'content_length', should be 'contentLength'"
            )


# ============================================================================
# Generic Field Naming Validation
# ============================================================================


class TestEventFieldNamingConvention:
    """Tests that all events use camelCase field names."""

    @pytest.mark.asyncio
    async def test_no_snake_case_fields_in_any_event(self) -> None:
        """All emitted events should use camelCase, not snake_case."""
        capture = EventCapture()

        async def comprehensive_stream():
            yield Event(type=EventType.TOKEN, text="Hello ")
            yield Event(type=EventType.TOKEN, text="world")
            yield Event(type=EventType.COMPLETE)

        from l0.types import CheckIntervals, Timeout

        result = await _internal_run(
            stream=comprehensive_stream,
            on_event=capture,
            timeout=Timeout(initial_token=30000, inter_token=10000),
            check_intervals=CheckIntervals(checkpoint=5),
        )
        async for _ in result:
            pass

        snake_case_violations = []
        for event in capture.events:
            for key in event.meta.keys():
                if "_" in key:
                    snake_case_violations.append(
                        f"{event.type.value}: field '{key}' uses snake_case"
                    )

        assert not snake_case_violations, (
            "Events should use camelCase field names:\n"
            + "\n".join(snake_case_violations)
        )


# ============================================================================
# Context Propagation Tests (Invariant: context-propagated)
# ============================================================================


class TestContextPropagation:
    """Tests that user context is propagated to all emitted events.

    Per canonical spec invariant 'context-propagated':
    'User context appears in all observability events'
    """

    @pytest.mark.asyncio
    async def test_context_included_in_all_events(self) -> None:
        """All emitted events should include the user-provided context."""
        capture = EventCapture()
        user_context = {"requestId": "req-123", "userId": "user-456"}

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
            context=user_context,
        )
        async for _ in result:
            pass

        assert len(capture.events) > 0, "Expected at least one event to be emitted"

        for event in capture.events:
            assert hasattr(event, "context"), (
                f"{event.type.value}: event missing 'context' attribute"
            )
            assert event.context == user_context, (
                f"{event.type.value}: context mismatch. "
                f"Expected {user_context}, got {event.context}"
            )

    @pytest.mark.asyncio
    async def test_empty_context_when_not_provided(self) -> None:
        """Events should have empty context dict when no context provided."""
        capture = EventCapture()

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
        )
        async for _ in result:
            pass

        for event in capture.events:
            assert hasattr(event, "context"), (
                f"{event.type.value}: event missing 'context' attribute"
            )
            assert event.context == {}, (
                f"{event.type.value}: expected empty context, got {event.context}"
            )

    @pytest.mark.asyncio
    async def test_context_is_immutable(self) -> None:
        """Context should be deeply cloned - mutations should not affect events."""
        capture = EventCapture()
        user_context: dict[str, Any] = {
            "requestId": "req-123",
            "nested": {"key": "original"},
        }

        async def simple_stream():
            yield Event(type=EventType.TOKEN, text="Hello")
            yield Event(type=EventType.COMPLETE)

        result = await _internal_run(
            stream=simple_stream,
            on_event=capture,
            context=user_context,
        )
        async for _ in result:
            pass

        # Mutate the original context after run
        user_context["requestId"] = "mutated"
        user_context["nested"]["key"] = "mutated"

        # Events should still have original values
        for event in capture.events:
            assert event.context.get("requestId") == "req-123", (
                f"{event.type.value}: context was mutated"
            )
