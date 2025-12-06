"""Event Parity Tests - Ensures Python events match canonical spec.

This test validates that Python does not emit events that are not
defined in the canonical lifecycle-scenarios.json specification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def load_lifecycle_scenarios() -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / "lifecycle-scenarios.json"
    with open(fixture_path) as f:
        return json.load(f)


LIFECYCLE_SCENARIOS = load_lifecycle_scenarios()


class TestNoUnknownEvents:
    """Tests that Python does not emit events not in the canonical spec."""

    @pytest.fixture
    def canonical_events(self) -> set[str]:
        """Get canonical event types from lifecycle-scenarios.json."""
        return set(LIFECYCLE_SCENARIOS["eventTypeReference"].keys())

    @pytest.fixture
    def allowed_python_only_events(self) -> set[str]:
        """Events allowed in Python but not in canonical spec."""
        return {
            "TOKEN",
            "STREAM_READY",
            "NETWORK_RECOVERY",
            "CONNECTION_DROPPED",
            "CONNECTION_RESTORED",
            "SESSION_END",
            "SESSION_SUMMARY",
            "ABORT_REQUESTED",
            "TOOL_REQUESTED",
            "TOOL_START",
            "TOOL_RESULT",
            "TOOL_ERROR",
            "TOOL_COMPLETED",
            "GUARDRAIL_CALLBACK_START",
            "GUARDRAIL_CALLBACK_END",
            "DRIFT_CHECK_RESULT",
            "DRIFT_CHECK_SKIPPED",
            "RETRY_FN_START",
            "RETRY_FN_RESULT",
            "RETRY_FN_ERROR",
            "FINALIZATION_START",
            "FINALIZATION_END",
            "CONSENSUS_START",
            "CONSENSUS_STREAM_START",
            "CONSENSUS_STREAM_END",
            "CONSENSUS_OUTPUT_COLLECTED",
            "CONSENSUS_ANALYSIS",
            "CONSENSUS_RESOLUTION",
            "CONSENSUS_END",
            "PARSE_START",
            "PARSE_END",
            "PARSE_ERROR",
            "SCHEMA_VALIDATION_START",
            "SCHEMA_VALIDATION_END",
            "SCHEMA_VALIDATION_ERROR",
            "AUTO_CORRECT_START",
            "AUTO_CORRECT_END",
            "STRUCTURED_PARSE_START",
            "STRUCTURED_PARSE_END",
            "STRUCTURED_PARSE_ERROR",
            "STRUCTURED_VALIDATION_START",
            "STRUCTURED_VALIDATION_END",
            "STRUCTURED_VALIDATION_ERROR",
            "STRUCTURED_AUTO_CORRECT_START",
            "STRUCTURED_AUTO_CORRECT_END",
            "TIMEOUT_TRIGGERED",
        }

    def test_no_unknown_events_emitted(
        self, canonical_events: set[str], allowed_python_only_events: set[str]
    ) -> None:
        """Ensure Python does not define unknown events."""
        from l0.events import ObservabilityEventType

        python_events = {e.value for e in ObservabilityEventType}
        allowed_events = canonical_events | allowed_python_only_events

        unknown_events = python_events - allowed_events
        assert not unknown_events, (
            f"Python emits unknown events: {unknown_events}. "
            f"Add to lifecycle-scenarios.json or allowed_python_only_events."
        )
