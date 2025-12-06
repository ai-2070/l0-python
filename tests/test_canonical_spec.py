"""Canonical Specification Tests for L0 Runtime (Python)

These tests validate the L0 runtime against the canonical specification
defined in fixtures/canonical-spec.json. This ensures consistency between
TypeScript and Python implementations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from l0.errors import Error, ErrorCode, ErrorCategory, ErrorContext


def load_spec() -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / "canonical-spec.json"
    with open(fixture_path) as f:
        return json.load(f)


SPEC = load_spec()


class TestL0ErrorToJSON:
    """Tests for Error.to_json() canonical format."""

    def test_returns_all_required_fields(self) -> None:
        error = Error(
            "Test error",
            ErrorContext(
                code=ErrorCode.STREAM_ABORTED,
                checkpoint="checkpoint-content",
                token_count=10,
                model_retry_count=2,
                network_retry_count=1,
                fallback_index=0,
                metadata={"violation": {"rule": "test"}},
                context={"requestId": "req-123"},
            ),
        )

        result = error.to_json()

        # Verify all fields from canonical spec (camelCase)
        assert result["name"] == "Error"
        assert result["code"] == "STREAM_ABORTED"
        assert "category" in result
        assert result["message"] == "Test error"
        assert "timestamp" in result
        assert result["hasCheckpoint"] is True
        assert result["checkpoint"] == "checkpoint-content"
        assert result["tokenCount"] == 10
        assert result["modelRetryCount"] == 2
        assert result["networkRetryCount"] == 1
        assert result["fallbackIndex"] == 0
        assert "metadata" in result
        assert "context" in result

    def test_includes_metadata_for_internal_state(self) -> None:
        error = Error(
            "Guardrail failed",
            ErrorContext(
                code=ErrorCode.GUARDRAIL_VIOLATION,
                metadata={
                    "violation": {
                        "rule": "no-pii",
                        "severity": "error",
                        "message": "PII detected",
                    }
                },
            ),
        )

        result = error.to_json()
        assert result["metadata"] == {
            "violation": {
                "rule": "no-pii",
                "severity": "error",
                "message": "PII detected",
            }
        }

    def test_includes_context_for_user_provided_data(self) -> None:
        error = Error(
            "Network error",
            ErrorContext(
                code=ErrorCode.NETWORK_ERROR,
                context={
                    "requestId": "req-456",
                    "userId": "user-789",
                    "nested": {"traceId": "trace-abc"},
                },
            ),
        )

        result = error.to_json()
        assert result["context"] == {
            "requestId": "req-456",
            "userId": "user-789",
            "nested": {"traceId": "trace-abc"},
        }

    def test_handles_undefined_optional_fields(self) -> None:
        error = Error(
            "Minimal error",
            ErrorContext(code=ErrorCode.INVALID_STREAM),
        )

        result = error.to_json()
        assert result["checkpoint"] is None
        assert result["tokenCount"] is None
        assert result["modelRetryCount"] is None
        assert result["networkRetryCount"] is None
        assert result["fallbackIndex"] is None
        assert result["metadata"] is None
        assert result["context"] is None

    def test_computes_has_checkpoint_correctly(self) -> None:
        # No checkpoint
        error1 = Error("No checkpoint", ErrorContext(code=ErrorCode.STREAM_ABORTED))
        assert error1.to_json()["hasCheckpoint"] is False

        # Empty checkpoint
        error2 = Error(
            "Empty checkpoint",
            ErrorContext(code=ErrorCode.STREAM_ABORTED, checkpoint=""),
        )
        assert error2.to_json()["hasCheckpoint"] is False

        # Valid checkpoint
        error3 = Error(
            "Has checkpoint",
            ErrorContext(code=ErrorCode.STREAM_ABORTED, checkpoint="content"),
        )
        assert error3.to_json()["hasCheckpoint"] is True

    def test_to_json_field_names_are_camel_case(self) -> None:
        """Verify all field names use camelCase to match TypeScript."""
        error = Error(
            "Test",
            ErrorContext(
                code=ErrorCode.STREAM_ABORTED,
                checkpoint="x",
                token_count=1,
                model_retry_count=1,
                network_retry_count=1,
                fallback_index=0,
            ),
        )

        result = error.to_json()
        keys = set(result.keys())

        # Should have camelCase keys
        assert "hasCheckpoint" in keys
        assert "tokenCount" in keys
        assert "modelRetryCount" in keys
        assert "networkRetryCount" in keys
        assert "fallbackIndex" in keys

        # Should NOT have snake_case keys
        assert "has_checkpoint" not in keys
        assert "token_count" not in keys
        assert "model_retry_count" not in keys
        assert "network_retry_count" not in keys
        assert "fallback_index" not in keys



class TestErrorCodeToCategoryMapping:
    """Tests for error code to category mapping."""

    def test_maps_network_error_to_network_category(self) -> None:
        error = Error("test", ErrorContext(code=ErrorCode.NETWORK_ERROR))
        assert error.category == ErrorCategory.NETWORK

    def test_maps_timeout_codes_to_transient_category(self) -> None:
        error1 = Error("test", ErrorContext(code=ErrorCode.INITIAL_TOKEN_TIMEOUT))
        assert error1.category == ErrorCategory.TRANSIENT

        error2 = Error("test", ErrorContext(code=ErrorCode.INTER_TOKEN_TIMEOUT))
        assert error2.category == ErrorCategory.TRANSIENT

    def test_maps_content_quality_codes_to_content_category(self) -> None:
        codes = [
            ErrorCode.GUARDRAIL_VIOLATION,
            ErrorCode.FATAL_GUARDRAIL_VIOLATION,
            ErrorCode.DRIFT_DETECTED,
            ErrorCode.ZERO_OUTPUT,
        ]
        for code in codes:
            error = Error("test", ErrorContext(code=code))
            assert error.category == ErrorCategory.CONTENT, f"{code} should map to CONTENT"

    def test_maps_internal_codes_to_internal_category(self) -> None:
        codes = [
            ErrorCode.INVALID_STREAM,
            ErrorCode.ADAPTER_NOT_FOUND,
            ErrorCode.FEATURE_NOT_ENABLED,
        ]
        for code in codes:
            error = Error("test", ErrorContext(code=code))
            assert error.category == ErrorCategory.INTERNAL, f"{code} should map to INTERNAL"

    def test_maps_provider_codes_to_provider_category(self) -> None:
        codes = [
            ErrorCode.STREAM_ABORTED,
            ErrorCode.ALL_STREAMS_EXHAUSTED,
        ]
        for code in codes:
            error = Error("test", ErrorContext(code=code))
            assert error.category == ErrorCategory.PROVIDER, f"{code} should map to PROVIDER"


class TestAllErrorCodesExist:
    """Tests that all spec error codes exist in Python."""

    @pytest.fixture
    def spec_error_codes(self) -> list[str]:
        return list(SPEC["errorHandling"]["L0ErrorCodes"]["values"].keys())

    def test_all_spec_error_codes_exist(self, spec_error_codes: list[str]) -> None:
        for code in spec_error_codes:
            assert hasattr(ErrorCode, code), f"ErrorCode.{code} should exist"


class TestObservabilityEvents:
    """Tests for observability event types."""

    @pytest.fixture
    def spec_events(self) -> list[str]:
        return list(SPEC["monitoring"]["observabilityEvents"]["events"].keys())

    def test_all_spec_events_exist(self, spec_events: list[str]) -> None:
        from l0.events import ObservabilityEventType

        for event in spec_events:
            assert hasattr(ObservabilityEventType, event), f"ObservabilityEventType.{event} should exist"

    def test_event_values_match_keys(self) -> None:
        from l0.events import ObservabilityEventType

        event_types = [
            "SESSION_START",
            "ATTEMPT_START",
            "FALLBACK_START",
            "RETRY_ATTEMPT",
            "ERROR",
            "COMPLETE",
            "CHECKPOINT_SAVED",
            "RESUME_START",
            "ABORT_COMPLETED",
            "GUARDRAIL_RULE_RESULT",
            "TIMEOUT_TRIGGERED",
        ]
        for evt in event_types:
            assert getattr(ObservabilityEventType, evt).value == evt


class TestLifecycleInvariants:
    """Tests for documented lifecycle invariants."""

    @pytest.fixture
    def invariants(self) -> list[dict[str, str]]:
        return SPEC["lifecycleInvariants"]["invariants"]

    def test_documents_all_critical_invariants(self, invariants: list[dict[str, str]]) -> None:
        invariant_ids = [i["id"] for i in invariants]
        expected = [
            "session-start-once",
            "attempt-start-retries-only",
            "fallback-not-attempt",
            "retry-precedes-attempt",
            "timestamps-monotonic",
            "stream-id-consistent",
            "context-immutable",
            "context-propagated",
        ]
        for inv_id in expected:
            assert inv_id in invariant_ids, f"Invariant {inv_id} should be documented"

    def test_invariants_have_rule_and_rationale(self, invariants: list[dict[str, str]]) -> None:
        for inv in invariants:
            assert inv.get("rule"), f"Invariant {inv[id]} should have a rule"
            assert inv.get("rationale"), f"Invariant {inv[id]} should have a rationale"

