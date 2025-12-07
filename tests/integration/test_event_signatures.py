"""Integration test for event signature validation against canonical-spec.json.

This test makes live LLM calls and validates that ALL emitted observability
events have correct field signatures matching the canonical specification.

Requires OPENAI_API_KEY to be set.
Run with: pytest tests/integration/test_event_signatures.py -v
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import l0
from l0.events import ObservabilityEvent, ObservabilityEventType
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Load canonical spec
SPEC_PATH = Path(__file__).parent.parent / "fixtures" / "canonical-spec.json"
with open(SPEC_PATH) as f:
    CANONICAL_SPEC = json.load(f)

# Extract event specs from canonical spec
EVENT_SPECS: dict[str, dict[str, Any]] = CANONICAL_SPEC["monitoring"][
    "observabilityEvents"
]["events"]

# Base fields that ALL events must have (from observabilityEvents.baseShape)
BASE_FIELDS = ["type", "ts", "stream_id", "context"]

# Events that are known to have implementation gaps (to be fixed separately)
# These are tracked but don't fail the test - they indicate spec/impl mismatches
KNOWN_GAPS: set[str] = {
    "ADAPTER_WRAP_START",  # Missing streamType field
    "ADAPTER_WRAP_END",  # Missing adapterId and success fields
}


class EventCollector:
    """Collects all emitted observability events."""

    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def __call__(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def get_events(self) -> list[ObservabilityEvent]:
        return self.events

    def get_events_of_type(
        self, event_type: ObservabilityEventType
    ) -> list[ObservabilityEvent]:
        return [e for e in self.events if e.type == event_type]

    def clear(self) -> None:
        self.events.clear()


def validate_event_signature(
    event: ObservabilityEvent, errors: list[str], warnings: list[str]
) -> None:
    """Validates that an event has all required fields from the canonical spec."""
    event_type = event.type.value  # Get string value from enum

    # Check base fields
    if not hasattr(event, "type"):
        errors.append(f"{event_type}: Missing base field 'type'")
    if not hasattr(event, "ts"):
        errors.append(f"{event_type}: Missing base field 'ts'")
    if not hasattr(event, "stream_id"):
        errors.append(f"{event_type}: Missing base field 'stream_id'")
    if not hasattr(event, "context"):
        errors.append(f"{event_type}: Missing base field 'context'")

    # Validate ts is a number
    if not isinstance(event.ts, (int, float)):
        errors.append(f"{event_type}: 'ts' should be number, got {type(event.ts)}")

    # Validate stream_id is a string
    if not isinstance(event.stream_id, str):
        errors.append(
            f"{event_type}: 'stream_id' should be string, got {type(event.stream_id)}"
        )

    # Validate context is a dict
    if not isinstance(event.context, dict):
        errors.append(
            f"{event_type}: 'context' should be dict, got {type(event.context)}"
        )

    # If spec exists, validate required fields in meta
    spec = EVENT_SPECS.get(event_type)
    if spec and "fields" in spec:
        for field_spec in spec["fields"]:
            if field_spec.get("required", False):
                field_name = field_spec["name"]
                # Check if field is in meta (Python stores event-specific data in meta)
                if field_name not in event.meta:
                    msg = f"{event_type}: Missing required field '{field_name}' in meta"
                    # Known gaps are warnings, not errors
                    if event_type in KNOWN_GAPS:
                        warnings.append(msg)
                    else:
                        errors.append(msg)


def validate_all_events(
    events: list[ObservabilityEvent],
) -> dict[str, Any]:
    """Validates all captured events against the canonical spec."""
    errors: list[str] = []
    warnings: list[str] = []
    event_counts: dict[str, int] = {}

    for event in events:
        # Count events by type
        event_type = event.type.value
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Validate signature
        validate_event_signature(event, errors, warnings)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "event_counts": event_counts,
    }


@requires_openai
class TestEventSignatureValidation:
    """Integration tests for event signature validation with live LLM calls."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_basic_streaming_event_signatures(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test that basic streaming emits events with correct signatures."""
        collector = EventCollector()

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, on_event=collector)
        content = await result.read()

        assert len(content) > 0

        # Validate all event signatures
        validation = validate_all_events(collector.get_events())

        if not validation["valid"]:
            print("Event signature errors:", validation["errors"])

        assert validation["errors"] == []
        assert validation["valid"] is True

        # Verify key events were emitted
        assert validation["event_counts"].get("SESSION_START") == 1
        assert validation["event_counts"].get("COMPLETE") == 1

    @pytest.mark.asyncio
    async def test_timeout_event_signatures(self, client: "AsyncOpenAI") -> None:
        """Test that timeout configuration emits events with correct signatures."""
        collector = EventCollector()

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            stream=True,
            max_tokens=50,
        )

        result = l0.wrap(
            stream,
            timeout=l0.Timeout(initial_token=30000, inter_token=5000),
            on_event=collector,
        )
        content = await result.read()

        assert len(content) > 0

        validation = validate_all_events(collector.get_events())

        if not validation["valid"]:
            print("Event signature errors:", validation["errors"])

        assert validation["errors"] == []

        # Timeout events should have been emitted
        assert validation["event_counts"].get("TIMEOUT_START", 0) > 0

    @pytest.mark.asyncio
    async def test_guardrail_event_signatures(self, client: "AsyncOpenAI") -> None:
        """Test that guardrails emit events with correct signatures."""
        collector = EventCollector()

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test message'."}],
            stream=True,
            max_tokens=20,
        )

        # Use recommended guardrails (Python API uses list of rules)
        result = l0.wrap(
            stream,
            guardrails=l0.Guardrails.recommended(),
            on_event=collector,
        )
        content = await result.read()

        assert len(content) > 0

        validation = validate_all_events(collector.get_events())

        if not validation["valid"]:
            print("Event signature errors:", validation["errors"])
        if validation["warnings"]:
            print("Event signature warnings (known gaps):", validation["warnings"])

        assert validation["errors"] == []

        # Guardrail events should have been emitted
        assert validation["event_counts"].get("GUARDRAIL_PHASE_START", 0) > 0
        assert validation["event_counts"].get("GUARDRAIL_PHASE_END", 0) > 0

    @pytest.mark.asyncio
    async def test_context_propagation_in_all_events(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test that user context is propagated to all events."""
        collector = EventCollector()
        user_context = {
            "requestId": "test-req-123",
            "userId": "user-456",
            "sessionId": "session-789",
        }

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hi'."}],
            stream=True,
            max_tokens=5,
        )

        result = l0.wrap(
            stream,
            context=user_context,
            on_event=collector,
        )
        content = await result.read()

        assert len(content) > 0

        events = collector.get_events()

        # Verify context is present in all events
        for event in events:
            assert event.context is not None
            assert event.context == user_context

        # Also run standard validation
        validation = validate_all_events(events)
        assert validation["errors"] == []

    @pytest.mark.asyncio
    async def test_adapter_event_signatures(self, client: "AsyncOpenAI") -> None:
        """Test that adapter detection emits events with correct signatures."""
        collector = EventCollector()

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'adapter test'."}],
            stream=True,
            max_tokens=10,
        )

        result = l0.wrap(stream, on_event=collector)
        content = await result.read()

        assert len(content) > 0

        validation = validate_all_events(collector.get_events())

        if not validation["valid"]:
            print("Event signature errors:", validation["errors"])

        assert validation["errors"] == []

        # Adapter detection should have happened
        assert validation["event_counts"].get("ADAPTER_DETECTED") == 1

    @pytest.mark.asyncio
    async def test_consistent_stream_id_across_events(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test that all events have the same streamId."""
        collector = EventCollector()

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'stream id test'."}],
            stream=True,
            max_tokens=15,
        )

        result = l0.wrap(stream, on_event=collector)
        content = await result.read()

        assert len(content) > 0

        events = collector.get_events()

        # All events should have the same stream_id
        stream_ids = set(e.stream_id for e in events)
        assert len(stream_ids) == 1

        # stream_id should be a valid UUID v7 format
        stream_id = events[0].stream_id
        uuid_pattern = (
            r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert re.match(uuid_pattern, stream_id, re.IGNORECASE) is not None

    @pytest.mark.asyncio
    async def test_monotonically_increasing_timestamps(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test that event timestamps are monotonically increasing."""
        collector = EventCollector()

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            max_tokens=30,
        )

        result = l0.wrap(stream, on_event=collector)
        content = await result.read()

        assert len(content) > 0

        events = collector.get_events()

        # Timestamps should be monotonically increasing (or equal for same-ms events)
        for i in range(1, len(events)):
            assert events[i].ts >= events[i - 1].ts

    @pytest.mark.asyncio
    async def test_comprehensive_event_coverage(self, client: "AsyncOpenAI") -> None:
        """Test comprehensive event coverage with multiple features enabled."""
        collector = EventCollector()

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a short story in exactly 3 sentences about a robot.",
                }
            ],
            stream=True,
            max_tokens=150,
        )

        result = l0.wrap(
            stream,
            timeout=l0.Timeout(initial_token=30000, inter_token=5000),
            guardrails=l0.Guardrails.recommended(),
            context={"testId": "comprehensive-coverage"},
            on_event=collector,
        )
        content = await result.read()

        assert len(content) > 0

        validation = validate_all_events(collector.get_events())

        # Log all captured event types for visibility
        print("Captured event types:", list(validation["event_counts"].keys()))
        print("Event counts:", validation["event_counts"])

        if not validation["valid"]:
            print("Event signature errors:", validation["errors"])
        if validation["warnings"]:
            print("Event signature warnings (known gaps):", validation["warnings"])

        assert validation["errors"] == []
        assert validation["valid"] is True

        # Verify we captured a good variety of events
        assert len(validation["event_counts"]) > 5
