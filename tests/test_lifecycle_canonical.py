"""Canonical Lifecycle Tests for L0 Runtime (Python)"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from l0 import Retry
from l0.events import ObservabilityEvent, ObservabilityEventType
from l0.runtime import _internal_run
from l0.types import AwaitableStreamFactory, Event, EventType, State


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def load_scenarios() -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / "lifecycle-scenarios.json"
    with open(fixture_path) as f:
        return json.load(f)


SCENARIOS = load_scenarios()


@dataclass
class CollectedEvent:
    type: str
    ts: float
    data: dict[str, Any]


@dataclass
class EventCollector:
    events: list[CollectedEvent] = field(default_factory=list)

    def handler(self, event: ObservabilityEvent) -> None:
        evt_type = (
            event.type.value
            if isinstance(event.type, ObservabilityEventType)
            else str(event.type)
        )
        self.events.append(
            CollectedEvent(
                type=evt_type,
                ts=event.ts,
                data={
                    "type": evt_type,
                    "ts": event.ts,
                    "stream_id": event.stream_id,
                    "context": event.context,
                    **event.meta,
                },
            )
        )

    def get_event_types(self) -> list[str]:
        return [e.type for e in self.events]

    def get_events_of_type(self, event_type: str) -> list[CollectedEvent]:
        return [e for e in self.events if e.type == event_type]


def get_nested_value(obj: dict[str, Any], path: str) -> Any:
    current: Any = obj
    for key in path.split("."):
        if current is None or not isinstance(current, dict):
            return None
        if key in current:
            current = current[key]
        else:
            current = current.get(camel_to_snake(key))
    return current


def validate_event_assertions(
    event: CollectedEvent, assertions: dict[str, Any]
) -> None:
    for path, expected in assertions.items():
        actual = get_nested_value(event.data, path)
        assert actual == expected, (
            f"Event {event.type}: {path} expected {expected!r}, got {actual!r}"
        )


def validate_observability_event_sequence(
    collector: EventCollector, expected_events: list[dict[str, Any]]
) -> None:
    collected_types = collector.get_event_types()
    last_idx = -1
    for expected in expected_events:
        event_type = expected["type"]
        assert len(collector.get_events_of_type(event_type)) > 0, (
            f"Expected {event_type} event"
        )
        try:
            found_idx = collected_types.index(event_type, last_idx + 1)
        except ValueError:
            found_idx = -1
        assert found_idx > last_idx, f"Expected {event_type} after index {last_idx}"
        if "assertions" in expected:
            validate_event_assertions(
                collector.events[found_idx], expected["assertions"]
            )
        last_idx = found_idx


async def create_token_stream(tokens: list[str]) -> AsyncIterator[Event]:
    for token in tokens:
        yield Event(type=EventType.TOKEN, text=token)
    yield Event(type=EventType.COMPLETE)


async def create_failing_stream(
    tokens: list[str], error: Exception | None = None
) -> AsyncIterator[Event]:
    for token in tokens:
        yield Event(type=EventType.TOKEN, text=token)
    raise (error or Exception("Stream failed"))


async def run_normal_success_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    collector = EventCollector()
    config = scenario["config"]
    tokens = config["tokens"]
    context = config.get("context", {})

    async def stream_factory() -> AsyncIterator[Event]:
        async for event in create_token_stream(tokens):
            yield event

    result = await _internal_run(
        stream=stream_factory,
        context=context,
        on_event=collector.handler,
        retry=Retry(attempts=1, max_retries=1),
    )
    async for _ in result:
        pass

    validate_observability_event_sequence(
        collector, scenario["expectedObservabilityEvents"]
    )
    return {"collector": collector}


async def run_error_context_propagation_scenario(
    scenario: dict[str, Any],
) -> dict[str, Any]:
    collector = EventCollector()
    config = scenario["config"]
    context = config.get("context", {})
    fallback_streams = config.get("fallbackStreams", [])

    async def failing_stream() -> AsyncIterator[Event]:
        async for event in create_failing_stream([]):
            yield event

    fallback_factories: list[AwaitableStreamFactory] = [
        lambda tokens=fs["tokens"]: create_token_stream(tokens)
        for fs in fallback_streams
    ]

    result = await _internal_run(
        stream=failing_stream,
        fallbacks=fallback_factories,
        retry=Retry(attempts=0, max_retries=0),
        context=context,
        on_event=collector.handler,
    )
    async for _ in result:
        pass

    session_starts = collector.get_events_of_type("SESSION_START")
    assert len(session_starts) == 1
    session_ctx = session_starts[0].data.get("context", {})
    assert session_ctx.get("requestId") == "error-ctx-404"
    assert session_ctx.get("userId") == "user-xyz"
    assert session_ctx.get("nested", {}).get("traceId") == "trace-abc"

    errors = collector.get_events_of_type("ERROR")
    assert len(errors) > 0
    assert errors[0].data.get("context", {}).get("requestId") == "error-ctx-404"

    fallback_starts = collector.get_events_of_type("FALLBACK_START")
    assert len(fallback_starts) == 1
    assert (
        fallback_starts[0].data.get("context", {}).get("requestId") == "error-ctx-404"
    )

    completes = collector.get_events_of_type("COMPLETE")
    assert len(completes) == 1
    assert completes[0].data.get("context", {}).get("requestId") == "error-ctx-404"

    return {"collector": collector}


class TestCanonicalLifecycle:
    @pytest.fixture
    def scenarios(self) -> list[dict[str, Any]]:
        return SCENARIOS["scenarios"]

    def get_scenario(
        self, scenarios: list[dict[str, Any]], scenario_id: str
    ) -> dict[str, Any]:
        for s in scenarios:
            if s["id"] == scenario_id:
                return s
        raise ValueError(f"Scenario {scenario_id} not found")


class TestNormalSuccessFlow(TestCanonicalLifecycle):
    @pytest.mark.asyncio
    async def test_normal_success(self, scenarios: list[dict[str, Any]]) -> None:
        scenario = self.get_scenario(scenarios, "normal-success")
        await run_normal_success_scenario(scenario)

    @pytest.mark.asyncio
    async def test_invariants(self, scenarios: list[dict[str, Any]]) -> None:
        scenario = self.get_scenario(scenarios, "normal-success")
        result = await run_normal_success_scenario(scenario)
        collector = result["collector"]
        assert len(collector.get_events_of_type("SESSION_START")) == 1
        assert collector.get_event_types()[-1] == "COMPLETE"


class TestErrorContextPropagation(TestCanonicalLifecycle):
    @pytest.mark.asyncio
    async def test_error_context_propagation(
        self, scenarios: list[dict[str, Any]]
    ) -> None:
        scenario = self.get_scenario(scenarios, "error-context-propagation")
        await run_error_context_propagation_scenario(scenario)


class TestCrossLanguageInvariants(TestCanonicalLifecycle):
    @pytest.mark.asyncio
    async def test_session_start_is_first(
        self, scenarios: list[dict[str, Any]]
    ) -> None:
        scenario = self.get_scenario(scenarios, "normal-success")
        result = await run_normal_success_scenario(scenario)
        assert result["collector"].get_event_types()[0] == "SESSION_START"

    @pytest.mark.asyncio
    async def test_complete_is_final(self, scenarios: list[dict[str, Any]]) -> None:
        scenario = self.get_scenario(scenarios, "normal-success")
        result = await run_normal_success_scenario(scenario)
        assert result["collector"].get_event_types()[-1] == "COMPLETE"

    @pytest.mark.asyncio
    async def test_timestamps_monotonic(self, scenarios: list[dict[str, Any]]) -> None:
        scenario = self.get_scenario(scenarios, "normal-success")
        result = await run_normal_success_scenario(scenario)
        events = result["collector"].events
        for i in range(1, len(events)):
            assert events[i].ts >= events[i - 1].ts

    @pytest.mark.asyncio
    async def test_stream_id_consistent(self, scenarios: list[dict[str, Any]]) -> None:
        scenario = self.get_scenario(scenarios, "normal-success")
        result = await run_normal_success_scenario(scenario)
        events = [e for e in result["collector"].events if e.data.get("stream_id")]
        assert len(events) > 0
        stream_id = events[0].data["stream_id"]
        for e in events:
            assert e.data["stream_id"] == stream_id
