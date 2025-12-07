"""L0 Event Sourcing - Record and replay streams for testing and debugging.

Provides in-memory and extensible storage for atomic, replayable events.

Usage:
    from l0 import (
        EventStore,
        EventRecorder,
        EventReplayer,
        create_in_memory_event_store,
        create_event_recorder,
        replay,
    )

    # Record a stream
    store = create_in_memory_event_store()
    recorder = create_event_recorder(store, "my-stream")

    await recorder.record_start({"prompt": "test", "model": "gpt-4"})
    await recorder.record_token("Hello", 0)
    await recorder.record_token(" World", 1)
    await recorder.record_complete("Hello World", 2)

    # Replay it - exact same output, no API calls
    result = await replay(
        stream_id="my-stream",
        event_store=store,
        fire_callbacks=True,
    )

    async for event in result.stream:
        print(event)  # Same events as original
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .types import Event, EventType, State

# ─────────────────────────────────────────────────────────────────────────────
# Event Types
# ─────────────────────────────────────────────────────────────────────────────


class RecordedEventType(str, Enum):
    """Types of recorded events."""

    START = "START"
    TOKEN = "TOKEN"
    CHECKPOINT = "CHECKPOINT"
    GUARDRAIL = "GUARDRAIL"
    DRIFT = "DRIFT"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    CONTINUATION = "CONTINUATION"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


@dataclass
class RecordedEvent:
    """A recorded event from L0 execution."""

    type: RecordedEventType
    ts: float  # Timestamp in milliseconds
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventEnvelope:
    """Wrapper around a recorded event with metadata."""

    stream_id: str
    seq: int  # Sequence number within the stream
    event: RecordedEvent


@dataclass
class Snapshot:
    """State snapshot for faster replay."""

    stream_id: str
    seq: int  # Sequence number at snapshot
    state: dict[str, Any]  # Serialized state
    ts: float  # Timestamp


@dataclass
class SerializedError:
    """Serialized error for storage."""

    name: str
    message: str
    stack: str | None = None
    code: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Event Store Interface
# ─────────────────────────────────────────────────────────────────────────────


class EventStore(ABC):
    """Abstract base class for event stores."""

    @abstractmethod
    async def append(self, stream_id: str, event: RecordedEvent) -> None:
        """Append an event to a stream."""
        ...

    @abstractmethod
    async def get_events(self, stream_id: str) -> list[EventEnvelope]:
        """Get all events for a stream."""
        ...

    @abstractmethod
    async def exists(self, stream_id: str) -> bool:
        """Check if a stream exists."""
        ...

    @abstractmethod
    async def get_last_event(self, stream_id: str) -> EventEnvelope | None:
        """Get the last event in a stream."""
        ...

    @abstractmethod
    async def get_events_after(
        self, stream_id: str, after_seq: int
    ) -> list[EventEnvelope]:
        """Get events after a sequence number."""
        ...

    @abstractmethod
    async def delete(self, stream_id: str) -> None:
        """Delete a stream."""
        ...

    @abstractmethod
    async def list_streams(self) -> list[str]:
        """List all stream IDs."""
        ...


class EventStoreWithSnapshots(EventStore):
    """Event store with snapshot support."""

    @abstractmethod
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a state snapshot."""
        ...

    @abstractmethod
    async def get_snapshot(self, stream_id: str) -> Snapshot | None:
        """Get the latest snapshot for a stream."""
        ...

    @abstractmethod
    async def get_snapshot_before(self, stream_id: str, seq: int) -> Snapshot | None:
        """Get the latest snapshot before a sequence number."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Event Store
# ─────────────────────────────────────────────────────────────────────────────


class InMemoryEventStore(EventStoreWithSnapshots):
    """In-memory event store for testing and short-lived sessions.

    Not suitable for production persistence - events are lost on process exit.
    Use for:
    - Unit/integration testing with record/replay
    - Development debugging
    - Short-lived serverless functions
    """

    def __init__(self) -> None:
        self._streams: dict[str, list[EventEnvelope]] = {}
        self._snapshots: dict[str, list[Snapshot]] = {}

    async def append(self, stream_id: str, event: RecordedEvent) -> None:
        """Append an event to a stream."""
        if stream_id not in self._streams:
            self._streams[stream_id] = []

        events = self._streams[stream_id]
        envelope = EventEnvelope(
            stream_id=stream_id,
            seq=len(events),
            event=event,
        )
        events.append(envelope)

    async def get_events(self, stream_id: str) -> list[EventEnvelope]:
        """Get all events for a stream."""
        return list(self._streams.get(stream_id, []))

    async def exists(self, stream_id: str) -> bool:
        """Check if a stream exists."""
        return stream_id in self._streams

    async def get_last_event(self, stream_id: str) -> EventEnvelope | None:
        """Get the last event in a stream."""
        events = self._streams.get(stream_id)
        if not events:
            return None
        return events[-1]

    async def get_events_after(
        self, stream_id: str, after_seq: int
    ) -> list[EventEnvelope]:
        """Get events after a sequence number."""
        events = self._streams.get(stream_id, [])
        return [e for e in events if e.seq > after_seq]

    async def delete(self, stream_id: str) -> None:
        """Delete a stream."""
        self._streams.pop(stream_id, None)
        self._snapshots.pop(stream_id, None)

    async def list_streams(self) -> list[str]:
        """List all stream IDs."""
        return list(self._streams.keys())

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a state snapshot."""
        if snapshot.stream_id not in self._snapshots:
            self._snapshots[snapshot.stream_id] = []
        self._snapshots[snapshot.stream_id].append(snapshot)

    async def get_snapshot(self, stream_id: str) -> Snapshot | None:
        """Get the latest snapshot for a stream."""
        snapshots = self._snapshots.get(stream_id)
        if not snapshots:
            return None
        return snapshots[-1]

    async def get_snapshot_before(self, stream_id: str, seq: int) -> Snapshot | None:
        """Get the latest snapshot before a sequence number."""
        snapshots = self._snapshots.get(stream_id, [])
        best: Snapshot | None = None
        for snapshot in snapshots:
            if snapshot.seq <= seq:
                if best is None or snapshot.seq > best.seq:
                    best = snapshot
        return best

    def clear(self) -> None:
        """Clear all data (useful for testing)."""
        self._streams.clear()
        self._snapshots.clear()

    def get_total_event_count(self) -> int:
        """Get total event count across all streams."""
        return sum(len(events) for events in self._streams.values())

    def get_stream_count(self) -> int:
        """Get stream count."""
        return len(self._streams)


# ─────────────────────────────────────────────────────────────────────────────
# Event Recorder
# ─────────────────────────────────────────────────────────────────────────────


def generate_stream_id() -> str:
    """Generate a unique stream ID."""
    return f"l0-{uuid.uuid4().hex[:12]}"


class EventRecorder:
    """Event recorder - wraps an event store with convenient recording methods."""

    def __init__(
        self,
        event_store: EventStore,
        stream_id: str | None = None,
    ) -> None:
        self._event_store = event_store
        self._stream_id = stream_id or generate_stream_id()
        self._seq = 0

    @property
    def stream_id(self) -> str:
        """Get the stream ID."""
        return self._stream_id

    @property
    def seq(self) -> int:
        """Get the current sequence number."""
        return self._seq

    async def record(self, event: RecordedEvent) -> None:
        """Record a generic event."""
        await self._event_store.append(self._stream_id, event)
        self._seq += 1

    async def record_start(self, options: dict[str, Any]) -> None:
        """Record stream start."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.START,
                ts=time.time() * 1000,
                data={"options": options},
            )
        )

    async def record_token(self, value: str, index: int) -> None:
        """Record a token."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.TOKEN,
                ts=time.time() * 1000,
                data={"value": value, "index": index},
            )
        )

    async def record_checkpoint(self, at: int, content: str) -> None:
        """Record a checkpoint."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.CHECKPOINT,
                ts=time.time() * 1000,
                data={"at": at, "content": content},
            )
        )

    async def record_guardrail(self, at: int, result: dict[str, Any]) -> None:
        """Record a guardrail check result."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.GUARDRAIL,
                ts=time.time() * 1000,
                data={"at": at, "result": result},
            )
        )

    async def record_drift(self, at: int, result: dict[str, Any]) -> None:
        """Record a drift detection result."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.DRIFT,
                ts=time.time() * 1000,
                data={"at": at, "result": result},
            )
        )

    async def record_retry(
        self,
        reason: str,
        attempt: int,
        counts_toward_limit: bool,
    ) -> None:
        """Record a retry."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.RETRY,
                ts=time.time() * 1000,
                data={
                    "reason": reason,
                    "attempt": attempt,
                    "counts_toward_limit": counts_toward_limit,
                },
            )
        )

    async def record_fallback(self, to: int) -> None:
        """Record a fallback switch."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.FALLBACK,
                ts=time.time() * 1000,
                data={"to": to},
            )
        )

    async def record_continuation(self, checkpoint: str, at: int) -> None:
        """Record a continuation from checkpoint."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.CONTINUATION,
                ts=time.time() * 1000,
                data={"checkpoint": checkpoint, "at": at},
            )
        )

    async def record_complete(self, content: str, token_count: int) -> None:
        """Record stream completion."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.COMPLETE,
                ts=time.time() * 1000,
                data={"content": content, "token_count": token_count},
            )
        )

    async def record_error(
        self,
        error: SerializedError,
        failure_type: str,
        recovery_strategy: str,
        policy: dict[str, Any] | None = None,
    ) -> None:
        """Record an error."""
        await self.record(
            RecordedEvent(
                type=RecordedEventType.ERROR,
                ts=time.time() * 1000,
                data={
                    "error": {
                        "name": error.name,
                        "message": error.message,
                        "stack": error.stack,
                        "code": error.code,
                    },
                    "failure_type": failure_type,
                    "recovery_strategy": recovery_strategy,
                    "policy": policy,
                },
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Event Replayer
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReplayedState:
    """State reconstructed from replay."""

    content: str = ""
    token_count: int = 0
    checkpoint: str = ""
    violations: list[Any] = field(default_factory=list)
    drift_detected: bool = False
    retry_attempts: int = 0
    network_retry_count: int = 0
    fallback_index: int = 0
    completed: bool = False
    error: SerializedError | None = None
    start_ts: float = 0
    end_ts: float = 0


class EventReplayer:
    """Event replayer - replays events from a store."""

    def __init__(self, event_store: EventStore) -> None:
        self._event_store = event_store

    async def replay(
        self,
        stream_id: str,
        *,
        speed: float = 0,
        from_seq: int = 0,
        to_seq: int | None = None,
    ) -> AsyncGenerator[EventEnvelope, None]:
        """Replay all events for a stream.

        Args:
            stream_id: The stream to replay.
            speed: Playback speed (0 = instant, 1 = real-time).
            from_seq: Start from this sequence.
            to_seq: Stop at this sequence (None = no limit).

        Yields:
            EventEnvelope for each recorded event.
        """
        import asyncio

        events = await self._event_store.get_events(stream_id)
        last_ts: float | None = None
        max_seq = to_seq if to_seq is not None else float("inf")

        for envelope in events:
            # Skip events outside range
            if envelope.seq < from_seq:
                continue
            if envelope.seq > max_seq:
                break

            # Simulate timing if speed > 0
            # Timestamps are in milliseconds, convert to seconds for asyncio.sleep
            if speed > 0 and last_ts is not None:
                delay = (envelope.event.ts - last_ts) / 1000 / speed
                if delay > 0:
                    await asyncio.sleep(delay)

            last_ts = envelope.event.ts
            yield envelope

    async def replay_to_state(self, stream_id: str) -> ReplayedState:
        """Replay and reconstruct final state."""
        state = ReplayedState()

        async for envelope in self.replay(stream_id):
            event = envelope.event
            data = event.data

            if event.type == RecordedEventType.START:
                state.start_ts = event.ts

            elif event.type == RecordedEventType.TOKEN:
                state.content += data.get("value", "")
                state.token_count = data.get("index", 0) + 1

            elif event.type == RecordedEventType.CHECKPOINT:
                state.checkpoint = data.get("content", "")

            elif event.type == RecordedEventType.GUARDRAIL:
                result = data.get("result", {})
                state.violations.extend(result.get("violations", []))

            elif event.type == RecordedEventType.DRIFT:
                result = data.get("result", {})
                if result.get("detected"):
                    state.drift_detected = True

            elif event.type == RecordedEventType.RETRY:
                if data.get("counts_toward_limit"):
                    state.retry_attempts += 1
                else:
                    state.network_retry_count += 1

            elif event.type == RecordedEventType.FALLBACK:
                state.fallback_index = data.get("to", 0)

            elif event.type == RecordedEventType.CONTINUATION:
                state.content = data.get("checkpoint", "")

            elif event.type == RecordedEventType.COMPLETE:
                state.completed = True
                state.content = data.get("content", "")
                state.token_count = data.get("token_count", 0)
                state.end_ts = event.ts

            elif event.type == RecordedEventType.ERROR:
                error_data = data.get("error", {})
                state.error = SerializedError(
                    name=error_data.get("name", "Error"),
                    message=error_data.get("message", ""),
                    stack=error_data.get("stack"),
                    code=error_data.get("code"),
                )
                state.end_ts = event.ts

        return state

    async def replay_tokens(
        self,
        stream_id: str,
        *,
        speed: float = 0,
    ) -> AsyncGenerator[str, None]:
        """Get stream as token async iterable (for replay mode)."""
        async for envelope in self.replay(stream_id, speed=speed):
            if envelope.event.type == RecordedEventType.TOKEN:
                yield envelope.event.data.get("value", "")


# ─────────────────────────────────────────────────────────────────────────────
# Replay Function
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReplayOptions:
    """Options for replaying a stream."""

    stream_id: str
    event_store: EventStore
    speed: float = 0  # 0 = instant, 1 = real-time
    fire_callbacks: bool = True
    from_seq: int = 0
    to_seq: int | None = None


@dataclass
class ReplayCallbacks:
    """Callbacks that fire during replay."""

    on_token: Callable[[str], None] | None = None
    on_violation: Callable[[Any], None] | None = None
    on_retry: Callable[[int, str], None] | None = None
    on_event: Callable[[Event], None] | None = None


@dataclass
class StreamMetadata:
    """Metadata about a stored stream."""

    stream_id: str
    event_count: int
    token_count: int
    start_ts: float
    end_ts: float
    completed: bool
    has_error: bool
    options: dict[str, Any]


@dataclass
class ReplayComparison:
    """Result of comparing two replays."""

    identical: bool
    differences: list[str]


@dataclass
class ReplayResult:
    """Result from replay operation."""

    stream: AsyncGenerator[Event, None]
    state: State
    errors: list[Exception]
    stream_id: str
    is_replay: bool = True
    original_options: dict[str, Any] = field(default_factory=dict)
    _callbacks: ReplayCallbacks = field(default_factory=ReplayCallbacks)
    _abort_requested: bool = False

    def abort(self) -> None:
        """Abort the replay."""
        self._abort_requested = True

    def set_callbacks(self, callbacks: ReplayCallbacks) -> None:
        """Set callbacks before iterating."""
        self._callbacks = callbacks


async def replay(
    stream_id: str,
    event_store: EventStore,
    *,
    speed: float = 0,
    fire_callbacks: bool = True,
    from_seq: int = 0,
    to_seq: int | None = None,
) -> ReplayResult:
    """Replay an L0 stream from stored events.

    This is a PURE replay - no network calls, no live computation.
    All events come from the event store.

    Args:
        stream_id: The stream ID to replay.
        event_store: The event store containing the stream.
        speed: Playback speed (0 = instant, 1 = real-time).
        fire_callbacks: Whether to fire callbacks during replay.
        from_seq: Start from this sequence.
        to_seq: Stop at this sequence (None = no limit).

    Returns:
        ReplayResult with stream, state, and metadata.

    Example:
        ```python
        # Record a stream
        store = create_in_memory_event_store()
        recorder = create_event_recorder(store, "my-stream")

        await recorder.record_start({"prompt": "test"})
        await recorder.record_token("Hello", 0)
        await recorder.record_complete("Hello", 1)

        # Replay it
        result = await replay("my-stream", store)

        async for event in result.stream:
            print(event)  # Same events as original
        ```
    """
    import asyncio

    # Verify stream exists
    exists = await event_store.exists(stream_id)
    if not exists:
        raise ValueError(f"Stream not found: {stream_id}")

    # Get all events
    envelopes = await event_store.get_events(stream_id)
    if not envelopes:
        raise ValueError(f"Stream has no events: {stream_id}")

    # Extract original options from START event
    start_event = next(
        (e for e in envelopes if e.event.type == RecordedEventType.START),
        None,
    )
    original_options = start_event.event.data.get("options", {}) if start_event else {}

    # Initialize state
    state = State()
    errors: list[Exception] = []

    # Create result with callbacks holder
    callbacks = ReplayCallbacks()
    abort_requested = False

    async def stream_generator() -> AsyncGenerator[Event, None]:
        nonlocal abort_requested
        last_ts: float | None = None
        max_seq = to_seq if to_seq is not None else float("inf")

        for envelope in envelopes:
            # Check abort
            if abort_requested:
                break

            # Skip events outside range
            if envelope.seq < from_seq:
                continue
            if envelope.seq > max_seq:
                break

            event = envelope.event
            data = event.data

            # Simulate timing if speed > 0
            # Timestamps are in milliseconds, convert to seconds for asyncio.sleep
            if speed > 0 and last_ts is not None:
                delay = (event.ts - last_ts) / 1000 / speed
                if delay > 0:
                    await asyncio.sleep(delay)
            last_ts = event.ts

            # Process each event type
            if event.type == RecordedEventType.START:
                # Nothing to emit, just metadata
                pass

            elif event.type == RecordedEventType.TOKEN:
                value = data.get("value", "")
                state.content += value
                state.token_count = data.get("index", 0) + 1

                token_event = Event(
                    type=EventType.TOKEN,
                    text=value,
                    timestamp=event.ts,
                )

                if fire_callbacks:
                    if callbacks.on_token:
                        callbacks.on_token(value)
                    if callbacks.on_event:
                        callbacks.on_event(token_event)

                yield token_event

            elif event.type == RecordedEventType.CHECKPOINT:
                state.checkpoint = data.get("content", "")

            elif event.type == RecordedEventType.GUARDRAIL:
                result = data.get("result", {})
                violations = result.get("violations", [])
                state.violations.extend(violations)

                if fire_callbacks and callbacks.on_violation:
                    for violation in violations:
                        callbacks.on_violation(violation)

            elif event.type == RecordedEventType.DRIFT:
                result = data.get("result", {})
                if result.get("detected"):
                    state.drift_detected = True

            elif event.type == RecordedEventType.RETRY:
                if data.get("counts_toward_limit"):
                    state.model_retry_count += 1
                else:
                    state.network_retry_count += 1

                if fire_callbacks and callbacks.on_retry:
                    callbacks.on_retry(
                        data.get("attempt", 0),
                        data.get("reason", ""),
                    )

            elif event.type == RecordedEventType.FALLBACK:
                state.fallback_index = data.get("to", 0)

            elif event.type == RecordedEventType.CONTINUATION:
                state.resumed = True
                state.resume_point = data.get("checkpoint", "")

            elif event.type == RecordedEventType.COMPLETE:
                state.completed = True
                state.content = data.get("content", "")
                state.token_count = data.get("token_count", 0)

                complete_event = Event(
                    type=EventType.COMPLETE,
                    timestamp=event.ts,
                )

                if fire_callbacks and callbacks.on_event:
                    callbacks.on_event(complete_event)

                yield complete_event

            elif event.type == RecordedEventType.ERROR:
                error_data = data.get("error", {})
                error = Exception(error_data.get("message", "Unknown error"))
                errors.append(error)

                error_event = Event(
                    type=EventType.ERROR,
                    error=error,
                    timestamp=event.ts,
                )

                if fire_callbacks and callbacks.on_event:
                    callbacks.on_event(error_event)

                yield error_event

    result = ReplayResult(
        stream=stream_generator(),
        state=state,
        errors=errors,
        stream_id=stream_id,
        is_replay=True,
        original_options=original_options,
        _callbacks=callbacks,
    )

    # Patch abort function
    def abort() -> None:
        nonlocal abort_requested
        abort_requested = True

    result.abort = abort  # type: ignore

    # Patch set_callbacks to update the shared callbacks object
    def set_callbacks(cbs: ReplayCallbacks) -> None:
        callbacks.on_token = cbs.on_token
        callbacks.on_violation = cbs.on_violation
        callbacks.on_retry = cbs.on_retry
        callbacks.on_event = cbs.on_event

    result.set_callbacks = set_callbacks  # type: ignore

    return result


async def get_stream_metadata(
    event_store: EventStore,
    stream_id: str,
) -> StreamMetadata | None:
    """Get stream metadata without full replay."""
    exists = await event_store.exists(stream_id)
    if not exists:
        return None

    events = await event_store.get_events(stream_id)
    if not events:
        return None

    start_event = next(
        (e for e in events if e.event.type == RecordedEventType.START),
        None,
    )
    complete_event = next(
        (e for e in events if e.event.type == RecordedEventType.COMPLETE),
        None,
    )
    error_event = next(
        (e for e in events if e.event.type == RecordedEventType.ERROR),
        None,
    )
    token_events = [e for e in events if e.event.type == RecordedEventType.TOKEN]

    return StreamMetadata(
        stream_id=stream_id,
        event_count=len(events),
        token_count=len(token_events),
        start_ts=start_event.event.ts if start_event else events[0].event.ts,
        end_ts=(complete_event or error_event or events[-1]).event.ts,
        completed=complete_event is not None,
        has_error=error_event is not None,
        options=start_event.event.data.get("options", {}) if start_event else {},
    )


def compare_replays(a: State, b: State) -> ReplayComparison:
    """Compare two replay results for equality.

    Useful for testing determinism.
    """
    differences: list[str] = []

    if a.content != b.content:
        differences.append(f'content: "{a.content[:50]}..." vs "{b.content[:50]}..."')
    if a.token_count != b.token_count:
        differences.append(f"token_count: {a.token_count} vs {b.token_count}")
    if a.completed != b.completed:
        differences.append(f"completed: {a.completed} vs {b.completed}")
    if a.model_retry_count != b.model_retry_count:
        differences.append(
            f"model_retry_count: {a.model_retry_count} vs {b.model_retry_count}"
        )
    if a.fallback_index != b.fallback_index:
        differences.append(f"fallback_index: {a.fallback_index} vs {b.fallback_index}")
    if len(a.violations) != len(b.violations):
        differences.append(f"violations: {len(a.violations)} vs {len(b.violations)}")
    if a.drift_detected != b.drift_detected:
        differences.append(f"drift_detected: {a.drift_detected} vs {b.drift_detected}")
    return ReplayComparison(
        identical=len(differences) == 0,
        differences=differences,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────


def create_in_memory_event_store() -> InMemoryEventStore:
    """Create an in-memory event store."""
    return InMemoryEventStore()


def create_event_recorder(
    event_store: EventStore,
    stream_id: str | None = None,
) -> EventRecorder:
    """Create an event recorder."""
    return EventRecorder(event_store, stream_id)


def create_event_replayer(event_store: EventStore) -> EventReplayer:
    """Create an event replayer."""
    return EventReplayer(event_store)


# ─────────────────────────────────────────────────────────────────────────────
# Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class EventSourcing:
    """Scoped API for event sourcing utilities.

    Provides utilities for recording and replaying L0 streams for testing,
    debugging, and deterministic reproduction of executions.

    Usage:
        ```python
        from l0 import EventSourcing

        # Create an in-memory store
        store = EventSourcing.create_store()

        # Create a recorder
        recorder = EventSourcing.create_recorder(store)

        # Record events
        await recorder.record_start({"prompt": "test"})
        await recorder.record_token("Hello", 0)
        await recorder.record_complete("Hello", 1)

        # Replay the stream
        result = await EventSourcing.replay(recorder.stream_id, store)
        async for event in result.stream:
            print(event)

        # Get stream metadata
        meta = await EventSourcing.get_metadata(store, recorder.stream_id)
        ```
    """

    # Re-export types for convenience
    Store = EventStore
    StoreWithSnapshots = EventStoreWithSnapshots
    InMemoryStore = InMemoryEventStore
    Recorder = EventRecorder
    Replayer = EventReplayer
    RecordedEvent = RecordedEvent
    RecordedEventType = RecordedEventType
    EventEnvelope = EventEnvelope
    Snapshot = Snapshot
    SerializedError = SerializedError
    ReplayedState = ReplayedState
    ReplayResult = ReplayResult
    ReplayCallbacks = ReplayCallbacks
    ReplayComparison = ReplayComparison
    StreamMetadata = StreamMetadata

    @staticmethod
    def generate_stream_id() -> str:
        """Generate a unique stream ID."""
        return generate_stream_id()

    @staticmethod
    async def replay(
        stream_id: str,
        event_store: EventStore,
        *,
        speed: float = 0,
        fire_callbacks: bool = True,
        from_seq: int = 0,
        to_seq: int | None = None,
    ) -> ReplayResult:
        """Replay an L0 stream from stored events.

        This is a PURE replay - no network calls, no live computation.
        All events come from the event store.

        Args:
            stream_id: The stream ID to replay.
            event_store: The event store containing the stream.
            speed: Playback speed (0 = instant, 1 = real-time).
            fire_callbacks: Whether to fire callbacks during replay.
            from_seq: Start from this sequence.
            to_seq: Stop at this sequence (None = no limit).

        Returns:
            ReplayResult with stream, state, and metadata.
        """
        return await replay(
            stream_id,
            event_store,
            speed=speed,
            fire_callbacks=fire_callbacks,
            from_seq=from_seq,
            to_seq=to_seq,
        )

    @staticmethod
    async def get_metadata(
        event_store: EventStore,
        stream_id: str,
    ) -> StreamMetadata | None:
        """Get stream metadata without full replay."""
        return await get_stream_metadata(event_store, stream_id)

    @staticmethod
    def compare(a: State, b: State) -> ReplayComparison:
        """Compare two replay results for equality.

        Useful for testing determinism.
        """
        return compare_replays(a, b)

    @staticmethod
    def create_store() -> InMemoryEventStore:
        """Create an in-memory event store."""
        return create_in_memory_event_store()

    @staticmethod
    def create_recorder(
        event_store: EventStore,
        stream_id: str | None = None,
    ) -> EventRecorder:
        """Create an event recorder."""
        return create_event_recorder(event_store, stream_id)

    @staticmethod
    def create_replayer(event_store: EventStore) -> EventReplayer:
        """Create an event replayer."""
        return create_event_replayer(event_store)
