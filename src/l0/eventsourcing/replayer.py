"""Event Replayer for Event Sourcing.

Replays events from a store to reconstruct state or stream tokens.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .types import (
    EventEnvelope,
    RecordedEventType,
    ReplayedState,
    ReplayOptions,
    SerializedOptions,
)

if TYPE_CHECKING:
    from .store import EventStore


@dataclass
class StreamMetadata:
    """Metadata about a recorded stream."""

    stream_id: str
    event_count: int
    token_count: int
    start_ts: float
    end_ts: float
    completed: bool
    has_error: bool
    options: SerializedOptions | None = None


@dataclass
class ReplayComparison:
    """Result of comparing two replays."""

    identical: bool
    differences: list[str]


class EventReplayer:
    """Event replayer - replays events from a store."""

    def __init__(self, event_store: EventStore):
        """Create an event replayer.

        Args:
            event_store: The event store to read from
        """
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
            stream_id: Stream ID to replay
            speed: Playback speed (0 = instant, 1 = real-time)
            from_seq: Start from this sequence
            to_seq: Stop at this sequence

        Yields:
            Event envelopes in order
        """
        events = await self._event_store.get_events(stream_id)
        last_ts: float | None = None

        for envelope in events:
            # Skip events outside range
            if envelope.seq < from_seq:
                continue
            if to_seq is not None and envelope.seq > to_seq:
                break

            # Simulate timing if speed > 0
            if speed > 0 and last_ts is not None:
                delay_ms = (envelope.event.ts - last_ts) / speed
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000)

            last_ts = envelope.event.ts
            yield envelope

    async def replay_to_state(self, stream_id: str) -> ReplayedState:
        """Replay and reconstruct final state.

        Args:
            stream_id: Stream ID to replay

        Returns:
            Reconstructed state from events
        """
        state = ReplayedState()
        events = await self._event_store.get_events(stream_id)

        for envelope in events:
            event = envelope.event

            if event.type == RecordedEventType.START:
                state.start_ts = event.ts

            elif event.type == RecordedEventType.TOKEN:
                state.content += event.value
                state.token_count = event.index + 1

            elif event.type == RecordedEventType.CHECKPOINT:
                state.checkpoint = event.content

            elif event.type == RecordedEventType.GUARDRAIL:
                state.violations.extend(event.result.violations)

            elif event.type == RecordedEventType.DRIFT:
                if event.result.detected:
                    state.drift_detected = True

            elif event.type == RecordedEventType.RETRY:
                if event.counts_toward_limit:
                    state.retry_attempts += 1
                else:
                    state.network_retry_count += 1

            elif event.type == RecordedEventType.FALLBACK:
                state.fallback_index = event.to

            elif event.type == RecordedEventType.CONTINUATION:
                state.content = event.checkpoint
                state.token_count = event.at

            elif event.type == RecordedEventType.COMPLETE:
                state.completed = True
                state.content = event.content
                state.token_count = event.token_count
                state.end_ts = event.ts

            elif event.type == RecordedEventType.ERROR:
                state.error = event.error
                state.end_ts = event.ts

        return state

    async def replay_tokens(
        self,
        stream_id: str,
        *,
        speed: float = 0,
    ) -> AsyncGenerator[str, None]:
        """Get stream as token async iterable (for replay mode).

        Args:
            stream_id: Stream ID to replay
            speed: Playback speed (0 = instant, 1 = real-time)

        Yields:
            Token values
        """
        async for envelope in self.replay(stream_id, speed=speed):
            if envelope.event.type == RecordedEventType.TOKEN:
                yield envelope.event.value


class ReplayResult:
    """Result of a replay operation with callbacks support."""

    def __init__(
        self,
        replayer: EventReplayer,
        stream_id: str,
        options: ReplayOptions,
    ):
        self._replayer = replayer
        self._stream_id = stream_id
        self._options = options
        self._callbacks: dict[str, Callable[..., None]] = {}
        self._state: ReplayedState | None = None

    def set_callbacks(
        self,
        *,
        on_token: Callable[[str], None] | None = None,
        on_violation: Callable[[dict[str, Any]], None] | None = None,
        on_retry: Callable[[int, str], None] | None = None,
        on_event: Callable[[EventEnvelope], None] | None = None,
    ) -> None:
        """Set callbacks for replay events.

        Args:
            on_token: Called for each token
            on_violation: Called for each guardrail violation
            on_retry: Called for each retry (attempt, reason)
            on_event: Called for every event
        """
        if on_token:
            self._callbacks["on_token"] = on_token
        if on_violation:
            self._callbacks["on_violation"] = on_violation
        if on_retry:
            self._callbacks["on_retry"] = on_retry
        if on_event:
            self._callbacks["on_event"] = on_event

    @property
    def state(self) -> ReplayedState:
        """Get the replayed state (after iteration completes)."""
        if self._state is None:
            self._state = ReplayedState()
        return self._state

    async def __aiter__(self) -> AsyncGenerator[EventEnvelope, None]:
        """Iterate over replay events."""
        self._state = ReplayedState()

        async for envelope in self._replayer.replay(
            self._stream_id,
            speed=self._options.speed,
            from_seq=self._options.from_seq,
            to_seq=self._options.to_seq,
        ):
            event = envelope.event

            # Update state
            if event.type == RecordedEventType.START:
                self._state.start_ts = event.ts

            elif event.type == RecordedEventType.TOKEN:
                self._state.content += event.value
                self._state.token_count = event.index + 1
                if self._options.fire_callbacks and "on_token" in self._callbacks:
                    self._callbacks["on_token"](event.value)

            elif event.type == RecordedEventType.CHECKPOINT:
                self._state.checkpoint = event.content

            elif event.type == RecordedEventType.GUARDRAIL:
                self._state.violations.extend(event.result.violations)
                if self._options.fire_callbacks and "on_violation" in self._callbacks:
                    for v in event.result.violations:
                        self._callbacks["on_violation"](v)

            elif event.type == RecordedEventType.DRIFT:
                if event.result.detected:
                    self._state.drift_detected = True

            elif event.type == RecordedEventType.RETRY:
                if event.counts_toward_limit:
                    self._state.retry_attempts += 1
                else:
                    self._state.network_retry_count += 1
                if self._options.fire_callbacks and "on_retry" in self._callbacks:
                    self._callbacks["on_retry"](event.attempt, event.reason)

            elif event.type == RecordedEventType.FALLBACK:
                self._state.fallback_index = event.to

            elif event.type == RecordedEventType.CONTINUATION:
                self._state.content = event.checkpoint
                self._state.token_count = event.at

            elif event.type == RecordedEventType.COMPLETE:
                self._state.completed = True
                self._state.content = event.content
                self._state.token_count = event.token_count
                self._state.end_ts = event.ts

            elif event.type == RecordedEventType.ERROR:
                self._state.error = event.error
                self._state.end_ts = event.ts

            # Fire general event callback
            if self._options.fire_callbacks and "on_event" in self._callbacks:
                self._callbacks["on_event"](envelope)

            yield envelope


async def replay(
    stream_id: str,
    event_store: EventStore,
    *,
    speed: float = 0,
    fire_callbacks: bool = False,
    from_seq: int = 0,
    to_seq: int | None = None,
) -> ReplayResult:
    """Replay a stream from an event store.

    Args:
        stream_id: Stream ID to replay
        event_store: Event store to read from
        speed: Playback speed (0 = instant, 1 = real-time)
        fire_callbacks: Whether to fire monitoring callbacks during replay
        from_seq: Start replay from this sequence number
        to_seq: Stop replay at this sequence number

    Returns:
        ReplayResult with stream and state access
    """
    replayer = EventReplayer(event_store)
    options = ReplayOptions(
        stream_id=stream_id,
        speed=speed,
        fire_callbacks=fire_callbacks,
        from_seq=from_seq,
        to_seq=to_seq,
    )
    return ReplayResult(replayer, stream_id, options)


async def get_stream_metadata(
    event_store: EventStore,
    stream_id: str,
) -> StreamMetadata | None:
    """Get metadata about a recorded stream.

    Args:
        event_store: Event store to read from
        stream_id: Stream ID to get metadata for

    Returns:
        Stream metadata or None if stream doesn't exist
    """
    if not await event_store.exists(stream_id):
        return None

    events = await event_store.get_events(stream_id)
    if not events:
        return None

    token_count = 0
    start_ts = 0.0
    end_ts = 0.0
    completed = False
    has_error = False
    options: SerializedOptions | None = None

    for envelope in events:
        event = envelope.event

        if event.type == RecordedEventType.START:
            start_ts = event.ts
            options = event.options

        elif event.type == RecordedEventType.TOKEN:
            token_count = event.index + 1

        elif event.type == RecordedEventType.COMPLETE:
            completed = True
            token_count = event.token_count
            end_ts = event.ts

        elif event.type == RecordedEventType.ERROR:
            has_error = True
            end_ts = event.ts

    return StreamMetadata(
        stream_id=stream_id,
        event_count=len(events),
        token_count=token_count,
        start_ts=start_ts,
        end_ts=end_ts,
        completed=completed,
        has_error=has_error,
        options=options,
    )


def compare_replays(state1: ReplayedState, state2: ReplayedState) -> ReplayComparison:
    """Compare two replay results.

    Args:
        state1: First replay state
        state2: Second replay state

    Returns:
        Comparison result with differences
    """
    differences: list[str] = []

    if state1.content != state2.content:
        differences.append(
            f"content: '{state1.content[:50]}...' vs '{state2.content[:50]}...'"
        )

    if state1.token_count != state2.token_count:
        differences.append(f"token_count: {state1.token_count} vs {state2.token_count}")

    if state1.checkpoint != state2.checkpoint:
        differences.append(
            f"checkpoint: '{state1.checkpoint}' vs '{state2.checkpoint}'"
        )

    if state1.completed != state2.completed:
        differences.append(f"completed: {state1.completed} vs {state2.completed}")

    if state1.drift_detected != state2.drift_detected:
        differences.append(
            f"drift_detected: {state1.drift_detected} vs {state2.drift_detected}"
        )

    if state1.retry_attempts != state2.retry_attempts:
        differences.append(
            f"retry_attempts: {state1.retry_attempts} vs {state2.retry_attempts}"
        )

    if state1.network_retry_count != state2.network_retry_count:
        differences.append(
            f"network_retry_count: {state1.network_retry_count} vs {state2.network_retry_count}"
        )

    if state1.fallback_index != state2.fallback_index:
        differences.append(
            f"fallback_index: {state1.fallback_index} vs {state2.fallback_index}"
        )

    if state1.violations != state2.violations:
        differences.append(
            f"violations: {state1.violations!r} vs {state2.violations!r}"
        )

    # Compare errors (check both presence and content)
    error1 = state1.error
    error2 = state2.error
    if (error1 is None) != (error2 is None):
        differences.append(f"error: {error1!r} vs {error2!r}")
    elif error1 is not None and error2 is not None:
        if error1.name != error2.name or error1.message != error2.message:
            differences.append(
                f"error: {error1.name}('{error1.message}') vs {error2.name}('{error2.message}')"
            )

    if state1.start_ts != state2.start_ts:
        differences.append(f"start_ts: {state1.start_ts} vs {state2.start_ts}")

    if state1.end_ts != state2.end_ts:
        differences.append(f"end_ts: {state1.end_ts} vs {state2.end_ts}")

    return ReplayComparison(
        identical=len(differences) == 0,
        differences=differences,
    )
