"""Event Store implementations for Event Sourcing.

Provides in-memory and extensible storage for atomic, replayable events.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, runtime_checkable

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt

    def _lock_file(f: IO[Any]) -> None:
        """Acquire exclusive lock on file (Windows)."""
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_file(f: IO[Any]) -> None:
        """Release lock on file (Windows)."""
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_file(f: IO[Any]) -> None:
        """Acquire exclusive lock on file (Unix)."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(f: IO[Any]) -> None:
        """Release lock on file (Unix)."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


from .types import (
    EventEnvelope,
    RecordedEvent,
    Snapshot,
    now_ms,
)

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class CompositeStoreError(Exception):
    """Raised when a CompositeEventStore operation partially fails.

    Attributes:
        succeeded_indices: Indices of stores that succeeded.
        failed_indices: Indices of stores that failed.
        errors: The exceptions from failed stores.
    """

    def __init__(
        self,
        message: str,
        succeeded_indices: list[int],
        failed_indices: list[int],
        errors: list[Exception],
    ):
        super().__init__(message)
        self.succeeded_indices = succeeded_indices
        self.failed_indices = failed_indices
        self.errors = errors


# ─────────────────────────────────────────────────────────────────────────────
# Event Store Protocol
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class EventStore(Protocol):
    """Event store interface for persistence."""

    async def append(self, stream_id: str, event: RecordedEvent) -> None:
        """Append an event to a stream."""
        ...

    async def get_events(self, stream_id: str) -> list[EventEnvelope]:
        """Get all events for a stream in order."""
        ...

    async def exists(self, stream_id: str) -> bool:
        """Check if a stream exists."""
        ...

    async def get_last_event(self, stream_id: str) -> EventEnvelope | None:
        """Get the last event for a stream."""
        ...

    async def get_events_after(
        self, stream_id: str, after_seq: int
    ) -> list[EventEnvelope]:
        """Get events after a sequence number (for resumption)."""
        ...

    async def delete(self, stream_id: str) -> None:
        """Delete all events for a stream."""
        ...

    async def list_streams(self) -> list[str]:
        """List all stream IDs."""
        ...


@runtime_checkable
class EventStoreWithSnapshots(EventStore, Protocol):
    """Extended event store with snapshot support."""

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a snapshot."""
        ...

    async def get_snapshot(self, stream_id: str) -> Snapshot | None:
        """Get the latest snapshot for a stream."""
        ...

    async def get_snapshot_before(self, stream_id: str, seq: int) -> Snapshot | None:
        """Get snapshot closest to but not after a sequence number."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Base Event Store
# ─────────────────────────────────────────────────────────────────────────────


class BaseEventStore(ABC):
    """Base class for implementing custom storage adapters.

    Provides default implementations that can be overridden.
    """

    def __init__(self, prefix: str = "l0", ttl: int = 0):
        """Initialize base event store.

        Args:
            prefix: Table/collection/key prefix
            ttl: TTL for events in milliseconds (0 = no expiry)
        """
        self.prefix = prefix
        self.ttl = ttl

    def get_stream_key(self, stream_id: str) -> str:
        """Get the storage key for a stream."""
        return f"{self.prefix}:stream:{stream_id}"

    def get_meta_key(self, stream_id: str) -> str:
        """Get the storage key for stream metadata."""
        return f"{self.prefix}:meta:{stream_id}"

    def is_expired(self, timestamp: float) -> bool:
        """Check if an event has expired based on TTL."""
        if self.ttl == 0:
            return False
        return now_ms() - timestamp > self.ttl

    @abstractmethod
    async def append(self, stream_id: str, event: RecordedEvent) -> None:
        """Append an event to a stream."""
        ...

    @abstractmethod
    async def get_events(self, stream_id: str) -> list[EventEnvelope]:
        """Get all events for a stream in order."""
        ...

    @abstractmethod
    async def exists(self, stream_id: str) -> bool:
        """Check if a stream exists."""
        ...

    @abstractmethod
    async def delete(self, stream_id: str) -> None:
        """Delete all events for a stream."""
        ...

    @abstractmethod
    async def list_streams(self) -> list[str]:
        """List all stream IDs."""
        ...

    async def get_last_event(self, stream_id: str) -> EventEnvelope | None:
        """Get the last event for a stream."""
        events = await self.get_events(stream_id)
        return events[-1] if events else None

    async def get_events_after(
        self, stream_id: str, after_seq: int
    ) -> list[EventEnvelope]:
        """Get events after a sequence number."""
        events = await self.get_events(stream_id)
        return [e for e in events if e.seq > after_seq]


class BaseEventStoreWithSnapshots(BaseEventStore, ABC):
    """Base class for storage adapters with snapshot support."""

    def get_snapshot_key(self, stream_id: str) -> str:
        """Get the storage key for snapshots."""
        return f"{self.prefix}:snapshot:{stream_id}"

    @abstractmethod
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a snapshot."""
        ...

    @abstractmethod
    async def get_snapshot(self, stream_id: str) -> Snapshot | None:
        """Get the latest snapshot for a stream."""
        ...

    async def get_snapshot_before(self, stream_id: str, seq: int) -> Snapshot | None:
        """Get snapshot closest to but not after a sequence number."""
        snapshot = await self.get_snapshot(stream_id)
        if snapshot and snapshot.seq <= seq:
            return snapshot
        return None


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Event Store
# ─────────────────────────────────────────────────────────────────────────────


class InMemoryEventStore(BaseEventStoreWithSnapshots):
    """In-memory event store for testing and short-lived sessions.

    Not suitable for production persistence - events are lost on process exit.
    Use for:
    - Unit/integration testing with record/replay
    - Development debugging
    - Short-lived serverless functions
    """

    def __init__(self, prefix: str = "l0", ttl: int = 0):
        super().__init__(prefix, ttl)
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
        """Get all events for a stream in order."""
        events = self._streams.get(stream_id, [])

        # Filter expired events if TTL is set
        if self.ttl > 0:
            return [e for e in events if not self.is_expired(e.event.ts)]

        return list(events)

    async def exists(self, stream_id: str) -> bool:
        """Check if a stream exists (with unexpired events if TTL is set)."""
        events = await self.get_events(stream_id)
        return len(events) > 0

    async def delete(self, stream_id: str) -> None:
        """Delete all events for a stream."""
        self._streams.pop(stream_id, None)
        self._snapshots.pop(stream_id, None)

    async def list_streams(self) -> list[str]:
        """List all stream IDs."""
        return list(self._streams.keys())

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a snapshot."""
        if snapshot.stream_id not in self._snapshots:
            self._snapshots[snapshot.stream_id] = []
        self._snapshots[snapshot.stream_id].append(snapshot)

    async def get_snapshot(self, stream_id: str) -> Snapshot | None:
        """Get the latest snapshot for a stream."""
        snapshots = self._snapshots.get(stream_id, [])
        return snapshots[-1] if snapshots else None

    async def get_snapshot_before(self, stream_id: str, seq: int) -> Snapshot | None:
        """Get snapshot closest to but not after a sequence number."""
        snapshots = self._snapshots.get(stream_id, [])
        if not snapshots:
            return None

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
# File Event Store
# ─────────────────────────────────────────────────────────────────────────────


def _event_to_dict(event: RecordedEvent) -> dict[str, Any]:
    """Convert a RecordedEvent to a dictionary for JSON serialization."""
    from dataclasses import asdict

    result = asdict(event)
    # Convert enum to string
    result["type"] = event.type.value
    return result


def _dict_to_event(data: dict[str, Any]) -> RecordedEvent:
    """Convert a dictionary back to a RecordedEvent."""
    from .types import (
        CheckpointEvent,
        CompleteEvent,
        ContinuationEvent,
        DriftEvent,
        DriftEventResult,
        ErrorEvent,
        FallbackEvent,
        GuardrailEvent,
        GuardrailEventResult,
        RecordedEventType,
        RetryEvent,
        SerializedError,
        SerializedOptions,
        StartEvent,
        TokenEvent,
    )

    event_type = data["type"]

    if event_type == RecordedEventType.START.value:
        opts_data = data.get("options", {})
        options = (
            SerializedOptions(**opts_data) if isinstance(opts_data, dict) else opts_data
        )
        return StartEvent(ts=data["ts"], options=options)

    elif event_type == RecordedEventType.TOKEN.value:
        return TokenEvent(ts=data["ts"], value=data["value"], index=data["index"])

    elif event_type == RecordedEventType.CHECKPOINT.value:
        return CheckpointEvent(ts=data["ts"], at=data["at"], content=data["content"])

    elif event_type == RecordedEventType.GUARDRAIL.value:
        result_data = data.get("result", {})
        result = (
            GuardrailEventResult(**result_data)
            if isinstance(result_data, dict)
            else result_data
        )
        return GuardrailEvent(ts=data["ts"], at=data["at"], result=result)

    elif event_type == RecordedEventType.DRIFT.value:
        result_data = data.get("result", {})
        result = (
            DriftEventResult(**result_data)
            if isinstance(result_data, dict)
            else result_data
        )
        return DriftEvent(ts=data["ts"], at=data["at"], result=result)

    elif event_type == RecordedEventType.RETRY.value:
        return RetryEvent(
            ts=data["ts"],
            reason=data["reason"],
            attempt=data["attempt"],
            counts_toward_limit=data.get("counts_toward_limit", True),
        )

    elif event_type == RecordedEventType.FALLBACK.value:
        return FallbackEvent(ts=data["ts"], to=data["to"])

    elif event_type == RecordedEventType.CONTINUATION.value:
        return ContinuationEvent(
            ts=data["ts"], checkpoint=data["checkpoint"], at=data["at"]
        )

    elif event_type == RecordedEventType.COMPLETE.value:
        return CompleteEvent(
            ts=data["ts"], content=data["content"], token_count=data["token_count"]
        )

    elif event_type == RecordedEventType.ERROR.value:
        error_data = data.get("error", {})
        error = (
            SerializedError(**error_data)
            if isinstance(error_data, dict)
            else error_data
        )
        return ErrorEvent(
            ts=data["ts"],
            error=error,
            failure_type=data.get("failure_type", ""),
            recovery_strategy=data.get("recovery_strategy", ""),
            policy=data.get("policy", ""),
        )

    else:
        raise ValueError(f"Unknown event type: {event_type}")


def _envelope_to_dict(envelope: EventEnvelope) -> dict[str, Any]:
    """Convert an EventEnvelope to a dictionary."""
    return {
        "stream_id": envelope.stream_id,
        "seq": envelope.seq,
        "event": _event_to_dict(envelope.event),
    }


def _dict_to_envelope(data: dict[str, Any]) -> EventEnvelope:
    """Convert a dictionary back to an EventEnvelope."""
    return EventEnvelope(
        stream_id=data["stream_id"],
        seq=data["seq"],
        event=_dict_to_event(data["event"]),
    )


def _snapshot_to_dict(snapshot: Snapshot) -> dict[str, Any]:
    """Convert a Snapshot to a dictionary."""
    from dataclasses import asdict

    return asdict(snapshot)


def _dict_to_snapshot(data: dict[str, Any]) -> Snapshot:
    """Convert a dictionary back to a Snapshot."""
    return Snapshot(**data)


class FileEventStore(BaseEventStoreWithSnapshots):
    """File-based event store for local persistence.

    Stores events as JSON files.
    """

    # Pattern for validating stream IDs
    _VALID_STREAM_ID = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __init__(
        self,
        base_path: str | Path = "./l0-events",
        prefix: str = "l0",
        ttl: int = 0,
    ):
        super().__init__(prefix, ttl)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_stream_id(cls, stream_id: str) -> str:
        """Validate stream ID to prevent path traversal attacks.

        Only allows alphanumeric characters, hyphens, and underscores.

        Raises:
            ValueError: If stream ID contains invalid characters
        """
        if not stream_id:
            raise ValueError("Invalid stream ID: must not be empty")
        if not cls._VALID_STREAM_ID.match(stream_id):
            raise ValueError(
                "Invalid stream ID: only alphanumeric characters, "
                "hyphens, and underscores are allowed"
            )
        return stream_id

    def _get_file_path(self, stream_id: str) -> Path:
        """Get file path for a stream."""
        safe_id = self.validate_stream_id(stream_id)
        return self.base_path / f"{safe_id}.json"

    def _get_snapshot_file_path(self, stream_id: str) -> Path:
        """Get file path for a stream's snapshot."""
        safe_id = self.validate_stream_id(stream_id)
        return self.base_path / f"{safe_id}.snapshot.json"

    async def append(self, stream_id: str, event: RecordedEvent) -> None:
        """Append an event to a stream.

        Uses file locking and atomic writes to prevent data loss
        from concurrent appends.
        """
        file_path = self._get_file_path(stream_id)
        lock_path = file_path.with_suffix(".lock")

        # Ensure lock file exists with at least 1 byte (required for Windows msvcrt.locking)
        if not lock_path.exists() or lock_path.stat().st_size == 0:
            lock_path.write_bytes(b"\x00")

        with open(lock_path, "r+b") as lock_file:
            # Acquire exclusive lock
            _lock_file(lock_file)
            try:
                # Read existing events
                events: list[dict[str, Any]] = []
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")
                    events = json.loads(content)

                envelope = EventEnvelope(
                    stream_id=stream_id,
                    seq=len(events),
                    event=event,
                )
                events.append(_envelope_to_dict(envelope))

                # Atomic write: write to temp file, then rename
                fd, tmp_path = tempfile.mkstemp(
                    dir=self.base_path, suffix=".tmp", prefix=stream_id
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                        json.dump(events, tmp_file, indent=2)
                    # Atomic rename
                    os.replace(tmp_path, file_path)
                except Exception:
                    # Clean up temp file on error
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
            finally:
                # Release lock
                _unlock_file(lock_file)

    async def get_events(self, stream_id: str) -> list[EventEnvelope]:
        """Get all events for a stream in order."""
        file_path = self._get_file_path(stream_id)

        if not file_path.exists():
            return []

        content = file_path.read_text(encoding="utf-8")
        events_data = json.loads(content)
        events = [_dict_to_envelope(e) for e in events_data]

        # Filter expired events if TTL is set
        if self.ttl > 0:
            return [e for e in events if not self.is_expired(e.event.ts)]

        return events

    async def exists(self, stream_id: str) -> bool:
        """Check if a stream exists."""
        file_path = self._get_file_path(stream_id)
        return file_path.exists()

    async def delete(self, stream_id: str) -> None:
        """Delete all events for a stream."""
        file_path = self._get_file_path(stream_id)
        snapshot_path = self._get_snapshot_file_path(stream_id)

        if file_path.exists():
            file_path.unlink()
        if snapshot_path.exists():
            snapshot_path.unlink()

    async def list_streams(self) -> list[str]:
        """List all stream IDs."""
        streams = []
        for path in self.base_path.glob("*.json"):
            if not path.name.endswith(".snapshot.json"):
                streams.append(path.stem)
        return streams

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a snapshot with atomic write and locking."""
        file_path = self._get_snapshot_file_path(snapshot.stream_id)
        lock_path = file_path.with_suffix(".snapshot.lock")

        # Ensure lock file exists with at least 1 byte (required for Windows msvcrt.locking)
        if not lock_path.exists() or lock_path.stat().st_size == 0:
            lock_path.write_bytes(b"\x00")

        with open(lock_path, "r+b") as lock_file:
            _lock_file(lock_file)
            try:
                # Write to temp file first
                temp_path = file_path.with_suffix(".snapshot.tmp")
                temp_path.write_text(
                    json.dumps(_snapshot_to_dict(snapshot), indent=2),
                    encoding="utf-8",
                )
                # Atomic replace
                os.replace(temp_path, file_path)
            finally:
                _unlock_file(lock_file)

    async def get_snapshot(self, stream_id: str) -> Snapshot | None:
        """Get the latest snapshot for a stream."""
        file_path = self._get_snapshot_file_path(stream_id)

        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8")
        return _dict_to_snapshot(json.loads(content))


# ─────────────────────────────────────────────────────────────────────────────
# Composite Event Store
# ─────────────────────────────────────────────────────────────────────────────


class CompositeEventStore:
    """Composite event store that writes to multiple backends.

    Useful for write-through caching or redundancy.
    """

    def __init__(self, stores: list[EventStore], primary_index: int = 0):
        """Create a composite event store.

        Args:
            stores: Array of event stores to write to
            primary_index: Index of the primary store for reads (default: 0)
        """
        if not stores:
            raise ValueError("CompositeEventStore requires at least one store")
        self._stores = stores
        self._primary_index = primary_index

    @property
    def _primary(self) -> EventStore:
        return self._stores[self._primary_index]

    async def append(self, stream_id: str, event: RecordedEvent) -> None:
        """Write to all stores.

        Raises:
            CompositeStoreError: If any store fails, with details of successes/failures.
        """
        import asyncio

        results = await asyncio.gather(
            *[store.append(stream_id, event) for store in self._stores],
            return_exceptions=True,
        )

        # Check for failures
        failures: list[tuple[int, Exception]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append((i, result))

        if failures:
            succeeded = [
                i for i, r in enumerate(results) if not isinstance(r, Exception)
            ]
            raise CompositeStoreError(
                f"Partial failure writing to CompositeEventStore: "
                f"{len(failures)}/{len(self._stores)} stores failed",
                succeeded_indices=succeeded,
                failed_indices=[i for i, _ in failures],
                errors=[e for _, e in failures],
            )

    async def get_events(self, stream_id: str) -> list[EventEnvelope]:
        """Read from primary only."""
        return await self._primary.get_events(stream_id)

    async def exists(self, stream_id: str) -> bool:
        return await self._primary.exists(stream_id)

    async def get_last_event(self, stream_id: str) -> EventEnvelope | None:
        return await self._primary.get_last_event(stream_id)

    async def get_events_after(
        self, stream_id: str, after_seq: int
    ) -> list[EventEnvelope]:
        return await self._primary.get_events_after(stream_id, after_seq)

    async def delete(self, stream_id: str) -> None:
        """Delete from all stores.

        Raises:
            CompositeStoreError: If any store fails, with details of successes/failures.
        """
        import asyncio

        results = await asyncio.gather(
            *[store.delete(stream_id) for store in self._stores],
            return_exceptions=True,
        )

        # Check for failures
        failures: list[tuple[int, Exception]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append((i, result))

        if failures:
            succeeded = [
                i for i, r in enumerate(results) if not isinstance(r, Exception)
            ]
            raise CompositeStoreError(
                f"Partial failure deleting from CompositeEventStore: "
                f"{len(failures)}/{len(self._stores)} stores failed",
                succeeded_indices=succeeded,
                failed_indices=[i for i, _ in failures],
                errors=[e for _, e in failures],
            )

    async def list_streams(self) -> list[str]:
        return await self._primary.list_streams()


# ─────────────────────────────────────────────────────────────────────────────
# TTL Event Store Wrapper
# ─────────────────────────────────────────────────────────────────────────────


class TTLEventStore:
    """Wrapper that adds TTL expiration to any event store."""

    def __init__(self, store: EventStore, ttl_ms: int):
        """Create a TTL wrapper.

        Args:
            store: The underlying event store
            ttl_ms: TTL in milliseconds
        """
        self._store = store
        self._ttl = ttl_ms

    def _is_expired(self, timestamp: float) -> bool:
        return now_ms() - timestamp > self._ttl

    def _filter_expired(self, events: list[EventEnvelope]) -> list[EventEnvelope]:
        return [e for e in events if not self._is_expired(e.event.ts)]

    async def append(self, stream_id: str, event: RecordedEvent) -> None:
        return await self._store.append(stream_id, event)

    async def get_events(self, stream_id: str) -> list[EventEnvelope]:
        events = await self._store.get_events(stream_id)
        return self._filter_expired(events)

    async def exists(self, stream_id: str) -> bool:
        events = await self.get_events(stream_id)
        return len(events) > 0

    async def get_last_event(self, stream_id: str) -> EventEnvelope | None:
        events = await self.get_events(stream_id)
        return events[-1] if events else None

    async def get_events_after(
        self, stream_id: str, after_seq: int
    ) -> list[EventEnvelope]:
        events = await self.get_events(stream_id)
        return [e for e in events if e.seq > after_seq]

    async def delete(self, stream_id: str) -> None:
        return await self._store.delete(stream_id)

    async def list_streams(self) -> list[str]:
        return await self._store.list_streams()
