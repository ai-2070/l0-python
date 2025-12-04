"""Scoped Event Sourcing API.

Provides a clean, unified API for all event sourcing operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .adapters import (
    StorageAdapterConfig,
    create_event_store,
    get_registered_adapters,
    register_storage_adapter,
    unregister_storage_adapter,
)
from .recorder import EventRecorder
from .replayer import (
    EventReplayer,
    ReplayComparison,
    ReplayResult,
    StreamMetadata,
    compare_replays,
    get_stream_metadata,
    replay,
)
from .store import (
    CompositeEventStore,
    FileEventStore,
    InMemoryEventStore,
    TTLEventStore,
)
from .types import (
    EventEnvelope,
    RecordedEvent,
    RecordedEventType,
    ReplayedState,
    ReplayOptions,
    Snapshot,
    generate_stream_id,
)

if TYPE_CHECKING:
    from .store import EventStore


class EventSourcing:
    """Scoped API for event sourcing operations.

    Usage:
        from l0 import EventSourcing

        # Create stores
        store = EventSourcing.memory()
        store = EventSourcing.file("./events")

        # Record events
        recorder = EventSourcing.recorder(store)
        await recorder.record_start({"model": "gpt-4"})
        await recorder.record_token("Hello", 0)
        await recorder.record_complete("Hello world", 2)

        # Replay events
        result = await EventSourcing.replay(recorder.stream_id, store)
        async for event in result:
            print(event)

        # Get metadata
        meta = await EventSourcing.metadata(recorder.stream_id, store)

        # Compare replays
        state1 = await EventSourcing.replayer(store).replay_to_state(stream_id_1)
        state2 = await EventSourcing.replayer(store).replay_to_state(stream_id_2)
        comparison = EventSourcing.compare(state1, state2)
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Store Factories
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def memory(prefix: str = "l0", ttl: int = 0) -> InMemoryEventStore:
        """Create an in-memory event store.

        Args:
            prefix: Key prefix
            ttl: TTL in milliseconds (0 = no expiry)

        Returns:
            In-memory event store
        """
        return InMemoryEventStore(prefix=prefix, ttl=ttl)

    @staticmethod
    def file(
        base_path: str | Path = "./l0-events",
        prefix: str = "l0",
        ttl: int = 0,
    ) -> FileEventStore:
        """Create a file-based event store.

        Args:
            base_path: Directory to store events
            prefix: Key prefix
            ttl: TTL in milliseconds (0 = no expiry)

        Returns:
            File-based event store
        """
        return FileEventStore(base_path=base_path, prefix=prefix, ttl=ttl)

    @staticmethod
    def composite(
        stores: list["EventStore"],
        primary_index: int = 0,
    ) -> CompositeEventStore:
        """Create a composite event store.

        Writes to all stores, reads from primary.

        Args:
            stores: List of stores to write to
            primary_index: Index of primary store for reads

        Returns:
            Composite event store
        """
        return CompositeEventStore(stores, primary_index)

    @staticmethod
    def with_ttl(store: "EventStore", ttl_ms: int) -> TTLEventStore:
        """Wrap a store with TTL expiration.

        Args:
            store: Underlying store
            ttl_ms: TTL in milliseconds

        Returns:
            TTL-wrapped store
        """
        return TTLEventStore(store, ttl_ms)

    @staticmethod
    async def create(config: StorageAdapterConfig) -> "EventStore":
        """Create a store using a registered adapter.

        Args:
            config: Storage adapter configuration

        Returns:
            Event store instance
        """
        return await create_event_store(config)

    # ─────────────────────────────────────────────────────────────────────────
    # Recorder & Replayer
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def recorder(
        store: "EventStore",
        stream_id: str | None = None,
    ) -> EventRecorder:
        """Create an event recorder.

        Args:
            store: Event store to write to
            stream_id: Custom stream ID (auto-generated if not provided)

        Returns:
            Event recorder instance
        """
        return EventRecorder(store, stream_id)

    @staticmethod
    def replayer(store: "EventStore") -> EventReplayer:
        """Create an event replayer.

        Args:
            store: Event store to read from

        Returns:
            Event replayer instance
        """
        return EventReplayer(store)

    # ─────────────────────────────────────────────────────────────────────────
    # Replay Functions
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def replay(
        stream_id: str,
        store: "EventStore",
        *,
        speed: float = 0,
        fire_callbacks: bool = False,
        from_seq: int = 0,
        to_seq: int | None = None,
    ) -> ReplayResult:
        """Replay a stream from an event store.

        Args:
            stream_id: Stream ID to replay
            store: Event store to read from
            speed: Playback speed (0 = instant, 1 = real-time)
            fire_callbacks: Whether to fire monitoring callbacks
            from_seq: Start from this sequence
            to_seq: Stop at this sequence

        Returns:
            Replay result with async iteration support
        """
        return await replay(
            stream_id,
            store,
            speed=speed,
            fire_callbacks=fire_callbacks,
            from_seq=from_seq,
            to_seq=to_seq,
        )

    @staticmethod
    async def metadata(
        stream_id: str,
        store: "EventStore",
    ) -> StreamMetadata | None:
        """Get metadata about a recorded stream.

        Args:
            stream_id: Stream ID
            store: Event store to read from

        Returns:
            Stream metadata or None if not found
        """
        return await get_stream_metadata(store, stream_id)

    @staticmethod
    def compare(
        state1: ReplayedState,
        state2: ReplayedState,
    ) -> ReplayComparison:
        """Compare two replay states.

        Args:
            state1: First replay state
            state2: Second replay state

        Returns:
            Comparison result
        """
        return compare_replays(state1, state2)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def generate_id() -> str:
        """Generate a unique stream ID.

        Returns:
            Unique stream ID
        """
        return generate_stream_id()

    # ─────────────────────────────────────────────────────────────────────────
    # Adapter Registry
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def register_adapter(
        adapter_type: str,
        factory: "StorageAdapterFactory",
    ) -> None:
        """Register a custom storage adapter.

        Args:
            adapter_type: Adapter type name
            factory: Factory function
        """
        register_storage_adapter(adapter_type, factory)

    @staticmethod
    def unregister_adapter(adapter_type: str) -> bool:
        """Unregister a storage adapter.

        Args:
            adapter_type: Adapter type name

        Returns:
            True if removed, False if not found
        """
        return unregister_storage_adapter(adapter_type)

    @staticmethod
    def list_adapters() -> list[str]:
        """List registered adapter types.

        Returns:
            List of adapter type names
        """
        return get_registered_adapters()

    # ─────────────────────────────────────────────────────────────────────────
    # Type Aliases (for convenience)
    # ─────────────────────────────────────────────────────────────────────────

    # Event types
    Event = RecordedEvent
    EventType = RecordedEventType
    Envelope = EventEnvelope

    # State types
    State = ReplayedState
    Snapshot = Snapshot

    # Result types
    Metadata = StreamMetadata
    Comparison = ReplayComparison

    # Config
    Config = StorageAdapterConfig


# Import for type annotation
from .adapters import StorageAdapterFactory  # noqa: E402
