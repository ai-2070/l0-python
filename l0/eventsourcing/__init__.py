"""L0 Event Sourcing - Atomic, Replayable Operations.

The key insight: Replayability MUST ignore external sources of non-determinism.
In replay mode, we're a pure faucet over stored events - no network, no retries,
no timeouts, no fallbacks, no live guardrail evaluation.

Derived computations (guardrails, drift, retries) are stored AS events,
not recomputed on replay.
"""

from .adapters import (
    StorageAdapterConfig,
    create_event_store,
    get_registered_adapters,
    register_storage_adapter,
    unregister_storage_adapter,
)
from .recorder import EventRecorder
from .replayer import EventReplayer, compare_replays, get_stream_metadata, replay
from .sourcing import EventSourcing
from .store import (
    BaseEventStore,
    BaseEventStoreWithSnapshots,
    CompositeEventStore,
    CompositeStoreError,
    EventStore,
    EventStoreWithSnapshots,
    FileEventStore,
    InMemoryEventStore,
    TTLEventStore,
)
from .types import (
    CheckpointEvent,
    CompleteEvent,
    ContinuationEvent,
    DriftEvent,
    DriftEventResult,
    ErrorEvent,
    # Supporting types
    EventEnvelope,
    FallbackEvent,
    GuardrailEvent,
    GuardrailEventResult,
    # Event types
    RecordedEvent,
    RecordedEventType,
    RecordOptions,
    ReplayedState,
    ReplayOptions,
    RetryEvent,
    SerializedError,
    SerializedOptions,
    Snapshot,
    StartEvent,
    TokenEvent,
    deserialize_error,
    generate_stream_id,
    # Utilities
    serialize_error,
)

__all__ = [
    # Main scoped API
    "EventSourcing",
    # Event types
    "RecordedEvent",
    "RecordedEventType",
    "StartEvent",
    "TokenEvent",
    "CheckpointEvent",
    "GuardrailEvent",
    "DriftEvent",
    "RetryEvent",
    "FallbackEvent",
    "ContinuationEvent",
    "CompleteEvent",
    "ErrorEvent",
    # Supporting types
    "EventEnvelope",
    "SerializedOptions",
    "SerializedError",
    "GuardrailEventResult",
    "DriftEventResult",
    "Snapshot",
    "ReplayedState",
    "ReplayOptions",
    "RecordOptions",
    # Utilities
    "serialize_error",
    "deserialize_error",
    "generate_stream_id",
    # Store
    "EventStore",
    "EventStoreWithSnapshots",
    "InMemoryEventStore",
    "FileEventStore",
    "CompositeEventStore",
    "CompositeStoreError",
    "TTLEventStore",
    "BaseEventStore",
    "BaseEventStoreWithSnapshots",
    # Recorder
    "EventRecorder",
    # Replayer
    "EventReplayer",
    "replay",
    "get_stream_metadata",
    "compare_replays",
    # Adapters
    "StorageAdapterConfig",
    "register_storage_adapter",
    "unregister_storage_adapter",
    "get_registered_adapters",
    "create_event_store",
]
