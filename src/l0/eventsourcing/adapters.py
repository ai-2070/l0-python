"""Storage Adapters for Event Sourcing.

Provides pluggable storage backends for event persistence.
Implement EventStore interface to create custom adapters.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .store import EventStore


@dataclass
class StorageAdapterConfig:
    """Storage adapter configuration."""

    type: str
    connection: str | None = None
    prefix: str = "l0"
    ttl: int = 0  # TTL for events in milliseconds (0 = no expiry)
    options: dict[str, Any] = field(default_factory=dict)


# Factory function type
StorageAdapterFactory = Callable[
    [StorageAdapterConfig],
    "EventStore | Awaitable[EventStore]",
]


# Registry of storage adapter factories
_adapter_registry: dict[str, StorageAdapterFactory] = {}


def register_storage_adapter(
    adapter_type: str,
    factory: StorageAdapterFactory,
) -> None:
    """Register a custom storage adapter factory.

    Example:
        ```python
        def create_redis_store(config):
            return RedisEventStore(config.connection, config.options)

        register_storage_adapter("redis", create_redis_store)

        store = await create_event_store(StorageAdapterConfig(
            type="redis",
            connection="redis://localhost"
        ))
        ```
    """
    _adapter_registry[adapter_type] = factory


def unregister_storage_adapter(adapter_type: str) -> bool:
    """Unregister a storage adapter.

    Returns:
        True if adapter was found and removed, False otherwise
    """
    if adapter_type in _adapter_registry:
        del _adapter_registry[adapter_type]
        return True
    return False


def get_registered_adapters() -> list[str]:
    """Get list of registered adapter types."""
    return list(_adapter_registry.keys())


async def create_event_store(config: StorageAdapterConfig) -> "EventStore":
    """Create an event store using a registered adapter.

    Example:
        ```python
        # Use built-in memory adapter
        mem_store = await create_event_store(StorageAdapterConfig(type="memory"))

        # Use built-in file adapter
        file_store = await create_event_store(StorageAdapterConfig(
            type="file",
            connection="./events",
            prefix="l0_events",
            ttl=7 * 24 * 60 * 60 * 1000,  # 7 days
        ))
        ```
    """
    factory = _adapter_registry.get(config.type)

    if factory is None:
        available = ", ".join(get_registered_adapters()) or "none"
        raise ValueError(
            f'Unknown storage adapter type: "{config.type}". '
            f"Available adapters: {available}"
        )

    result = factory(config)

    # Handle async factories
    if inspect.isawaitable(result):
        return await result

    # Result is already an EventStore
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Register Built-in Adapters
# ─────────────────────────────────────────────────────────────────────────────


def _create_memory_store(config: StorageAdapterConfig) -> "EventStore":
    """Create an in-memory event store."""
    from .store import InMemoryEventStore

    return InMemoryEventStore(prefix=config.prefix, ttl=config.ttl)


def _create_file_store(config: StorageAdapterConfig) -> "EventStore":
    """Create a file-based event store."""
    from .store import FileEventStore

    return FileEventStore(
        base_path=config.connection or "./l0-events",
        prefix=config.prefix,
        ttl=config.ttl,
    )


# Register built-in adapters
register_storage_adapter("memory", _create_memory_store)
register_storage_adapter("file", _create_file_store)
