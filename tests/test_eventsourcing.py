"""Tests for src.l0.eventsourcing module."""

import shutil
import tempfile
from collections.abc import Generator
from typing import cast

import pytest

from l0.eventsourcing import (
    EventRecorder,
    EventReplayer,
    EventSourcing,
    FileEventStore,
    InMemoryEventStore,
    RecordedEventType,
    StorageAdapterConfig,
    deserialize_error,
    generate_stream_id,
    serialize_error,
)
from l0.eventsourcing.types import CompleteEvent, TokenEvent


class TestGenerateStreamId:
    def test_generates_unique_ids(self):
        id1 = generate_stream_id()
        id2 = generate_stream_id()
        assert id1 != id2

    def test_generates_valid_format(self):
        stream_id = generate_stream_id()
        assert isinstance(stream_id, str)
        assert len(stream_id) > 0


class TestSerializeError:
    def test_serialize_basic_error(self):
        error = ValueError("test error")
        serialized = serialize_error(error)
        assert serialized.name == "ValueError"
        assert serialized.message == "test error"

    def test_deserialize_error(self):
        error = ValueError("test error")
        serialized = serialize_error(error)
        deserialized = deserialize_error(serialized)
        assert str(deserialized) == "test error"

    def test_deserialize_error_preserves_stack(self):
        """Test that deserialized errors preserve the stored stack trace."""
        captured_error = None
        try:
            raise ValueError("test with traceback")
        except ValueError as e:
            captured_error = e

        serialized = serialize_error(captured_error)
        deserialized = deserialize_error(serialized)

        assert hasattr(deserialized, "stack")
        assert deserialized.stack == serialized.stack  # type: ignore[attr-defined]
        assert "ValueError" in deserialized.stack  # type: ignore[attr-defined]
        assert "test with traceback" in deserialized.stack  # type: ignore[attr-defined]

    def test_serialize_error_captures_stack_outside_except_block(self):
        """Test that stack trace is captured from error, not ambient state."""
        # Capture an error with a real traceback
        captured_error = None
        try:
            raise ValueError("test with traceback")
        except ValueError as e:
            captured_error = e

        # Serialize OUTSIDE the except block (ambient exc_info is now None)
        serialized = serialize_error(captured_error)

        # Stack should contain the actual traceback, not 'NoneType: None'
        assert serialized.stack is not None
        assert "NoneType" not in serialized.stack
        assert "ValueError" in serialized.stack
        assert "test with traceback" in serialized.stack


class TestInMemoryEventStore:
    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_append_and_get_events(self, store: InMemoryEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        stream_id = "test-stream"
        event = TokenEvent(ts=now_ms(), value="Hello", index=0)

        await store.append(stream_id, event)
        events = await store.get_events(stream_id)

        assert len(events) == 1
        assert events[0].stream_id == stream_id
        assert events[0].seq == 0
        assert cast(TokenEvent, events[0].event).value == "Hello"

    @pytest.mark.asyncio
    async def test_exists(self, store: InMemoryEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        stream_id = "test-stream"
        assert await store.exists(stream_id) is False

        event = TokenEvent(ts=now_ms(), value="Hello", index=0)
        await store.append(stream_id, event)
        assert await store.exists(stream_id) is True

    @pytest.mark.asyncio
    async def test_delete(self, store: InMemoryEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        stream_id = "test-stream"
        event = TokenEvent(ts=now_ms(), value="Hello", index=0)
        await store.append(stream_id, event)

        await store.delete(stream_id)
        assert await store.exists(stream_id) is False

    @pytest.mark.asyncio
    async def test_list_streams(self, store: InMemoryEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        event = TokenEvent(ts=now_ms(), value="Hello", index=0)
        await store.append("stream-1", event)
        await store.append("stream-2", event)

        streams = await store.list_streams()
        assert "stream-1" in streams
        assert "stream-2" in streams

    @pytest.mark.asyncio
    async def test_get_last_event(self, store: InMemoryEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        stream_id = "test-stream"
        await store.append(stream_id, TokenEvent(ts=now_ms(), value="A", index=0))
        await store.append(stream_id, TokenEvent(ts=now_ms(), value="B", index=1))

        last = await store.get_last_event(stream_id)
        assert last is not None
        assert cast(TokenEvent, last.event).value == "B"

    @pytest.mark.asyncio
    async def test_get_events_after(self, store: InMemoryEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        stream_id = "test-stream"
        await store.append(stream_id, TokenEvent(ts=now_ms(), value="A", index=0))
        await store.append(stream_id, TokenEvent(ts=now_ms(), value="B", index=1))
        await store.append(stream_id, TokenEvent(ts=now_ms(), value="C", index=2))

        events = await store.get_events_after(stream_id, 0)
        assert len(events) == 2
        assert cast(TokenEvent, events[0].event).value == "B"
        assert cast(TokenEvent, events[1].event).value == "C"

    def test_clear(self, store: InMemoryEventStore) -> None:
        store._streams["test"] = []
        store.clear()
        assert store.get_stream_count() == 0

    def test_get_counts(self, store: InMemoryEventStore) -> None:
        assert store.get_stream_count() == 0
        assert store.get_total_event_count() == 0


class TestFileEventStore:
    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    @pytest.fixture
    def store(self, temp_dir: str) -> FileEventStore:
        return FileEventStore(base_path=temp_dir)

    def test_validate_stream_id_valid(self):
        assert FileEventStore.validate_stream_id("valid-id_123") == "valid-id_123"

    def test_validate_stream_id_invalid(self):
        with pytest.raises(ValueError, match="Invalid stream ID"):
            FileEventStore.validate_stream_id("../invalid")

    def test_validate_stream_id_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            FileEventStore.validate_stream_id("")

    @pytest.mark.asyncio
    async def test_append_and_get_events(self, store: FileEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        stream_id = "test-stream"
        event = TokenEvent(ts=now_ms(), value="Hello", index=0)

        await store.append(stream_id, event)
        events = await store.get_events(stream_id)

        assert len(events) == 1
        assert cast(TokenEvent, events[0].event).value == "Hello"

    @pytest.mark.asyncio
    async def test_exists(self, store: FileEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        stream_id = "test-stream"
        assert await store.exists(stream_id) is False

        event = TokenEvent(ts=now_ms(), value="Hello", index=0)
        await store.append(stream_id, event)
        assert await store.exists(stream_id) is True

    @pytest.mark.asyncio
    async def test_delete(self, store: FileEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        stream_id = "test-stream"
        event = TokenEvent(ts=now_ms(), value="Hello", index=0)
        await store.append(stream_id, event)

        await store.delete(stream_id)
        assert await store.exists(stream_id) is False

    @pytest.mark.asyncio
    async def test_list_streams(self, store: FileEventStore) -> None:
        from l0.eventsourcing.types import TokenEvent, now_ms

        event = TokenEvent(ts=now_ms(), value="Hello", index=0)
        await store.append("stream-1", event)
        await store.append("stream-2", event)

        streams = await store.list_streams()
        assert "stream-1" in streams
        assert "stream-2" in streams

    @pytest.mark.asyncio
    async def test_snapshot(self, store: FileEventStore) -> None:
        from l0.eventsourcing.types import Snapshot

        snapshot = Snapshot(
            stream_id="test-stream",
            seq=10,
            ts=1000.0,
            content="Hello world",
            token_count=2,
            checkpoint="Hello",
        )

        await store.save_snapshot(snapshot)
        loaded = await store.get_snapshot("test-stream")

        assert loaded is not None
        assert loaded.content == "Hello world"
        assert loaded.seq == 10


class TestEventRecorder:
    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.fixture
    def recorder(self, store: InMemoryEventStore) -> EventRecorder:
        return EventRecorder(store, "test-stream")

    @pytest.mark.asyncio
    async def test_record_start(
        self, recorder: EventRecorder, store: InMemoryEventStore
    ) -> None:
        await recorder.record_start({"model": "gpt-4"})

        events = await store.get_events("test-stream")
        assert len(events) == 1
        assert events[0].event.type == RecordedEventType.START

    @pytest.mark.asyncio
    async def test_record_token(
        self, recorder: EventRecorder, store: InMemoryEventStore
    ) -> None:
        await recorder.record_token("Hello", 0)
        await recorder.record_token(" world", 1)

        events = await store.get_events("test-stream")
        assert len(events) == 2
        assert cast(TokenEvent, events[0].event).value == "Hello"
        assert cast(TokenEvent, events[1].event).value == " world"

    @pytest.mark.asyncio
    async def test_record_complete(
        self, recorder: EventRecorder, store: InMemoryEventStore
    ) -> None:
        await recorder.record_complete("Hello world", 2)

        events = await store.get_events("test-stream")
        assert len(events) == 1
        assert events[0].event.type == RecordedEventType.COMPLETE
        assert cast(CompleteEvent, events[0].event).content == "Hello world"

    @pytest.mark.asyncio
    async def test_record_error(
        self, recorder: EventRecorder, store: InMemoryEventStore
    ) -> None:
        await recorder.record_error(
            ValueError("test error"),
            failure_type="validation",
            recovery_strategy="halt",
        )

        events = await store.get_events("test-stream")
        assert len(events) == 1
        assert events[0].event.type == RecordedEventType.ERROR

    @pytest.mark.asyncio
    async def test_record_checkpoint(
        self, recorder: EventRecorder, store: InMemoryEventStore
    ) -> None:
        await recorder.record_checkpoint(5, "Hello world")

        events = await store.get_events("test-stream")
        assert len(events) == 1
        assert events[0].event.type == RecordedEventType.CHECKPOINT

    @pytest.mark.asyncio
    async def test_record_retry(
        self, recorder: EventRecorder, store: InMemoryEventStore
    ) -> None:
        await recorder.record_retry("rate_limit", 1, True)

        events = await store.get_events("test-stream")
        assert len(events) == 1
        assert events[0].event.type == RecordedEventType.RETRY

    @pytest.mark.asyncio
    async def test_record_fallback(
        self, recorder: EventRecorder, store: InMemoryEventStore
    ) -> None:
        await recorder.record_fallback(1)

        events = await store.get_events("test-stream")
        assert len(events) == 1
        assert events[0].event.type == RecordedEventType.FALLBACK

    @pytest.mark.asyncio
    async def test_seq_increments(self, recorder: EventRecorder) -> None:
        assert recorder.seq == 0
        await recorder.record_token("A", 0)
        assert recorder.seq == 1
        await recorder.record_token("B", 1)
        assert recorder.seq == 2


class TestEventReplayer:
    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_replay_tokens(self, store: InMemoryEventStore) -> None:
        recorder = EventRecorder(store, "test-stream")
        await recorder.record_start({})
        await recorder.record_token("Hello", 0)
        await recorder.record_token(" ", 1)
        await recorder.record_token("world", 2)
        await recorder.record_complete("Hello world", 3)

        replayer = EventReplayer(store)
        tokens = []
        async for token in replayer.replay_tokens("test-stream"):
            tokens.append(token)

        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_replay_to_state(self, store: InMemoryEventStore) -> None:
        recorder = EventRecorder(store, "test-stream")
        await recorder.record_start({})
        await recorder.record_token("Hello", 0)
        await recorder.record_token(" world", 1)
        await recorder.record_complete("Hello world", 2)

        replayer = EventReplayer(store)
        state = await replayer.replay_to_state("test-stream")

        assert state.content == "Hello world"
        assert state.token_count == 2
        assert state.completed is True

    @pytest.mark.asyncio
    async def test_replay_with_retries(self, store: InMemoryEventStore) -> None:
        recorder = EventRecorder(store, "test-stream")
        await recorder.record_start({})
        await recorder.record_retry("rate_limit", 1, True)
        await recorder.record_retry("network", 2, False)
        await recorder.record_complete("", 0)

        replayer = EventReplayer(store)
        state = await replayer.replay_to_state("test-stream")

        assert state.retry_attempts == 1
        assert state.network_retry_count == 1

    @pytest.mark.asyncio
    async def test_replay_continuation_rewinds_token_count(
        self, store: InMemoryEventStore
    ) -> None:
        """Test that continuation events properly rewind token_count."""
        recorder = EventRecorder(store, "test-stream")
        await recorder.record_start({})
        # Stream some tokens
        await recorder.record_token("Hello", 0)
        await recorder.record_token(" ", 1)
        await recorder.record_token("world", 2)
        # Simulate a continuation from checkpoint at token 1
        await recorder.record_continuation("Hello ", 2)
        # Continue with new tokens
        await recorder.record_token("there", 2)
        await recorder.record_complete("Hello there", 3)

        replayer = EventReplayer(store)
        state = await replayer.replay_to_state("test-stream")

        assert state.content == "Hello there"
        assert state.token_count == 3
        assert state.completed is True

    @pytest.mark.asyncio
    async def test_replay_result_continuation_rewinds_token_count(
        self, store: InMemoryEventStore
    ) -> None:
        """Test that ReplayResult also rewinds token_count on continuation."""
        from l0.eventsourcing.replayer import replay

        recorder = EventRecorder(store, "test-stream")
        await recorder.record_start({})
        await recorder.record_token("Hello", 0)
        await recorder.record_token(" ", 1)
        await recorder.record_token("world", 2)
        # Continuation rewinds to checkpoint
        await recorder.record_continuation("Hello ", 2)
        await recorder.record_token("there", 2)
        await recorder.record_complete("Hello there", 3)

        result = await replay("test-stream", store)

        # Consume all events
        async for _ in result:
            pass

        assert result.state.content == "Hello there"
        assert result.state.token_count == 3


class TestEventSourcing:
    def test_memory_store(self):
        store = EventSourcing.memory()
        assert isinstance(store, InMemoryEventStore)

    def test_file_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EventSourcing.file(temp_dir)
            assert isinstance(store, FileEventStore)

    def test_recorder(self):
        store = EventSourcing.memory()
        recorder = EventSourcing.recorder(store)
        assert isinstance(recorder, EventRecorder)

    def test_replayer(self):
        store = EventSourcing.memory()
        replayer = EventSourcing.replayer(store)
        assert isinstance(replayer, EventReplayer)

    def test_generate_id(self):
        id1 = EventSourcing.generate_id()
        id2 = EventSourcing.generate_id()
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        # Create store
        store = EventSourcing.memory()

        # Record events
        recorder = EventSourcing.recorder(store)
        await recorder.record_start({"model": "gpt-4"})
        await recorder.record_token("Hello", 0)
        await recorder.record_token(" world", 1)
        await recorder.record_complete("Hello world", 2)

        # Replay
        result = await EventSourcing.replay(recorder.stream_id, store)
        events = []
        async for event in result:
            events.append(event)

        assert len(events) == 4
        assert result.state.content == "Hello world"
        assert result.state.completed is True

    @pytest.mark.asyncio
    async def test_metadata(self):
        store = EventSourcing.memory()
        recorder = EventSourcing.recorder(store)
        await recorder.record_start({"model": "gpt-4"})
        await recorder.record_token("Hello", 0)
        await recorder.record_complete("Hello", 1)

        meta = await EventSourcing.metadata(recorder.stream_id, store)

        assert meta is not None
        assert meta.stream_id == recorder.stream_id
        assert meta.event_count == 3
        assert meta.token_count == 1
        assert meta.completed is True

    def test_compare_identical(self):
        from l0.eventsourcing import ReplayedState

        state1 = ReplayedState(content="Hello", token_count=1, completed=True)
        state2 = ReplayedState(content="Hello", token_count=1, completed=True)

        comparison = EventSourcing.compare(state1, state2)
        assert comparison.identical is True
        assert len(comparison.differences) == 0

    def test_compare_different(self):
        from l0.eventsourcing import ReplayedState

        state1 = ReplayedState(content="Hello", token_count=1, completed=True)
        state2 = ReplayedState(content="World", token_count=2, completed=False)

        comparison = EventSourcing.compare(state1, state2)
        assert comparison.identical is False
        assert len(comparison.differences) > 0

    def test_compare_different_violations_same_length(self):
        """Test that violations with same length but different contents are detected."""
        from l0.eventsourcing import ReplayedState

        state1 = ReplayedState(
            content="Hello",
            token_count=1,
            completed=True,
            violations=[{"rule": "violation_a"}],
        )
        state2 = ReplayedState(
            content="Hello",
            token_count=1,
            completed=True,
            violations=[{"rule": "violation_b"}],
        )

        comparison = EventSourcing.compare(state1, state2)
        assert comparison.identical is False
        assert any("violations" in diff for diff in comparison.differences)

    def test_compare_different_errors(self):
        """Test that different errors are detected."""
        from l0.eventsourcing import ReplayedState
        from l0.eventsourcing.types import SerializedError

        state1 = ReplayedState(
            content="Hello",
            token_count=1,
            completed=False,
            error=SerializedError(name="ValueError", message="error 1"),
        )
        state2 = ReplayedState(
            content="Hello",
            token_count=1,
            completed=False,
            error=SerializedError(name="TypeError", message="error 2"),
        )

        comparison = EventSourcing.compare(state1, state2)
        assert comparison.identical is False
        assert any("error" in diff for diff in comparison.differences)

    def test_compare_error_vs_no_error(self):
        """Test that presence/absence of error is detected."""
        from l0.eventsourcing import ReplayedState
        from l0.eventsourcing.types import SerializedError

        state1 = ReplayedState(
            content="Hello",
            token_count=1,
            completed=False,
            error=SerializedError(name="ValueError", message="error"),
        )
        state2 = ReplayedState(
            content="Hello",
            token_count=1,
            completed=True,
            error=None,
        )

        comparison = EventSourcing.compare(state1, state2)
        assert comparison.identical is False
        assert any("error" in diff for diff in comparison.differences)

    def test_compare_different_network_retry_count(self):
        """Test that different network_retry_count is detected."""
        from l0.eventsourcing import ReplayedState

        state1 = ReplayedState(content="Hello", network_retry_count=0)
        state2 = ReplayedState(content="Hello", network_retry_count=3)

        comparison = EventSourcing.compare(state1, state2)
        assert comparison.identical is False
        assert any("network_retry_count" in diff for diff in comparison.differences)

    def test_compare_different_timestamps(self):
        """Test that different timestamps are detected."""
        from l0.eventsourcing import ReplayedState

        state1 = ReplayedState(content="Hello", start_ts=1000.0, end_ts=2000.0)
        state2 = ReplayedState(content="Hello", start_ts=1000.0, end_ts=3000.0)

        comparison = EventSourcing.compare(state1, state2)
        assert comparison.identical is False
        assert any("end_ts" in diff for diff in comparison.differences)

    def test_compare_different_checkpoint(self):
        """Test that different checkpoints are detected."""
        from l0.eventsourcing import ReplayedState

        state1 = ReplayedState(content="Hello", checkpoint="check1")
        state2 = ReplayedState(content="Hello", checkpoint="check2")

        comparison = EventSourcing.compare(state1, state2)
        assert comparison.identical is False
        assert any("checkpoint" in diff for diff in comparison.differences)

    def test_list_adapters(self):
        adapters = EventSourcing.list_adapters()
        assert "memory" in adapters
        assert "file" in adapters


class TestCompositeEventStore:
    @pytest.mark.asyncio
    async def test_writes_to_all_stores(self):
        from l0.eventsourcing import CompositeEventStore
        from l0.eventsourcing.types import TokenEvent, now_ms

        store1 = InMemoryEventStore()
        store2 = InMemoryEventStore()
        composite = CompositeEventStore([store1, store2])

        event = TokenEvent(ts=now_ms(), value="Hello", index=0)
        await composite.append("test-stream", event)

        # Both stores should have the event
        events1 = await store1.get_events("test-stream")
        events2 = await store2.get_events("test-stream")

        assert len(events1) == 1
        assert len(events2) == 1

    @pytest.mark.asyncio
    async def test_reads_from_primary(self):
        from l0.eventsourcing import CompositeEventStore
        from l0.eventsourcing.types import TokenEvent, now_ms

        store1 = InMemoryEventStore()
        store2 = InMemoryEventStore()
        composite = CompositeEventStore([store1, store2], primary_index=0)

        # Add event only to store1 (primary)
        event = TokenEvent(ts=now_ms(), value="Hello", index=0)
        await store1.append("test-stream", event)

        events = await composite.get_events("test-stream")
        assert len(events) == 1


class TestTTLEventStore:
    @pytest.mark.asyncio
    async def test_filters_expired_events(self):
        from l0.eventsourcing import TTLEventStore
        from l0.eventsourcing.types import TokenEvent

        store = InMemoryEventStore()
        ttl_store = TTLEventStore(store, ttl_ms=1000)  # 1 second TTL

        # Add event with old timestamp
        old_event = TokenEvent(ts=1.0, value="Old", index=0)  # Very old timestamp
        await store.append("test-stream", old_event)

        events = await ttl_store.get_events("test-stream")
        assert len(events) == 0  # Should be filtered out

    @pytest.mark.asyncio
    async def test_exists_respects_ttl(self):
        """exists() should return False when all events have expired."""
        from l0.eventsourcing import TTLEventStore
        from l0.eventsourcing.types import TokenEvent

        store = InMemoryEventStore()
        ttl_store = TTLEventStore(store, ttl_ms=1000)  # 1 second TTL

        # Add event with old timestamp
        old_event = TokenEvent(ts=1.0, value="Old", index=0)  # Very old timestamp
        await store.append("test-stream", old_event)

        # Stream exists in underlying store but all events are expired
        assert await ttl_store.exists("test-stream") is False

    @pytest.mark.asyncio
    async def test_exists_true_with_unexpired_events(self):
        """exists() should return True when there are unexpired events."""
        from l0.eventsourcing import TTLEventStore
        from l0.eventsourcing.types import TokenEvent, now_ms

        store = InMemoryEventStore()
        ttl_store = TTLEventStore(store, ttl_ms=1000)  # 1 second TTL

        # Add event with current timestamp
        event = TokenEvent(ts=now_ms(), value="Current", index=0)
        await ttl_store.append("test-stream", event)

        assert await ttl_store.exists("test-stream") is True


class TestInMemoryEventStoreTTL:
    @pytest.mark.asyncio
    async def test_exists_respects_ttl(self):
        """InMemoryEventStore.exists() should return False when all events have expired."""
        from l0.eventsourcing.types import TokenEvent

        store = InMemoryEventStore(ttl=1000)  # 1 second TTL

        # Add event with old timestamp
        old_event = TokenEvent(ts=1.0, value="Old", index=0)  # Very old timestamp
        await store.append("test-stream", old_event)

        # Stream has events but all are expired
        assert await store.exists("test-stream") is False

    @pytest.mark.asyncio
    async def test_exists_true_with_unexpired_events(self):
        """InMemoryEventStore.exists() should return True when there are unexpired events."""
        from l0.eventsourcing.types import TokenEvent, now_ms

        store = InMemoryEventStore(ttl=1000)  # 1 second TTL

        # Add event with current timestamp
        event = TokenEvent(ts=now_ms(), value="Current", index=0)
        await store.append("test-stream", event)

        assert await store.exists("test-stream") is True

    @pytest.mark.asyncio
    async def test_exists_without_ttl(self):
        """InMemoryEventStore.exists() should work normally without TTL."""
        from l0.eventsourcing.types import TokenEvent

        store = InMemoryEventStore()  # No TTL

        # Add event with old timestamp
        old_event = TokenEvent(ts=1.0, value="Old", index=0)
        await store.append("test-stream", old_event)

        # Without TTL, stream should exist
        assert await store.exists("test-stream") is True


class TestStorageAdapters:
    @pytest.mark.asyncio
    async def test_create_memory_store(self):
        from l0.eventsourcing import StorageAdapterConfig, create_event_store

        store = await create_event_store(StorageAdapterConfig(type="memory"))
        assert isinstance(store, InMemoryEventStore)

    @pytest.mark.asyncio
    async def test_create_file_store(self):
        from l0.eventsourcing import StorageAdapterConfig, create_event_store

        with tempfile.TemporaryDirectory() as temp_dir:
            store = await create_event_store(
                StorageAdapterConfig(type="file", connection=temp_dir)
            )
            assert isinstance(store, FileEventStore)

    @pytest.mark.asyncio
    async def test_unknown_adapter_raises(self):
        from l0.eventsourcing import StorageAdapterConfig, create_event_store

        with pytest.raises(ValueError, match="Unknown storage adapter"):
            await create_event_store(StorageAdapterConfig(type="unknown"))

    def test_register_custom_adapter(self):
        from l0.eventsourcing import (
            get_registered_adapters,
            register_storage_adapter,
            unregister_storage_adapter,
        )

        def custom_factory(config: StorageAdapterConfig) -> InMemoryEventStore:
            return InMemoryEventStore()

        register_storage_adapter("custom", custom_factory)
        assert "custom" in get_registered_adapters()

        unregister_storage_adapter("custom")
        assert "custom" not in get_registered_adapters()
