"""Event sourcing integration tests with real API calls.

These tests verify event recording and replay with actual LLM streams,
ensuring byte-for-byte identical replay of recorded sessions.

Requires OPENAI_API_KEY to be set.
Run with: pytest tests/integration/test_event_sourcing.py -v
"""

from typing import TYPE_CHECKING

import pytest

import l0
from l0.eventsourcing import (
    EventRecorder,
    EventReplayer,
    InMemoryEventStore,
    RecordedEventType,
    compare_replays,
)
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@requires_openai
class TestRecordingRealLLMStreams:
    """Test recording real LLM streams."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_record_all_events_from_stream(
        self, client: "AsyncOpenAI", store: InMemoryEventStore
    ) -> None:
        """Test recording all events from a real LLM stream."""
        stream_id = "test-recording"
        recorder = EventRecorder(store, stream_id)

        # Record start
        await recorder.record_start({"prompt": "Say 'Hello, World!'"})

        # Stream and record tokens
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
            ],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)
        token_index = 0
        content = ""

        async for event in result:
            if event.is_token and event.text:
                await recorder.record_token(event.text, token_index)
                content += event.text
                token_index += 1

        # Record completion
        await recorder.record_complete(content, token_index)

        # Verify events were recorded
        events = await store.get_events(stream_id)
        assert len(events) > 0

        # Should have START event
        start_events = [e for e in events if e.event.type == RecordedEventType.START]
        assert len(start_events) == 1

        # Should have TOKEN events
        token_events = [e for e in events if e.event.type == RecordedEventType.TOKEN]
        assert len(token_events) > 0

        # Should have COMPLETE event
        complete_events = [
            e for e in events if e.event.type == RecordedEventType.COMPLETE
        ]
        assert len(complete_events) == 1

    @pytest.mark.asyncio
    async def test_record_events_with_sequence_numbers(
        self, client: "AsyncOpenAI", store: InMemoryEventStore
    ) -> None:
        """Test that events are recorded with correct sequence numbers."""
        stream_id = "test-sequence"
        recorder = EventRecorder(store, stream_id)

        await recorder.record_start({"prompt": "Count from 1 to 3."})

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)
        token_index = 0
        content = ""

        async for event in result:
            if event.is_token and event.text:
                await recorder.record_token(event.text, token_index)
                content += event.text
                token_index += 1

        await recorder.record_complete(content, token_index)

        events = await store.get_events(stream_id)

        # Verify sequence numbers are sequential
        for i, event in enumerate(events):
            assert event.seq == i


@requires_openai
class TestReplayingRecordedStreams:
    """Test replaying recorded streams."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_replay_reconstructs_state(
        self, client: "AsyncOpenAI", store: InMemoryEventStore
    ) -> None:
        """Test that replay reconstructs the original state."""
        stream_id = "test-replay"
        recorder = EventRecorder(store, stream_id)

        # Record a stream
        await recorder.record_start({"prompt": "Say 'Replay Test'"})

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Replay Test' exactly."}],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)
        original_content = await result.read()

        # Record tokens (simulate - in real code you'd record during streaming)
        await recorder.record_token(original_content, 0)
        await recorder.record_complete(original_content, 1)

        # Replay and verify
        replayer = EventReplayer(store)
        replayed_state = await replayer.replay_to_state(stream_id)

        assert replayed_state.content == original_content
        assert replayed_state.completed is True

    @pytest.mark.asyncio
    async def test_replay_tokens_in_order(
        self, client: "AsyncOpenAI", store: InMemoryEventStore
    ) -> None:
        """Test that tokens are replayed in correct order."""
        stream_id = "test-token-order"
        recorder = EventRecorder(store, stream_id)

        await recorder.record_start({"prompt": "Say 'A B C'"})

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'A B C' with spaces."}],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)
        original_tokens: list[str] = []
        token_index = 0

        async for event in result:
            if event.is_token and event.text:
                original_tokens.append(event.text)
                await recorder.record_token(event.text, token_index)
                token_index += 1

        original_content = "".join(original_tokens)
        await recorder.record_complete(original_content, token_index)

        # Replay tokens
        replayer = EventReplayer(store)
        replayed_tokens: list[str] = []

        async for token in replayer.replay_tokens(stream_id):
            replayed_tokens.append(token)

        # Tokens should match
        assert replayed_tokens == original_tokens
        assert "".join(replayed_tokens) == original_content


@requires_openai
class TestEventStorePersistence:
    """Test event store persistence across streams."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_separate_streams_maintained(
        self, client: "AsyncOpenAI", store: InMemoryEventStore
    ) -> None:
        """Test that separate streams are maintained correctly."""
        stream_id_1 = "test-stream-1"
        stream_id_2 = "test-stream-2"

        # Record first stream
        recorder1 = EventRecorder(store, stream_id_1)
        await recorder1.record_start({"prompt": "Say 'First'"})
        await recorder1.record_token("First", 0)
        await recorder1.record_complete("First", 1)

        # Record second stream
        recorder2 = EventRecorder(store, stream_id_2)
        await recorder2.record_start({"prompt": "Say 'Second'"})
        await recorder2.record_token("Second", 0)
        await recorder2.record_complete("Second", 1)

        # Verify streams are separate
        events1 = await store.get_events(stream_id_1)
        events2 = await store.get_events(stream_id_2)

        assert len(events1) > 0
        assert len(events2) > 0

        # All events in stream1 should have stream_id_1
        for event in events1:
            assert event.stream_id == stream_id_1

        # All events in stream2 should have stream_id_2
        for event in events2:
            assert event.stream_id == stream_id_2

    @pytest.mark.asyncio
    async def test_list_all_streams(
        self, client: "AsyncOpenAI", store: InMemoryEventStore
    ) -> None:
        """Test listing all recorded streams."""
        stream_ids = ["list-test-1", "list-test-2", "list-test-3"]

        for stream_id in stream_ids:
            recorder = EventRecorder(store, stream_id)
            await recorder.record_start({"prompt": "test"})
            await recorder.record_token("test", 0)
            await recorder.record_complete("test", 1)

        listed_streams = await store.list_streams()

        for stream_id in stream_ids:
            assert stream_id in listed_streams


@requires_openai
class TestManualEventRecording:
    """Test manual event recording."""

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_manual_token_recording(self, store: InMemoryEventStore) -> None:
        """Test manual token recording and replay."""
        stream_id = "test-manual"
        recorder = EventRecorder(store, stream_id)

        # Record manually
        await recorder.record_start({"prompt": "Test prompt"})
        await recorder.record_token("Hello", 0)
        await recorder.record_token(" ", 1)
        await recorder.record_token("World", 2)
        await recorder.record_complete("Hello World", 3)

        # Verify all events were recorded
        events = await store.get_events(stream_id)
        assert len(events) == 5  # START + 3 TOKENS + COMPLETE

        # Replay and verify
        replayer = EventReplayer(store)
        state = await replayer.replay_to_state(stream_id)

        assert state.content == "Hello World"
        assert state.completed is True


@requires_openai
class TestErrorRecording:
    """Test error event recording."""

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_record_error_events(self, store: InMemoryEventStore) -> None:
        """Test recording error events."""
        stream_id = "test-error"
        recorder = EventRecorder(store, stream_id)

        # Simulate error recording
        await recorder.record_start({"prompt": "Test"})
        await recorder.record_token("Partial", 0)
        await recorder.record_error(
            error={"name": "NetworkError", "message": "Connection lost"},
            failure_type="network",
            recovery_strategy="retry",
        )

        events = await store.get_events(stream_id)
        error_events = [e for e in events if e.event.type == RecordedEventType.ERROR]

        assert len(error_events) == 1
        error_event = error_events[0].event
        assert error_event.error.name == "NetworkError"  # type: ignore[union-attr]
        assert error_event.failure_type == "network"  # type: ignore[union-attr]
        assert error_event.recovery_strategy == "retry"  # type: ignore[union-attr]


@requires_openai
class TestRetryAndFallbackRecording:
    """Test retry and fallback event recording."""

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_record_retry_events(self, store: InMemoryEventStore) -> None:
        """Test recording retry events."""
        stream_id = "test-retry-record"
        recorder = EventRecorder(store, stream_id)

        await recorder.record_start({"prompt": "Test"})
        await recorder.record_retry("rate_limit", 1, True)
        await recorder.record_retry("timeout", 2, True)
        await recorder.record_token("Success", 0)
        await recorder.record_complete("Success", 1)

        events = await store.get_events(stream_id)
        retry_events = [e for e in events if e.event.type == RecordedEventType.RETRY]

        assert len(retry_events) == 2

        # Replay and check retry count
        replayer = EventReplayer(store)
        state = await replayer.replay_to_state(stream_id)

        assert state.retry_attempts == 2

    @pytest.mark.asyncio
    async def test_record_fallback_events(self, store: InMemoryEventStore) -> None:
        """Test recording fallback events."""
        stream_id = "test-fallback-record"
        recorder = EventRecorder(store, stream_id)

        await recorder.record_start({"prompt": "Test"})
        await recorder.record_fallback(1)  # Fallback to model index 1
        await recorder.record_fallback(2)  # Fallback to model index 2
        await recorder.record_token("Success", 0)
        await recorder.record_complete("Success", 1)

        events = await store.get_events(stream_id)
        fallback_events = [
            e for e in events if e.event.type == RecordedEventType.FALLBACK
        ]

        assert len(fallback_events) == 2

        # Replay and check fallback index
        replayer = EventReplayer(store)
        state = await replayer.replay_to_state(stream_id)

        assert state.fallback_index == 2


@requires_openai
class TestByteForByteIdenticalReplay:
    """Test byte-for-byte identical replay."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_byte_identical_replay(
        self, client: "AsyncOpenAI", store: InMemoryEventStore
    ) -> None:
        """Test that replay produces byte-for-byte identical output."""
        stream_id = "byte-identical"
        recorder = EventRecorder(store, stream_id)

        # Record a real LLM stream
        await recorder.record_start({"prompt": "Output exactly: 'The quick brown fox'"})

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Output exactly: 'The quick brown fox'",
                }
            ],
            stream=True,
            max_tokens=20,
        )

        result = l0.wrap(stream)
        original_tokens: list[str] = []
        token_index = 0

        async for event in result:
            if event.is_token and event.text:
                original_tokens.append(event.text)
                await recorder.record_token(event.text, token_index)
                token_index += 1

        original_output = "".join(original_tokens)
        await recorder.record_complete(original_output, token_index)

        # Replay and reconstruct
        replayer = EventReplayer(store)

        # Method 1: Replay tokens
        replayed_tokens: list[str] = []
        async for token in replayer.replay_tokens(stream_id):
            replayed_tokens.append(token)
        replayed_from_tokens = "".join(replayed_tokens)

        # Method 2: Replay to state
        replayed_state = await replayer.replay_to_state(stream_id)

        # Verify byte-for-byte identical
        assert replayed_from_tokens == original_output
        assert replayed_state.content == original_output

        # Verify token-by-token identical
        assert len(replayed_tokens) == len(original_tokens)
        for i, (orig, replayed) in enumerate(zip(original_tokens, replayed_tokens)):
            assert orig == replayed, f"Token {i} mismatch: {orig!r} vs {replayed!r}"


@requires_openai
class TestPartialReplay:
    """Test partial replay from halfway point."""

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_partial_replay(self, store: InMemoryEventStore) -> None:
        """Test replaying only first N events."""
        stream_id = "partial-replay"
        recorder = EventRecorder(store, stream_id)

        # Record known tokens
        all_tokens = ["One", " ", "Two", " ", "Three", " ", "Four"]
        await recorder.record_start({"prompt": "Test partial replay"})
        for i, token in enumerate(all_tokens):
            await recorder.record_token(token, i)
        await recorder.record_complete("".join(all_tokens), len(all_tokens))

        # Get all events
        all_events = await store.get_events(stream_id)
        token_events = [
            e for e in all_events if e.event.type == RecordedEventType.TOKEN
        ]

        # Replay only first half of tokens
        halfway_count = len(token_events) // 2
        replayer = EventReplayer(store)

        # Replay with to_seq limit
        partial_tokens: list[str] = []
        async for envelope in replayer.replay(stream_id, to_seq=halfway_count):
            if envelope.event.type == RecordedEventType.TOKEN:
                partial_tokens.append(envelope.event.value)

        # Verify partial output
        expected_partial = all_tokens[:halfway_count]
        assert partial_tokens == expected_partial
        assert len(partial_tokens) < len(all_tokens)

    @pytest.mark.asyncio
    async def test_replay_interrupted_stream(self, store: InMemoryEventStore) -> None:
        """Test replaying an interrupted stream (no COMPLETE event)."""
        stream_id = "interrupted"
        recorder = EventRecorder(store, stream_id)

        # Simulate an interrupted stream (no COMPLETE event)
        await recorder.record_start({"prompt": "Test"})
        await recorder.record_token("Hello", 0)
        await recorder.record_token(" ", 1)
        await recorder.record_token("World", 2)
        # Note: No record_complete - simulates crash/interrupt

        replayer = EventReplayer(store)
        state = await replayer.replay_to_state(stream_id)

        # Should reconstruct partial content
        assert state.content == "Hello World"
        assert state.completed is False  # Not marked complete
        assert state.token_count == 3


@requires_openai
class TestDeterministicReplays:
    """Test deterministic replays across multiple runs."""

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_deterministic_fallback_replay(
        self, store: InMemoryEventStore
    ) -> None:
        """Test that fallback sequence replays deterministically."""
        stream_id = "fallback-determinism"
        recorder = EventRecorder(store, stream_id)

        # Record a sequence with fallbacks
        await recorder.record_start({"prompt": "Test", "fallback_count": 2})
        await recorder.record_fallback(1)
        await recorder.record_fallback(2)
        await recorder.record_token("Success on fallback", 0)
        await recorder.record_complete("Success on fallback", 1)

        # Replay multiple times - should be identical each time
        replayer = EventReplayer(store)

        replay1 = await replayer.replay_to_state(stream_id)
        replay2 = await replayer.replay_to_state(stream_id)
        replay3 = await replayer.replay_to_state(stream_id)

        # All replays must produce identical results
        assert replay1.content == replay2.content == replay3.content
        assert (
            replay1.fallback_index == replay2.fallback_index == replay3.fallback_index
        )
        assert replay1.fallback_index == 2

    @pytest.mark.asyncio
    async def test_deterministic_retry_replay(self, store: InMemoryEventStore) -> None:
        """Test that retry sequence replays deterministically."""
        stream_id = "retry-determinism"
        recorder = EventRecorder(store, stream_id)

        # Record a sequence with retries
        await recorder.record_start({"prompt": "Test"})
        await recorder.record_retry("rate_limit", 1, True)
        await recorder.record_retry("timeout", 2, True)
        await recorder.record_token("Finally", 0)
        await recorder.record_complete("Finally", 1)

        replayer = EventReplayer(store)

        # Replay and verify retry count is deterministic
        state = await replayer.replay_to_state(stream_id)
        assert state.retry_attempts == 2
        assert state.content == "Finally"

        # Replay again - same result
        state2 = await replayer.replay_to_state(stream_id)
        assert state2.retry_attempts == 2
        assert state2.content == "Finally"


@requires_openai
class TestReplayComparison:
    """Test replay comparison utilities."""

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_compare_identical_replays(self, store: InMemoryEventStore) -> None:
        """Test comparing identical replays."""
        stream_id = "compare-test"
        recorder = EventRecorder(store, stream_id)

        await recorder.record_start({"prompt": "Test"})
        await recorder.record_token("Hello", 0)
        await recorder.record_complete("Hello", 1)

        replayer = EventReplayer(store)

        state1 = await replayer.replay_to_state(stream_id)
        state2 = await replayer.replay_to_state(stream_id)

        comparison = compare_replays(state1, state2)

        assert comparison.identical is True
        assert len(comparison.differences) == 0


@requires_openai
class TestContinuationRecording:
    """Test continuation event recording."""

    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_record_continuation(self, store: InMemoryEventStore) -> None:
        """Test recording continuation events."""
        stream_id = "continuation-test"
        recorder = EventRecorder(store, stream_id)

        await recorder.record_start({"prompt": "Write a story"})

        # First attempt - partial
        await recorder.record_token("Once upon", 0)
        await recorder.record_checkpoint(1, "Once upon")

        # Network error, continuation triggered
        await recorder.record_continuation("Once upon", 1)

        # Continued from checkpoint
        await recorder.record_token(" a time", 1)
        await recorder.record_complete("Once upon a time", 2)

        replayer = EventReplayer(store)
        state = await replayer.replay_to_state(stream_id)

        # Content should reflect the continuation
        assert state.content == "Once upon a time"
        assert state.checkpoint == "Once upon"
