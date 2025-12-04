"""Event Recorder for Event Sourcing.

Wraps an event store with convenient recording methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    RecordedEvent,
    RetryEvent,
    SerializedError,
    SerializedOptions,
    StartEvent,
    TokenEvent,
    generate_stream_id,
    now_ms,
)

if TYPE_CHECKING:
    from .store import EventStore


class EventRecorder:
    """Event recorder - wraps an event store with convenient recording methods."""

    def __init__(self, event_store: EventStore, stream_id: str | None = None):
        """Create an event recorder.

        Args:
            event_store: The event store to write to
            stream_id: Custom stream ID (auto-generated if not provided)
        """
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
        """Record an event to the store."""
        await self._event_store.append(self._stream_id, event)
        self._seq += 1

    async def record_start(
        self,
        options: SerializedOptions | dict[str, Any] | None = None,
    ) -> None:
        """Record a START event.

        Args:
            options: Serialized options for the stream
        """
        if options is None:
            options = SerializedOptions()
        elif isinstance(options, dict):
            options = SerializedOptions(**options)

        await self.record(StartEvent(ts=now_ms(), options=options))

    async def record_token(self, value: str, index: int) -> None:
        """Record a TOKEN event.

        Args:
            value: Token content
            index: Zero-based token index
        """
        await self.record(TokenEvent(ts=now_ms(), value=value, index=index))

    async def record_checkpoint(self, at: int, content: str) -> None:
        """Record a CHECKPOINT event.

        Args:
            at: Token index at checkpoint
            content: Accumulated content at checkpoint
        """
        await self.record(CheckpointEvent(ts=now_ms(), at=at, content=content))

    async def record_guardrail(
        self,
        at: int,
        result: GuardrailEventResult | dict[str, Any],
    ) -> None:
        """Record a GUARDRAIL event.

        Args:
            at: Token index when check occurred
            result: Guardrail evaluation result
        """
        if isinstance(result, dict):
            result = GuardrailEventResult(**result)

        await self.record(GuardrailEvent(ts=now_ms(), at=at, result=result))

    async def record_drift(
        self,
        at: int,
        result: DriftEventResult | dict[str, Any],
    ) -> None:
        """Record a DRIFT event.

        Args:
            at: Token index when check occurred
            result: Drift detection result
        """
        if isinstance(result, dict):
            result = DriftEventResult(**result)

        await self.record(DriftEvent(ts=now_ms(), at=at, result=result))

    async def record_retry(
        self,
        reason: str,
        attempt: int,
        counts_toward_limit: bool = True,
    ) -> None:
        """Record a RETRY event.

        Args:
            reason: Reason for retry
            attempt: Attempt number (1-based)
            counts_toward_limit: Whether this counts toward model retry limit
        """
        await self.record(
            RetryEvent(
                ts=now_ms(),
                reason=reason,
                attempt=attempt,
                counts_toward_limit=counts_toward_limit,
            )
        )

    async def record_fallback(self, to: int) -> None:
        """Record a FALLBACK event.

        Args:
            to: Index of stream we're falling back to
        """
        await self.record(FallbackEvent(ts=now_ms(), to=to))

    async def record_continuation(self, checkpoint: str, at: int) -> None:
        """Record a CONTINUATION event.

        Args:
            checkpoint: Checkpoint content used for continuation
            at: Token index of checkpoint
        """
        await self.record(ContinuationEvent(ts=now_ms(), checkpoint=checkpoint, at=at))

    async def record_complete(self, content: str, token_count: int) -> None:
        """Record a COMPLETE event.

        Args:
            content: Final accumulated content
            token_count: Total token count
        """
        await self.record(
            CompleteEvent(ts=now_ms(), content=content, token_count=token_count)
        )

    async def record_error(
        self,
        error: SerializedError | dict[str, Any] | Exception,
        failure_type: str = "",
        recovery_strategy: str = "",
        policy: str = "",
    ) -> None:
        """Record an ERROR event.

        Args:
            error: Serialized error or exception
            failure_type: What went wrong - the root cause
            recovery_strategy: What was decided to do next
            policy: Policy that determined the recovery strategy
        """
        from .types import serialize_error

        if isinstance(error, Exception):
            error = serialize_error(error)
        elif isinstance(error, dict):
            error = SerializedError(**error)

        await self.record(
            ErrorEvent(
                ts=now_ms(),
                error=error,
                failure_type=failure_type,
                recovery_strategy=recovery_strategy,
                policy=policy,
            )
        )
