"""Main L0 runtime engine."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from .adapters import detect_adapter
from .events import EventBus, ObservabilityEventType
from .logging import logger
from .retry import RetryManager
from .state import append_token, create_state, mark_completed, update_checkpoint
from .types import Event, EventType, Retry, State, Stream, Timeout


class TimeoutError(Exception):
    """Raised when a timeout occurs during streaming."""

    def __init__(self, message: str, timeout_type: str, timeout_seconds: float):
        super().__init__(message)
        self.timeout_type = timeout_type  # "initial_token" or "inter_token"
        self.timeout_seconds = timeout_seconds


if TYPE_CHECKING:
    from .events import ObservabilityEvent
    from .guardrails import GuardrailRule


async def _internal_run(
    stream: Callable[[], AsyncIterator[Any]],
    *,
    fallbacks: list[Callable[[], AsyncIterator[Any]]] | None = None,
    guardrails: list[GuardrailRule] | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    adapter: Any | str | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
    meta: dict[str, Any] | None = None,
) -> Stream:
    """Internal implementation of the L0 runtime.

    Args:
        stream: Factory function that returns an async LLM stream
        fallbacks: Optional list of fallback stream factories
        guardrails: Optional list of guardrail rules to apply
        retry: Optional retry configuration
        timeout: Optional timeout configuration
        adapter: Optional adapter hint ("openai", "litellm", or Adapter instance)
        on_event: Optional callback for observability events
        meta: Optional metadata attached to all events

    Returns:
        Stream - async iterator with .state, .abort(), and .read()
    """
    fallbacks = fallbacks or []
    guardrails = guardrails or []

    state = create_state()
    retry_mgr = RetryManager(retry)
    event_bus = EventBus(on_event, meta=meta)
    errors: list[Exception] = []
    aborted = False

    logger.debug(f"Starting L0 stream: {event_bus.stream_id}")

    def abort() -> None:
        nonlocal aborted
        aborted = True
        state.aborted = True
        logger.debug("Abort requested")

    async def run_stream() -> AsyncIterator[Event]:
        nonlocal state

        streams = [stream] + fallbacks

        for fallback_idx, stream_fn in enumerate(streams):
            state.fallback_index = fallback_idx

            if fallback_idx > 0:
                logger.debug(f"Trying fallback {fallback_idx}")
                event_bus.emit(
                    ObservabilityEventType.FALLBACK_START, index=fallback_idx
                )

            while True:
                if aborted:
                    return

                try:
                    event_bus.emit(ObservabilityEventType.STREAM_INIT)
                    raw_stream = stream_fn()

                    # Handle both sync and async stream factories
                    # stream_fn() might return a coroutine or an async iterator directly
                    if inspect.iscoroutine(raw_stream):
                        raw_stream = await raw_stream

                    detected_adapter = detect_adapter(raw_stream, adapter)
                    event_bus.emit(ObservabilityEventType.STREAM_READY)

                    # Set up timeout tracking
                    first_token_received = False
                    initial_timeout = timeout.initial_token if timeout else None
                    inter_timeout = timeout.inter_token if timeout else None

                    async def get_next_event() -> Event:
                        return await detected_adapter.wrap(raw_stream).__anext__()

                    adapted_stream = detected_adapter.wrap(raw_stream)

                    while True:
                        if aborted:
                            return

                        # Determine which timeout to use
                        current_timeout = (
                            inter_timeout if first_token_received else initial_timeout
                        )

                        try:
                            if current_timeout is not None:
                                event = await asyncio.wait_for(
                                    adapted_stream.__anext__(), timeout=current_timeout
                                )
                            else:
                                event = await adapted_stream.__anext__()
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            timeout_type = (
                                "inter_token"
                                if first_token_received
                                else "initial_token"
                            )
                            raise TimeoutError(
                                f"Timeout waiting for {'next' if first_token_received else 'first'} token "
                                f"(timeout={current_timeout}s)",
                                timeout_type=timeout_type,
                                timeout_seconds=current_timeout,
                            )

                        if event.type == EventType.TOKEN and event.text:
                            first_token_received = True
                            append_token(state, event.text)

                            # Check guardrails periodically
                            if state.token_count % 5 == 0 and guardrails:
                                # Import here to avoid circular import
                                from .guardrails import check_guardrails

                                event_bus.emit(
                                    ObservabilityEventType.GUARDRAIL_PHASE_START
                                )
                                violations = check_guardrails(state, guardrails)
                                if violations:
                                    state.violations.extend(violations)
                                    event_bus.emit(
                                        ObservabilityEventType.GUARDRAIL_RULE_RESULT,
                                        violations=[v.__dict__ for v in violations],
                                    )
                                event_bus.emit(
                                    ObservabilityEventType.GUARDRAIL_PHASE_END
                                )

                        yield event

                    # Success
                    mark_completed(state)
                    event_bus.emit(
                        ObservabilityEventType.COMPLETE, token_count=state.token_count
                    )
                    logger.debug(f"Stream complete: {state.token_count} tokens")
                    return

                except Exception as e:
                    errors.append(e)
                    logger.debug(f"Stream error: {e}")
                    event_bus.emit(ObservabilityEventType.NETWORK_ERROR, error=str(e))

                    if retry_mgr.should_retry(e):
                        retry_mgr.record_attempt(e)
                        state.model_retry_count = retry_mgr.model_retry_count
                        state.network_retry_count = retry_mgr.network_retry_count
                        event_bus.emit(
                            ObservabilityEventType.RETRY_ATTEMPT,
                            attempt=retry_mgr.total_retries,
                        )
                        await retry_mgr.wait(e)
                        update_checkpoint(state)
                        state.resumed = True
                        continue
                    else:
                        # Try next fallback
                        event_bus.emit(
                            ObservabilityEventType.FALLBACK_END, index=fallback_idx
                        )
                        break

        # All fallbacks exhausted
        event_bus.emit(ObservabilityEventType.RETRY_GIVE_UP)
        if errors:
            raise errors[-1]
        raise RuntimeError("All streams and fallbacks exhausted")

    return Stream(
        iterator=run_stream(),
        state=state,
        abort=abort,
        errors=errors,
    )
