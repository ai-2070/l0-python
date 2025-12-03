"""Main L0 runtime engine."""

from __future__ import annotations

from typing import AsyncIterator

from .adapters import detect_adapter
from .events import EventBus, ObservabilityEventType
from .logging import logger
from .retry import RetryManager
from .state import append_token, create_state, mark_completed, update_checkpoint
from .types import EventType, L0Event, L0Options, L0Result


async def l0(options: L0Options) -> L0Result:
    """Main L0 wrapper function."""

    state = create_state()
    retry_mgr = RetryManager(options.retry)
    event_bus = EventBus(options.on_event, meta=options.meta)
    errors: list[Exception] = []
    aborted = False

    logger.debug(f"Starting L0 stream: {event_bus.stream_id}")

    def abort() -> None:
        nonlocal aborted
        aborted = True
        state.aborted = True
        logger.debug("Abort requested")

    async def run_stream() -> AsyncIterator[L0Event]:
        nonlocal state

        streams = [options.stream] + options.fallbacks

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
                    if hasattr(raw_stream, "__await__"):
                        raw_stream = await raw_stream

                    adapter = detect_adapter(raw_stream, options.adapter)
                    event_bus.emit(ObservabilityEventType.STREAM_READY)

                    async for event in adapter.wrap(raw_stream):
                        if aborted:
                            return

                        if event.type == EventType.TOKEN and event.value:
                            append_token(state, event.value)

                            # Check guardrails periodically
                            if state.token_count % 5 == 0 and options.guardrails:
                                # Import here to avoid circular import
                                from .guardrails import check_guardrails

                                event_bus.emit(
                                    ObservabilityEventType.GUARDRAIL_PHASE_START
                                )
                                violations = check_guardrails(state, options.guardrails)
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

    return L0Result(
        stream=run_stream(),
        state=state,
        abort=abort,
        errors=errors,
    )
