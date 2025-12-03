"""Main L0 runtime engine."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from .adapters import Adapters
from .events import EventBus, ObservabilityEventType
from .logging import logger
from .retry import RetryManager
from .state import append_token, create_state, mark_completed, update_checkpoint
from .types import Event, EventType, Retry, Stream, Timeout


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
    event_bus.emit(ObservabilityEventType.SESSION_START, session_id=event_bus.stream_id)

    def abort() -> None:
        nonlocal aborted
        aborted = True
        state.aborted = True
        logger.debug("Abort requested")
        event_bus.emit(ObservabilityEventType.ABORT_REQUESTED, source="user")

    async def run_stream() -> AsyncIterator[Event]:
        nonlocal state

        streams = [stream] + fallbacks

        for fallback_idx, stream_fn in enumerate(streams):
            state.fallback_index = fallback_idx

            if fallback_idx > 0:
                logger.debug(f"Trying fallback {fallback_idx}")
                event_bus.emit(
                    ObservabilityEventType.FALLBACK_START,
                    index=fallback_idx,
                    from_index=fallback_idx - 1,
                    reason="previous_failed",
                )
                event_bus.emit(
                    ObservabilityEventType.FALLBACK_MODEL_SELECTED, index=fallback_idx
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

                    event_bus.emit(ObservabilityEventType.ADAPTER_WRAP_START)
                    detected_adapter = Adapters.detect(raw_stream, adapter)
                    event_bus.emit(
                        ObservabilityEventType.ADAPTER_DETECTED,
                        adapter=detected_adapter.name,
                    )
                    event_bus.emit(ObservabilityEventType.STREAM_READY)

                    # Set up timeout tracking
                    first_token_received = False
                    initial_timeout = timeout.initial_token if timeout else None
                    inter_timeout = timeout.inter_token if timeout else None

                    if initial_timeout is not None:
                        event_bus.emit(
                            ObservabilityEventType.TIMEOUT_START,
                            timeout_type="initial_token",
                            duration_seconds=initial_timeout,
                        )

                    adapted_stream = detected_adapter.wrap(raw_stream)
                    event_bus.emit(ObservabilityEventType.ADAPTER_WRAP_END)

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
                        except asyncio.TimeoutError as e:
                            timeout_type = (
                                "inter_token"
                                if first_token_received
                                else "initial_token"
                            )
                            # current_timeout is guaranteed to be float here
                            # (asyncio.TimeoutError only raised if wait_for was called)
                            assert current_timeout is not None
                            event_bus.emit(
                                ObservabilityEventType.TIMEOUT_TRIGGERED,
                                timeout_type=timeout_type,
                                elapsed_seconds=current_timeout,
                            )
                            token_desc = "next" if first_token_received else "first"
                            raise TimeoutError(
                                f"Timeout waiting for {token_desc} token "
                                f"(timeout={current_timeout}s)",
                                timeout_type=timeout_type,
                                timeout_seconds=current_timeout,
                            ) from e

                        if event.type == EventType.TOKEN and event.text:
                            if not first_token_received and inter_timeout is not None:
                                # First token received, switch to inter-token timeout
                                event_bus.emit(
                                    ObservabilityEventType.TIMEOUT_RESET,
                                    timeout_type="inter_token",
                                    token_index=state.token_count,
                                )
                            first_token_received = True
                            append_token(state, event.text)

                            # Check guardrails periodically
                            if state.token_count % 5 == 0 and guardrails:
                                event_bus.emit(
                                    ObservabilityEventType.GUARDRAIL_PHASE_START,
                                    context_size=len(state.content),
                                    rule_count=len(guardrails),
                                )

                                all_violations = []
                                for idx, rule in enumerate(guardrails):
                                    event_bus.emit(
                                        ObservabilityEventType.GUARDRAIL_RULE_START,
                                        index=idx,
                                        rule_id=rule.name,
                                    )
                                    rule_violations = rule.check(state)
                                    if rule_violations:
                                        all_violations.extend(rule_violations)
                                        event_bus.emit(
                                            ObservabilityEventType.GUARDRAIL_RULE_RESULT,
                                            index=idx,
                                            rule_id=rule.name,
                                            violations=[
                                                v.__dict__ for v in rule_violations
                                            ],
                                        )
                                    event_bus.emit(
                                        ObservabilityEventType.GUARDRAIL_RULE_END,
                                        index=idx,
                                        rule_id=rule.name,
                                    )

                                if all_violations:
                                    state.violations.extend(all_violations)

                                event_bus.emit(
                                    ObservabilityEventType.GUARDRAIL_PHASE_END,
                                    rule_count=len(guardrails),
                                    violation_count=len(all_violations),
                                )

                        yield event

                    # Success
                    mark_completed(state)

                    # Run final guardrail check (for completion-only rules)
                    if guardrails:
                        event_bus.emit(
                            ObservabilityEventType.GUARDRAIL_PHASE_START,
                            context_size=len(state.content),
                            rule_count=len(guardrails),
                        )

                        all_violations = []
                        for idx, rule in enumerate(guardrails):
                            event_bus.emit(
                                ObservabilityEventType.GUARDRAIL_RULE_START,
                                index=idx,
                                rule_id=rule.name,
                            )
                            rule_violations = rule.check(state)
                            if rule_violations:
                                all_violations.extend(rule_violations)
                                event_bus.emit(
                                    ObservabilityEventType.GUARDRAIL_RULE_RESULT,
                                    index=idx,
                                    rule_id=rule.name,
                                    violations=[v.__dict__ for v in rule_violations],
                                )
                            event_bus.emit(
                                ObservabilityEventType.GUARDRAIL_RULE_END,
                                index=idx,
                                rule_id=rule.name,
                            )

                        if all_violations:
                            state.violations.extend(all_violations)

                        event_bus.emit(
                            ObservabilityEventType.GUARDRAIL_PHASE_END,
                            rule_count=len(guardrails),
                            violation_count=len(all_violations),
                        )

                    if fallback_idx > 0:
                        event_bus.emit(
                            ObservabilityEventType.FALLBACK_END, index=fallback_idx
                        )
                    event_bus.emit(ObservabilityEventType.FINALIZATION_START)
                    event_bus.emit(
                        ObservabilityEventType.COMPLETE, token_count=state.token_count
                    )
                    event_bus.emit(ObservabilityEventType.FINALIZATION_END)
                    event_bus.emit(
                        ObservabilityEventType.SESSION_SUMMARY,
                        token_count=state.token_count,
                        duration=state.duration,
                        guardrail_violations=len(state.violations),
                        fallback_depth=state.fallback_index,
                        retry_count=state.model_retry_count + state.network_retry_count,
                    )
                    event_bus.emit(ObservabilityEventType.SESSION_END)
                    logger.debug(f"Stream complete: {state.token_count} tokens")
                    return

                except Exception as e:
                    errors.append(e)
                    logger.debug(f"Stream error: {e}")

                    # Determine error category
                    from .errors import categorize_error
                    from .types import ErrorCategory

                    error_category = categorize_error(e)
                    is_network = error_category == ErrorCategory.NETWORK

                    if is_network:
                        event_bus.emit(
                            ObservabilityEventType.NETWORK_ERROR,
                            error=str(e),
                            retryable=retry_mgr.should_retry(e),
                        )
                    else:
                        event_bus.emit(
                            ObservabilityEventType.ERROR,
                            error=str(e),
                            category=error_category.value,
                        )

                    if retry_mgr.should_retry(e):
                        event_bus.emit(
                            ObservabilityEventType.RETRY_START,
                            attempt=retry_mgr.total_retries + 1,
                            max_attempts=retry_mgr.config.max_retries,
                        )
                        retry_mgr.record_attempt(e)
                        state.model_retry_count = retry_mgr.model_retry_count
                        state.network_retry_count = retry_mgr.network_retry_count
                        event_bus.emit(
                            ObservabilityEventType.RETRY_ATTEMPT,
                            attempt=retry_mgr.total_retries,
                            reason=str(e),
                            is_network=is_network,
                        )
                        await retry_mgr.wait(e)
                        event_bus.emit(
                            ObservabilityEventType.CHECKPOINT_SAVED,
                            checkpoint=state.checkpoint,
                            token_count=state.token_count,
                        )
                        update_checkpoint(state)
                        state.resumed = True
                        event_bus.emit(ObservabilityEventType.RETRY_END, success=False)
                        continue
                    else:
                        # Try next fallback
                        event_bus.emit(
                            ObservabilityEventType.FALLBACK_END, index=fallback_idx
                        )
                        break

        # All fallbacks exhausted
        event_bus.emit(
            ObservabilityEventType.RETRY_GIVE_UP,
            attempts=retry_mgr.total_retries,
            last_error=str(errors[-1]) if errors else None,
        )
        event_bus.emit(ObservabilityEventType.SESSION_END)
        if errors:
            raise errors[-1]
        raise RuntimeError("All streams and fallbacks exhausted")

    return Stream(
        iterator=run_stream(),
        state=state,
        abort=abort,
        errors=errors,
    )
