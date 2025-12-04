"""Main L0 runtime engine."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from .adapters import Adapters
from .continuation import ContinuationConfig, deduplicate_continuation, detect_overlap
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


def _validate_checkpoint(
    checkpoint: str,
    guardrails: list[GuardrailRule],
    event_bus: EventBus,
) -> bool:
    """Validate checkpoint content against guardrails.

    Returns True if checkpoint is valid for continuation, False otherwise.
    """
    if not checkpoint or not guardrails:
        return True

    # Create temporary state with checkpoint content
    temp_state = State(content=checkpoint, completed=False)

    event_bus.emit(
        ObservabilityEventType.CHECKPOINT_START,
        checkpoint_length=len(checkpoint),
    )

    has_fatal_violation = False

    for rule in guardrails:
        # Only check streaming rules (completion rules need completed=True)
        if not rule.streaming:
            continue

        violations = rule.check(temp_state)
        for v in violations:
            if v.severity == "error":
                has_fatal_violation = True
                event_bus.emit(
                    ObservabilityEventType.GUARDRAIL_RULE_RESULT,
                    rule_id=rule.name,
                    violations=[v.__dict__],
                    checkpoint_validation=True,
                )

    event_bus.emit(
        ObservabilityEventType.CHECKPOINT_END,
        valid=not has_fatal_violation,
    )

    return not has_fatal_violation


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
    buffer_tool_calls: bool = False,
    continue_from_last_good_token: ContinuationConfig | bool = True,
    build_continuation_prompt: Callable[[str], str] | None = None,
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
        buffer_tool_calls: Buffer tool call arguments until complete (default: False)
        continue_from_last_good_token: Resume from checkpoint on retry (default: True)
        build_continuation_prompt: Callback to modify prompt for continuation

    Returns:
        Stream - async iterator with .state, .abort(), and .read()
    """
    fallbacks = fallbacks or []
    guardrails = guardrails or []

    # Normalize continuation config
    continuation_config: ContinuationConfig | None = None
    if continue_from_last_good_token is True:
        continuation_config = ContinuationConfig.default()
    elif continue_from_last_good_token is False:
        continuation_config = None
    elif isinstance(continue_from_last_good_token, ContinuationConfig):
        continuation_config = (
            continue_from_last_good_token
            if continue_from_last_good_token.enabled
            else None
        )

    state = create_state()
    retry_mgr = RetryManager(retry)
    event_bus = EventBus(on_event, meta=meta)
    errors: list[Exception] = []
    aborted = False

    # Track if we should use continuation on next retry
    use_continuation_on_retry = False
    pending_checkpoint: str | None = None

    logger.debug(f"Starting L0 stream: {event_bus.stream_id}")
    event_bus.emit(ObservabilityEventType.SESSION_START, session_id=event_bus.stream_id)

    def abort() -> None:
        nonlocal aborted
        aborted = True
        state.aborted = True
        logger.debug("Abort requested")
        event_bus.emit(ObservabilityEventType.ABORT_REQUESTED, source="user")

    async def run_stream() -> AsyncIterator[Event]:
        nonlocal state, use_continuation_on_retry, pending_checkpoint

        streams = [stream] + fallbacks

        # Determine checkpoint interval
        checkpoint_interval = 5
        if continuation_config:
            checkpoint_interval = continuation_config.checkpoint_interval

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

                # Check if we should emit continuation from checkpoint
                should_continue_from_checkpoint = (
                    use_continuation_on_retry
                    and pending_checkpoint
                    and continuation_config is not None
                )

                if should_continue_from_checkpoint and pending_checkpoint:
                    # Validate checkpoint before using
                    checkpoint_valid = True
                    if continuation_config.validate_checkpoint and guardrails:
                        checkpoint_valid = _validate_checkpoint(
                            pending_checkpoint, guardrails, event_bus
                        )

                    if checkpoint_valid:
                        event_bus.emit(
                            ObservabilityEventType.CONTINUATION_START,
                            checkpoint_length=len(pending_checkpoint),
                        )

                        # Emit checkpoint content as tokens first
                        state.resume_point = pending_checkpoint
                        state.resume_from = 0
                        state.continuation_used = True

                        # Yield checkpoint as a single token event
                        # (The content is already in state from before the failure)
                        yield Event(
                            type=EventType.TOKEN, text=""
                        )  # Empty - content already tracked

                        # Call build_continuation_prompt if provided
                        if build_continuation_prompt:
                            # This allows the user to modify the prompt for the next call
                            build_continuation_prompt(pending_checkpoint)

                        event_bus.emit(
                            ObservabilityEventType.CONTINUATION_END,
                            checkpoint_length=len(pending_checkpoint),
                        )

                        logger.debug(
                            f"Continuing from checkpoint ({len(pending_checkpoint)} chars)"
                        )
                    else:
                        # Checkpoint invalid - start fresh
                        logger.debug("Checkpoint validation failed, starting fresh")
                        state.content = ""
                        state.token_count = 0
                        pending_checkpoint = None

                use_continuation_on_retry = False

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

                    # Tool call buffering state (when buffer_tool_calls=True)
                    # Maps tool call index -> {id, name, arguments}
                    tool_call_buffers: dict[int, dict[str, Any]] = {}

                    async def emit_buffered_tool_calls() -> AsyncIterator[Event]:
                        """Emit all buffered tool calls."""
                        for idx in sorted(tool_call_buffers.keys()):
                            buffered = tool_call_buffers[idx]
                            yield Event(
                                type=EventType.TOOL_CALL,
                                data={
                                    "id": buffered.get("id"),
                                    "name": buffered.get("name"),
                                    "arguments": buffered.get("arguments", ""),
                                },
                            )
                        tool_call_buffers.clear()

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
                            token_text = event.text

                            # Handle deduplication for continuation
                            if (
                                state.continuation_used
                                and not state.deduplication_applied
                                and continuation_config
                                and continuation_config.deduplicate
                                and state.resume_point
                            ):
                                # Check for overlap with checkpoint
                                event_bus.emit(
                                    ObservabilityEventType.DEDUPLICATION_START
                                )
                                overlap_result = detect_overlap(
                                    state.resume_point,
                                    token_text,
                                    continuation_config.deduplication_options,
                                )

                                if overlap_result.has_overlap:
                                    token_text = overlap_result.deduplicated
                                    state.deduplication_applied = True
                                    state.overlap_removed = overlap_result.overlap_text
                                    event_bus.emit(
                                        ObservabilityEventType.DEDUPLICATION_END,
                                        overlap_detected=True,
                                        overlap_length=overlap_result.overlap_length,
                                        overlap_text=overlap_result.overlap_text,
                                    )
                                    logger.debug(
                                        f"Deduplication removed {overlap_result.overlap_length} chars overlap"
                                    )

                                    # Update the event with deduplicated text
                                    if not token_text:
                                        # Entire token was overlap, skip it
                                        continue
                                    event = Event(type=EventType.TOKEN, text=token_text)
                                else:
                                    state.deduplication_applied = (
                                        True  # Mark as checked
                                    )
                                    event_bus.emit(
                                        ObservabilityEventType.DEDUPLICATION_END,
                                        overlap_detected=False,
                                    )

                            if not first_token_received and inter_timeout is not None:
                                # First token received, switch to inter-token timeout
                                event_bus.emit(
                                    ObservabilityEventType.TIMEOUT_RESET,
                                    timeout_type="inter_token",
                                    token_index=state.token_count,
                                )
                            first_token_received = True
                            append_token(state, token_text)

                            # Save checkpoint periodically
                            if state.token_count % checkpoint_interval == 0:
                                update_checkpoint(state)

                            # Check guardrails periodically
                            if (
                                state.token_count % checkpoint_interval == 0
                                and guardrails
                            ):
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

                        # Handle tool call buffering
                        if buffer_tool_calls and event.type == EventType.TOOL_CALL:
                            # Buffer tool call data by index
                            tc_data = event.data or {}
                            # Use 'index' if provided, otherwise use length as index
                            tc_index = tc_data.get("index", len(tool_call_buffers))

                            if tc_index not in tool_call_buffers:
                                # New tool call - initialize buffer
                                tool_call_buffers[tc_index] = {
                                    "id": tc_data.get("id"),
                                    "name": tc_data.get("name"),
                                    "arguments": tc_data.get("arguments") or "",
                                }
                            else:
                                # Existing tool call - accumulate arguments
                                buffered = tool_call_buffers[tc_index]
                                # Update id/name if provided (first chunk has these)
                                if tc_data.get("id"):
                                    buffered["id"] = tc_data["id"]
                                if tc_data.get("name"):
                                    buffered["name"] = tc_data["name"]
                                # Append arguments
                                if tc_data.get("arguments"):
                                    buffered["arguments"] += tc_data["arguments"]

                            # Don't yield partial tool calls - they'll be emitted
                            # when stream ends
                            continue

                        yield event

                    # Emit any buffered tool calls at end of stream
                    if buffer_tool_calls and tool_call_buffers:
                        async for tc_event in emit_buffered_tool_calls():
                            yield tc_event

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

                        # Save checkpoint for potential continuation
                        update_checkpoint(state)
                        event_bus.emit(
                            ObservabilityEventType.CHECKPOINT_SAVED,
                            checkpoint=state.checkpoint,
                            token_count=state.token_count,
                        )

                        # Enable continuation on next retry if configured
                        if continuation_config and state.checkpoint:
                            use_continuation_on_retry = True
                            pending_checkpoint = state.checkpoint

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
