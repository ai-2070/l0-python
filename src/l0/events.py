from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from uuid6 import uuid7

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Event Types (matches TS EventType - UPPER_CASE values)
# ─────────────────────────────────────────────────────────────────────────────


class ObservabilityEventType(str, Enum):
    # Session
    SESSION_START = "SESSION_START"
    ATTEMPT_START = "ATTEMPT_START"
    SESSION_END = "SESSION_END"
    SESSION_SUMMARY = "SESSION_SUMMARY"

    # Stream
    STREAM_INIT = "STREAM_INIT"
    STREAM_READY = "STREAM_READY"
    TOKEN = "TOKEN"

    # Adapter
    ADAPTER_DETECTED = "ADAPTER_DETECTED"
    ADAPTER_WRAP_START = "ADAPTER_WRAP_START"
    ADAPTER_WRAP_END = "ADAPTER_WRAP_END"

    # Timeout
    TIMEOUT_START = "TIMEOUT_START"
    TIMEOUT_RESET = "TIMEOUT_RESET"
    TIMEOUT_TRIGGERED = "TIMEOUT_TRIGGERED"

    # Network
    NETWORK_ERROR = "NETWORK_ERROR"
    NETWORK_RECOVERY = "NETWORK_RECOVERY"
    CONNECTION_DROPPED = "CONNECTION_DROPPED"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"

    # Abort
    ABORT_REQUESTED = "ABORT_REQUESTED"
    ABORT_COMPLETED = "ABORT_COMPLETED"

    # Tool
    TOOL_REQUESTED = "TOOL_REQUESTED"
    TOOL_START = "TOOL_START"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_ERROR = "TOOL_ERROR"
    TOOL_COMPLETED = "TOOL_COMPLETED"

    # Guardrail
    GUARDRAIL_PHASE_START = "GUARDRAIL_PHASE_START"
    GUARDRAIL_PHASE_END = "GUARDRAIL_PHASE_END"
    GUARDRAIL_RULE_START = "GUARDRAIL_RULE_START"
    GUARDRAIL_RULE_RESULT = "GUARDRAIL_RULE_RESULT"
    GUARDRAIL_RULE_END = "GUARDRAIL_RULE_END"
    GUARDRAIL_CALLBACK_START = "GUARDRAIL_CALLBACK_START"
    GUARDRAIL_CALLBACK_END = "GUARDRAIL_CALLBACK_END"

    # Drift
    DRIFT_CHECK_RESULT = "DRIFT_CHECK_RESULT"
    DRIFT_CHECK_SKIPPED = "DRIFT_CHECK_SKIPPED"

    # Checkpoint
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"

    # Resume
    RESUME_START = "RESUME_START"

    # Retry
    RETRY_START = "RETRY_START"
    RETRY_ATTEMPT = "RETRY_ATTEMPT"
    RETRY_END = "RETRY_END"
    RETRY_GIVE_UP = "RETRY_GIVE_UP"
    RETRY_FN_START = "RETRY_FN_START"
    RETRY_FN_RESULT = "RETRY_FN_RESULT"
    RETRY_FN_ERROR = "RETRY_FN_ERROR"

    # Fallback
    FALLBACK_START = "FALLBACK_START"
    FALLBACK_MODEL_SELECTED = "FALLBACK_MODEL_SELECTED"
    FALLBACK_END = "FALLBACK_END"

    # Completion
    FINALIZATION_START = "FINALIZATION_START"
    FINALIZATION_END = "FINALIZATION_END"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"

    # Consensus
    CONSENSUS_START = "CONSENSUS_START"
    CONSENSUS_STREAM_START = "CONSENSUS_STREAM_START"
    CONSENSUS_STREAM_END = "CONSENSUS_STREAM_END"
    CONSENSUS_OUTPUT_COLLECTED = "CONSENSUS_OUTPUT_COLLECTED"
    CONSENSUS_ANALYSIS = "CONSENSUS_ANALYSIS"
    CONSENSUS_RESOLUTION = "CONSENSUS_RESOLUTION"
    CONSENSUS_END = "CONSENSUS_END"

    # Structured output
    PARSE_START = "PARSE_START"
    PARSE_END = "PARSE_END"
    PARSE_ERROR = "PARSE_ERROR"
    SCHEMA_VALIDATION_START = "SCHEMA_VALIDATION_START"
    SCHEMA_VALIDATION_END = "SCHEMA_VALIDATION_END"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    AUTO_CORRECT_START = "AUTO_CORRECT_START"
    AUTO_CORRECT_END = "AUTO_CORRECT_END"
    # Alternate naming for compatibility with TS
    STRUCTURED_PARSE_START = "STRUCTURED_PARSE_START"
    STRUCTURED_PARSE_END = "STRUCTURED_PARSE_END"
    STRUCTURED_PARSE_ERROR = "STRUCTURED_PARSE_ERROR"
    STRUCTURED_VALIDATION_START = "STRUCTURED_VALIDATION_START"
    STRUCTURED_VALIDATION_END = "STRUCTURED_VALIDATION_END"
    STRUCTURED_VALIDATION_ERROR = "STRUCTURED_VALIDATION_ERROR"
    STRUCTURED_AUTO_CORRECT_START = "STRUCTURED_AUTO_CORRECT_START"
    STRUCTURED_AUTO_CORRECT_END = "STRUCTURED_AUTO_CORRECT_END"

    # Continuation
    CONTINUATION_START = "CONTINUATION_START"
    # Alternate naming for compatibility with TS


# ─────────────────────────────────────────────────────────────────────────────
# Observability Event (matches TS L0ObservabilityEvent)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ObservabilityEvent:
    type: ObservabilityEventType
    ts: float
    stream_id: str
    context: dict[str, Any] = field(
        default_factory=dict
    )  # User-provided metadata (request_id, tenant, etc.)
    meta: dict[str, Any] = field(
        default_factory=dict
    )  # Event-specific data (attempt, reason, etc.)


class EventBus:
    """Central event bus for all L0 observability."""

    def __init__(
        self,
        handler: Callable[[ObservabilityEvent], None] | None = None,
        context: dict[str, Any] | None = None,
    ):
        self._handler = handler
        self._stream_id = str(uuid7())
        self._context = context or {}

    @property
    def stream_id(self) -> str:
        return self._stream_id

    def emit(self, event_type: ObservabilityEventType, **event_meta: Any) -> None:
        if not self._handler:
            return

        event = ObservabilityEvent(
            type=event_type,
            ts=time.time() * 1000,
            stream_id=self._stream_id,
            context=self._context.copy(),
            meta=event_meta,
        )
        try:
            self._handler(event)
        except Exception:
            # Callback errors are non-fatal - log but don't crash the stream
            _logger.debug("Event handler failed for %s", event_type, exc_info=True)
