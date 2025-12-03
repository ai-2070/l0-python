from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from uuid6 import uuid7

# ─────────────────────────────────────────────────────────────────────────────
# Event Types (matches TS EventType - UPPER_CASE values)
# ─────────────────────────────────────────────────────────────────────────────


class ObservabilityEventType(str, Enum):
    # Session
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"

    # Stream
    STREAM_INIT = "STREAM_INIT"
    STREAM_READY = "STREAM_READY"

    # Retry
    RETRY_START = "RETRY_START"
    RETRY_ATTEMPT = "RETRY_ATTEMPT"
    RETRY_END = "RETRY_END"
    RETRY_GIVE_UP = "RETRY_GIVE_UP"

    # Fallback
    FALLBACK_START = "FALLBACK_START"
    FALLBACK_END = "FALLBACK_END"

    # Guardrail
    GUARDRAIL_PHASE_START = "GUARDRAIL_PHASE_START"
    GUARDRAIL_RULE_RESULT = "GUARDRAIL_RULE_RESULT"
    GUARDRAIL_PHASE_END = "GUARDRAIL_PHASE_END"

    # Drift
    DRIFT_CHECK_RESULT = "DRIFT_CHECK_RESULT"

    # Network
    NETWORK_ERROR = "NETWORK_ERROR"
    NETWORK_RECOVERY = "NETWORK_RECOVERY"

    # Checkpoint
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"

    # Completion
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# Observability Event (matches TS L0ObservabilityEvent)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ObservabilityEvent:
    type: ObservabilityEventType
    ts: float
    stream_id: str
    meta: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Central event bus for all L0 observability."""

    def __init__(
        self,
        handler: Callable[[ObservabilityEvent], None] | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self._handler = handler
        self._stream_id = str(uuid7())
        self._meta = meta or {}

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
            meta={**self._meta, **event_meta},
        )
        self._handler(event)
