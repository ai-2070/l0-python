"""Event Sourcing Types for Atomic, Replayable Operations.

Key insight: Replayability MUST ignore external sources of non-determinism.
In replay mode, we're a pure faucet over stored events - no network, no retries,
no timeouts, no fallbacks, no live guardrail evaluation.

Derived computations (guardrails, drift, retries) are stored AS events,
not recomputed on replay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from uuid6 import uuid7

# ─────────────────────────────────────────────────────────────────────────────
# Event Type Constants
# ─────────────────────────────────────────────────────────────────────────────


class RecordedEventType(str, Enum):
    """Recorded event types for event sourcing."""

    START = "START"
    TOKEN = "TOKEN"
    CHECKPOINT = "CHECKPOINT"
    GUARDRAIL = "GUARDRAIL"
    DRIFT = "DRIFT"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    CONTINUATION = "CONTINUATION"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# Serialized Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SerializedOptions:
    """Serialized options for event storage.

    Strips functions and non-serializable fields.
    """

    prompt: str | None = None
    model: str | None = None
    retry: dict[str, Any] | None = None
    timeout: dict[str, Any] | None = None
    check_intervals: dict[str, Any] | None = None
    continue_from_checkpoint: bool = False
    detect_drift: bool = False
    detect_zero_tokens: bool = False
    fallback_count: int = 0
    guardrail_count: int = 0
    metadata: dict[str, Any] | None = None


@dataclass
class SerializedError:
    """Serialized error for event storage."""

    name: str
    message: str
    code: str | None = None
    stack: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class GuardrailEventResult:
    """Guardrail evaluation result for event storage."""

    violations: list[dict[str, Any]] = field(default_factory=list)
    should_retry: bool = False
    should_halt: bool = False


@dataclass
class DriftEventResult:
    """Drift detection result for event storage."""

    detected: bool = False
    types: list[str] = field(default_factory=list)
    confidence: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Event Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StartEvent:
    """Stream execution started."""

    type: Literal[RecordedEventType.START] = RecordedEventType.START
    ts: float = 0.0  # Unix timestamp in milliseconds
    options: SerializedOptions = field(default_factory=SerializedOptions)


@dataclass
class TokenEvent:
    """Token received from stream."""

    type: Literal[RecordedEventType.TOKEN] = RecordedEventType.TOKEN
    ts: float = 0.0
    value: str = ""
    index: int = 0  # Zero-based token index


@dataclass
class CheckpointEvent:
    """Checkpoint saved (for continuation support)."""

    type: Literal[RecordedEventType.CHECKPOINT] = RecordedEventType.CHECKPOINT
    ts: float = 0.0
    at: int = 0  # Token index at checkpoint
    content: str = ""  # Accumulated content at checkpoint


@dataclass
class GuardrailEvent:
    """Guardrail evaluation occurred.

    Stored as event because it's a derived computation.
    """

    type: Literal[RecordedEventType.GUARDRAIL] = RecordedEventType.GUARDRAIL
    ts: float = 0.0
    at: int = 0  # Token index when check occurred
    result: GuardrailEventResult = field(default_factory=GuardrailEventResult)


@dataclass
class DriftEvent:
    """Drift detection occurred.

    Stored as event because it's a derived computation.
    """

    type: Literal[RecordedEventType.DRIFT] = RecordedEventType.DRIFT
    ts: float = 0.0
    at: int = 0  # Token index when check occurred
    result: DriftEventResult = field(default_factory=DriftEventResult)


@dataclass
class RetryEvent:
    """Retry triggered."""

    type: Literal[RecordedEventType.RETRY] = RecordedEventType.RETRY
    ts: float = 0.0
    reason: str = ""
    attempt: int = 1  # Attempt number (1-based)
    counts_toward_limit: bool = True  # Whether this counts toward model retry limit


@dataclass
class FallbackEvent:
    """Fallback to next stream triggered."""

    type: Literal[RecordedEventType.FALLBACK] = RecordedEventType.FALLBACK
    ts: float = 0.0
    to: int = 0  # Index of stream we're falling back to (1-based for fallbacks)


@dataclass
class ContinuationEvent:
    """Continuation from checkpoint used."""

    type: Literal[RecordedEventType.CONTINUATION] = RecordedEventType.CONTINUATION
    ts: float = 0.0
    checkpoint: str = ""  # Checkpoint content used for continuation
    at: int = 0  # Token index of checkpoint


@dataclass
class CompleteEvent:
    """Stream completed successfully."""

    type: Literal[RecordedEventType.COMPLETE] = RecordedEventType.COMPLETE
    ts: float = 0.0
    content: str = ""  # Final accumulated content
    token_count: int = 0  # Total token count


@dataclass
class ErrorEvent:
    """Stream failed with error."""

    type: Literal[RecordedEventType.ERROR] = RecordedEventType.ERROR
    ts: float = 0.0
    error: SerializedError = field(default_factory=lambda: SerializedError("", ""))
    failure_type: str = ""  # What went wrong - the root cause
    recovery_strategy: str = ""  # What was decided to do next
    policy: str = ""  # Policy that determined the recovery strategy


# Union type for all recorded events
RecordedEvent = (
    StartEvent
    | TokenEvent
    | CheckpointEvent
    | GuardrailEvent
    | DriftEvent
    | RetryEvent
    | FallbackEvent
    | ContinuationEvent
    | CompleteEvent
    | ErrorEvent
)


# ─────────────────────────────────────────────────────────────────────────────
# Event Envelope
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EventEnvelope:
    """Event envelope with stream identity."""

    stream_id: str
    seq: int  # Sequence number within stream (0-based)
    event: RecordedEvent


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """Snapshot of state at a point in time.

    Used for faster replay of long streams.
    """

    stream_id: str
    seq: int  # Sequence number this snapshot is valid at
    ts: float  # Unix timestamp when snapshot was taken
    content: str
    token_count: int
    checkpoint: str
    violations: list[dict[str, Any]] = field(default_factory=list)
    drift_detected: bool = False
    retry_attempts: int = 0
    network_retry_count: int = 0
    fallback_index: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Replayed State
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReplayedState:
    """State reconstructed from replay."""

    content: str = ""
    token_count: int = 0
    checkpoint: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)
    drift_detected: bool = False
    retry_attempts: int = 0
    network_retry_count: int = 0
    fallback_index: int = 0
    completed: bool = False
    error: SerializedError | None = None
    start_ts: float = 0.0
    end_ts: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReplayOptions:
    """Options for replay mode."""

    stream_id: str
    speed: float = 0.0  # Playback speed (0 = instant, 1 = real-time)
    fire_callbacks: bool = False  # Whether to fire monitoring callbacks
    from_seq: int = 0  # Start replay from this sequence number
    to_seq: int | None = None  # Stop replay at this sequence number


@dataclass
class RecordOptions:
    """Options for record mode."""

    stream_id: str | None = None  # Custom stream ID (auto-generated if not provided)
    save_snapshots: bool = False  # Whether to also save snapshots periodically
    snapshot_interval: int = 100  # Snapshot interval (every N events)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def serialize_error(error: Exception) -> SerializedError:
    """Serialize an Exception to SerializedError."""
    import traceback

    return SerializedError(
        name=type(error).__name__,
        message=str(error),
        code=getattr(error, "code", None),
        stack="".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        metadata=getattr(error, "metadata", None),
    )


def deserialize_error(stored: SerializedError) -> Exception:
    """Deserialize a SerializedError back to Exception.

    Note: The original error type cannot be fully restored,
    but the name is preserved in the error's attributes.
    """
    error = Exception(stored.message)
    setattr(error, "original_type", stored.name)
    if stored.code:
        setattr(error, "code", stored.code)
    if stored.stack:
        setattr(error, "stack", stored.stack)
    if stored.metadata:
        setattr(error, "metadata", stored.metadata)
    return error


def generate_stream_id() -> str:
    """Generate a unique stream ID using UUID7."""
    return str(uuid7())


def now_ms() -> float:
    """Get current timestamp in milliseconds."""
    return time.time() * 1000
