"""Pydantic models for L0 state machine types.

These models mirror the dataclasses in l0.state_machine for runtime validation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class RuntimeStateModel(str, Enum):
    """Runtime state constants."""

    INIT = "init"
    WAITING_FOR_TOKEN = "waiting_for_token"
    STREAMING = "streaming"
    TOOL_CALL_DETECTED = "tool_call_detected"
    CONTINUATION_MATCHING = "continuation_matching"
    CHECKPOINT_VERIFYING = "checkpoint_verifying"
    RETRYING = "retrying"
    FALLBACK = "fallback"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    ERROR = "error"


class StateTransitionModel(BaseModel):
    """Pydantic model for a state transition record."""

    model_config = ConfigDict(extra="forbid")

    from_state: RuntimeStateModel
    to_state: RuntimeStateModel
    timestamp: float
