"""L0 - Reliability layer for AI/LLM streaming."""

from .adapters import (
    Adapter,
    LiteLLMAdapter,
    OpenAIAdapter,
    detect_adapter,
    register_adapter,
)
from .consensus import consensus
from .events import EventBus, ObservabilityEvent, ObservabilityEventType
from .guardrails import (
    GuardrailRule,
    GuardrailViolation,
    check_guardrails,
    json_rule,
    pattern_rule,
    recommended_guardrails,
    repetition_rule,
    stall_rule,
    strict_guardrails,
    strict_json_rule,
    zero_output_rule,
)
from .logging import enable_debug
from .parallel import batched, parallel, race
from .runtime import l0
from .stream import consume_stream, get_text
from .structured import structured
from .types import (
    BackoffStrategy,
    ErrorCategory,
    EventType,
    L0Event,
    L0Options,
    L0Result,
    L0State,
    RetryConfig,
    TimeoutConfig,
)
from .version import __version__

__all__ = [
    # Version
    "__version__",
    # Core
    "l0",
    "L0Event",
    "L0State",
    "L0Options",
    "L0Result",
    "RetryConfig",
    "TimeoutConfig",
    "EventType",
    "ErrorCategory",
    "BackoffStrategy",
    # Events
    "ObservabilityEvent",
    "ObservabilityEventType",
    "EventBus",
    # Stream
    "consume_stream",
    "get_text",
    # Adapters
    "Adapter",
    "OpenAIAdapter",
    "LiteLLMAdapter",
    "register_adapter",
    "detect_adapter",
    # Guardrails
    "GuardrailRule",
    "GuardrailViolation",
    "check_guardrails",
    "json_rule",
    "strict_json_rule",
    "pattern_rule",
    "zero_output_rule",
    "stall_rule",
    "repetition_rule",
    "recommended_guardrails",
    "strict_guardrails",
    # Structured
    "structured",
    # Parallel
    "parallel",
    "race",
    "batched",
    # Consensus
    "consensus",
    # Debug
    "enable_debug",
]
