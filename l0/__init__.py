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
from .runtime import _internal_run
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


# Public API: run() is the main entrypoint, l0() is an alias
async def run(options: L0Options) -> L0Result:
    """Main L0 streaming runtime with guardrails and retry logic.

    This is the primary entrypoint for the L0 library. It wraps LLM streams
    with deterministic behavior, retry logic, fallbacks, and guardrails.

    Args:
        options: L0Options configuration object

    Returns:
        L0Result with stream, state, and abort function

    Example:
        ```python
        import l0

        result = await l0.run(l0.L0Options(
            stream=lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            ),
            guardrails=l0.recommended_guardrails(),
        ))

        async for event in result.stream:
            if event.type == l0.EventType.TOKEN:
                print(event.value, end="")
        ```
    """
    return await _internal_run(options)


# Alias for backwards compatibility and convenience
l0 = run

__all__ = [
    # Version
    "__version__",
    # Core
    "run",
    "l0",  # Alias to run()
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
