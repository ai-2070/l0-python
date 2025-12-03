"""Multi-model consensus utilities for L0."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

from .events import EventBus, ObservabilityEvent, ObservabilityEventType

T = TypeVar("T")

Strategy = Literal["unanimous", "majority", "best"]


async def consensus(
    tasks: list[Callable[[], Awaitable[T]]],
    strategy: Strategy = "majority",
    on_event: Callable[[ObservabilityEvent], None] | None = None,
) -> T:
    """Run multiple tasks and resolve consensus.

    Args:
        tasks: List of async callables that return comparable results
        strategy: Consensus strategy to use:
            - "unanimous": All results must match
            - "majority": Most common result wins
            - "best": Return first result (useful with scoring)
        on_event: Optional callback for observability events

    Returns:
        The consensus result

    Raises:
        ValueError: If consensus cannot be reached
        RuntimeError: If no tasks provided
    """
    if not tasks:
        raise RuntimeError("No tasks provided")

    event_bus = EventBus(on_event)
    event_bus.emit(ObservabilityEventType.CONSENSUS_START)
    consensus_start = time.time()

    # Run all tasks and track per-stream events
    async def run_with_events(idx: int, task: Callable[[], Awaitable[T]]) -> T:
        event_bus.emit(
            ObservabilityEventType.CONSENSUS_STREAM_START,
            stream_index=idx,
        )
        stream_start = time.time()
        try:
            result = await task()
            duration = (time.time() - stream_start) * 1000
            event_bus.emit(
                ObservabilityEventType.CONSENSUS_STREAM_END,
                stream_index=idx,
                duration_ms=duration,
                status="success",
            )
            event_bus.emit(
                ObservabilityEventType.CONSENSUS_OUTPUT_COLLECTED,
                stream_index=idx,
                length=len(str(result)),
                has_errors=False,
            )
            return result
        except Exception as e:
            duration = (time.time() - stream_start) * 1000
            event_bus.emit(
                ObservabilityEventType.CONSENSUS_STREAM_END,
                stream_index=idx,
                duration_ms=duration,
                status="error",
                error=str(e),
            )
            raise

    results = await asyncio.gather(
        *[run_with_events(i, t) for i, t in enumerate(tasks)]
    )

    # Calculate similarity for analysis event
    unique_results = set(str(r) for r in results)
    agreement_ratio = 1.0 - (len(unique_results) - 1) / max(len(results), 1)

    event_bus.emit(
        ObservabilityEventType.CONSENSUS_ANALYSIS,
        agreement_ratio=agreement_ratio,
        strategy=strategy,
        unique_results=len(unique_results),
        total_results=len(results),
    )

    match strategy:
        case "unanimous":
            # All results must be equal
            if len(unique_results) == 1:
                event_bus.emit(
                    ObservabilityEventType.CONSENSUS_RESOLUTION,
                    method="unanimous",
                    confidence=1.0,
                )
                duration = (time.time() - consensus_start) * 1000
                event_bus.emit(
                    ObservabilityEventType.CONSENSUS_END,
                    status="success",
                    confidence=1.0,
                    duration_ms=duration,
                )
                return results[0]
            event_bus.emit(
                ObservabilityEventType.CONSENSUS_END,
                status="failed",
                confidence=0.0,
                duration_ms=(time.time() - consensus_start) * 1000,
            )
            raise ValueError("No unanimous consensus: results differ")

        case "majority":
            # Most common result wins
            counter = Counter(str(r) for r in results)
            winner, count = counter.most_common(1)[0]
            confidence = count / len(results)

            # Check if majority actually exists (more than half)
            if count <= len(results) // 2:
                event_bus.emit(
                    ObservabilityEventType.CONSENSUS_END,
                    status="failed",
                    confidence=confidence,
                    duration_ms=(time.time() - consensus_start) * 1000,
                )
                raise ValueError(
                    f"No majority consensus: highest count {count}/{len(results)}"
                )

            # Return the actual result object, not the string
            for r in results:
                if str(r) == winner:
                    event_bus.emit(
                        ObservabilityEventType.CONSENSUS_RESOLUTION,
                        method="majority",
                        confidence=confidence,
                    )
                    duration = (time.time() - consensus_start) * 1000
                    event_bus.emit(
                        ObservabilityEventType.CONSENSUS_END,
                        status="success",
                        confidence=confidence,
                        duration_ms=duration,
                    )
                    return r

            event_bus.emit(
                ObservabilityEventType.CONSENSUS_END,
                status="failed",
                confidence=0.0,
                duration_ms=(time.time() - consensus_start) * 1000,
            )
            raise ValueError("No majority consensus")

        case "best":
            # First result wins (caller can pre-sort by score if needed)
            event_bus.emit(
                ObservabilityEventType.CONSENSUS_RESOLUTION,
                method="best",
                final_selection=0,
                confidence=1.0,
            )
            duration = (time.time() - consensus_start) * 1000
            event_bus.emit(
                ObservabilityEventType.CONSENSUS_END,
                status="success",
                confidence=1.0,
                duration_ms=duration,
            )
            return results[0]

        case _:
            event_bus.emit(
                ObservabilityEventType.CONSENSUS_END,
                status="failed",
                duration_ms=(time.time() - consensus_start) * 1000,
            )
            raise ValueError(f"Unknown strategy: {strategy}")
