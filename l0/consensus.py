"""Multi-model consensus utilities for L0."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Awaitable, Callable, Literal, TypeVar

T = TypeVar("T")

Strategy = Literal["unanimous", "majority", "best"]


async def consensus(
    tasks: list[Callable[[], Awaitable[T]]],
    strategy: Strategy = "majority",
) -> T:
    """Run multiple tasks and resolve consensus.

    Args:
        tasks: List of async callables that return comparable results
        strategy: Consensus strategy to use:
            - "unanimous": All results must match
            - "majority": Most common result wins
            - "best": Return first result (useful with scoring)

    Returns:
        The consensus result

    Raises:
        ValueError: If consensus cannot be reached
        RuntimeError: If no tasks provided
    """
    if not tasks:
        raise RuntimeError("No tasks provided")

    results = await asyncio.gather(*[t() for t in tasks])

    match strategy:
        case "unanimous":
            # All results must be equal
            if len(set(str(r) for r in results)) == 1:
                return results[0]
            raise ValueError("No unanimous consensus: results differ")

        case "majority":
            # Most common result wins
            counter = Counter(str(r) for r in results)
            winner, count = counter.most_common(1)[0]

            # Check if majority actually exists (more than half)
            if count <= len(results) // 2:
                raise ValueError(
                    f"No majority consensus: highest count {count}/{len(results)}"
                )

            # Return the actual result object, not the string
            for r in results:
                if str(r) == winner:
                    return r

            raise ValueError("No majority consensus")

        case "best":
            # First result wins (caller can pre-sort by score if needed)
            return results[0]

        case _:
            raise ValueError(f"Unknown strategy: {strategy}")
