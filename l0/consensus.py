"""Multi-model consensus utilities for L0.

Multi-generation consensus for high-confidence results. Run multiple generations,
compare outputs, and resolve disagreements.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

from .events import EventBus, ObservabilityEvent, ObservabilityEventType

T = TypeVar("T")

Strategy = Literal["unanimous", "majority", "weighted", "best"]
ConflictResolution = Literal["vote", "merge", "best", "fail"]
AgreementType = Literal["exact", "similar", "structural", "semantic"]
DisagreementSeverity = Literal["minor", "moderate", "major", "critical"]


# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Agreement:
    """What outputs agreed on."""

    content: Any  # Agreed content
    path: str | None = None  # Field path (for structured)
    count: int = 0  # How many agreed
    ratio: float = 0.0  # Agreement ratio
    indices: list[int] = field(default_factory=list)  # Which outputs agreed
    type: AgreementType = "exact"


@dataclass
class DisagreementValue:
    """A single value in a disagreement."""

    value: Any
    count: int
    indices: list[int]


@dataclass
class Disagreement:
    """Where outputs differed."""

    path: str | None = None  # Field path (for structured)
    values: list[DisagreementValue] = field(default_factory=list)
    severity: DisagreementSeverity = "minor"
    resolution: str | None = None
    resolution_confidence: float | None = None


@dataclass
class ConsensusAnalysis:
    """Detailed statistics about consensus."""

    total_outputs: int = 0
    successful_outputs: int = 0
    failed_outputs: int = 0
    identical_outputs: int = 0
    similarity_matrix: list[list[float]] = field(default_factory=list)
    average_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    total_agreements: int = 0
    total_disagreements: int = 0
    strategy: str = ""
    conflict_resolution: str = ""
    duration_ms: float = 0.0


@dataclass
class FieldConsensusInfo:
    """Per-field consensus information."""

    value: Any
    agreement: float  # 0-1
    count: int
    indices: list[int]


@dataclass
class FieldConsensus:
    """Field-by-field consensus for structured outputs."""

    fields: dict[str, FieldConsensusInfo] = field(default_factory=dict)


@dataclass
class ConsensusOutput:
    """Individual output from a stream."""

    value: Any
    index: int
    success: bool
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class ConsensusResult(Generic[T]):
    """Result of consensus operation."""

    consensus: T  # Final agreed output
    confidence: float  # 0-1 overall confidence
    outputs: list[ConsensusOutput]  # Individual outputs
    agreements: list[Agreement]  # What matched
    disagreements: list[Disagreement]  # What differed
    analysis: ConsensusAnalysis  # Detailed stats
    type: Literal["text", "structured"] = "text"
    field_consensus: FieldConsensus | None = None  # For structured
    status: Literal["success", "partial", "failed"] = "success"


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConsensusPreset:
    """Preset configuration for consensus."""

    strategy: Strategy
    threshold: float
    resolve_conflicts: ConflictResolution
    minimum_agreement: float


# Strict: all must agree
strict_consensus = ConsensusPreset(
    strategy="unanimous",
    threshold=1.0,
    resolve_conflicts="fail",
    minimum_agreement=1.0,
)

# Standard: majority rules (default)
standard_consensus = ConsensusPreset(
    strategy="majority",
    threshold=0.8,
    resolve_conflicts="vote",
    minimum_agreement=0.6,
)

# Lenient: flexible
lenient_consensus = ConsensusPreset(
    strategy="majority",
    threshold=0.7,
    resolve_conflicts="merge",
    minimum_agreement=0.5,
)

# Best: choose highest quality
best_consensus = ConsensusPreset(
    strategy="best",
    threshold=0.5,
    resolve_conflicts="best",
    minimum_agreement=0.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def _calculate_similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings (0-1)."""
    return SequenceMatcher(None, a, b).ratio()


def _build_similarity_matrix(outputs: list[str]) -> list[list[float]]:
    """Build NxN similarity matrix for outputs."""
    n = len(outputs)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            elif j > i:
                sim = _calculate_similarity(outputs[i], outputs[j])
                matrix[i][j] = sim
                matrix[j][i] = sim
    return matrix


def _determine_severity(ratio: float) -> DisagreementSeverity:
    """Determine disagreement severity based on agreement ratio."""
    if ratio >= 0.8:
        return "minor"
    elif ratio >= 0.6:
        return "moderate"
    elif ratio >= 0.4:
        return "major"
    else:
        return "critical"


def quick_consensus(outputs: list[Any], threshold: float = 0.8) -> bool:
    """Quick check if outputs have consensus at given threshold.

    Args:
        outputs: List of outputs to check
        threshold: Minimum agreement ratio (default 0.8 = 80%)

    Returns:
        True if agreement ratio >= threshold
    """
    if not outputs:
        return False

    counter = Counter(str(o) for o in outputs)
    most_common_count = counter.most_common(1)[0][1]
    ratio = most_common_count / len(outputs)
    return ratio >= threshold


def get_consensus_value(outputs: list[T]) -> T | None:
    """Get the most common value from outputs.

    Args:
        outputs: List of outputs

    Returns:
        Most common value, or None if empty
    """
    if not outputs:
        return None

    counter = Counter(str(o) for o in outputs)
    winner = counter.most_common(1)[0][0]

    # Return the actual object, not the string
    for o in outputs:
        if str(o) == winner:
            return o
    return outputs[0]


def validate_consensus(
    result: ConsensusResult[Any],
    min_confidence: float = 0.8,
    max_disagreements: int = 0,
) -> bool:
    """Validate consensus result meets requirements.

    Args:
        result: ConsensusResult to validate
        min_confidence: Minimum confidence required (default 0.8)
        max_disagreements: Maximum major/critical disagreements allowed (default 0)

    Returns:
        True if result meets requirements
    """
    if result.confidence < min_confidence:
        return False

    major_disagreements = sum(
        1 for d in result.disagreements if d.severity in ("major", "critical")
    )
    if major_disagreements > max_disagreements:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main Consensus Function
# ─────────────────────────────────────────────────────────────────────────────


async def consensus(
    tasks: list[Callable[[], Awaitable[T]]],
    *,
    strategy: Strategy = "majority",
    threshold: float = 0.8,
    resolve_conflicts: ConflictResolution = "vote",
    weights: list[float] | None = None,
    minimum_agreement: float = 0.6,
    schema: type[BaseModel] | None = None,
    on_event: Callable[[ObservabilityEvent], None] | None = None,
) -> ConsensusResult[T]:
    """Run multiple tasks and resolve consensus.

    Args:
        tasks: List of async callables that return comparable results (min 2)
        strategy: Consensus strategy:
            - "unanimous": All must agree
            - "majority": Most common result wins
            - "weighted": Weight by model/confidence
            - "best": Choose first/highest quality output
        threshold: Similarity threshold for matching (default 0.8)
        resolve_conflicts: How to resolve disagreements:
            - "vote": Take majority vote
            - "merge": Combine information
            - "best": Choose highest confidence
            - "fail": Throw error on disagreement
        weights: Weights for each task (for weighted strategy)
        minimum_agreement: Minimum agreement ratio required (default 0.6)
        schema: Pydantic schema for structured consensus
        on_event: Optional callback for observability events

    Returns:
        ConsensusResult with consensus value and analysis

    Raises:
        ValueError: If consensus cannot be reached
        RuntimeError: If no tasks provided or fewer than 2
    """
    if not tasks:
        raise RuntimeError("No tasks provided")
    if len(tasks) < 2:
        raise RuntimeError("At least 2 tasks required for consensus")

    event_bus = EventBus(on_event)
    event_bus.emit(ObservabilityEventType.CONSENSUS_START)
    consensus_start = time.time()

    # Initialize weights
    if weights is None:
        weights = [1.0] * len(tasks)
    elif len(weights) != len(tasks):
        raise ValueError("Weights must match number of tasks")

    # Run all tasks and collect outputs
    outputs: list[ConsensusOutput] = []
    successful_values: list[tuple[int, Any, float]] = []  # (index, value, weight)

    async def run_task(idx: int, task: Callable[[], Awaitable[T]]) -> ConsensusOutput:
        event_bus.emit(
            ObservabilityEventType.CONSENSUS_STREAM_START,
            stream_index=idx,
        )
        start = time.time()
        try:
            result = await task()
            duration = (time.time() - start) * 1000
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
            return ConsensusOutput(
                value=result,
                index=idx,
                success=True,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            event_bus.emit(
                ObservabilityEventType.CONSENSUS_STREAM_END,
                stream_index=idx,
                duration_ms=duration,
                status="error",
                error=str(e),
            )
            return ConsensusOutput(
                value=None,
                index=idx,
                success=False,
                error=str(e),
                duration_ms=duration,
            )

    # Run all tasks concurrently
    outputs = await asyncio.gather(*[run_task(i, t) for i, t in enumerate(tasks)])

    # Collect successful outputs
    for out in outputs:
        if out.success:
            successful_values.append((out.index, out.value, weights[out.index]))

    if not successful_values:
        event_bus.emit(
            ObservabilityEventType.CONSENSUS_END,
            status="failed",
            duration_ms=(time.time() - consensus_start) * 1000,
        )
        raise ValueError("All tasks failed, no consensus possible")

    # Convert to strings for comparison
    string_outputs = [str(v) for _, v, _ in successful_values]

    # Build similarity matrix
    similarity_matrix = _build_similarity_matrix(string_outputs)
    flat_similarities = [
        similarity_matrix[i][j]
        for i in range(len(string_outputs))
        for j in range(i + 1, len(string_outputs))
    ]
    avg_similarity = (
        sum(flat_similarities) / len(flat_similarities) if flat_similarities else 1.0
    )
    min_similarity = min(flat_similarities) if flat_similarities else 1.0
    max_similarity = max(flat_similarities) if flat_similarities else 1.0

    # Count identical outputs
    unique_outputs = set(string_outputs)
    identical_count = (
        len(string_outputs) - len(unique_outputs) + 1
        if len(unique_outputs) < len(string_outputs)
        else 0
    )

    event_bus.emit(
        ObservabilityEventType.CONSENSUS_ANALYSIS,
        agreement_ratio=avg_similarity,
        strategy=strategy,
        unique_results=len(unique_outputs),
        total_results=len(string_outputs),
        similarity_matrix=similarity_matrix,
        average_similarity=avg_similarity,
    )

    # Determine consensus based on strategy
    consensus_value: Any = None
    confidence: float = 0.0
    agreements: list[Agreement] = []
    disagreements: list[Disagreement] = []
    status: Literal["success", "partial", "failed"] = "success"

    if strategy == "unanimous":
        # All must match (within threshold)
        all_similar = all(
            similarity_matrix[0][j] >= threshold for j in range(1, len(string_outputs))
        )
        if all_similar:
            consensus_value = successful_values[0][1]
            confidence = min_similarity
            agreements.append(
                Agreement(
                    content=consensus_value,
                    count=len(successful_values),
                    ratio=1.0,
                    indices=[i for i, _, _ in successful_values],
                    type="exact" if len(unique_outputs) == 1 else "similar",
                )
            )
        else:
            if resolve_conflicts == "fail":
                event_bus.emit(
                    ObservabilityEventType.CONSENSUS_END,
                    status="failed",
                    duration_ms=(time.time() - consensus_start) * 1000,
                )
                raise ValueError(
                    "No unanimous consensus: outputs differ beyond threshold"
                )
            # Try to resolve
            consensus_value, confidence = _resolve_conflict(
                successful_values, resolve_conflicts, weights
            )
            status = "partial"

    elif strategy == "majority":
        # Group by similarity
        groups = _group_by_similarity(successful_values, threshold)
        largest_group = max(groups, key=lambda g: sum(w for _, _, w in g))
        group_weight = sum(w for _, _, w in largest_group)
        total_weight = sum(w for _, _, w in successful_values)
        ratio = group_weight / total_weight

        if ratio >= minimum_agreement:
            consensus_value = largest_group[0][1]
            confidence = ratio
            agreements.append(
                Agreement(
                    content=consensus_value,
                    count=len(largest_group),
                    ratio=ratio,
                    indices=[i for i, _, _ in largest_group],
                    type="exact"
                    if len(set(str(v) for _, v, _ in largest_group)) == 1
                    else "similar",
                )
            )
            # Record disagreements
            for group in groups:
                if group != largest_group:
                    disagreements.append(
                        Disagreement(
                            values=[
                                DisagreementValue(
                                    value=v,
                                    count=1,
                                    indices=[i],
                                )
                                for i, v, _ in group
                            ],
                            severity=_determine_severity(ratio),
                        )
                    )
        else:
            if resolve_conflicts == "fail":
                event_bus.emit(
                    ObservabilityEventType.CONSENSUS_END,
                    status="failed",
                    duration_ms=(time.time() - consensus_start) * 1000,
                )
                raise ValueError(
                    f"No majority consensus: highest agreement {ratio:.0%} < {minimum_agreement:.0%}"
                )
            consensus_value, confidence = _resolve_conflict(
                successful_values, resolve_conflicts, weights
            )
            status = "partial"

    elif strategy == "weighted":
        # Weight-based selection
        groups = _group_by_similarity(successful_values, threshold)
        # Find group with highest total weight
        best_group = max(groups, key=lambda g: sum(w for _, _, w in g))
        total_weight = sum(w for _, _, w in successful_values)
        group_weight = sum(w for _, _, w in best_group)

        consensus_value = best_group[0][1]
        confidence = group_weight / total_weight
        agreements.append(
            Agreement(
                content=consensus_value,
                count=len(best_group),
                ratio=confidence,
                indices=[i for i, _, _ in best_group],
                type="similar",
            )
        )

    elif strategy == "best":
        # Take first successful result
        consensus_value = successful_values[0][1]
        confidence = 1.0
        agreements.append(
            Agreement(
                content=consensus_value,
                count=1,
                ratio=1.0 / len(successful_values),
                indices=[successful_values[0][0]],
                type="exact",
            )
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Build analysis
    duration_ms = (time.time() - consensus_start) * 1000
    analysis = ConsensusAnalysis(
        total_outputs=len(tasks),
        successful_outputs=len(successful_values),
        failed_outputs=len(tasks) - len(successful_values),
        identical_outputs=identical_count,
        similarity_matrix=similarity_matrix,
        average_similarity=avg_similarity,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        total_agreements=len(agreements),
        total_disagreements=len(disagreements),
        strategy=strategy,
        conflict_resolution=resolve_conflicts,
        duration_ms=duration_ms,
    )

    # Handle structured consensus if schema provided
    field_consensus: FieldConsensus | None = None
    result_type: Literal["text", "structured"] = "text"

    if schema is not None:
        result_type = "structured"
        field_consensus = _compute_field_consensus(
            [v for _, v, _ in successful_values],
            schema,
            threshold,
        )

    event_bus.emit(
        ObservabilityEventType.CONSENSUS_RESOLUTION,
        method=strategy,
        confidence=confidence,
    )
    event_bus.emit(
        ObservabilityEventType.CONSENSUS_END,
        status=status,
        confidence=confidence,
        duration_ms=duration_ms,
    )

    return ConsensusResult(
        consensus=consensus_value,
        confidence=confidence,
        outputs=list(outputs),
        agreements=agreements,
        disagreements=disagreements,
        analysis=analysis,
        type=result_type,
        field_consensus=field_consensus,
        status=status,
    )


def _group_by_similarity(
    values: list[tuple[int, Any, float]],
    threshold: float,
) -> list[list[tuple[int, Any, float]]]:
    """Group values by similarity threshold."""
    if not values:
        return []

    groups: list[list[tuple[int, Any, float]]] = []
    used = set()

    for i, (idx, val, weight) in enumerate(values):
        if i in used:
            continue

        group = [(idx, val, weight)]
        used.add(i)

        for j, (idx2, val2, weight2) in enumerate(values):
            if j in used:
                continue
            if _calculate_similarity(str(val), str(val2)) >= threshold:
                group.append((idx2, val2, weight2))
                used.add(j)

        groups.append(group)

    return groups


def _resolve_conflict(
    values: list[tuple[int, Any, float]],
    resolution: ConflictResolution,
    weights: list[float],
) -> tuple[Any, float]:
    """Resolve conflict between values."""
    if resolution == "vote":
        # Take most common
        counter = Counter(str(v) for _, v, _ in values)
        winner = counter.most_common(1)[0][0]
        count = counter.most_common(1)[0][1]
        for _, v, _ in values:
            if str(v) == winner:
                return v, count / len(values)
        return values[0][1], 1.0 / len(values)

    elif resolution == "merge":
        # For strings, concatenate unique parts (simplified)
        # In production, would use more sophisticated merging
        unique_parts = []
        seen = set()
        for _, v, _ in values:
            s = str(v)
            if s not in seen:
                unique_parts.append(s)
                seen.add(s)
        merged = " | ".join(unique_parts)
        return merged, 0.5

    elif resolution == "best":
        # Take highest weighted
        best_idx = max(range(len(values)), key=lambda i: values[i][2])
        return values[best_idx][1], values[best_idx][2] / sum(w for _, _, w in values)

    else:  # "fail"
        raise ValueError("Consensus conflict with resolve_conflicts='fail'")


def _compute_field_consensus(
    values: list[Any],
    schema: type[BaseModel],
    threshold: float,
) -> FieldConsensus:
    """Compute field-by-field consensus for structured outputs."""
    field_consensus = FieldConsensus()

    # Get field names from schema
    field_names = list(schema.model_fields.keys())

    for field_name in field_names:
        field_values: list[tuple[int, Any]] = []

        for i, val in enumerate(values):
            if isinstance(val, BaseModel):
                field_val = getattr(val, field_name, None)
            elif isinstance(val, dict):
                field_val = val.get(field_name)
            else:
                continue
            field_values.append((i, field_val))

        if not field_values:
            continue

        # Count occurrences
        counter = Counter(str(fv) for _, fv in field_values)
        most_common, count = counter.most_common(1)[0]
        agreement = count / len(field_values)

        # Find the actual value and indices
        winning_value = None
        indices = []
        for i, fv in field_values:
            if str(fv) == most_common:
                if winning_value is None:
                    winning_value = fv
                indices.append(i)

        field_consensus.fields[field_name] = FieldConsensusInfo(
            value=winning_value,
            agreement=agreement,
            count=count,
            indices=indices,
        )

    return field_consensus
