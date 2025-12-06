"""Multi-model consensus utilities for L0.

Multi-generation consensus for high-confidence results. Run multiple generations,
compare outputs, and resolve disagreements.

Example:
    ```python
    from l0 import Consensus

    # Run consensus with multiple tasks
    result = await Consensus.run(tasks, strategy="majority")

    # Use presets
    result = await Consensus.strict(tasks)   # All must agree
    result = await Consensus.standard(tasks) # Majority rules
    result = await Consensus.lenient(tasks)  # Flexible
    result = await Consensus.best(tasks)     # Pick best

    # Quick check
    if Consensus.quick(outputs, threshold=0.8):
        print("Consensus reached!")

    # Get most common value
    value = Consensus.get_value(outputs)

    # Validate result
    if Consensus.validate(result, min_confidence=0.8):
        print("Valid consensus")
    ```
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
class FieldAgreement:
    """Per-field consensus information (matches TS FieldAgreement)."""

    path: str  # Field path
    value: Any  # Consensus value for this field
    agreement: float  # 0-1 agreement ratio
    votes: dict[str, int]  # Vote counts per value
    values: list[Any]  # All values seen
    unanimous: bool  # Whether field had unanimous agreement
    confidence: float  # 0-1 confidence in this field's consensus


# Alias for backwards compatibility
FieldConsensusInfo = FieldAgreement


@dataclass
class FieldConsensus:
    """Field-by-field consensus for structured outputs."""

    fields: dict[str, FieldAgreement] = field(default_factory=dict)
    overall_agreement: float = 0.0  # 0-1 overall field agreement ratio
    agreed_fields: list[str] = field(default_factory=list)  # Fields with full agreement
    disagreed_fields: list[str] = field(
        default_factory=list
    )  # Fields with disagreement


@dataclass
class ConsensusOutput:
    """Individual output from a stream."""

    index: int  # Output index
    text: str  # Raw text output
    value: Any  # Parsed value (for compatibility)
    success: bool  # Status of this output
    data: Any = None  # Parsed data (if structured)
    l0_result: Any = None  # L0 result (if text-based)
    structured_result: Any = None  # Structured result (if schema)
    error: str | None = None  # Error if output failed
    duration_ms: float = 0.0  # Execution duration (ms)
    weight: float = 1.0  # Weight assigned to this output
    similarities: list[float] | None = None  # Similarity scores with other outputs


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
# Presets (kept for type reference)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConsensusPreset:
    """Preset configuration for consensus."""

    strategy: Strategy
    threshold: float
    resolve_conflicts: ConflictResolution
    minimum_agreement: float


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_for_comparison(value: Any) -> Any:
    """Normalize a value for consistent comparison.

    Returns a normalized version of the value with consistent ordering,
    preserving the actual structure (not converting to strings prematurely).
    """
    if isinstance(value, dict):
        # Recursively normalize nested dicts with sorted keys
        return {k: _normalize_for_comparison(v) for k, v in sorted(value.items())}
    elif isinstance(value, BaseModel):
        # Convert to dict and recursively normalize
        return _normalize_for_comparison(value.model_dump())
    elif isinstance(value, list):
        return [_normalize_for_comparison(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(_normalize_for_comparison(v) for v in value)
    else:
        return value


def _stable_repr(value: Any) -> str:
    """Get a stable string representation for equality comparison.

    Handles dicts with consistent key ordering and other types that may
    have inconsistent string representations.
    """
    # First normalize the structure, then convert to repr
    # Using repr() preserves type information (e.g., strings have quotes)
    return repr(_normalize_for_comparison(value))


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
            if (
                _calculate_similarity(_stable_repr(val), _stable_repr(val2))
                >= threshold
            ):
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
        counter = Counter(_stable_repr(v) for _, v, _ in values)
        winner = counter.most_common(1)[0][0]
        count = counter.most_common(1)[0][1]
        for _, v, _ in values:
            if _stable_repr(v) == winner:
                return v, count / len(values)
        return values[0][1], 1.0 / len(values)

    elif resolution == "merge":
        # Merge values based on their type
        first_value = values[0][1]

        # For dicts, merge keys (later values override earlier)
        if isinstance(first_value, dict):
            merged: dict[str, Any] = {}
            for _, v, _ in values:
                if isinstance(v, dict):
                    merged.update(v)
            return merged, 0.5

        # For lists, concatenate unique items
        if isinstance(first_value, list):
            merged_list: list[Any] = []
            seen_reprs: set[str] = set()
            for _, v, _ in values:
                if isinstance(v, list):
                    for item in v:
                        repr_item = _stable_repr(item)
                        if repr_item not in seen_reprs:
                            merged_list.append(item)
                            seen_reprs.add(repr_item)
            return merged_list, 0.5

        # For strings, concatenate unique parts
        unique_parts = []
        seen: set[str] = set()
        for _, v, _ in values:
            s = str(v)
            if s not in seen:
                unique_parts.append(s)
                seen.add(s)
        merged_str = " | ".join(unique_parts)
        return merged_str, 0.5

    elif resolution == "best":
        # Take highest weighted
        best_idx = max(range(len(values)), key=lambda i: values[i][2])
        total_weight = sum(w for _, _, w in values)
        if total_weight == 0:
            return values[best_idx][1], 0.0
        return values[best_idx][1], values[best_idx][2] / total_weight

    else:  # "fail"
        raise ValueError("Consensus conflict with resolve_conflicts='fail'")


def _compute_field_consensus(
    values: list[Any],
    schema: type[BaseModel],
    threshold: float,
) -> FieldConsensus:
    """Compute field-by-field consensus for structured outputs."""
    fields: dict[str, FieldAgreement] = {}
    agreed_fields: list[str] = []
    disagreed_fields: list[str] = []

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

        # Count occurrences (votes)
        votes: dict[str, int] = {}
        all_values: list[Any] = []
        for _, fv in field_values:
            key = _stable_repr(fv)
            votes[key] = votes.get(key, 0) + 1
            all_values.append(fv)

        # Find winning value
        most_common = max(votes.items(), key=lambda x: x[1])
        most_common_repr, count = most_common
        agreement = count / len(field_values)
        unanimous = count == len(field_values)

        # Find the actual value
        winning_value = None
        for _, fv in field_values:
            if _stable_repr(fv) == most_common_repr:
                winning_value = fv
                break

        fields[field_name] = FieldAgreement(
            path=field_name,
            value=winning_value,
            agreement=agreement,
            votes=votes,
            values=all_values,
            unanimous=unanimous,
            confidence=agreement,
        )

        if unanimous:
            agreed_fields.append(field_name)
        else:
            disagreed_fields.append(field_name)

    # Calculate overall agreement
    overall_agreement = (
        sum(f.agreement for f in fields.values()) / len(fields) if fields else 0.0
    )

    return FieldConsensus(
        fields=fields,
        overall_agreement=overall_agreement,
        agreed_fields=agreed_fields,
        disagreed_fields=disagreed_fields,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Utilities (matches TS consensusUtils.ts)
# ─────────────────────────────────────────────────────────────────────────────


def calculate_similarity_matrix(outputs: list[ConsensusOutput]) -> list[list[float]]:
    """Calculate pairwise similarity matrix for all outputs.

    Args:
        outputs: Array of consensus outputs

    Returns:
        Similarity matrix (NxN)
    """
    n = len(outputs)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0  # Self-similarity

        for j in range(i + 1, n):
            similarity = calculate_output_similarity(outputs[i], outputs[j])
            matrix[i][j] = similarity
            matrix[j][i] = similarity

    return matrix


def calculate_output_similarity(a: ConsensusOutput, b: ConsensusOutput) -> float:
    """Calculate similarity between two outputs.

    Args:
        a: First output
        b: Second output

    Returns:
        Similarity score (0-1)
    """
    # If both have structured data, compare structurally
    if a.data is not None and b.data is not None:
        return calculate_structural_similarity(a.data, b.data)

    # Otherwise, compare text
    return _calculate_similarity(a.text, b.text)


def calculate_structural_similarity(a: Any, b: Any) -> float:
    """Calculate structural similarity between two objects.

    Optimized with early termination for identical values.

    Args:
        a: First object
        b: Second object

    Returns:
        Similarity score (0-1)
    """
    # Fast path: reference equality
    if a is b:
        return 1.0

    # Fast path: null/None
    if a is None:
        return 1.0 if b is None else 0.0
    if b is None:
        return 0.0

    type_a = type(a)
    type_b = type(b)

    # Type mismatch - early termination
    if type_a != type_b:
        # Allow BaseModel vs dict comparison
        if isinstance(a, BaseModel) and isinstance(b, dict):
            a = a.model_dump() if hasattr(a, "model_dump") else dict(a)
        elif isinstance(b, BaseModel) and isinstance(a, dict):
            b = b.model_dump() if hasattr(b, "model_dump") else dict(b)
        elif not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return 0.0

    # Primitives
    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return 1.0
        return _calculate_similarity(a, b)

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return 1.0
        max_diff = max(abs(a), abs(b))
        if max_diff == 0:
            return 1.0
        return float(1 - abs(a - b) / max_diff)

    if isinstance(a, bool) and isinstance(b, bool):
        return 1.0 if a == b else 0.0

    # Array comparison
    if isinstance(a, list):
        if not isinstance(b, list):
            return 0.0

        len_a = len(a)
        len_b = len(b)
        max_length = max(len_a, len_b)

        if max_length == 0:
            return 1.0

        # Fast path: check if arrays are equal
        if len_a == len_b and a == b:
            return 1.0

        matches = 0.0
        min_length = min(len_a, len_b)
        for i in range(min_length):
            matches += calculate_structural_similarity(a[i], b[i])

        return matches / max_length

    # Object/dict comparison
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return 0.0

        keys_a = set(a.keys())
        keys_b = set(b.keys())

        # Fast path: check if dicts are equal
        if keys_a == keys_b and a == b:
            return 1.0

        all_keys = keys_a | keys_b
        total = len(all_keys)

        if total == 0:
            return 1.0

        matches = 0.0
        for key in all_keys:
            if key in a and key in b:
                matches += calculate_structural_similarity(a[key], b[key])

        return matches / total

    # BaseModel comparison
    if isinstance(a, BaseModel) and isinstance(b, BaseModel):
        a_dict = a.model_dump()
        b_dict = b.model_dump()
        return calculate_structural_similarity(a_dict, b_dict)

    # Fallback: string comparison
    return 1.0 if _stable_repr(a) == _stable_repr(b) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Agreement/Disagreement Utilities (matches TS consensusUtils.ts)
# ─────────────────────────────────────────────────────────────────────────────


def find_agreements(
    outputs: list[ConsensusOutput],
    threshold: float = 0.8,
) -> list[Agreement]:
    """Find agreements across outputs.

    Args:
        outputs: Array of consensus outputs
        threshold: Similarity threshold for agreement

    Returns:
        Array of agreements
    """
    agreements: list[Agreement] = []

    # For text-based consensus
    if outputs and outputs[0].data is None:
        agreements.extend(_find_text_agreements(outputs, threshold))
    else:
        # For structured consensus
        agreements.extend(_find_structured_agreements(outputs, threshold))

    return agreements


def _find_text_agreements(
    outputs: list[ConsensusOutput],
    threshold: float,
) -> list[Agreement]:
    """Find text-based agreements."""
    agreements: list[Agreement] = []

    # Group similar outputs
    groups: list[list[int]] = []
    used: set[int] = set()

    for i in range(len(outputs)):
        if i in used:
            continue

        group = [i]
        used.add(i)

        for j in range(i + 1, len(outputs)):
            if j in used:
                continue

            similarity = calculate_output_similarity(outputs[i], outputs[j])
            if similarity >= threshold:
                group.append(j)
                used.add(j)

        if len(group) > 1:
            groups.append(group)

    # Create agreements from groups
    for group in groups:
        content = outputs[group[0]].text
        agreement_type: AgreementType = (
            "exact" if len(group) == len(outputs) else "similar"
        )

        agreements.append(
            Agreement(
                content=content,
                count=len(group),
                ratio=len(group) / len(outputs),
                indices=group,
                type=agreement_type,
            )
        )

    return agreements


def _find_structured_agreements(
    outputs: list[ConsensusOutput],
    threshold: float,
) -> list[Agreement]:
    """Find structured agreements (field-by-field)."""
    agreements: list[Agreement] = []

    # Get all field paths
    all_paths: set[str] = set()
    for output in outputs:
        if output.data:
            all_paths.update(_get_all_paths(output.data))

    # Check agreement for each field
    for path in all_paths:
        values = [
            _get_value_at_path(o.data, path) for o in outputs if o.data is not None
        ]
        values = [v for v in values if v is not None]

        if not values:
            continue

        # Count identical values
        value_counts: dict[str, list[int]] = {}
        for i, v in enumerate(values):
            key = _stable_repr(v)
            if key not in value_counts:
                value_counts[key] = []
            value_counts[key].append(i)

        # Find majority
        max_count = 0
        majority_value: Any = None
        majority_indices: list[int] = []

        for key, indices in value_counts.items():
            if len(indices) > max_count:
                max_count = len(indices)
                majority_value = values[indices[0]] if indices else None
                majority_indices = indices

        ratio = max_count / len(outputs)
        if ratio >= threshold:
            agreements.append(
                Agreement(
                    content=majority_value,
                    path=path,
                    count=max_count,
                    ratio=ratio,
                    indices=majority_indices,
                    type="exact" if ratio == 1.0 else "structural",
                )
            )

    return agreements


def find_disagreements(
    outputs: list[ConsensusOutput],
    threshold: float = 0.8,
) -> list[Disagreement]:
    """Find disagreements across outputs.

    Args:
        outputs: Array of consensus outputs
        threshold: Disagreement threshold

    Returns:
        Array of disagreements
    """
    disagreements: list[Disagreement] = []

    # For structured outputs
    if outputs and outputs[0].data is not None:
        disagreements.extend(_find_structured_disagreements(outputs, threshold))
    else:
        # For text outputs
        disagreements.extend(_find_text_disagreements(outputs, threshold))

    return disagreements


def _find_text_disagreements(
    outputs: list[ConsensusOutput],
    threshold: float,
) -> list[Disagreement]:
    """Find text-based disagreements."""
    disagreements: list[Disagreement] = []

    # Group outputs by similarity
    value_counts: dict[str, list[int]] = {}

    for i, output in enumerate(outputs):
        text = output.text.strip()
        grouped = False

        # Try to group with existing
        for key in list(value_counts.keys()):
            similarity = _calculate_similarity(text, key)
            if similarity >= threshold:
                value_counts[key].append(i)
                grouped = True
                break

        if not grouped:
            value_counts[text] = [i]

    # If more than one group, it's a disagreement
    if len(value_counts) > 1:
        values = [
            DisagreementValue(
                value=value,
                count=len(indices),
                indices=indices,
            )
            for value, indices in value_counts.items()
        ]

        max_count = max(v.count for v in values)
        ratio = max_count / len(outputs)
        severity = _determine_severity(ratio)

        disagreements.append(
            Disagreement(
                values=values,
                severity=severity,
            )
        )

    return disagreements


def _find_structured_disagreements(
    outputs: list[ConsensusOutput],
    threshold: float,
) -> list[Disagreement]:
    """Find structured disagreements (field-by-field)."""
    disagreements: list[Disagreement] = []

    # Get all field paths
    all_paths: set[str] = set()
    for output in outputs:
        if output.data:
            all_paths.update(_get_all_paths(output.data))

    # Check each field for disagreement
    for path in all_paths:
        values_with_indices: list[tuple[Any, int]] = []
        for i, output in enumerate(outputs):
            if output.data is not None:
                value = _get_value_at_path(output.data, path)
                if value is not None:
                    values_with_indices.append((value, i))

        # Group by value
        value_counts: dict[str, list[int]] = {}
        for value, index in values_with_indices:
            key = _stable_repr(value)
            if key not in value_counts:
                value_counts[key] = []
            value_counts[key].append(index)

        # If more than one value, check if it's a significant disagreement
        if len(value_counts) > 1:
            distinct_values = []
            for key, indices in value_counts.items():
                # Find actual value
                actual_value = None
                for value, idx in values_with_indices:
                    if _stable_repr(value) == key:
                        actual_value = value
                        break
                distinct_values.append(
                    DisagreementValue(
                        value=actual_value,
                        count=len(indices),
                        indices=indices,
                    )
                )

            # Find the majority agreement ratio
            max_count = max(v.count for v in distinct_values)
            majority_ratio = max_count / len(outputs)

            # Skip if majority agrees above threshold
            if majority_ratio >= threshold:
                continue

            severity = _determine_severity(majority_ratio)

            disagreements.append(
                Disagreement(
                    path=path,
                    values=distinct_values,
                    severity=severity,
                )
            )

    return disagreements


def calculate_field_consensus(outputs: list[ConsensusOutput]) -> FieldConsensus:
    """Calculate field-level consensus for structured outputs.

    Args:
        outputs: Array of consensus outputs

    Returns:
        Field consensus information
    """
    fields: dict[str, FieldAgreement] = {}

    # Get all field paths
    all_paths: set[str] = set()
    for output in outputs:
        if output.data:
            all_paths.update(_get_all_paths(output.data))

    # Calculate consensus for each field
    for path in all_paths:
        values_with_indices: list[tuple[Any, int]] = []
        for i, output in enumerate(outputs):
            if output.data is not None:
                value = _get_value_at_path(output.data, path)
                if value is not None:
                    values_with_indices.append((value, i))

        if not values_with_indices:
            continue

        # Count votes
        votes: dict[str, int] = {}
        all_values: list[Any] = []

        for value, _ in values_with_indices:
            key = _stable_repr(value)
            votes[key] = votes.get(key, 0) + 1
            all_values.append(value)

        # Find consensus value
        max_votes = 0
        consensus_value: Any = None
        for key, count in votes.items():
            if count > max_votes:
                max_votes = count
                # Find actual value
                for value, _ in values_with_indices:
                    if _stable_repr(value) == key:
                        consensus_value = value
                        break

        agreement = max_votes / len(outputs)
        unanimous = max_votes == len(outputs)
        confidence = agreement

        fields[path] = FieldAgreement(
            path=path,
            value=consensus_value,
            agreement=agreement,
            votes=votes,
            values=all_values,
            unanimous=unanimous,
            confidence=confidence,
        )

    # Calculate overall metrics
    agreed_fields = [k for k, v in fields.items() if v.unanimous]
    disagreed_fields = [k for k, v in fields.items() if not v.unanimous]
    overall_agreement = (
        sum(f.agreement for f in fields.values()) / len(fields) if fields else 0.0
    )

    return FieldConsensus(
        fields=fields,
        overall_agreement=overall_agreement,
        agreed_fields=agreed_fields,
        disagreed_fields=disagreed_fields,
    )


def _get_all_paths(obj: Any, prefix: str = "") -> list[str]:
    """Get all paths in an object (dot notation)."""
    paths: list[str] = []

    if isinstance(obj, BaseModel):
        obj = obj.model_dump() if hasattr(obj, "model_dump") else {}

    if isinstance(obj, dict):
        for key in obj.keys():
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)

            value = obj[key]
            if isinstance(value, dict) or isinstance(value, BaseModel):
                paths.extend(_get_all_paths(value, path))

    return paths


def _get_value_at_path(obj: Any, path: str) -> Any:
    """Get value at path in object."""
    if obj is None:
        return None

    if isinstance(obj, BaseModel):
        obj = obj.model_dump() if hasattr(obj, "model_dump") else {}

    parts = path.split(".")
    current = obj

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, BaseModel) and hasattr(current, part):
            current = getattr(current, part)
        else:
            return None

    return current


# ─────────────────────────────────────────────────────────────────────────────
# Resolution Utilities (matches TS consensusUtils.ts)
# ─────────────────────────────────────────────────────────────────────────────


def resolve_majority(
    outputs: list[ConsensusOutput],
    weights: list[float] | None = None,
) -> ConsensusOutput:
    """Resolve consensus using majority vote.

    Args:
        outputs: Array of consensus outputs
        weights: Optional weights for each output

    Returns:
        Consensus output
    """
    if not outputs:
        raise ValueError("No outputs to resolve")

    # Use weights if provided
    output_weights = weights or [o.weight for o in outputs]

    # For structured outputs, do field-by-field voting
    if outputs[0].data is not None:
        field_consensus = calculate_field_consensus(outputs)
        consensus_data: dict[str, Any] = {}

        for path, field_info in field_consensus.fields.items():
            _set_value_at_path(consensus_data, path, field_info.value)

        return ConsensusOutput(
            index=outputs[0].index,
            text=_stable_repr(consensus_data),
            value=consensus_data,
            success=True,
            data=consensus_data,
            weight=outputs[0].weight,
        )

    # For text outputs, find most similar to all
    best_index = 0
    best_score = -1.0

    for i in range(len(outputs)):
        score = 0.0
        for j in range(len(outputs)):
            if i != j:
                similarity = calculate_output_similarity(outputs[i], outputs[j])
                score += similarity * output_weights[j]

        if score > best_score:
            best_score = score
            best_index = i

    return outputs[best_index]


def resolve_best(
    outputs: list[ConsensusOutput],
    weights: list[float] | None = None,
) -> ConsensusOutput:
    """Resolve consensus by choosing best output.

    Args:
        outputs: Array of consensus outputs
        weights: Optional weights for each output

    Returns:
        Best output
    """
    if not outputs:
        raise ValueError("No outputs to resolve")

    output_weights = weights or [o.weight for o in outputs]

    # Find output with highest weight
    best_index = 0
    best_weight = output_weights[0]

    for i in range(1, len(outputs)):
        if output_weights[i] > best_weight:
            best_weight = output_weights[i]
            best_index = i

    return outputs[best_index]


def resolve_merge(outputs: list[ConsensusOutput]) -> ConsensusOutput:
    """Resolve consensus by merging all outputs.

    Args:
        outputs: Array of consensus outputs

    Returns:
        Merged output
    """
    if not outputs:
        raise ValueError("No outputs to resolve")

    if len(outputs) == 1:
        return outputs[0]

    # For structured outputs, merge field by field
    if outputs[0].data is not None:
        merged: dict[str, Any] = {}
        all_paths: set[str] = set()

        for output in outputs:
            if output.data:
                all_paths.update(_get_all_paths(output.data))

        for path in all_paths:
            values = [
                _get_value_at_path(o.data, path) for o in outputs if o.data is not None
            ]
            values = [v for v in values if v is not None]

            # Take first non-None value
            if values:
                _set_value_at_path(merged, path, values[0])

        return ConsensusOutput(
            index=outputs[0].index,
            text=_stable_repr(merged),
            value=merged,
            success=True,
            data=merged,
            weight=outputs[0].weight,
        )

    # For text outputs, concatenate unique parts
    unique_texts = list(dict.fromkeys(o.text.strip() for o in outputs))
    merged_text = "\n\n".join(unique_texts)

    return ConsensusOutput(
        index=outputs[0].index,
        text=merged_text,
        value=merged_text,
        success=True,
        weight=outputs[0].weight,
    )


def _set_value_at_path(obj: dict[str, Any], path: str, value: Any) -> None:
    """Set value at path in object."""
    parts = path.split(".")
    current = obj

    for i in range(len(parts) - 1):
        part = parts[i]
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def meets_minimum_agreement(
    agreements: list[Agreement],
    outputs_count: int,
    threshold: float,
) -> bool:
    """Check if consensus meets minimum agreement threshold.

    Args:
        agreements: Array of agreements
        outputs_count: Total outputs
        threshold: Minimum agreement ratio

    Returns:
        Whether consensus is sufficient
    """
    # If threshold is 0, any level of agreement is acceptable
    if threshold == 0:
        return True

    # Single output is trivially unanimous
    if outputs_count == 1:
        return True

    if not agreements:
        return False

    # Find highest agreement ratio
    max_ratio = max(a.count / outputs_count for a in agreements)
    return max_ratio >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Consensus Class - Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class Consensus:
    """Scoped API for multi-model consensus operations.

    Usage:
        from l0 import Consensus

        # Run consensus
        result = await Consensus.run(tasks, strategy="majority")

        # Presets (as methods)
        result = await Consensus.strict(tasks)   # All must agree
        result = await Consensus.standard(tasks) # Majority rules (default)
        result = await Consensus.lenient(tasks)  # Flexible
        result = await Consensus.best(tasks)     # Pick best single output

        # Presets (as config objects)
        Consensus.STRICT   # ConsensusPreset for unanimous agreement
        Consensus.STANDARD # ConsensusPreset for majority rules
        Consensus.LENIENT  # ConsensusPreset for flexible matching
        Consensus.BEST     # ConsensusPreset for best output

        # Quick check
        if Consensus.quick(outputs, threshold=0.8):
            print("Consensus reached!")

        # Get most common value
        value = Consensus.get_value(outputs)

        # Validate result
        if Consensus.validate(result, min_confidence=0.8):
            print("Valid consensus")
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Preset Configurations (class attributes)
    # ─────────────────────────────────────────────────────────────────────────

    # Strict consensus - all must agree
    STRICT: ConsensusPreset = ConsensusPreset(
        strategy="unanimous",
        threshold=1.0,
        resolve_conflicts="fail",
        minimum_agreement=1.0,
    )

    # Standard consensus - majority rules (default)
    STANDARD: ConsensusPreset = ConsensusPreset(
        strategy="majority",
        threshold=0.8,
        resolve_conflicts="vote",
        minimum_agreement=0.6,
    )

    # Lenient consensus - flexible agreement
    LENIENT: ConsensusPreset = ConsensusPreset(
        strategy="majority",
        threshold=0.7,
        resolve_conflicts="merge",
        minimum_agreement=0.5,
    )

    # Best-of consensus - choose highest quality
    BEST: ConsensusPreset = ConsensusPreset(
        strategy="best",
        threshold=0.8,
        resolve_conflicts="best",
        minimum_agreement=0.5,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Run Method
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def run(
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
        successful_values: list[tuple[int, Any, float]] = []

        async def run_task(
            idx: int, task: Callable[[], Awaitable[T]]
        ) -> ConsensusOutput:
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
                # Determine text representation and data
                text = str(result) if result is not None else ""
                data = None
                if isinstance(result, BaseModel):
                    data = result.model_dump() if hasattr(result, "model_dump") else {}
                    text = _stable_repr(data)
                elif isinstance(result, dict):
                    data = result
                    text = _stable_repr(data)

                event_bus.emit(
                    ObservabilityEventType.CONSENSUS_OUTPUT_COLLECTED,
                    stream_index=idx,
                    length=len(text),
                    has_errors=False,
                )
                return ConsensusOutput(
                    index=idx,
                    text=text,
                    value=result,
                    success=True,
                    data=data,
                    duration_ms=duration,
                    weight=weights[idx],
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
                    index=idx,
                    text="",
                    value=None,
                    success=False,
                    error=str(e),
                    duration_ms=duration,
                    weight=weights[idx],
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
        string_outputs = [_stable_repr(v) for _, v, _ in successful_values]

        # Build similarity matrix
        similarity_matrix = _build_similarity_matrix(string_outputs)
        flat_similarities = [
            similarity_matrix[i][j]
            for i in range(len(string_outputs))
            for j in range(i + 1, len(string_outputs))
        ]
        avg_similarity = (
            sum(flat_similarities) / len(flat_similarities)
            if flat_similarities
            else 1.0
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
                similarity_matrix[0][j] >= threshold
                for j in range(1, len(string_outputs))
            )
            if all_similar and len(successful_values) == len(tasks):
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
            ratio = group_weight / total_weight if total_weight > 0 else 0.0

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
                        if len(set(_stable_repr(v) for _, v, _ in largest_group)) == 1
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
            confidence = group_weight / total_weight if total_weight > 0 else 0.0
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

    # ─────────────────────────────────────────────────────────────────────────
    # Presets
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def strict(
        tasks: list[Callable[[], Awaitable[T]]],
        *,
        schema: type[BaseModel] | None = None,
        on_event: Callable[[ObservabilityEvent], None] | None = None,
    ) -> ConsensusResult[T]:
        """Run consensus with strict preset - all must agree.

        Args:
            tasks: List of async callables (min 2)
            schema: Optional Pydantic schema for structured consensus
            on_event: Optional callback for observability events

        Returns:
            ConsensusResult

        Raises:
            ValueError: If outputs don't unanimously agree
        """
        return await Consensus.run(
            tasks,
            strategy="unanimous",
            threshold=1.0,
            resolve_conflicts="fail",
            minimum_agreement=1.0,
            schema=schema,
            on_event=on_event,
        )

    @staticmethod
    async def standard(
        tasks: list[Callable[[], Awaitable[T]]],
        *,
        schema: type[BaseModel] | None = None,
        on_event: Callable[[ObservabilityEvent], None] | None = None,
    ) -> ConsensusResult[T]:
        """Run consensus with standard preset - majority rules.

        Args:
            tasks: List of async callables (min 2)
            schema: Optional Pydantic schema for structured consensus
            on_event: Optional callback for observability events

        Returns:
            ConsensusResult
        """
        return await Consensus.run(
            tasks,
            strategy="majority",
            threshold=0.8,
            resolve_conflicts="vote",
            minimum_agreement=0.6,
            schema=schema,
            on_event=on_event,
        )

    @staticmethod
    async def lenient(
        tasks: list[Callable[[], Awaitable[T]]],
        *,
        schema: type[BaseModel] | None = None,
        on_event: Callable[[ObservabilityEvent], None] | None = None,
    ) -> ConsensusResult[T]:
        """Run consensus with lenient preset - flexible matching.

        Args:
            tasks: List of async callables (min 2)
            schema: Optional Pydantic schema for structured consensus
            on_event: Optional callback for observability events

        Returns:
            ConsensusResult
        """
        return await Consensus.run(
            tasks,
            strategy="majority",
            threshold=0.7,
            resolve_conflicts="merge",
            minimum_agreement=0.5,
            schema=schema,
            on_event=on_event,
        )

    @staticmethod
    async def best(
        tasks: list[Callable[[], Awaitable[T]]],
        *,
        schema: type[BaseModel] | None = None,
        on_event: Callable[[ObservabilityEvent], None] | None = None,
    ) -> ConsensusResult[T]:
        """Run consensus with best preset - pick highest quality output.

        Args:
            tasks: List of async callables (min 2)
            schema: Optional Pydantic schema for structured consensus
            on_event: Optional callback for observability events

        Returns:
            ConsensusResult
        """
        return await Consensus.run(
            tasks,
            strategy="best",
            threshold=0.5,
            resolve_conflicts="best",
            minimum_agreement=0.0,
            schema=schema,
            on_event=on_event,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def quick(outputs: list[Any], threshold: float = 0.8) -> bool:
        """Quick check if outputs have consensus at given threshold.

        Args:
            outputs: List of outputs to check
            threshold: Minimum agreement ratio (default 0.8 = 80%)

        Returns:
            True if agreement ratio >= threshold
        """
        if not outputs:
            return False

        counter = Counter(_stable_repr(o) for o in outputs)
        most_common_count = counter.most_common(1)[0][1]
        ratio = most_common_count / len(outputs)
        return ratio >= threshold

    @staticmethod
    def get_value(outputs: list[T]) -> T | None:
        """Get the most common value from outputs.

        Args:
            outputs: List of outputs

        Returns:
            Most common value, or None if empty
        """
        if not outputs:
            return None

        counter = Counter(_stable_repr(o) for o in outputs)
        winner = counter.most_common(1)[0][0]

        # Return the actual object, not the string
        for o in outputs:
            if _stable_repr(o) == winner:
                return o
        return outputs[0]

    @staticmethod
    def validate(
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

    # ─────────────────────────────────────────────────────────────────────────
    # Similarity Utilities (scoped)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def similarity_matrix(outputs: list[ConsensusOutput]) -> list[list[float]]:
        """Calculate pairwise similarity matrix for all outputs.

        Args:
            outputs: Array of consensus outputs

        Returns:
            Similarity matrix (NxN)
        """
        return calculate_similarity_matrix(outputs)

    @staticmethod
    def output_similarity(a: ConsensusOutput, b: ConsensusOutput) -> float:
        """Calculate similarity between two outputs.

        Args:
            a: First output
            b: Second output

        Returns:
            Similarity score (0-1)
        """
        return calculate_output_similarity(a, b)

    @staticmethod
    def structural_similarity(a: Any, b: Any) -> float:
        """Calculate structural similarity between two objects.

        Args:
            a: First object
            b: Second object

        Returns:
            Similarity score (0-1)
        """
        return calculate_structural_similarity(a, b)

    # ─────────────────────────────────────────────────────────────────────────
    # Agreement/Disagreement Utilities (scoped)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def agreements(
        outputs: list[ConsensusOutput],
        threshold: float = 0.8,
    ) -> list[Agreement]:
        """Find agreements across outputs.

        Args:
            outputs: Array of consensus outputs
            threshold: Similarity threshold for agreement

        Returns:
            Array of agreements
        """
        return find_agreements(outputs, threshold)

    @staticmethod
    def disagreements(
        outputs: list[ConsensusOutput],
        threshold: float = 0.8,
    ) -> list[Disagreement]:
        """Find disagreements across outputs.

        Args:
            outputs: Array of consensus outputs
            threshold: Disagreement threshold

        Returns:
            Array of disagreements
        """
        return find_disagreements(outputs, threshold)

    @staticmethod
    def field_consensus(outputs: list[ConsensusOutput]) -> FieldConsensus:
        """Calculate field-level consensus for structured outputs.

        Args:
            outputs: Array of consensus outputs

        Returns:
            Field consensus information
        """
        return calculate_field_consensus(outputs)

    # ─────────────────────────────────────────────────────────────────────────
    # Resolution Utilities (scoped)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def majority(
        outputs: list[ConsensusOutput],
        weights: list[float] | None = None,
    ) -> ConsensusOutput:
        """Resolve consensus using majority vote.

        Args:
            outputs: Array of consensus outputs
            weights: Optional weights for each output

        Returns:
            Consensus output
        """
        return resolve_majority(outputs, weights)

    @staticmethod
    def best_output(
        outputs: list[ConsensusOutput],
        weights: list[float] | None = None,
    ) -> ConsensusOutput:
        """Resolve consensus by choosing best output.

        Args:
            outputs: Array of consensus outputs
            weights: Optional weights for each output

        Returns:
            Best output
        """
        return resolve_best(outputs, weights)

    @staticmethod
    def merge(outputs: list[ConsensusOutput]) -> ConsensusOutput:
        """Resolve consensus by merging all outputs.

        Args:
            outputs: Array of consensus outputs

        Returns:
            Merged output
        """
        return resolve_merge(outputs)

    @staticmethod
    def meets_agreement(
        agreements: list[Agreement],
        outputs_count: int,
        threshold: float,
    ) -> bool:
        """Check if consensus meets minimum agreement threshold.

        Args:
            agreements: Array of agreements
            outputs_count: Total outputs
            threshold: Minimum agreement ratio

        Returns:
            Whether consensus is sufficient
        """
        return meets_minimum_agreement(agreements, outputs_count, threshold)


# Convenience alias - consensus() triggers model calls, so shorthand is useful
consensus = Consensus.run


# ─────────────────────────────────────────────────────────────────────────────
# Module-Level Presets (matches TS consensus presets)
# ─────────────────────────────────────────────────────────────────────────────

# Strict consensus - all must agree
strict_consensus: ConsensusPreset = ConsensusPreset(
    strategy="unanimous",
    threshold=1.0,
    resolve_conflicts="fail",
    minimum_agreement=1.0,
)

# Standard consensus - majority rules (default)
standard_consensus: ConsensusPreset = ConsensusPreset(
    strategy="majority",
    threshold=0.8,
    resolve_conflicts="vote",
    minimum_agreement=0.6,
)

# Lenient consensus - flexible agreement
lenient_consensus: ConsensusPreset = ConsensusPreset(
    strategy="majority",
    threshold=0.7,
    resolve_conflicts="merge",
    minimum_agreement=0.5,
)

# Best-of consensus - choose highest quality
best_consensus: ConsensusPreset = ConsensusPreset(
    strategy="best",
    threshold=0.8,
    resolve_conflicts="best",
    minimum_agreement=0.5,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-Level Utility Functions (matches TS helper functions)
# ─────────────────────────────────────────────────────────────────────────────


def quick_consensus(outputs: list[Any], threshold: float = 0.8) -> bool:
    """Quick consensus check - returns true if outputs agree.

    Args:
        outputs: Array of text outputs
        threshold: Similarity threshold (default 0.8)

    Returns:
        Whether outputs have consensus

    Example:
        ```python
        outputs = ['answer A', 'answer A', 'answer B']
        quick_consensus(outputs)  # False (not 80% agreement)
        quick_consensus(outputs, 0.6)  # True (66% >= 60%)
        ```
    """
    return Consensus.quick(outputs, threshold)


def get_consensus_value(outputs: list[T]) -> T | None:
    """Get consensus value from array of outputs.

    Args:
        outputs: Array of outputs

    Returns:
        Most common output

    Example:
        ```python
        get_consensus_value(['A', 'A', 'B'])  # 'A'
        get_consensus_value([1, 2, 1, 1])  # 1
        ```
    """
    return Consensus.get_value(outputs)


def validate_consensus(
    result: ConsensusResult[Any],
    min_confidence: float = 0.8,
    max_disagreements: int = 0,
) -> bool:
    """Validate consensus result meets criteria.

    Args:
        result: Consensus result to validate
        min_confidence: Minimum confidence required (default 0.8)
        max_disagreements: Maximum disagreements allowed (default 0)

    Returns:
        True if confidence >= min_confidence and no major/critical disagreements
    """
    return Consensus.validate(result, min_confidence, max_disagreements)
