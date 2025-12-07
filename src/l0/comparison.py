"""Comparison utilities for L0.

Provides string similarity algorithms, object comparison, and deep equality
utilities for evaluation and testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal


class DifferenceSeverity(str, Enum):
    """Severity level for a difference."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DifferenceType(str, Enum):
    """Type of difference found."""

    MISSING = "missing"
    EXTRA = "extra"
    DIFFERENT = "different"
    TYPE_MISMATCH = "type-mismatch"


@dataclass
class Difference:
    """A difference found during comparison."""

    path: str
    """JSON path to the differing value."""

    expected: Any
    """Expected value."""

    actual: Any
    """Actual value."""

    type: DifferenceType
    """Type of difference."""

    severity: DifferenceSeverity
    """Severity level."""

    message: str
    """Human-readable description."""

    similarity: float | None = None
    """Similarity score for string comparisons (0-1)."""


@dataclass
class StringComparisonOptions:
    """Options for string comparison."""

    case_sensitive: bool = True
    """Whether comparison is case-sensitive."""

    normalize_whitespace: bool = True
    """Whether to normalize whitespace before comparison."""

    algorithm: Literal["levenshtein", "jaro-winkler", "cosine"] = "levenshtein"
    """Similarity algorithm to use."""


@dataclass
class ObjectComparisonOptions:
    """Options for object comparison."""

    style: Literal["strict", "lenient"] = "strict"
    """Comparison style."""

    ignore_extra_fields: bool = False
    """Whether to ignore extra fields in actual value."""

    ignore_array_order: bool = False
    """Whether to ignore order when comparing arrays."""

    numeric_tolerance: float = 0.001
    """Acceptable difference for numeric comparisons."""

    custom_comparisons: dict[str, Callable[[Any, Any], bool | float]] = field(
        default_factory=dict
    )
    """Custom comparison functions for specific paths."""


# ─────────────────────────────────────────────────────────────────────────────
# String Similarity Algorithms
# ─────────────────────────────────────────────────────────────────────────────


def compare_strings(
    a: str,
    b: str,
    *,
    case_sensitive: bool = True,
    normalize_whitespace: bool = True,
    algorithm: Literal["levenshtein", "jaro-winkler", "cosine"] = "levenshtein",
) -> float:
    """Compare two strings with similarity scoring.

    Args:
        a: First string
        b: Second string
        case_sensitive: Whether comparison is case-sensitive
        normalize_whitespace: Whether to normalize whitespace
        algorithm: Similarity algorithm to use

    Returns:
        Similarity score (0-1)
    """
    str1 = a
    str2 = b

    # Normalize
    if not case_sensitive:
        str1 = str1.lower()
        str2 = str2.lower()

    if normalize_whitespace:
        import re

        str1 = re.sub(r"\s+", " ", str1).strip()
        str2 = re.sub(r"\s+", " ", str2).strip()

    # Exact match
    if str1 == str2:
        return 1.0

    # Choose algorithm
    if algorithm == "levenshtein":
        return levenshtein_similarity(str1, str2)
    elif algorithm == "jaro-winkler":
        return jaro_winkler_similarity(str1, str2)
    elif algorithm == "cosine":
        return cosine_similarity(str1, str2)
    else:
        return levenshtein_similarity(str1, str2)


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate Levenshtein distance (edit distance).

    Args:
        a: First string
        b: Second string

    Returns:
        Number of edits needed to transform a into b
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Create matrix
    matrix: list[list[int]] = []

    # Initialize matrix
    for i in range(len(b) + 1):
        matrix.append([i])

    for j in range(1, len(a) + 1):
        matrix[0].append(j)

    # Fill matrix
    for i in range(1, len(b) + 1):
        for j in range(1, len(a) + 1):
            if b[i - 1] == a[j - 1]:
                matrix[i].append(matrix[i - 1][j - 1])
            else:
                matrix[i].append(
                    min(
                        matrix[i - 1][j - 1] + 1,  # substitution
                        matrix[i][j - 1] + 1,  # insertion
                        matrix[i - 1][j] + 1,  # deletion
                    )
                )

    return matrix[len(b)][len(a)]


def levenshtein_similarity(a: str, b: str) -> float:
    """Levenshtein distance similarity (0-1).

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score (0-1)
    """
    if a == b:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0

    distance = levenshtein_distance(a, b)
    max_length = max(len(a), len(b))

    return 1 - distance / max_length


def _jaro_similarity(a: str, b: str) -> float:
    """Calculate Jaro similarity.

    Args:
        a: First string
        b: Second string

    Returns:
        Jaro similarity score (0-1)
    """
    if a == b:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0

    match_window = max(len(a), len(b)) // 2 - 1
    if match_window < 0:
        match_window = 0

    a_matches = [False] * len(a)
    b_matches = [False] * len(b)

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len(a)):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len(b))

        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Find transpositions
    k = 0
    for i in range(len(a)):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1

    return (
        matches / len(a) + matches / len(b) + (matches - transpositions / 2) / matches
    ) / 3


def _common_prefix_length(a: str, b: str, max_length: int) -> int:
    """Get common prefix length.

    Args:
        a: First string
        b: Second string
        max_length: Maximum prefix length to consider

    Returns:
        Length of common prefix
    """
    length = 0
    limit = min(len(a), len(b), max_length)

    for i in range(limit):
        if a[i] == b[i]:
            length += 1
        else:
            break

    return length


def jaro_winkler_similarity(a: str, b: str) -> float:
    """Jaro-Winkler similarity (0-1).

    Adds a bonus for matching prefixes.

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score (0-1)
    """
    if a == b:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0

    # Jaro similarity
    jaro_sim = _jaro_similarity(a, b)

    # Jaro-Winkler adds bonus for matching prefix
    prefix_length = _common_prefix_length(a, b, 4)
    prefix_scale = 0.1

    return jaro_sim + prefix_length * prefix_scale * (1 - jaro_sim)


def _string_to_vector(text: str) -> dict[str, int]:
    """Convert string to term frequency vector.

    Args:
        text: Text to vectorize

    Returns:
        Dictionary of word frequencies
    """
    words = text.lower().split()
    vector: dict[str, int] = {}

    for word in words:
        vector[word] = vector.get(word, 0) + 1

    return vector


def cosine_similarity(a: str, b: str) -> float:
    """Cosine similarity (0-1).

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score (0-1)
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    vector_a = _string_to_vector(a)
    vector_b = _string_to_vector(b)

    dot_product = sum(vector_a.get(key, 0) * vector_b.get(key, 0) for key in vector_a)

    magnitude_a = sum(val * val for val in vector_a.values()) ** 0.5
    magnitude_b = sum(val * val for val in vector_b.values()) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return float(dot_product / (magnitude_a * magnitude_b))


# ─────────────────────────────────────────────────────────────────────────────
# Numeric Comparison
# ─────────────────────────────────────────────────────────────────────────────


def compare_numbers(a: float, b: float, tolerance: float = 0.001) -> bool:
    """Compare two numbers with tolerance.

    Args:
        a: First number
        b: Second number
        tolerance: Acceptable difference

    Returns:
        Whether numbers are equal within tolerance
    """
    return abs(a - b) <= tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Type Utilities
# ─────────────────────────────────────────────────────────────────────────────


def get_type(value: Any) -> str:
    """Get type of value as string.

    Args:
        value: Value to check

    Returns:
        Type string ("null", "boolean", "number", "string", "array", "object", "unknown")
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Deep Equality
# ─────────────────────────────────────────────────────────────────────────────


def deep_equal(a: Any, b: Any) -> bool:
    """Deep equality check with early termination.

    Returns False as soon as a difference is found.

    Args:
        a: First value
        b: Second value

    Returns:
        Whether values are deeply equal
    """
    # Fast path: reference equality
    if a is b:
        return True

    # Fast path: null/None checks
    if a is None or b is None:
        return a is b

    type_a = type(a)
    type_b = type(b)

    # Fast path: type mismatch
    if type_a != type_b:
        # Special case: int and float
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            numeric_eq: bool = a == b
            return numeric_eq
        return False

    # Primitives that aren't equal
    if isinstance(a, (str, int, float, bool)):
        primitive_eq: bool = a == b
        return primitive_eq

    # Array comparison with early termination
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not deep_equal(a[i], b[i]):
                return False
        return True

    # Object comparison with early termination
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for key in a:
            if not deep_equal(a[key], b[key]):
                return False
        return True

    # Fallback to regular equality
    fallback_eq: bool = a == b
    return fallback_eq


# ─────────────────────────────────────────────────────────────────────────────
# Object/Array Comparison with Differences
# ─────────────────────────────────────────────────────────────────────────────


def compare_values(
    expected: Any,
    actual: Any,
    options: ObjectComparisonOptions | None = None,
    path: str = "",
) -> list[Difference]:
    """Compare two values (generic).

    Args:
        expected: Expected value
        actual: Actual value
        options: Comparison options
        path: Current path

    Returns:
        Array of differences
    """
    if options is None:
        options = ObjectComparisonOptions()

    # Exact match
    if expected == actual:
        return []

    # Type mismatch
    expected_type = get_type(expected)
    actual_type = get_type(actual)

    if expected_type != actual_type:
        return [
            Difference(
                path=path,
                expected=expected,
                actual=actual,
                type=DifferenceType.TYPE_MISMATCH,
                severity=DifferenceSeverity.ERROR,
                message=f"Type mismatch: expected {expected_type}, got {actual_type}",
            )
        ]

    # Type-specific comparison
    if expected_type in ("null",):
        if expected != actual:
            return [
                Difference(
                    path=path,
                    expected=expected,
                    actual=actual,
                    type=DifferenceType.DIFFERENT,
                    severity=DifferenceSeverity.ERROR,
                    message=f"Expected {expected}, got {actual}",
                )
            ]
        return []

    if expected_type == "number":
        if compare_numbers(expected, actual, options.numeric_tolerance):
            return []
        return [
            Difference(
                path=path,
                expected=expected,
                actual=actual,
                type=DifferenceType.DIFFERENT,
                severity=DifferenceSeverity.ERROR,
                message=f"Numbers differ: {expected} vs {actual}",
            )
        ]

    if expected_type == "string":
        if expected == actual:
            return []

        similarity = compare_strings(
            expected,
            actual,
            case_sensitive=True,
            normalize_whitespace=True,
            algorithm="levenshtein",
        )

        if options.style == "lenient" and similarity >= 0.8:
            return [
                Difference(
                    path=path,
                    expected=expected,
                    actual=actual,
                    type=DifferenceType.DIFFERENT,
                    severity=DifferenceSeverity.WARNING,
                    message=f"Strings differ but similar ({similarity * 100:.0f}%)",
                    similarity=similarity,
                )
            ]

        return [
            Difference(
                path=path,
                expected=expected,
                actual=actual,
                type=DifferenceType.DIFFERENT,
                severity=DifferenceSeverity.ERROR,
                message="Strings differ",
                similarity=similarity,
            )
        ]

    if expected_type == "boolean":
        return [
            Difference(
                path=path,
                expected=expected,
                actual=actual,
                type=DifferenceType.DIFFERENT,
                severity=DifferenceSeverity.ERROR,
                message=f"Boolean mismatch: {expected} vs {actual}",
            )
        ]

    if expected_type == "array":
        return compare_arrays(expected, actual, options, path)

    if expected_type == "object":
        return compare_objects(expected, actual, options, path)

    return [
        Difference(
            path=path,
            expected=expected,
            actual=actual,
            type=DifferenceType.DIFFERENT,
            severity=DifferenceSeverity.ERROR,
            message="Values differ",
        )
    ]


def compare_arrays(
    a: list[Any],
    b: list[Any],
    options: ObjectComparisonOptions,
    path: str = "",
) -> list[Difference]:
    """Compare two arrays.

    Args:
        a: First array (expected)
        b: Second array (actual)
        options: Comparison options
        path: Current path

    Returns:
        Array of differences
    """
    import json

    differences: list[Difference] = []

    if options.ignore_array_order:
        # Compare as sets (order doesn't matter)
        a_set = {json.dumps(item, sort_keys=True) for item in a}
        b_set = {json.dumps(item, sort_keys=True) for item in b}

        # Find items in a but not in b
        for item in a_set:
            if item not in b_set:
                differences.append(
                    Difference(
                        path=f"{path}[]",
                        expected=json.loads(item),
                        actual=None,
                        type=DifferenceType.MISSING,
                        severity=(
                            DifferenceSeverity.ERROR
                            if options.style == "strict"
                            else DifferenceSeverity.WARNING
                        ),
                        message="Item missing in actual array",
                    )
                )

        # Find items in b but not in a
        for item in b_set:
            if item not in a_set:
                differences.append(
                    Difference(
                        path=f"{path}[]",
                        expected=None,
                        actual=json.loads(item),
                        type=DifferenceType.EXTRA,
                        severity=(
                            DifferenceSeverity.INFO
                            if options.ignore_extra_fields
                            else DifferenceSeverity.WARNING
                        ),
                        message="Extra item in actual array",
                    )
                )
    else:
        # Compare with order
        max_length = max(len(a), len(b))

        for i in range(max_length):
            item_path = f"{path}[{i}]"

            if i >= len(a):
                differences.append(
                    Difference(
                        path=item_path,
                        expected=None,
                        actual=b[i],
                        type=DifferenceType.EXTRA,
                        severity=(
                            DifferenceSeverity.INFO
                            if options.ignore_extra_fields
                            else DifferenceSeverity.WARNING
                        ),
                        message=f"Extra item at index {i}",
                    )
                )
            elif i >= len(b):
                differences.append(
                    Difference(
                        path=item_path,
                        expected=a[i],
                        actual=None,
                        type=DifferenceType.MISSING,
                        severity=DifferenceSeverity.ERROR,
                        message=f"Missing item at index {i}",
                    )
                )
            else:
                # Compare items
                item_diffs = compare_values(a[i], b[i], options, item_path)
                differences.extend(item_diffs)

    return differences


def compare_objects(
    expected: dict[str, Any],
    actual: dict[str, Any],
    options: ObjectComparisonOptions,
    path: str = "",
) -> list[Difference]:
    """Compare two objects deeply.

    Args:
        expected: Expected object
        actual: Actual object
        options: Comparison options
        path: Current path

    Returns:
        Array of differences
    """
    differences: list[Difference] = []

    # Get all keys
    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())
    all_keys = expected_keys | actual_keys

    for key in all_keys:
        field_path = f"{path}.{key}" if path else key
        has_expected = key in expected
        has_actual = key in actual

        # Check for custom comparison
        if field_path in options.custom_comparisons:
            custom_result = options.custom_comparisons[field_path](
                expected.get(key),
                actual.get(key),
            )

            if isinstance(custom_result, bool) and not custom_result:
                differences.append(
                    Difference(
                        path=field_path,
                        expected=expected.get(key),
                        actual=actual.get(key),
                        type=DifferenceType.DIFFERENT,
                        severity=DifferenceSeverity.ERROR,
                        message=f"Custom comparison failed for {field_path}",
                    )
                )
            elif isinstance(custom_result, (int, float)) and custom_result < 0.8:
                differences.append(
                    Difference(
                        path=field_path,
                        expected=expected.get(key),
                        actual=actual.get(key),
                        type=DifferenceType.DIFFERENT,
                        severity=DifferenceSeverity.WARNING,
                        message=f"Custom comparison score too low: {custom_result:.2f}",
                        similarity=float(custom_result),
                    )
                )
            continue

        if not has_expected and has_actual:
            # Extra field in actual
            if not options.ignore_extra_fields:
                differences.append(
                    Difference(
                        path=field_path,
                        expected=None,
                        actual=actual[key],
                        type=DifferenceType.EXTRA,
                        severity=(
                            DifferenceSeverity.ERROR
                            if options.style == "strict"
                            else DifferenceSeverity.INFO
                        ),
                        message=f"Extra field: {key}",
                    )
                )
        elif has_expected and not has_actual:
            # Missing field in actual
            differences.append(
                Difference(
                    path=field_path,
                    expected=expected[key],
                    actual=None,
                    type=DifferenceType.MISSING,
                    severity=DifferenceSeverity.ERROR,
                    message=f"Missing field: {key}",
                )
            )
        else:
            # Both exist, compare values
            value_diffs = compare_values(
                expected[key],
                actual[key],
                options,
                field_path,
            )
            differences.extend(value_diffs)

    return differences


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def calculate_similarity_score(
    differences: list[Difference],
    total_fields: int,
) -> float:
    """Calculate overall similarity score from differences.

    Args:
        differences: Array of differences
        total_fields: Total number of fields compared

    Returns:
        Similarity score (0-1)
    """
    if total_fields == 0:
        return 1.0

    # Weight differences by severity
    weights = {
        DifferenceSeverity.ERROR: 1.0,
        DifferenceSeverity.WARNING: 0.5,
        DifferenceSeverity.INFO: 0.1,
    }

    total_penalty = sum(weights[diff.severity] for diff in differences)

    max_penalty = total_fields
    return max(0.0, 1 - total_penalty / max_penalty)


def count_fields(value: Any) -> int:
    """Count total fields in a value (for scoring).

    Args:
        value: Value to count fields in

    Returns:
        Total number of fields
    """
    value_type = get_type(value)

    if value_type == "object":
        return sum(1 + count_fields(v) for v in value.values())

    if value_type == "array":
        return sum(count_fields(item) for item in value)

    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class Compare:
    """Scoped API for comparison utilities.

    Provides string similarity algorithms, object comparison, and deep equality
    utilities for evaluation and testing.

    Usage:
        ```python
        from l0 import Compare

        # String similarity
        score = Compare.strings("hello", "hallo")
        distance = Compare.levenshtein_distance("cat", "bat")
        similarity = Compare.levenshtein("cat", "bat")
        similarity = Compare.jaro_winkler("hello", "hallo")
        similarity = Compare.cosine("hello world", "hello there")

        # Numeric comparison
        equal = Compare.numbers(1.0001, 1.0002, tolerance=0.001)

        # Deep equality
        equal = Compare.deep_equal({"a": 1}, {"a": 1})

        # Object comparison with differences
        diffs = Compare.objects(expected, actual)
        diffs = Compare.arrays([1, 2], [1, 3])
        diffs = Compare.values(expected, actual)

        # Scoring
        score = Compare.similarity_score(differences, total_fields)
        count = Compare.count_fields({"a": {"b": 1}})

        # Type utilities
        type_str = Compare.get_type(value)
        ```
    """

    # Re-export types for convenience
    Difference = Difference
    DifferenceSeverity = DifferenceSeverity
    DifferenceType = DifferenceType
    StringComparisonOptions = StringComparisonOptions
    ObjectComparisonOptions = ObjectComparisonOptions

    @staticmethod
    def strings(
        a: str,
        b: str,
        *,
        case_sensitive: bool = True,
        normalize_whitespace: bool = True,
        algorithm: Literal["levenshtein", "jaro-winkler", "cosine"] = "levenshtein",
    ) -> float:
        """Compare two strings with similarity scoring.

        Args:
            a: First string
            b: Second string
            case_sensitive: Whether comparison is case-sensitive
            normalize_whitespace: Whether to normalize whitespace
            algorithm: Similarity algorithm to use

        Returns:
            Similarity score (0-1)
        """
        return compare_strings(
            a,
            b,
            case_sensitive=case_sensitive,
            normalize_whitespace=normalize_whitespace,
            algorithm=algorithm,
        )

    @staticmethod
    def levenshtein_distance(a: str, b: str) -> int:
        """Calculate Levenshtein distance (edit distance).

        Args:
            a: First string
            b: Second string

        Returns:
            Number of edits needed to transform a into b
        """
        return levenshtein_distance(a, b)

    @staticmethod
    def levenshtein(a: str, b: str) -> float:
        """Levenshtein distance similarity (0-1).

        Args:
            a: First string
            b: Second string

        Returns:
            Similarity score (0-1)
        """
        return levenshtein_similarity(a, b)

    @staticmethod
    def jaro_winkler(a: str, b: str) -> float:
        """Jaro-Winkler similarity (0-1).

        Adds a bonus for matching prefixes.

        Args:
            a: First string
            b: Second string

        Returns:
            Similarity score (0-1)
        """
        return jaro_winkler_similarity(a, b)

    @staticmethod
    def cosine(a: str, b: str) -> float:
        """Cosine similarity (0-1).

        Args:
            a: First string
            b: Second string

        Returns:
            Similarity score (0-1)
        """
        return cosine_similarity(a, b)

    @staticmethod
    def numbers(a: float, b: float, tolerance: float = 0.001) -> bool:
        """Compare two numbers with tolerance.

        Args:
            a: First number
            b: Second number
            tolerance: Acceptable difference

        Returns:
            Whether numbers are equal within tolerance
        """
        return compare_numbers(a, b, tolerance)

    @staticmethod
    def deep_equal(a: Any, b: Any) -> bool:
        """Deep equality check with early termination.

        Returns False as soon as a difference is found.

        Args:
            a: First value
            b: Second value

        Returns:
            Whether values are deeply equal
        """
        return deep_equal(a, b)

    @staticmethod
    def values(
        expected: Any,
        actual: Any,
        options: ObjectComparisonOptions | None = None,
        path: str = "",
    ) -> list[Difference]:
        """Compare two values (generic).

        Args:
            expected: Expected value
            actual: Actual value
            options: Comparison options
            path: Current path

        Returns:
            Array of differences
        """
        return compare_values(expected, actual, options, path)

    @staticmethod
    def arrays(
        a: list[Any],
        b: list[Any],
        options: ObjectComparisonOptions | None = None,
        path: str = "",
    ) -> list[Difference]:
        """Compare two arrays.

        Args:
            a: First array (expected)
            b: Second array (actual)
            options: Comparison options
            path: Current path

        Returns:
            Array of differences
        """
        if options is None:
            options = ObjectComparisonOptions()
        return compare_arrays(a, b, options, path)

    @staticmethod
    def objects(
        expected: dict[str, Any],
        actual: dict[str, Any],
        options: ObjectComparisonOptions | None = None,
        path: str = "",
    ) -> list[Difference]:
        """Compare two objects deeply.

        Args:
            expected: Expected object
            actual: Actual object
            options: Comparison options
            path: Current path

        Returns:
            Array of differences
        """
        if options is None:
            options = ObjectComparisonOptions()
        return compare_objects(expected, actual, options, path)

    @staticmethod
    def similarity_score(
        differences: list[Difference],
        total_fields: int,
    ) -> float:
        """Calculate overall similarity score from differences.

        Args:
            differences: Array of differences
            total_fields: Total number of fields compared

        Returns:
            Similarity score (0-1)
        """
        return calculate_similarity_score(differences, total_fields)

    @staticmethod
    def count_fields(value: Any) -> int:
        """Count total fields in a value (for scoring).

        Args:
            value: Value to count fields in

        Returns:
            Total number of fields
        """
        return count_fields(value)

    @staticmethod
    def get_type(value: Any) -> str:
        """Get type of value as string.

        Args:
            value: Value to check

        Returns:
            Type string ("null", "boolean", "number", "string", "array", "object", "unknown")
        """
        return get_type(value)
