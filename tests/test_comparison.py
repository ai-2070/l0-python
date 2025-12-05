"""Tests for comparison utilities."""

import pytest

from l0 import (
    Difference,
    DifferenceSeverity,
    DifferenceType,
    ObjectComparisonOptions,
    calculate_similarity_score,
    compare_arrays,
    compare_numbers,
    compare_objects,
    compare_strings,
    compare_values,
    cosine_similarity,
    count_fields,
    deep_equal,
    get_type,
    jaro_winkler_similarity,
    levenshtein_distance,
    levenshtein_similarity,
)


class TestLevenshteinDistance:
    """Tests for levenshtein_distance."""

    def test_identical_strings(self) -> None:
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self) -> None:
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "hello") == 5

    def test_single_substitution(self) -> None:
        assert levenshtein_distance("hello", "hallo") == 1

    def test_single_insertion(self) -> None:
        assert levenshtein_distance("hello", "helloo") == 1

    def test_single_deletion(self) -> None:
        assert levenshtein_distance("hello", "hell") == 1

    def test_completely_different(self) -> None:
        assert levenshtein_distance("abc", "xyz") == 3


class TestLevenshteinSimilarity:
    """Tests for levenshtein_similarity."""

    def test_identical_strings(self) -> None:
        assert levenshtein_similarity("hello", "hello") == 1.0

    def test_empty_strings(self) -> None:
        assert levenshtein_similarity("", "") == 1.0
        assert levenshtein_similarity("hello", "") == 0.0
        assert levenshtein_similarity("", "hello") == 0.0

    def test_similar_strings(self) -> None:
        # "hello" vs "hallo" = 1 edit, max length 5
        # similarity = 1 - 1/5 = 0.8
        assert levenshtein_similarity("hello", "hallo") == 0.8

    def test_very_different_strings(self) -> None:
        similarity = levenshtein_similarity("abc", "xyz")
        assert similarity == 0.0  # 3 edits, max length 3


class TestJaroWinklerSimilarity:
    """Tests for jaro_winkler_similarity."""

    def test_identical_strings(self) -> None:
        assert jaro_winkler_similarity("hello", "hello") == 1.0

    def test_empty_strings(self) -> None:
        assert jaro_winkler_similarity("", "") == 1.0
        assert jaro_winkler_similarity("hello", "") == 0.0

    def test_similar_strings(self) -> None:
        # Jaro-Winkler gives bonus for matching prefix
        similarity = jaro_winkler_similarity("hello", "hallo")
        assert similarity > 0.8

    def test_prefix_bonus(self) -> None:
        # Jaro-Winkler gives bonus for matching prefix
        # "hello" vs "hella" should score higher than "hello" vs "xello"
        sim1 = jaro_winkler_similarity("hello", "hella")
        sim2 = jaro_winkler_similarity("hello", "xello")
        assert sim1 > sim2  # Same prefix "hell" vs different first char


class TestCosineSimilarity:
    """Tests for cosine_similarity."""

    def test_identical_strings(self) -> None:
        assert cosine_similarity("hello world", "hello world") == 1.0

    def test_empty_strings(self) -> None:
        assert cosine_similarity("", "") == 1.0
        assert cosine_similarity("hello", "") == 0.0

    def test_similar_content(self) -> None:
        similarity = cosine_similarity(
            "the quick brown fox",
            "the quick brown dog",
        )
        assert similarity > 0.7  # 3/4 words match

    def test_completely_different(self) -> None:
        similarity = cosine_similarity("hello world", "foo bar baz")
        assert similarity == 0.0


class TestCompareStrings:
    """Tests for compare_strings."""

    def test_identical_strings(self) -> None:
        assert compare_strings("hello", "hello") == 1.0

    def test_case_insensitive(self) -> None:
        assert compare_strings("Hello", "hello", case_sensitive=False) == 1.0
        assert compare_strings("Hello", "hello", case_sensitive=True) < 1.0

    def test_whitespace_normalization(self) -> None:
        assert (
            compare_strings(
                "hello   world",
                "hello world",
                normalize_whitespace=True,
            )
            == 1.0
        )

    def test_levenshtein_algorithm(self) -> None:
        similarity = compare_strings("hello", "hallo", algorithm="levenshtein")
        assert similarity == 0.8

    def test_jaro_winkler_algorithm(self) -> None:
        similarity = compare_strings("hello", "hallo", algorithm="jaro-winkler")
        assert similarity > 0.8

    def test_cosine_algorithm(self) -> None:
        similarity = compare_strings(
            "the quick brown",
            "the quick brown",
            algorithm="cosine",
        )
        assert similarity == 1.0


class TestCompareNumbers:
    """Tests for compare_numbers."""

    def test_equal_numbers(self) -> None:
        assert compare_numbers(1.0, 1.0) is True

    def test_within_tolerance(self) -> None:
        assert compare_numbers(1.0, 1.0005, tolerance=0.001) is True

    def test_outside_tolerance(self) -> None:
        assert compare_numbers(1.0, 1.01, tolerance=0.001) is False

    def test_integers(self) -> None:
        assert compare_numbers(5, 5) is True
        assert compare_numbers(5, 6) is False


class TestGetType:
    """Tests for get_type."""

    def test_null(self) -> None:
        assert get_type(None) == "null"

    def test_boolean(self) -> None:
        assert get_type(True) == "boolean"
        assert get_type(False) == "boolean"

    def test_number(self) -> None:
        assert get_type(42) == "number"
        assert get_type(3.14) == "number"

    def test_string(self) -> None:
        assert get_type("hello") == "string"

    def test_array(self) -> None:
        assert get_type([1, 2, 3]) == "array"

    def test_object(self) -> None:
        assert get_type({"key": "value"}) == "object"


class TestDeepEqual:
    """Tests for deep_equal."""

    def test_primitives(self) -> None:
        assert deep_equal(1, 1) is True
        assert deep_equal(1, 2) is False
        assert deep_equal("hello", "hello") is True
        assert deep_equal("hello", "world") is False
        assert deep_equal(True, True) is True
        assert deep_equal(True, False) is False

    def test_none(self) -> None:
        assert deep_equal(None, None) is True
        assert deep_equal(None, 1) is False

    def test_arrays(self) -> None:
        assert deep_equal([1, 2, 3], [1, 2, 3]) is True
        assert deep_equal([1, 2, 3], [1, 2, 4]) is False
        assert deep_equal([1, 2], [1, 2, 3]) is False

    def test_nested_arrays(self) -> None:
        assert deep_equal([[1, 2], [3, 4]], [[1, 2], [3, 4]]) is True
        assert deep_equal([[1, 2], [3, 4]], [[1, 2], [3, 5]]) is False

    def test_objects(self) -> None:
        assert deep_equal({"a": 1}, {"a": 1}) is True
        assert deep_equal({"a": 1}, {"a": 2}) is False
        assert deep_equal({"a": 1}, {"b": 1}) is False

    def test_nested_objects(self) -> None:
        assert (
            deep_equal(
                {"a": {"b": 1}},
                {"a": {"b": 1}},
            )
            is True
        )
        assert (
            deep_equal(
                {"a": {"b": 1}},
                {"a": {"b": 2}},
            )
            is False
        )

    def test_mixed_types(self) -> None:
        assert deep_equal(1, "1") is False
        assert deep_equal([1], {"0": 1}) is False

    def test_int_float_equality(self) -> None:
        # Special case: int and float should compare by value
        assert deep_equal(1, 1.0) is True
        assert deep_equal(1, 1.5) is False


class TestCompareValues:
    """Tests for compare_values."""

    def test_identical_values(self) -> None:
        assert compare_values("hello", "hello") == []

    def test_type_mismatch(self) -> None:
        diffs = compare_values("hello", 123)
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.TYPE_MISMATCH

    def test_string_difference(self) -> None:
        diffs = compare_values("hello", "world")
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.DIFFERENT
        assert diffs[0].similarity is not None

    def test_number_within_tolerance(self) -> None:
        options = ObjectComparisonOptions(numeric_tolerance=0.01)
        diffs = compare_values(1.0, 1.005, options)
        assert diffs == []

    def test_number_outside_tolerance(self) -> None:
        options = ObjectComparisonOptions(numeric_tolerance=0.001)
        diffs = compare_values(1.0, 1.1, options)
        assert len(diffs) == 1


class TestCompareObjects:
    """Tests for compare_objects."""

    def test_identical_objects(self) -> None:
        obj = {"a": 1, "b": 2}
        assert compare_objects(obj, obj, ObjectComparisonOptions()) == []

    def test_missing_field(self) -> None:
        expected = {"a": 1, "b": 2}
        actual = {"a": 1}
        diffs = compare_objects(expected, actual, ObjectComparisonOptions())
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.MISSING
        assert diffs[0].path == "b"

    def test_extra_field(self) -> None:
        expected = {"a": 1}
        actual = {"a": 1, "b": 2}
        options = ObjectComparisonOptions(ignore_extra_fields=False)
        diffs = compare_objects(expected, actual, options)
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.EXTRA

    def test_ignore_extra_fields(self) -> None:
        expected = {"a": 1}
        actual = {"a": 1, "b": 2}
        options = ObjectComparisonOptions(ignore_extra_fields=True)
        diffs = compare_objects(expected, actual, options)
        assert diffs == []

    def test_nested_difference(self) -> None:
        expected = {"a": {"b": 1}}
        actual = {"a": {"b": 2}}
        diffs = compare_objects(expected, actual, ObjectComparisonOptions())
        assert len(diffs) == 1
        assert diffs[0].path == "a.b"


class TestCompareArrays:
    """Tests for compare_arrays."""

    def test_identical_arrays(self) -> None:
        arr = [1, 2, 3]
        assert compare_arrays(arr, arr, ObjectComparisonOptions()) == []

    def test_missing_item(self) -> None:
        expected = [1, 2, 3]
        actual = [1, 2]
        diffs = compare_arrays(expected, actual, ObjectComparisonOptions())
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.MISSING

    def test_extra_item(self) -> None:
        expected = [1, 2]
        actual = [1, 2, 3]
        diffs = compare_arrays(expected, actual, ObjectComparisonOptions())
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.EXTRA

    def test_ignore_array_order(self) -> None:
        expected = [1, 2, 3]
        actual = [3, 1, 2]
        options = ObjectComparisonOptions(ignore_array_order=True)
        diffs = compare_arrays(expected, actual, options)
        assert diffs == []


class TestCountFields:
    """Tests for count_fields."""

    def test_primitive(self) -> None:
        assert count_fields(1) == 1
        assert count_fields("hello") == 1

    def test_simple_object(self) -> None:
        assert count_fields({"a": 1, "b": 2}) == 4  # 2 keys + 2 values

    def test_nested_object(self) -> None:
        obj = {"a": {"b": 1}}
        # "a" (1) + nested object: "b" (1) + value (1) = 3
        assert count_fields(obj) == 3

    def test_array(self) -> None:
        assert count_fields([1, 2, 3]) == 3


class TestCalculateSimilarityScore:
    """Tests for calculate_similarity_score."""

    def test_no_differences(self) -> None:
        assert calculate_similarity_score([], 10) == 1.0

    def test_all_errors(self) -> None:
        diffs = [
            Difference(
                path="a",
                expected=1,
                actual=2,
                type=DifferenceType.DIFFERENT,
                severity=DifferenceSeverity.ERROR,
                message="test",
            ),
        ]
        score = calculate_similarity_score(diffs, 1)
        assert score == 0.0

    def test_warnings_weighted(self) -> None:
        diffs = [
            Difference(
                path="a",
                expected=1,
                actual=2,
                type=DifferenceType.DIFFERENT,
                severity=DifferenceSeverity.WARNING,
                message="test",
            ),
        ]
        score = calculate_similarity_score(diffs, 2)
        # 1 warning (0.5 weight) out of 2 fields = 1 - 0.5/2 = 0.75
        assert score == 0.75

    def test_zero_fields(self) -> None:
        assert calculate_similarity_score([], 0) == 1.0
