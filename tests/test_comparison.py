"""Tests for comparison utilities."""

import pytest

from l0 import (
    Compare,
    Difference,
    DifferenceSeverity,
    DifferenceType,
    ObjectComparisonOptions,
)


class TestLevenshteinDistance:
    """Tests for Compare.levenshtein_distance."""

    def test_identical_strings(self) -> None:
        assert Compare.levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self) -> None:
        assert Compare.levenshtein_distance("", "") == 0
        assert Compare.levenshtein_distance("hello", "") == 5
        assert Compare.levenshtein_distance("", "hello") == 5

    def test_single_substitution(self) -> None:
        assert Compare.levenshtein_distance("hello", "hallo") == 1

    def test_single_insertion(self) -> None:
        assert Compare.levenshtein_distance("hello", "helloo") == 1

    def test_single_deletion(self) -> None:
        assert Compare.levenshtein_distance("hello", "hell") == 1

    def test_completely_different(self) -> None:
        assert Compare.levenshtein_distance("abc", "xyz") == 3


class TestLevenshteinSimilarity:
    """Tests for Compare.levenshtein."""

    def test_identical_strings(self) -> None:
        assert Compare.levenshtein("hello", "hello") == 1.0

    def test_empty_strings(self) -> None:
        assert Compare.levenshtein("", "") == 1.0
        assert Compare.levenshtein("hello", "") == 0.0
        assert Compare.levenshtein("", "hello") == 0.0

    def test_similar_strings(self) -> None:
        # "hello" vs "hallo" = 1 edit, max length 5
        # similarity = 1 - 1/5 = 0.8
        assert Compare.levenshtein("hello", "hallo") == 0.8

    def test_very_different_strings(self) -> None:
        similarity = Compare.levenshtein("abc", "xyz")
        assert similarity == 0.0  # 3 edits, max length 3


class TestJaroWinklerSimilarity:
    """Tests for Compare.jaro_winkler."""

    def test_identical_strings(self) -> None:
        assert Compare.jaro_winkler("hello", "hello") == 1.0

    def test_empty_strings(self) -> None:
        assert Compare.jaro_winkler("", "") == 1.0
        assert Compare.jaro_winkler("hello", "") == 0.0

    def test_similar_strings(self) -> None:
        # Jaro-Winkler gives bonus for matching prefix
        similarity = Compare.jaro_winkler("hello", "hallo")
        assert similarity > 0.8

    def test_prefix_bonus(self) -> None:
        # Jaro-Winkler gives bonus for matching prefix
        # "hello" vs "hella" should score higher than "hello" vs "xello"
        sim1 = Compare.jaro_winkler("hello", "hella")
        sim2 = Compare.jaro_winkler("hello", "xello")
        assert sim1 > sim2  # Same prefix "hell" vs different first char


class TestCosineSimilarity:
    """Tests for Compare.cosine."""

    def test_identical_strings(self) -> None:
        assert Compare.cosine("hello world", "hello world") == 1.0

    def test_empty_strings(self) -> None:
        assert Compare.cosine("", "") == 1.0
        assert Compare.cosine("hello", "") == 0.0

    def test_similar_content(self) -> None:
        similarity = Compare.cosine(
            "the quick brown fox",
            "the quick brown dog",
        )
        assert similarity > 0.7  # 3/4 words match

    def test_completely_different(self) -> None:
        similarity = Compare.cosine("hello world", "foo bar baz")
        assert similarity == 0.0


class TestCompareStrings:
    """Tests for Compare.strings."""

    def test_identical_strings(self) -> None:
        assert Compare.strings("hello", "hello") == 1.0

    def test_case_insensitive(self) -> None:
        assert Compare.strings("Hello", "hello", case_sensitive=False) == 1.0
        assert Compare.strings("Hello", "hello", case_sensitive=True) < 1.0

    def test_whitespace_normalization(self) -> None:
        assert (
            Compare.strings(
                "hello   world",
                "hello world",
                normalize_whitespace=True,
            )
            == 1.0
        )

    def test_levenshtein_algorithm(self) -> None:
        similarity = Compare.strings("hello", "hallo", algorithm="levenshtein")
        assert similarity == 0.8

    def test_jaro_winkler_algorithm(self) -> None:
        similarity = Compare.strings("hello", "hallo", algorithm="jaro-winkler")
        assert similarity > 0.8

    def test_cosine_algorithm(self) -> None:
        similarity = Compare.strings(
            "the quick brown",
            "the quick brown",
            algorithm="cosine",
        )
        assert similarity == 1.0


class TestCompareNumbers:
    """Tests for Compare.numbers."""

    def test_equal_numbers(self) -> None:
        assert Compare.numbers(1.0, 1.0) is True

    def test_within_tolerance(self) -> None:
        assert Compare.numbers(1.0, 1.0005, tolerance=0.001) is True

    def test_outside_tolerance(self) -> None:
        assert Compare.numbers(1.0, 1.01, tolerance=0.001) is False

    def test_integers(self) -> None:
        assert Compare.numbers(5, 5) is True
        assert Compare.numbers(5, 6) is False


class TestGetType:
    """Tests for Compare.get_type."""

    def test_null(self) -> None:
        assert Compare.get_type(None) == "null"

    def test_boolean(self) -> None:
        assert Compare.get_type(True) == "boolean"
        assert Compare.get_type(False) == "boolean"

    def test_number(self) -> None:
        assert Compare.get_type(42) == "number"
        assert Compare.get_type(3.14) == "number"

    def test_string(self) -> None:
        assert Compare.get_type("hello") == "string"

    def test_array(self) -> None:
        assert Compare.get_type([1, 2, 3]) == "array"

    def test_object(self) -> None:
        assert Compare.get_type({"key": "value"}) == "object"


class TestDeepEqual:
    """Tests for Compare.deep_equal."""

    def test_primitives(self) -> None:
        assert Compare.deep_equal(1, 1) is True
        assert Compare.deep_equal(1, 2) is False
        assert Compare.deep_equal("hello", "hello") is True
        assert Compare.deep_equal("hello", "world") is False
        assert Compare.deep_equal(True, True) is True
        assert Compare.deep_equal(True, False) is False

    def test_none(self) -> None:
        assert Compare.deep_equal(None, None) is True
        assert Compare.deep_equal(None, 1) is False

    def test_arrays(self) -> None:
        assert Compare.deep_equal([1, 2, 3], [1, 2, 3]) is True
        assert Compare.deep_equal([1, 2, 3], [1, 2, 4]) is False
        assert Compare.deep_equal([1, 2], [1, 2, 3]) is False

    def test_nested_arrays(self) -> None:
        assert Compare.deep_equal([[1, 2], [3, 4]], [[1, 2], [3, 4]]) is True
        assert Compare.deep_equal([[1, 2], [3, 4]], [[1, 2], [3, 5]]) is False

    def test_objects(self) -> None:
        assert Compare.deep_equal({"a": 1}, {"a": 1}) is True
        assert Compare.deep_equal({"a": 1}, {"a": 2}) is False
        assert Compare.deep_equal({"a": 1}, {"b": 1}) is False

    def test_nested_objects(self) -> None:
        assert (
            Compare.deep_equal(
                {"a": {"b": 1}},
                {"a": {"b": 1}},
            )
            is True
        )
        assert (
            Compare.deep_equal(
                {"a": {"b": 1}},
                {"a": {"b": 2}},
            )
            is False
        )

    def test_mixed_types(self) -> None:
        assert Compare.deep_equal(1, "1") is False
        assert Compare.deep_equal([1], {"0": 1}) is False

    def test_int_float_equality(self) -> None:
        # Special case: int and float should compare by value
        assert Compare.deep_equal(1, 1.0) is True
        assert Compare.deep_equal(1, 1.5) is False


class TestCompareValues:
    """Tests for Compare.values."""

    def test_identical_values(self) -> None:
        assert Compare.values("hello", "hello") == []

    def test_type_mismatch(self) -> None:
        diffs = Compare.values("hello", 123)
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.TYPE_MISMATCH

    def test_string_difference(self) -> None:
        diffs = Compare.values("hello", "world")
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.DIFFERENT
        assert diffs[0].similarity is not None

    def test_number_within_tolerance(self) -> None:
        options = ObjectComparisonOptions(numeric_tolerance=0.01)
        diffs = Compare.values(1.0, 1.005, options)
        assert diffs == []

    def test_number_outside_tolerance(self) -> None:
        options = ObjectComparisonOptions(numeric_tolerance=0.001)
        diffs = Compare.values(1.0, 1.1, options)
        assert len(diffs) == 1


class TestCompareObjects:
    """Tests for Compare.objects."""

    def test_identical_objects(self) -> None:
        obj = {"a": 1, "b": 2}
        assert Compare.objects(obj, obj) == []

    def test_missing_field(self) -> None:
        expected = {"a": 1, "b": 2}
        actual = {"a": 1}
        diffs = Compare.objects(expected, actual)
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.MISSING
        assert diffs[0].path == "b"

    def test_extra_field(self) -> None:
        expected = {"a": 1}
        actual = {"a": 1, "b": 2}
        options = ObjectComparisonOptions(ignore_extra_fields=False)
        diffs = Compare.objects(expected, actual, options)
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.EXTRA

    def test_ignore_extra_fields(self) -> None:
        expected = {"a": 1}
        actual = {"a": 1, "b": 2}
        options = ObjectComparisonOptions(ignore_extra_fields=True)
        diffs = Compare.objects(expected, actual, options)
        assert diffs == []

    def test_nested_difference(self) -> None:
        expected = {"a": {"b": 1}}
        actual = {"a": {"b": 2}}
        diffs = Compare.objects(expected, actual)
        assert len(diffs) == 1
        assert diffs[0].path == "a.b"


class TestCompareArrays:
    """Tests for Compare.arrays."""

    def test_identical_arrays(self) -> None:
        arr = [1, 2, 3]
        assert Compare.arrays(arr, arr) == []

    def test_missing_item(self) -> None:
        expected = [1, 2, 3]
        actual = [1, 2]
        diffs = Compare.arrays(expected, actual)
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.MISSING

    def test_extra_item(self) -> None:
        expected = [1, 2]
        actual = [1, 2, 3]
        diffs = Compare.arrays(expected, actual)
        assert len(diffs) == 1
        assert diffs[0].type == DifferenceType.EXTRA

    def test_ignore_array_order(self) -> None:
        expected = [1, 2, 3]
        actual = [3, 1, 2]
        options = ObjectComparisonOptions(ignore_array_order=True)
        diffs = Compare.arrays(expected, actual, options)
        assert diffs == []


class TestCountFields:
    """Tests for Compare.count_fields."""

    def test_primitive(self) -> None:
        assert Compare.count_fields(1) == 1
        assert Compare.count_fields("hello") == 1

    def test_simple_object(self) -> None:
        assert Compare.count_fields({"a": 1, "b": 2}) == 4  # 2 keys + 2 values

    def test_nested_object(self) -> None:
        obj = {"a": {"b": 1}}
        # "a" (1) + nested object: "b" (1) + value (1) = 3
        assert Compare.count_fields(obj) == 3

    def test_array(self) -> None:
        assert Compare.count_fields([1, 2, 3]) == 3


class TestCalculateSimilarityScore:
    """Tests for Compare.similarity_score."""

    def test_no_differences(self) -> None:
        assert Compare.similarity_score([], 10) == 1.0

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
        score = Compare.similarity_score(diffs, 1)
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
        score = Compare.similarity_score(diffs, 2)
        # 1 warning (0.5 weight) out of 2 fields = 1 - 0.5/2 = 0.75
        assert score == 0.75

    def test_zero_fields(self) -> None:
        assert Compare.similarity_score([], 0) == 1.0


class TestCompareTypesAccessible:
    """Tests that types are accessible via Compare class."""

    def test_difference_type(self) -> None:
        assert Compare.Difference is Difference

    def test_difference_severity(self) -> None:
        assert Compare.DifferenceSeverity is DifferenceSeverity

    def test_difference_type_enum(self) -> None:
        assert Compare.DifferenceType is DifferenceType

    def test_options_types(self) -> None:
        assert Compare.StringComparisonOptions is not None
        assert Compare.ObjectComparisonOptions is not None
