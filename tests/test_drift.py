"""Comprehensive drift detection tests for L0.

Tests for DriftDetector class and helper functions, matching TypeScript implementation.
"""

from __future__ import annotations

import pytest

from l0.drift import (
    DriftConfig,
    DriftDetector,
    DriftResult,
    check_drift,
    create_drift_detector,
)


class TestDriftDetectorInitialization:
    """Tests for DriftDetector initialization."""

    def test_initialize_with_default_config(self):
        """Should initialize with default config."""
        detector = DriftDetector()
        assert detector is not None

    def test_initialize_with_custom_config(self):
        """Should initialize with custom config."""
        custom_detector = DriftDetector(
            DriftConfig(
                detect_tone_shift=False,
                detect_meta_commentary=True,
                repetition_threshold=5,
            )
        )
        assert custom_detector is not None

    def test_initialize_with_all_detection_disabled(self):
        """Should initialize with all detection disabled."""
        no_detection = DriftDetector(
            DriftConfig(
                detect_tone_shift=False,
                detect_meta_commentary=False,
                detect_repetition=False,
                detect_entropy_spike=False,
            )
        )
        assert no_detection is not None

    def test_apply_default_thresholds(self):
        """Should apply default thresholds."""
        default_detector = DriftDetector(DriftConfig())
        assert default_detector is not None


class TestMetaCommentaryDetection:
    """Tests for meta commentary detection."""

    def test_detect_as_an_ai_patterns(self):
        """Should detect 'as an ai' patterns."""
        detector = DriftDetector()
        content = "As an AI language model, I think this is interesting."
        result = detector.check(content)
        assert result.detected is True
        assert "meta_commentary" in result.types
        assert result.confidence >= 0.8

    def test_detect_im_an_ai_patterns(self):
        """Should detect 'i'm an ai' patterns."""
        detector = DriftDetector()
        content = "I'm an AI assistant and I can help you with that."
        result = detector.check(content)
        assert result.detected is True
        assert "meta_commentary" in result.types

    def test_detect_apology_patterns(self):
        """Should detect apology patterns."""
        detector = DriftDetector()
        content = "I apologize, but I cannot provide that information."
        result = detector.check(content)
        assert result.detected is True
        assert "meta_commentary" in result.types

    def test_detect_cannot_actually_patterns(self):
        """Should detect 'i cannot actually' patterns."""
        detector = DriftDetector()
        content = "I cannot actually perform that task for you."
        result = detector.check(content)
        assert result.detected is True
        assert "meta_commentary" in result.types

    def test_detect_clarification_patterns(self):
        """Should detect clarification patterns."""
        detector = DriftDetector()
        content = "Let me explain how this works in more detail."
        result = detector.check(content)
        assert result.detected is True
        assert "meta_commentary" in result.types

    def test_not_detect_in_normal_text(self):
        """Should not detect meta commentary in normal text."""
        detector = DriftDetector()
        content = "The weather today is sunny and warm."
        result = detector.check(content)
        assert "meta_commentary" not in result.types

    def test_detect_in_recent_text(self):
        """Should detect meta commentary in recent text."""
        detector = DriftDetector()
        long_content = (
            "This is a long response. " * 20 + "As an AI, I should mention that."
        )
        result = detector.check(long_content)
        assert result.detected is True
        assert "meta_commentary" in result.types


class TestToneShiftDetection:
    """Tests for tone shift detection."""

    def test_detect_formal_to_informal_shift(self):
        """Should detect shift from formal to informal."""
        detector = DriftDetector()
        previous = (
            "Therefore, we must consider the implications. "
            "Thus, we proceed with the analysis. "
            "Hence, we conclude the matter. "
        ) * 3
        detector.check(previous)
        current = (
            previous + " Yeah, that's gonna be awesome! Wanna try it? Gonna be great!"
        )
        result = detector.check(current)
        assert "tone_shift" in result.types

    def test_detect_informal_to_formal_shift(self):
        """Should detect shift from informal to formal."""
        detector = DriftDetector()
        previous = "Yeah, that's cool and stuff. Gonna be great! Wanna try it out? " * 3
        detector.check(previous)
        current = (
            previous
            + " Therefore, we must consequently analyze furthermore. "
            + "Thus, we proceed accordingly. Hence, we conclude."
        )
        result = detector.check(current)
        assert "tone_shift" in result.types

    def test_not_detect_with_consistent_formal_tone(self):
        """Should not detect tone shift with consistent formal tone."""
        detector = DriftDetector()
        previous = "Therefore, we must consider the implications. Thus, we proceed."
        detector.check(previous)
        current = previous + " Hence, we conclude the analysis."
        result = detector.check(current)
        assert "tone_shift" not in result.types

    def test_not_detect_with_consistent_informal_tone(self):
        """Should not detect tone shift with consistent informal tone."""
        detector = DriftDetector()
        previous = "Yeah, that's cool. Gonna be great!"
        detector.check(previous)
        current = previous + " Wanna try it out?"
        result = detector.check(current)
        assert "tone_shift" not in result.types

    def test_not_detect_on_first_check(self):
        """Should not detect tone shift on first check."""
        detector = DriftDetector()
        content = "Yeah, that's gonna be awesome!"
        result = detector.check(content)
        assert "tone_shift" not in result.types

    def test_require_sufficient_previous_content(self):
        """Should require sufficient previous content."""
        detector = DriftDetector()
        detector.check("Short")
        result = detector.check("Short text")
        assert "tone_shift" not in result.types


class TestRepetitionDetection:
    """Tests for repetition detection."""

    def test_detect_repeated_sentences(self):
        """Should detect repeated sentences."""
        detector = DriftDetector()
        content = (
            "This is a test sentence that repeats. "
            "This is a test sentence that repeats. "
            "This is a test sentence that repeats."
        )
        result = detector.check(content)
        assert result.detected is True
        assert "repetition" in result.types

    def test_detect_repeated_phrases(self):
        """Should detect repeated phrases."""
        detector = DriftDetector()
        content = (
            "The quick brown fox jumped over the lazy dog. "
            "The quick brown fox jumped over the fence. "
            "The quick brown fox jumped over the wall."
        )
        result = detector.check(content)
        assert result.detected is True
        assert "repetition" in result.types

    def test_not_detect_normal_varied_content(self):
        """Should not detect normal varied content."""
        detector = DriftDetector()
        content = (
            "First sentence here. Second different sentence. Third unique sentence."
        )
        result = detector.check(content)
        assert "repetition" not in result.types

    def test_respect_repetition_threshold(self):
        """Should respect repetition threshold."""
        high_threshold = DriftDetector(DriftConfig(repetition_threshold=10))
        content = "This repeats. This repeats. This repeats. This repeats."
        result = high_threshold.check(content)
        assert "repetition" not in result.types

    def test_detect_phrase_repetition_in_longer_text(self):
        """Should detect phrase repetition in longer text."""
        detector = DriftDetector()
        phrase = "we need to consider the implications"
        content = (
            f"First, {phrase}. Second, {phrase}. Third, {phrase}. Finally, {phrase}."
        )
        result = detector.check(content)
        assert result.detected is True
        assert "repetition" in result.types


class TestFormatCollapseDetection:
    """Tests for format collapse detection."""

    def test_detect_here_is_pattern_at_start(self):
        """Should detect 'here is' pattern at start."""
        detector = DriftDetector()
        content = "Here is the response you requested: Some content."
        result = detector.check(content)
        assert result.detected is True
        assert "format_collapse" in result.types

    def test_detect_let_me_pattern_at_start(self):
        """Should detect 'let me' pattern at start."""
        detector = DriftDetector()
        content = "Let me create a solution for you."
        result = detector.check(content)
        assert result.detected is True
        assert "format_collapse" in result.types

    def test_detect_here_you_go_pattern(self):
        """Should detect 'here you go' pattern."""
        detector = DriftDetector()
        content = "Here you go: the answer is 42."
        result = detector.check(content)
        assert result.detected is True
        assert "format_collapse" in result.types

    def test_not_detect_in_middle_of_text(self):
        """Should not detect format collapse in middle of text."""
        detector = DriftDetector()
        content = "Some content. Later I will say here is something."
        result = detector.check(content)
        assert "format_collapse" not in result.types

    def test_only_check_beginning(self):
        """Should only check beginning of content."""
        detector = DriftDetector()
        content = "Normal start. " * 20 + "Here is something."
        result = detector.check(content)
        assert "format_collapse" not in result.types


class TestMarkdownCollapseDetection:
    """Tests for markdown collapse detection."""

    def test_not_detect_with_consistent_markdown(self):
        """Should not detect collapse with consistent markdown."""
        detector = DriftDetector()
        previous = "# Title\n\n```javascript\ncode\n```"
        detector.check(previous)
        current = previous + "\n\n## Another heading\n\n**More bold**"
        result = detector.check(current)
        assert "markdown_collapse" not in result.types

    def test_not_detect_with_no_initial_markdown(self):
        """Should not detect collapse with no initial markdown."""
        detector = DriftDetector()
        previous = "Just plain text here."
        detector.check(previous)
        current = previous + " More plain text."
        result = detector.check(current)
        assert "markdown_collapse" not in result.types

    def test_require_sufficient_previous_content(self):
        """Should require sufficient previous content."""
        detector = DriftDetector()
        detector.check("# H")
        result = detector.check("# H\nPlain")
        assert "markdown_collapse" not in result.types


class TestHedgingDetection:
    """Tests for excessive hedging detection."""

    def test_detect_sure_at_start(self):
        """Should detect 'sure' at start."""
        detector = DriftDetector()
        content = "Sure!\nHere is the content."
        result = detector.check(content)
        assert result.detected is True
        assert "hedging" in result.types

    def test_detect_certainly_at_start(self):
        """Should detect 'certainly' at start."""
        detector = DriftDetector()
        content = "Certainly\nI can help with that."
        result = detector.check(content)
        assert result.detected is True
        assert "hedging" in result.types

    def test_detect_of_course_at_start(self):
        """Should detect 'of course' at start."""
        detector = DriftDetector()
        content = "Of course!\nLet me explain."
        result = detector.check(content)
        assert result.detected is True
        assert "hedging" in result.types

    def test_detect_absolutely_at_start(self):
        """Should detect 'absolutely' at start."""
        detector = DriftDetector()
        content = "Absolutely\nThat's correct."
        result = detector.check(content)
        assert result.detected is True
        assert "hedging" in result.types

    def test_not_detect_in_middle_of_text(self):
        """Should not detect hedging in middle of text."""
        detector = DriftDetector()
        content = "The answer is sure to be correct."
        result = detector.check(content)
        assert "hedging" not in result.types

    def test_only_check_first_line(self):
        """Should only check first line."""
        detector = DriftDetector()
        content = "Normal start\nSure thing on second line"
        result = detector.check(content)
        assert "hedging" not in result.types


class TestEntropySpikeDetection:
    """Tests for entropy spike detection."""

    def test_track_entropy_over_time(self):
        """Should track entropy over time."""
        detector = DriftDetector()
        # Send consistent tokens
        for i in range(15):
            detector.check("normal text content", "text")

        # Send high-entropy token
        result = detector.check("normal text content", "x" * 50)
        assert result is not None

    def test_require_sufficient_history(self):
        """Should require sufficient history."""
        detector = DriftDetector()
        result = detector.check("test", "test")
        assert "entropy_spike" not in result.types

    def test_handle_no_delta_gracefully(self):
        """Should handle no delta gracefully."""
        detector = DriftDetector()
        result = detector.check("test content")
        assert result is not None
        assert result.detected is not None

    def test_maintain_entropy_window(self):
        """Should maintain entropy window."""
        short_window = DriftDetector(DriftConfig(entropy_window=5))
        for i in range(20):
            short_window.check("content", "token")
        history = short_window.get_history()
        assert len(history["entropy"]) <= 5


class TestMultipleDriftTypes:
    """Tests for multiple drift types detection."""

    def test_detect_multiple_types_simultaneously(self):
        """Should detect multiple drift types simultaneously."""
        detector = DriftDetector()
        content = (
            "As an AI, I must say. Here is the answer: "
            "This repeats. This repeats. This repeats."
        )
        result = detector.check(content)
        assert result.detected is True
        assert len(result.types) > 1

    def test_use_highest_confidence(self):
        """Should use highest confidence."""
        detector = DriftDetector()
        content = "As an AI language model, here is the response."
        result = detector.check(content)
        assert result.detected is True
        assert result.confidence > 0.5

    def test_aggregate_details(self):
        """Should aggregate details."""
        detector = DriftDetector()
        content = "As an AI. Here is the answer."
        result = detector.check(content)
        assert result.details is not None
        assert "detected" in result.details.lower()


class TestStateManagement:
    """Tests for state management."""

    def test_reset_state(self):
        """Should reset state."""
        detector = DriftDetector()
        detector.check("Some content")
        detector.check("More content")
        detector.reset()
        history = detector.get_history()
        assert len(history["entropy"]) == 0
        assert len(history["tokens"]) == 0
        assert history["last_content"] == ""

    def test_maintain_history_between_checks(self):
        """Should maintain history between checks."""
        detector = DriftDetector()
        detector.check("First check")
        history1 = detector.get_history()
        assert history1["last_content"] == "First check"

        detector.check("Second check")
        history2 = detector.get_history()
        assert history2["last_content"] == "Second check"

    def test_track_tokens_with_delta(self):
        """Should track tokens with delta."""
        detector = DriftDetector()
        detector.check("content", "token1")
        detector.check("content", "token2")
        history = detector.get_history()
        assert len(history["tokens"]) > 0

    def test_limit_token_history_to_window_size(self):
        """Should limit token history to window size."""
        small_window = DriftDetector(DriftConfig(entropy_window=3))
        for i in range(10):
            small_window.check("content", f"token{i}")
        history = small_window.get_history()
        assert len(history["tokens"]) <= 3


class TestConfigurationOptions:
    """Tests for configuration options."""

    def test_respect_detect_tone_shift_config(self):
        """Should respect detectToneShift config."""
        no_tone = DriftDetector(DriftConfig(detect_tone_shift=False))
        no_tone.check("Therefore, we conclude.")
        result = no_tone.check("Therefore, we conclude. Yeah, cool!")
        assert "tone_shift" not in result.types

    def test_respect_detect_meta_commentary_config(self):
        """Should respect detectMetaCommentary config."""
        no_meta = DriftDetector(DriftConfig(detect_meta_commentary=False))
        result = no_meta.check("As an AI language model, I think...")
        assert "meta_commentary" not in result.types

    def test_respect_detect_repetition_config(self):
        """Should respect detectRepetition config."""
        no_rep = DriftDetector(DriftConfig(detect_repetition=False))
        content = "Repeat. " * 10
        result = no_rep.check(content)
        assert "repetition" not in result.types

    def test_respect_detect_entropy_spike_config(self):
        """Should respect detectEntropySpike config."""
        no_entropy = DriftDetector(DriftConfig(detect_entropy_spike=False))
        for i in range(20):
            no_entropy.check("content", "token")
        result = no_entropy.check("content", "xyz")
        assert "entropy_spike" not in result.types

    def test_use_custom_repetition_threshold(self):
        """Should use custom repetition threshold."""
        high_threshold = DriftDetector(DriftConfig(repetition_threshold=100))
        content = "This repeats. " * 4
        result = high_threshold.check(content)
        assert "repetition" not in result.types

    def test_use_custom_entropy_threshold(self):
        """Should use custom entropy threshold."""
        high_entropy = DriftDetector(DriftConfig(entropy_threshold=10))
        for i in range(20):
            high_entropy.check("content", "token")
        result = high_entropy.check("content", "different")
        assert "entropy_spike" not in result.types


class TestEdgeCases:
    """Tests for edge cases."""

    def test_handle_empty_content(self):
        """Should handle empty content."""
        detector = DriftDetector()
        result = detector.check("")
        assert result is not None
        assert result.detected is not None

    def test_handle_very_short_content(self):
        """Should handle very short content."""
        detector = DriftDetector()
        result = detector.check("Hi")
        assert result is not None

    def test_handle_very_long_content(self):
        """Should handle very long content."""
        detector = DriftDetector()
        long_content = "word " * 10000
        result = detector.check(long_content)
        assert result is not None

    def test_handle_unicode_content(self):
        """Should handle unicode content."""
        detector = DriftDetector()
        content = "你好世界 مرحبا 😀"
        result = detector.check(content)
        assert result is not None

    def test_handle_only_punctuation(self):
        """Should handle content with only punctuation."""
        detector = DriftDetector()
        result = detector.check("...")
        assert result is not None

    def test_handle_special_characters(self):
        """Should handle content with special characters."""
        detector = DriftDetector()
        content = "@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        result = detector.check(content)
        assert result is not None

    def test_handle_none_delta_gracefully(self):
        """Should handle None delta gracefully."""
        detector = DriftDetector()
        result = detector.check("content", None)
        assert result is not None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_drift_detector_with_config(self):
        """Should create detector with config."""
        detector = create_drift_detector(
            DriftConfig(
                detect_tone_shift=False,
                repetition_threshold=5,
            )
        )
        assert detector is not None
        assert isinstance(detector, DriftDetector)

    def test_create_drift_detector_with_no_config(self):
        """Should create detector with no config."""
        detector = create_drift_detector()
        assert detector is not None

    def test_check_drift_without_instance(self):
        """Should check content without instance."""
        result = check_drift("As an AI language model, I think...")
        assert result.detected is True
        assert "meta_commentary" in result.types

    def test_check_drift_with_normal_content(self):
        """Should work with normal content."""
        result = check_drift("This is normal content.")
        assert result.detected is False

    def test_check_drift_multiple_types(self):
        """Should detect multiple drift types."""
        result = check_drift("As an AI. Here is the answer. Repeat. Repeat. Repeat.")
        assert result.detected is True
        assert len(result.types) > 0


class TestIntegrationScenarios:
    """Tests for integration scenarios."""

    def test_streaming_scenario(self):
        """Should handle streaming scenario."""
        detector = DriftDetector()
        content = ""
        chunks = ["The quick ", "brown fox ", "jumps over ", "the lazy dog."]

        for chunk in chunks:
            content += chunk
            result = detector.check(content, chunk)
            assert result is not None

    def test_detect_drift_during_streaming(self):
        """Should detect drift during streaming."""
        detector = DriftDetector()
        content = (
            "This is a professional response. "
            "Therefore, we must consider the implications. "
            "Thus, we proceed. Hence, we analyze. "
        ) * 2
        detector.check(content)

        content += "Yeah, that's gonna be awesome! Wanna try it? Gonna be cool!"
        result = detector.check(content)
        assert "tone_shift" in result.types

    def test_track_entropy_across_stream(self):
        """Should track entropy across stream."""
        detector = DriftDetector()
        for i in range(15):
            detector.check("building content", "word")
        result = detector.check("building content", "word")
        assert result is not None

    def test_handle_reset_mid_stream(self):
        """Should handle reset mid-stream."""
        detector = DriftDetector()
        detector.check("Initial content")
        detector.check("More content")
        detector.reset()
        result = detector.check("Fresh start")
        assert "tone_shift" not in result.types

    def test_detect_progressive_repetition(self):
        """Should detect progressive repetition."""
        detector = DriftDetector()
        content = (
            "This is a meaningful sentence that has enough words to be detected "
            "as substantial content for testing. "
        )
        detector.check(content)

        content += (
            "Another different sentence with plenty of words to be considered "
            "meaningful by the detector. "
        )
        detector.check(content)

        content += (
            "This is a meaningful sentence that has enough words to be detected "
            "as substantial content for testing. "
        ) * 3
        result = detector.check(content)
        assert "repetition" in result.types


class TestPhraseRepetitionSlidingWindow:
    """Tests for phrase repetition sliding window edge cases.

    Note: The _detect_repetition method requires at least 3 sentences (split by .!?)
    with length > 20 chars before it checks for phrase repetition. Tests must include
    proper sentence structure to reach the phrase detection logic.
    """

    def test_detect_repeated_phrase_at_end_of_content(self):
        """Should detect repeated 5-word phrase at the end of content.

        This tests the fix for the off-by-one error where range(len(words) - 5)
        missed the last valid 5-word phrase starting position.
        """
        detector = DriftDetector()
        # Need 3+ sentences with >20 chars each to pass early return check
        # The phrase "one two three four five" repeats 3 times
        content = (
            "This is the first sentence here. "
            "This is the second sentence now. "
            "This is the third sentence too. "
            "one two three four five extra. "
            "some other words here now done. "
            "one two three four five extra. "
            "more different words between us. "
            "one two three four five extra"
        )
        result = detector.check(content)
        assert result.detected is True
        assert "repetition" in result.types

    def test_detect_phrase_starting_at_last_valid_position(self):
        """Should detect phrase that starts at the last valid sliding window position."""
        detector = DriftDetector()
        # The phrase "alpha beta gamma delta epsilon" repeats 3 times
        content = (
            "This is the first sentence here. "
            "This is the second sentence now. "
            "This is the third sentence too. "
            "prefix word here now today. "
            "alpha beta gamma delta epsilon. "
            "middle section now here today. "
            "alpha beta gamma delta epsilon. "
            "alpha beta gamma delta epsilon"
        )
        result = detector.check(content)
        assert result.detected is True
        assert "repetition" in result.types

    def test_exactly_five_words_repeated(self):
        """Should detect repetition when content has exactly 5 words repeated."""
        detector = DriftDetector()
        # Exactly 5 words repeated 3 times (threshold default is 3)
        content = (
            "This is the first sentence here. "
            "This is the second sentence now. "
            "This is the third sentence too. "
            "hello world foo bar baz. "
            "hello world foo bar baz. "
            "hello world foo bar baz"
        )
        result = detector.check(content)
        assert result.detected is True
        assert "repetition" in result.types

    def test_six_words_with_phrase_at_boundary(self):
        """Should detect 5-word phrase when repeated at content boundary."""
        detector = DriftDetector()
        # The phrase "quick brown fox jumps over" repeats 3 times
        content = (
            "This is the first sentence here. "
            "This is the second sentence now. "
            "This is the third sentence too. "
            "quick brown fox jumps over lazy. "
            "dog runs fast now here today. "
            "quick brown fox jumps over fence. "
            "the yard today here now done. "
            "quick brown fox jumps over wall"
        )
        result = detector.check(content)
        assert result.detected is True
        assert "repetition" in result.types

    def test_no_false_positive_with_four_word_near_match(self):
        """Should not detect repetition when only 4 words match at boundary."""
        detector = DriftDetector()
        # Only 4 words match, not 5 - should not trigger phrase repetition
        # Need sentences for the check to proceed
        content = (
            "This is the first sentence here. "
            "This is the second sentence now. "
            "This is the third sentence too. "
            "one two three four DIFFERENT word. "
            "some other content here now done. "
            "one two three four ANOTHER word"
        )
        result = detector.check(content)
        # Should not detect phrase repetition (only 4 words match)
        assert "repetition" not in result.types

    def test_minimum_words_for_phrase_detection(self):
        """Should handle content with exactly 5 words (minimum for phrase detection)."""
        detector = DriftDetector()
        # With exactly 5 words, there's only one possible 5-word phrase
        content = "one two three four five"
        result = detector.check(content)
        # No repetition possible with just one phrase
        assert "repetition" not in result.types

    def test_four_words_no_phrase_detection(self):
        """Should handle content with fewer than 5 words gracefully."""
        detector = DriftDetector()
        content = "one two three four"
        result = detector.check(content)
        # Should not crash, and no repetition detected
        assert result is not None
        assert "repetition" not in result.types


class TestPerformance:
    """Tests for performance."""

    def test_handle_many_checks_efficiently(self):
        """Should handle many checks efficiently."""
        detector = DriftDetector()
        import time

        start = time.time()
        for i in range(100):
            detector.check(f"Content {i}", f"token{i}")
        duration = time.time() - start
        assert duration < 1.0

    def test_handle_large_content_efficiently(self):
        """Should handle large content efficiently."""
        detector = DriftDetector()
        import time

        large_content = "word " * 10000
        start = time.time()
        detector.check(large_content)
        duration = time.time() - start
        assert duration < 0.5


class TestConfidenceScoring:
    """Tests for confidence scoring."""

    def test_return_zero_confidence_for_no_drift(self):
        """Should return 0 confidence for no drift."""
        detector = DriftDetector()
        result = detector.check("Normal content here.")
        assert result.detected is False
        assert result.confidence == 0

    def test_return_high_confidence_for_meta_commentary(self):
        """Should return high confidence for meta commentary."""
        detector = DriftDetector()
        result = detector.check("As an AI language model...")
        assert result.detected is True
        assert result.confidence >= 0.9

    def test_use_maximum_confidence_from_multiple_types(self):
        """Should use maximum confidence from multiple types."""
        detector = DriftDetector()
        result = detector.check("As an AI. Sure! Here is the response.")
        assert result.detected is True
        assert result.confidence > 0
