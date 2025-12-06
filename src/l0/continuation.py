"""Continuation and deduplication for L0 checkpoint resumption.

When a stream fails mid-generation and resumes from a checkpoint,
LLMs often repeat words from the end of the checkpoint. This module
provides deduplication to handle overlap detection and removal.

Example:
    ```python
    from l0.continuation import detect_overlap, deduplicate_continuation

    # Full result with metadata
    result = detect_overlap("Hello world", "world is great")
    # OverlapResult(
    #     has_overlap=True,
    #     overlap_length=5,
    #     overlap_text="world",
    #     deduplicated="is great"
    # )

    # Convenience - just the deduplicated string
    text = deduplicate_continuation("Hello world", "world is great")
    # " is great"
    ```
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class DeduplicationOptions(BaseModel):
    """Configuration for continuation deduplication.

    Attributes:
        min_overlap: Minimum overlap length to detect (avoids false positives)
        max_overlap: Maximum overlap length to check (performance limit)
        case_sensitive: Whether matching is case-sensitive
        normalize_whitespace: Normalize whitespace when matching
    """

    min_overlap: int = Field(default=2, ge=1)
    max_overlap: int = Field(default=500, ge=1)
    case_sensitive: bool = True
    normalize_whitespace: bool = False


@dataclass
class OverlapResult:
    """Result of overlap detection.

    Attributes:
        has_overlap: Whether overlap was detected
        overlap_length: Length of the overlapping portion
        overlap_text: The overlapping text
        deduplicated: The continuation with overlap removed
    """

    has_overlap: bool
    overlap_length: int
    overlap_text: str
    deduplicated: str


def _normalize_text(text: str, options: DeduplicationOptions) -> str:
    """Normalize text according to options."""
    result = text
    if not options.case_sensitive:
        result = result.lower()
    if options.normalize_whitespace:
        # Collapse multiple whitespace to single space
        import re

        result = re.sub(r"\s+", " ", result)
    return result


def detect_overlap(
    checkpoint: str,
    continuation: str,
    options: DeduplicationOptions | None = None,
) -> OverlapResult:
    """Detect overlap between checkpoint suffix and continuation prefix.

    Finds the longest suffix of checkpoint that matches a prefix of continuation.

    Args:
        checkpoint: The checkpoint content (text before failure)
        continuation: The continuation content (text from resumed stream)
        options: Deduplication options

    Returns:
        OverlapResult with overlap details and deduplicated continuation

    Example:
        ```python
        result = detect_overlap("Hello world", "world is great")
        assert result.has_overlap is True
        assert result.overlap_text == "world"
        assert result.deduplicated == " is great"
        ```
    """
    if options is None:
        options = DeduplicationOptions()

    if not checkpoint or not continuation:
        return OverlapResult(
            has_overlap=False,
            overlap_length=0,
            overlap_text="",
            deduplicated=continuation,
        )

    # Normalize for comparison
    norm_checkpoint = _normalize_text(checkpoint, options)
    norm_continuation = _normalize_text(continuation, options)

    # Limit search to max_overlap
    search_len = min(len(checkpoint), len(continuation), options.max_overlap)

    best_overlap = 0
    best_overlap_text = ""

    # Try each possible overlap length, starting from longest
    for overlap_len in range(search_len, options.min_overlap - 1, -1):
        # Get suffix of checkpoint and prefix of continuation
        checkpoint_suffix = norm_checkpoint[-overlap_len:]
        continuation_prefix = norm_continuation[:overlap_len]

        if checkpoint_suffix == continuation_prefix:
            # Found overlap - get the original text (not normalized)
            best_overlap = overlap_len
            best_overlap_text = continuation[:overlap_len]
            break

    if best_overlap >= options.min_overlap:
        return OverlapResult(
            has_overlap=True,
            overlap_length=best_overlap,
            overlap_text=best_overlap_text,
            deduplicated=continuation[best_overlap:],
        )

    return OverlapResult(
        has_overlap=False,
        overlap_length=0,
        overlap_text="",
        deduplicated=continuation,
    )


def deduplicate_continuation(
    checkpoint: str,
    continuation: str,
    options: DeduplicationOptions | None = None,
) -> str:
    """Remove overlapping text from continuation.

    Convenience wrapper around detect_overlap that returns just the
    deduplicated continuation string.

    Args:
        checkpoint: The checkpoint content
        continuation: The continuation content
        options: Deduplication options

    Returns:
        Continuation with any overlapping prefix removed

    Example:
        ```python
        text = deduplicate_continuation("Hello world", "world is great")
        assert text == " is great"
        ```
    """
    result = detect_overlap(checkpoint, continuation, options)
    return result.deduplicated


class ContinuationConfig(BaseModel):
    """Configuration for checkpoint continuation.

    Usage:
        ```python
        import l0
        from l0.continuation import ContinuationConfig

        config = ContinuationConfig(
            enabled=True,
            checkpoint_interval=10,
            deduplicate=True,
        )

        result = await l0.run(
            stream=lambda: client.chat.completions.create(...),
            continuation=config,
        )
        ```

    Attributes:
        enabled: Enable continuation from last known good checkpoint
        checkpoint_interval: Save checkpoint every N tokens (default: 5)
        deduplicate: Enable deduplication of overlapping content (default: True)
        deduplication_options: Options for deduplication
        validate_checkpoint: Run guardrails against checkpoint before using
    """

    enabled: bool = True
    checkpoint_interval: int = Field(default=5, ge=1)
    deduplicate: bool = True
    deduplication_options: DeduplicationOptions = Field(
        default_factory=DeduplicationOptions
    )
    validate_checkpoint: bool = True

    @classmethod
    def default(cls) -> ContinuationConfig:
        """Get default continuation configuration."""
        return cls()

    @classmethod
    def disabled(cls) -> ContinuationConfig:
        """Get disabled continuation configuration."""
        return cls(enabled=False)


# ─────────────────────────────────────────────────────────────────────────────
# Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class Continuation:
    """Scoped API for continuation and deduplication utilities.

    Provides utilities for handling checkpoint resumption and deduplication
    when streams fail mid-generation and resume from a checkpoint.

    Usage:
        ```python
        from l0 import Continuation

        # Detect overlap between checkpoint and continuation
        result = Continuation.detect_overlap("Hello world", "world is great")
        if result.has_overlap:
            print(f"Overlap: {result.overlap_text}")

        # Deduplicate continuation (convenience method)
        text = Continuation.deduplicate("Hello world", "world is great")
        # Returns: " is great"
        ```
    """

    # Re-export types for convenience
    Config = ContinuationConfig
    Options = DeduplicationOptions
    OverlapResult = OverlapResult

    @staticmethod
    def detect_overlap(
        checkpoint: str,
        continuation: str,
        options: DeduplicationOptions | None = None,
    ) -> OverlapResult:
        """Detect overlap between checkpoint suffix and continuation prefix.

        Args:
            checkpoint: The checkpoint content (text before failure)
            continuation: The continuation content (text from resumed stream)
            options: Deduplication options

        Returns:
            OverlapResult with overlap details and deduplicated continuation
        """
        return detect_overlap(checkpoint, continuation, options)

    @staticmethod
    def deduplicate(
        checkpoint: str,
        continuation: str,
        options: DeduplicationOptions | None = None,
    ) -> str:
        """Remove overlapping text from continuation.

        Args:
            checkpoint: The checkpoint content
            continuation: The continuation content
            options: Deduplication options

        Returns:
            Continuation with any overlapping prefix removed
        """
        return deduplicate_continuation(checkpoint, continuation, options)
