"""Document windowing for processing long documents in chunks.

Automatic chunking and navigation for long documents.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from .runtime import _internal_run
from .types import AwaitableStreamFactory, Retry, Stream, Timeout

T = TypeVar("T")

ChunkingStrategy = Literal["token", "char", "paragraph", "sentence"]
TokenEstimator = Callable[[str], int]
ContextRestorationStrategy = Literal["adjacent", "overlap", "full"]


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DocumentChunk:
    """A chunk of a document."""

    index: int  # Position (0-based)
    content: str  # Chunk text
    start_pos: int  # Start position in original document
    end_pos: int  # End position in original document
    token_count: int  # Estimated tokens
    char_count: int  # Character count
    is_first: bool  # Is this the first chunk?
    is_last: bool  # Is this the last chunk?
    total_chunks: int  # Total number of chunks
    metadata: dict[str, Any] | None = None  # Custom metadata


@dataclass
class WindowConfig:
    """Configuration for document windowing."""

    size: int = 2000  # Tokens per chunk
    overlap: int = 200  # Overlap between chunks
    strategy: ChunkingStrategy = "token"
    estimate_tokens: TokenEstimator | None = None  # Custom token estimator
    preserve_paragraphs: bool = True  # Preserve paragraph boundaries
    preserve_sentences: bool = False  # Preserve sentence boundaries
    metadata: dict[str, Any] | None = None  # Custom metadata for all chunks


@dataclass
class ContextRestorationOptions:
    """Options for context restoration on drift detection."""

    enabled: bool = True  # Enable automatic context restoration
    strategy: ContextRestorationStrategy = "adjacent"  # Restoration strategy
    max_attempts: int = 2  # Maximum restoration attempts
    on_restore: Callable[[int, int], None] | None = None  # Callback (from, to)


@dataclass
class WindowStats:
    """Statistics about a document window."""

    total_chunks: int  # Total number of chunks
    total_chars: int  # Total document length (characters)
    total_tokens: int  # Estimated total tokens
    avg_chunk_size: int  # Average chunk size (characters)
    avg_chunk_tokens: int  # Average chunk tokens
    overlap_size: int  # Overlap size (tokens)
    strategy: ChunkingStrategy  # Chunking strategy used


@dataclass
class ChunkProcessConfig:
    """Configuration for processing a chunk."""

    stream: AwaitableStreamFactory
    retry: Retry | None = None
    timeout: Timeout | None = None
    fallbacks: list[AwaitableStreamFactory] | None = None


@dataclass
class ChunkResult(Generic[T]):
    """Result of processing a chunk."""

    chunk: DocumentChunk
    status: Literal["success", "error"]
    result: "Stream[Any] | None" = None
    content: str = ""
    error: str | None = None
    duration: float = 0.0  # Duration in milliseconds


@dataclass
class ProcessingStats:
    """Statistics from processing results."""

    total: int  # Total chunks processed
    successful: int  # Successfully processed
    failed: int  # Failed to process
    success_rate: float  # Success percentage (0-100)
    avg_duration: float  # Average duration in ms
    total_duration: float  # Total duration in ms


# ─────────────────────────────────────────────────────────────────────────────
# Token Estimation
# ─────────────────────────────────────────────────────────────────────────────


def _get_time_ms() -> float:
    """Get current time in milliseconds."""
    return time.time() * 1000


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token for English text.
    For more accurate counts, use tiktoken or similar.
    """
    return len(text) // 4


def estimate_chars_for_tokens(token_count: int) -> int:
    """Estimate character count for a given token count."""
    return token_count * 4


# ─────────────────────────────────────────────────────────────────────────────
# Chunking Functions
# ─────────────────────────────────────────────────────────────────────────────


def _estimate_chars_per_token(document: str, token_estimator: TokenEstimator) -> float:
    """Estimate the chars-per-token ratio using the provided estimator.

    Samples the document to determine the ratio, falling back to default if needed.
    """
    # Sample up to 1000 chars from the document for estimation
    sample = document[:1000] if len(document) > 1000 else document
    if not sample:
        return 4.0  # Default ratio

    token_count = token_estimator(sample)
    if token_count <= 0:
        return 4.0  # Default ratio

    return len(sample) / token_count


def _chunk_by_char(
    document: str,
    size: int,
    overlap: int,
    token_estimator: TokenEstimator,
) -> list[tuple[int, int]]:
    """Chunk document by character count."""
    chunks: list[tuple[int, int]] = []
    pos = 0

    # Use the token estimator to calculate chars-per-token ratio
    chars_per_token = _estimate_chars_per_token(document, token_estimator)
    char_size = int(size * chars_per_token)
    char_overlap = int(overlap * chars_per_token)

    # Ensure overlap is less than size to guarantee forward progress
    if char_overlap >= char_size:
        char_overlap = max(0, char_size - 1)

    while pos < len(document):
        end = min(pos + char_size, len(document))
        chunks.append((pos, end))
        new_pos = end - char_overlap
        # Ensure we always advance by at least 1 character to prevent infinite loop
        if new_pos <= pos:
            new_pos = pos + 1
        pos = new_pos
        if pos >= len(document):
            break
        # Avoid tiny final chunks
        if len(document) - pos < char_overlap:
            break

    return chunks


def _chunk_by_token(
    document: str,
    size: int,
    overlap: int,
    token_estimator: TokenEstimator,
) -> list[tuple[int, int]]:
    """Chunk document by estimated token count."""
    # For token-based chunking, we convert to character positions
    return _chunk_by_char(document, size, overlap, token_estimator)


def _find_paragraph_boundaries(document: str) -> list[int]:
    """Find paragraph boundary positions (double newlines)."""
    boundaries = [0]
    for match in re.finditer(r"\n\s*\n", document):
        boundaries.append(match.end())
    boundaries.append(len(document))
    return boundaries


def _find_sentence_boundaries(document: str) -> list[int]:
    """Find sentence boundary positions."""
    boundaries = [0]
    # Match sentence endings: . ! ? followed by space or end
    for match in re.finditer(r"[.!?]+\s+", document):
        boundaries.append(match.end())
    boundaries.append(len(document))
    return boundaries


def _chunk_by_boundaries(
    document: str,
    boundaries: list[int],
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Chunk document respecting boundaries (paragraphs or sentences)."""
    chunks: list[tuple[int, int]] = []
    char_size = estimate_chars_for_tokens(size)
    char_overlap = estimate_chars_for_tokens(overlap)

    # Ensure overlap is less than size to guarantee forward progress
    if char_overlap >= char_size:
        char_overlap = max(0, char_size - 1)

    i = 0
    while i < len(boundaries) - 1:
        start_pos = boundaries[i]
        end_pos = start_pos

        # Accumulate boundaries until we reach size
        j = i + 1
        while j < len(boundaries):
            next_pos = boundaries[j]
            if next_pos - start_pos > char_size and end_pos > start_pos:
                break
            end_pos = next_pos
            j += 1

        if end_pos > start_pos:
            chunks.append((start_pos, end_pos))

        # If we've reached the end of the document, stop
        if end_pos >= len(document):
            break

        # Find overlap start position
        overlap_start = max(start_pos, end_pos - char_overlap)
        # Find the boundary closest to overlap_start
        new_i = i
        for k in range(i, len(boundaries)):
            if boundaries[k] >= overlap_start:
                new_i = k
                break

        # Ensure progress
        if new_i <= i:
            new_i = i + 1
        i = new_i

        if i >= len(boundaries) - 1:
            break

    return chunks


def _chunk_by_paragraph(
    document: str,
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Chunk document by paragraphs."""
    boundaries = _find_paragraph_boundaries(document)
    return _chunk_by_boundaries(document, boundaries, size, overlap)


def _chunk_by_sentence(
    document: str,
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Chunk document by sentences."""
    boundaries = _find_sentence_boundaries(document)
    return _chunk_by_boundaries(document, boundaries, size, overlap)


def chunk_document(
    document: str,
    config: WindowConfig,
) -> list[DocumentChunk]:
    """Chunk a document according to configuration."""
    token_estimator = config.estimate_tokens or estimate_tokens

    if config.strategy == "char":
        positions = _chunk_by_char(
            document, config.size, config.overlap, token_estimator
        )
    elif config.strategy == "token":
        positions = _chunk_by_token(
            document, config.size, config.overlap, token_estimator
        )
    elif config.strategy == "paragraph":
        positions = _chunk_by_paragraph(document, config.size, config.overlap)
    elif config.strategy == "sentence":
        positions = _chunk_by_sentence(document, config.size, config.overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")

    total = len(positions)
    chunks: list[DocumentChunk] = []

    for i, (start, end) in enumerate(positions):
        content = document[start:end]
        chunks.append(
            DocumentChunk(
                index=i,
                content=content,
                start_pos=start,
                end_pos=end,
                token_count=token_estimator(content),
                char_count=len(content),
                is_first=(i == 0),
                is_last=(i == total - 1),
                total_chunks=total,
                metadata=config.metadata.copy() if config.metadata else None,
            )
        )

    return chunks


def merge_chunks(
    chunks: list[DocumentChunk],
    preserve_overlap: bool = False,
) -> str:
    """Merge chunks back into a single string.

    Args:
        chunks: List of document chunks to merge
        preserve_overlap: If True, preserves overlap; if False, removes it

    Returns:
        Merged content string
    """
    if not chunks:
        return ""

    if len(chunks) == 1:
        return chunks[0].content

    if preserve_overlap:
        return "".join(chunk.content for chunk in chunks)

    # Remove overlap by using position info
    result_parts: list[str] = []
    last_end = 0

    for chunk in sorted(chunks, key=lambda c: c.start_pos):
        if chunk.start_pos >= last_end:
            # No overlap with previous
            result_parts.append(chunk.content)
        else:
            # Overlap - take only the new part
            overlap_chars = last_end - chunk.start_pos
            if overlap_chars < len(chunk.content):
                result_parts.append(chunk.content[overlap_chars:])
        last_end = chunk.end_pos

    return "".join(result_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Document Window Class
# ─────────────────────────────────────────────────────────────────────────────


class DocumentWindow:
    """Window for navigating and processing document chunks.

    Example:
        ```python
        from l0 import Window

        window = Window.create(long_document, size=2000, overlap=200)

        # Navigate
        chunk = window.current()
        window.next()
        window.prev()
        window.jump(5)

        # Search
        matches = window.find_chunks("keyword")

        # Stats
        stats = window.get_stats()

        # Process all chunks
        results = await window.process_all(
            lambda chunk: ChunkProcessConfig(
                stream=lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": chunk.content}],
                    stream=True,
                )
            )
        )
        ```
    """

    def __init__(self, document: str, config: WindowConfig):
        """Create a document window.

        Args:
            document: The document text to chunk
            config: Window configuration
        """
        self._document = document
        self._config = config
        self._token_estimator = config.estimate_tokens or estimate_tokens
        self._chunks = chunk_document(document, config)
        self._current_index = 0

    @property
    def total_chunks(self) -> int:
        """Total number of chunks."""
        return len(self._chunks)

    @property
    def current_index(self) -> int:
        """Current chunk index."""
        return self._current_index

    @property
    def document(self) -> str:
        """Original document."""
        return self._document

    @property
    def config(self) -> WindowConfig:
        """Window configuration."""
        return self._config

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────────────────────────────────

    def current(self) -> DocumentChunk | None:
        """Get current chunk."""
        if 0 <= self._current_index < len(self._chunks):
            return self._chunks[self._current_index]
        return None

    def get(self, index: int) -> DocumentChunk | None:
        """Get chunk at specific index."""
        if 0 <= index < len(self._chunks):
            return self._chunks[index]
        return None

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Get all chunks."""
        return list(self._chunks)

    def get_range(self, start: int, end: int) -> list[DocumentChunk]:
        """Get a range of chunks.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            List of chunks in the range
        """
        valid_start = max(0, start)
        valid_end = min(len(self._chunks), end)
        return self._chunks[valid_start:valid_end]

    def next(self) -> DocumentChunk | None:
        """Move to and return next chunk."""
        if self.has_next():
            self._current_index += 1
            return self.current()
        return None

    def prev(self) -> DocumentChunk | None:
        """Move to and return previous chunk."""
        if self.has_prev():
            self._current_index -= 1
            return self.current()
        return None

    def jump(self, index: int) -> DocumentChunk | None:
        """Jump to specific chunk index."""
        if 0 <= index < len(self._chunks):
            self._current_index = index
            return self.current()
        return None

    def reset(self) -> DocumentChunk | None:
        """Reset to first chunk."""
        self._current_index = 0
        return self.current()

    def has_next(self) -> bool:
        """Check if there's a next chunk."""
        return self._current_index < len(self._chunks) - 1

    def has_prev(self) -> bool:
        """Check if there's a previous chunk."""
        return self._current_index > 0

    # ─────────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────────

    def find_chunks(
        self,
        search_text: str,
        case_sensitive: bool = False,
    ) -> list[DocumentChunk]:
        """Find chunks containing specific text.

        Args:
            search_text: Text to search for
            case_sensitive: Whether search is case-sensitive (default False)

        Returns:
            List of chunks containing the search text
        """
        search = search_text if case_sensitive else search_text.lower()

        return [
            chunk
            for chunk in self._chunks
            if search in (chunk.content if case_sensitive else chunk.content.lower())
        ]

    def get_context(
        self,
        index: int,
        *,
        before: int = 0,
        after: int = 0,
    ) -> str:
        """Get surrounding context for a chunk.

        Merges the content of adjacent chunks to provide context.

        Args:
            index: Index of the center chunk
            before: Number of chunks before to include
            after: Number of chunks after to include

        Returns:
            Merged content of the context chunks
        """
        start = max(0, index - before)
        end = min(len(self._chunks), index + after + 1)

        context_chunks = self._chunks[start:end]
        return merge_chunks(context_chunks)

    def get_chunks_in_range(
        self,
        start_pos: int,
        end_pos: int,
    ) -> list[DocumentChunk]:
        """Get chunks that overlap with a character position range.

        Args:
            start_pos: Start position in original document (characters)
            end_pos: End position in original document (characters)

        Returns:
            List of chunks that overlap with the specified range
        """
        return [
            chunk
            for chunk in self._chunks
            if (
                # Chunk starts within range
                (chunk.start_pos >= start_pos and chunk.start_pos < end_pos)
                # Chunk ends within range
                or (chunk.end_pos > start_pos and chunk.end_pos <= end_pos)
                # Chunk fully contains range
                or (chunk.start_pos <= start_pos and chunk.end_pos >= end_pos)
            )
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> WindowStats:
        """Get statistics about this window.

        Returns:
            WindowStats with chunk and document statistics
        """
        total_chars = len(self._document)
        total_tokens = self._token_estimator(self._document)

        if self._chunks:
            avg_chunk_size = sum(c.char_count for c in self._chunks) // len(
                self._chunks
            )
            avg_chunk_tokens = sum(c.token_count for c in self._chunks) // len(
                self._chunks
            )
        else:
            avg_chunk_size = 0
            avg_chunk_tokens = 0

        return WindowStats(
            total_chunks=len(self._chunks),
            total_chars=total_chars,
            total_tokens=total_tokens,
            avg_chunk_size=avg_chunk_size,
            avg_chunk_tokens=avg_chunk_tokens,
            overlap_size=self._config.overlap,
            strategy=self._config.strategy,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Processing
    # ─────────────────────────────────────────────────────────────────────────

    async def process_all(
        self,
        processor: Callable[[DocumentChunk], ChunkProcessConfig],
        *,
        concurrency: int = 5,
    ) -> list[ChunkResult[Any]]:
        """Process all chunks in parallel.

        Args:
            processor: Function that takes a chunk and returns processing config
            concurrency: Maximum concurrent processing (default 5)

        Returns:
            List of ChunkResult for each chunk

        Raises:
            ValueError: If concurrency is less than 1
        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        semaphore = asyncio.Semaphore(concurrency)

        async def process_chunk(chunk: DocumentChunk) -> ChunkResult[Any]:
            async with semaphore:
                start_time = _get_time_ms()
                try:
                    config = processor(chunk)
                    result = await _internal_run(
                        stream=config.stream,
                        fallbacks=config.fallbacks,
                        retry=config.retry,
                        timeout=config.timeout,
                    )
                    content = await result.read()
                    return ChunkResult(
                        chunk=chunk,
                        status="success",
                        result=result,
                        content=content,
                        duration=_get_time_ms() - start_time,
                    )
                except Exception as e:
                    return ChunkResult(
                        chunk=chunk,
                        status="error",
                        error=str(e),
                        duration=_get_time_ms() - start_time,
                    )

        tasks = [process_chunk(chunk) for chunk in self._chunks]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def process_sequential(
        self,
        processor: Callable[[DocumentChunk], ChunkProcessConfig],
    ) -> list[ChunkResult[Any]]:
        """Process all chunks sequentially.

        Args:
            processor: Function that takes a chunk and returns processing config

        Returns:
            List of ChunkResult for each chunk
        """
        results: list[ChunkResult[Any]] = []

        for chunk in self._chunks:
            start_time = _get_time_ms()
            try:
                config = processor(chunk)
                result = await _internal_run(
                    stream=config.stream,
                    fallbacks=config.fallbacks,
                    retry=config.retry,
                    timeout=config.timeout,
                )
                content = await result.read()
                results.append(
                    ChunkResult(
                        chunk=chunk,
                        status="success",
                        result=result,
                        content=content,
                        duration=_get_time_ms() - start_time,
                    )
                )
            except Exception as e:
                results.append(
                    ChunkResult(
                        chunk=chunk,
                        status="error",
                        error=str(e),
                        duration=_get_time_ms() - start_time,
                    )
                )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Python Special Methods
    # ─────────────────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[DocumentChunk]:
        """Iterate over all chunks."""
        return iter(self._chunks)

    def __len__(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

    def __getitem__(self, index: int) -> DocumentChunk:
        """Get chunk by index."""
        return self._chunks[index]


# ─────────────────────────────────────────────────────────────────────────────
# Window Class - Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class Window:
    """Scoped API for document windowing operations.

    Usage:
        from l0 import Window

        # Create a window
        window = Window.create(document, size=2000, overlap=200)

        # Use presets
        window = Window.small(document)    # 1000 tokens
        window = Window.medium(document)   # 2000 tokens (default)
        window = Window.large(document)    # 4000 tokens
        window = Window.paragraph(document) # Paragraph-based
        window = Window.sentence(document)  # Sentence-based

        # Chunk a document
        chunks = Window.chunk(document, config)

        # Estimate tokens
        count = Window.estimate_tokens(text)
    """

    @staticmethod
    def create(
        document: str,
        config: WindowConfig | None = None,
        *,
        size: int = 2000,
        overlap: int = 200,
        strategy: ChunkingStrategy = "token",
        estimate_tokens: TokenEstimator | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentWindow:
        """Create a document window for chunking and processing.

        Args:
            document: The document text to chunk
            config: Optional WindowConfig (overrides other args if provided)
            size: Tokens per chunk (default 2000)
            overlap: Overlap between chunks (default 200)
            strategy: Chunking strategy (default "token")
            estimate_tokens: Custom token estimator function
            metadata: Custom metadata to attach to each chunk

        Returns:
            DocumentWindow for navigating and processing chunks
        """
        if config is None:
            config = WindowConfig(
                size=size,
                overlap=overlap,
                strategy=strategy,
                estimate_tokens=estimate_tokens,
                metadata=metadata,
            )
        return DocumentWindow(document, config)

    @staticmethod
    def small(document: str) -> DocumentWindow:
        """Create a small window (1000 tokens, 100 overlap)."""
        return DocumentWindow(
            document, WindowConfig(size=1000, overlap=100, strategy="token")
        )

    @staticmethod
    def medium(document: str) -> DocumentWindow:
        """Create a medium window (2000 tokens, 200 overlap)."""
        return DocumentWindow(
            document, WindowConfig(size=2000, overlap=200, strategy="token")
        )

    @staticmethod
    def large(document: str) -> DocumentWindow:
        """Create a large window (4000 tokens, 400 overlap)."""
        return DocumentWindow(
            document, WindowConfig(size=4000, overlap=400, strategy="token")
        )

    @staticmethod
    def paragraph(document: str) -> DocumentWindow:
        """Create a paragraph-based window (2000 tokens, 200 overlap)."""
        return DocumentWindow(
            document, WindowConfig(size=2000, overlap=200, strategy="paragraph")
        )

    @staticmethod
    def sentence(document: str) -> DocumentWindow:
        """Create a sentence-based window (1500 tokens, 150 overlap)."""
        return DocumentWindow(
            document, WindowConfig(size=1500, overlap=150, strategy="sentence")
        )

    @staticmethod
    def chunk(document: str, config: WindowConfig) -> list[DocumentChunk]:
        """Chunk a document according to configuration."""
        return chunk_document(document, config)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token for English text.
        """
        return estimate_tokens(text)

    @staticmethod
    async def process(
        document: str,
        processor: "Callable[[DocumentChunk], ChunkProcessConfig]",
        config: WindowConfig | None = None,
        *,
        size: int = 2000,
        overlap: int = 200,
        strategy: "ChunkingStrategy" = "token",
        concurrency: int = 3,
    ) -> "list[ChunkResult[Any]]":
        """Process a document through windowed chunks.

        Args:
            document: Document to process
            processor: Function that takes a chunk and returns processing config
            config: Optional WindowConfig
            size: Tokens per chunk
            overlap: Overlap between chunks
            strategy: Chunking strategy
            concurrency: Max concurrent chunk processing

        Returns:
            List of ChunkResult objects
        """
        return await process_with_window(
            document,
            processor,
            config,
            size=size,
            overlap=overlap,
            strategy=strategy,
            concurrency=concurrency,
        )

    @staticmethod
    def merge_results(
        results: "list[ChunkResult[str]]",
        separator: str = "\n\n",
    ) -> str:
        """Merge string results from multiple chunks.

        Args:
            results: List of ChunkResult with string data
            separator: Separator between chunks (default "\\n\\n")

        Returns:
            Merged string with overlap handling
        """
        return merge_results(results, separator=separator)

    @staticmethod
    def merge_chunks(
        chunks: list[DocumentChunk],
        preserve_overlap: bool = False,
    ) -> str:
        """Merge document chunks back into a single document.

        Args:
            chunks: List of DocumentChunk objects
            preserve_overlap: If True, preserves overlap; if False, removes it

        Returns:
            Merged document text with overlap handling
        """
        return merge_chunks(chunks, preserve_overlap=preserve_overlap)

    @staticmethod
    def get_stats(results: "list[ChunkResult[Any]]") -> ProcessingStats:
        """Get processing statistics from chunk results.

        Args:
            results: List of ChunkResult objects

        Returns:
            ProcessingStats with aggregated metrics
        """
        return get_processing_stats(results)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


async def process_with_window(
    document: str,
    processor: Callable[[DocumentChunk], ChunkProcessConfig],
    config: WindowConfig | None = None,
    *,
    size: int = 2000,
    overlap: int = 200,
    strategy: ChunkingStrategy = "token",
    concurrency: int = 5,
) -> list[ChunkResult[Any]]:
    """Process a document with L0 using a sliding window.

    Convenience function that creates a window and processes all chunks.

    Args:
        document: Document to process
        processor: Function to process each chunk
        config: Optional WindowConfig (overrides other args if provided)
        size: Tokens per chunk (default 2000)
        overlap: Overlap between chunks (default 200)
        strategy: Chunking strategy (default "token")
        concurrency: Maximum concurrent processing (default 5)

    Returns:
        List of ChunkResult for each chunk

    Example:
        ```python
        results = await process_with_window(
            document,
            lambda chunk: ChunkProcessConfig(
                stream=lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"Summarize: {chunk.content}"}],
                    stream=True,
                )
            ),
            size=2000,
            overlap=200,
        )
        ```
    """
    if config is None:
        config = WindowConfig(
            size=size,
            overlap=overlap,
            strategy=strategy,
        )
    window = DocumentWindow(document, config)
    return await window.process_all(processor, concurrency=concurrency)


def merge_results(
    results: list[ChunkResult[Any]],
    separator: str = "\n\n",
) -> str:
    """Merge results from multiple chunk processing into a single text.

    Args:
        results: List of ChunkResult from processing
        separator: Separator between chunks (default "\\n\\n")

    Returns:
        Merged text from all successful results
    """
    return separator.join(
        r.content for r in results if r.status == "success" and r.content
    )


def get_processing_stats(results: list[ChunkResult[Any]]) -> ProcessingStats:
    """Get processing statistics from results.

    Args:
        results: List of ChunkResult from processing

    Returns:
        ProcessingStats with success/failure counts and timing info
    """
    total = len(results)
    successful = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "error")
    success_rate = (successful / total * 100) if total > 0 else 0.0
    total_duration = sum(r.duration for r in results)
    avg_duration = (total_duration / total) if total > 0 else 0.0

    return ProcessingStats(
        total=total,
        successful=successful,
        failed=failed,
        success_rate=round(success_rate, 2),
        avg_duration=round(avg_duration, 2),
        total_duration=round(total_duration, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context Restoration
# ─────────────────────────────────────────────────────────────────────────────


async def l0_with_window(
    window: DocumentWindow,
    chunk_index: int,
    stream: AwaitableStreamFactory,
    *,
    context_restoration: ContextRestorationOptions | None = None,
    retry: Retry | None = None,
    timeout: Timeout | None = None,
    fallbacks: list[AwaitableStreamFactory] | None = None,
) -> ChunkResult[Any]:
    """Process a chunk with context restoration on drift detection.

    If drift is detected during processing, automatically retries with
    adjacent chunks to restore context.

    Args:
        window: Document window containing chunks
        chunk_index: Index of chunk to process
        stream: Stream factory for processing
        context_restoration: Options for context restoration
        retry: Retry configuration
        timeout: Timeout configuration
        fallbacks: Fallback stream factories

    Returns:
        ChunkResult from processing

    Example:
        ```python
        window = Window.create(document, size=2000)
        result = await l0_with_window(
            window,
            chunk_index=0,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": window.get(0).content}],
                stream=True,
            ),
            context_restoration=ContextRestorationOptions(
                enabled=True,
                strategy="adjacent",
                max_attempts=2,
            ),
        )
        ```
    """
    chunk = window.get(chunk_index)
    if chunk is None:
        raise ValueError(f"Invalid chunk index: {chunk_index}")

    restoration = context_restoration or ContextRestorationOptions()
    current_chunk_index = chunk_index
    attempts = 0

    while attempts <= restoration.max_attempts:
        current_chunk = window.get(current_chunk_index)
        if current_chunk is None:
            break

        start_time = _get_time_ms()
        try:
            result = await _internal_run(
                stream=stream,
                fallbacks=fallbacks,
                retry=retry,
                timeout=timeout,
            )
            content = await result.read()

            # Check for drift (if result has drift detection)
            drift_detected = getattr(result, "drift_detected", False)

            if (
                drift_detected
                and restoration.enabled
                and attempts < restoration.max_attempts
            ):
                # Try context restoration
                next_chunk_index = _get_restoration_chunk(
                    window, current_chunk_index, restoration.strategy
                )

                if next_chunk_index is not None:
                    if restoration.on_restore:
                        restoration.on_restore(current_chunk_index, next_chunk_index)
                    current_chunk_index = next_chunk_index
                    attempts += 1
                    continue

            return ChunkResult(
                chunk=current_chunk,
                status="success",
                result=result,
                content=content,
                duration=_get_time_ms() - start_time,
            )

        except Exception as e:
            if attempts >= restoration.max_attempts:
                return ChunkResult(
                    chunk=current_chunk,
                    status="error",
                    error=str(e),
                    duration=_get_time_ms() - start_time,
                )
            attempts += 1

    # Should not reach here, but return error if we do
    return ChunkResult(
        chunk=chunk,
        status="error",
        error="Context restoration failed after max attempts",
        duration=0.0,
    )


def _get_restoration_chunk(
    window: DocumentWindow,
    current_index: int,
    strategy: ContextRestorationStrategy,
) -> int | None:
    """Get the next chunk index for context restoration.

    Args:
        window: Document window
        current_index: Current chunk index
        strategy: Restoration strategy

    Returns:
        Next chunk index to try, or None if no more options
    """
    if strategy == "adjacent":
        # Try next chunk first, then previous
        if current_index < window.total_chunks - 1:
            return current_index + 1
        elif current_index > 0:
            return current_index - 1
    elif strategy == "overlap":
        # Try chunk with more overlap (next chunk)
        if current_index < window.total_chunks - 1:
            return current_index + 1
    elif strategy == "full":
        # Try adjacent chunks for full context
        if current_index < window.total_chunks - 1:
            return current_index + 1
        elif current_index > 0:
            return current_index - 1

    return None
