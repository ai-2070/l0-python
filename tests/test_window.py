"""Tests for l0.window module."""

import pytest

from l0 import (
    ChunkResult,
    ContextRestorationOptions,
    DocumentChunk,
    DocumentWindow,
    ProcessingStats,
    Window,
    WindowConfig,
    WindowStats,
)


class TestEstimateTokens:
    def test_estimate_tokens_empty(self):
        assert Window.estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        # "hello" = 5 chars, ~1 token
        tokens = Window.estimate_tokens("hello")
        assert tokens == 1

    def test_estimate_tokens_longer(self):
        # 100 chars should be ~25 tokens
        text = "a" * 100
        tokens = Window.estimate_tokens(text)
        assert tokens == 25


class TestWindowChunk:
    def test_chunk_empty_document(self):
        chunks = Window.chunk("", WindowConfig(size=100))
        assert len(chunks) == 0

    def test_chunk_small_document(self):
        doc = "Hello world"
        chunks = Window.chunk(doc, WindowConfig(size=100))
        assert len(chunks) == 1
        assert chunks[0].content == doc
        assert chunks[0].is_first
        assert chunks[0].is_last

    def test_chunk_by_char(self):
        doc = "a" * 1000
        # 100 tokens = 400 chars, 20 overlap = 80 chars
        chunks = Window.chunk(doc, WindowConfig(size=100, overlap=20, strategy="char"))
        assert len(chunks) > 1
        # Check overlap
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1].end_pos
            curr_start = chunks[i].start_pos
            assert curr_start < prev_end  # Overlap exists

    def test_chunk_by_token(self):
        doc = "word " * 500  # ~500 words
        chunks = Window.chunk(doc, WindowConfig(size=100, overlap=10, strategy="token"))
        assert len(chunks) > 1

    def test_chunk_by_paragraph(self):
        doc = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = Window.chunk(
            doc, WindowConfig(size=1000, overlap=100, strategy="paragraph")
        )
        # With large size, should be 1 chunk
        assert len(chunks) >= 1
        assert "Paragraph one" in chunks[0].content

    def test_chunk_by_sentence(self):
        doc = "First sentence. Second sentence. Third sentence."
        chunks = Window.chunk(
            doc, WindowConfig(size=1000, overlap=100, strategy="sentence")
        )
        assert len(chunks) >= 1

    def test_chunk_metadata(self):
        doc = "Hello world. This is a test."
        chunks = Window.chunk(doc, WindowConfig(size=1000))
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.index == 0
        assert chunk.start_pos == 0
        assert chunk.end_pos == len(doc)
        assert chunk.char_count == len(doc)
        assert chunk.token_count > 0
        assert chunk.is_first
        assert chunk.is_last
        assert chunk.total_chunks == 1


class TestDocumentWindow:
    @pytest.fixture
    def sample_doc(self) -> str:
        return "a" * 4000  # Large enough for multiple chunks

    @pytest.fixture
    def window(self, sample_doc: str) -> DocumentWindow:
        return Window.create(sample_doc, size=500, overlap=50)

    def test_create_window(self, window: DocumentWindow) -> None:
        assert window.total_chunks > 1

    def test_current(self, window: DocumentWindow) -> None:
        chunk = window.current()
        assert chunk is not None
        assert chunk.index == 0
        assert chunk.is_first

    def test_get(self, window: DocumentWindow) -> None:
        chunk = window.get(0)
        assert chunk is not None
        assert chunk.index == 0

        last_chunk = window.get(window.total_chunks - 1)
        assert last_chunk is not None
        assert last_chunk.is_last

        # Out of bounds
        assert window.get(-1) is None
        assert window.get(999) is None

    def test_get_all_chunks(self, window: DocumentWindow) -> None:
        chunks = window.get_all_chunks()
        assert len(chunks) == window.total_chunks

    def test_navigation_next(self, window: DocumentWindow) -> None:
        assert window.current_index == 0
        chunk = window.next()
        assert chunk is not None
        assert window.current_index == 1

    def test_navigation_prev(self, window: DocumentWindow) -> None:
        window.next()
        assert window.current_index == 1
        chunk = window.prev()
        assert chunk is not None
        assert window.current_index == 0

    def test_navigation_jump(self, window: DocumentWindow) -> None:
        chunk = window.jump(2)
        assert chunk is not None
        assert window.current_index == 2

    def test_navigation_reset(self, window: DocumentWindow) -> None:
        window.jump(2)
        chunk = window.reset()
        assert chunk is not None
        assert window.current_index == 0

    def test_has_next(self, window: DocumentWindow) -> None:
        assert window.has_next()
        window.jump(window.total_chunks - 1)
        assert not window.has_next()

    def test_has_prev(self, window: DocumentWindow) -> None:
        assert not window.has_prev()
        window.next()
        assert window.has_prev()

    def test_iteration(self, window: DocumentWindow) -> None:
        chunks = list(window)
        assert len(chunks) == window.total_chunks

    def test_len(self, window: DocumentWindow) -> None:
        assert len(window) == window.total_chunks

    def test_getitem(self, window: DocumentWindow) -> None:
        chunk = window[0]
        assert chunk.index == 0


class TestWindowCreate:
    def test_create_with_kwargs(self):
        doc = "Test document"
        window = Window.create(doc, size=100, overlap=10, strategy="token")
        assert window.config.size == 100
        assert window.config.overlap == 10
        assert window.config.strategy == "token"

    def test_create_with_config(self):
        doc = "Test document"
        config = WindowConfig(size=500, overlap=50, strategy="paragraph")
        window = Window.create(doc, config=config)
        assert window.config.size == 500
        assert window.config.overlap == 50
        assert window.config.strategy == "paragraph"


class TestWindowPresets:
    def test_small(self):
        doc = "Test document"
        window = Window.small(doc)
        assert window.config.size == 1000
        assert window.config.overlap == 100
        assert window.config.strategy == "token"

    def test_medium(self):
        doc = "Test document"
        window = Window.medium(doc)
        assert window.config.size == 2000
        assert window.config.overlap == 200
        assert window.config.strategy == "token"

    def test_large(self):
        doc = "Test document"
        window = Window.large(doc)
        assert window.config.size == 4000
        assert window.config.overlap == 400
        assert window.config.strategy == "token"

    def test_paragraph(self):
        doc = "Test document"
        window = Window.paragraph(doc)
        assert window.config.strategy == "paragraph"

    def test_sentence(self):
        doc = "Test document"
        window = Window.sentence(doc)
        assert window.config.strategy == "sentence"


class TestParagraphChunking:
    def test_respects_paragraph_boundaries(self):
        doc = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content."""

        window = Window.create(doc, size=1000, overlap=100, strategy="paragraph")
        chunks = window.get_all_chunks()

        # With large size, should fit in one chunk
        assert len(chunks) >= 1
        # Content should be preserved
        assert "First paragraph" in chunks[0].content

    def test_multiple_paragraph_chunks(self):
        # Create document with many paragraphs
        paragraphs = [
            f"Paragraph {i} with some longer content to make it bigger."
            for i in range(20)
        ]
        doc = "\n\n".join(paragraphs)

        window = Window.create(doc, size=100, overlap=10, strategy="paragraph")
        chunks = window.get_all_chunks()

        # Should create multiple chunks
        assert len(chunks) >= 1


class TestSentenceChunking:
    def test_respects_sentence_boundaries(self):
        doc = "First sentence. Second sentence. Third sentence."
        window = Window.create(doc, size=1000, overlap=100, strategy="sentence")
        chunks = window.get_all_chunks()

        # All should fit in one chunk
        assert len(chunks) == 1
        assert "First sentence" in chunks[0].content

    def test_multiple_sentence_chunks(self):
        # Create document with many sentences
        sentences = [f"This is sentence number {i}." for i in range(50)]
        doc = " ".join(sentences)

        window = Window.create(doc, size=100, overlap=10, strategy="sentence")
        chunks = window.get_all_chunks()

        # Should create multiple chunks
        assert len(chunks) >= 1


class TestOverlap:
    def test_chunks_have_overlap(self):
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        chunks = window.get_all_chunks()

        if len(chunks) > 1:
            # Check that chunks overlap
            for i in range(1, len(chunks)):
                prev_end = chunks[i - 1].end_pos
                curr_start = chunks[i].start_pos
                # Overlap means current starts before previous ends
                assert curr_start < prev_end, (
                    f"Chunk {i} should overlap with chunk {i - 1}"
                )


class TestProcessAll:
    """Tests for process_all method."""

    @pytest.mark.asyncio
    async def test_process_all_zero_concurrency_raises(self):
        """Test that concurrency=0 raises ValueError."""
        doc = "Test document"
        window = Window.create(doc, size=100)

        async def processor(chunk: DocumentChunk) -> str:  # type: ignore[misc]
            return chunk.content

        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await window.process_all(processor, concurrency=0)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_process_all_negative_concurrency_raises(self):
        """Test that negative concurrency raises ValueError."""
        doc = "Test document"
        window = Window.create(doc, size=100)

        async def processor(chunk: DocumentChunk) -> str:  # type: ignore[misc]
            return chunk.content

        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await window.process_all(processor, concurrency=-1)  # type: ignore[arg-type]


class TestOverlapGreaterThanSize:
    """Test that overlap >= size doesn't cause infinite loop."""

    def test_overlap_equal_to_size_no_hang(self):
        """Test that overlap == size doesn't cause infinite loop."""
        doc = "This is a test document with some content."
        # overlap == size would cause infinite loop without fix
        window = Window.create(doc, size=100, overlap=100, strategy="char")
        chunks = window.get_all_chunks()
        # Should complete without hanging and produce at least one chunk
        assert len(chunks) >= 1

    def test_overlap_greater_than_size_no_hang(self):
        """Test that overlap > size doesn't cause infinite loop."""
        doc = "This is a test document with some content."
        # overlap > size would cause infinite loop without fix
        window = Window.create(doc, size=50, overlap=100, strategy="char")
        chunks = window.get_all_chunks()
        # Should complete without hanging and produce at least one chunk
        assert len(chunks) >= 1

    def test_sentence_overlap_greater_than_size_no_hang(self):
        """Test sentence chunking with overlap >= size."""
        doc = "First sentence. Second sentence. Third sentence."
        window = Window.create(doc, size=10, overlap=20, strategy="sentence")
        chunks = window.get_all_chunks()
        assert len(chunks) >= 1


class TestGetRange:
    """Tests for get_range method."""

    def test_get_range_basic(self):
        """Test basic range retrieval."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        chunks = window.get_range(0, 2)
        assert len(chunks) == 2
        assert chunks[0].index == 0
        assert chunks[1].index == 1

    def test_get_range_middle(self):
        """Test range in the middle."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        chunks = window.get_range(1, 3)
        assert len(chunks) == 2
        assert chunks[0].index == 1
        assert chunks[1].index == 2

    def test_get_range_clamps_start(self):
        """Test that negative start is clamped to 0."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        chunks = window.get_range(-5, 2)
        assert len(chunks) == 2
        assert chunks[0].index == 0

    def test_get_range_clamps_end(self):
        """Test that end beyond total_chunks is clamped."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        total = window.total_chunks
        chunks = window.get_range(total - 1, total + 10)
        assert len(chunks) == 1
        assert chunks[0].index == total - 1

    def test_get_range_empty(self):
        """Test empty range when start >= end."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        chunks = window.get_range(5, 2)
        assert len(chunks) == 0


class TestFindChunks:
    """Tests for find_chunks method."""

    def test_find_chunks_basic(self):
        """Test basic text search."""
        doc = "Hello world. Goodbye world. Hello again."
        window = Window.create(doc, size=1000)
        matches = window.find_chunks("Hello")
        assert len(matches) == 1
        assert "Hello" in matches[0].content

    def test_find_chunks_case_insensitive(self):
        """Test case-insensitive search (default)."""
        doc = "Hello world. HELLO again. hello there."
        window = Window.create(doc, size=1000)
        matches = window.find_chunks("hello")
        assert len(matches) == 1  # All in one chunk
        assert "Hello" in matches[0].content

    def test_find_chunks_case_sensitive(self):
        """Test case-sensitive search."""
        doc = "Hello world. HELLO again. hello there."
        window = Window.create(doc, size=1000)
        matches = window.find_chunks("HELLO", case_sensitive=True)
        assert len(matches) == 1
        assert "HELLO" in matches[0].content

    def test_find_chunks_no_match(self):
        """Test when no chunks match."""
        doc = "Hello world. Goodbye world."
        window = Window.create(doc, size=1000)
        matches = window.find_chunks("Python")
        assert len(matches) == 0

    def test_find_chunks_multiple_chunks(self):
        """Test finding across multiple chunks."""
        # Create doc with keyword in multiple positions
        doc = "Keyword here. " + "a" * 2000 + " Keyword there."
        window = Window.create(doc, size=500, overlap=50)
        matches = window.find_chunks("Keyword")
        assert len(matches) >= 2


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_basic(self):
        """Test basic stats retrieval."""
        doc = "Hello world. This is a test document."
        window = Window.create(doc, size=1000, overlap=100, strategy="token")
        stats = window.get_stats()

        assert isinstance(stats, WindowStats)
        assert stats.total_chunks == 1
        assert stats.total_chars == len(doc)
        assert stats.total_tokens > 0
        assert stats.avg_chunk_size == len(doc)
        assert stats.overlap_size == 100
        assert stats.strategy == "token"

    def test_get_stats_multiple_chunks(self):
        """Test stats with multiple chunks."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        stats = window.get_stats()

        assert stats.total_chunks > 1
        assert stats.total_chars == 4000
        assert stats.avg_chunk_size > 0
        assert stats.avg_chunk_tokens > 0

    def test_get_stats_empty_document(self):
        """Test stats with empty document."""
        doc = ""
        window = Window.create(doc, size=1000)
        stats = window.get_stats()

        assert stats.total_chunks == 0
        assert stats.total_chars == 0
        assert stats.avg_chunk_size == 0
        assert stats.avg_chunk_tokens == 0


class TestMetadata:
    """Tests for metadata field."""

    def test_chunk_metadata_from_config(self):
        """Test that metadata flows from config to chunks."""
        doc = "Hello world"
        metadata = {"source": "test", "priority": 1}
        window = Window.create(doc, size=1000, metadata=metadata)
        chunks = window.get_all_chunks()

        assert len(chunks) == 1
        assert chunks[0].metadata == {"source": "test", "priority": 1}

    def test_chunk_metadata_is_copied(self):
        """Test that each chunk gets a copy of metadata."""
        doc = "a" * 4000
        metadata = {"source": "test"}
        window = Window.create(doc, size=500, overlap=50, metadata=metadata)
        chunks = window.get_all_chunks()

        # Modify one chunk's metadata
        assert chunks[0].metadata is not None
        chunks[0].metadata["modified"] = True

        # Other chunks should not be affected
        assert chunks[1].metadata is not None
        assert "modified" not in chunks[1].metadata

    def test_chunk_metadata_none_by_default(self):
        """Test that metadata is None by default."""
        doc = "Hello world"
        window = Window.create(doc, size=1000)
        chunks = window.get_all_chunks()

        assert chunks[0].metadata is None


class TestCustomTokenEstimator:
    """Tests for custom estimate_tokens callback."""

    def test_custom_estimator_used(self):
        """Test that custom estimator is used for chunking."""
        # Custom estimator that returns 1 token per character
        custom_estimator = lambda text: len(text)

        doc = "a" * 100
        # With default estimator (4 chars/token): 100 chars = 25 tokens
        # With custom estimator: 100 chars = 100 tokens
        window = Window.create(
            doc, size=50, overlap=5, estimate_tokens=custom_estimator
        )
        chunks = window.get_all_chunks()

        # Custom estimator means we need more chunks
        # 100 tokens with size=50 should give multiple chunks
        assert len(chunks) >= 2

    def test_custom_estimator_in_stats(self):
        """Test that custom estimator is used in get_stats."""
        custom_estimator = lambda text: len(text) * 2  # 2 tokens per char

        doc = "Hello"  # 5 chars
        window = Window.create(doc, size=1000, estimate_tokens=custom_estimator)
        stats = window.get_stats()

        # 5 chars * 2 = 10 tokens
        assert stats.total_tokens == 10

    def test_custom_estimator_in_chunk_token_count(self):
        """Test that custom estimator is used in chunk token_count."""
        custom_estimator = lambda text: len(text)  # 1 token per char

        doc = "Hello"  # 5 chars
        window = Window.create(doc, size=1000, estimate_tokens=custom_estimator)
        chunks = window.get_all_chunks()

        assert chunks[0].token_count == 5

    def test_config_with_custom_estimator(self):
        """Test WindowConfig with custom estimator."""
        custom_estimator = lambda text: len(text)
        config = WindowConfig(size=100, overlap=10, estimate_tokens=custom_estimator)

        doc = "a" * 200
        window = Window.create(doc, config=config)
        chunks = window.get_all_chunks()

        # With 1 token per char, 200 chars with size 100 = multiple chunks
        assert len(chunks) >= 2

    def test_custom_estimator_affects_chunk_boundaries(self):
        """Test that custom estimator affects where chunk boundaries are placed."""
        doc = "a" * 400  # 400 chars

        # Default estimator: ~4 chars/token, so 400 chars = ~100 tokens
        # With size=50 tokens, expect ~2 chunks
        default_window = Window.create(doc, size=50, overlap=0)
        default_chunks = default_window.get_all_chunks()

        # Custom estimator: 1 char = 1 token, so 400 chars = 400 tokens
        # With size=50 tokens, expect ~8 chunks
        custom_estimator = lambda text: len(text)
        custom_window = Window.create(
            doc, size=50, overlap=0, estimate_tokens=custom_estimator
        )
        custom_chunks = custom_window.get_all_chunks()

        # Custom estimator should produce more chunks because it estimates more tokens
        assert len(custom_chunks) > len(default_chunks)

        # Verify chunk sizes are different
        # Default: each chunk ~200 chars (50 tokens * 4 chars/token)
        # Custom: each chunk ~50 chars (50 tokens * 1 char/token)
        assert default_chunks[0].char_count > custom_chunks[0].char_count


# ─────────────────────────────────────────────────────────────────────────────
# New Feature Tests - TypeScript Parity
# ─────────────────────────────────────────────────────────────────────────────


class TestGetContext:
    """Tests for get_context method."""

    def test_get_context_single_chunk(self):
        """Test getting context with no surrounding chunks."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        # Get context for chunk 2 with no surrounding chunks
        context = window.get_context(2, before=0, after=0)
        chunk = window.get(2)
        assert chunk is not None
        assert context == chunk.content

    def test_get_context_with_before(self):
        """Test getting context with chunks before."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        context = window.get_context(2, before=1, after=0)
        # Should include chunks 1 and 2
        chunk2 = window.get(2)
        assert chunk2 is not None
        assert len(context) > len(chunk2.content)

    def test_get_context_with_after(self):
        """Test getting context with chunks after."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        # Use chunk 1 which has chunks after it with content
        context = window.get_context(1, before=0, after=1)
        # Should include chunks 1 and 2 (with overlap removed)
        chunk1 = window.get(1)
        assert chunk1 is not None
        # Context should be at least as long as the single chunk
        assert len(context) >= len(chunk1.content)

    def test_get_context_with_before_and_after(self):
        """Test getting context with surrounding chunks."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        context = window.get_context(2, before=1, after=1)
        # Should include chunks 1, 2, and 3
        chunk2 = window.get(2)
        assert chunk2 is not None
        assert len(context) > len(chunk2.content)

    def test_get_context_clamps_to_bounds(self):
        """Test that context is clamped to document bounds."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        # Request more context than available at start
        context = window.get_context(0, before=5, after=1)
        # Should not fail, just clamp
        assert len(context) > 0

    def test_get_context_at_end(self):
        """Test getting context at end of document."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        last_idx = window.total_chunks - 1
        context = window.get_context(last_idx, before=1, after=5)
        # Should not fail
        assert len(context) > 0


class TestGetChunksInRange:
    """Tests for get_chunks_in_range method."""

    def test_get_chunks_in_range_basic(self):
        """Test basic range retrieval."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        chunks = window.get_chunks_in_range(0, 500)
        # Should get first chunk at minimum
        assert len(chunks) >= 1
        assert chunks[0].index == 0

    def test_get_chunks_in_range_middle(self):
        """Test range in middle of document."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        # Get chunks overlapping with middle portion
        chunks = window.get_chunks_in_range(1000, 2000)
        assert len(chunks) >= 1

    def test_get_chunks_in_range_overlapping(self):
        """Test that overlapping chunks are included."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        all_chunks = window.get_all_chunks()

        # Get chunks in a range that spans multiple chunks
        if len(all_chunks) >= 3:
            # Get range that should include middle chunks
            start = all_chunks[1].start_pos
            end = all_chunks[2].end_pos
            chunks = window.get_chunks_in_range(start, end)
            assert len(chunks) >= 2

    def test_get_chunks_in_range_no_match(self):
        """Test when no chunks are in range."""
        doc = "a" * 100  # Small doc, one chunk
        window = Window.create(doc, size=1000)
        chunks = window.get_chunks_in_range(5000, 6000)
        assert len(chunks) == 0

    def test_get_chunks_in_range_full_doc(self):
        """Test range covering entire document."""
        doc = "a" * 4000
        window = Window.create(doc, size=500, overlap=50)
        chunks = window.get_chunks_in_range(0, 4000)
        assert len(chunks) == window.total_chunks


class TestMergeChunks:
    """Tests for Window.merge_chunks method."""

    def test_merge_chunks_empty(self):
        """Test merging empty list."""
        result = Window.merge_chunks([])
        assert result == ""

    def test_merge_chunks_single(self):
        """Test merging single chunk."""
        chunk = DocumentChunk(
            index=0,
            content="Hello world",
            start_pos=0,
            end_pos=11,
            token_count=2,
            char_count=11,
            is_first=True,
            is_last=True,
            total_chunks=1,
        )
        result = Window.merge_chunks([chunk])
        assert result == "Hello world"

    def test_merge_chunks_no_overlap(self):
        """Test merging chunks without overlap."""
        chunk1 = DocumentChunk(
            index=0,
            content="Hello",
            start_pos=0,
            end_pos=5,
            token_count=1,
            char_count=5,
            is_first=True,
            is_last=False,
            total_chunks=2,
        )
        chunk2 = DocumentChunk(
            index=1,
            content=" world",
            start_pos=5,
            end_pos=11,
            token_count=1,
            char_count=6,
            is_first=False,
            is_last=True,
            total_chunks=2,
        )
        result = Window.merge_chunks([chunk1, chunk2])
        assert result == "Hello world"

    def test_merge_chunks_with_overlap_removed(self):
        """Test merging chunks with overlap removed."""
        # Simulating overlap where chunk2 starts at position 3
        chunk1 = DocumentChunk(
            index=0,
            content="Hello",
            start_pos=0,
            end_pos=5,
            token_count=1,
            char_count=5,
            is_first=True,
            is_last=False,
            total_chunks=2,
        )
        chunk2 = DocumentChunk(
            index=1,
            content="lo world",  # Overlaps with "lo" from chunk1
            start_pos=3,
            end_pos=11,
            token_count=2,
            char_count=8,
            is_first=False,
            is_last=True,
            total_chunks=2,
        )
        result = Window.merge_chunks([chunk1, chunk2], preserve_overlap=False)
        # Should remove the overlap and produce clean merge
        assert result == "Hello world"

    def test_merge_chunks_preserve_overlap(self):
        """Test merging chunks with overlap preserved."""
        chunk1 = DocumentChunk(
            index=0,
            content="Hello",
            start_pos=0,
            end_pos=5,
            token_count=1,
            char_count=5,
            is_first=True,
            is_last=False,
            total_chunks=2,
        )
        chunk2 = DocumentChunk(
            index=1,
            content="lo world",
            start_pos=3,
            end_pos=11,
            token_count=2,
            char_count=8,
            is_first=False,
            is_last=True,
            total_chunks=2,
        )
        result = Window.merge_chunks([chunk1, chunk2], preserve_overlap=True)
        # Should concatenate directly
        assert result == "Hellolo world"


class TestMergeResults:
    """Tests for Window.merge_results method."""

    def test_merge_results_empty(self):
        """Test merging empty results."""
        result = Window.merge_results([])
        assert result == ""

    def test_merge_results_success_only(self):
        """Test that only successful results are merged."""
        chunk = DocumentChunk(
            index=0,
            content="test",
            start_pos=0,
            end_pos=4,
            token_count=1,
            char_count=4,
            is_first=True,
            is_last=True,
            total_chunks=1,
        )
        results = [
            ChunkResult(chunk=chunk, status="success", content="Result 1"),
            ChunkResult(chunk=chunk, status="error", content="", error="Failed"),
            ChunkResult(chunk=chunk, status="success", content="Result 2"),
        ]
        merged = Window.merge_results(results)
        assert merged == "Result 1\n\nResult 2"

    def test_merge_results_custom_separator(self):
        """Test custom separator."""
        chunk = DocumentChunk(
            index=0,
            content="test",
            start_pos=0,
            end_pos=4,
            token_count=1,
            char_count=4,
            is_first=True,
            is_last=True,
            total_chunks=1,
        )
        results = [
            ChunkResult(chunk=chunk, status="success", content="A"),
            ChunkResult(chunk=chunk, status="success", content="B"),
        ]
        merged = Window.merge_results(results, separator="\n---\n")
        assert merged == "A\n---\nB"

    def test_merge_results_skips_empty_content(self):
        """Test that empty content is skipped."""
        chunk = DocumentChunk(
            index=0,
            content="test",
            start_pos=0,
            end_pos=4,
            token_count=1,
            char_count=4,
            is_first=True,
            is_last=True,
            total_chunks=1,
        )
        results = [
            ChunkResult(chunk=chunk, status="success", content="A"),
            ChunkResult(chunk=chunk, status="success", content=""),
            ChunkResult(chunk=chunk, status="success", content="B"),
        ]
        merged = Window.merge_results(results)
        assert merged == "A\n\nB"


class TestGetProcessingStats:
    """Tests for Window.get_stats method."""

    def test_get_processing_stats_empty(self):
        """Test stats from empty results."""
        stats = Window.get_stats([])
        assert stats.total == 0
        assert stats.successful == 0
        assert stats.failed == 0
        assert stats.success_rate == 0.0
        assert stats.avg_duration == 0.0
        assert stats.total_duration == 0.0

    def test_get_processing_stats_all_success(self):
        """Test stats when all succeed."""
        chunk = DocumentChunk(
            index=0,
            content="test",
            start_pos=0,
            end_pos=4,
            token_count=1,
            char_count=4,
            is_first=True,
            is_last=True,
            total_chunks=1,
        )
        results = [
            ChunkResult(chunk=chunk, status="success", content="A", duration=100.0),
            ChunkResult(chunk=chunk, status="success", content="B", duration=200.0),
        ]
        stats = Window.get_stats(results)
        assert stats.total == 2
        assert stats.successful == 2
        assert stats.failed == 0
        assert stats.success_rate == 100.0
        assert stats.avg_duration == 150.0
        assert stats.total_duration == 300.0

    def test_get_processing_stats_mixed(self):
        """Test stats with mixed success/failure."""
        chunk = DocumentChunk(
            index=0,
            content="test",
            start_pos=0,
            end_pos=4,
            token_count=1,
            char_count=4,
            is_first=True,
            is_last=True,
            total_chunks=1,
        )
        results = [
            ChunkResult(chunk=chunk, status="success", content="A", duration=100.0),
            ChunkResult(chunk=chunk, status="error", error="Failed", duration=50.0),
            ChunkResult(chunk=chunk, status="success", content="B", duration=200.0),
            ChunkResult(chunk=chunk, status="error", error="Failed", duration=50.0),
        ]
        stats = Window.get_stats(results)
        assert stats.total == 4
        assert stats.successful == 2
        assert stats.failed == 2
        assert stats.success_rate == 50.0
        assert stats.total_duration == 400.0
        assert stats.avg_duration == 100.0


class TestChunkResultDuration:
    """Tests for duration field in ChunkResult."""

    def test_chunk_result_has_duration(self):
        """Test that ChunkResult has duration field."""
        chunk = DocumentChunk(
            index=0,
            content="test",
            start_pos=0,
            end_pos=4,
            token_count=1,
            char_count=4,
            is_first=True,
            is_last=True,
            total_chunks=1,
        )
        result = ChunkResult(
            chunk=chunk,
            status="success",
            content="Result",
            duration=150.5,
        )
        assert result.duration == 150.5

    def test_chunk_result_duration_default(self):
        """Test that duration defaults to 0."""
        chunk = DocumentChunk(
            index=0,
            content="test",
            start_pos=0,
            end_pos=4,
            token_count=1,
            char_count=4,
            is_first=True,
            is_last=True,
            total_chunks=1,
        )
        result = ChunkResult(chunk=chunk, status="success", content="Result")
        assert result.duration == 0.0


class TestContextRestorationOptions:
    """Tests for ContextRestorationOptions."""

    def test_default_options(self):
        """Test default context restoration options."""
        options = ContextRestorationOptions()
        assert options.enabled is True
        assert options.strategy == "adjacent"
        assert options.max_attempts == 2
        assert options.on_restore is None

    def test_custom_options(self):
        """Test custom context restoration options."""
        callback_called = []

        def on_restore(from_idx: int, to_idx: int):
            callback_called.append((from_idx, to_idx))

        options = ContextRestorationOptions(
            enabled=False,
            strategy="full",
            max_attempts=5,
            on_restore=on_restore,
        )
        assert options.enabled is False
        assert options.strategy == "full"
        assert options.max_attempts == 5

        # Test callback
        assert options.on_restore is not None
        options.on_restore(1, 2)
        assert callback_called == [(1, 2)]


class TestGetRestorationChunk:
    """Tests for _get_restoration_chunk function."""

    def test_adjacent_strategy_returns_next_chunk(self):
        """Test adjacent strategy returns next chunk when available."""
        from l0.window import _get_restoration_chunk

        doc = "word " * 100
        window = Window.create(doc, size=10, overlap=2)

        # From middle chunk, should return next chunk
        result = _get_restoration_chunk(window, current_index=1, strategy="adjacent")
        assert result == 2

    def test_adjacent_strategy_returns_previous_at_end(self):
        """Test adjacent strategy returns previous chunk when at end."""
        from l0.window import _get_restoration_chunk

        doc = "word " * 100
        window = Window.create(doc, size=10, overlap=2)
        last_index = window.total_chunks - 1

        # From last chunk, should return previous chunk
        result = _get_restoration_chunk(
            window, current_index=last_index, strategy="adjacent"
        )
        assert result == last_index - 1

    def test_adjacent_strategy_independent_of_window_cursor(self):
        """Test that adjacent strategy uses current_index, not window's internal cursor."""
        from l0.window import _get_restoration_chunk

        doc = "word " * 100
        window = Window.create(doc, size=10, overlap=2)

        # Move window's internal cursor to end
        while window.has_next():
            window.next()

        # Even though window cursor is at end, passing current_index=0 should return 1
        result = _get_restoration_chunk(window, current_index=0, strategy="adjacent")
        assert result == 1

    def test_overlap_strategy_returns_next_only(self):
        """Test overlap strategy only returns next chunk."""
        from l0.window import _get_restoration_chunk

        doc = "word " * 100
        window = Window.create(doc, size=10, overlap=2)
        last_index = window.total_chunks - 1

        # From middle, returns next
        result = _get_restoration_chunk(window, current_index=1, strategy="overlap")
        assert result == 2

        # From last chunk, returns None (doesn't try previous)
        result = _get_restoration_chunk(
            window, current_index=last_index, strategy="overlap"
        )
        assert result is None

    def test_full_strategy_returns_next_then_previous(self):
        """Test full strategy returns next chunk first, then previous."""
        from l0.window import _get_restoration_chunk

        doc = "word " * 100
        window = Window.create(doc, size=10, overlap=2)
        last_index = window.total_chunks - 1

        # From middle, returns next
        result = _get_restoration_chunk(window, current_index=1, strategy="full")
        assert result == 2

        # From last chunk, returns previous
        result = _get_restoration_chunk(
            window, current_index=last_index, strategy="full"
        )
        assert result == last_index - 1

    def test_returns_none_for_single_chunk(self):
        """Test returns None when document has only one chunk."""
        from l0.window import _get_restoration_chunk

        doc = "short"
        window = Window.create(doc, size=1000, overlap=0)
        assert window.total_chunks == 1

        result = _get_restoration_chunk(window, current_index=0, strategy="adjacent")
        assert result is None


class TestWindowConfigPreserveOptions:
    """Tests for preserve_paragraphs and preserve_sentences options."""

    def test_default_preserve_options(self):
        """Test default preserve options."""
        config = WindowConfig()
        assert config.preserve_paragraphs is True
        assert config.preserve_sentences is False

    def test_custom_preserve_options(self):
        """Test custom preserve options."""
        config = WindowConfig(
            preserve_paragraphs=False,
            preserve_sentences=True,
        )
        assert config.preserve_paragraphs is False
        assert config.preserve_sentences is True

    def test_window_with_preserve_options(self):
        """Test creating window with preserve options."""
        doc = "Paragraph one.\n\nParagraph two."
        config = WindowConfig(
            size=1000,
            preserve_paragraphs=True,
            preserve_sentences=True,
        )
        window = Window.create(doc, config=config)
        assert window.config.preserve_paragraphs is True
        assert window.config.preserve_sentences is True


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_processing_stats_fields(self):
        """Test ProcessingStats has all required fields."""
        stats = ProcessingStats(
            total=10,
            successful=8,
            failed=2,
            success_rate=80.0,
            avg_duration=100.5,
            total_duration=1005.0,
        )
        assert stats.total == 10
        assert stats.successful == 8
        assert stats.failed == 2
        assert stats.success_rate == 80.0
        assert stats.avg_duration == 100.5
        assert stats.total_duration == 1005.0
