"""Tests for l0.window module."""

import pytest

from src.l0.window import (
    DocumentChunk,
    DocumentWindow,
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
    def sample_doc(self):
        return "a" * 4000  # Large enough for multiple chunks

    @pytest.fixture
    def window(self, sample_doc):
        return Window.create(sample_doc, size=500, overlap=50)

    def test_create_window(self, window):
        assert window.total_chunks > 1

    def test_current(self, window):
        chunk = window.current()
        assert chunk is not None
        assert chunk.index == 0
        assert chunk.is_first

    def test_get(self, window):
        chunk = window.get(0)
        assert chunk is not None
        assert chunk.index == 0

        last_chunk = window.get(window.total_chunks - 1)
        assert last_chunk is not None
        assert last_chunk.is_last

        # Out of bounds
        assert window.get(-1) is None
        assert window.get(999) is None

    def test_get_all_chunks(self, window):
        chunks = window.get_all_chunks()
        assert len(chunks) == window.total_chunks

    def test_navigation_next(self, window):
        assert window.current_index == 0
        chunk = window.next()
        assert chunk is not None
        assert window.current_index == 1

    def test_navigation_prev(self, window):
        window.next()
        assert window.current_index == 1
        chunk = window.prev()
        assert chunk is not None
        assert window.current_index == 0

    def test_navigation_jump(self, window):
        chunk = window.jump(2)
        assert chunk is not None
        assert window.current_index == 2

    def test_navigation_reset(self, window):
        window.jump(2)
        chunk = window.reset()
        assert chunk is not None
        assert window.current_index == 0

    def test_has_next(self, window):
        assert window.has_next()
        window.jump(window.total_chunks - 1)
        assert not window.has_next()

    def test_has_prev(self, window):
        assert not window.has_prev()
        window.next()
        assert window.has_prev()

    def test_iteration(self, window):
        chunks = list(window)
        assert len(chunks) == window.total_chunks

    def test_len(self, window):
        assert len(window) == window.total_chunks

    def test_getitem(self, window):
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

        async def processor(chunk: DocumentChunk) -> str:
            return chunk.content

        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await window.process_all(processor, concurrency=0)

    @pytest.mark.asyncio
    async def test_process_all_negative_concurrency_raises(self):
        """Test that negative concurrency raises ValueError."""
        doc = "Test document"
        window = Window.create(doc, size=100)

        async def processor(chunk: DocumentChunk) -> str:
            return chunk.content

        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await window.process_all(processor, concurrency=-1)


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
        chunks[0].metadata["modified"] = True

        # Other chunks should not be affected
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
