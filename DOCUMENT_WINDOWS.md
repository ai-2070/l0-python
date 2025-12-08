# Document Windows Guide

Automatic chunking and navigation for long documents.

## Quick Start

```python
from l0 import Window

# Create a window over a long document
window = Window.create(
    long_document,
    size=2000,      # Tokens per chunk
    overlap=200,    # Overlap between chunks
    strategy="paragraph",  # "token" | "char" | "paragraph" | "sentence"
)

# Process all chunks
results = await window.process_all(
    lambda chunk: ChunkProcessConfig(
        stream=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Summarize: {chunk.content}"}],
            stream=True,
        )
    )
)

# Merge results
summary = Window.merge_results(results)
```

---

## Table of Contents

- [Chunking Strategies](#chunking-strategies)
- [Window Options](#window-options)
- [Navigation](#navigation)
- [Processing](#processing)
- [Chunk Structure](#chunk-structure)
- [Window Statistics](#window-statistics)
- [Overlap](#overlap)
- [Context Restoration](#context-restoration)
- [Helper Functions](#helper-functions)
- [Presets](#presets)
- [Examples](#examples)
- [Best Practices](#best-practices)

---

## Chunking Strategies

| Strategy    | Best For        | Behavior                        |
| ----------- | --------------- | ------------------------------- |
| `token`     | General purpose | Chunks by estimated token count |
| `char`      | Fixed-length    | Chunks by character count       |
| `paragraph` | Structured docs | Preserves paragraph boundaries  |
| `sentence`  | Precision       | Never splits sentences          |

```python
from l0 import Window

# Token-based (default)
window = Window.create(doc, size=2000, strategy="token")

# Paragraph-based
window = Window.create(doc, size=2000, strategy="paragraph")

# Sentence-based
window = Window.create(doc, size=1500, strategy="sentence")

# Character-based
window = Window.create(doc, size=8000, strategy="char")
```

---

## Window Options

```python
from l0 import WindowConfig

@dataclass
class WindowConfig:
    """Configuration for document windowing."""
    
    size: int = 2000
    """Size of each chunk (in tokens or characters)."""
    
    overlap: int = 200
    """Overlap between chunks (in tokens or characters)."""
    
    strategy: ChunkingStrategy = "token"
    """Chunking strategy: "token" | "char" | "paragraph" | "sentence"."""
    
    estimate_tokens: Callable[[str], int] | None = None
    """Custom token estimator function.
    If not provided, uses rough estimate (1 token ≈ 4 chars)."""
    
    preserve_paragraphs: bool = True
    """Preserve paragraph boundaries when chunking."""
    
    preserve_sentences: bool = False
    """Preserve sentence boundaries when chunking."""
    
    metadata: dict[str, Any] | None = None
    """Custom metadata to attach to each chunk."""
```

### Using WindowConfig

```python
from l0 import Window, WindowConfig

# Create config object
config = WindowConfig(
    size=2000,
    overlap=200,
    strategy="paragraph",
    metadata={"source": "legal_doc.pdf"},
)

# Use with Window.create
window = Window.create(document, config=config)

# Or use keyword arguments directly
window = Window.create(
    document,
    size=2000,
    overlap=200,
    strategy="paragraph",
)
```

---

## Navigation

```python
from l0 import Window

window = Window.create(document, size=2000)

# Get chunks
window.current()           # Current chunk
window.get(0)              # Specific chunk by index
window.get_all_chunks()    # All chunks
window.get_range(0, 5)     # Range of chunks (start inclusive, end exclusive)

# Navigate
window.next()              # Move to and return next chunk
window.prev()              # Move to and return previous chunk
window.jump(5)             # Jump to chunk 5 and return it
window.reset()             # Back to first chunk

# Check bounds
window.has_next()          # Has more chunks?
window.has_prev()          # Has previous?
window.total_chunks        # Total count
window.current_index       # Current position (0-based)

# Search and context
window.find_chunks("search term")        # Find chunks containing text
window.find_chunks("term", case_sensitive=True)  # Case-sensitive search
window.get_context(3, before=1, after=1) # Get surrounding context merged
window.get_chunks_in_range(0, 500)       # Get chunks within character range

# Statistics
stats = window.get_stats()  # Get window statistics
```

### Python Special Methods

DocumentWindow supports Python idioms:

```python
# Iteration
for chunk in window:
    print(chunk.content)

# Length
num_chunks = len(window)

# Indexing
first_chunk = window[0]
last_chunk = window[-1]
```

---

## Processing

### Parallel (Default)

```python
from l0 import Window, ChunkProcessConfig

window = Window.create(document, size=2000)

results = await window.process_all(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": c.content}],
            stream=True,
        )
    )
)
```

### Parallel with Concurrency Limit

```python
# Limit to 3 concurrent streams (default is 5)
results = await window.process_all(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": c.content}],
            stream=True,
        )
    ),
    concurrency=3,
)
```

### Sequential

```python
results = await window.process_sequential(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": c.content}],
            stream=True,
        )
    )
)
```

### With Retry & Fallbacks

```python
from l0 import Retry

results = await window.process_all(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": c.content}],
            stream=True,
        ),
        retry=Retry(attempts=3),
        fallbacks=[
            lambda c=chunk: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": c.content}],
                stream=True,
            ),
        ],
    )
)
```

---

## Chunk Structure

```python
from l0 import DocumentChunk

@dataclass
class DocumentChunk:
    index: int              # Position (0-based)
    content: str            # Chunk text
    start_pos: int          # Start position in original document
    end_pos: int            # End position in original document
    token_count: int        # Estimated tokens
    char_count: int         # Character count
    is_first: bool          # Is this the first chunk?
    is_last: bool           # Is this the last chunk?
    total_chunks: int       # Total number of chunks
    metadata: dict[str, Any] | None = None  # Custom metadata
```

### Accessing Chunk Properties

```python
chunk = window.get(0)

print(f"Chunk {chunk.index + 1} of {chunk.total_chunks}")
print(f"Position: {chunk.start_pos}-{chunk.end_pos}")
print(f"Tokens: {chunk.token_count}, Characters: {chunk.char_count}")
print(f"First: {chunk.is_first}, Last: {chunk.is_last}")
print(f"Content: {chunk.content[:100]}...")
```

---

## Window Statistics

```python
from l0 import WindowStats

@dataclass
class WindowStats:
    total_chunks: int       # Total chunks
    total_chars: int        # Total document length (characters)
    total_tokens: int       # Estimated total tokens
    avg_chunk_size: int     # Average chunk size (characters)
    avg_chunk_tokens: int   # Average chunk tokens
    overlap_size: int       # Overlap size (tokens)
    strategy: ChunkingStrategy  # Chunking strategy used

# Get statistics
stats = window.get_stats()
print(f"Total chunks: {stats.total_chunks}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Avg chunk size: {stats.avg_chunk_size} chars")
print(f"Avg chunk tokens: {stats.avg_chunk_tokens}")
print(f"Strategy: {stats.strategy}")
```

---

## Overlap

Overlap maintains context between chunks:

```python
window = Window.create(document, size=2000, overlap=200)

# Chunk 0: tokens 0-2000
# Chunk 1: tokens 1800-3800 (200 overlap with chunk 0)
# Chunk 2: tokens 3600-5600 (200 overlap with chunk 1)
```

**Recommendation:** Use 10% overlap (e.g., 200 for 2000-token chunks)

---

## Context Restoration

Auto-retry with adjacent chunks if drift is detected:

```python
from l0 import Window, ContextRestorationOptions
from l0.window import l0_with_window

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
        strategy="adjacent",  # "adjacent" | "overlap" | "full"
        max_attempts=2,
        on_restore=lambda from_idx, to_idx: print(f"Restored from chunk {from_idx} to {to_idx}"),
    ),
)
```

### ContextRestorationOptions

```python
from l0 import ContextRestorationOptions

@dataclass
class ContextRestorationOptions:
    enabled: bool = True
    """Enable automatic context restoration."""
    
    strategy: ContextRestorationStrategy = "adjacent"
    """Restoration strategy: "adjacent" | "overlap" | "full"."""
    
    max_attempts: int = 2
    """Maximum restoration attempts."""
    
    on_restore: Callable[[int, int], None] | None = None
    """Callback when restoration occurs (from_index, to_index)."""
```

---

## Helper Functions

### process_with_window

Process a document directly without creating a window instance:

```python
from l0.window import process_with_window

results = await process_with_window(
    document,
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Summarize: {c.content}"}],
            stream=True,
        )
    ),
    size=2000,
    overlap=200,
)
```

### merge_results

Merge results from multiple chunk processing into a single text:

```python
from l0 import Window

results = await window.process_all(processor)

# Default separator: "\n\n"
merged = Window.merge_results(results)

# Custom separator
merged = Window.merge_results(results, separator="\n---\n")
```

### merge_chunks

Merge document chunks back into original document:

```python
from l0 import Window

# Get chunks
chunks = window.get_all_chunks()

# Merge without overlap (default)
text = Window.merge_chunks(chunks)

# Merge preserving overlap
text = Window.merge_chunks(chunks, preserve_overlap=True)
```

### get_stats

Get processing statistics from results:

```python
from l0 import Window, ProcessingStats

results = await window.process_all(processor)

stats = Window.get_stats(results)
# ProcessingStats(
#     total=10,
#     successful=9,
#     failed=1,
#     success_rate=90.0,
#     avg_duration=1500.0,
#     total_duration=15000.0
# )
```

### estimate_tokens

Estimate token count for text:

```python
from l0 import Window

token_count = Window.estimate_tokens(text)
# Uses heuristic: ~4 characters per token
```

---

## Presets

Quick window creation with common configurations:

```python
from l0 import Window

# Small: 1000 tokens, 100 overlap, token strategy
window = Window.small(document)

# Medium: 2000 tokens, 200 overlap, token strategy (default)
window = Window.medium(document)

# Large: 4000 tokens, 400 overlap, token strategy
window = Window.large(document)

# Paragraph: 2000 tokens, 200 overlap, paragraph strategy
window = Window.paragraph(document)

# Sentence: 1500 tokens, 150 overlap, sentence strategy
window = Window.sentence(document)
```

---

## ChunkResult Structure

```python
from l0 import ChunkResult

@dataclass
class ChunkResult(Generic[T]):
    chunk: DocumentChunk           # The processed chunk
    status: Literal["success", "error"]  # Processing status
    result: Stream[Any] | None = None    # L0 Stream result (if success)
    content: str = ""              # Extracted text content
    error: str | None = None       # Error message (if error)
    duration: float = 0.0          # Duration in milliseconds
```

### Handling Results

```python
results = await window.process_all(processor)

for result in results:
    if result.status == "success":
        print(f"Chunk {result.chunk.index}: {result.content[:50]}...")
        print(f"Duration: {result.duration}ms")
    else:
        print(f"Chunk {result.chunk.index} failed: {result.error}")
```

---

## Examples

### Legal Document Analysis

```python
from l0 import Window, ChunkProcessConfig

window = Window.paragraph(contract)  # Use paragraph boundaries

results = await window.process_all(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Extract legal clauses from:\n\n{c.content}"
            }],
            stream=True,
        )
    )
)

clauses = Window.merge_results(results)
```

### Transcript Summarization

```python
from l0 import Window, ChunkProcessConfig

window = Window.sentence(transcript)  # Preserve sentence boundaries

summaries = await window.process_sequential(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Summarize this section:\n\n{c.content}"
            }],
            stream=True,
        )
    )
)

summary = Window.merge_results(summaries)
```

### Code Documentation

```python
from l0 import Window, ChunkProcessConfig

window = Window.create(source_code, size=1500, strategy="paragraph")

docs = await window.process_all(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Generate documentation for:\n\n{c.content}"
            }],
            stream=True,
        )
    ),
    concurrency=3,
)

documentation = Window.merge_results(docs)
```

### Custom Token Estimation

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

window = Window.create(
    document,
    size=2000,
    overlap=200,
    estimate_tokens=lambda text: len(enc.encode(text)),
)
```

### Searching and Context

```python
window = Window.create(document, size=2000)

# Find all chunks containing a term
relevant_chunks = window.find_chunks("important keyword")

# Get context around a specific chunk (merge before/after)
context = window.get_context(5, before=2, after=2)

# Get chunks within a specific position range
range_chunks = window.get_chunks_in_range(1000, 5000)
```

### Rate-Limited Processing

```python
from l0 import Window, ChunkProcessConfig, Retry

window = Window.create(document, size=2000)

# Low concurrency for rate-limited APIs
results = await window.process_all(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": c.content}],
            stream=True,
        ),
        retry=Retry(attempts=3),  # Retry on rate limits
    ),
    concurrency=2,  # Only 2 concurrent requests
)
```

---

## Best Practices

1. **Chunk size** - Leave room for prompt + response (e.g., 2000 for 8k context)
2. **Overlap** - Use 10% for context continuity
3. **Strategy** - Match to content type (paragraph for docs, sentence for transcripts)
4. **Concurrency** - Limit for rate-limited APIs
5. **Error handling** - Check `result.status == "error"` for failures
6. **Custom token estimation** - Use tiktoken for accurate counts with OpenAI models

```python
from l0 import Window, ChunkProcessConfig, Retry

# Recommended setup
window = Window.create(
    document,
    size=2000,
    overlap=200,
    strategy="paragraph",
)

results = await window.process_all(
    lambda chunk: ChunkProcessConfig(
        stream=lambda c=chunk: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": c.content}],
            stream=True,
        ),
        retry=Retry(attempts=3),
    ),
    concurrency=3,
)

# Handle failures
stats = Window.get_stats(results)
if stats.failed > 0:
    print(f"{stats.failed} chunks failed ({stats.success_rate}% success rate)")

# Get merged output
output = Window.merge_results(results)
```

---

## Types Summary

| Type                        | Description                           |
| --------------------------- | ------------------------------------- |
| `Window`                    | Scoped API for window operations      |
| `DocumentWindow`            | Window instance for navigation/processing |
| `WindowConfig`              | Configuration for window creation     |
| `DocumentChunk`             | A single chunk of the document        |
| `ChunkProcessConfig`        | Processing configuration for a chunk  |
| `ChunkResult`               | Result of processing a chunk          |
| `WindowStats`               | Statistics about the window           |
| `ProcessingStats`           | Statistics from processing results    |
| `ContextRestorationOptions` | Options for drift context restoration |
| `ChunkingStrategy`          | Strategy type: `"token"` \| `"char"` \| `"paragraph"` \| `"sentence"` |
| `ContextRestorationStrategy`| Strategy type: `"adjacent"` \| `"overlap"` \| `"full"` |
