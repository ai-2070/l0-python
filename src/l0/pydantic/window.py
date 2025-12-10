"""Pydantic models for L0 window types.

These models mirror the dataclasses in l0.window for runtime validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkingStrategy = Literal["token", "char", "paragraph", "sentence"]
ContextRestorationStrategy = Literal["adjacent", "overlap", "full"]


class DocumentChunkModel(BaseModel):
    """Pydantic model for a chunk of a document."""

    model_config = ConfigDict(extra="forbid")

    index: int
    content: str
    start_pos: int
    end_pos: int
    token_count: int
    char_count: int
    is_first: bool
    is_last: bool
    total_chunks: int
    metadata: dict[str, Any] | None = None


class WindowConfigModel(BaseModel):
    """Pydantic model for document windowing configuration."""

    model_config = ConfigDict(extra="forbid")

    size: int = 2000
    overlap: int = 200
    strategy: ChunkingStrategy = "token"
    preserve_paragraphs: bool = True
    preserve_sentences: bool = False
    metadata: dict[str, Any] | None = None
    # estimate_tokens callback is not serializable


class ContextRestorationOptionsModel(BaseModel):
    """Pydantic model for context restoration options."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    strategy: ContextRestorationStrategy = "adjacent"
    max_attempts: int = 2
    # on_restore callback is not serializable


class WindowStatsModel(BaseModel):
    """Pydantic model for document window statistics."""

    model_config = ConfigDict(extra="forbid")

    total_chunks: int
    total_chars: int
    total_tokens: int
    avg_chunk_size: int
    avg_chunk_tokens: int
    overlap_size: int
    strategy: ChunkingStrategy


class ChunkResultModel(BaseModel):
    """Pydantic model for result of processing a chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk: DocumentChunkModel
    status: Literal["success", "error"]
    content: str = ""
    error: str | None = None
    duration: float = 0.0
    # result (Stream) is not serializable


class ProcessingStatsModel(BaseModel):
    """Pydantic model for processing statistics."""

    model_config = ConfigDict(extra="forbid")

    total: int
    successful: int
    failed: int
    success_rate: float
    avg_duration: float
    total_duration: float
