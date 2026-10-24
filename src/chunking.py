"""Deterministic word-based chunking module for the Personal AI Knowledge Assistant.

Splits loaded DocumentRecord objects into bounded word chunks with configurable overlap.
Preserves source path, page numbers, and position while computing stable, deterministic
SHA-256 identifiers.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Optional

from src.loaders import DocumentRecord

DEFAULT_CHUNK_SIZE = 120
DEFAULT_CHUNK_OVERLAP = 25


@dataclass(frozen=True)
class ChunkRecord:
    """Represents a discrete word chunk produced from a DocumentRecord.
    
    Attributes:
        chunk_id: Stable, deterministic identifier incorporating the full source path,
                  page identity, chunk position, and a hash of the content.
                  Example: 'project_aurora__data_documents__p0_c000_3a8f1b9c'.
                  Identical text in files with the same stem but different directories
                  or extensions will produce different IDs.
        text: The text content of the chunk.
        source: Normalized relative source path (e.g. 'data/documents/project_aurora.md').
        page: 1-indexed page number for multi-page documents; None for text/markdown.
        chunk_index: 0-indexed position of this chunk within the parent document/page.
        content_hash: Full SHA-256 hex digest of the chunk's text content.
        word_count: Number of whitespace-separated words in this chunk.
        metadata: Inherited and chunk-level metadata.
    """
    chunk_id: str
    text: str
    source: str
    page: Optional[int]
    chunk_index: int
    content_hash: str
    word_count: int
    metadata: dict = field(default_factory=dict)


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate that chunk size and overlap parameters are mathematically sound.
    
    Args:
        chunk_size: Target number of words per chunk. Must be > 0.
        chunk_overlap: Number of overlapping words between consecutive chunks.
                       Must be >= 0 and strictly less than chunk_size.
                       
    Raises:
        ValueError: If chunk settings violate boundary rules.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got: {chunk_size}")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be a non-negative integer, got: {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size}). "
            "Overlap equal to or greater than chunk size would cause an infinite windowing loop."
        )


def _compute_chunk_id(source: str, page: Optional[int], chunk_index: int, content_hash: str) -> str:
    """Derive a stable, human-readable, deterministic identifier for a chunk.

    The ID incorporates:
    - The full normalized source path (not just the file stem), so files with the
      same name in different directories or with different extensions are distinct.
    - The page identity (1-indexed for PDFs; 0 as sentinel for non-paginated text).
    - The chunk's position (chunk_index) within its document/page.
    - The SHA-256 hash of the chunk's text content.

    Uses SHA-256 throughout; never uses Python's non-deterministic process-dependent hash().
    """
    # Readable prefix: file stem from the source path
    stem = Path(source).stem.replace(" ", "_")
    page_tag = f"p{page}" if page is not None else "p0"
    # Hash the full normalized source path (not the stem) to distinguish same-stem files
    # in different directories or with different extensions (e.g. subdir_a/report.md vs
    # subdir_b/report.md, or report.md vs report.pdf).
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
    # Use first 8 characters of the content hash for readability
    return f"{stem}_{source_hash}_{page_tag}_c{chunk_index:03d}_{content_hash[:8]}"


def chunk_document(
    doc: DocumentRecord,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Split a single DocumentRecord into word-based ChunkRecord objects.
    
    Args:
        doc: The source DocumentRecord to split.
        chunk_size: Number of words per chunk (default 120).
        chunk_overlap: Number of overlapping words between consecutive chunks (default 25).
        
    Returns:
        List of ChunkRecord objects. Empty if document text is empty.
    """
    validate_chunk_settings(chunk_size, chunk_overlap)

    words = doc.text.split()
    total_words = len(words)
    if total_words == 0:
        return []

    # If the document fits entirely in one chunk, return it immediately
    if total_words <= chunk_size:
        chunk_text = " ".join(words)
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        chunk_id = _compute_chunk_id(doc.source, doc.page, 0, content_hash)
        return [
            ChunkRecord(
                chunk_id=chunk_id,
                text=chunk_text,
                source=doc.source,
                page=doc.page,
                chunk_index=0,
                content_hash=content_hash,
                word_count=total_words,
                metadata={**doc.metadata, "total_chunks": 1},
            )
        ]

    chunks: list[ChunkRecord] = []
    step = chunk_size - chunk_overlap
    start = 0
    chunk_index = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        chunk_id = _compute_chunk_id(doc.source, doc.page, chunk_index, content_hash)

        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                text=chunk_text,
                source=doc.source,
                page=doc.page,
                chunk_index=chunk_index,
                content_hash=content_hash,
                word_count=len(chunk_words),
                metadata={**doc.metadata},
            )
        )

        chunk_index += 1
        if end >= total_words:
            break
        start += step

    # Annotate chunks with the total chunk count for this document/page
    annotated_chunks = [
        ChunkRecord(
            chunk_id=c.chunk_id,
            text=c.text,
            source=c.source,
            page=c.page,
            chunk_index=c.chunk_index,
            content_hash=c.content_hash,
            word_count=c.word_count,
            metadata={**c.metadata, "total_chunks": len(chunks)},
        )
        for c in chunks
    ]

    return annotated_chunks


def chunk_documents(
    docs: list[DocumentRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Split a list of DocumentRecord objects into ChunkRecord objects.
    
    Ensures document and page boundaries are strictly respected: chunks are NEVER
    merged across distinct DocumentRecord boundaries.
    
    Args:
        docs: List of DocumentRecord objects.
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words between consecutive chunks.
        
    Returns:
        Flattened list of all generated ChunkRecord objects.
    """
    all_chunks: list[ChunkRecord] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
