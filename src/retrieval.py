"""Retrieval module for the Personal AI Knowledge Assistant.

Bridges embedding and vector_store: accepts a question string, embeds it,
queries the index, and returns ranked evidence records ready for the
generation pipeline.

Per ARCHITECTURE.md:
- Start with configurable top_k=4 and cosine distance.
- Distance is NOT confidence or probability; do not display it as such.
- Inspect expected evidence before adding score thresholds.
- Bound evidence length and inspect truncation risk.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from src.config import DEFAULT_TOP_K
from src.embeddings import embed_query, EMBEDDING_DIMENSION
from src.vector_store import query_index


def validate_query(query: str) -> str:
    """Validate and strip a retrieval query string.

    Args:
        query: Raw query string from user or pipeline.

    Returns:
        Stripped, non-empty query string.

    Raises:
        ValueError: If query is empty or whitespace only.
    """
    stripped = query.strip()
    if not stripped:
        raise ValueError(
            "Query must be a non-empty string. "
            "Empty queries cannot be embedded or retrieved against."
        )
    return stripped


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage_dir: Optional[Path] = None,
) -> list[dict]:
    """Embed a question and retrieve the top-k most similar chunks from the index.

    Args:
        question: The user's question string.
        top_k: Number of candidate evidence passages to retrieve. Must be >= 1.
        storage_dir: Override default storage path (useful for tests).

    Returns:
        List of evidence dicts ordered by ascending cosine distance
        (most relevant first). Each dict contains:
            chunk_id, text, source, page, chunk_index, distance, metadata.
        'distance' is a cosine distance in [0, 2] for L2-normalized embeddings.
        Smaller values indicate greater similarity. Distance is NOT a confidence
        score or probability.

    Raises:
        ValueError: If question is empty, top_k < 1, or the index is absent/empty.
    """
    question = validate_query(question)

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got: {top_k!r}")

    query_vec = embed_query(question)
    passages = query_index(
        query_embedding=query_vec,
        top_k=top_k,
        storage_dir=storage_dir,
    )
    return passages


def format_evidence_for_display(passages: list[dict], max_chars_per_chunk: int = 800) -> str:
    """Format retrieved passages for human-readable terminal display.

    Truncates individual passage text to max_chars_per_chunk to prevent
    very long documents from dominating output. Truncation is noted explicitly.

    Args:
        passages: List of passage dicts from retrieve().
        max_chars_per_chunk: Maximum characters shown per passage.

    Returns:
        Formatted multi-line string.
    """
    if not passages:
        return "(No passages retrieved)"

    lines = []
    for rank, p in enumerate(passages, start=1):
        page_label = f"page {p['page']}" if p.get("page") else "no page"
        text = p["text"]
        truncated = False
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk]
            truncated = True

        lines.append(
            f"[{rank}] source={p['source']} | {page_label} | "
            f"chunk_index={p['chunk_index']} | distance={p['distance']:.4f}"
        )
        lines.append(f"    {text}" + (" [TRUNCATED]" if truncated else ""))
        lines.append("")

    return "\n".join(lines)
