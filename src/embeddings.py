"""Embedding module for the Personal AI Knowledge Assistant.

Loads and caches the sentence-transformers model once per process, computes
normalized float32 embeddings from lists of strings, and exposes the model
identifier and output dimension for index configuration recording.

IMPORTANT (per TECH_STACK.md):
- Use the SAME model instance and normalization for indexing and for queries.
- Never silently switch embedding providers.
- Record the model identifier alongside every index build.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

# Lazy-load sentinel: the model is created on first use and cached here.
_MODEL = None
_MODEL_NAME: Optional[str] = None

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384   # Fixed output size for all-MiniLM-L6-v2


def _get_model():
    """Return the cached SentenceTransformer model, loading it on first call.

    The model is loaded once per Python process. If this is the first call in
    a fresh environment, sentence-transformers will download the model weights
    from the Hugging Face Hub (~90 MB) and cache them locally in
    ~/.cache/torch/sentence_transformers or SENTENCE_TRANSFORMERS_HOME.
    Subsequent calls use the cached weights without re-downloading.

    Returns:
        SentenceTransformer instance for 'all-MiniLM-L6-v2'.
    """
    global _MODEL, _MODEL_NAME
    if _MODEL is None:
        # Import here so the module can be imported without triggering model load.
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _MODEL_NAME = EMBEDDING_MODEL_NAME
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Compute normalized float32 embeddings for a list of strings.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIMENSION), dtype float32.
        Embeddings are L2-normalized (unit vectors) — consistent with cosine
        similarity search in ChromaDB.

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("embed_texts received an empty list; nothing to embed.")
    model = _get_model()
    # normalize_embeddings=True produces unit-length vectors.
    # Converting to float32 ensures consistent dtype for Chroma.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Compute a normalized float32 embedding for a single query string.

    Uses the exact same model and normalization as embed_texts so that cosine
    similarity between query and indexed chunk embeddings is meaningful.

    Args:
        query: A non-empty query string.

    Returns:
        numpy array of shape (EMBEDDING_DIMENSION,), dtype float32.

    Raises:
        ValueError: If query is empty or whitespace only.
    """
    stripped = query.strip()
    if not stripped:
        raise ValueError("embed_query received an empty or whitespace-only query.")
    result = embed_texts([stripped])
    return result[0]


def get_model_metadata() -> dict:
    """Return embedding configuration metadata for recording alongside an index.

    This dict must be stored when an index is built and checked when an existing
    index is reused, so that incompatible (different model or dimension) reuse
    is caught and reported rather than silently producing wrong results.
    """
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
    }
