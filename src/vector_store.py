"""Vector store module for the Personal AI Knowledge Assistant.

Wraps a ChromaDB PersistentClient collection. Handles:
- Building and persisting a fresh collection from chunk records and embeddings.
- Recording embedding model and chunk settings alongside the index.
- Rejecting incompatible reuse of an existing index (different model or dimension).
- Querying by a pre-computed embedding vector.
- Clearing a collection for an explicit rebuild.

Per ARCHITECTURE.md:
- For this corpus size, use a full rebuild (not incremental upsert) on corpus change.
- Activate the new collection only after a successful build.
- Preserve the last usable index on failure.
- Do NOT automatically rebuild at app startup.
- Do NOT use Chroma's default embedding function — pass explicitly computed embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.chunking import ChunkRecord
from src.config import PATHS, DEFAULT_TOP_K
from src.embeddings import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

COLLECTION_NAME = "knowledge_chunks"
COLLECTION_STAGING = "knowledge_chunks_staging"
COLLECTION_BACKUP = "knowledge_chunks_backup"

# Metadata stored in Chroma collection to detect incompatible index reuse
_META_MODEL_KEY = "embedding_model"
_META_DIM_KEY = "embedding_dimension"
_META_NORMALIZED_KEY = "embedding_normalized"
_META_METRIC_KEY = "distance_metric"
_META_CHUNK_SIZE_KEY = "chunk_size"
_META_CHUNK_OVERLAP_KEY = "chunk_overlap"

DISTANCE_METRIC = "cosine"
EMBEDDING_NORMALIZED = True


def _get_client(storage_dir: Optional[Path] = None) -> chromadb.PersistentClient:
    """Return a ChromaDB PersistentClient for the configured storage directory."""
    path = storage_dir or PATHS.storage_dir
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def recover_orphaned_collections(client: chromadb.PersistentClient) -> None:
    """Handle leftover collections from an interrupted process without data loss.

    Recovery logic:
    1. If COLLECTION_NAME is missing, but COLLECTION_BACKUP exists:
       An interrupted run crashed after backing up the active index or during promotion.
       Restore COLLECTION_BACKUP -> COLLECTION_NAME immediately so the usable
       index is restored.
    2. If COLLECTION_NAME exists and COLLECTION_BACKUP also exists:
       Promotion had succeeded before an interruption occurred during backup cleanup.
       Safely delete COLLECTION_BACKUP.
    3. If COLLECTION_STAGING exists:
       Staging is an incomplete unpromoted draft. Safely delete COLLECTION_STAGING.
    """
    colls = {c.name: c for c in client.list_collections()}
    has_active = COLLECTION_NAME in colls
    has_backup = COLLECTION_BACKUP in colls
    has_staging = COLLECTION_STAGING in colls

    if not has_active and has_backup:
        # Crucial: Backup is the ONLY usable index. Restore it!
        colls[COLLECTION_BACKUP].modify(name=COLLECTION_NAME)
        has_active = True
        has_backup = False

    if has_active and has_backup:
        try:
            client.delete_collection(COLLECTION_BACKUP)
        except Exception:
            pass

    if has_staging:
        try:
            client.delete_collection(COLLECTION_STAGING)
        except Exception:
            pass


def build_index(
    chunks: list[ChunkRecord],
    embeddings,               # np.ndarray shape (len(chunks), dim)
    chunk_size: int,
    chunk_overlap: int,
    storage_dir: Optional[Path] = None,
) -> dict:
    """Build (or replace) the persistent Chroma collection from chunk records.

    Uses a recoverable replacement strategy:
    1. Validates inputs and embedding shapes completely before making any storage changes.
    2. Recovers any orphaned collections from an earlier interrupted run.
    3. Fully populates and validates a temporary staging collection (`knowledge_chunks_staging`).
       If staging fails, staging is deleted and active index remains untouched.
    4. Preserves the current active collection as a backup (`knowledge_chunks_backup`)
       before promotion.
    5. Promotes staging to the active name (`knowledge_chunks`).
    6. Verifies the promoted collection.
    7. Removes the backup only after successful verification.
    8. If promotion or verification fails, restores the previous active collection
       from backup.

    NOTE ON CONCURRENCY AND ATOMICITY:
    This design is tailored for a single-user local application.
    Replacement is recoverable, but NOT atomic across all steps.
    Queries should not run concurrently with rebuilding.

    Args:
        chunks: List of ChunkRecord objects produced by chunking.py.
        embeddings: numpy float32 array of shape (len(chunks), embedding_dim).
        chunk_size: The chunk_size used during chunking (recorded in metadata).
        chunk_overlap: The chunk_overlap used during chunking (recorded in metadata).
        storage_dir: Override the default storage path (useful for tests).

    Returns:
        Summary dict with collection name, chunk count, model, and storage path.

    Raises:
        ValueError: If chunks is empty or embeddings shape is incompatible.
        RuntimeError: If staging population, promotion, or verification fails.
    """
    import numpy as np

    # Step 1: Pre-validation of all inputs before any disk/collection mutation
    if not chunks:
        raise ValueError("build_index received an empty chunk list. Nothing to index.")

    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.shape != (len(chunks), EMBEDDING_DIMENSION):
        raise ValueError(
            f"Embeddings shape {embeddings.shape} does not match "
            f"expected ({len(chunks)}, {EMBEDDING_DIMENSION})."
        )

    client = _get_client(storage_dir)

    # Clean up or recover any orphaned collections from prior interrupted runs
    recover_orphaned_collections(client)

    collection_metadata = {
        _META_MODEL_KEY: EMBEDDING_MODEL_NAME,
        _META_DIM_KEY: EMBEDDING_DIMENSION,
        _META_NORMALIZED_KEY: EMBEDDING_NORMALIZED,
        _META_METRIC_KEY: DISTANCE_METRIC,
        _META_CHUNK_SIZE_KEY: chunk_size,
        _META_CHUNK_OVERLAP_KEY: chunk_overlap,
        "hnsw:space": DISTANCE_METRIC,
    }

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source": c.source,
            "page": c.page if c.page is not None else -1,  # Chroma requires scalar
            "chunk_index": c.chunk_index,
            "word_count": c.word_count,
            "content_hash": c.content_hash,
            **{k: str(v) for k, v in c.metadata.items()},  # ensure all values are strings
        }
        for c in chunks
    ]

    # Step 2: Build and populate a staging collection
    try:
        staging_collection = client.create_collection(
            name=COLLECTION_STAGING,
            metadata=collection_metadata,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create staging collection: {exc}. Active index preserved."
        ) from exc

    try:
        staging_collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )
        if staging_collection.count() != len(chunks):
            raise RuntimeError(
                f"Staging collection chunk count ({staging_collection.count()}) "
                f"does not match expected ({len(chunks)})."
            )
    except Exception as exc:
        # Failure during insertion: delete staging collection; existing index is preserved
        try:
            client.delete_collection(COLLECTION_STAGING)
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to populate staging collection during index rebuild: {exc}. "
            "The existing usable index was preserved."
        ) from exc

    # Step 3: Preserve current active collection as backup before promotion
    has_prior_active = False
    try:
        active_coll = client.get_collection(COLLECTION_NAME)
        active_coll.modify(name=COLLECTION_BACKUP)
        has_prior_active = True
    except Exception:
        has_prior_active = False  # First build; no active collection existed

    # Step 4: Promote staging to active name
    try:
        staging_collection.modify(name=COLLECTION_NAME)
    except Exception as prom_exc:
        # Rollback: restore active from backup if it existed
        if has_prior_active:
            try:
                client.get_collection(COLLECTION_BACKUP).modify(name=COLLECTION_NAME)
            except Exception:
                pass
        try:
            client.delete_collection(COLLECTION_STAGING)
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to promote staging collection to '{COLLECTION_NAME}': {prom_exc}. "
            "Previous active index was restored from backup."
        ) from prom_exc

    # Step 5: Verify the promoted collection
    try:
        promoted = client.get_collection(COLLECTION_NAME)
        if promoted.count() != len(chunks):
            raise RuntimeError(
                f"Promoted chunk count ({promoted.count()}) does not match expected ({len(chunks)})."
            )
    except Exception as verif_exc:
        # Rollback: delete broken promoted collection and restore backup
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        if has_prior_active:
            try:
                client.get_collection(COLLECTION_BACKUP).modify(name=COLLECTION_NAME)
            except Exception:
                pass
        raise RuntimeError(
            f"Verification of promoted collection failed: {verif_exc}. "
            "Previous active index was restored from backup."
        ) from verif_exc

    # Step 6: Remove the backup only after successful verification
    if has_prior_active:
        try:
            client.delete_collection(COLLECTION_BACKUP)
        except Exception:
            pass

    return {
        "collection": COLLECTION_NAME,
        "chunk_count": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_normalized": EMBEDDING_NORMALIZED,
        "distance_metric": DISTANCE_METRIC,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "storage_dir": str(storage_dir or PATHS.storage_dir),
    }


def query_index(
    query_embedding,           # np.ndarray shape (dim,)
    top_k: int = DEFAULT_TOP_K,
    storage_dir: Optional[Path] = None,
) -> list[dict]:
    """Query the persistent Chroma collection with a pre-computed embedding.

    First verifies that the existing index is compatible with the active
    embedding configuration (model, dimension, normalization, and distance metric).

    Returns ranked passages ordered by ascending cosine distance
    (smaller = more similar). Distances are similarity measures, not
    confidence or probability scores.

    Args:
        query_embedding: float32 array of shape (EMBEDDING_DIMENSION,).
        top_k: Number of nearest neighbours to return. Must be >= 1.
        storage_dir: Override the default storage path (useful for tests).

    Returns:
        List of result dicts, each containing:
            chunk_id, text, source, page, chunk_index, distance, metadata.
        Ordered from most similar (lowest distance) to least.

    Raises:
        ValueError: If top_k < 1, index is missing/empty, or incompatible.
    """
    import numpy as np

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got: {top_k!r}")

    query_embedding = np.array(query_embedding, dtype=np.float32)
    if query_embedding.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"query_embedding shape {query_embedding.shape} does not match "
            f"expected ({EMBEDDING_DIMENSION},)."
        )

    # Confirm index exists and is compatible with current embedding configuration
    compatible, compat_msg = check_index_compatibility(storage_dir)
    if not compatible:
        raise ValueError(
            f"Cannot query incompatible index at '{storage_dir or PATHS.storage_dir}': "
            f"{compat_msg} Run ingest.py to rebuild the index."
        )

    client = _get_client(storage_dir)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        raise ValueError(
            f"No index found at '{storage_dir or PATHS.storage_dir}'. "
            "Run ingest.py to build the index before querying."
        )

    count = collection.count()
    if count == 0:
        raise ValueError(
            "The index exists but contains no chunks. "
            "Re-run ingest.py to populate the index."
        )

    # Clamp top_k to the number of stored chunks to avoid Chroma errors
    effective_k = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
        meta_dict = meta or {}
        page_raw = meta_dict.get("page", -1)
        passages.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": meta_dict.get("source", ""),
                "page": int(page_raw) if int(page_raw) != -1 else None,
                "chunk_index": meta_dict.get("chunk_index"),
                "distance": dist,   # smaller = more similar; NOT a confidence score
                "metadata": meta_dict,
            }
        )

    return passages


def check_index_compatibility(
    storage_dir: Optional[Path] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> tuple[bool, str]:
    """Check whether an existing index was built with the current configuration.

    Validates:
      1. Model name (must match EMBEDDING_MODEL_NAME).
      2. Embedding dimension (must match EMBEDDING_DIMENSION).
      3. Normalization setting (must match EMBEDDING_NORMALIZED).
      4. Distance metric (must match DISTANCE_METRIC).
      5. Chunk settings (if chunk_size/chunk_overlap provided, checks if settings changed).

    Returns:
        Tuple of (compatible: bool, message: str).
    """
    client = _get_client(storage_dir)
    recover_orphaned_collections(client)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return True, "No existing index found; a fresh build will be created."

    meta = collection.metadata or {}
    stored_model = meta.get(_META_MODEL_KEY, "")
    stored_dim = meta.get(_META_DIM_KEY, -1)
    stored_norm = meta.get(_META_NORMALIZED_KEY, None)
    stored_metric = meta.get(_META_METRIC_KEY, "")

    if stored_model != EMBEDDING_MODEL_NAME:
        return False, (
            f"Existing index was built with model '{stored_model}' but current "
            f"configuration uses '{EMBEDDING_MODEL_NAME}'. "
            "Re-run ingest.py to rebuild."
        )
    if int(stored_dim) != EMBEDDING_DIMENSION:
        return False, (
            f"Existing index has embedding dimension {stored_dim} but current "
            f"model produces {EMBEDDING_DIMENSION}. "
            "Re-run ingest.py to rebuild."
        )
    if stored_norm is not None and bool(stored_norm) != EMBEDDING_NORMALIZED:
        return False, (
            f"Existing index embedding_normalized={stored_norm} but current "
            f"configuration expects {EMBEDDING_NORMALIZED}. "
            "Re-run ingest.py to rebuild."
        )
    if stored_metric and stored_metric != DISTANCE_METRIC:
        return False, (
            f"Existing index distance_metric='{stored_metric}' but current "
            f"configuration expects '{DISTANCE_METRIC}'. "
            "Re-run ingest.py to rebuild."
        )

    # Check if chunk settings changed
    stored_chunk_size = meta.get(_META_CHUNK_SIZE_KEY)
    stored_chunk_overlap = meta.get(_META_CHUNK_OVERLAP_KEY)
    if chunk_size is not None and stored_chunk_size is not None and int(stored_chunk_size) != chunk_size:
        return False, (
            f"Existing index was built with chunk_size={stored_chunk_size}, but "
            f"current request uses chunk_size={chunk_size}. Re-run ingest.py to rebuild."
        )
    if chunk_overlap is not None and stored_chunk_overlap is not None and int(stored_chunk_overlap) != chunk_overlap:
        return False, (
            f"Existing index was built with chunk_overlap={stored_chunk_overlap}, but "
            f"current request uses chunk_overlap={chunk_overlap}. Re-run ingest.py to rebuild."
        )

    return True, (
        f"Index is compatible: model='{stored_model}', dim={stored_dim}, "
        f"normalized={stored_norm}, metric='{stored_metric}', "
        f"chunk_size={stored_chunk_size}, chunk_overlap={stored_chunk_overlap}."
    )
