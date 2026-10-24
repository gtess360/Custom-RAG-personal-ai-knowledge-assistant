r"""Corpus ingestion script for the Personal AI Knowledge Assistant.

Reads all supported documents from data/documents/, chunks them using the
documented word-based settings, computes embeddings with all-MiniLM-L6-v2,
and writes a persistent ChromaDB index to storage/chroma.

Usage:
    .\.venv\Scripts\python.exe ingest.py                  # Use defaults
    .\.venv\Scripts\python.exe ingest.py --force-rebuild  # Rebuild even if compatible

FIRST-RUN NOTE
--------------
On the first run (and any time the model cache is empty), sentence-transformers
will download the 'all-MiniLM-L6-v2' model weights from the Hugging Face Hub
(~90 MB). The download happens automatically to:
    Windows: %USERPROFILE%\.cache\torch\sentence_transformers
After the first run, the weights are cached locally and no download occurs.
You can pre-set SENTENCE_TRANSFORMERS_HOME to a custom cache directory.

POLICY
------
This script performs a FULL REBUILD every run:
- Reads fresh from data/documents/
- Deletes the old collection before writing the new one
- Checks embedding model compatibility before overwriting

Do NOT run this script as part of the test suite.
Tests create their own temporary indices using controlled fixtures.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Allow running from the project root without src/ on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PATHS
from src.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from src.embeddings import EMBEDDING_MODEL_NAME, embed_texts
from src.loaders import load_documents_dir
from src.vector_store import (
    COLLECTION_NAME,
    _get_client,
    build_index,
    check_index_compatibility,
)


def ingest_corpus(
    force_rebuild: bool = False,
    allow_partial_ingestion: bool = False,
    allow_empty: bool = False,
    data_dir: Optional[Path] = None,
    storage_dir: Optional[Path] = None,
) -> dict:
    """Run the 5-step corpus ingestion pipeline.

    Args:
        force_rebuild: Rebuild even if existing index is incompatible.
        allow_partial_ingestion: Proceed even if some files fail to load.
        allow_empty: Allow replacing a non-empty index with an empty corpus.
        data_dir: Directory to read documents from (defaults to PATHS.data_dir).
        storage_dir: Directory to store ChromaDB index (defaults to PATHS.storage_dir).

    Returns:
        Summary dict of the built index.

    Raises:
        ValueError: If compatibility check fails or corpus is empty without permission.
        RuntimeError: If file load errors occur without allow_partial_ingestion.
    """
    target_data_dir = data_dir or PATHS.data_dir
    target_storage_dir = storage_dir or PATHS.storage_dir

    print("=" * 60)
    print("Personal AI Knowledge Assistant — Corpus Ingestion")
    print("=" * 60)

    # 1. Compatibility check
    compatible, compat_msg = check_index_compatibility(storage_dir=target_storage_dir)
    print(f"\n[1/5] Index compatibility check: {compat_msg}")
    if not compatible:
        if not force_rebuild:
            raise ValueError(
                f"Existing index is incompatible: {compat_msg}. "
                "Pass force_rebuild=True or --force-rebuild to rebuild."
            )
        print("[INFO] --force-rebuild specified; proceeding with rebuild.")

    # 2. Load documents
    print(f"\n[2/5] Loading documents from: {target_data_dir}")
    load_result = load_documents_dir(target_data_dir)
    records = load_result.records
    errors = load_result.errors
    warnings = load_result.warnings

    if warnings:
        print(f"  Page-level warnings ({len(warnings)} skipped/textless page(s)):")
        for warn in warnings:
            print(f"    ~ {warn}")

    # Guard: Require explicit action before replacing nonempty index with empty corpus
    if not records:
        client = _get_client(target_storage_dir)
        try:
            coll = client.get_collection(COLLECTION_NAME)
            existing_count = coll.count()
        except Exception:
            existing_count = 0

        if existing_count > 0 and not allow_empty:
            raise RuntimeError(
                f"Attempting to replace a non-empty index ({existing_count} chunks) "
                "with an empty corpus! Refusing to wipe the existing index. "
                "Pass --allow-empty to explicitly clear the index."
            )
        if not allow_empty:
            raise ValueError("No documents were loaded and corpus is empty. Aborting.")

        # If allow_empty is True, clear existing collection cleanly
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return {
            "collection": COLLECTION_NAME,
            "chunk_count": 0,
            "storage_dir": str(target_storage_dir),
        }

    if errors:
        print(f"  File-loading errors ({len(errors)} file(s)):")
        for err in errors:
            print(f"    ! {err}")
        if not allow_partial_ingestion:
            raise RuntimeError(
                f"Loading produced {len(errors)} file-level failure(s). "
                "Aborting rebuild to preserve the existing usable index. "
                "Pass --allow-partial-ingestion to proceed with readable documents."
            )
        print("  [WARN] --allow-partial-ingestion specified; proceeding with readable documents.")

    print(f"  Loaded {len(records)} document record(s) (pages) from disk.")

    # 3. Chunk documents
    print(
        f"\n[3/5] Chunking (chunk_size={DEFAULT_CHUNK_SIZE}, "
        f"chunk_overlap={DEFAULT_CHUNK_OVERLAP}) ..."
    )
    chunks = chunk_documents(
        records, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    print(f"  Produced {len(chunks)} chunk(s).")

    # 4. Compute embeddings
    print(f"\n[4/5] Computing embeddings with '{EMBEDDING_MODEL_NAME}' ...")
    print(
        "  NOTE: On first run, the model weights will be downloaded from "
        "Hugging Face Hub (~90 MB)."
    )
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  Embedding array shape: {embeddings.shape}")

    # 5. Build index (recoverable staged replacement)
    print(f"\n[5/5] Writing ChromaDB collection '{COLLECTION_NAME}' to {target_storage_dir} ...")
    summary = build_index(
        chunks=chunks,
        embeddings=embeddings,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        storage_dir=target_storage_dir,
    )
    print("\nIngestion complete.")
    print(f"  Chunks indexed  : {summary['chunk_count']}")
    print(f"  Embedding model : {summary['embedding_model']}")
    print(f"  Embedding dim   : {summary['embedding_dimension']}")
    print(f"  Normalized      : {summary['embedding_normalized']}")
    print(f"  Distance metric : {summary['distance_metric']}")
    print(f"  Chunk size      : {summary['chunk_size']}")
    print(f"  Chunk overlap   : {summary['chunk_overlap']}")
    print(f"  Storage path    : {summary['storage_dir']}")
    print("\nThe index is ready. You can now run retrieval queries.")
    return summary


def main(
    force_rebuild: bool = False,
    allow_partial_ingestion: bool = False,
    allow_empty: bool = False,
) -> None:
    try:
        ingest_corpus(
            force_rebuild=force_rebuild,
            allow_partial_ingestion=allow_partial_ingestion,
            allow_empty=allow_empty,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into ChromaDB index.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if an incompatible index already exists.",
    )
    parser.add_argument(
        "--allow-partial-ingestion",
        action="store_true",
        help="Proceed with ingestion even if some files fail to load.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow replacing a non-empty index with an empty corpus.",
    )
    args = parser.parse_args()
    main(
        force_rebuild=args.force_rebuild,
        allow_partial_ingestion=args.allow_partial_ingestion,
        allow_empty=args.allow_empty,
    )
