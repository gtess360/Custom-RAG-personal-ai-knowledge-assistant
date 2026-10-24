"""Tests for Phase 3: Embeddings, vector store, and retrieval.

Isolation:
- Routine unit tests (all tests NOT marked with @pytest.mark.slow) use a
  controlled embedding fixture that produces deterministic fake embeddings
  WITHOUT loading the real sentence-transformers model. They are fast (~0.5s)
  and fully offline.
- The smoke test (test_real_model_smoke) is marked @pytest.mark.slow and
  loads the actual all-MiniLM-L6-v2 model. It requires an internet connection
  on first run to download model weights (~90 MB). Run it explicitly:
      pytest -v -m slow
  It is excluded from the default test run (see pytest.ini: addopts or markers).
- All tests use pytest's tmp_path fixture for storage; the real
  storage/chroma directory is never touched.
"""

import numpy as np
import pytest
import chromadb

from src.chunking import ChunkRecord
from src.embeddings import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME, embed_query, get_model_metadata
from src.vector_store import (
    COLLECTION_NAME,
    build_index,
    check_index_compatibility,
    query_index,
)
from src.retrieval import format_evidence_for_display, retrieve, validate_query


# ---------------------------------------------------------------------------
# Controlled embedding fixture (no real model)
# ---------------------------------------------------------------------------

def _fake_embeddings(n: int, seed: int = 42) -> np.ndarray:
    """Return reproducible unit-norm float32 embeddings without the real model."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, EMBEDDING_DIMENSION)).astype(np.float32)
    # L2-normalize each row to match real model output
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / norms).astype(np.float32)


def _make_chunk(chunk_id: str, text: str, source: str, page=None, chunk_index: int = 0) -> ChunkRecord:
    """Construct a ChunkRecord for testing without going through the full pipeline."""
    import hashlib
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    return ChunkRecord(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page=page,
        chunk_index=chunk_index,
        content_hash=content_hash,
        word_count=len(text.split()),
        metadata={"total_chunks": 1},
    )


# ---------------------------------------------------------------------------
# src/embeddings.py unit tests
# ---------------------------------------------------------------------------

def test_get_model_metadata_returns_correct_keys():
    """get_model_metadata must return the model name and dimension without loading the model."""
    meta = get_model_metadata()
    assert meta["embedding_model"] == EMBEDDING_MODEL_NAME
    assert meta["embedding_dimension"] == EMBEDDING_DIMENSION
    assert EMBEDDING_DIMENSION == 384  # all-MiniLM-L6-v2 fixed dimension


def test_embed_query_raises_on_empty_string():
    """embed_query must reject empty and whitespace-only strings before touching the model."""
    with pytest.raises(ValueError) as exc:
        embed_query("")
    assert "empty" in str(exc.value).lower()

    with pytest.raises(ValueError) as exc:
        embed_query("   ")
    assert "empty" in str(exc.value).lower()


def test_validate_query_strips_and_rejects_empty():
    """validate_query returns stripped text and raises on empty/whitespace."""
    assert validate_query("  What is the project?  ") == "What is the project?"

    with pytest.raises(ValueError):
        validate_query("")

    with pytest.raises(ValueError):
        validate_query("   \n\t  ")


# ---------------------------------------------------------------------------
# src/vector_store.py unit tests (offline — fake embeddings)
# ---------------------------------------------------------------------------

def test_build_index_and_query_roundtrip(tmp_path):
    """build_index stores chunks and query_index retrieves them correctly."""
    chunks = [
        _make_chunk("c1", "Alex Mercer leads the fraud detection project Aurora.", "profile.md"),
        _make_chunk("c2", "Project Nexus is an internal RAG assistant.", "nexus.md"),
        _make_chunk("c3", "Project Beacon defines the model evaluation framework.", "beacon.md"),
    ]
    embs = _fake_embeddings(3, seed=42)

    summary = build_index(chunks, embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    assert summary["chunk_count"] == 3
    assert summary["embedding_model"] == EMBEDDING_MODEL_NAME

    # Query with the first chunk's embedding — it should come back as closest
    results = query_index(embs[0], top_k=2, storage_dir=tmp_path)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert "Alex Mercer" in results[0]["text"]
    assert results[0]["distance"] <= results[1]["distance"]


def test_query_index_returns_all_metadata_fields(tmp_path):
    """Returned passages must include source, page, chunk_index, distance, metadata."""
    chunk = _make_chunk("meta_c1", "Metadata completeness test text.", "spec.pdf", page=3)
    emb = _fake_embeddings(1, seed=7)

    build_index([chunk], emb, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)
    results = query_index(emb[0], top_k=1, storage_dir=tmp_path)

    assert len(results) == 1
    r = results[0]
    assert r["chunk_id"] == "meta_c1"
    assert r["source"] == "spec.pdf"
    assert r["page"] == 3
    assert r["chunk_index"] == 0
    assert "distance" in r
    assert isinstance(r["distance"], float)
    assert "metadata" in r


def test_query_distance_is_not_labelled_as_confidence(tmp_path):
    """Result dicts must include 'distance' key (not 'confidence' or 'score')."""
    chunks = [_make_chunk("d1", "Some text here.", "doc.md")]
    emb = _fake_embeddings(1)
    build_index(chunks, emb, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)
    results = query_index(emb[0], top_k=1, storage_dir=tmp_path)
    assert "distance" in results[0]
    assert "confidence" not in results[0]
    assert "score" not in results[0]


def test_build_index_rejects_empty_chunks(tmp_path):
    """build_index must raise ValueError when given an empty chunk list."""
    embs = _fake_embeddings(0)
    with pytest.raises(ValueError) as exc:
        build_index([], embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)
    assert "empty" in str(exc.value).lower()


def test_build_index_rejects_mismatched_embedding_shape(tmp_path):
    """build_index must raise ValueError if embedding dimensions don't match expectations."""
    chunks = [_make_chunk("shape_c1", "Test text.", "doc.md")]
    wrong_emb = np.zeros((1, 128), dtype=np.float32)  # Wrong dimension
    with pytest.raises(ValueError) as exc:
        build_index(chunks, wrong_emb, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)
    assert "shape" in str(exc.value).lower() or "dimension" in str(exc.value).lower()


def test_query_index_raises_if_no_index(tmp_path):
    """query_index must raise a descriptive ValueError when no index exists."""
    query_emb = _fake_embeddings(1)[0]
    with pytest.raises(ValueError) as exc:
        query_index(query_emb, top_k=3, storage_dir=tmp_path)
    assert "ingest" in str(exc.value).lower() or "no index" in str(exc.value).lower()


def test_query_index_invalid_top_k(tmp_path):
    """query_index must reject top_k < 1 before touching the index."""
    query_emb = _fake_embeddings(1)[0]
    with pytest.raises(ValueError) as exc:
        query_index(query_emb, top_k=0, storage_dir=tmp_path)
    assert "positive integer" in str(exc.value).lower()


def test_build_index_full_replace(tmp_path):
    """A second build_index call replaces the first — old chunks must not appear."""
    chunk_v1 = _make_chunk("old_chunk", "Old content from version one.", "old.md")
    emb_v1 = _fake_embeddings(1, seed=1)
    build_index([chunk_v1], emb_v1, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    chunk_v2 = _make_chunk("new_chunk", "New content from version two.", "new.md")
    emb_v2 = _fake_embeddings(1, seed=2)
    build_index([chunk_v2], emb_v2, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    results = query_index(emb_v2[0], top_k=3, storage_dir=tmp_path)
    ids = [r["chunk_id"] for r in results]
    assert "new_chunk" in ids
    assert "old_chunk" not in ids, "Old chunk must not persist after full rebuild."


def test_check_index_compatibility_no_index(tmp_path):
    """check_index_compatibility returns (True, message) when no index exists."""
    compatible, msg = check_index_compatibility(storage_dir=tmp_path)
    assert compatible is True
    assert "no existing index" in msg.lower() or "fresh" in msg.lower()


def test_check_index_compatibility_after_build(tmp_path):
    """A freshly-built index must be reported as compatible."""
    chunks = [_make_chunk("compat_c1", "Compatibility check text.", "doc.md")]
    emb = _fake_embeddings(1)
    build_index(chunks, emb, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    compatible, msg = check_index_compatibility(storage_dir=tmp_path)
    assert compatible is True
    assert EMBEDDING_MODEL_NAME in msg


def test_top_k_clamped_to_collection_size(tmp_path):
    """If top_k > number of stored chunks, all chunks are returned without error."""
    chunks = [_make_chunk(f"clamp_{i}", f"Chunk content {i}.", "doc.md", chunk_index=i)
              for i in range(2)]
    embs = _fake_embeddings(2)
    build_index(chunks, embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    # Request more results than exist
    results = query_index(embs[0], top_k=10, storage_dir=tmp_path)
    assert len(results) == 2  # Clamped, not an error


# ---------------------------------------------------------------------------
# src/retrieval.py unit tests (offline — monkeypatched embedding)
# ---------------------------------------------------------------------------

def test_retrieve_returns_ranked_passages(tmp_path, monkeypatch):
    """retrieve() must return passages ordered by distance (lowest first)."""
    chunks = [
        _make_chunk("r1", "Alex Mercer is the primary candidate.", "profile.md"),
        _make_chunk("r2", "Project Aurora detects fraud with 94% precision.", "aurora.md"),
    ]
    embs = _fake_embeddings(2, seed=99)
    build_index(chunks, embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    # Patch embed_query to return the first chunk's embedding deterministically
    monkeypatch.setattr("src.retrieval.embed_query", lambda q: embs[0])
    results = retrieve("Who is the candidate?", top_k=2, storage_dir=tmp_path)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "r1"  # Most similar first
    assert results[0]["distance"] <= results[1]["distance"]


def test_retrieve_raises_on_empty_question(tmp_path, monkeypatch):
    """retrieve() must reject empty questions before calling the embedding model."""
    monkeypatch.setattr("src.retrieval.embed_query", lambda q: _fake_embeddings(1)[0])
    with pytest.raises(ValueError):
        retrieve("", storage_dir=tmp_path)


def test_retrieve_raises_when_index_empty(tmp_path, monkeypatch):
    """retrieve() must propagate a descriptive error when no index exists."""
    monkeypatch.setattr("src.retrieval.embed_query", lambda q: _fake_embeddings(1)[0])
    with pytest.raises(ValueError) as exc:
        retrieve("Any question?", storage_dir=tmp_path)
    assert "ingest" in str(exc.value).lower() or "index" in str(exc.value).lower()


def test_format_evidence_no_results():
    """format_evidence_for_display must handle empty passage list gracefully."""
    output = format_evidence_for_display([])
    assert "no passages" in output.lower()


def test_format_evidence_labels_distance_not_confidence(tmp_path):
    """Formatted output must NOT use the word 'confidence' for the distance value."""
    passage = {
        "chunk_id": "c1", "text": "Sample text.", "source": "doc.md",
        "page": None, "chunk_index": 0, "distance": 0.123, "metadata": {}
    }
    output = format_evidence_for_display([passage])
    assert "confidence" not in output.lower()
    assert "0.1230" in output or "distance" in output.lower()


def test_format_evidence_truncates_long_text():
    """Very long chunk text must be truncated and marked [TRUNCATED]."""
    long_text = "word " * 600  # ~3000 chars
    passage = {
        "chunk_id": "c1", "text": long_text, "source": "doc.md",
        "page": 1, "chunk_index": 0, "distance": 0.05, "metadata": {}
    }
    output = format_evidence_for_display([passage], max_chars_per_chunk=400)
    assert "[TRUNCATED]" in output


# ---------------------------------------------------------------------------
# Rebuild failure handling & Compatibility tests
# ---------------------------------------------------------------------------

def test_build_index_preserves_existing_index_on_insertion_failure(tmp_path, monkeypatch):
    """If insertion into the staging collection fails, the existing usable index
    must remain completely intact and searchable."""
    # 1. Build an initial valid index with 2 chunks
    initial_chunks = [
        _make_chunk("init_1", "Initial document one.", "doc1.md"),
        _make_chunk("init_2", "Initial document two.", "doc2.md"),
    ]
    initial_embs = _fake_embeddings(2, seed=10)
    build_index(initial_chunks, initial_embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    # Verify initial index has 2 chunks
    res_before = query_index(initial_embs[0], top_k=5, storage_dir=tmp_path)
    assert len(res_before) == 2

    # 2. Attempt a rebuild where insertion fails (simulate disk error / Chroma failure)
    new_chunks = [_make_chunk("new_1", "Brand new document.", "new.md")]
    new_embs = _fake_embeddings(1, seed=20)

    # Monkeypatch the Collection.add method to simulate a failure during staging insertion
    from chromadb.api.models.Collection import Collection
    original_add = Collection.add

    def failing_add(self, *args, **kwargs):
        raise RuntimeError("Simulated failure during collection insertion")

    monkeypatch.setattr(Collection, "add", failing_add)

    with pytest.raises(RuntimeError) as exc:
        build_index(new_chunks, new_embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    assert "preserved" in str(exc.value).lower()

    # Restore add method
    monkeypatch.setattr(Collection, "add", original_add)

    # 3. Verify the original index was preserved intact and is still searchable
    res_after = query_index(initial_embs[0], top_k=5, storage_dir=tmp_path)
    assert len(res_after) == 2
    chunk_ids = [r["chunk_id"] for r in res_after]
    assert "init_1" in chunk_ids
    assert "init_2" in chunk_ids


def test_query_index_rejects_incompatible_embedding_model(tmp_path):
    """query_index must refuse to query an index built with an incompatible model."""
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path))
    # Create collection with incompatible model metadata
    coll = client.create_collection(
        name="knowledge_chunks",
        metadata={
            "embedding_model": "text-embedding-ada-002",
            "embedding_dimension": 384,
            "embedding_normalized": True,
            "distance_metric": "cosine",
            "chunk_size": 120,
            "chunk_overlap": 25,
        },
    )
    coll.add(ids=["bad_1"], documents=["test text"], embeddings=[[0.1] * 384])

    query_emb = _fake_embeddings(1)[0]
    with pytest.raises(ValueError) as exc:
        query_index(query_emb, top_k=1, storage_dir=tmp_path)
    assert "incompatible" in str(exc.value).lower()
    assert "text-embedding-ada-002" in str(exc.value)


def test_check_index_compatibility_chunk_setting_change(tmp_path):
    """check_index_compatibility detects changes to chunk_size or chunk_overlap."""
    chunks = [_make_chunk("c1", "Chunk setting test.", "doc.md")]
    embs = _fake_embeddings(1)
    build_index(chunks, embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    # Same settings -> compatible
    compat, _ = check_index_compatibility(storage_dir=tmp_path, chunk_size=120, chunk_overlap=25)
    assert compat is True

    # Changed chunk_size -> incompatible, instructs rebuild
    compat_size, msg_size = check_index_compatibility(storage_dir=tmp_path, chunk_size=200, chunk_overlap=25)
    assert compat_size is False
    assert "chunk_size" in msg_size

    # Changed chunk_overlap -> incompatible, instructs rebuild
    compat_overlap, msg_overlap = check_index_compatibility(storage_dir=tmp_path, chunk_size=120, chunk_overlap=50)
    assert compat_overlap is False
    assert "chunk_overlap" in msg_overlap


def test_load_documents_dir_distinguishes_page_warnings_from_file_failures(tmp_path):
    """load_documents_dir isolates file-level failures from page-level warnings."""
    from src.loaders import load_documents_dir
    from tests.test_loaders import _create_minimal_pdf_with_text

    # 1. Valid markdown file
    (tmp_path / "valid.md").write_text("Valid text.", encoding="utf-8")

    # 2. Corrupt file that causes a file-level error
    (tmp_path / "corrupt.pdf").write_bytes(b"NOT A REAL PDF CONTENT \x00\xff")

    # 3. Valid multi-page PDF with readable page 1 and empty page 2
    pdf_path = tmp_path / "mixed.pdf"
    _create_minimal_pdf_with_text(pdf_path, ["Readable page one content", ""])

    result = load_documents_dir(tmp_path, root=tmp_path)

    # Test backward-compatible unpacking
    records, errors = result
    assert isinstance(result.warnings, list)
    assert isinstance(result.errors, list)

    # File loading error contains the corrupt file
    assert any("corrupt.pdf" in err for err in result.errors)
    # Page warning contains the empty page report
    assert any("page 2" in warn.lower() for warn in result.warnings)
    # The two lists are distinct
    assert not any("page 2" in err.lower() for err in result.errors)
    # Valid records were loaded despite the corrupt file and empty page
    assert len(records) >= 2  # valid.md + mixed.pdf page 1


def test_promotion_failure_restores_previous_index(tmp_path, monkeypatch):
    """If promotion of staging to active fails, the previous active index
    must be restored from backup."""
    # 1. Build an initial valid index with 2 chunks
    initial_chunks = [
        _make_chunk("p_init_1", "Initial document one.", "doc1.md"),
        _make_chunk("p_init_2", "Initial document two.", "doc2.md"),
    ]
    initial_embs = _fake_embeddings(2, seed=30)
    build_index(initial_chunks, initial_embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    # 2. Attempt a rebuild where promotion (rename to knowledge_chunks) fails
    new_chunks = [_make_chunk("p_new_1", "Brand new content.", "new.md")]
    new_embs = _fake_embeddings(1, seed=40)

    from chromadb.api.models.Collection import Collection
    original_modify = Collection.modify

    def failing_modify(self, name=None, metadata=None):
        if name == COLLECTION_NAME and self.name == "knowledge_chunks_staging":
            raise RuntimeError("Simulated failure during promotion of staging collection")
        return original_modify(self, name=name, metadata=metadata)

    monkeypatch.setattr(Collection, "modify", failing_modify)

    with pytest.raises(RuntimeError) as exc:
        build_index(new_chunks, new_embs, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    assert "restored from backup" in str(exc.value).lower()

    # Restore modify
    monkeypatch.setattr(Collection, "modify", original_modify)

    # 3. Verify the previous active index was restored from backup and is searchable
    res = query_index(initial_embs[0], top_k=5, storage_dir=tmp_path)
    assert len(res) == 2
    ids = [r["chunk_id"] for r in res]
    assert "p_init_1" in ids
    assert "p_init_2" in ids


def test_recovery_from_interrupted_replacement(tmp_path):
    """If a previous run was interrupted leaving only knowledge_chunks_backup,
    it must be recovered rather than deleted or lost."""
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path))

    # Simulate an interrupted crash state: active is gone, but backup exists
    backup_coll = client.create_collection(
        name="knowledge_chunks_backup",
        metadata={
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "embedding_normalized": True,
            "distance_metric": "cosine",
            "chunk_size": 120,
            "chunk_overlap": 25,
        },
    )
    backup_coll.add(
        ids=["saved_chunk_1", "saved_chunk_2"],
        documents=["Saved text 1", "Saved text 2"],
        embeddings=[[0.05] * 384, [0.06] * 384],
    )

    # Ensure COLLECTION_NAME does not exist initially
    existing = [c.name for c in client.list_collections()]
    assert "knowledge_chunks" not in existing
    assert "knowledge_chunks_backup" in existing

    # Running check_index_compatibility or query_index triggers recovery
    compatible, msg = check_index_compatibility(storage_dir=tmp_path)
    assert compatible is True

    # Confirm knowledge_chunks was restored and backup was cleaned up
    collections_after = [c.name for c in client.list_collections()]
    assert "knowledge_chunks" in collections_after
    assert "knowledge_chunks_backup" not in collections_after

    # Confirm the chunks are searchable
    results = query_index([0.05] * 384, top_k=5, storage_dir=tmp_path)
    assert len(results) == 2
    assert {r["chunk_id"] for r in results} == {"saved_chunk_1", "saved_chunk_2"}


def test_refuse_empty_replacement_without_permission(tmp_path, monkeypatch):
    """ingest_corpus must refuse to replace a non-empty index with an empty corpus
    unless allow_empty=True is explicitly passed."""
    import ingest
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: _fake_embeddings(len(texts)))

    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()

    # 1. Build an initial non-empty index
    (data_dir / "doc.md").write_text("Valid text content.", encoding="utf-8")
    summary = ingest.ingest_corpus(data_dir=data_dir, storage_dir=storage_dir)
    assert summary["chunk_count"] > 0

    # 2. Empty the documents directory
    (data_dir / "doc.md").unlink()

    # Attempting to ingest empty corpus without permission must raise
    with pytest.raises(RuntimeError) as exc:
        ingest.ingest_corpus(data_dir=data_dir, storage_dir=storage_dir, allow_empty=False)
    assert "refusing to wipe" in str(exc.value).lower() or "empty corpus" in str(exc.value).lower()

    # The existing index is preserved intact
    client = chromadb.PersistentClient(path=str(storage_dir))
    coll = client.get_collection(COLLECTION_NAME)
    assert coll.count() > 0

    # 3. Explicit permission allows empty replacement
    empty_summary = ingest.ingest_corpus(data_dir=data_dir, storage_dir=storage_dir, allow_empty=True)
    assert empty_summary["chunk_count"] == 0


def test_abort_ingestion_on_file_errors_unless_partial(tmp_path, monkeypatch):
    """ingest_corpus must abort on file errors unless allow_partial_ingestion=True."""
    import ingest
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: _fake_embeddings(len(texts)))

    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()

    # Create one good file and one corrupted PDF
    (data_dir / "good.md").write_text("Good document text.", encoding="utf-8")
    (data_dir / "bad.pdf").write_bytes(b"CORRUPT GARBAGE NOT A PDF \x00\xff")

    # Without allow_partial_ingestion -> raises RuntimeError
    with pytest.raises(RuntimeError) as exc:
        ingest.ingest_corpus(data_dir=data_dir, storage_dir=storage_dir, allow_partial_ingestion=False)
    assert "aborting rebuild" in str(exc.value).lower()

    # With allow_partial_ingestion=True -> proceeds and indexes good file
    summary = ingest.ingest_corpus(data_dir=data_dir, storage_dir=storage_dir, allow_partial_ingestion=True)
    assert summary["chunk_count"] > 0

def test_real_model_smoke(tmp_path):
    """Load the actual all-MiniLM-L6-v2 model, embed two strings, build a tiny
    index, and confirm the more similar string ranks first.

    FIRST RUN: Downloads ~90 MB model weights from Hugging Face Hub.
    Subsequent runs use the local cache.

    Run with: pytest -v -m slow
    """
    from src.embeddings import embed_texts

    texts = [
        "Alex Mercer is an AI engineer specializing in machine learning.",
        "The Eiffel Tower is located in Paris, France.",
    ]
    embeddings = embed_texts(texts)

    assert embeddings.shape == (2, EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32

    # Unit-norm check: L2-normalized embeddings should have norm ≈ 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Embeddings are not unit-norm: {norms}"

    # Build a tiny index and verify the closest string to "AI engineer" is texts[0]
    chunks = [_make_chunk(f"smoke_{i}", t, "smoke.md", chunk_index=i) for i, t in enumerate(texts)]
    build_index(chunks, embeddings, chunk_size=120, chunk_overlap=25, storage_dir=tmp_path)

    from src.embeddings import embed_query as real_embed_query
    query_vec = real_embed_query("What does Alex Mercer specialize in?")
    results = query_index(query_vec, top_k=2, storage_dir=tmp_path)

    assert results[0]["chunk_id"] == "smoke_0"  # AI engineer text should rank first
    assert results[0]["distance"] < results[1]["distance"]
    print(f"\n[smoke] Top result: '{results[0]['text']}' | distance={results[0]['distance']:.4f}")
