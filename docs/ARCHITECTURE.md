# Architecture

## Two workflows
Indexing: files → loaded document records → chunks → local embeddings → persistent Chroma collection.
Answering: Next.js frontend (Vercel) → HTTP JSON API → FastAPI backend (`src/api.py`) → `src.pipeline.query_pipeline` → local embeddings → Chroma retrieval → OpenAI evidence-grounded generation → structured response.

The Next.js frontend communicates exclusively via HTTP with the FastAPI backend; it contains no RAG or embedding logic. Ingestion remains an explicit, offline operation executed via `ingest.py` when documents change.

## Phase 5 Subphases
- **5A: FastAPI backend**: Create `src/api.py`, exposing `/health` and `/query`. Reuses `src.pipeline.query_pipeline` without blocking the async event loop; binds to `127.0.0.1`; configures CORS for `http://localhost:3000`; tested with FastAPI `TestClient`.
- **5B: Next.js frontend**: Create `frontend/` using Next.js/React; connects to local FastAPI backend; displays question input, answer, citations, expandable evidence, and status states.
- **5C: Deployment preparation**: Configuration and container/platform setup for deploying the Next.js frontend to Vercel and the Python backend to a hosted environment with persistent Chroma storage and production authentication/rate limits.

## Target structure
```text
personal-ai-assistant/
    frontend/            # Next.js frontend application (Phase 5B/5C)
    ingest.py
    README.md
    requirements.txt
    .env                 # local only
    .env.example
    .gitignore
    docs/                # these project documents and review log
    data/documents/      # sanitized source files only
    storage/chroma/      # generated, ignored by Git
    src/
        __init__.py
        config.py
        loaders.py
        chunking.py
        embeddings.py
        vector_store.py
        retrieval.py
        generation.py
        pipeline.py
        api.py           # FastAPI backend application (Phase 5A)
    tests/
    evaluation/
        questions.json
        evaluate.py
        results/
```
Create files only as needed in the active phase. Use docs/ consistently, not both doc/ and docs/.

## Module contracts
| Module | Input | Output / responsibility |
|---|---|---|
| config.py | Environment variables and project root | Validated settings and absolute paths |
| loaders.py | Supported file paths | Document records with text, relative source path, optional page |
| chunking.py | Document records and chunk settings | Chunk records with stable IDs, text, source, page and position |
| embeddings.py | List of strings | Normalized vectors; expose model metadata |
| vector_store.py | Chunk records and vectors, or query vector | Persistent writes and similarity results |
| retrieval.py | Question and retrieval settings | Ranked evidence records and distances |
| generation.py | Question and selected evidence | Answer with allowed source IDs, or abstention |
| pipeline.py | Question | Answer, cited sources, retrieved evidence and timings |
| ingest.py | Corpus folder | Validated index build summary |
| api.py | HTTP request (`/query`, `/health`) | Structured JSON response via `pipeline.py` |
| frontend/ | User interaction in browser | Web UI rendering question, answer, citations, and evidence |

Use simple dictionaries or dataclasses for records; explain the choice. Avoid a new abstraction/class for every operation.

## Data lineage
Each chunk contains chunk_id, text, source, page (optional), chunk_index and content hash. Derive stable IDs from relative source, page, position, and content with a deterministic hash, never Python's process-dependent hash(). Store text and metadata alongside vectors. Reject unreadable/empty content with an actionable report.

## Persistence and updates
For the first small corpus, use an explicit full rebuild to a fresh collection on corpus change; activate it only after a successful build. Preserve the last usable index on failure. Document this simple policy instead of claiming incremental updates. Record corpus and settings fingerprints. Do not automatically rebuild at app startup.

## Retrieval baseline
Start with configurable top_k=4 and cosine distance. Smaller cosine distance means more similar; do not display it as confidence or probability. Inspect expected evidence before adding thresholds. Multi-document questions may require tuning k or retrieval strategy. Bound evidence length and inspect truncation.

## Generation and trust boundary
Include a source-ID-labelled context and clear evidence-only instructions. Documents cannot override system instructions. Use a structured response schema if supported by the verified model/API; validate cited IDs against the supplied evidence. Unsupported/unknown IDs cause a controlled retry or error, not fabricated citations. Semantic support still requires evaluation.

## Operational rules
Never log keys or full sensitive documents. Local storage does not imply all-local processing: retrieved excerpts and questions go to OpenAI. Use sanitized samples. Load the embedding model once per process; cache expensive resources in the UI without caching secrets into output. Empty index and API failures must be understandable.
