# Technology Stack and Decisions

## Selected stack
| Component | Decision | Reason |
|---|---|---|
| Language | Python; verify a supported installed version before setup | One language across pipeline and UI |
| Environment | .venv and pip | Explicit, familiar dependency isolation |
| Source control | Git and GitHub | Reviewable incremental changes |
| Text ingestion | pathlib and UTF-8 file reading | Inspectable standard-library implementation |
| PDF ingestion | pypdf, added in Phase 2 | Text-based PDF extraction; no OCR |
| Chunking | Small custom word-based chunker | Makes size, overlap, and metadata visible |
| Embeddings | Sentence Transformers, all-MiniLM-L6-v2 | Local embedding baseline; no embedding API key required |
| Storage | Chroma PersistentClient | Local persistence without a separate server |
| Generation | Official OpenAI Python SDK | Direct, inspectable API integration |
| Configuration | python-dotenv and environment variables | Keep secrets separate from code |
| Interface | FastAPI backend + Next.js frontend (Vercel) | Decoupled modern web app: FastAPI exposes RAG pipeline, Next.js provides responsive UI |
| Checks | pytest and a JSON evaluation set | Component checks plus observable RAG behavior |

## Interface architecture evolution
Initially, a single-process Streamlit UI was planned. In Phase 5, the project evolved to a decoupled production-ready architecture:
- **Frontend**: Next.js (React) optimized for deployment to Vercel.
- **Backend API**: FastAPI (`src/api.py`) exposing the Python RAG pipeline via REST endpoints.
- **Benefits**: Clean separation of frontend UI from backend pipeline, prevention of Python UI re-run quirks, asynchronous request handling without blocking the model pipeline, strict CORS isolation, and independent deployability.

## OpenAI boundary
The OpenAI API key is for answer generation. Embeddings remain local in the initial design. Select an available text-generation model at Phase 4, verify current official SDK documentation, and record its exact identifier in configuration and evaluation results. Do not assume a ChatGPT subscription includes API billing. Do not silently switch embedding providers.

## Embedding consistency
Use the same embedding model and normalization for indexing and queries. Record model identifier, dimension, chunk settings, and corpus fingerprint with the index. Rebuild after an embedding model or chunking change. Chroma must receive explicitly computed embeddings; do not accidentally use its default embedding function.

## Initial chunking baseline
Start with approximately 120 words and 25 words overlap, splitting within source/page boundaries. These are tuning candidates, not optimized settings. Inspect the embedding tokenizer's actual maximum sequence length and check token counts: word counts do not guarantee no truncation. Adjust before indexing if needed. Never join separate documents into a chunk.

## Database tradeoffs
Chroma fits this local learning scope. FAISS is a similarity-search library requiring separate document/metadata management. Qdrant is an alternative dedicated database; Pinecone is a managed service; pgvector fits applications already using PostgreSQL. A future migration still requires adapter changes, data export/re-indexing, and validation.

## Dependency policy
Install only packages required by the active phase. Verify compatibility in .venv and record tested versions in requirements.txt. Do not install a large framework by default. In Phase 5A, FastAPI, Uvicorn, and HTTPX are installed to power the API backend and its test client. LangChain, LlamaIndex, Docker, and reranking are deferred until a measured need exists.
