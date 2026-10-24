# Phased Implementation and Review Gates

## Mandatory workflow
Only the current phase is authorized. Completing a phase does not authorize the next. After each phase, stop and wait for the user's explicit instruction: “I reviewed Phase N. Proceed to Phase N+1.” Clarification or error reports are not phase approval. Fix current-phase issues without advancing.

For each phase: explain the goal briefly → implement only its scope → run meaningful checks → explain code flow → give a small learner exercise → request review and stop. Keep documentation consistent with actual behavior.

## Phase 1 — Foundation and corpus
Deliver: verify Windows terminal, Python and .venv; create root setup files, minimal config.py and package marker; add 3–5 clearly labelled fictional/sanitized Markdown samples; create initial question records. No embeddings, database, generation, UI or future module stubs.
Configuration: OPENAI_API_KEY and OPENAI_MODEL placeholders in .env.example. Never read or print an existing secret to explain setup. Local loading of the key does not require an API call.
Checks: interpreter resolves to .venv; configuration resolves paths independent of working directory; missing required configuration produces an actionable message when its feature is invoked; Git ignores secrets and storage.
Learn: environment vs interpreter, imports, configuration vs code, source-of-truth documents.
Exercise: change one sample project's status and identify which expected answer must change.
Gate: user understands setup and corpus; no external API call needed.

## Phase 2 — Loading and chunking
Deliver: loaders.py and chunking.py. Start with .md/.txt; add pypdf and text PDFs in this phase only after text loading works. Preserve page boundaries. Report empty/scanned PDFs as unsupported extraction; never pretend they were read.
Checks: normal/empty/unsupported files, Unicode, overlap, deterministic IDs, source/page tracing; inspect a few real chunks. Validate chunk size/overlap inputs and measure embedding tokenizer lengths before Phase 3 indexing.
Learn: document vs chunk, metadata, overlap, functions, deterministic identifiers.
Exercise: change chunk size and explain the effect on count/context.
Gate: learner can trace one file into its chunks.

## Phase 3 — Embeddings and retrieval
Deliver: embeddings.py, vector_store.py, retrieval.py, ingest.py; local persistent index, safe rebuild policy, a terminal retrieval demonstration. No OpenAI generation or Streamlit.
Checks: same embedding settings on both paths, persistence across restart, repeated ingestion without duplicates, document removal after explicit rebuild, known-question evidence in top-k, missing-index error. Verify retrieval with several questions before proceeding.
Learn: vector dimensions, cosine distance, top-k, index lifecycle, retrieval failure vs generation failure.
Exercise: compare top_k=2 and top_k=4 on a multi-document question.
Gate: inspect returned text and sources without an LLM.

## Phase 4 — Evidence-grounded generation
Deliver: generation.py and pipeline.py, official OpenAI SDK integration, configurable supported model, citations, insufficient-evidence response, partial-answer behavior, clear API errors.
Checks: mocked API/component tests plus a small user-authorized live smoke check; direct, multi-document, missing-evidence, conflicting-status and document-instruction cases. Track time and model ID. No invented claims of guaranteed safety.
Learn: prompt roles, context construction, structured output, token budget, evidence vs model knowledge.
Exercise: explain why a plausible answer must still be rejected if unsupported.
Gate: learner can trace question → evidence → request → validated answer.

## Phase 5 — FastAPI Backend, Next.js Frontend, and Deployment Preparation

### Phase 5A — FastAPI backend, tested locally
Deliver: `src/api.py` wrapping `src.pipeline.query_pipeline`, `GET /health`, `POST /query` with Pydantic validation (nonblank question, top_k 1–10), CORS restricted to `http://localhost:3000`, 127.0.0.1 binding, non-blocking execution via worker thread, singleton embedding model reuse, and safe HTTP error mapping.
Checks: FastAPI `TestClient` suite with mocked pipeline; tests for health, valid questions, invalid inputs, all 5 result statuses (`answered`, `partial`, `conflict`, `abstain`, `error`), and sanitized failures without leaking secrets.

### Phase 5B — Next.js frontend, tested against local backend
Deliver: `frontend/` directory with Next.js/React application connecting to `http://localhost:8000/query`, rendering responsive search, clean markdown answers, citation badges resolved from storage, expandable evidence panels with distance indicators, and clear loading/error/abstention states.
Checks: End-to-end user interaction in browser against local FastAPI server; empty query handling, network error recovery, citation drawer inspection.

### Phase 5C — Deployment preparation (Vercel + Hosted Backend)
Deliver: Deployment configuration for Next.js frontend on Vercel and separate containerized/cloud deployment for the Python backend with persistent Chroma storage. Production environment variable management, CORS production domain configuration, authentication considerations, and rate limiting strategy.
Checks: Build checks for Next.js bundle, container build test, persistent volume verification for Chroma index, security review of exposed endpoints.

## Phase 6 — Evaluation and portfolio packaging
Deliver: fixed 20-case evaluation, saved actual results, reproducible README, architecture description and limitations. Prepare GitHub commits; do not publish or push without instruction.
Checks: evaluation definitions applied consistently, retrieval and generation separated, no fake results, clean setup instructions, secret exclusion, source traceability.
Learn: failure analysis, reproducibility, tradeoffs and interview demonstration.
Exercise: explain one failure and justify one targeted improvement.
Gate: user reviews measured results and repository readiness.

## Phase completion report format
1. Scope completed and files changed.
2. One concrete execution trace naming functions and their inputs/outputs.
3. Key code sections explained in small pieces.
4. Commands to run and expected output shape (label examples as illustrative).
5. Checks actually run, results, and checks not run.
6. Two understanding questions and one small modification exercise.
7. Outstanding issues; proposed next phase; STOP for review.

Maintain docs/PHASE_REVIEW_LOG.md starting in Phase 1: phase, implementation status, verification, learner-review status, explicit approval, next authorized phase. Never mark user approval on the user's behalf.
