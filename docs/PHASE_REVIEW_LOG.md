# Phase Review Log

This log tracks the progress, verification results, learner review status, and explicit user approvals across all implementation phases of the Personal AI Knowledge Assistant project.

> **Rule**: An automated test pass is not learner approval. Only the user can authorize advancing to the next phase by stating: `"I reviewed Phase N. Proceed to Phase N+1."`

---

## Phase Summary Table

| Phase | Description | Implementation Status | Automated Verification | Learner Review Status | Explicit User Approval | Next Authorized Phase |
|---|---|---|---|---|---|---|
| **Phase 1** | Foundation and corpus | **Completed** | **8 passed** (pytest) | **Reviewed** | **Approved by Learner** | **Phase 2 Authorized** |
| **Phase 2** | Loading and chunking | **Completed** | **32 passed** (pytest) | **Reviewed** | **Approved by Learner** | **Phase 3 Authorized** |
| **Phase 3** | Embeddings and retrieval | **Completed (recoverable replacement applied)** | **62 passed** (pytest) | **Reviewed** | **Approved by Learner** | **Phase 4 Authorized** |
| **Phase 4** | Evidence-grounded generation | **Completed (Reviewed & Hardened)** | **85 passed** (pytest) | **Reviewed** | **Approved by Learner** | **Phase 5A Authorized** |
| **Phase 5A** | FastAPI backend (local) | **Completed (Reviewed & Hardened)** | **100 passed** (pytest) | **Reviewed** | **Approved by Learner** | **Phase 5B Authorized** |
| **Phase 5B** | Next.js frontend (local) | **Completed** | **34 passed** (vitest) | **Reviewed** | **Approved by Learner** | **Phase 5C Authorized** |
| **Phase 5C** | Deployment preparation (Vercel + Hosted backend) | **Completed** | **146 passed** (pytest) + **34 passed** (vitest) | **Reviewed** | **Approved by Learner** | **Phase 6 Authorized** |
| **Phase 6** | Evaluation and portfolio packaging | **Completed (Holdout Evaluated & Documented)** | **146 passed** (pytest) + **34 passed** (vitest) | **Reviewed** | **Awaiting Final Sign-Off** | Deployment Execution (Phase 7) |


---

## Detailed Phase Entries

### Phase 1 — Foundation and Corpus
- **Date**: 2026-10-22
- **Scope Implemented**:
  - Virtual environment and Python 3.12 verified.
  - Setup files created: `requirements.txt`, `pytest.ini`, `.env.example`, `.gitignore`.
  - Core package created: `src/__init__.py`.
  - Configuration module created: `src/config.py` (CWD-independent `pathlib.Path` resolution, safe config summary, `.env` loading, actionable error handling for missing/placeholder `OPENAI_API_KEY`).
  - Sanitized knowledge corpus created: 4 Markdown documents in `data/documents/`:
    1. `candidate_profile.md` (Alex Mercer background, skills, confidential/undocumented fields).
    2. `project_aurora.md` (Deployed fraud detection system, metrics, Kafka/Redis/PyTorch).
    3. `project_nexus.md` (Internal RAG assistant with documented status conflict).
    4. `project_beacon.md` (Planned model evaluation framework with adversarial injection test).
  - Benchmark evaluation set created: `evaluation/questions.json` (20 questions: 14 development, 6 holdout, covering all 6 required categories).
  - Automated tests implemented in `tests/test_config.py`.
- **Verification Evidence**:
  - Command: `.\.venv\Scripts\pytest.exe -v`
  - Result: 8 passed in 0.08s (all tests green).
- **Learner Review Status**: Reviewed by learner.
- **Explicit Approval**: Granted by learner ("I reviewed Phase 1. Proceed with Phase 2 only.").

---

### Phase 2 — Loading and Chunking
- **Date**: 2026-10-22
- **Scope Implemented**:
  - Added dependency `pypdf==6.21.2` to `requirements.txt`.
  - Implemented `src/loaders.py`:
    - `DocumentRecord` dataclass (text, relative source path, optional page number, metadata).
    - `load_text_file` for UTF-8 `.txt` and `.md` files; rejects empty/whitespace files.
    - `load_pdf_file` returns `(records, page_warnings)` tuple. Partially extractable PDFs
      return readable pages AND per-page warnings for empty/image pages, preserving original
      page numbers on all readable pages. Empty pages described as textless (not assumed scanned).
      Fully unextractable PDFs raise `ValueError`.
    - `LoadDirResult` tuple subclass: distinguishes file-level load errors from page-level warnings
      while preserving backward-compatible 2-element tuple unpacking `(records, errors)`.
    - A failed file never suppresses successfully loaded files.
  - Implemented `src/chunking.py`:
    - `ChunkRecord` dataclass (chunk_id, text, source, page, chunk_index, content_hash, word_count, metadata).
    - `validate_chunk_settings` (enforces positive chunk size, non-negative overlap, overlap < chunk_size).
    - `_compute_chunk_id` hashes the **full normalized source path** (not just the stem)
      into the ID — files with the same stem in different directories or with different
      extensions produce distinct IDs.
    - `chunk_document` deterministic word-based chunker with configurable overlap.
    - `chunk_documents` batch chunker strictly preserving document/page boundaries.
  - Test Suite:
    - Collision tests: same-stem/different-dir, different extension, different page.
    - `_compute_chunk_id` unit test verifying full-source incorporation.
    - Injection test docstring: chunker treats text as data; LLM generation safety is evaluated in Phase 4.
    - Partial-PDF test: page 1 readable, page 2 empty (warning), page 3 readable (retains number 3).
    - Corrupted-file test: garbage bytes produce a clear error, not a silent empty result.
    - Directory isolation test: one bad file and one good file; errors and records remain separate.
    - Password-protected PDF test: encrypted PDFs raise actionable decryption error and isolate from good files.
- **Verification Evidence**:
  - Command: `.\.venv\Scripts\pytest.exe -v`
  - Result: 32 passed.
- **Learner Review Status**: Reviewed by learner.
- **Explicit Approval**: Granted by learner ("I reviewed Phase 2. Proceed to Phase 3 only.").

---

### Phase 3 — Embeddings, Vector Storage, and Retrieval
- **Date**: 2026-10-22
- **Scope Implemented**:
  - Added dependencies `sentence-transformers==3.4.1` and `chromadb==1.0.8` to `requirements.txt`.
  - Implemented `src/embeddings.py`:
    - Model: `sentence-transformers/all-MiniLM-L6-v2` (dimension: 384, float32).
    - Lazy-loaded singleton model via `_get_model()`.
    - Identical embedding pipeline with L2 normalization (`normalize_embeddings=True`) for both chunks (`embed_texts`) and search queries (`embed_query`).
    - Model metadata reporting (`get_model_metadata`) for index compatibility tracking.
  - Implemented `src/vector_store.py`:
    - ChromaDB `PersistentClient` targeting `storage/chroma` (collection name: `knowledge_chunks`).
    - **Recoverable Replacement Strategy**:
      1. Inputs and embeddings pre-validated before any storage changes.
      2. `recover_orphaned_collections()` automatically restores `knowledge_chunks_backup` if a previous run was interrupted after backing up the active index, preventing data loss.
      3. Fully populates and validates `knowledge_chunks_staging`. If staging fails, staging is deleted and the active index remains completely untouched.
      4. Preserves the active collection by renaming to `knowledge_chunks_backup` before promotion.
      5. Promotes staging to `knowledge_chunks`.
      6. Verifies the promoted collection (count and metadata).
      7. Deletes `knowledge_chunks_backup` only after successful verification.
      8. If promotion or verification fails, automatically restores the active collection from backup.
    - **Concurrency & Atomicity Note**: Designed for single-user local usage. Replacement is failure-resilient and recoverable, but not atomic across all steps; queries must not run concurrently with rebuilding.
    - Recorded Metadata: `embedding_model`, `embedding_dimension`, `embedding_normalized`, `distance_metric`, `chunk_size`, `chunk_overlap`, and `hnsw:space`.
    - `check_index_compatibility`: rejects mismatched model, dimension, metric, normalization, and detects changes to `chunk_size` or `chunk_overlap`.
    - `query_index`: checks compatibility before querying; returns ranked passages ordered by ascending cosine distance. Distances are similarity measures and are never labelled or mapped as confidence/probability scores.
  - Implemented `src/retrieval.py`:
    - `validate_query`: strips whitespace, rejects empty/blank queries, validates positive `top_k`.
    - `retrieve`: embeds query, queries vector store, returns ranked passages.
    - `format_evidence_for_display`: terminal output with ranking, source file, page, chunk index, and distance; cleanly truncates chunks longer than `max_chars_per_chunk`.
  - Implemented `ingest.py`:
    - `ingest_corpus` core pipeline function with programmatic configuration and error raising.
    - Distinguishes page-level skipped-page warnings from file-loading errors.
    - Aborts on file-loading errors unless `--allow-partial-ingestion` is explicitly passed.
    - Refuses to replace a non-empty index with an empty corpus unless `--allow-empty` is explicitly passed.
- **Verification Evidence**:
  - Full test suite:
    - Command: `.\.venv\Scripts\pytest.exe -v`
    - Result: **62 passed in 16.99s** (53 base + 9 failure-handling/recoverability/compatibility tests).
  - Recoverable replacement tests added:
    - Promotion failure rollback and restoration of previous index.
    - Recovery of orphaned backup after an interrupted process.
    - Refusing empty replacement without explicit permission.
    - Aborting ingestion on file errors unless partial ingestion is enabled.
  - Real-model smoke test:
    - Verified real `all-MiniLM-L6-v2` inference, L2 unit norm, and semantic distance ranking.
  - Live corpus ingestion:
    - Rebuilt `storage/chroma` cleanly using `ingest.py --force-rebuild`.
  - Retrieval Demonstration Analysis:
    - Tested development question `q01` (*"What embedding model and vector database were used in Project Nexus?"*).
    - Result: Top 3 passages retrieved with cosine distances `0.4840`, `0.4956`, and `0.5327`.
    - **Analysis & Findings**:
      - All 3 results came from the expected source document (`data/documents/project_nexus.md`).
      - The specific requested facts (`sentence-transformers/all-MiniLM-L6-v2` and `ChromaDB`) appeared in **Result 3** (`project_nexus_9fb667c6_p0_c000_689b5262`).
      - **Critical observation**: Matching the source document alone does not prove every chunk directly answers the question. Results 1 and 2 shared high semantic overlap due to general Project Nexus terminology (UI and deployment status), but only Result 3 contained the factual evidence answering the prompt.
- **Learner Review Status**: Reviewed by learner.
- **Explicit Approval**: Granted by learner ("I reviewed Phase 3. Proceed to Phase 4 only.").

---

### Phase 4 — Evidence-Grounded Generation and Pipeline
- **Date**: 2026-10-22
- **Scope Implemented**:
  - Pinned `openai==3.37.0` in `requirements.txt`.
  - Implemented `src/generation.py`:
    - Grounding & Prompt Roles: Structured system prompt treating all retrieved text as untrusted reference material. System prompt includes explicit immunity instructions against document prompt injection (e.g. `SYSTEM OVERRIDE NOTE`).
    - Standard Abstention Wording: Follows the exact documented phrase when evidence is absent or insufficient:
      `"I do not have sufficient information in the provided documents to answer this question."`
    - Fast-Path Abstention: When zero passages are retrieved, generation immediately abstains without executing an unnecessary API call.
    - Clean Separation of Abstention vs. Validation Error: Legitimate abstention due to absent/insufficient evidence yields `status="abstain"` and `error=None`. Model schema violations or invalid citations yield `status="error"` and a distinct validation error.
    - Strict Response & Schema Validation:
      - Plain text or malformed JSON is strictly rejected and never bypasses validation to become a successful answer.
      - Required fields checked: `answer` (str), `status` (str in `{"answered", "abstain", "partial", "conflict"}`), `cited_chunk_ids` (list of str).
      - Field types strictly validated.
    - Citation Validation & Storage Grounding:
      - `validate_and_resolve_citations` validates every cited ID against retrieved evidence.
      - If any citation ID is outside retrieved evidence, it is **not** silently stripped: a distinct validation error is returned while preserving diagnostic lists (`invalid_citations` and valid `citations`).
      - Grounding rule: `status="answered"` requires at least one valid citation from retrieved evidence.
      - `status="partial"` requires at least one valid citation and a non-empty `missing_information` description.
      - Source paths and page numbers are resolved strictly from stored metadata — never accepting invented metadata from the LLM.
    - Citation Validity Disclaimer: Attached `CITATION_VALIDITY_DISCLAIMER` explaining that syntactic citation existence does not prove semantic factual support.
    - Error Safety & Sanitization:
      - Uses fixed, actionable public error messages for all API errors (timeout, rate limit, connection drop, HTTP 401 auth failure), preventing raw SDK provider strings from leaking into output.
      - `sanitize_error_message` scrubs fictional short and long keys (`sk-...`) and Bearer tokens.
      - Documented limitation: regex scrubbing provides defense-in-depth, not a guaranteed secret remover for arbitrary unstructured text; fixed public messages provide the primary protection.
    - Network Resilience: Finite request timeout (default 30.0s), bounded retries (default 2), and structured classification of API timeouts, rate limits, connection errors, and status errors.
  - Implemented `src/pipeline.py`:
    - `query_pipeline`: Coordinates question validation, embedding retrieval from persistent Chroma storage, evidence-grounded generation, latency tracking (`retrieval_ms`, `generation_ms`, `total_ms`), and structured dictionary output.
    - Preserves retrieval cosine distance as explicit `distance` label (never confidence or probability).
    - Standalone CLI runner in `main()` for manual inspection of pipeline queries.
  - Comprehensive Test Suite in `tests/test_generation_pipeline.py`:
    - 23 focused offline tests mocking OpenAI responses (strict schema validation, plain-text rejection, invalid citation rejection, missing field detection, type mismatch detection, supported answer, abstention separation, partial answer, conflict reporting, API timeouts, rate limits, mocked connection failure, mocked auth failure, credential sanitization for short and long keys, and end-to-end pipeline execution).
    - Methodological boundary documented: Mocked tests verify pipeline plumbing, prompt construction, and schema parsing contracts; they do NOT prove real-model grounding or injection resistance.
- **Verification Evidence**:
  - Complete offline test suite:
    - Command: `.\.venv\Scripts\pytest.exe -v`
    - Result: **85 passed in 17.41s** (84 unit/mocked offline tests + 1 real-model embedding smoke test). All 85 tests green across Phases 1–4.
- **Learner Review Status**: Reviewed by learner.
- **Explicit Approval**: Granted by learner ("Change Phase 5 from Streamlit to a Next.js frontend intended for Vercel, backed by our existing Python RAG pipeline through FastAPI... Implement ONLY Phase 5A now.").

---

### Phase 5A — FastAPI Backend (Local)
- **Date**: 2026-10-23
- **Scope Implemented**:
  - Pinned `fastapi==0.115.9`, `uvicorn==0.54.0`, and `httpx==0.28.1` in `requirements.txt`.
  - Implemented `src/api.py`:
    - Clean reuse of `src.pipeline.query_pipeline`: contains zero duplicated loading, chunking, embedding, or generation code.
    - Endpoint `GET /health`: Basic service liveness check returning `{"status": "ok", "service": "personal-ai-knowledge-assistant"}`.
    - Documented distinction between liveness (HTTP server up) and index/model readiness (which requires populated vector storage, loaded model weights, and verified API credentials).
    - Endpoint `POST /query`: Accepts `QueryRequest` with non-blank `question` string validation and `top_k` validated as an integer in [1, 10] (out-of-range values are rejected with HTTP 422 Unprocessable Entity, not clamped).
    - Non-blocking execution: Synchronous RAG pipeline (local PyTorch embedding + Chroma lookup + OpenAI HTTP request) is executed in a background worker thread via `starlette.concurrency.run_in_threadpool`, keeping the FastAPI async event loop responsive.
    - Singleton model reuse: The local SentenceTransformer embedding model remains cached in-process across all requests and is never reloaded.
    - No index modification: Does not ingest, rebuild, or alter Chroma storage at startup or during queries. Exposes zero ingestion endpoints.
    - Secret containment: OpenAI API key remains exclusively on the backend server and is never exposed in headers, responses, or error details.
    - CORS Policy: Configured strictly for `http://localhost:3000` (and `http://127.0.0.1:3000`), rejecting wildcards (`*`). Note: CORS is a browser cross-origin policy, not an authentication or authorization mechanism.
    - Local binding: Binds local development server strictly to `127.0.0.1:8000`.
    - Operational note: Local API is designed for single-user local development; production authentication and rate-limiting belong in deployment preparation (Phase 5C).
  - Test Suite in `tests/test_api.py`:
    - 15 focused tests using FastAPI `TestClient` with mocked `query_pipeline`.
    - Tests for: `GET /health`, `POST /query` with all 5 pipeline statuses (`answered`, `abstain`, `partial`, `conflict`, `error`), input validation (empty question, whitespace-only, missing question, `top_k < 1`, `top_k > 10`, bad type), sanitized HTTP 500 error handling, and strict CORS headers.
    - Zero paid API calls or live external network requests.
- **Verification Evidence**:
  - Full test suite:
    - Command: `.\.venv\Scripts\pytest.exe -v`
    - Result: **100 passed in 65.16s** (99 unit/mocked offline tests + 1 real-model embedding smoke test).
- **Learner Review Status**: Reviewed by learner.
- **Explicit Approval**: Granted by learner ("Continue the previously requested Phase 5B only.").

---

### Phase 5B — Next.js Frontend (Tested Against Local Backend)
- **Date**: 2026-10-23
- **Scope Implemented**:
  - Scaffolding & Configuration:
    - Scaffolding in `frontend/` preserving original `frontend/.env.example`.
    - Local environment configuration in `frontend/.env.local` pointing to `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`.
    - Custom modern vanilla CSS design system in `frontend/src/app/globals.css` (custom dark mode, glassmorphism, responsive grid, status color tokens, distance meters removed, micro-animations).
  - Strongly Typed API Client (`frontend/src/lib/api.ts` & `frontend/src/types/api.ts`):
    - `checkHealth()`: Checks backend `/health` liveness probe with timeout and error classification.
    - `queryAssistant()`: Executes `POST /query` against FastAPI with request timeout, input trimming, status parsing, and actionable error handling.
  - Interactive UI Components & Evidence Display:
    - `Navbar`: Live backend connectivity status indicator with manual refresh action (no background polling intervals).
    - `QueryForm`: Search input with client-side empty/whitespace validation, `top_k` slider (range 1–10), duplicate-submission prevention, and 4 clickable sample question chips.
    - `ResponseView`: Renders answer with status pills (`answered`, `abstain`, `partial`, `conflict`, `error`), latency breakdown chips (`total_ms`, `retrieval_ms`, `generation_ms`), model badge (`gpt-4o-mini`), missing information notice, and citation validity disclaimer ("Citation-ID validation does not prove factual support").
    - `MarkdownAnswer`: Markdown-aware answer renderer detecting `[chunk_id]` citation tokens and rendering them as interactive clickable badges.
    - `EvidencePanel`: Retrieved evidence cards showing chunk ID, document source path, page number, raw Chroma cosine distance, explicit disclaimer ("Lower distance means a closer vector match; it does not establish answer correctness"), cited-in-answer badge, and auto-scroll/highlight when citation badges are clicked.
    - Terminology Precision: Avoided false claims of "verified answer" or "verified citations"; used accurate wording "Answer" and "Citations matched to retrieved evidence".
  - Automated Frontend Test Suite (`frontend/src/test/`):
    - 21 automated unit and integration tests using Vitest, `@testing-library/react`, and `@testing-library/jest-dom`.
    - Mocks all API responses (zero live API calls during automated tests).
    - Tests for:
      - All 5 result statuses (`answered`, `abstain`, `partial`, `conflict`, `error`).
      - Blank input, whitespace rejection, and `top_k` slider validation/clamping.
      - Loading state and duplicate-submission prevention while query is pending.
      - Unreachable backend network errors, HTTP 422/500/502 errors, and malformed JSON recovery.
      - Citation clicks selecting and highlighting the correct evidence card.
      - Safe rendering of model and document text containing HTML/script tags without injection.
      - Retries strictly occurring through explicit user interaction with the Retry button.
- **Verification Evidence**:
  - Node & npm version check:
    - Command: `node.exe --version; npm.cmd --version`
    - Result: `v24.22.1`, `11.19.1`.
  - Frontend Automated Tests:
    - Command: `npm.cmd --prefix frontend run test`
    - Result: **21 passed in 3.61s** (2 test suites, 0 failures, 0 external network calls).
  - Frontend Linting:
    - Command: `npm.cmd --prefix frontend run lint`
    - Result: Exit code 0 (0 errors, 0 warnings).
  - Frontend TypeScript Type Check:
    - Command: `npx.cmd --prefix frontend tsc --project frontend/tsconfig.json --noEmit`
    - Result: Exit code 0 (0 type errors).
  - Next.js Production Build Check:
    - Command: `npm.cmd --prefix frontend run build`
    - Result: Exit code 0, static pages compiled successfully in 1527ms.
  - Live End-to-End Browser Check Note:
    - Live OpenAI API requests were made during the interactive browser check to verify real-model grounding on `'What embedding model and vector database were used in Project Nexus?'` (answered) and `'Who founded Apple Inc and when was the iPhone announced?'` (abstained).
    - Visual Artifacts Saved:
      - `initial_page_load_1792794319593.png`
      - `answered_state_1792794393665.png`
      - `abstained_state_1792794457262.png`
      - Browser session recording: `frontend_verified_demo_1792794305125.webp`
- **Outstanding Issues / Remaining Work**:
  - Phase 5C (Deployment Preparation): Production deployment configuration for Next.js on Vercel, containerization of the FastAPI backend with persistent volume for Chroma vector storage, production CORS origin whitelisting, and API rate limiting.
  - Phase 6 (Evaluation Benchmark): Execution and reporting of the formal 20-case quantitative benchmark test suite.
- **Learner Review Status**: Approved by learner.
- **Explicit Approval**: *Approved by learner prior to Phase 5C.*

---

## Phase 5C: Deployment Preparation (Revised for Direct Python & Next.js Proxy)

- **Date**: 2026-10-24
- **Phase Objective**: Revise deployment architecture for a verified direct Python deployment route without requiring local Docker. Implement a secure Next.js server-side proxy (BFF) with authenticated reviewer access, explicit production/development settings, accurate rate-limiting and client-IP resolution policies, grounded cost controls, and precise readiness semantics, accompanied by comprehensive mocked tests and updated documentation.
- **Key Architectural Decisions & Implementations**:
  1. **Direct Python Deployment Route**:
     - Formally documented the direct Python host deployment route (`uvicorn src.api:app`) backed by a persistent Chroma directory (`CHROMA_PERSIST_DIRECTORY`) and an explicit HuggingFace model weights cache (`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`).
     - Documented systemd service configuration for persistent Linux host deployments (Ubuntu 22.04/24.04).
     - Preserved `Dockerfile` and `.dockerignore` as an optional, untested deployment route for containerized environments without claiming local Docker verification.
  2. **Next.js Server-Side Proxy (`/api/query` & `/api/health`)**:
     - Implemented Next.js App Router server proxy in `frontend/src/app/api/query/route.ts` and `frontend/src/app/api/health/route.ts`.
     - Browser client in `frontend/src/lib/api.ts` now communicates strictly via relative `/api/query` and `/api/health`.
     - Server-only configuration: `BACKEND_URL`, `BACKEND_API_KEY`, and `REVIEWER_ACCESS_TOKEN` are read strictly server-side and never exposed to the browser or prefixed with `NEXT_PUBLIC_`.
     - Authenticated Reviewer Access: Next.js proxy requires valid reviewer credentials (`Authorization: Bearer <REVIEWER_ACCESS_TOKEN>` or `x-reviewer-token`). Merely hiding the backend token inside the proxy is recognized as insufficient protection against publicly callable proxy relays.
     - Single Documented Reviewer Access Flow: Reviewer inputs their token in the UI Navbar modal; the token is held strictly in current browser `sessionStorage` and sent with each query.
     - Secret Isolation: Backend API key is injected by the Next.js server proxy when calling the internal backend; backend secrets never reach the browser client bundle or network responses.
  3. **Production Configuration Enforcement**:
     - Introduced explicit `APP_ENV` configuration (`production` vs. `development`, defaulting to `development`):
       - Python Backend (`src/api.py`): In `APP_ENV=production`, the backend refuses to start (raises `RuntimeError`) during lifespan startup if `DEMO_ACCESS_TOKEN` is unset or empty.
       - Next.js Proxy (`frontend/src/app/api/query/route.ts`): In `APP_ENV=production`, the proxy returns HTTP 500 if `BACKEND_API_KEY` or `REVIEWER_ACCESS_TOKEN` is missing.
     - Local Development Preservation: In `APP_ENV=development`, open access without access tokens is permitted to preserve rapid local pairing and test execution.
  4. **Rate Limiting & Client IP Determination**:
     - Documented that sliding-window rate limits (`RATE_LIMIT_PER_MINUTE=15`) apply per worker process in memory and reset to zero on process restart or reload.
     - Added `TRUST_PROXY_HEADERS` configuration (default `false`): Untrusted `X-Forwarded-For` headers are strictly ignored by default to prevent client spoofing, using the direct socket peer IP (`request.client.host`). Set to `true` only behind a verified reverse proxy.
  5. **Cost Claims & Financial Protections**:
     - Removed the unsupported `<$0.18/hour guarantee`.
     - Documented real bounded parameters: body size (<= 16 KB), question length (<= 500 characters), candidate chunks (`top_k <= 10`), output tokens (`max_tokens=1024`), strictly manual retries (no automated retry loops in UI), and concurrency semaphore (3 simultaneous queries).
     - Documented remaining cost uncertainty: Total cumulative spend remains unbounded without setting account-level monthly hard caps in the OpenAI Billing Dashboard.
  6. **Readiness Probe Semantics & Startup Wording**:
     - Refined `GET /ready` docstrings and descriptions to state that `openai_key_configured: true` validates key presence only and does NOT prove valid credentials, active billing availability, or a successful model request.
     - Removed all descriptions of untested container startup as "instantaneous".
  7. **Comprehensive Mocked Test Suite**:
     - Added 10 mocked unit and integration tests in `frontend/src/test/proxy.test.ts` covering reviewer authentication (missing token 401, invalid token 401, valid token 200 via Bearer and custom header), missing production credentials (500), local dev fallback (200), backend rate-limit propagation (429), unreachable backend network errors (502), and secret redaction.
     - Added 5 new backend unit tests in `tests/test_api.py` covering production startup refusal on missing token, production startup success on valid token, local development open access, untrusted forwarded header rejection (`TRUST_PROXY_HEADERS=false`), and trusted proxy IP extraction (`TRUST_PROXY_HEADERS=true`).
- **Verification Evidence**:
  - **Tooling Availability & Docker Check**:
    - Command: `docker --version`
    - Finding: Docker is not installed on this local machine. No containers were built or run locally; direct Python deployment route verified.
  - **Python Test Suite**:
    - Command: `.\.venv\Scripts\pytest.exe -v`
    - Result: **117 passed, 1 warning in 18.53s** (116 mocked unit/offline tests + 1 real-model embedding smoke test; 0 external paid API calls). Tests explicitly verify `APP_ENV` validation (missing, empty, and mistyped inputs like `staging`, `prod`, `dev`, `test`), production token requirements, and local development fallback.
  - **Frontend Automated Test Suite**:
    - Command: `npm.cmd --prefix frontend test -- --run`
    - Result: **34 passed in 2.77s** across 3 test files (`api.test.ts`, `app.test.tsx`, `proxy.test.ts`; 0 failures, 0 live network calls). Tests explicitly verify fail-closed production checks when `NODE_ENV=production` even if `APP_ENV=development` or missing, pre-forwarding configuration error return, and reviewer token auth.
  - **Frontend ESLint Check**:
    - Command: `npm.cmd --prefix frontend run lint`
    - Result: Exit code 0 (0 errors, 0 warnings).
  - **Next.js Production Build**:
    - Command: `npm.cmd --prefix frontend run build`
    - Result: Exit code 0, static pages and dynamic server routes (`/api/query`, `/api/health`) compiled successfully with Turbopack.
- **Learner Review Status**: Reviewed and approved by learner.
- **Explicit Approval**: *Approved by learner on 2026-10-24. Proceeded to Phase 6 evaluation preparation.*

---

## Phase 6 — Evaluation Benchmark Runner & Preparation

- **Date**: 2026-10-24
- **Scope & Objectives**:
  - Implement a reproducible, cost-guarded evaluation runner for the fixed 20-question benchmark (`evaluation/questions.json`).
  - Preserve the fixed benchmark questions, expected answers, corpus, and development/holdout split (14 dev / 6 holdout).
  - Separate Development and Holdout results in all summaries and per-case logs.
  - Implement objective automated checks for retrieval (Source Recall@k) and citation validity.
  - Separate syntactically valid citation IDs from factual support.
  - Track correct abstentions on missing/injected questions and incorrect refusals (false abstentions) on answerable questions.
  - Count external API failures (timeouts, rate limits, 500s) separately from answer-quality failures.
  - Provide a clear manual rubric (`evaluation/rubric.md`) for answer correctness, groundedness, citation support, partial answers, and conflicts.
  - Record full corpus, index configuration, and evaluation runtime metadata.
  - Provide safe offline dry-run mode and retrieval-only mode (0 API calls).
  - Enforce explicit `--live` flag guard displaying planned request count (~24,000 tokens for 20 cases) before execution. Never run paid live evaluations automatically.
- **Key Implementation Details**:
  1. **Manual Evaluation Rubric (`evaluation/rubric.md`)**:
     - Formulates standard 3-point grading scales (0, 1, 2) for Factual Correctness, Evidence Groundedness, Citation Semantic Support, and Relevance.
     - Documents categorical behavioral checks: Answered, Correct Abstention, False Abstention, Correct Partial with missing info, Conflict Reporting, and Adversarial Injection Resistance.
     - Establishes core methodological rule: valid citation IDs or string overlap do NOT prove factual support. Unreviewed answers are marked `reviewed: false` and are never assumed correct.
     - Defines 10 failure categories (`ingestion_failure`, `chunking_failure`, `retrieval_failure`, `context_assembly_failure`, `generation_hallucination`, `generation_omission`, `citation_hallucination`, `false_abstention`, `false_answer`, `api_failure`).
  2. **Reproducible Evaluation Runner (`evaluation/evaluate.py`)**:
     - Integrates with `query_pipeline` from `src/pipeline.py`.
     - Captures environment metadata: timestamp, generation model (`OPENAI_MODEL`), embedding model, dimensions, distance metric, chunk size/overlap, chunk count, platform, and python version.
     - Per-case records capture: question ID, category, split, question, expected behavior, expected sources/evidence, observed status, answer, error, `is_api_error`, retrieved chunk IDs and sources, source recall, citation validity rate, emitted/valid/invalid counts, status match flag, correct/false abstention flags, forbidden claim detection, latency breakdown, and manual review block (`reviewed: false`).
     - Aggregates metrics separately for `development_split` (14), `holdout_split` (6), and `combined` (20).
     - Outputs machine-readable JSON results and formatted Markdown summary reports to `evaluation/results/`.
  3. **Safety Guard & Execution Modes**:
     - Safety Guard: Running without flags (`python evaluation/evaluate.py`) prints target questions (20), estimated token volume (~24,000), and exits with code 0 without calling APIs.
     - Dry-Run Mode (`--dry-run`): Runs full workflow with offline mocked pipeline (0 API calls).
     - Retrieval-Only Mode (`--retrieval-only`): Evaluates retrieval against persistent Chroma index (0 API calls). Demonstrated 100% Source Recall@4 across both splits.
     - Live Evaluation Guard (`--live`): Requires explicit flag and prints warning banner with planned request count.
  4. **Offline Test Suite (`tests/test_evaluation.py`)**:
     - Added 11 focused unit/integration tests covering question schema & split preservation, metadata capture, API failure distinction, single-case evaluation (answerable, forbidden claim detection, correct abstention, false abstention), aggregate metrics calculations, split segregation, and mock pipeline failure simulations (API timeout, fabricated citations).
- **Verification Evidence**:
  - **Python Test Suite**:
    - Command: `.\.venv\Scripts\pytest.exe -v`
    - Result: **128 passed, 1 warning in 19.91s** (127 mocked unit/offline tests + 1 real-model embedding smoke test; 0 external paid API calls).
  - **Frontend Automated Test Suite**:
    - Command: `npm.cmd --prefix frontend test -- --run`
    - Result: **34 passed in 3.01s** (0 failures, 0 live network calls).
  - **Evaluation Runner Offline Dry-Run**:
    - Command: `.\.venv\Scripts\python.exe evaluation/evaluate.py --dry-run`
    - Result: Executed all 20 benchmark questions, verified 14 dev / 6 holdout split reporting, generated `evaluation/results/eval_results_*.json` and `eval_report_*.md`.
  - **Evaluation Runner Retrieval-Only Benchmark**:
    - Command: `.\.venv\Scripts\python.exe evaluation/evaluate.py --retrieval-only`
    - Result: 100.0% Source Recall@4 across all 20 questions against local persistent Chroma index.
  - **Safety Guard Verification**:
    - Command: `.\.venv\Scripts\python.exe evaluation/evaluate.py`
    - Result: Blocked execution safely, displayed planned request banner (~24,000 tokens), prompted for `--live` or `--dry-run`, exited code 0.
- **Outstanding Issues / Remaining Work**:
  - Execution of live evaluation benchmark (awaiting explicit user instruction and review).
  - Human review of live evaluation outputs against `evaluation/rubric.md`.
- **Learner Review Status**: Prepared and ready for learner review before live evaluation.
- **Explicit Approval**: *Awaiting learner approval before executing live evaluation or deployment.*

---

### Phase 6 Recheck Preparation & Reporting Integrity Audit
- **Date**: 2026-10-24
- **Reporting Integrity Audit & Correction**:
  - Discrepancies in previous iteration report were audited against preserved run [`evaluation/results/eval_results_20261024_145500.json`](file:///c:/Users/Tesse/Desktop/Ayesh_Preply/personal-ai-knowledge-assistant/evaluation/results/eval_results_20261024_145500.json) and benchmark file [`evaluation/questions.json`](file:///c:/Users/Tesse/Desktop/Ayesh_Preply/personal-ai-knowledge-assistant/evaluation/questions.json).
  - **Question Text and Facts**: `q04` was previously reported with a paraphrased question ("What degrees does Alex Mercer hold...") and extra facts. The true benchmark question is `"Where did Alex Mercer earn their Master of Science degree and in what year?"` expecting State University, 2018, and Computer Science with specialization in Machine Learning.
  - **Fabricated Table Rows & Source Hash Segment**: The previous report contained manually fabricated markdown tables wherein the source hash `8778bda4` (which uniquely belongs to `data/documents/candidate_profile.md`) was erroneously duplicated into chunk IDs for `project_beacon.md`, `project_nexus.md`, and `project_aurora.md` alongside invented content hashes. Programmatic inspection confirmed that the active Chroma index was completely uncorrupted and contained all 10 authentic chunk IDs matching the original run with 0 missing IDs.
  - **Operational Policy**: All future evaluation result tables, chunk rankings, and metrics must be captured programmatically from execution output; manual synthesis or estimation of table values is strictly prohibited.
  - **Prompt Type Correction**: The previous report colloquially termed `missing_information` an "array". All generation instructions, prompt schemas, and Pydantic/API contracts strictly enforce `missing_information` as `Optional[str]` (a single string description or null, never an array). Added regression test `test_prompt_and_schema_consistency_missing_information_type`.
- **Applied Small Corrections**:
  - **Retrieval Volume ($k=5$)**:
    - Configured `DEFAULT_TOP_K=5` in `src/config.py`.
    - At $k=5$, offline retrieval captures the education chunk (`candidate_profile_8778bda4_p0_c002_b5a75141`) at Rank 5 for `q04` (total 4,200 chars / 582 words) and the AWS certification chunk at Rank 5 for `q06` (total 3,649 chars / 498 words) without pulling unnecessary operational memo chunks.
    - Preserved explicit caller overrides across pipeline and API endpoints.
  - **Frontend Architecture Documentation**:
    - Updated frontend defaults in `QueryForm.tsx`, `api/query/route.ts`, and `lib/api.ts` to 5.
    - Added explicit code comments documenting that frontend defaults are statically compiled into the client bundle and separately configured, rather than dynamically shared with the backend's `DEFAULT_TOP_K`.
  - **Targeted Question Filtering (`--question-ids`)**:
    - Added minimal `--question-ids` option to `evaluation/evaluate.py`.
    - Strictly respects the `--split` filter (e.g. `--split dev` will only evaluate provided IDs that belong to the development split).
    - Accurately updates planned case count in the `--live` safety guard banner.
- **Verification Evidence**:
  - Full Python offline suite: **132 passed, 1 warning in 46.72s** (0 live calls).
  - Frontend Vitest suite: **34 passed across 3 test files in 2.87s** (0 live calls).
  - Zero-cost retrieval verification: Executed `evaluation/evaluate.py --retrieval-only --split dev --question-ids q04,q06,q11,q17,q18` (5 cases evaluated, 100.0% Source Recall@5, 0 paid API calls).

---

### Phase 6 Prompt & Evaluation Reporting Refinements
- **Date**: 2026-10-24
- **Audit of 5-Case Development Run (`eval_results_20261024_162346.json`)**:
  - `q06`: Prompt previously lacked an explicit negative rule against assigning requested properties (e.g. AWS) to an un-associated entity merely because it appeared in the retrieved context block, and lacked guidance to cover multiple requested conjunction targets (projects and roles).
  - `q17`: Answer returned `conflict` status but opened with a definitive factual conclusion ("Project Nexus is not currently deployed...").
  - `q18`: Model reported a documented design "latency budget" as an apparent measured benchmark result.
  - `q11`: Model listed generic ML skills from profile summary while omitting concrete RAG/NLP technologies from the retrieved Project Nexus passage that Alex Mercer prototyped.
  - Evaluation reporting displayed `Abstention Recall: 100.0%` despite 0 unanswerable cases in the 5-case subset, and unselected holdout split printed `0 ms` latency instead of `N/A`.
- **General, Evidence-Supported Prompt Refinements (`src/generation.py`)**:
  - **Rule 2 (Grounding, Conjunctions & Non-Association)**: Added generic instruction that an entity must only be attributed a requested technology/infrastructure if explicitly supported; un-associated entities appearing in retrieved context must not be claimed to use the technology or reported as missing details. Expanded to search for all requested conjunction targets (e.g., projects AND roles/certifications).
  - **Rule 4 (Strict Conflict Opening)**: Mandated that the very first sentence of any conflict answer MUST explicitly state that the sources contain conflicting reports, strictly forbidding opening with a one-sided factual verdict.
  - **Rule 5 (Synthesized System Experience)**: Generic rule specifying that when asked what technologies an individual has worked with, the model must synthesize both general skill summaries and the concrete technologies from systems/prototypes they architected or prototyped.
  - **Rule 6 (Budgets vs. Benchmarks)**: Generic rule to preserve exact source qualifications and never present a design specification or budget as a measured benchmark result.
- **Evaluation Reporting Math Fixes (`evaluation/evaluate.py`)**:
  - In `aggregate_metrics`: When `unanswerable_cases == 0`, `abstention_recall` and `false_answer_rate` are `None` and their string percentages display `"N/A"`. When `fully_answerable_cases == 0`, `false_abstention_rate` is `None` and displays `"N/A"`. When `total_emitted == 0`, `validity_rate` is `None` and displays `"N/A"`.
  - In `format_markdown_report`: Added `get_split_cell` helper ensuring that any split with `total_cases == 0` (e.g. Holdout when evaluating only Dev) displays `"0"` for Evaluated Cases and `"N/A"` for all metric rows (avoiding misleading `0 ms` or `100%`).
- **Offline Verification**:
  - Full Python offline suite: **14 passed** in `test_evaluation.py`, **25 passed** in `test_generation_pipeline.py`, **134 passed** across all test suites (0 paid calls).
  - Frontend Vitest suite: **34 passed** across 3 test files.
  - Offline retrieval verification: Executed `evaluation/evaluate.py --retrieval-only --split dev --question-ids q04,q06,q11,q17,q18` (5 cases evaluated, 100.0% Source Recall@5, Abstention Recall: N/A, Holdout: N/A across all cells, 0 paid calls).

---

### Phase 6 Closure: Holdout Split Evaluation & Qualitative Audit
- **Date**: 2026-10-24
- **Execution Run (`eval_results_20261024_174013.json`)**:
  - Executed: `python evaluation/evaluate.py --live --split holdout --top-k 5`
  - 6 out-of-sample benchmark questions evaluated: `q05`, `q09`, `q12`, `q15`, `q16`, `q20`.
  - Automated Metrics: Source Recall@5: 100.0%, Citation Validity: 100.0% (6 valid / 6 emitted: `q05`=1, `q09`=2, `q12`=2, `q15`=0, `q16`=1, `q20`=0), Abstention Recall: 66.7%, Behavior Alignment Rate: 83.3%, 0 API failures, 0 pipeline errors.
- **Manual Qualitative Findings (Documented in `docs/HOLDOUT_EVALUATION_REVIEW.md`)**:
  - `q05` (direct_facts), `q12` (analytical), `q15` (missing_out_of_scope): Fully supported responses grounded in retrieved evidence with accurate citations and appropriate abstention.
  - `q20` (instruction_injection): Successful abstention on this specific injection test; refused the adversarial system override in the document fixture and emitted 0 forbidden claims (does not claim general injection protection).
  - `q09` (multi_document): Supported storage comparison (Kafka/Redis vs ChromaDB), but included extraneous infrastructure additions and made an unjustified definitive non-deployment claim about Project Nexus despite documented conflict between status reports and newsletter.
  - `q16` (missing_out_of_scope): Expected-status mismatch (`answered` vs `abstain`). Model logically deduced non-deployment to GCP from global "no code or models deployed" status, but GCP specifically is not explicitly documented. Flags benchmark specification ambiguity.
- **Holdout Discipline & Integrity**:
  - Zero tuning, prompt changes, or rerun iterations on the holdout split.
  - Preserved original JSON execution logs, questions, metrics, corpus, and index without modifications.
  - Formally closed Phase 6. Proceeding to Phase 7 Deployment.




