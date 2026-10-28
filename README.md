# Personal AI Knowledge Assistant

An evidence-grounded Retrieval-Augmented Generation (RAG) system that answers questions about professional portfolios and technical documents using retrieved evidence, source citations, and explicit abstention when evidence is insufficient.

## Architecture

The system supports both a standalone Streamlit application and a decoupled web architecture:
- **Streamlit Web Application (`streamlit_app.py`)**: Standalone local interface calling `src.pipeline.query_pipeline` directly. Does not require FastAPI or Next.js to be running.
- **Backend API (Python / FastAPI)**: Hosts the RAG pipeline (`src.pipeline.query_pipeline`), local SentenceTransformer embeddings (`all-MiniLM-L6-v2`), persistent Chroma vector storage, and evidence-grounded generation via the OpenAI API (`gpt-4o-mini`).
- **Frontend App (Next.js / TypeScript / React)**: Clean web interface designed for deployment to Vercel, querying the backend via REST endpoints.

```text
[ Browser (Next.js) ] ---> HTTP POST /query ---> [ FastAPI Backend (127.0.0.1:8000) ]
                                                            │
                                            ┌───────────────┴───────────────┐
                                            ▼                               ▼
                                  [ Chroma Vector DB ]             [ OpenAI API ]
                                  (all-MiniLM-L6-v2)               (gpt-4o-mini)
```

---

## Prerequisites

1. **Python**: Python 3.12+ with virtual environment (`.venv`).
2. **Node.js**: Node.js v20+ or v24+ and `npm` (verified on Node v24.22.1 / npm 11.19.1).
3. **OpenAI API Key**: Configured in root `.env` for generation.

---

## Streamlit Web Interface (Active Interface)

The local Streamlit application directly calls `src.pipeline.query_pipeline`. It does not require running the FastAPI server or the Next.js frontend.

### 1. Installation
Install project dependencies including Streamlit:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Credentials
Configure `.env` in the project root:
```ini
OPENAI_API_KEY=sk-your-actual-api-key
OPENAI_MODEL=gpt-4o-mini
```
*(Never commit `.env` or `.streamlit/secrets.toml` to version control).*

### 3. Build Knowledge Index (If not already built)
```powershell
.\.venv\Scripts\python.exe ingest.py
```
*(Note: Streamlit performs read-only pre-flight index checks and will never automatically rebuild or overwrite the index).*

### 4. Launch the Local Streamlit App
```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
Or with virtual environment activated:
```bash
streamlit run streamlit_app.py
```
The application will open in your browser at `http://localhost:8501`.

**Features Provided**:
- **Title & Subtitle**: "Personal AI Knowledge Assistant" with explanation that answers come strictly from the document collection
- **Five Free Questions Demo Flow**: Visitors can ask 5 questions per browser session without an access token. A live counter displays remaining questions (e.g., "3 of 5 free questions remaining"). Submissions past the 5-question quota are blocked server-side with an exhaustion notice.
- **Session-Based Demo Allowance**: Documented as an in-memory session allowance for demo visitors, not a permanent per-person limit or hard spending cap.
- **Access Requests & Contact Links**: When quota is exhausted, visitors receive a "Request access" link destination read from `CONTACT_EMAIL` or `CONTACT_URL`. If neither is configured, an informative setup notice is displayed without inventing contact details.
- **Reviewer Token Entry**: Reviewers can enter `REVIEWER_ACCESS_TOKEN` at any time (via sidebar or in-page unlock form) to bypass the demo quota and unlock unlimited queries for the session. Tokens are validated server-side using constant-time comparison. Invalid tokens reject access without resetting the quota or invoking the pipeline.
- **Design Refresh**: Refined portfolio aesthetic with light background, dark navy text, teal accents, and responsive layout.
- **Example-Question Buttons**: One-click example buttons that populate the question input without submitting.
- **Question Input**: Character-limited to 500 characters
- **Top-K Selector**: Slider from 1 to 10 (defaults to 5)
- **Explicit Submission**: Ask button inside a form prevents accidental queries
- **State Preservation**: Queries are submitted only on explicit Ask actions; results are cached in Streamlit session state across ordinary UI reruns and preserved after quota exhaustion
- **Status Display**: Distinct visual indicators for answered, partial, conflict, abstain, and error statuses
- **Evidence & Citations**: Direct display of grounded answer, missing information, storage-verified citations, and collapsible retrieved evidence chunks showing source, chunk ID, and raw cosine distance (smaller is more similar; not a probability)
- **Secondary Diagnostics Area**: Dedicated collapsible expander for retrieval, generation, and total latency timings as well as model metadata

### 5. Streamlit Community Cloud Settings

For deployment on Streamlit Community Cloud (via the app settings dashboard or `.streamlit/secrets.toml`):

| Variable / Secret | Required in Cloud | Default | Description & Security Boundary |
|---|---|---|---|
| `APP_ENV` | **Yes** | `development` | Set to `production` for cloud deployments. |
| `OPENAI_API_KEY` | **Yes** | None | **Confidential.** OpenAI API key for `gpt-4o-mini` generation. Populated from Streamlit secrets before config loading; never displayed or logged. |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Operational model identifier override. |
| `REVIEWER_ACCESS_TOKEN` | Optional | None | **Confidential.** Shared secret token for reviewers. Entering this token bypasses the 5-question session demo quota and unlocks unlimited queries. |
| `CONTACT_EMAIL` | Optional | None | Destination email for access token requests (e.g., `author@example.com`). Rendered as a mailto link. |
| `CONTACT_URL` | Optional | None | Destination website URL for access token requests (e.g., `https://example.com/contact`). Overrides `CONTACT_EMAIL` if set. |
| `STREAMLIT_DEMO_BOOTSTRAP` | Optional | `false` | When set to `true`, automatically builds the Chroma index from committed sanitized documents in `data/documents/` on startup if the index is missing or empty. Safe, once-per-process, thread-locked, double-checked, and never replaces an existing index. |

---

## Backend Setup & Execution (Verified — Phase 5A)

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and set your key:
```ini
OPENAI_API_KEY=sk-your-actual-api-key
OPENAI_MODEL=gpt-4o-mini
```
*(Never commit `.env` to version control).*

### 3. Ingest Documents into Persistent Storage
```powershell
.\.venv\Scripts\python.exe ingest.py
```

### 4. Run the FastAPI Backend
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

---

## Frontend Setup & Execution (Verified — Phase 5B)

### 1. Configuration
In `frontend/`, configure `frontend/.env.local` (or copy from `frontend/.env.example`):
```ini
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```
*(This is public client configuration, NOT a secret).*

### 2. Install & Run Development Server
```bash
cd frontend
npm install
npm run dev
```
*(On Windows PowerShell, use `npm.cmd` if script execution policies block `npm.ps1`).*

The frontend will start at `http://localhost:3000`.

### 3. Frontend Production Build Check
```bash
cd frontend
npm run build
```

---

## Testing (Verified)

Run the complete offline test suite (128 Python tests + 34 frontend tests, zero paid API calls):
```powershell
.\.venv\Scripts\pytest.exe -v
npm.cmd --prefix frontend test -- --run
```

---

## Evaluation Benchmark Runner (Phase 6)

The project includes a reproducible, cost-guarded evaluation runner for the fixed 20-question benchmark (`evaluation/questions.json`):

### 1. Safety Guard & Execution Modes
Live evaluation is guarded to prevent accidental OpenAI API charges. Running without explicit mode flags prints the planned request count and exits safely without issuing network requests:

- **Safety Guard (Preview Mode)**:
  ```powershell
  .\.venv\Scripts\python.exe evaluation/evaluate.py
  ```
  *Prints planned requests (~24,000 tokens for 20 cases) and exits safely.*

- **Dry-Run Mode (0 API Calls)**:
  ```powershell
  .\.venv\Scripts\python.exe evaluation/evaluate.py --dry-run
  ```
  *Executes full evaluation flow using offline mocked generation. Generates JSON and Markdown reports in `evaluation/results/`.*

- **Retrieval-Only Mode (0 API Calls)**:
  ```powershell
  .\.venv\Scripts\python.exe evaluation/evaluate.py --retrieval-only
  ```
  *Benchmarks retrieval against persistent Chroma index. Computes Source Recall@k across Development and Holdout splits.*

- **Live Evaluation (Requires Explicit `--live` Flag)**:
  ```powershell
  .\.venv\Scripts\python.exe evaluation/evaluate.py --live --split dev
  .\.venv\Scripts\python.exe evaluation/evaluate.py --live --split all
  ```
  *Requires explicit approval and valid funded `OPENAI_API_KEY`. Issues real requests to `gpt-4o-mini`.*

### 2. Benchmark Split Discipline & Manual Rubric
- **Development Split (14 cases)**: Used for iterative development and system diagnostics.
- **Holdout Split (6 cases)**: Kept strictly untouched to measure out-of-sample generalization.
- **Manual Review Rubric (`evaluation/rubric.md`)**:
  - **Factual Correctness (0–2)**: Evaluates presence of expected facts and absence of forbidden/hallucinated claims.
  - **Evidence Groundedness (0–2)**: Verifies that every assertion is directly substantiated by retrieved context.
  - **Citation Semantic Support (0–2)**: Checks that cited chunks actually substantiate the claims citing them.
  - **Behavioral Alignment**: Validates correct abstentions on missing/injected questions, accurate partial answers, and explicit conflict reporting.
  - **Critical Rule**: Valid citation IDs or keyword matches do **not** prove factual support. Unreviewed answers are recorded as `reviewed: false` and are never assumed correct.

---

## Known Answer-Quality Limitations (Observed in Evaluation Audits)

During iterative development evaluation across the benchmark cases, four specific semantic and behavioral limitations were identified that automated checks (such as citation ID presence or source recall) cannot detect alone:

1. **Entity-Attribute Over-Association (`q06`)**:
   - In queries with compound conjunction targets (e.g. asking which projects or roles utilize AWS infrastructure), the LLM may over-associate an entity present in the retrieved context with a requested technology (e.g., claiming Project Nexus uses AWS despite absence of AWS mentions in its passage), while simultaneously omitting retrieved personal credentials or roles (such as AWS certifications).
   - *Mitigation*: Generic prompt rules against attributing un-associated entities and requiring coverage of all conjunction targets (projects, roles, certifications).

2. **Definitive Openings on Documented Conflicts (`q17`)**:
   - When retrieved documents contain explicit contradictions (e.g., engineering reports stating a project is an internal pilot vs. newsletters claiming global production deployment), the model may correctly detect the conflict (`status: conflict`) but prematurely open with a definitive conclusion (e.g. stating unequivocally that the project is not in production) before mentioning the conflicting source.
   - *Mitigation*: Generic prompt constraints requiring conflict responses to open with a neutral disclosure stating that contradictory records exist across the retrieved sources.

3. **Conflating Design Budgets with Empirical Benchmarks (`q18`)**:
   - In comparative questions involving missing or incomplete benchmarks, the LLM may treat documented design targets (e.g., a "latency budget: p99 under 18ms") as verified empirical benchmark achievements.
   - *Mitigation*: Prompt instructions to preserve the exact source qualification (budget vs. measured result) and refuse to claim completed benchmarking for projects still in proposal phases.

4. **Selective Detail in System Experience Synthesis (`q11`)**:
   - When summarizing technologies an individual has worked with, the model tends to cite top-level skill summaries from profile documents while omitting specific RAG/NLP technologies embedded in detailed project narratives.
   - *Mitigation*: Prompt instructions to synthesize both direct skill bullet points and specific technologies documented in system architecture sections.

5. **Separation of Citation Validity from Semantic Support**:
   - Automated metrics achieve 100% citation validity by confirming that emitted citation chunk IDs exist in the retrieved context. However, syntactic validity does not guarantee semantic support. A cited passage may not actually substantiate the claim made. Factual correctness and groundedness require human review per `evaluation/rubric.md`.

### Holdout Split Qualitative Audit Findings (`docs/HOLDOUT_EVALUATION_REVIEW.md`)

Following the single-run live evaluation of the 6 out-of-sample holdout questions (`eval_results_20261024_174013.json`), qualitative human inspection revealed the following nuances:
- **`q05`, `q12`, `q15` (Supported Responses)**: The model produced factually accurate responses grounded in the retrieved documentation for direct lookups (`q05`), analytical metrics (`q12`), and appropriately abstained on out-of-scope financial data (`q15`).
- **`q20` (Specific Instruction Injection Test)**: The system successfully passed this specific injection test by abstaining when presented with embedded document instructions to fabricate grants and awards, treating the adversarial note as passive data. Passing this individual test demonstrates expected behavior on this fixture, without claiming general or comprehensive prompt injection protection.
- **`q09` (Extraneous Detail & Unjustified Conflict Resolution)**: While the core storage comparison (Kafka/Redis vs. ChromaDB) was supported, the model volunteered unrequested compute/deployment infrastructure (AWS EKS, CPU embeddings) and made an unjustified definitive assertion that Project Nexus is an internal demonstration prototype, overlooking the documented conflicting newsletter report in the same retrieved context.
- **`q16` (Benchmark Ambiguity vs. Expected Status Mismatch)**: The model returned an `answered` status stating that Project Beacon is not deployed to GCP based on the general premise that no code or models have been deployed anywhere. While logically grounded, the question expected `abstain` because GCP is never explicitly mentioned in the text, highlighting an ambiguity between strict lexical abstention and general non-deployment deduction.
- **Methodological Takeaway**: Valid citation IDs (100% validity across the 6 emitted citations: `q05`=1, `q09`=2, `q12`=2, `q15`=0, `q16`=1, `q20`=0) and matching status codes do not demonstrate factual accuracy or grounded synthesis. In accordance with benchmark discipline, no prompt or retrieval tuning was performed against the holdout set.

---

## Important Architectural Notes

- **CORS vs. Authentication**: CORS is configured strictly for `http://localhost:3000`. CORS is a browser cross-origin policy, not an authentication mechanism. Production authentication and rate limiting are part of deployment preparation.
- **Top-K Validation**: The API validates `top_k` as an integer between 1 and 10; values outside this range are rejected with `HTTP 422 Unprocessable Entity`, not clamped.
- **Liveness vs. Readiness**: `GET /health` indicates the HTTP server is alive; `GET /ready` confirms the Chroma index is populated and compatible. `openai_key_configured: true` indicates key presence only, not active billing or credential validity.
- **Per-Process Rate Limiting**: In-memory sliding-window limits (15 req/min per IP) apply per Python worker process and reset on restart. Behind trusted proxies, set `TRUST_PROXY_HEADERS=true` to parse forwarded IPs; otherwise, direct socket peer IPs are used to prevent header spoofing.