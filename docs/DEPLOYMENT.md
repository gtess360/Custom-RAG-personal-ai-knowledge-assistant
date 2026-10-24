# Deployment Guide: Personal AI Knowledge Assistant

This guide documents the deployment architecture, configuration parameters, operational workflows, and security protections for the **Personal AI Knowledge Assistant**.

---

## 1. Architectural Overview

The application follows a decoupled architecture featuring a Next.js server-side proxy (Backend-for-Frontend / BFF) and a Python FastAPI service backed by persistent vector storage:

```
┌─────────────────────────────────┐           HTTPS (REST)            ┌─────────────────────────────────────────┐
│     Client Browser              │──────────────────────────────────▶│        Next.js Server Proxy (BFF)       │
│     (React 19 Frontend)         │◀──────────────────────────────────│        (App Router: /api/query)         │
│                                 │   Relative: /api/query            │                                         │
│ Session Token in sessionStorage │   Auth: Reviewer Bearer Token     │ Validates: REVIEWER_ACCESS_TOKEN        │
└─────────────────────────────────┘                                   │ Injects:   BACKEND_API_KEY (Server Secret│
                                                                      └────────────────────┬────────────────────┘
                                                                                           │ Private Network / HTTPS
                                                                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│        Python Backend Service (FastAPI + Uvicorn)                                                             │
│                                                                                                               │
│        - Validates: DEMO_ACCESS_TOKEN (from BACKEND_API_KEY)                                                  │
│        - IP Protection: Untrusted X-Forwarded-For ignored unless TRUST_PROXY_HEADERS=true                     │
│        - In-Memory Protections: Semaphore (max 3 concurrent), Rate Limiter (15 req/min per IP, per process)  │
│        - Payload Limits: Body <= 16 KB, Question <= 500 characters                                           │
└──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │             Persistent Storage & Cache      │
                │ 1. ChromaDB Vector Store: ./storage/chroma  │
                │ 2. Model Weights Cache:   ./.cache/hf       │
                └─────────────────────────────────────────────┘
```

---

## 2. Deployment Routes

### Primary Route: Direct Python Host Deployment (Recommended)
This is the verified deployment route that does not require Docker on the host system:
- **Operating Environment**: Any persistent Linux host, Virtual Machine, or VPS (e.g. Ubuntu 22.04/24.04 on AWS EC2, DigitalOcean Droplet, Hetzner, or persistent PaaS like Render native Python).
- **Python Version**: Python 3.10, 3.11, or 3.12.
- **Process Supervision**: Systemd service, Supervisord, or PM2 managing `uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1`.
- **Chroma Storage**: Mounted local directory (`./storage/chroma` or `/var/lib/rag-assistant/storage/chroma`).
- **Embedding Cache**: Explicit cache directory (`./.cache/huggingface` or `/var/cache/rag-assistant/huggingface`).

### Optional Route: Docker Container (Untested Locally)
A `Dockerfile` and `.dockerignore` are provided in the repository root as an **optional, untested deployment route** for containerized platforms (e.g. AWS ECS, Fly.io, Railway, Kubernetes). 
> [!IMPORTANT]
> Docker CLI is not installed on the local development machine, so container builds and container runtime behavior have not been tested locally. Container startup must not be assumed to be instantaneous; runtime initialization requires loading PyTorch (~1.1 GB in memory), mounting persistent storage, and loading embedding weights.

---

## 3. Environment Variables & Security Matrix

Environment variables are partitioned strictly by security boundary. **Backend credentials must never be exposed to browser bundles.**

### Frontend (Next.js Server-Side Settings)

| Variable | Required | Default | Security Boundary | Purpose & Constraints |
|---|---|---|---|---|
| `APP_ENV` | **Yes** in prod | `development` | Server-Only | Set to `production` for live deployments or `development` for local pairing. In production, missing credentials refuse to serve. |
| `BACKEND_URL` | **Yes** in prod | `http://127.0.0.1:8000` | Server-Only | Internal URL of the Python FastAPI backend service. Not accessible to public internet. |
| `BACKEND_API_KEY` | **Yes** in prod | None | **Confidential** (Server-Only) | Shared secret required to authenticate with the backend (`DEMO_ACCESS_TOKEN`). **Never expose in browser code or prefix with `NEXT_PUBLIC_`.** |
| `REVIEWER_ACCESS_TOKEN`| **Yes** in prod | None | **Confidential** (Server-Only) | Secret token required from reviewers to execute queries through the Next.js proxy. In production, proxy returns HTTP 500 if unset. |

> [!CAUTION]
> **Zero Client Secrets:** The frontend codebase does not expose `OPENAI_API_KEY` or `BACKEND_API_KEY` to the client. Browser requests target the relative endpoint `/api/query`.

---

### Backend (Python FastAPI Host Environment)

| Variable | Required | Default | Security Boundary | Purpose & Constraints |
|---|---|---|---|---|
| `APP_ENV` | **Yes** in prod | `development` | Server Environment | Set to `production` or `development`. When set to `production`, backend **refuses to start** if `DEMO_ACCESS_TOKEN` is missing. |
| `OPENAI_API_KEY` | **Yes** | None | **Confidential** | Secret key for OpenAI API (`gpt-4o-mini`). Kept strictly server-side. |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Operational | Generation model name. |
| `DEMO_ACCESS_TOKEN` | **Yes** in prod | Empty in dev | **Confidential** | Shared secret token validated against `Authorization: Bearer <token>`. In `APP_ENV=production`, startup fails if unset. In `development`, open access is permitted if unset. |
| `ALLOWED_ORIGINS` | **Yes** | `http://localhost:3000,http://127.0.0.1:3000` | Security Policy | Comma-separated list of exact allowed frontend origins (e.g. `https://my-rag-app.vercel.app`). Wildcards (`*`) are strictly prohibited and stripped. |
| `TRUST_PROXY_HEADERS` | No | `false` | Security Policy | When `false` (default), uses direct socket peer IP (`request.client.host`) and **ignores arbitrary `X-Forwarded-For` headers** to prevent spoofed header bypass. Set `true` only behind a trusted reverse proxy. |
| `CHROMA_PERSIST_DIRECTORY`| No | `./storage/chroma` | Operational | Directory for persistent Chroma vector database. |
| `DOCUMENTS_DIRECTORY` | No | `./data/documents` | Operational | Directory containing corpus documents for ingestion. |
| `HF_HOME` | No | `./.cache/huggingface` | Operational | Explicit cache location for HuggingFace model weights. |
| `SENTENCE_TRANSFORMERS_HOME` | No | `./.cache/huggingface` | Operational | Explicit cache location for SentenceTransformer weights. |
| `PORT` | No | `8000` | Operational | TCP listening port. |
| `HOST` | No | `127.0.0.1` (`0.0.0.0` in container) | Operational | Network interface binding. |
| `MAX_CONCURRENT_QUERIES` | No | `3` | Protection | Concurrency semaphore bounding simultaneous RAG pipeline runs. |
| `RATE_LIMIT_PER_MINUTE` | No | `15` | Protection | In-memory sliding-window request cap per client IP. |

---

## 4. Authentication & Reviewer Access Flow

Merely hiding the backend token inside a server-side proxy does not protect the system if the proxy itself is publicly callable without authentication. To prevent unauthorized proxy abuse and LLM credit exhaustion, a single, documented **Reviewer Access Flow** is enforced:

### The Reviewer Access Workflow
1. **Passphrase Distribution**: The deployment administrator generates a secure reviewer token (e.g. `openssl rand -hex 16`) and sets it as `REVIEWER_ACCESS_TOKEN` on the Next.js server.
2. **Reviewer UI Session**:
   - The reviewer opens the deployed application.
   - The reviewer clicks the **"🔑 Reviewer Access"** button in the navigation bar.
   - The reviewer inputs the provided reviewer token.
   - The token is saved **only** in the browser's `sessionStorage` (`reviewer_access_token`) for the active session. It is never persisted to disk or sent to analytics.
3. **Proxy Request Execution**:
   - When the reviewer submits a query, `frontend/src/lib/api.ts` sends a `POST /api/query` request with header `Authorization: Bearer <reviewer-token>`.
4. **Proxy Verification**:
   - Next.js route handler (`frontend/src/app/api/query/route.ts`) checks the header against `process.env.REVIEWER_ACCESS_TOKEN`.
   - If missing or mismatched: Returns `HTTP 401 Unauthorized` (`detail: "Unauthorized reviewer access. A valid reviewer access token is required."`).
5. **Backend Dispatch**:
   - Upon successful reviewer validation, the Next.js proxy injects the server-only `BACKEND_API_KEY` into `Authorization: Bearer <BACKEND_API_KEY>` and forwards the request to `${BACKEND_URL}/query`.
   - The backend validates `DEMO_ACCESS_TOKEN` and executes the pipeline.
6. **Local Development Preservation**:
   - When `APP_ENV=development` and `REVIEWER_ACCESS_TOKEN` is unset, the proxy permits open queries without requiring a token, preserving rapid local development and test suite execution.

---

## 5. Production Configuration Enforcement (Fail Closed)

To eliminate configuration drift and accidental exposure, an explicit environment setting is enforced across both services:

- **Backend Enforcement**:
  - `APP_ENV` is **mandatory** and must be explicitly set to `'development'` or `'production'`.
  - Missing, empty, or unrecognized values (e.g. `staging`, `prod`, `unknown`) are rejected during startup lifespan with an actionable `RuntimeError` (`"Backend startup failed: APP_ENV must be explicitly configured"`).
  - When `APP_ENV=production`, `DEMO_ACCESS_TOKEN` is mandatory; the backend refuses to start without it.
  - When `APP_ENV=development`, open local development without access tokens is permitted. Local setup templates (`.env.example` and local `.env`) document `APP_ENV=development`.

- **Frontend Proxy Enforcement**:
  - **Fails Closed on `NODE_ENV=production`**: If `NODE_ENV=production`, production authentication is required **even if `APP_ENV` is missing or mistakenly set to `development`**.
  - **Required Production Settings**: The proxy verifies that `BACKEND_URL`, `BACKEND_API_KEY`, and `REVIEWER_ACCESS_TOKEN` are all configured and non-empty.
  - **Pre-Forwarding Rejection**: If any required setting is missing in production, the proxy immediately returns `HTTP 500 Internal Server Error` with a safe configuration message **before reading the query body or forwarding any request**.
  - When running in local development (`NODE_ENV!=production` and `APP_ENV=development`), open queries are permitted when `REVIEWER_ACCESS_TOKEN` is unset.

---

## 6. Rate Limiting Reality & Client IP Determination

### In-Memory Scope and Reset Behavior
- **Per-Process Limitation**: The rate limiter (`RATE_LIMIT_PER_MINUTE=15`) is implemented in Python process memory (`_ip_request_timestamps` sliding-window dictionary).
- **Multi-Process / Multi-Instance Caveat**: In multi-worker or horizontally scaled setups, rate limits are enforced independently per worker process; state is not synchronized across workers.
- **Restart Reset**: When the Python process restarts, reloads, or redeploys, all in-memory timestamp windows reset to zero.
- **Distributed Needs**: For strict cluster-wide rate limiting across multiple instances, an external key-value store (e.g. Redis) is required.

### Client IP Determination Behind Proxies
- **Default Policy (`TRUST_PROXY_HEADERS=false`)**:
  - The backend uses the direct TCP socket IP (`raw_request.client.host`).
  - **Arbitrary forwarded headers (`X-Forwarded-For`) are completely ignored**. This prevents malicious clients from spoofing client IPs to bypass the sliding-window limit.
- **Explicit Proxy Trust (`TRUST_PROXY_HEADERS=true`)**:
  - Must only be enabled when the backend is deployed in an isolated private network behind a trusted reverse proxy (such as the Next.js BFF proxy or an AWS ALB) that strips and overwrites client headers.
  - When enabled, the backend extracts the client IP from the first entry of `X-Forwarded-For`.

---

## 7. Direct Python Deployment Runbook

Follow these steps to deploy the backend directly on a persistent Linux host without Docker:

### Step 1: Provision Host and Prerequisites
- Ubuntu 22.04 LTS or 24.04 LTS with at least **2 GB RAM** and **1 vCPU**.
- Install Python 3.10+ and python3-venv:
  ```bash
  sudo apt update && sudo apt install -y python3 python3-venv python3-pip
  ```

### Step 2: Clone and Setup Virtual Environment
```bash
git clone <repo-url> /opt/rag-assistant
cd /opt/rag-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create `/opt/rag-assistant/.env`:
```ini
APP_ENV=production
OPENAI_API_KEY=sk-your-actual-openai-api-key
OPENAI_MODEL=gpt-4o-mini
DEMO_ACCESS_TOKEN=your-randomly-generated-backend-secret
ALLOWED_ORIGINS=https://your-frontend.vercel.app
TRUST_PROXY_HEADERS=false
HOST=127.0.0.1
PORT=8000
CHROMA_PERSIST_DIRECTORY=/opt/rag-assistant/storage/chroma
DOCUMENTS_DIRECTORY=/opt/rag-assistant/data/documents
HF_HOME=/opt/rag-assistant/.cache/huggingface
SENTENCE_TRANSFORMERS_HOME=/opt/rag-assistant/.cache/huggingface
```

### Step 4: Run Corpus Ingestion Once
Execute ingestion to embed documents into persistent Chroma storage:
```bash
source .venv/bin/activate
python ingest.py
```
Verify that Chroma creates the database files in `/opt/rag-assistant/storage/chroma`.

### Step 5: Configure Systemd Service
Create `/etc/systemd/system/rag-assistant.service`:
```ini
[Unit]
Description=Personal AI Knowledge Assistant Backend (FastAPI)
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/rag-assistant
EnvironmentFile=/opt/rag-assistant/.env
ExecStart=/opt/rag-assistant/.venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```
Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rag-assistant
sudo systemctl status rag-assistant
```

---

## 8. Verification: Liveness vs. Readiness Semantics

Distinguish service liveness from index readiness:

### Liveness Probe (`GET /health`)
- Confirms the HTTP process is alive and accepting requests.
- Returns `200 OK`: `{"status": "ok", "service": "personal-ai-knowledge-assistant"}`.
- Use this for process supervisor health checks.

### Readiness Probe (`GET /ready`)
- Confirms the Chroma index exists, is compatible, and contains indexed chunks (`collection.count() > 0`), and confirms `OPENAI_API_KEY` is present.
- Returns `200 OK`:
  ```json
  {
    "status": "ready",
    "service": "personal-ai-knowledge-assistant",
    "document_chunks": 4,
    "model": "gpt-4o-mini",
    "openai_key_configured": true
  }
  ```
- **Important Note on Credential Validity**:
  A configured API key (`openai_key_configured: true`) confirms key presence in the environment only; **it is NOT proof of valid credentials, active billing availability, or a successful model request.** Verifying live model requests requires an end-to-end query test, not a repeated readiness probe.

---

## 9. Cost Controls, Retries, and Spending Uncertainty

### Built-in Structural Cost Bounding
The system includes technical guardrails to mitigate unintentional OpenAI API spending:
1. **Bounded Input Size**: Requests exceeding 16 KB are rejected with `HTTP 413`. Questions exceeding 500 characters are rejected with `HTTP 422`.
2. **Bounded Evidence Volume**: Candidate chunk retrieval is capped at `top_k <= 10` chunks (~4,000 tokens maximum input context).
3. **Bounded Output Tokens**: Answer generation requests are constrained to `max_tokens=1024`.
4. **Rate Limiting & Concurrency**: In-memory rate limits (15 queries/min per IP) and concurrency semaphore (3 concurrent executions) prevent burst exhaustion from a single client.

### Retry Architecture: SDK Retries vs. User-Initiated Retries
Retry mechanisms are explicitly separated across system layers:
- **Automatic SDK Retries (Backend)**:
  - The backend's OpenAI client (`src/generation.py`) is initialized with `max_retries=2`.
  - On transient network connection drops, temporary provider timeouts, or HTTP 429 / 5xx responses, the official OpenAI Python SDK automatically retries up to 2 times with exponential backoff before surfacing an error.
  - These internal retries occur transparently within a single pipeline execution to protect against transient provider hiccups.
- **User-Initiated Retries (Frontend UI)**:
  - The Next.js frontend UI (`frontend/src/app/page.tsx`) contains **zero automatic retry loops**.
  - When an error occurs (such as a 401, 429, 500, or 502), the error message is presented in a prominent error banner.
  - Retries occur strictly when an operator or reviewer manually clicks the **"Retry"** button. This ensures failed requests never trigger cascading retry storms or automated quota drain.

### Spending Uncertainty: Provider Budget Alerts vs. Enforced Limits
> [!WARNING]
> **No Application-Level Spending Cap**: The application code does **not** enforce a hard cumulative monthly financial cap. Rate limits and token bounds throttle per-minute consumption, but cumulative spending remains unbounded if queries accumulate across users or time.
>
> **Provider Dashboard Budgets — Alerts vs. Enforced Limits**:
> - **Notification Alerts**: Configuring a budget in provider dashboards often sets only soft notification thresholds (e.g. email warnings when spend reaches $10, $50, or $100). **These alerts do NOT halt API traffic or prevent charges.**
> - **Enforced Usage Limits**: To prevent unexpected costs, an operator must verify whether the provider's billing dashboard supports a strictly enforced hard monthly limit (usage cutoff) and confirm it is active. Do not assume or promise that provider budgets are hard spending caps unless explicitly verified in the billing settings. Even with hard caps enabled, delayed usage aggregation and concurrent in-flight requests may incur minor overages before the cutoff takes effect.

