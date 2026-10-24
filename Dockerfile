# ==============================================================================
# Dockerfile for Personal AI Knowledge Assistant Backend (Phase 5C)
#
# Operational Principles:
# 1. Runs as a non-root system user (appuser, UID 10001).
# 2. Configurable listening port via PORT (default 8000).
# 3. Explicit embedding model cache at /app/.cache/huggingface (HF_HOME).
# 4. Configurable persistent vector storage at /app/storage/chroma (CHROMA_PERSIST_DIRECTORY).
# 5. Does NOT automatically invoke OpenAI or rebuild the index at startup.
# 6. Starts with a single application worker initially (--workers 1).
# ==============================================================================

FROM python:3.12-slim

# Set environment variables for clean container execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    CHROMA_PERSIST_DIRECTORY=/app/storage/chroma \
    DOCUMENTS_DIRECTORY=/app/data/documents \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

# Install essential runtime tools (curl for healthchecks)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create a dedicated non-root user and group
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Install Python dependencies before copying source code for optimal layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Create necessary runtime directories and grant ownership to non-root appuser
RUN mkdir -p /app/storage/chroma \
             /app/.cache/huggingface \
             /app/data/documents && \
    chown -R appuser:appgroup /app

# Copy application source code, corpus documents, and ingestion script
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup data/ /app/data/
COPY --chown=appuser:appgroup ingest.py /app/ingest.py

# Switch to non-root user for all subsequent operations
USER appuser

# Expose the configured HTTP listening port
EXPOSE ${PORT}

# Health check probe against liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the FastAPI application with a single worker process
CMD ["sh", "-c", "uvicorn src.api:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers 1"]
