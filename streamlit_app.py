"""Streamlit web interface for the Personal AI Knowledge Assistant.

Directly calls src.pipeline.query_pipeline without requiring FastAPI or Next.js.
Supports Streamlit Community Cloud deployment with:
- Pre-import Streamlit secrets propagation to os.environ (preserving local .env)
- Explicit STREAMLIT_DEMO_BOOTSTRAP mode for automatic missing-index initialization
- Thread-locked, double-checked, once-per-process bootstrap guard
- Production access protection requiring REVIEWER_ACCESS_TOKEN (failing closed if unset)
- Configurable candidate retrieval (top_k between 1 and 10, default 5)
- Character-limited question input (<= 500 characters)
- Explicit form submission with session-state caching across UI reruns
- Clear rendering of status (answered, partial, conflict, abstain, error)
- Grounded answers, missing information, storage-verified citations, and expandable evidence
- Latency timing breakdowns (retrieval, generation, total)
- Safe HTML rendering and strict credential protection
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Optional
import streamlit as st

logger = logging.getLogger("streamlit_app")


# -----------------------------------------------------------------------------
# 1. Streamlit Cloud Secrets Propagation (Pre-Config Import)
# -----------------------------------------------------------------------------

def _sync_streamlit_secrets_to_env() -> None:
    """Propagate Streamlit Cloud secrets to os.environ before config imports.

    Streamlit Cloud secrets override local defaults while preserving
    any environment variables not defined in st.secrets.
    Never prints or logs secret values.
    """
    try:
        if hasattr(st, "secrets"):
            for key, val in st.secrets.items():
                if isinstance(val, (str, int, float, bool)):
                    os.environ[key] = str(val)
    except Exception:
        pass


_sync_streamlit_secrets_to_env()

# Now import configuration and pipeline modules (which read environment variables)
from src.config import (
    PATHS,
    OPENAI_MODEL,
    DEFAULT_TOP_K,
    get_safe_config_summary,
    has_openai_api_key,
)
import src.pipeline as pipeline
import src.vector_store as vector_store


# -----------------------------------------------------------------------------
# 2. Process-Level Demo Bootstrap Mechanism
# -----------------------------------------------------------------------------

_bootstrap_lock = threading.Lock()
_bootstrap_attempted: bool = False
_bootstrap_error: Optional[str] = None


def is_bootstrap_enabled() -> bool:
    """Check if demo bootstrap mode is explicitly enabled."""
    val = os.getenv("STREAMLIT_DEMO_BOOTSTRAP", "").strip().lower()
    return val in ("true", "1", "yes", "enabled")


def reset_bootstrap_state_for_tests() -> None:
    """Reset process-level bootstrap tracking flags (used only in tests)."""
    global _bootstrap_attempted, _bootstrap_error
    with _bootstrap_lock:
        _bootstrap_attempted = False
        _bootstrap_error = None


def maybe_bootstrap_index(
    storage_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> tuple[bool, Optional[str]]:
    """Bootstrap Chroma index if STREAMLIT_DEMO_BOOTSTRAP is enabled and index is missing or empty.

    Concurrency & Safety Guarantees:
    - Thread-safe: Uses process-level lock and double-checked inspection.
    - Idempotent: Executes at most once per process.
    - Read-only condition: Only bootstraps if collection is missing or collection.count() == 0.
    - Never rebuilds or overwrites an existing non-empty index (compatible or incompatible).
    - Never replaces an existing local non-empty index.
    - Blocks subsequent queries if bootstrap fails.

    Returns:
        tuple[bool, Optional[str]]: (success, error_message_if_any)
    """
    global _bootstrap_attempted, _bootstrap_error

    if not is_bootstrap_enabled():
        return True, None

    if _bootstrap_error is not None:
        return False, _bootstrap_error

    target_storage = storage_dir or PATHS.storage_dir
    target_data = data_dir or PATHS.data_dir

    # 1. First index check (outside lock)
    try:
        client = vector_store._get_client(target_storage)
        try:
            coll = client.get_collection(vector_store.COLLECTION_NAME)
            needs_bootstrap = (coll.count() == 0)
        except Exception:
            needs_bootstrap = True
    except Exception as exc:
        _bootstrap_error = f"Bootstrap pre-check error: {exc}"
        logger.error("Bootstrap pre-check error: %s\n%s", exc, traceback.format_exc())
        return False, _bootstrap_error

    # If index already has chunks, do not bootstrap and NEVER replace it
    if not needs_bootstrap:
        return True, None

    # 2. Acquire lock for process-level single initialization
    with _bootstrap_lock:
        if _bootstrap_attempted:
            if _bootstrap_error is not None:
                return False, _bootstrap_error
            return True, None

        # 3. Second index check (inside lock)
        try:
            client = vector_store._get_client(target_storage)
            try:
                coll = client.get_collection(vector_store.COLLECTION_NAME)
                still_needs_bootstrap = (coll.count() == 0)
            except Exception:
                still_needs_bootstrap = True
        except Exception as exc:
            _bootstrap_attempted = True
            _bootstrap_error = f"Bootstrap locked-check error: {exc}"
            logger.error("Bootstrap locked-check error: %s\n%s", exc, traceback.format_exc())
            return False, _bootstrap_error

        if not still_needs_bootstrap:
            _bootstrap_attempted = True
            return True, None

        # 4. Perform ingestion from committed sanitized files
        try:
            from ingest import ingest_corpus
            ingest_corpus(
                force_rebuild=False,
                allow_partial_ingestion=False,
                data_dir=target_data,
                storage_dir=target_storage,
            )
            _bootstrap_attempted = True
            return True, None
        except Exception as exc:
            _bootstrap_attempted = True
            _bootstrap_error = f"STREAMLIT_DEMO_BOOTSTRAP failed: {exc}"
            from src.embeddings import get_embedding_package_versions
            versions = get_embedding_package_versions()
            logger.error(
                "STREAMLIT_DEMO_BOOTSTRAP failed: %s (packages: %s)\nTraceback:\n%s",
                exc,
                versions,
                traceback.format_exc(),
            )
            return False, _bootstrap_error


# -----------------------------------------------------------------------------
# 3. Read-Only Index Health Check
# -----------------------------------------------------------------------------

def check_index_health(storage_dir: Optional[Path] = None) -> tuple[bool, str]:
    """Check if the persistent Chroma index exists, is populated, and is compatible.

    Per architecture guidelines, this inspection is strictly read-only.
    It NEVER rebuilds or alters the index automatically.

    Returns:
        tuple[bool, str]: (is_healthy, message)
    """
    try:
        target_dir = storage_dir or PATHS.storage_dir
        client = vector_store._get_client(target_dir)

        try:
            collection = client.get_collection(vector_store.COLLECTION_NAME)
        except Exception:
            return (
                False,
                f"Chroma index collection '{vector_store.COLLECTION_NAME}' not found at '{target_dir}'. "
                "Action required: Run 'python ingest.py' from your terminal to build the knowledge index before submitting queries.",
            )

        count = collection.count()
        if count == 0:
            return (
                False,
                f"Chroma index at '{target_dir}' exists but is empty (0 chunks). "
                "Action required: Run 'python ingest.py' to populate the index.",
            )

        compatible, compat_msg = vector_store.check_index_compatibility(target_dir)
        if not compatible:
            return (
                False,
                f"Chroma index is incompatible with current configuration: {compat_msg} "
                "Action required: Run 'python ingest.py' to rebuild the index.",
            )

        return True, f"Index ready ({count} chunks indexed)."
    except Exception as exc:
        return (
            False,
            f"Error inspecting Chroma index: {exc}. "
            "Action required: Verify storage directory or run 'python ingest.py'.",
        )


# -----------------------------------------------------------------------------
# 4. Production Reviewer Access Protection
# -----------------------------------------------------------------------------

def check_production_access() -> bool:
    """Enforce REVIEWER_ACCESS_TOKEN authentication in production (APP_ENV=production).

    Fails closed if REVIEWER_ACCESS_TOKEN is missing or empty in production mode.
    Does not require FastAPI authentication because Streamlit calls the pipeline directly.
    In development mode, allows open access for local workflow.

    Returns:
        bool: True if authorized to view query interface, False otherwise.
    """
    app_env = os.getenv("APP_ENV", "").strip().lower()
    is_prod = (app_env == "production")

    if not is_prod:
        return True

    expected_token = os.getenv("REVIEWER_ACCESS_TOKEN", "").strip()
    if not expected_token:
        st.error(
            "🔒 **Production Access Blocked:** `REVIEWER_ACCESS_TOKEN` is not configured on the server. "
            "In production mode (`APP_ENV=production`), queries are strictly disabled until an access token is provided."
        )
        return False

    if st.session_state.get("authenticated", False):
        return True

    st.subheader("🔒 Reviewer Authentication")
    st.info("This instance is running in production mode. Please enter the reviewer access token to unlock the assistant.")

    with st.form("reviewer_auth_form"):
        token_input = st.text_input(
            "Reviewer Access Token",
            type="password",
            placeholder="Enter reviewer access token...",
            help="Provided by the project administrator.",
        )
        submitted = st.form_submit_button("Unlock Assistant", type="primary")

    if submitted:
        if token_input and hmac.compare_digest(token_input.strip(), expected_token):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid access token. Access denied.")

    return False


# -----------------------------------------------------------------------------
# 5. Application Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Personal AI Knowledge Assistant",
        page_icon="🧠",
        layout="wide",
    )

    # Initialize session state for persistent results across reruns
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Title & Subtitle
    st.title("Personal AI Knowledge Assistant")
    st.caption("Evidence-grounded local knowledge assistant powered by ChromaDB and OpenAI.")

    # 1. Execute optional bootstrap if STREAMLIT_DEMO_BOOTSTRAP is enabled
    bootstrap_ok, bootstrap_err = maybe_bootstrap_index()

    # 2. Check index health
    if bootstrap_ok:
        is_healthy, health_msg = check_index_health()
    else:
        is_healthy = False
        health_msg = f"Bootstrap initialization failed: {bootstrap_err}"

    # 3. Sidebar: Safe System & Index Status
    app_env = os.getenv("APP_ENV", "").strip().lower()
    is_prod = (app_env == "production")

    with st.sidebar:
        st.header("System Status")
        if is_healthy:
            st.success(f"Index: {health_msg}")
        else:
            st.error(f"Index Warning: {health_msg}")

        st.subheader("Configuration")
        safe_cfg = get_safe_config_summary()
        st.text(f"Environment: {safe_cfg.get('app_env', 'development')}")
        st.text(f"OpenAI Model: {safe_cfg.get('openai_model', OPENAI_MODEL)}")
        st.text(f"API Key Configured: {'Yes' if has_openai_api_key() else 'No'}")
        st.text(f"Demo Bootstrap: {'Enabled' if is_bootstrap_enabled() else 'Disabled'}")
        if is_prod:
            has_token = bool(os.getenv("REVIEWER_ACCESS_TOKEN", "").strip())
            st.text(f"Reviewer Token: {'Configured' if has_token else 'Missing'}")

        st.divider()
        if st.button("Clear Saved Result", use_container_width=True):
            st.session_state["last_result"] = None
            st.rerun()

        if is_prod and st.session_state.get("authenticated", False):
            if st.button("Lock Assistant", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["last_result"] = None
                st.rerun()

    # 4. Enforce Production Reviewer Access Gate
    if not check_production_access():
        return

    # 5. Display Index Warning Banner if Unhealthy
    if not is_healthy:
        st.error(f"⚠️ **Index Issue Detected:** {health_msg}")

    # 6. Query Form
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            max_chars=500,
            placeholder="Ask a question about the indexed knowledge base (maximum 500 characters)...",
            help="Queries are answered strictly using retrieved evidence. External ungrounded knowledge is rejected.",
        )
        top_k = st.slider(
            "Number of evidence chunks (top_k)",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            step=1,
            help="Number of candidate evidence passages to retrieve from the vector index (default 5).",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    # 7. Handle Explicit Form Submission
    if submitted:
        clean_q = (question or "").strip()
        if not clean_q:
            st.warning("Please enter a non-empty question before clicking Ask.")
        elif len(clean_q) > 500:
            st.error("Question exceeds the 500-character limit. Please shorten your question.")
        elif not is_healthy:
            st.error(
                f"Cannot execute query because the Chroma index is not ready:\n\n{health_msg}"
            )
        else:
            with st.spinner("Retrieving evidence and generating grounded answer..."):
                result = pipeline.query_pipeline(question=clean_q, top_k=top_k)
                st.session_state["last_result"] = result

    # 8. Render Last Result from Session State
    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()

        # Status Display
        status = result.get("status", "unknown")
        if status == "answered":
            st.success("✅ **Status: Answered** — Grounded answer generated from retrieved evidence.")
        elif status == "partial":
            st.warning("⚠️ **Status: Partial** — Grounded answer provided with some requested information missing.")
        elif status == "conflict":
            st.warning("⚠️ **Status: Conflict** — Contradictory evidence detected in the indexed documents.")
        elif status == "abstain":
            st.info("ℹ️ **Status: Abstain** — Insufficient evidence found in the indexed corpus to answer the question.")
        elif status == "error":
            err = result.get("error") or "An error occurred during query processing."
            st.error(f"❌ **Status: Error** — {err}")
        else:
            st.info(f"**Status: {status.title()}**")

        # Answer
        st.subheader("Answer")
        answer_text = result.get("answer", "")
        if answer_text:
            st.write(answer_text)
        else:
            st.caption("No answer text provided.")

        # Missing Information
        missing = result.get("missing_information")
        if missing:
            st.subheader("Missing Information")
            st.info(missing)

        # Citations
        st.subheader("Citations")
        citations = result.get("citations", [])
        if citations:
            for cit in citations:
                page_part = f", Page {cit['page']}" if cit.get("page") is not None else ""
                st.markdown(f"- **`{cit.get('chunk_id', '')}`** → Source: `{cit.get('source', '')}`{page_part}")
        else:
            st.caption("No citations cited in this answer.")

        invalid_citations = result.get("invalid_citations", [])
        if invalid_citations:
            st.warning(f"Unresolved citations detected: {', '.join(invalid_citations)}")

        disclaimer = result.get("citation_disclaimer")
        if disclaimer:
            st.caption(f"*{disclaimer}*")

        # Latency Timings
        st.subheader("Latency Timings")
        timings = result.get("timings", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval Latency", f"{timings.get('retrieval_ms', 0.0):.1f} ms")
        col2.metric("Generation Latency", f"{timings.get('generation_ms', 0.0):.1f} ms")
        col3.metric("Total Latency", f"{timings.get('total_ms', 0.0):.1f} ms")

        # Expandable Retrieved Evidence
        st.subheader("Retrieved Evidence Chunks")
        evidence_list = result.get("retrieved_evidence", [])
        if not evidence_list:
            st.caption("No candidate evidence retrieved.")
        else:
            for i, ev in enumerate(evidence_list, start=1):
                chunk_id = ev.get("chunk_id", f"chunk_{i}")
                source = ev.get("source", "Unknown")
                distance = ev.get("distance")
                dist_str = f"{distance:.4f}" if distance is not None else "N/A"
                page = ev.get("page")
                page_part = f" | Page {page}" if page is not None else ""

                expander_label = f"Evidence #{i}: [{chunk_id}] (Cosine Distance: {dist_str})"
                with st.expander(expander_label, expanded=False):
                    st.markdown(f"**Chunk ID:** `{chunk_id}`")
                    st.markdown(f"**Source Document:** `{source}`{page_part}")
                    st.markdown(
                        f"**Raw Cosine Distance:** `{dist_str}` "
                        "*(smaller indicates closer similarity; not a probability or confidence score)*"
                    )
                    if ev.get("chunk_index") is not None:
                        st.markdown(f"**Chunk Index:** `{ev['chunk_index']}`")
                    st.markdown("**Passage Excerpt:**")
                    st.text(ev.get("text", ""))


if __name__ == "__main__":
    main()
