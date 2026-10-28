"""Streamlit web interface for the Personal AI Knowledge Assistant.

Directly calls src.pipeline.query_pipeline without requiring FastAPI or Next.js.
Supports Streamlit Community Cloud deployment with:
- Pre-import Streamlit secrets propagation to os.environ (preserving local .env)
- Explicit STREAMLIT_DEMO_BOOTSTRAP mode for automatic missing-index initialization
- Thread-locked, double-checked, once-per-process bootstrap guard
- Session-based demo allowance (5 free questions per browser session without requiring an access token)
  Note: This is an in-memory session allowance for demo visitors, not a permanent per-person quota or hard API spending cap.
- Token entry available at any time using REVIEWER_ACCESS_TOKEN (validated server-side with constant-time comparison)
- Valid token bypasses the demo quota for that browser session
- Configurable candidate retrieval (top_k between 1 and 10, default 5)
- Character-limited question input (<= 500 characters)
- Example question buttons to populate the question input without submitting
- Explicit form submission with session-state caching across UI reruns
- Clear rendering of status (answered, partial, conflict, abstain, error)
- Grounded answers, missing information, storage-verified citations, and collapsible evidence
- Secondary diagnostics area for latency timings and technical metadata
- Restrained professional knowledge-search interface (theme in .streamlit/config.toml, small static CSS)
- Safe text rendering and strict credential protection (never interpolating untrusted content into HTML)
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
# 2. Quota, Contact & Example Question Constants
# -----------------------------------------------------------------------------

MAX_FREE_QUESTIONS: int = 5

EXAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "nexus",
        "label": "Project Nexus",
        "query": "What database does Project Nexus use?",
    },
    {
        "id": "aurora",
        "label": "Project Aurora",
        "query": "Where was Project Aurora deployed?",
    },
    {
        "id": "beacon",
        "label": "Project Beacon",
        "query": "What was the latency reduction achieved in Project Beacon?",
    },
]

# Static custom CSS. Colours and radii come from .streamlit/config.toml; this block only
# adds what the theme cannot express. Selectors use stable hooks (data-testid attributes
# and the st-key-<key> classes Streamlit adds to keyed containers), never generated
# class names. Strictly static declarations; NEVER interpolates dynamic or untrusted text.
STATIC_CUSTOM_CSS = """
<style>
/* White surfaces for the main workspace cards. */
.st-key-ask_card,
.st-key-answer_card,
.st-key-empty_state,
.st-key-access_panel {
    background-color: #FFFFFF;
}

/* Secondary text colour for captions. */
[data-testid="stCaptionContainer"] {
    color: #64748B;
}

/* Sample questions: compact secondary controls. */
.st-key-sample_questions button p {
    font-size: 0.875rem;
}

/* Visible keyboard focus for interactive controls. */
button:focus-visible,
a:focus-visible,
summary:focus-visible,
[role="slider"]:focus-visible {
    outline: 2px solid #2563EB;
    outline-offset: 2px;
}
</style>
"""

# Restrained status labels: (badge label, badge colour, one-line explanation).
STATUS_DISPLAY: dict[str, tuple[str, str, str]] = {
    "answered": ("Answered", "green", "Answer drawn from your documents."),
    "partial": ("Partial answer", "orange", "Some requested details were not found in your documents."),
    "conflict": ("Conflicting sources", "orange", "Your documents disagree. Review the supporting evidence."),
    "abstain": ("Insufficient evidence", "gray", "Your documents do not contain enough information to answer this."),
}

_MD_SPECIAL_CHARS = set("\\`*_{}[]()#+-.!|<>~$:")


def _md_escape(text: object) -> str:
    """Escape Markdown control characters so user or document text renders literally."""
    return "".join(f"\\{ch}" if ch in _MD_SPECIAL_CHARS else ch for ch in str(text))


def _md_code(text: object) -> str:
    """Render a short identifier (file name, chunk ID) as an inline code span."""
    return "`" + str(text).replace("`", "'").replace("\n", " ") + "`"


# -----------------------------------------------------------------------------
# 3. Process-Level Demo Bootstrap Mechanism
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
# 4. Read-Only Index Health Check
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
# 5. Access Control, Quota & Contact Destination Helpers
# -----------------------------------------------------------------------------

def get_contact_info() -> dict[str, Optional[str] | bool]:
    """Retrieve contact link destination from CONTACT_URL or CONTACT_EMAIL.

    Priority:
    1. CONTACT_URL if configured and non-empty.
    2. CONTACT_EMAIL if configured and non-empty (formatted as a mailto: link).
    3. Neither configured (returns configured=False with None destination).

    Returns:
        dict:
            - 'url': Optional[str] (destination link, e.g. https://... or mailto:...)
            - 'display': Optional[str] (human-readable destination text)
            - 'configured': bool (True if a contact link is configured)
    """
    raw_url = os.getenv("CONTACT_URL", "").strip()
    raw_email = os.getenv("CONTACT_EMAIL", "").strip()

    if raw_url:
        return {"url": raw_url, "display": raw_url, "configured": True}
    if raw_email:
        mail_target = raw_email if raw_email.startswith("mailto:") else f"mailto:{raw_email}?subject=Personal%20AI%20Knowledge%20Assistant%20Access%20Token%20Request"
        return {"url": mail_target, "display": raw_email, "configured": True}
    return {"url": None, "display": None, "configured": False}


def validate_reviewer_token(token: str) -> bool:
    """Validate entered reviewer token using constant-time comparison.

    Fails safely if REVIEWER_ACCESS_TOKEN is unset or empty on the server,
    or if the entered token does not match.

    Returns:
        bool: True if token matches REVIEWER_ACCESS_TOKEN, False otherwise.
    """
    expected_token = os.getenv("REVIEWER_ACCESS_TOKEN", "").strip()
    if not expected_token or not token or not token.strip():
        return False
    return hmac.compare_digest(token.strip(), expected_token)


def check_production_access() -> bool:
    """Check if the user is authorized to submit queries.

    Visitors have a session-based demo allowance of up to MAX_FREE_QUESTIONS.
    Entering a valid REVIEWER_ACCESS_TOKEN bypasses this demo limit for the session.

    Returns:
        bool: True if authorized (authenticated or questions_used < MAX_FREE_QUESTIONS).
    """
    if st.session_state.get("authenticated", False):
        return True
    return st.session_state.get("questions_used", 0) < MAX_FREE_QUESTIONS


def _populate_example(query: str) -> None:
    """Callback for example buttons to populate question input without submitting."""
    st.session_state["question_input"] = query


# -----------------------------------------------------------------------------
# 6. Interface Rendering Helpers
# -----------------------------------------------------------------------------

def _try_unlock(token: str) -> None:
    """Validate a submitted token; unlock the session or show an inline error."""
    if validate_reviewer_token(token):
        st.session_state["authenticated"] = True
        st.rerun()
    else:
        st.error("Invalid access token. Access denied.")


def _render_request_access_button(contact_info: dict, *, primary: bool = False) -> None:
    """Render the 'Request access' contact control."""
    btn_type = "primary" if primary else "secondary"
    if contact_info["configured"]:
        st.link_button("Request access", str(contact_info["url"]), type=btn_type, width="stretch")
    else:
        st.button(
            "Request access",
            key=f"request_access_disabled_{btn_type}",
            disabled=True,
            width="stretch",
            help="Contact details are not configured on this server.",
        )


def _render_sidebar(
    *,
    is_authenticated: bool,
    questions_used: int,
    is_healthy: bool,
    health_msg: str,
    contact_info: dict,
) -> int:
    """Render the compact sidebar and return the selected top_k value."""
    remaining = max(0, MAX_FREE_QUESTIONS - questions_used)
    exhausted = not is_authenticated and remaining == 0

    with st.sidebar:
        st.subheader("Knowledge Assistant")
        st.caption("Search your documents with source-backed answers.")
        st.space("small")

        if is_authenticated:
            st.success("Access unlocked", icon=":material/lock_open:")
            st.caption("Unlimited questions for this browser session.")
            if st.button("Lock session", key="sidebar_lock_btn", type="tertiary"):
                st.session_state["authenticated"] = False
                st.rerun()
        else:
            st.markdown("**Free preview**")
            st.progress(min(1.0, questions_used / MAX_FREE_QUESTIONS))
            st.caption(
                f"{remaining} of {MAX_FREE_QUESTIONS} free questions remaining",
                help=(
                    "Visitors can ask 5 free questions per browser session without an access token. "
                    "This is an in-memory session allowance, not a permanent per-person limit "
                    "or hard spending cap."
                ),
            )
            _render_request_access_button(contact_info)
            with st.expander("Have an access token?"):
                with st.form("sidebar_reviewer_auth_form", border=False):
                    token_input = st.text_input(
                        "Access token",
                        key="sidebar_token_input",
                        type="password",
                        help="Provided by the project administrator.",
                    )
                    unlock = st.form_submit_button("Unlock", width="stretch")
                if unlock:
                    _try_unlock(token_input)

        st.space("small")
        with st.expander("System status and settings"):
            if is_healthy:
                st.badge("Index ready", color="green", icon=":material/check:")
                st.caption(health_msg)
            else:
                st.badge("Index not ready", color="red", icon=":material/error:")
                st.caption("Details are shown in the main workspace.")

            top_k = st.slider(
                "Passages to retrieve",
                min_value=1,
                max_value=10,
                value=DEFAULT_TOP_K,
                step=1,
                help="Number of candidate evidence passages retrieved from the vector index (top_k, default 5).",
            )

            safe_cfg = get_safe_config_summary()
            has_server_token = bool(os.getenv("REVIEWER_ACCESS_TOKEN", "").strip())
            st.caption(
                f"Environment: {_md_escape(safe_cfg.get('app_env', 'development'))}  \n"
                f"Model: {_md_escape(safe_cfg.get('openai_model', OPENAI_MODEL))}  \n"
                f"API key configured: {'Yes' if has_openai_api_key() else 'No'}  \n"
                f"Demo bootstrap: {'Enabled' if is_bootstrap_enabled() else 'Disabled'}  \n"
                f"Access token: {'Configured' if has_server_token else 'Missing'}"
            )

            if st.button("Clear current answer", key="clear_result_btn", type="tertiary"):
                st.session_state["last_result"] = None
                st.rerun()

    return top_k


def _render_access_panel(contact_info: dict) -> None:
    """Single, clean panel shown once the free preview is used up."""
    with st.container(border=True, key="access_panel"):
        st.subheader("Your free preview is complete")
        st.markdown("You’ve used your five questions. Request an access token to continue.")
        if contact_info["configured"]:
            st.link_button("Request access", str(contact_info["url"]), type="primary")
        else:
            st.info(
                "Contact information is not configured on this server. "
                "Set `CONTACT_EMAIL` or `CONTACT_URL` in your environment."
            )
        with st.form("in_page_token_form", border=False):
            in_page_token = st.text_input("Access token", key="in_page_token_input", type="password")
            in_page_unlock = st.form_submit_button("Unlock")
        if in_page_unlock:
            _try_unlock(in_page_token)


def _render_empty_state() -> None:
    """Meaningful placeholder before the first question."""
    with st.container(border=True, key="empty_state"):
        st.markdown("**Your answer will appear here**")
        st.caption(
            "Ask about anything in the indexed documents. Each answer lists the documents "
            "it draws on, and you can open the exact passages that support it."
        )


def _unique_sources(citations: list[dict]) -> list[tuple[str, Optional[object]]]:
    """Collapse citations to unique (source, page) pairs, preserving order."""
    seen: list[tuple[str, Optional[object]]] = []
    for cit in citations:
        key = (str(cit.get("source", "Unknown")), cit.get("page"))
        if key not in seen:
            seen.append(key)
    return seen


def _render_technical_details(result: dict, evidence_list: list[dict], top_k_used: object) -> None:
    """Raw IDs, distance metrics and timings, kept out of the main reading flow."""
    with st.expander("Technical details"):
        timings = result.get("timings") or {}
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval Latency", f"{timings.get('retrieval_ms', 0.0):.1f} ms")
        col2.metric("Generation Latency", f"{timings.get('generation_ms', 0.0):.1f} ms")
        col3.metric("Total Latency", f"{timings.get('total_ms', 0.0):.1f} ms")
        st.caption(
            f"Model: {_md_code(result.get('model', OPENAI_MODEL))} · "
            f"Candidate top_k: {_md_code(top_k_used)} · "
            f"Evidence chunks: {_md_code(len(evidence_list))}"
        )

        citations = result.get("citations") or []
        if citations:
            st.markdown("**Cited chunks**")
            for cit in citations:
                st.markdown(f"- {_md_code(cit.get('chunk_id', ''))} → {_md_code(cit.get('source', ''))}")

        invalid_citations = result.get("invalid_citations") or []
        if invalid_citations:
            st.markdown("**Unresolved citations**")
            st.markdown(", ".join(_md_code(c) for c in invalid_citations))

        if evidence_list:
            st.markdown("**Retrieved chunks**")
            for i, ev in enumerate(evidence_list, start=1):
                distance = ev.get("distance")
                dist_str = f"{distance:.4f}" if distance is not None else "N/A"
                chunk_index = ev.get("chunk_index")
                index_part = f" · chunk index {_md_code(chunk_index)}" if chunk_index is not None else ""
                st.markdown(
                    f"{i}. {_md_code(ev.get('chunk_id', f'chunk_{i}'))} · "
                    f"Raw Cosine Distance {_md_code(dist_str)}{index_part}"
                )
            st.caption("Smaller distance indicates closer similarity; it is not a probability or confidence score.")


def _render_result(result: dict, fallback_question: str, top_k_used: object) -> None:
    """Render the submitted question, then the answer (or error) with its sources."""
    status = result.get("status", "unknown")
    evidence_list = result.get("retrieved_evidence") or []
    question_text = result.get("question") or fallback_question

    st.space("small")
    if question_text:
        st.caption("Your question")
        st.markdown(f"**{_md_escape(question_text)}**")

    # Errors are displayed on their own, never styled as an answer.
    if status == "error":
        err = result.get("error") or "An error occurred during query processing."
        st.error(f"Unable to answer this question.\n\n{_md_escape(err)}", icon=":material/error:")
        _render_technical_details(result, evidence_list, top_k_used)
        return

    label, color, explanation = STATUS_DISPLAY.get(
        status, (str(status).title(), "gray", "")
    )

    with st.container(border=True, key="answer_card"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge(label, color=color)
            if explanation:
                st.caption(explanation, width="content")

        answer_text = result.get("answer", "")
        if answer_text:
            # Markdown (never raw HTML); escape '$' so currency is not parsed as math.
            st.markdown(str(answer_text).replace("$", "\\$"))
        else:
            st.caption("No answer text provided.")

        missing = result.get("missing_information")
        if missing:
            st.info(missing)

        citations = result.get("citations") or []
        st.markdown("**Sources**")
        if citations:
            for i, (source, page) in enumerate(_unique_sources(citations), start=1):
                page_part = f" · page {_md_escape(page)}" if page is not None else ""
                st.markdown(f"{i}. {_md_code(source)}{page_part}")
        else:
            st.caption("No sources were cited for this answer.")

        invalid_citations = result.get("invalid_citations") or []
        if invalid_citations:
            st.caption(
                f"{len(invalid_citations)} citation(s) could not be matched to stored documents. "
                "See technical details."
            )

        disclaimer = result.get("citation_disclaimer")
        if disclaimer:
            st.caption(_md_escape(disclaimer))

    if evidence_list:
        with st.expander(f"View supporting evidence ({len(evidence_list)})"):
            for i, ev in enumerate(evidence_list, start=1):
                if i > 1:
                    st.divider()
                page = ev.get("page")
                page_part = f" · page {_md_escape(page)}" if page is not None else ""
                st.markdown(f"**{i}.** {_md_code(ev.get('source', 'Unknown'))}{page_part}")
                st.text(ev.get("text", ""))
    else:
        st.caption("No supporting passages were retrieved.")

    _render_technical_details(result, evidence_list, top_k_used)


# -----------------------------------------------------------------------------
# 7. Application Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Knowledge Assistant",
        page_icon=":material/manage_search:",
        layout="centered",
    )

    # Small static stylesheet (no dynamic content).
    st.html(STATIC_CUSTOM_CSS)

    # Initialize session state for persistent results across reruns
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "questions_used" not in st.session_state:
        st.session_state["questions_used"] = 0
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "in_page_token_input" not in st.session_state:
        st.session_state["in_page_token_input"] = ""
    if "sidebar_token_input" not in st.session_state:
        st.session_state["sidebar_token_input"] = ""

    # 1. Execute optional bootstrap if STREAMLIT_DEMO_BOOTSTRAP is enabled
    bootstrap_ok, bootstrap_err = maybe_bootstrap_index()

    # 2. Check index health (read-only)
    if bootstrap_ok:
        is_healthy, health_msg = check_index_health()
    else:
        is_healthy = False
        health_msg = f"Bootstrap initialization failed: {bootstrap_err}"

    is_authenticated = st.session_state.get("authenticated", False)
    questions_used = st.session_state.get("questions_used", 0)
    exhausted = not is_authenticated and questions_used >= MAX_FREE_QUESTIONS
    contact_info = get_contact_info()

    # 3. Sidebar: product, allowance, access and (collapsed) system settings
    top_k = _render_sidebar(
        is_authenticated=is_authenticated,
        questions_used=questions_used,
        is_healthy=is_healthy,
        health_msg=health_msg,
        contact_info=contact_info,
    )

    # 4. Header
    st.title("Ask your knowledge base")
    st.caption("Find answers across your documents and inspect the sources.")

    # 5. Blocking notices (shown once, in the workspace)
    if not is_healthy:
        st.error(f"**The knowledge index is not ready.** {health_msg}", icon=":material/error:")
    if exhausted:
        _render_access_panel(contact_info)

    input_locked = exhausted or not is_healthy

    # 6. Question form and sample questions
    with st.container(border=True, key="ask_card"):
        with st.form("query_form", clear_on_submit=False, border=False):
            question = st.text_area(
                "Your question",
                key="question_input",
                max_chars=500,
                height=110,
                placeholder="For example: What database does Project Nexus use?",
                help="Answers use only evidence retrieved from your indexed documents.",
                disabled=input_locked,
            )
            submitted = st.form_submit_button("Ask assistant", type="primary", disabled=input_locked)

        st.caption("Sample questions")
        with st.container(horizontal=True, gap="small", key="sample_questions"):
            for ex in EXAMPLE_QUESTIONS:
                st.button(
                    ex["query"],
                    key=f"ex_btn_{ex['id']}",
                    on_click=_populate_example,
                    args=(ex["query"],),
                    help="Fills in the question without submitting",
                    disabled=input_locked,
                )

    # 7. Handle explicit form submission with server-side quota enforcement
    if submitted:
        clean_q = (question or "").strip()
        is_auth = st.session_state.get("authenticated", False)
        q_used = st.session_state.get("questions_used", 0)

        if not clean_q:
            st.warning("Please enter a non-empty question before asking.")
        elif len(clean_q) > 500:
            st.error("Question exceeds the 500-character limit. Please shorten your question.")
        elif not is_healthy:
            # The index notice above already explains the problem and remedy.
            pass
        elif not is_auth and q_used >= MAX_FREE_QUESTIONS:
            # The access panel above already explains how to continue.
            pass
        else:
            with st.spinner("Searching your documents..."):
                result = pipeline.query_pipeline(question=clean_q, top_k=top_k)
                st.session_state["last_result"] = result
                st.session_state["last_question"] = clean_q
                st.session_state["last_top_k"] = top_k
                if not is_auth:
                    st.session_state["questions_used"] = q_used + 1
                st.rerun()

    # 8. Last result (preserved after the preview is exhausted) or empty state
    result = st.session_state.get("last_result")
    if result is not None:
        _render_result(
            result,
            fallback_question=st.session_state.get("last_question", ""),
            top_k_used=st.session_state.get("last_top_k", top_k),
        )
    else:
        _render_empty_state()


if __name__ == "__main__":
    main()
