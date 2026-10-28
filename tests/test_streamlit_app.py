"""Unit and interface tests for the Streamlit web application.

Verifies:
1. Pre-flight index health and compatibility checks (read-only, actionable).
2. Demo bootstrap mechanism:
   - Disabled by default.
   - Initializing missing or empty index once per process with double-checked locking.
   - Reusing existing index without rebuilding (even if incompatible).
   - Initialization failure handling and blocking subsequent queries.
3. Reviewer token validation & constant-time security:
   - Constant-time comparison rejecting mismatched, empty, or missing tokens.
   - Accepting valid token.
   - Unlocking unlimited session access via sidebar and main form.
4. Streamlit secrets propagation to os.environ before config import.
5. Initial UI rendering and design refresh:
   - Title: "Personal AI Knowledge Assistant"
   - Subtitle explaining answers come from document collection
   - Question text area (<= 500 characters)
   - Top-k slider (1-10, default 5)
   - Example-question buttons
   - Free-question counter
   - Ask button
6. Feature 1 & 2 Verification Tests:
   - Five allowed submissions and a blocked sixth request.
   - Reruns and invalid inputs (blank or >500 chars) do not consume quota.
   - Independent session counters across distinct browser sessions.
   - Valid token bypasses quota; invalid token is rejected without quota reset.
   - Previous answer and evidence remain visible after exhaustion.
   - Contact-link configuration (CONTACT_URL, CONTACT_EMAIL) and setup notice when unconfigured.
   - Example-question buttons populate input without submitting.
   - Secondary diagnostics expander for timings and metadata.
   - Status badge rendering for answered, partial, conflict, abstain, and error.
   - Index-unhealthy blocking behavior with actionable remediation messages.

All tests run completely offline with mocked pipeline responses and external services.
No live OpenAI API calls or index rebuilds are executed.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    check_index_health,
    maybe_bootstrap_index,
    is_bootstrap_enabled,
    check_production_access,
    validate_reviewer_token,
    get_contact_info,
    MAX_FREE_QUESTIONS,
    reset_bootstrap_state_for_tests,
    _sync_streamlit_secrets_to_env,
)


ASK_LABEL = "Ask assistant"
EXHAUSTED_HEADING = "Your free preview is complete"
EXHAUSTED_MESSAGE = "You’ve used your five questions. Request an access token to continue."


def _texts(at):
    """All markdown and caption text rendered in the main area and sidebar."""
    return [m.value for m in at.markdown] + [c.value for c in at.caption]


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure clean process-level state between test executions."""
    reset_bootstrap_state_for_tests()
    yield
    reset_bootstrap_state_for_tests()


# -----------------------------------------------------------------------------
# 1. Index Health & Pre-flight Diagnostics Tests
# -----------------------------------------------------------------------------

def test_check_index_health_healthy():
    """Verify check_index_health returns True when collection exists, is populated and compatible."""
    mock_coll = MagicMock()
    mock_coll.count.return_value = 10

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll

    with patch("src.vector_store._get_client", return_value=mock_client), \
         patch("src.vector_store.check_index_compatibility", return_value=(True, "All good")):
        healthy, msg = check_index_health(Path("/mock/dir"))
        assert healthy is True
        assert "10 chunks indexed" in msg


def test_check_index_health_missing_collection():
    """Verify check_index_health returns False with actionable instructions when index is missing."""
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")

    with patch("src.vector_store._get_client", return_value=mock_client):
        healthy, msg = check_index_health(Path("/mock/dir"))
        assert healthy is False
        assert "not found" in msg.lower()
        assert "python ingest.py" in msg


def test_check_index_health_empty_collection():
    """Verify check_index_health returns False when collection exists but contains 0 chunks."""
    mock_coll = MagicMock()
    mock_coll.count.return_value = 0

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll

    with patch("src.vector_store._get_client", return_value=mock_client):
        healthy, msg = check_index_health(Path("/mock/dir"))
        assert healthy is False
        assert "empty" in msg.lower()
        assert "python ingest.py" in msg


def test_check_index_health_incompatible():
    """Verify check_index_health returns False when index compatibility check fails."""
    mock_coll = MagicMock()
    mock_coll.count.return_value = 10

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll

    with patch("src.vector_store._get_client", return_value=mock_client), \
         patch("src.vector_store.check_index_compatibility", return_value=(False, "Model mismatch")):
        healthy, msg = check_index_health(Path("/mock/dir"))
        assert healthy is False
        assert "incompatible" in msg.lower()
        assert "Model mismatch" in msg
        assert "python ingest.py" in msg


# -----------------------------------------------------------------------------
# 2. Demo Bootstrap Mechanism Tests
# -----------------------------------------------------------------------------

def test_bootstrap_disabled_by_default(monkeypatch):
    """Verify demo bootstrap is disabled by default and does not invoke ingestion."""
    monkeypatch.delenv("STREAMLIT_DEMO_BOOTSTRAP", raising=False)
    assert not is_bootstrap_enabled()

    with patch("ingest.ingest_corpus") as mock_ingest:
        ok, err = maybe_bootstrap_index()
        assert ok is True
        assert err is None
        mock_ingest.assert_not_called()


def test_bootstrap_missing_index_initialization(monkeypatch):
    """Verify missing index triggers ingestion when STREAMLIT_DEMO_BOOTSTRAP=true."""
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "true")

    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")

    with patch("src.vector_store._get_client", return_value=mock_client), \
         patch("ingest.ingest_corpus", return_value={"chunk_count": 10}) as mock_ingest:
        ok, err = maybe_bootstrap_index(
            storage_dir=Path("/mock/storage"),
            data_dir=Path("/mock/data"),
        )
        assert ok is True
        assert err is None
        mock_ingest.assert_called_once_with(
            force_rebuild=False,
            allow_partial_ingestion=False,
            data_dir=Path("/mock/data"),
            storage_dir=Path("/mock/storage"),
        )


def test_bootstrap_empty_index_initialization(monkeypatch):
    """Verify empty index (0 chunks) triggers ingestion when STREAMLIT_DEMO_BOOTSTRAP=true."""
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "1")

    mock_coll = MagicMock()
    mock_coll.count.return_value = 0
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll

    with patch("src.vector_store._get_client", return_value=mock_client), \
         patch("ingest.ingest_corpus", return_value={"chunk_count": 10}) as mock_ingest:
        ok, err = maybe_bootstrap_index(
            storage_dir=Path("/mock/storage"),
            data_dir=Path("/mock/data"),
        )
        assert ok is True
        assert err is None
        mock_ingest.assert_called_once()


def test_bootstrap_reuse_existing_index(monkeypatch):
    """Verify existing populated index is reused and NEVER rebuilt or replaced."""
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "true")

    mock_coll = MagicMock()
    mock_coll.count.return_value = 10
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll

    with patch("src.vector_store._get_client", return_value=mock_client), \
         patch("ingest.ingest_corpus") as mock_ingest:
        ok, err = maybe_bootstrap_index()
        assert ok is True
        assert err is None
        mock_ingest.assert_not_called()


def test_bootstrap_incompatible_index_not_rebuilt(monkeypatch):
    """Verify incompatible existing index is NOT rebuilt automatically by bootstrap."""
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "true")

    mock_coll = MagicMock()
    mock_coll.count.return_value = 10
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll

    with patch("src.vector_store._get_client", return_value=mock_client), \
         patch("src.vector_store.check_index_compatibility", return_value=(False, "Model mismatch")), \
         patch("ingest.ingest_corpus") as mock_ingest:
        ok, err = maybe_bootstrap_index()
        assert ok is True
        assert err is None
        mock_ingest.assert_not_called()


def test_bootstrap_initialization_failure(monkeypatch, caplog):
    """Verify bootstrap failure records error, logs traceback with packages, and blocks subsequent operations."""
    import logging
    caplog.set_level(logging.ERROR)
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "true")

    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Not found")

    with patch("src.vector_store._get_client", return_value=mock_client), \
         patch("ingest.ingest_corpus", side_effect=RuntimeError("Disk write error")):
        ok, err = maybe_bootstrap_index()
        assert ok is False
        assert "STREAMLIT_DEMO_BOOTSTRAP failed" in err
        assert "Disk write error" in err

        # Verify server-side traceback and package diagnostic logging occurred
        assert "STREAMLIT_DEMO_BOOTSTRAP failed: Disk write error" in caplog.text
        assert "Traceback" in caplog.text
        assert "packages:" in caplog.text

        # Subsequent call returns cached error without re-running ingest
        ok2, err2 = maybe_bootstrap_index()
        assert ok2 is False
        assert err2 == err


# -----------------------------------------------------------------------------
# 3. Reviewer Token Security & Constant-Time Validation
# -----------------------------------------------------------------------------

def test_validate_reviewer_token_constant_time(monkeypatch):
    """Verify constant-time validation safely compares tokens and rejects invalid inputs."""
    monkeypatch.setenv("REVIEWER_ACCESS_TOKEN", "valid-secret-token")

    # Correct token
    assert validate_reviewer_token("valid-secret-token") is True
    # Strip whitespace handling
    assert validate_reviewer_token("  valid-secret-token  ") is True
    # Incorrect token
    assert validate_reviewer_token("wrong-token") is False
    # Empty token
    assert validate_reviewer_token("") is False
    assert validate_reviewer_token("   ") is False

    # Server token unset: fails closed
    monkeypatch.delenv("REVIEWER_ACCESS_TOKEN", raising=False)
    assert validate_reviewer_token("valid-secret-token") is False


def test_check_production_access_allowance_and_bypass():
    """Verify check_production_access respects the 5-question allowance and token bypass."""
    with patch("streamlit.session_state", {"questions_used": 0, "authenticated": False}):
        assert check_production_access() is True

    with patch("streamlit.session_state", {"questions_used": 4, "authenticated": False}):
        assert check_production_access() is True

    with patch("streamlit.session_state", {"questions_used": 5, "authenticated": False}):
        assert check_production_access() is False

    with patch("streamlit.session_state", {"questions_used": 5, "authenticated": True}):
        assert check_production_access() is True


def test_sidebar_access_form_rejects_invalid_token(monkeypatch):
    """Verify entering an invalid token in sidebar displays error and does not authenticate."""
    monkeypatch.setenv("REVIEWER_ACCESS_TOKEN", "valid-reviewer-token")

    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

    # Enter wrong token in sidebar form
    at.sidebar.text_input[0].input("wrong-token")
    sidebar_unlock = [b for b in at.sidebar.button if "unlock" in b.label.lower()][0]
    sidebar_unlock.click().run()

    assert at.session_state["authenticated"] is False
    assert any("Invalid access token" in err.value for err in at.sidebar.error)


def test_sidebar_access_form_grants_access_on_valid_token(monkeypatch):
    """Verify entering valid token in sidebar unlocks assistant for the session."""
    monkeypatch.setenv("REVIEWER_ACCESS_TOKEN", "valid-reviewer-token")

    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

    at.sidebar.text_input[0].input("valid-reviewer-token")
    sidebar_unlock = [b for b in at.sidebar.button if "unlock" in b.label.lower()][0]
    sidebar_unlock.click().run()

    assert at.session_state["authenticated"] is True
    assert any("Access unlocked" in s.value for s in at.success)


# -----------------------------------------------------------------------------
# 4. Streamlit Secrets Propagation Test
# -----------------------------------------------------------------------------

def test_sync_streamlit_secrets_to_env(monkeypatch):
    """Verify secrets in st.secrets are populated into os.environ without printing."""
    fake_secrets = {
        "CUSTOM_DEPLOY_SECRET": "deploy_val_123",
        "APP_ENV": "production",
    }
    monkeypatch.delenv("CUSTOM_DEPLOY_SECRET", raising=False)

    with patch("streamlit.secrets", fake_secrets, create=True):
        _sync_streamlit_secrets_to_env()
        assert os.environ.get("CUSTOM_DEPLOY_SECRET") == "deploy_val_123"
        assert os.environ.get("APP_ENV") == "production"


# -----------------------------------------------------------------------------
# 5. Streamlit UI Rendering and Form Behavior Tests
# -----------------------------------------------------------------------------

def test_streamlit_initial_render(monkeypatch):
    """Verify initial UI components: title, subtitle, counter, example buttons, slider, ask button."""
    monkeypatch.setenv("APP_ENV", "development")
    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

    # Title & Subtitle
    assert len(at.title) >= 1
    assert at.title[0].value == "Ask your knowledge base"
    assert any("inspect the sources" in c.value for c in at.caption)

    # Free Question Counter (starts at 5 of 5)
    assert any("5 of 5 free questions remaining" in t for t in _texts(at))

    # Example question buttons present
    example_btn_labels = [b.label for b in at.button if "Project" in b.label]
    assert len(example_btn_labels) >= 3
    assert "What database does Project Nexus use?" in example_btn_labels
    assert "Where was Project Aurora deployed?" in example_btn_labels
    assert "What was the latency reduction achieved in Project Beacon?" in example_btn_labels

    # Question Input with 500-character max limit
    assert len(at.text_area) >= 1
    assert at.text_area[0].max_chars == 500

    # Slider: default 5, min 1, max 10
    assert len(at.slider) >= 1
    slider = at.slider[0]
    assert slider.value == 5
    assert slider.min == 1
    assert slider.max == 10

    # Ask button
    ask_btn = [b for b in at.button if b.label == ASK_LABEL]
    assert len(ask_btn) == 1

    # Initial session state
    assert at.session_state["last_result"] is None
    assert at.session_state["questions_used"] == 0
    assert at.session_state["authenticated"] is False


def test_streamlit_empty_submission(monkeypatch):
    """Verify submitting empty question displays warning and does not consume quota."""
    monkeypatch.setenv("APP_ENV", "development")
    with patch("src.pipeline.query_pipeline") as mock_pipeline:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        at.text_area[0].input("   ")
        ask_btn = [b for b in at.button if b.label == ASK_LABEL][0]
        ask_btn.click().run()

        mock_pipeline.assert_not_called()
        assert at.session_state["questions_used"] == 0
        assert len(at.warning) >= 1
        assert "non-empty question" in at.warning[0].value.lower()


def test_streamlit_successful_query_flow(monkeypatch):
    """Verify full query lifecycle: user query -> mocked query_pipeline -> answer & evidence rendered."""
    monkeypatch.setenv("APP_ENV", "development")
    mock_result = {
        "question": "What database does Project Nexus use?",
        "answer": "Project Nexus uses ChromaDB for persistent vector storage.",
        "status": "answered",
        "missing_information": None,
        "citations": [
            {"chunk_id": "nexus_01", "source": "project_nexus.md", "page": 1}
        ],
        "invalid_citations": [],
        "citation_disclaimer": "Citations are resolved strictly from storage metadata.",
        "retrieved_evidence": [
            {
                "chunk_id": "nexus_01",
                "text": "Project Nexus utilizes ChromaDB as its persistent vector database.",
                "source": "project_nexus.md",
                "page": 1,
                "chunk_index": 0,
                "distance": 0.1234,
            }
        ],
        "timings": {
            "retrieval_ms": 12.5,
            "generation_ms": 340.2,
            "total_ms": 352.7,
        },
        "model": "gpt-4o-mini",
        "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Input query and submit
        at.text_area[0].input("What database does Project Nexus use?")
        at.slider[0].set_value(4)
        ask_btn = [b for b in at.button if b.label == ASK_LABEL][0]
        ask_btn.click().run()

        mock_pipe.assert_called_once_with(
            question="What database does Project Nexus use?",
            top_k=4,
        )

        # Quota incremented to 1
        assert at.session_state["questions_used"] == 1
        assert at.session_state["last_result"] == mock_result

        # Restrained status badge for "answered" status
        assert any("Answered" in t for t in _texts(at))

        # Citations rendered
        markdown_texts = [m.value for m in at.markdown]
        assert any("nexus_01" in m for m in markdown_texts)
        assert any("project_nexus.md" in m for m in markdown_texts)

        # Raw distance and chunk ID displayed in evidence
        assert any("Raw Cosine Distance" in m for m in markdown_texts)

        # Metrics rendered for latency timings inside secondary expander
        metric_labels = [m.label for m in at.metric]
        assert "Retrieval Latency" in metric_labels
        assert "Generation Latency" in metric_labels
        assert "Total Latency" in metric_labels
        retrieval_m = [m for m in at.metric if m.label == "Retrieval Latency"][0]
        assert "12.5 ms" in retrieval_m.value


def test_streamlit_preserves_session_state_on_subsequent_rerun(monkeypatch):
    """Verify that ordinary UI reruns preserve the last result without re-invoking query_pipeline."""
    monkeypatch.setenv("APP_ENV", "development")
    mock_result = {
        "question": "What is Project Aurora?",
        "answer": "Project Aurora was deployed to AWS EKS.",
        "status": "answered",
        "missing_information": None,
        "citations": [{"chunk_id": "aurora_01", "source": "project_aurora.md", "page": None}],
        "invalid_citations": [],
        "citation_disclaimer": "Verified disclaimer",
        "retrieved_evidence": [
            {
                "chunk_id": "aurora_01",
                "text": "Aurora deployment details.",
                "source": "project_aurora.md",
                "page": None,
                "chunk_index": 0,
                "distance": 0.2222,
            }
        ],
        "timings": {"retrieval_ms": 10.0, "generation_ms": 200.0, "total_ms": 210.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # 1. Ask question
        at.text_area[0].input("What is Project Aurora?")
        ask_btn = [b for b in at.button if b.label == ASK_LABEL][0]
        ask_btn.click().run()

        assert mock_pipe.call_count == 1
        assert at.session_state["questions_used"] == 1
        assert at.session_state["last_result"] == mock_result

        # 2. Trigger a normal rerun without clicking Ask
        at.run()

        # query_pipeline must NOT be called again, questions_used remains 1
        assert mock_pipe.call_count == 1
        assert at.session_state["questions_used"] == 1

        # Content must still be rendered from session state
        markdown_texts = [m.value for m in at.markdown]
        assert any("aurora_01" in m for m in markdown_texts)


@pytest.mark.parametrize("status,expected_widget,expected_text", [
    ("partial", "markdown", "Partial answer"),
    ("conflict", "markdown", "Conflicting sources"),
    ("abstain", "markdown", "Insufficient evidence"),
    ("error", "error", "Simulated pipeline failure"),
])
def test_streamlit_status_rendering(monkeypatch, status, expected_widget, expected_text):
    """Verify distinct display for partial, conflict, abstain, and error statuses."""
    monkeypatch.setenv("APP_ENV", "development")
    mock_result = {
        "question": "Test query",
        "answer": "Answer content or fallback phrase.",
        "status": status,
        "missing_information": "Information about missing component." if status == "partial" else None,
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "Test disclaimer",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 5.0, "generation_ms": 50.0, "total_ms": 55.0},
        "model": "gpt-4o-mini",
        "error": "Simulated pipeline failure" if status == "error" else None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result):
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        at.text_area[0].input("Test query")
        ask_btn = [b for b in at.button if b.label == ASK_LABEL][0]
        ask_btn.click().run()

        widget_list = getattr(at, expected_widget)
        assert len(widget_list) >= 1
        assert any(expected_text in w.value for w in widget_list)

        if status == "partial":
            assert any("Information about missing component." in i.value for i in at.info)


def test_streamlit_blocks_submission_when_index_unhealthy(monkeypatch):
    """Verify that an unhealthy index prevents query execution and renders actionable guidance."""
    monkeypatch.setenv("APP_ENV", "development")
    actionable_msg = "Chroma collection missing"

    with patch("src.vector_store._get_client", side_effect=Exception(actionable_msg)), \
         patch("src.pipeline.query_pipeline") as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Error banner should be rendered immediately mentioning ingest.py
        all_errors = [e.value for e in at.error] + [e.value for e in at.sidebar.error]
        assert any("python ingest.py" in err for err in all_errors)

        # Attempt to submit question
        at.text_area[0].input("Valid question text")
        ask_btn = [b for b in at.button if b.label == ASK_LABEL][0]
        ask_btn.click().run()

        # Pipeline must not have been executed and quota must not be consumed
        mock_pipe.assert_not_called()
        assert at.session_state["questions_used"] == 0
        assert at.session_state["last_result"] is None


# -----------------------------------------------------------------------------
# 6. Focused Offline Verification Tests for Features 1 & 2
# -----------------------------------------------------------------------------

def test_five_allowed_submissions_and_blocked_sixth(monkeypatch):
    """Verify 5 questions are allowed without a token, and the 6th is blocked server-side."""
    monkeypatch.setenv("APP_ENV", "production")
    mock_result = {
        "question": "q",
        "answer": "answer content",
        "status": "answered",
        "missing_information": None,
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        assert at.session_state["questions_used"] == 0
        assert any("5 of 5 free questions remaining" in t for t in _texts(at))

        # Perform 5 submissions
        for i in range(1, 6):
            at.text_area[0].input(f"Question number {i}")
            ask_btn = [b for b in at.button if b.label == ASK_LABEL][0]
            ask_btn.click().run()

            assert mock_pipe.call_count == i
            assert at.session_state["questions_used"] == i
            rem = MAX_FREE_QUESTIONS - i
            if rem > 0:
                assert any(f"{rem} of 5 free questions remaining" in t for t in _texts(at))

        # Quota is now exhausted (5 used)
        assert at.session_state["questions_used"] == 5
        assert any(EXHAUSTED_HEADING in h.value for h in at.subheader)
        assert any(EXHAUSTED_MESSAGE in m.value for m in at.markdown)

        # Attempt 6th submission
        at.text_area[0].input("Attempting 6th question")
        ask_btn = [b for b in at.button if b.label == ASK_LABEL][0]
        ask_btn.click().run()

        # Pipeline must NOT be invoked on 6th request (call count remains 5)
        assert mock_pipe.call_count == 5
        assert at.session_state["questions_used"] == 5
        assert any(EXHAUSTED_HEADING in h.value for h in at.subheader)
        assert any(EXHAUSTED_MESSAGE in m.value for m in at.markdown)


def test_reruns_and_invalid_inputs_do_not_consume_quota(monkeypatch):
    """Verify ordinary Streamlit reruns and blank/whitespace inputs do not consume quota."""
    monkeypatch.setenv("APP_ENV", "production")

    with patch("src.pipeline.query_pipeline") as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        assert at.session_state["questions_used"] == 0
        assert at.text_area[0].max_chars == 500

        # 1. Blank input with whitespace
        at.text_area[0].input("   ")
        [b for b in at.button if b.label == ASK_LABEL][0].click().run()
        assert mock_pipe.call_count == 0
        assert at.session_state["questions_used"] == 0
        assert any("non-empty question" in w.value.lower() for w in at.warning)

        # 2. Completely empty input
        at.text_area[0].input("")
        [b for b in at.button if b.label == ASK_LABEL][0].click().run()
        assert mock_pipe.call_count == 0
        assert at.session_state["questions_used"] == 0
        assert any("non-empty question" in w.value.lower() for w in at.warning)

        # 3. Ordinary rerun
        at.run()
        assert mock_pipe.call_count == 0
        assert at.session_state["questions_used"] == 0
        assert any("5 of 5 free questions remaining" in t for t in _texts(at))


def test_sessions_have_independent_counters(monkeypatch):
    """Verify that distinct browser sessions have independent question counters."""
    monkeypatch.setenv("APP_ENV", "production")
    mock_result = {
        "question": "q",
        "answer": "ans",
        "status": "answered",
        "missing_information": None,
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 1.0, "generation_ms": 1.0, "total_ms": 2.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result):
        # Session A: submits 2 questions
        session_a = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        session_a.text_area[0].input("Session A - Question 1")
        [b for b in session_a.button if b.label == ASK_LABEL][0].click().run()
        session_a.text_area[0].input("Session A - Question 2")
        [b for b in session_a.button if b.label == ASK_LABEL][0].click().run()

        assert session_a.session_state["questions_used"] == 2
        assert any("3 of 5 free questions remaining" in t for t in _texts(session_a))

        # Session B: independent session starts at 0 used
        session_b = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        assert session_b.session_state["questions_used"] == 0
        assert any("5 of 5 free questions remaining" in t for t in _texts(session_b))


def test_valid_token_bypass_and_invalid_token_rejection(monkeypatch):
    """Verify invalid token rejection and valid token bypass of the 5-question limit."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("REVIEWER_ACCESS_TOKEN", "valid-secret-reviewer-token")

    mock_result = {
        "question": "q",
        "answer": "ans",
        "status": "answered",
        "missing_information": None,
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 1.0, "generation_ms": 1.0, "total_ms": 2.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Exhaust 5 free questions
        for i in range(5):
            at.text_area[0].input(f"Exhaustion query {i}")
            [b for b in at.button if b.label == ASK_LABEL][0].click().run()
        assert at.session_state["questions_used"] == 5
        assert mock_pipe.call_count == 5

        # 1. Attempt unlock with INVALID token
        token_inputs = [ti for ti in at.text_input if "token" in ti.label.lower()]
        assert len(token_inputs) >= 1
        token_inputs[0].input("incorrect-token")
        unlock_btns = [b for b in at.button if "unlock" in b.label.lower()]
        assert len(unlock_btns) >= 1
        unlock_btns[0].click().run()

        # Quota remains exhausted, unauthenticated, pipeline not called
        assert at.session_state["authenticated"] is False
        assert at.session_state["questions_used"] == 5
        assert mock_pipe.call_count == 5
        assert any("Invalid access token" in err.value for err in at.error)

        # 2. Attempt unlock with VALID token
        token_inputs = [ti for ti in at.text_input if "token" in ti.label.lower()]
        token_inputs[0].input("valid-secret-reviewer-token")
        unlock_btns = [b for b in at.button if "unlock" in b.label.lower()]
        unlock_btns[0].click().run()

        # Session authenticated and quota bypassed
        assert at.session_state["authenticated"] is True
        assert any("Access unlocked" in s.value for s in at.success)

        # 3. 6th query now executes successfully!
        at.text_area[0].input("Bypassed 6th question")
        [b for b in at.button if b.label == ASK_LABEL][0].click().run()
        assert mock_pipe.call_count == 6


def test_previous_results_remain_visible_after_exhaustion(monkeypatch):
    """Verify that previously rendered answer and evidence remain visible when quota is exhausted."""
    monkeypatch.setenv("APP_ENV", "production")
    mock_fifth_result = {
        "question": "What is Project Beacon?",
        "answer": "Project Beacon achieved 42% latency reduction.",
        "status": "answered",
        "missing_information": None,
        "citations": [{"chunk_id": "beacon_01", "source": "project_beacon.md", "page": 1}],
        "invalid_citations": [],
        "citation_disclaimer": "Verified citation",
        "retrieved_evidence": [
            {
                "chunk_id": "beacon_01",
                "text": "Project Beacon achieved 42% latency reduction.",
                "source": "project_beacon.md",
                "page": 1,
                "chunk_index": 0,
                "distance": 0.05,
            }
        ],
        "timings": {"retrieval_ms": 8.0, "generation_ms": 120.0, "total_ms": 128.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_fifth_result) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Submit 5 questions
        for i in range(4):
            at.text_area[0].input(f"Query {i}")
            [b for b in at.button if b.label == ASK_LABEL][0].click().run()

        at.text_area[0].input("What is Project Beacon?")
        [b for b in at.button if b.label == ASK_LABEL][0].click().run()
        assert mock_pipe.call_count == 5
        assert at.session_state["last_result"] == mock_fifth_result

        # Verify answer rendered
        assert any("beacon_01" in m.value for m in at.markdown)

        # Attempt 6th blocked submission
        at.text_area[0].input("Attempting 6th question")
        [b for b in at.button if b.label == ASK_LABEL][0].click().run()

        # Pipeline not called on 6th
        assert mock_pipe.call_count == 5
        # Previous result is preserved in session state
        assert at.session_state["last_result"] == mock_fifth_result
        # Previous answer and citation still rendered on screen
        assert any("beacon_01" in m.value for m in at.markdown)


def test_contact_link_configuration_and_error_rendering(monkeypatch):
    """Verify contact link destination selection and notice when neither is configured."""
    # 1. When CONTACT_EMAIL is configured
    monkeypatch.setenv("CONTACT_EMAIL", "reviewer@example.com")
    monkeypatch.delenv("CONTACT_URL", raising=False)
    info_email = get_contact_info()
    assert info_email["configured"] is True
    assert "mailto:reviewer@example.com" in info_email["url"]

    # 2. When CONTACT_URL is configured (priority)
    monkeypatch.setenv("CONTACT_URL", "https://example.com/contact")
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    info_url = get_contact_info()
    assert info_url["configured"] is True
    assert info_url["url"] == "https://example.com/contact"

    # 3. When neither is configured: setup notice rendered without invented contact
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("CONTACT_URL", raising=False)
    info_none = get_contact_info()
    assert info_none["configured"] is False
    assert info_none["url"] is None

    mock_result = {
        "question": "q", "answer": "a", "status": "answered", "missing_information": None,
        "citations": [], "invalid_citations": [], "citation_disclaimer": "", "retrieved_evidence": [],
        "timings": {"retrieval_ms": 1.0, "generation_ms": 1.0, "total_ms": 2.0}, "model": "gpt-4o-mini", "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result):
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        for _ in range(5):
            at.text_area[0].input("query")
            [b for b in at.button if b.label == ASK_LABEL][0].click().run()

        # Setup notice must be displayed when neither is configured
        assert any("Contact information is not configured on this server" in info.value for info in at.info)


def test_example_question_buttons_populate_input_without_submitting(monkeypatch):
    """Verify example question buttons populate question input without calling pipeline."""
    monkeypatch.setenv("APP_ENV", "development")

    with patch("src.pipeline.query_pipeline") as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Click "Project Nexus" example button
        nexus_btn = [b for b in at.button if b.label == "What database does Project Nexus use?"][0]
        nexus_btn.click().run()

        # Text area value should be populated with the example query
        assert at.text_area[0].value == "What database does Project Nexus use?"
        # Pipeline must NOT be called
        mock_pipe.assert_not_called()
        # Quota must NOT be consumed
        assert at.session_state["questions_used"] == 0
