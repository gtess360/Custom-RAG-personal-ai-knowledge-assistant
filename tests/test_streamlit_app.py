"""Unit and interface tests for the Streamlit web application.

Verifies:
1. Pre-flight index health and compatibility checks (read-only, actionable).
2. Demo bootstrap mechanism:
   - Disabled by default.
   - Initializing missing or empty index once per process with double-checked locking.
   - Reusing existing index without rebuilding (even if incompatible).
   - Initialization failure handling and blocking subsequent queries.
3. Production security gate:
   - Failing closed when APP_ENV=production and REVIEWER_ACCESS_TOKEN is missing.
   - Password access form rejecting invalid tokens.
   - Authenticating and unlocking queries on valid token.
   - Allowing open access in non-production.
4. Streamlit secrets propagation to os.environ before config import.
5. Initial UI rendering (Title, Question input <=500 chars, top_k slider 1-10 defaulting to 5, Ask button).
6. Explicit submission behavior (empty input rejected, character limit enforced).
7. Grounded answer and evidence display (answer, citations, source, chunk ID, raw cosine distance, timings).
8. Status badge rendering for answered, partial, conflict, abstain, and error.
9. Session state caching across UI reruns without duplicate pipeline calls.
10. Index-unhealthy blocking behavior with actionable remediation messages.

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
    reset_bootstrap_state_for_tests,
    _sync_streamlit_secrets_to_env,
)


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
        # Existing index must NOT be rebuilt
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


def test_bootstrap_initialization_failure(monkeypatch):
    """Verify bootstrap failure records error and blocks subsequent operations."""
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "true")

    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Not found")

    with patch("src.vector_store._get_client", return_value=mock_client), \
         patch("ingest.ingest_corpus", side_effect=RuntimeError("Disk write error")):
        ok, err = maybe_bootstrap_index()
        assert ok is False
        assert "STREAMLIT_DEMO_BOOTSTRAP failed" in err
        assert "Disk write error" in err

        # Subsequent call returns cached error without re-running ingest
        ok2, err2 = maybe_bootstrap_index()
        assert ok2 is False
        assert err2 == err


# -----------------------------------------------------------------------------
# 3. Production Access & Reviewer Token Security Tests
# -----------------------------------------------------------------------------

def test_production_access_fails_closed_when_token_missing(monkeypatch):
    """Verify production access fails closed when REVIEWER_ACCESS_TOKEN is missing."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("REVIEWER_ACCESS_TOKEN", raising=False)

    with patch("streamlit.error") as mock_st_error:
        authorized = check_production_access()
        assert authorized is False
        mock_st_error.assert_called_once()
        assert "Production Access Blocked" in mock_st_error.call_args[0][0]


def test_production_access_form_rejects_invalid_token(monkeypatch):
    """Verify production access form rejects invalid token via AppTest."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("REVIEWER_ACCESS_TOKEN", "valid-reviewer-token")

    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

    # Form should be shown
    assert len(at.text_input) >= 1
    at.text_input[0].input("wrong-token")
    unlock_btn = [b for b in at.button if b.label == "Unlock Assistant"][0]
    unlock_btn.click().run()

    assert at.session_state["authenticated"] is False
    assert any("Invalid access token" in err.value for err in at.error)


def test_production_access_form_grants_access_on_valid_token(monkeypatch):
    """Verify entering valid token grants access in production."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("REVIEWER_ACCESS_TOKEN", "valid-reviewer-token")

    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

    # Enter correct token
    at.text_input[0].input("valid-reviewer-token")
    unlock_btn = [b for b in at.button if b.label == "Unlock Assistant"][0]
    unlock_btn.click().run()

    assert at.session_state["authenticated"] is True
    # Query form should now be visible
    assert any(b.label == "Ask" for b in at.button)


def test_development_access_open_without_token(monkeypatch):
    """Verify development mode allows open access without reviewer token form."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("REVIEWER_ACCESS_TOKEN", raising=False)

    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

    # Query form is directly available without login gate
    assert any(b.label == "Ask" for b in at.button)


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
# 5. Streamlit UI Rendering and Submission Behavior Tests (AppTest)
# -----------------------------------------------------------------------------

def test_streamlit_initial_render(monkeypatch):
    """Verify initial UI components: title, form, question area, top_k slider, ask button."""
    monkeypatch.setenv("APP_ENV", "development")
    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

    # Check Title
    assert len(at.title) >= 1
    assert at.title[0].value == "Personal AI Knowledge Assistant"

    # Check Question Input with 500-character max limit
    assert len(at.text_area) >= 1
    assert at.text_area[0].max_chars == 500

    # Check Slider: default 5, min 1, max 10
    assert len(at.slider) >= 1
    slider = at.slider[0]
    assert slider.value == 5
    assert slider.min == 1
    assert slider.max == 10

    # Check Form Submit Button
    ask_btn = [b for b in at.button if b.label == "Ask"]
    assert len(ask_btn) == 1

    # Last result should be None initially
    assert at.session_state["last_result"] is None


def test_streamlit_empty_submission(monkeypatch):
    """Verify submitting empty question displays warning and does not call pipeline."""
    monkeypatch.setenv("APP_ENV", "development")
    with patch("src.pipeline.query_pipeline") as mock_pipeline:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        at.text_area[0].input("   ")
        ask_btn = [b for b in at.button if b.label == "Ask"][0]
        ask_btn.click().run()

        mock_pipeline.assert_not_called()
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
        ask_btn = [b for b in at.button if b.label == "Ask"][0]
        ask_btn.click().run()

        # Pipeline must be called with expected parameters
        mock_pipe.assert_called_once_with(
            question="What database does Project Nexus use?",
            top_k=4,
        )

        # Result preserved in session state
        assert at.session_state["last_result"] == mock_result

        # Success banner for "answered" status
        assert any("Answered" in s.value for s in at.success)

        # Metrics rendered for timings
        metric_labels = [m.label for m in at.metric]
        assert "Retrieval Latency" in metric_labels
        assert "Generation Latency" in metric_labels
        assert "Total Latency" in metric_labels

        retrieval_m = [m for m in at.metric if m.label == "Retrieval Latency"][0]
        assert "12.5 ms" in retrieval_m.value

        # Citations rendered
        markdown_texts = [m.value for m in at.markdown]
        assert any("nexus_01" in m for m in markdown_texts)
        assert any("project_nexus.md" in m for m in markdown_texts)
        assert any("0.1234" in m for m in markdown_texts)

        # Raw distance and chunk ID displayed for evidence
        assert any("Raw Cosine Distance" in m for m in markdown_texts)


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
        ask_btn = [b for b in at.button if b.label == "Ask"][0]
        ask_btn.click().run()

        assert mock_pipe.call_count == 1
        assert at.session_state["last_result"] == mock_result

        # 2. Trigger a normal rerun without clicking Ask
        at.run()

        # query_pipeline must NOT be called again
        assert mock_pipe.call_count == 1

        # Content must still be rendered from session state
        markdown_texts = [m.value for m in at.markdown]
        assert any("aurora_01" in m for m in markdown_texts)


@pytest.mark.parametrize("status,expected_widget,expected_text", [
    ("partial", "warning", "Partial"),
    ("conflict", "warning", "Conflict"),
    ("abstain", "info", "Abstain"),
    ("error", "error", "Error"),
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
        ask_btn = [b for b in at.button if b.label == "Ask"][0]
        ask_btn.click().run()

        widget_list = getattr(at, expected_widget)
        assert len(widget_list) >= 1
        assert any(expected_text in w.value for w in widget_list)

        # If partial, missing information section should also be present
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
        ask_btn = [b for b in at.button if b.label == "Ask"][0]
        ask_btn.click().run()

        # Pipeline must not have been executed
        mock_pipe.assert_not_called()
        assert at.session_state["last_result"] is None
