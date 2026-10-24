"""Tests for evidence-grounded generation and the RAG pipeline.

All tests in this module run offline with mocked OpenAI API interactions.
No paid API calls are made during automated testing.

CRITICAL METHODOLOGICAL LIMITATION:
Mocked unit tests verify prompt assembly, API client configuration, response parsing,
schema validation, citation validation, error sanitization, and structured result contracts.
They DO NOT prove real-world model grounding, semantic claim support, or robust
resistance to prompt injection. Real-model grounding and injection resistance are
empirically measured on live benchmark questions in Phase 6.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import httpx
from openai import (
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    APIStatusError,
)

from src.generation import (
    INSUFFICIENT_EVIDENCE_PHRASE,
    CITATION_VALIDITY_DISCLAIMER,
    validate_and_resolve_citations,
    parse_model_response,
    generate_answer,
    sanitize_error_message,
    build_generation_messages,
    build_system_prompt,
)
from src.pipeline import query_pipeline
from src.chunking import ChunkRecord, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import embed_texts
from src.vector_store import build_index


def _make_mock_openai_client(content: str) -> MagicMock:
    """Helper to construct a mock OpenAI client returning a specific string content."""
    mock_choice = MagicMock()
    mock_choice.message.content = content

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_passages() -> list[dict]:
    return [
        {
            "chunk_id": "chunk_nexus_00",
            "text": "Project Nexus uses ChromaDB and sentence-transformers/all-MiniLM-L6-v2.",
            "source": "data/documents/project_nexus.md",
            "page": None,
            "chunk_index": 0,
            "distance": 0.15,
        },
        {
            "chunk_id": "chunk_aurora_01",
            "text": "Project Aurora was deployed to AWS EKS in August 2024.",
            "source": "data/documents/project_aurora.md",
            "page": 1,
            "chunk_index": 1,
            "distance": 0.22,
        },
    ]


# -----------------------------------------------------------------------------
# 1. Error Sanitization & Public Error Safety
# -----------------------------------------------------------------------------

def test_sanitize_error_message_redacts_short_and_long_fictional_credentials():
    """Verify regex sanitization redacts short and long fictional keys and Bearer tokens.

    Limitation note: Regex sanitization is defense-in-depth and does not guarantee
    complete redaction of arbitrary unstructured secrets; fixed public messages are preferred.
    """
    short_key = "sk-1234567890abcdef1234"
    long_key = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_-"
    bearer = "Bearer fictional-bearer-token-1234567890"

    raw_error = f"Error with {short_key}, {long_key}, and {bearer} in payload."
    sanitized = sanitize_error_message(raw_error)

    assert short_key not in sanitized
    assert long_key not in sanitized
    assert bearer not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "Bearer [REDACTED_TOKEN]" in sanitized


# -----------------------------------------------------------------------------
# 2. Strict Response & Schema Validation
# -----------------------------------------------------------------------------

def test_parse_model_response_rejects_empty_string(sample_passages):
    """Empty model output must return a distinct validation error, not a successful answer."""
    parsed = parse_model_response("", sample_passages)
    assert parsed["status"] == "error"
    assert "Validation error" in parsed["error"]
    assert "empty response" in parsed["error"].lower()


def test_parse_model_response_rejects_plain_text(sample_passages):
    """Plain text responses must not bypass validation and become successful answers."""
    raw_text = "Project Nexus uses ChromaDB [chunk_nexus_00]."
    parsed = parse_model_response(raw_text, sample_passages)

    assert parsed["status"] == "error"
    assert "Validation error" in parsed["error"]
    assert "not valid JSON" in parsed["error"]


def test_parse_model_response_rejects_malformed_json(sample_passages):
    """Truncated or invalid JSON must return a validation error."""
    bad_json = '{"answer": "Incomplete answer", "status": "answered", "cited_chunk_ids":'
    parsed = parse_model_response(bad_json, sample_passages)

    assert parsed["status"] == "error"
    assert "Validation error" in parsed["error"]
    assert "not valid JSON" in parsed["error"]


def test_parse_model_response_validates_required_fields(sample_passages):
    """Missing required keys ('status', 'answer', or 'cited_chunk_ids') must fail validation."""
    missing_status = json.dumps({"answer": "Some answer", "cited_chunk_ids": ["chunk_nexus_00"]})
    parsed = parse_model_response(missing_status, sample_passages)
    assert parsed["status"] == "error"
    assert "Missing required field" in parsed["error"]
    assert "status" in parsed["error"]

    missing_citations = json.dumps({"answer": "Some answer", "status": "answered"})
    parsed2 = parse_model_response(missing_citations, sample_passages)
    assert parsed2["status"] == "error"
    assert "Missing required field" in parsed2["error"]
    assert "cited_chunk_ids" in parsed2["error"]


def test_parse_model_response_validates_field_types(sample_passages):
    """Invalid types for required fields must return distinct validation errors."""
    bad_citations_type = json.dumps({
        "answer": "Some answer",
        "status": "answered",
        "cited_chunk_ids": "chunk_nexus_00",  # String instead of list
    })
    parsed = parse_model_response(bad_citations_type, sample_passages)
    assert parsed["status"] == "error"
    assert "must be a list of strings" in parsed["error"]

    bad_item_type = json.dumps({
        "answer": "Some answer",
        "status": "answered",
        "cited_chunk_ids": [123],  # Int inside list
    })
    parsed2 = parse_model_response(bad_item_type, sample_passages)
    assert parsed2["status"] == "error"
    assert "must be strings" in parsed2["error"]


def test_parse_model_response_validates_allowed_status_values(sample_passages):
    """Status outside the allowed set must return a validation error."""
    invalid_status = json.dumps({
        "answer": "Some answer",
        "status": "guessed",  # Disallowed
        "cited_chunk_ids": ["chunk_nexus_00"],
    })
    parsed = parse_model_response(invalid_status, sample_passages)
    assert parsed["status"] == "error"
    assert "Invalid status 'guessed'" in parsed["error"]


# -----------------------------------------------------------------------------
# 3. Citation Handling & Factual Grounding Constraints
# -----------------------------------------------------------------------------

def test_validate_and_resolve_citations_derives_from_storage(sample_passages):
    """Source paths and page numbers must be resolved strictly from stored metadata."""
    cited_ids = ["chunk_nexus_00", "chunk_aurora_01", "chunk_hallucinated_99"]

    valid_cits, invalid_ids = validate_and_resolve_citations(cited_ids, sample_passages)

    assert len(valid_cits) == 2
    assert valid_cits[0] == {
        "chunk_id": "chunk_nexus_00",
        "source": "data/documents/project_nexus.md",
        "page": None,
    }
    assert valid_cits[1] == {
        "chunk_id": "chunk_aurora_01",
        "source": "data/documents/project_aurora.md",
        "page": 1,
    }

    assert invalid_ids == ["chunk_hallucinated_99"]


def test_generate_answer_rejects_invalid_citations_with_validation_error(sample_passages):
    """If model cites chunk ID outside retrieved evidence, return validation error (do not silently drop)."""
    mock_payload = json.dumps({
        "answer": "Project Aurora runs on AWS EKS.",
        "cited_chunk_ids": ["chunk_aurora_01", "hallucinated_chunk_id_42"],
        "status": "answered",
        "missing_information": None,
    })
    mock_client = _make_mock_openai_client(mock_payload)

    result = generate_answer("Where does Aurora run?", sample_passages, client=mock_client)

    # Must return validation error, NOT status="answered"
    assert result["status"] == "error"
    assert "Validation error" in result["error"]
    assert "hallucinated_chunk_id_42" in result["error"]

    # Diagnostics are preserved separately
    assert result["invalid_citations"] == ["hallucinated_chunk_id_42"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk_aurora_01"


def test_generate_answer_requires_at_least_one_valid_citation_for_answered(sample_passages):
    """status='answered' requires at least one valid citation; empty citations must fail validation."""
    mock_payload = json.dumps({
        "answer": "Project Nexus is an internal assistant.",
        "cited_chunk_ids": [],  # Empty citations for factual answer
        "status": "answered",
        "missing_information": None,
    })
    mock_client = _make_mock_openai_client(mock_payload)

    result = generate_answer("What is Project Nexus?", sample_passages, client=mock_client)

    assert result["status"] == "error"
    assert "requires at least one valid citation" in result["error"]


def test_generate_answer_requires_missing_information_for_partial(sample_passages):
    """status='partial' requires non-empty missing_information explaining undocumented details."""
    mock_payload = json.dumps({
        "answer": "Project Aurora was deployed in August 2024.",
        "cited_chunk_ids": ["chunk_aurora_01"],
        "status": "partial",
        "missing_information": None,  # Missing required explanation
    })
    mock_client = _make_mock_openai_client(mock_payload)

    result = generate_answer("When were Aurora and Beacon deployed?", sample_passages, client=mock_client)

    assert result["status"] == "error"
    assert "requires non-empty 'missing_information'" in result["error"]


def test_insufficient_evidence_abstention_distinct_from_validation_errors(sample_passages):
    """Abstention due to insufficient evidence has status='abstain' and error=None."""
    mock_payload = json.dumps({
        "answer": INSUFFICIENT_EVIDENCE_PHRASE,
        "cited_chunk_ids": [],
        "status": "abstain",
        "missing_information": "Exact salary details are undocumented in the excerpts.",
    })
    mock_client = _make_mock_openai_client(mock_payload)

    result = generate_answer("What is Alex's exact salary?", sample_passages, client=mock_client)

    # Clean abstention, NOT an error
    assert result["status"] == "abstain"
    assert result["answer"] == INSUFFICIENT_EVIDENCE_PHRASE
    assert result["error"] is None
    assert result["citations"] == []


# -----------------------------------------------------------------------------
# 4. Prompt Construction & System Guardrails
# -----------------------------------------------------------------------------

def test_build_generation_messages_includes_system_guardrails(sample_passages):
    messages = build_generation_messages("What is Project Nexus?", sample_passages)

    assert len(messages) == 2
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]

    assert "Untrusted Reference Material" in system_content
    assert "SYSTEM OVERRIDE NOTE" in system_content
    assert INSUFFICIENT_EVIDENCE_PHRASE in system_content
    assert "Do not infer 'not deployed' from 'in development'" in system_content

    assert "[Chunk ID: chunk_nexus_00]" in user_content
    assert "[Chunk ID: chunk_aurora_01]" in user_content
    assert "QUESTION:\nWhat is Project Nexus?" in user_content


# -----------------------------------------------------------------------------
# 5. Core Generation Behaviors (Mocked)
# -----------------------------------------------------------------------------

def test_generate_answer_supported_claim(sample_passages):
    mock_payload = json.dumps({
        "answer": "Project Nexus uses ChromaDB and all-MiniLM-L6-v2.",
        "cited_chunk_ids": ["chunk_nexus_00"],
        "status": "answered",
        "missing_information": None,
    })
    mock_client = _make_mock_openai_client(mock_payload)

    result = generate_answer("What does Nexus use?", sample_passages, client=mock_client)

    assert result["status"] == "answered"
    assert "ChromaDB" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source"] == "data/documents/project_nexus.md"
    assert result["invalid_citations"] == []
    assert result["error"] is None


def test_generate_answer_abstention_fast_path_when_no_passages():
    """When retrieval returns zero passages, generation immediately abstains without calling API."""
    mock_client = MagicMock()
    result = generate_answer("What is Alex's private salary?", [], client=mock_client)

    assert result["status"] == "abstain"
    assert result["answer"] == INSUFFICIENT_EVIDENCE_PHRASE
    assert result["citations"] == []
    mock_client.chat.completions.create.assert_not_called()


def test_generate_answer_partial_answer_success(sample_passages):
    mock_payload = json.dumps({
        "answer": "Project Aurora was deployed in August 2024.",
        "cited_chunk_ids": ["chunk_aurora_01"],
        "status": "partial",
        "missing_information": "Project Beacon deployment schedule is not documented.",
    })
    mock_client = _make_mock_openai_client(mock_payload)

    result = generate_answer("When were Aurora and Beacon deployed?", sample_passages, client=mock_client)

    assert result["status"] == "partial"
    assert "August 2024" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["missing_information"] == "Project Beacon deployment schedule is not documented."
    assert result["error"] is None


def test_generate_answer_conflict_reporting_success(sample_passages):
    mock_payload = json.dumps({
        "answer": "There is a conflict: one document states Nexus is an internal staging pilot, while another claims full global deployment.",
        "cited_chunk_ids": ["chunk_nexus_00"],
        "status": "conflict",
        "missing_information": None,
    })
    mock_client = _make_mock_openai_client(mock_payload)

    result = generate_answer("Is Nexus deployed?", sample_passages, client=mock_client)

    assert result["status"] == "conflict"
    assert "conflict" in result["answer"].lower()
    assert len(result["citations"]) == 1
    assert result["error"] is None


# -----------------------------------------------------------------------------
# 6. Fixed Public Error Handling & Network Failure Demonstration
# -----------------------------------------------------------------------------

def test_generate_answer_handles_timeout_error_with_fixed_message(sample_passages):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

    result = generate_answer("What is Project Aurora?", sample_passages, client=mock_client)

    assert result["status"] == "error"
    assert "request timed out" in result["error"].lower()
    assert "check network latency" in result["error"].lower()


def test_generate_answer_handles_rate_limit_with_fixed_message(sample_passages):
    mock_client = MagicMock()
    mock_response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    mock_client.chat.completions.create.side_effect = RateLimitError(
        message="Raw provider leak with sk-internal-key-1234567890abcdef",
        response=mock_response,
        body=None,
    )

    result = generate_answer("What is Project Aurora?", sample_passages, client=mock_client)

    assert result["status"] == "error"
    # Fixed actionable message is used, raw provider string is not reflected
    assert "rate limit exceeded" in result["error"].lower()
    assert "sk-internal-key" not in result["error"]


def test_demonstrate_network_connection_failure_mocked(sample_passages):
    """Demonstrate network unreachability with a mocked APIConnectionError.

    Correction note for review exercise:
    Passing an invalid API key tests authentication failure (HTTP 401), not network
    unreachability. Network unreachability occurs when connection cannot be established
    and is represented by APIConnectionError.
    """
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )

    result = generate_answer("What is Project Aurora?", sample_passages, client=mock_client)

    assert result["status"] == "error"
    assert "connection failed" in result["error"].lower()
    assert "verify your internet connection" in result["error"].lower()


def test_demonstrate_authentication_failure_mocked(sample_passages):
    """Demonstrate authentication failure (HTTP 401) with a mocked APIStatusError."""
    mock_client = MagicMock()
    mock_response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    mock_client.chat.completions.create.side_effect = APIStatusError(
        message="Incorrect API key provided",
        response=mock_response,
        body=None,
    )

    result = generate_answer("What is Project Aurora?", sample_passages, client=mock_client)

    assert result["status"] == "error"
    assert "authentication failed (HTTP 401)" in result["error"]
    assert "verify that your OPENAI_API_KEY in .env is valid" in result["error"]


# -----------------------------------------------------------------------------
# 7. End-to-End Pipeline
# -----------------------------------------------------------------------------

def test_query_pipeline_end_to_end_mocked(tmp_path):
    """Verify query_pipeline coordinates retrieval and generation with distance labeling."""
    storage_dir = tmp_path / "chroma"
    test_chunks = [
        ChunkRecord(
            chunk_id="chunk_aurora_test",
            text="Project Aurora is deployed in production on AWS EKS since August 2024.",
            source="data/documents/project_aurora.md",
            page=1,
            chunk_index=0,
            content_hash="hash_aurora",
            word_count=12,
        )
    ]
    embeddings = embed_texts([c.text for c in test_chunks])
    build_index(
        chunks=test_chunks,
        embeddings=embeddings,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        storage_dir=storage_dir,
    )

    mock_payload = json.dumps({
        "answer": "Project Aurora was deployed in August 2024 on AWS EKS.",
        "cited_chunk_ids": ["chunk_aurora_test"],
        "status": "answered",
        "missing_information": None,
    })
    mock_client = _make_mock_openai_client(mock_payload)

    result = query_pipeline(
        question="When was Project Aurora deployed?",
        top_k=2,
        storage_dir=storage_dir,
        client=mock_client,
    )

    assert result["status"] == "answered"
    assert "August 2024" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk_aurora_test"
    assert result["citations"][0]["source"] == "data/documents/project_aurora.md"
    assert result["citations"][0]["page"] == 1

    # Verify retrieved evidence explicitly labels distance
    assert len(result["retrieved_evidence"]) == 1
    assert "distance" in result["retrieved_evidence"][0]
    assert result["retrieved_evidence"][0]["distance"] >= 0.0

    # Verify latency breakdown
    assert "retrieval_ms" in result["timings"]
    assert "generation_ms" in result["timings"]
    assert "total_ms" in result["timings"]
    assert result["timings"]["total_ms"] >= result["timings"]["retrieval_ms"]

    # Verify citation disclaimer
    assert result["citation_disclaimer"] == CITATION_VALIDITY_DISCLAIMER


def test_query_pipeline_empty_query_rejected():
    """Empty query should return structured error result without unhandled exception."""
    result = query_pipeline("   ")

    assert result["status"] == "error"
    assert "empty" in result["error"].lower()
    assert result["retrieved_evidence"] == []
    assert result["citations"] == []


def test_system_prompt_includes_refined_conflict_and_comparison_rules():
    """Verify that build_system_prompt includes refined conflict, comparison, and tool-role instructions.

    NOTE: This test verifies prompt construction only. Mocked unit tests verify
    prompt assembly and schema contracts; they DO NOT prove live model compliance.
    """
    prompt = build_system_prompt()

    # Rule 2: Entity focus without volunteering unrelated projects
    assert "Focus directly on the entities, projects, or persons asked about" in prompt
    assert "Do not volunteer or introduce unrelated projects" in prompt

    # Rule 3: Asymmetric comparisons with missing metrics must use partial
    assert "Partial Answers & Missing Metrics in Comparisons" in prompt
    assert "when asked to compare two entities or projects on a requested metric" in prompt
    assert "return status 'partial' and specify the missing metric in 'missing_information'" in prompt

    # Rule 4: Neutral conflict opening
    assert "You MUST open your answer by explicitly acknowledging that the provided sources contain conflicting reports" in prompt
    assert "without selecting one side as definitive truth" in prompt

    # Rule 5: Preserving tool roles without inventing narrow categories
    assert "Tool Roles and Categories" in prompt
    assert "preserve their documented functional roles" in prompt
    assert "list the tool names neutrally rather than inventing an inaccurate narrower category" in prompt


def test_prompt_and_schema_consistency_missing_information_type(sample_passages):
    """Verify prompt and parser consistency: missing_information is strictly string or null, never an array.

    Ensures that:
    1. System prompt defines missing_information as string or null, explicitly forbidding arrays.
    2. Parser rejects array/list types with a descriptive validation error.
    3. Parser accepts valid string or null values.
    """
    prompt = build_system_prompt()
    assert "must be a single string or null; never an array or list" in prompt
    valid_id = sample_passages[0]["chunk_id"]

    # Test parser rejects array payload
    array_payload = (
        '{\n'
        '  "answer": "Project Aurora achieves p99 < 18ms.",\n'
        f'  "cited_chunk_ids": ["{valid_id}"],\n'
        '  "status": "partial",\n'
        '  "missing_information": ["Project Beacon latency benchmarks are missing"]\n'
        '}'
    )
    result_array = parse_model_response(array_payload, sample_passages)
    assert result_array["status"] == "error"
    assert "Field 'missing_information' must be a string or null" in result_array["error"]

    # Test parser accepts string payload
    str_payload = (
        '{\n'
        '  "answer": "Project Aurora achieves p99 < 18ms.",\n'
        f'  "cited_chunk_ids": ["{valid_id}"],\n'
        '  "status": "partial",\n'
        '  "missing_information": "Project Beacon latency benchmarks are missing."\n'
        '}'
    )
    result_str = parse_model_response(str_payload, sample_passages)
    assert result_str["status"] == "partial"
    assert result_str["missing_information"] == "Project Beacon latency benchmarks are missing."
    assert result_str["error"] is None

    # Test parser accepts null payload
    null_payload = (
        '{\n'
        '  "answer": "Based on documents, Project Aurora achieves p99 < 18ms.",\n'
        f'  "cited_chunk_ids": ["{valid_id}"],\n'
        '  "status": "answered",\n'
        '  "missing_information": null\n'
        '}'
    )
    result_null = parse_model_response(null_payload, sample_passages)
    assert result_null["status"] == "answered"
    assert result_null["missing_information"] is None
    assert result_null["error"] is None
