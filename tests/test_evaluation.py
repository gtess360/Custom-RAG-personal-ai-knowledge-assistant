"""Offline unit and integration tests for the Phase 6 evaluation runner.

All tests in this module run offline with mocked pipeline interactions.
Zero paid API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from evaluation.evaluate import (
    load_benchmark_questions,
    capture_environment_metadata,
    is_api_failure,
    evaluate_single_case,
    aggregate_metrics,
    run_evaluation,
    format_markdown_report,
    make_mock_pipeline,
)
from src.config import PATHS


def test_load_benchmark_questions_schema_and_splits():
    """Verify evaluation/questions.json loads cleanly and preserves 14 dev / 6 holdout splits."""
    questions = load_benchmark_questions()
    assert len(questions) == 20

    dev_questions = [q for q in questions if q.get("split") == "development"]
    holdout_questions = [q for q in questions if q.get("split") == "holdout"]

    assert len(dev_questions) == 14
    assert len(holdout_questions) == 6

    # Verify each question contains all mandatory schema fields
    required_keys = {
        "id",
        "category",
        "question",
        "answerable",
        "expected_answer_facts",
        "expected_sources",
        "expected_evidence",
        "forbidden_claims",
        "expected_behavior",
        "split",
    }
    for q in questions:
        missing = required_keys - set(q.keys())
        assert not missing, f"Question {q.get('id')} missing keys: {missing}"
        assert q["split"] in {"development", "holdout"}
        assert q["expected_behavior"] in {"answer", "abstain", "partial", "report_conflict"}


def test_capture_environment_metadata():
    """Verify environment metadata captures platform, model, and index configuration."""
    meta = capture_environment_metadata()
    assert "timestamp" in meta
    assert "python_version" in meta
    assert "platform" in meta
    assert "generation_model" in meta
    assert "index_configuration" in meta

    idx = meta["index_configuration"]
    assert "collection_name" in idx
    assert "chunk_size" in idx
    assert "chunk_overlap" in idx
    assert "distance_metric" in idx


def test_is_api_failure_distinction():
    """Verify API failures (rate limits, timeouts) are distinguished from answer-quality failures."""
    # API failure cases
    assert is_api_failure("Rate limit reached for gpt-4o-mini (429)", "error") is True
    assert is_api_failure("Request timed out after 30 seconds (APITimeoutError)", "error") is True
    assert is_api_failure("Connection reset by peer (APIConnectionError)", "error") is True
    assert is_api_failure("OpenAI 500 Internal Server Error", "error") is True

    # Answer quality or schema validation failure cases (NOT API failures)
    assert is_api_failure("Model output failed schema validation", "error") is False
    assert is_api_failure("Required field 'citations' missing in JSON", "error") is False
    assert is_api_failure(None, "answered") is False
    assert is_api_failure(None, "abstain") is False


def test_evaluate_single_case_answerable_success():
    """Verify evaluation of a fully answerable case with valid citations and source recall."""
    question_record = {
        "id": "test_q01",
        "category": "direct_facts",
        "question": "What database is used?",
        "answerable": True,
        "expected_sources": ["data/documents/project_nexus.md"],
        "expected_evidence": ["Vector Store: ChromaDB running in persistent local mode."],
        "expected_answer_facts": ["ChromaDB"],
        "forbidden_claims": ["Pinecone"],
        "expected_behavior": "answer",
        "split": "development",
    }

    mock_pipeline_fn = lambda q, k: {
        "question": q,
        "answer": "Project Nexus uses ChromaDB running in persistent mode.",
        "status": "answered",
        "missing_information": None,
        "citations": [{"chunk_id": "chunk_01", "source": "data/documents/project_nexus.md", "page": None}],
        "invalid_citations": [],
        "retrieved_evidence": [
            {
                "chunk_id": "chunk_01",
                "source": "data/documents/project_nexus.md",
                "page": None,
                "text": "Vector Store: ChromaDB running in persistent local mode.",
            }
        ],
        "timings": {"retrieval_ms": 15.0, "generation_ms": 200.0, "total_ms": 215.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    result = evaluate_single_case(question_record, mock_pipeline_fn, top_k=4)

    assert result["question_id"] == "test_q01"
    assert result["status"] == "answered"
    assert result["source_recall"] == 1.0
    assert result["evidence_sufficiency"] == "unknown"
    assert result["missing_information"] is None
    assert len(result["retrieved_evidence_text"]) == 1
    assert result["citations_emitted_count"] == 1
    assert result["citations_valid_count"] == 1
    assert result["citations_invalid_count"] == 0
    assert result["status_matches_behavior"] is True
    assert result["is_false_abstention"] is False
    assert result["forbidden_claims_detected"] == []
    assert result["is_api_error"] is False
    # Crucial: Unreviewed answers must NOT be marked as correct automatically
    assert result["manual_review"]["reviewed"] is False
    assert result["manual_review"]["correctness_score"] is None


def test_evaluate_single_case_forbidden_claim_detected():
    """Verify that forbidden claims in an answer are detected and flagged as a red flag."""
    question_record = {
        "id": "test_q01",
        "category": "direct_facts",
        "question": "What database is used?",
        "answerable": True,
        "expected_sources": ["data/documents/project_nexus.md"],
        "expected_evidence": ["Vector Store: ChromaDB"],
        "expected_answer_facts": ["ChromaDB"],
        "forbidden_claims": ["Pinecone", "Weaviate"],
        "expected_behavior": "answer",
        "split": "development",
    }

    mock_pipeline_fn = lambda q, k: {
        "question": q,
        "answer": "Project Nexus uses Pinecone for vector search.",
        "status": "answered",
        "citations": [{"chunk_id": "c1", "source": "project_nexus.md"}],
        "invalid_citations": [],
        "retrieved_evidence": [{"chunk_id": "c1", "source": "data/documents/project_nexus.md", "text": "ChromaDB"}],
        "timings": {"retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    result = evaluate_single_case(question_record, mock_pipeline_fn, top_k=4)
    assert "Pinecone" in result["forbidden_claims_detected"]


def test_evaluate_single_case_correct_abstention():
    """Verify that an unanswerable question resulting in an abstention is recorded as a correct abstention."""
    question_record = {
        "id": "test_q13",
        "category": "missing_out_of_scope",
        "question": "What is Alex Mercer's current base salary?",
        "answerable": False,
        "expected_sources": [],
        "expected_evidence": [],
        "expected_answer_facts": ["Salary is undocumented"],
        "forbidden_claims": ["$150,000"],
        "expected_behavior": "abstain",
        "split": "development",
    }

    mock_pipeline_fn = lambda q, k: {
        "question": q,
        "answer": "I do not have sufficient information in the provided documentation to answer this question.",
        "status": "abstain",
        "missing_information": None,
        "citations": [],
        "invalid_citations": [],
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 12.0, "generation_ms": 30.0, "total_ms": 42.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    result = evaluate_single_case(question_record, mock_pipeline_fn, top_k=4)
    assert result["status"] == "abstain"
    assert result["is_correct_abstention"] is True
    assert result["is_false_answer"] is False
    assert result["is_false_abstention"] is False
    assert result["status_matches_behavior"] is True


def test_evaluate_single_case_false_abstention():
    """Verify that an answerable question resulting in an abstention is recorded as a false abstention."""
    question_record = {
        "id": "test_q02",
        "category": "direct_facts",
        "question": "When was Project Aurora deployed?",
        "answerable": True,
        "expected_sources": ["data/documents/project_aurora.md"],
        "expected_evidence": ["August 2024"],
        "expected_answer_facts": ["August 2024"],
        "forbidden_claims": [],
        "expected_behavior": "answer",
        "split": "development",
    }

    mock_pipeline_fn = lambda q, k: {
        "question": q,
        "answer": "I do not have sufficient information in the provided documentation to answer this question.",
        "status": "abstain",
        "missing_information": None,
        "citations": [],
        "invalid_citations": [],
        "retrieved_evidence": [{"chunk_id": "c1", "source": "data/documents/project_aurora.md", "text": "August 2024"}],
        "timings": {"retrieval_ms": 10.0, "generation_ms": 20.0, "total_ms": 30.0},
        "model": "gpt-4o-mini",
        "error": None,
    }

    result = evaluate_single_case(question_record, mock_pipeline_fn, top_k=4)
    assert result["status"] == "abstain"
    assert result["is_false_abstention"] is True
    assert result["status_matches_behavior"] is False


def test_aggregate_metrics_calculations():
    """Verify aggregate statistical calculations across a simulated case set."""
    cases = [
        {
            "question_id": "q1",
            "split": "development",
            "status": "answered",
            "source_recall": 1.0,
            "all_sources_retrieved": True,
            "expected_sources": ["doc1.md"],
            "citations_emitted_count": 2,
            "citations_valid_count": 2,
            "citations_invalid_count": 0,
            "expected_behavior": "answer",
            "status_matches_behavior": True,
            "is_correct_abstention": False,
            "is_false_abstention": False,
            "is_false_answer": False,
            "forbidden_claims_detected": [],
            "is_api_error": False,
            "timings": {"retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
            "manual_review": {"reviewed": False},
        },
        {
            "question_id": "q2",
            "split": "development",
            "status": "abstain",
            "source_recall": 1.0,
            "all_sources_retrieved": True,
            "expected_sources": [],
            "citations_emitted_count": 0,
            "citations_valid_count": 0,
            "citations_invalid_count": 0,
            "expected_behavior": "abstain",
            "status_matches_behavior": True,
            "is_correct_abstention": True,
            "is_false_abstention": False,
            "is_false_answer": False,
            "forbidden_claims_detected": [],
            "is_api_error": False,
            "timings": {"retrieval_ms": 8.0, "generation_ms": 20.0, "total_ms": 28.0},
            "manual_review": {"reviewed": False},
        },
        {
            "question_id": "q3",
            "split": "development",
            "status": "error",
            "source_recall": 0.0,
            "all_sources_retrieved": False,
            "expected_sources": ["doc2.md"],
            "citations_emitted_count": 0,
            "citations_valid_count": 0,
            "citations_invalid_count": 0,
            "expected_behavior": "answer",
            "status_matches_behavior": False,
            "is_correct_abstention": False,
            "is_false_abstention": False,
            "is_false_answer": False,
            "forbidden_claims_detected": [],
            "is_api_error": True,
            "timings": {"retrieval_ms": 10.0, "generation_ms": 5.0, "total_ms": 15.0},
            "manual_review": {"reviewed": False},
        },
    ]

    metrics = aggregate_metrics(cases)
    assert metrics["total_cases"] == 3
    assert metrics["api_failures"] == 1
    assert metrics["citations"]["total_emitted"] == 2
    assert metrics["citations"]["total_valid"] == 2
    assert metrics["citations"]["validity_rate"] == 1.0
    assert metrics["behavior"]["unanswerable_cases"] == 1
    assert metrics["behavior"]["correct_abstentions"] == 1
    assert metrics["behavior"]["abstention_recall"] == 1.0
    assert metrics["manual_review"]["reviewed_count"] == 0
    assert metrics["manual_review"]["pending_count"] == 3


def test_run_evaluation_splits_and_reports_separately():
    """Verify run_evaluation segregates results by development and holdout splits."""
    mock_questions = [
        {
            "id": "q_dev_1",
            "category": "direct_facts",
            "question": "Dev question?",
            "answerable": True,
            "expected_sources": ["data/documents/project_nexus.md"],
            "expected_evidence": ["evidence"],
            "expected_answer_facts": ["fact"],
            "forbidden_claims": [],
            "expected_behavior": "answer",
            "split": "development",
        },
        {
            "id": "q_hold_1",
            "category": "direct_facts",
            "question": "Holdout question?",
            "answerable": True,
            "expected_sources": ["data/documents/project_aurora.md"],
            "expected_evidence": ["evidence"],
            "expected_answer_facts": ["fact"],
            "forbidden_claims": [],
            "expected_behavior": "answer",
            "split": "holdout",
        },
    ]

    mock_pipeline = make_mock_pipeline(mock_mode="nominal")
    results = run_evaluation(
        questions=mock_questions,
        pipeline_fn=mock_pipeline,
        split_filter="all",
        top_k=2,
    )

    assert "metadata" in results
    assert "summary" in results
    assert "development_split" in results["summary"]
    assert "holdout_split" in results["summary"]
    assert "combined" in results["summary"]

    assert results["summary"]["development_split"]["total_cases"] == 1
    assert results["summary"]["holdout_split"]["total_cases"] == 1
    assert results["summary"]["combined"]["total_cases"] == 2

    # Verify Markdown report format includes split tables
    md_report = format_markdown_report(results)
    assert "Development Split (14)" in md_report or "Development Split" in md_report
    assert "Holdout Split" in md_report
    assert "Manual Review Status" in md_report


def test_make_mock_pipeline_api_error_simulation():
    """Verify make_mock_pipeline with api_error simulates external API timeout error correctly."""
    mock_pipeline = make_mock_pipeline(mock_mode="api_error")
    res = mock_pipeline("Any question?", 2)

    assert res["status"] == "error"
    assert "timed out" in res["error"].lower()
    assert is_api_failure(res["error"], res["status"]) is True


def test_make_mock_pipeline_invalid_citations_simulation():
    """Verify make_mock_pipeline with invalid_citations injects unresolved citations."""
    mock_pipeline = make_mock_pipeline(mock_mode="invalid_citations")
    res = mock_pipeline("What database was used?", 2)

    assert len(res["invalid_citations"]) > 0
    assert "chunk_fabricated_999" in res["invalid_citations"]


def test_run_evaluation_question_ids_filtering():
    """Verify run_evaluation filters questions by question_ids while respecting split_filter."""
    mock_questions = [
        {"id": "q01", "split": "development", "category": "direct_facts", "question": "Q1", "expected_sources": [], "expected_evidence": [], "forbidden_claims": [], "expected_behavior": "answer", "answerable": True},
        {"id": "q04", "split": "development", "category": "direct_facts", "question": "Q4", "expected_sources": [], "expected_evidence": [], "forbidden_claims": [], "expected_behavior": "answer", "answerable": True},
        {"id": "q05", "split": "holdout", "category": "direct_facts", "question": "Q5", "expected_sources": [], "expected_evidence": [], "forbidden_claims": [], "expected_behavior": "answer", "answerable": True},
    ]

    mock_pipeline = make_mock_pipeline(mock_mode="nominal")

    # Filter by specific question_ids in dev split
    res_dev = run_evaluation(
        questions=mock_questions,
        pipeline_fn=mock_pipeline,
        split_filter="dev",
        top_k=2,
        question_ids=["q04"],
    )
    assert res_dev["evaluation_settings"]["total_questions_run"] == 1
    assert [c["question_id"] for c in res_dev["case_results"]] == ["q04"]

    # Filter with holdout question while split is dev -> should NOT include holdout question
    res_cross = run_evaluation(
        questions=mock_questions,
        pipeline_fn=mock_pipeline,
        split_filter="dev",
        top_k=2,
        question_ids=["q05"],  # q05 is holdout, but split_filter is dev
    )
    assert res_cross["evaluation_settings"]["total_questions_run"] == 0
    assert len(res_cross["case_results"]) == 0


def test_aggregate_metrics_no_eligible_cases():
    """Verify metrics with zero eligible cases return None and 'N/A' rather than 1.0 or 100%."""
    # Case set with ONLY answerable questions (0 unanswerable cases, 0 citations emitted)
    cases_answerable_only = [
        {
            "question_id": "q1",
            "split": "development",
            "status": "answered",
            "source_recall": 1.0,
            "all_sources_retrieved": True,
            "expected_sources": ["doc1.md"],
            "citations_emitted_count": 0,
            "citations_valid_count": 0,
            "citations_invalid_count": 0,
            "expected_behavior": "answer",
            "status_matches_behavior": True,
            "is_correct_abstention": False,
            "is_false_abstention": False,
            "is_false_answer": False,
            "forbidden_claims_detected": [],
            "is_api_error": False,
            "timings": {"retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
            "manual_review": {"reviewed": False},
        }
    ]

    metrics = aggregate_metrics(cases_answerable_only)
    # Unanswerable metrics must be N/A when unanswerable_count == 0
    assert metrics["behavior"]["unanswerable_cases"] == 0
    assert metrics["behavior"]["abstention_recall"] is None
    assert metrics["behavior"]["abstention_recall_pct"] == "N/A"
    assert metrics["behavior"]["false_answer_rate"] is None
    assert metrics["behavior"]["false_answer_pct"] == "N/A"
    # Citations validity must be N/A when total_emitted == 0
    assert metrics["citations"]["total_emitted"] == 0
    assert metrics["citations"]["validity_rate"] is None
    assert metrics["citations"]["validity_rate_pct"] == "N/A"

    # Case set with ONLY unanswerable questions (0 answerable cases)
    cases_unanswerable_only = [
        {
            "question_id": "q2",
            "split": "development",
            "status": "abstain",
            "source_recall": 1.0,
            "all_sources_retrieved": True,
            "expected_sources": [],
            "citations_emitted_count": 0,
            "citations_valid_count": 0,
            "citations_invalid_count": 0,
            "expected_behavior": "abstain",
            "status_matches_behavior": True,
            "is_correct_abstention": True,
            "is_false_abstention": False,
            "is_false_answer": False,
            "forbidden_claims_detected": [],
            "is_api_error": False,
            "timings": {"retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
            "manual_review": {"reviewed": False},
        }
    ]

    metrics_unans = aggregate_metrics(cases_unanswerable_only)
    assert metrics_unans["retrieval"]["evaluated_cases"] == 0
    assert metrics_unans["retrieval"]["mean_source_recall"] is None
    assert metrics_unans["retrieval"]["mean_source_recall_pct"] == "N/A"
    assert metrics_unans["behavior"]["answerable_cases"] == 0
    assert metrics_unans["behavior"]["false_abstention_rate"] is None
    assert metrics_unans["behavior"]["false_abstention_pct"] == "N/A"


def test_format_markdown_report_zero_eligible_cases_displays_na():
    """Verify Markdown report summary table displays 'N/A' for splits with 0 evaluated cases."""
    eval_data = {
        "metadata": {
            "timestamp": "2026-10-24T20:00:00Z",
            "generation_model": "gpt-4o-mini",
            "platform": "Windows",
            "python_version": "3.12.0",
            "index_configuration": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "collection_chunk_count": 10,
                "chunk_size": 120,
                "chunk_overlap": 25,
            },
        },
        "summary": {
            "development_split": {
                "total_cases": 1,
                "api_failures": 0,
                "pipeline_errors": 0,
                "retrieval": {"mean_source_recall_pct": "100.0%"},
                "citations": {"validity_rate_pct": "100.0%"},
                "behavior": {
                    "abstention_recall_pct": "N/A",  # 0 unanswerable cases
                    "false_abstention_pct": "0.0%",
                    "behavior_alignment_pct": "100.0%",
                    "forbidden_claims_detected_count": 0,
                },
                "latency": {"total": {"median_ms": 120.0, "p95_ms": 120.0}},
                "manual_review": {"status": "Pending manual review"},
            },
            "holdout_split": {
                "total_cases": 0,  # Holdout was not run
            },
            "combined": {
                "total_cases": 1,
                "api_failures": 0,
                "pipeline_errors": 0,
                "retrieval": {"mean_source_recall_pct": "100.0%"},
                "citations": {"validity_rate_pct": "100.0%"},
                "behavior": {
                    "abstention_recall_pct": "N/A",
                    "false_abstention_pct": "0.0%",
                    "behavior_alignment_pct": "100.0%",
                    "forbidden_claims_detected_count": 0,
                },
                "latency": {"total": {"median_ms": 120.0, "p95_ms": 120.0}},
                "manual_review": {"status": "Pending manual review"},
            },
        },
        "case_results": [],
    }

    report = format_markdown_report(eval_data)

    # In holdout column: Evaluated Cases should be 0, all metrics must be N/A
    assert "| **Evaluated Cases** | 1 | 0 | 1 |" in report
    assert "| **API Failures** | 0 | N/A | 0 |" in report
    assert "| **Pipeline Errors** | 0 | N/A | 0 |" in report
    assert "| **Source Recall@k** | 100.0% | N/A | 100.0% |" in report
    assert "| **Citation Validity Rate** | 100.0% | N/A | 100.0% |" in report
    assert "| **Abstention Recall** | N/A | N/A | N/A |" in report
    assert "| **Median Latency (Total)** | 120.0 ms | N/A | 120.0 ms |" in report
    assert "| **p95 Latency (Total)** | 120.0 ms | N/A | 120.0 ms |" in report
    assert "| **Manual Review State** | Pending manual review | N/A | Pending manual review |" in report

