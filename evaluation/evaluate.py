"""Evaluation runner for the Personal AI Knowledge Assistant.

Executes and measures the fixed 20-question benchmark (docs/EVALUATION_PLAN.md).
Features:
1. Strict separation of Development (14) and Holdout (6) splits.
2. Objective measurement of retrieval (Source Recall@k) and citation validity.
3. Separation of syntactically valid citation IDs from semantic factual support.
4. Tracking of correct abstentions and incorrect refusals (false abstentions).
5. Separate accounting for API failures (rate limits, timeouts) vs answer-quality failures.
6. Recording of full corpus/index configuration and evaluation runtime metadata.
7. Safe offline dry-run mode and retrieval-only mode (0 API calls).
8. Strict guard requiring explicit '--live' flag before executing live OpenAI API requests.
9. Unreviewed answers are marked 'reviewed: false' and never assumed correct.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


# Add project root to sys.path if running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PATHS,
    OPENAI_MODEL,
    DEFAULT_TOP_K,
)

class PipelineCallable(Protocol):
    """Protocol for pipeline callables accepting question string and optional top_k."""
    def __call__(self, question: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        ...
from src.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src.pipeline import query_pipeline
from src.retrieval import retrieve
from src.vector_store import (
    _get_client,
    COLLECTION_NAME,
    check_index_compatibility,
)

# Status to behavior mapping
BEHAVIOR_TO_STATUS = {
    "answer": "answered",
    "abstain": "abstain",
    "partial": "partial",
    "report_conflict": "conflict",
}


def load_benchmark_questions(questions_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load and validate the benchmark questions from questions.json."""
    path = questions_path or (PATHS.evaluation_dir / "questions.json")
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark questions file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError(f"Questions file at {path} must contain a non-empty list.")

    return questions


def capture_environment_metadata(storage_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Capture corpus, index configuration, and system runtime metadata."""
    storage_path = storage_dir or PATHS.storage_dir
    index_meta: Dict[str, Any] = {
        "storage_dir": str(storage_path),
        "collection_name": COLLECTION_NAME,
        "collection_chunk_count": 0,
        "embedding_model": "unknown",
        "embedding_dimension": 0,
        "distance_metric": "cosine",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "is_compatible": False,
        "compatibility_message": "",
    }

    try:
        client = _get_client(storage_path)
        coll = client.get_collection(COLLECTION_NAME)
        index_meta["collection_chunk_count"] = coll.count()
        meta = coll.metadata or {}
        index_meta["embedding_model"] = meta.get("embedding_model", "unknown")
        index_meta["embedding_dimension"] = meta.get("embedding_dimension", 0)
        index_meta["distance_metric"] = meta.get("distance_metric", "cosine")
        index_meta["chunk_size"] = meta.get("chunk_size", DEFAULT_CHUNK_SIZE)
        index_meta["chunk_overlap"] = meta.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

        compatible, msg = check_index_compatibility(storage_path)
        index_meta["is_compatible"] = compatible
        index_meta["compatibility_message"] = msg
    except Exception as e:
        index_meta["compatibility_message"] = f"Index inspect error: {e}"

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "generation_model": OPENAI_MODEL,
        "corpus_dir": str(PATHS.data_dir),
        "index_configuration": index_meta,
    }


def is_api_failure(error_msg: Optional[str], status: str) -> bool:
    """Determine whether an execution failure was caused by an external API or network error."""
    if not error_msg and status != "error":
        return False
    err_lower = (error_msg or "").lower()
    api_indicators = [
        "rate limit",
        "ratelimit",
        "429",
        "timeout",
        "connection",
        "network",
        "authentication",
        "401",
        "api error",
        "openai",
        "httpx",
        "service unavailable",
        "503",
        "502",
        "500",
    ]
    return any(indicator in err_lower for indicator in api_indicators)


def evaluate_single_case(
    question_record: Dict[str, Any],
    pipeline_fn: PipelineCallable,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """Execute a single evaluation case through the pipeline and compute objective metrics."""
    q_id = question_record["id"]
    category = question_record["category"]
    split = question_record["split"]
    question_text = question_record["question"]
    expected_behavior = question_record.get("expected_behavior", "answer")
    expected_sources = question_record.get("expected_sources", [])
    expected_evidence = question_record.get("expected_evidence", [])
    expected_facts = question_record.get("expected_answer_facts", [])
    forbidden_claims = question_record.get("forbidden_claims", [])
    is_answerable = question_record.get("answerable")

    # Run query through provided pipeline function
    result = pipeline_fn(question_text, top_k)

    answer = result.get("answer", "")
    status = result.get("status", "error")
    error = result.get("error")
    missing_information = result.get("missing_information")
    citations = result.get("citations", [])
    invalid_citations = result.get("invalid_citations", [])
    retrieved_evidence = result.get("retrieved_evidence", [])
    exact_evidence_texts = [ev.get("text", "") for ev in retrieved_evidence]
    timings = result.get("timings", {"retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": 0.0})
    model = result.get("model", "unknown")

    # 1. Retrieval Analysis
    retrieved_chunk_ids = [ev.get("chunk_id", "") for ev in retrieved_evidence if ev.get("chunk_id")]
    retrieved_sources = list({ev.get("source", "") for ev in retrieved_evidence if ev.get("source")})

    if expected_sources:
        # Check fraction of expected sources appearing in retrieved passages
        found_sources = set(expected_sources) & set(retrieved_sources)
        source_recall = len(found_sources) / len(expected_sources)
        all_sources_retrieved = len(found_sources) == len(expected_sources)
    else:
        source_recall = None
        all_sources_retrieved = None

    # Semantic evidence sufficiency cannot be reliably measured by heuristic substring
    # or keyword matching. It requires human inspection per evaluation/rubric.md.
    # We label automated sufficiency as 'unknown' rather than zero or pseudo-numeric.
    evidence_sufficiency = "unknown"

    # 2. Citation Analysis
    # Emitted citation IDs are the combination of resolved and invalid IDs
    emitted_citation_ids = [c.get("chunk_id") for c in citations if isinstance(c, dict)] + list(invalid_citations)
    valid_citation_ids = [c.get("chunk_id") for c in citations if isinstance(c, dict)]

    citations_emitted_count = len(emitted_citation_ids)
    citations_valid_count = len(valid_citation_ids)
    citations_invalid_count = len(invalid_citations)

    if citations_emitted_count > 0:
        citation_validity_rate = round(citations_valid_count / citations_emitted_count, 4)
    else:
        citation_validity_rate = None  # No emitted citations: rate is not applicable.

    # 3. Behavioral Alignment Checks
    expected_status = BEHAVIOR_TO_STATUS.get(expected_behavior, "answered")
    status_matches_behavior = (status == expected_status)

    is_unanswerable_case = (is_answerable is False) or (category in {"missing_out_of_scope", "instruction_injection"})
    is_fully_answerable = (is_answerable is True)

    is_correct_abstention = is_unanswerable_case and (status == "abstain")
    is_false_abstention = is_fully_answerable and (status == "abstain")
    is_false_answer = is_unanswerable_case and (status == "answered")

    # 4. Forbidden Claim Red Flag (verbatim string check)
    # Important: Absence of forbidden claims does NOT prove correctness.
    forbidden_detected = []
    answer_lower = answer.lower()
    for forbidden in forbidden_claims:
        if forbidden.strip().lower() in answer_lower:
            forbidden_detected.append(forbidden)

    # 5. API Failure Detection
    is_api_err = is_api_failure(error, status)

    return {
        "question_id": q_id,
        "category": category,
        "split": split,
        "question": question_text,
        "expected_behavior": expected_behavior,
        "expected_status": expected_status,
        "status": status,
        "answer": answer,
        "missing_information": missing_information,
        "error": error,
        "is_api_error": is_api_err,
        "model": model,
        "timings": timings,
        # Retrieval fields
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_sources": retrieved_sources,
        "retrieved_evidence_text": exact_evidence_texts,
        "expected_sources": expected_sources,
        "source_recall": round(source_recall, 4) if source_recall is not None else None,
        "all_sources_retrieved": all_sources_retrieved,
        "evidence_sufficiency": evidence_sufficiency,
        # Citation fields
        "citations": citations,
        "invalid_citations": invalid_citations,
        "citations_emitted_count": citations_emitted_count,
        "citations_valid_count": citations_valid_count,
        "citations_invalid_count": citations_invalid_count,
        "citation_validity_rate": citation_validity_rate,
        # Behavioral fields
        "status_matches_behavior": status_matches_behavior,
        "is_correct_abstention": is_correct_abstention,
        "is_false_abstention": is_false_abstention,
        "is_false_answer": is_false_answer,
        # Red flags
        "forbidden_claims_detected": forbidden_detected,
        # Manual Review Template (unreviewed by default)
        "manual_review": {
            "reviewed": False,
            "reviewer": None,
            "correctness_score": None,      # 0, 1, or 2 (see evaluation/rubric.md)
            "groundedness_score": None,     # 0, 1, or 2
            "citation_support_score": None, # 0, 1, or 2
            "relevance_score": None,        # 0, 1, or 2
            "notes": "Pending human manual review. Automated checks do not establish factual support.",
        },
    }


def aggregate_metrics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistical metrics across a subset of evaluation cases."""
    total_cases = len(cases)
    if total_cases == 0:
        return {"total_cases": 0}

    # Failure counts
    api_errors = sum(1 for c in cases if c.get("is_api_error"))
    pipeline_errors = sum(1 for c in cases if c.get("status") == "error" and not c.get("is_api_error"))

    # Status distribution
    status_counts: Dict[str, int] = {}
    for c in cases:
        st = c.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    # Retrieval metrics (over answerable cases where sources are expected)
    answerable_cases = [c for c in cases if c.get("expected_sources")]
    if answerable_cases:
        mean_source_recall = sum(c["source_recall"] for c in answerable_cases) / len(answerable_cases)
        perfect_source_recall_count = sum(1 for c in answerable_cases if c["all_sources_retrieved"])
        source_recall_str = f"{round(mean_source_recall * 100, 1)}%"
    else:
        mean_source_recall = None
        perfect_source_recall_count = 0
        source_recall_str = "N/A"

    # Citation metrics
    total_emitted = sum(c["citations_emitted_count"] for c in cases)
    total_valid = sum(c["citations_valid_count"] for c in cases)
    total_invalid = sum(c["citations_invalid_count"] for c in cases)
    if total_emitted > 0:
        overall_citation_validity = round(total_valid / total_emitted, 4)
        citation_validity_str = f"{round(overall_citation_validity * 100, 1)}%"
    else:
        overall_citation_validity = None
        citation_validity_str = "N/A"

    # Abstention metrics
    unanswerable = [c for c in cases if c.get("expected_behavior") == "abstain"]
    unanswerable_count = len(unanswerable)
    if unanswerable_count > 0:
        correct_abstentions = sum(1 for c in unanswerable if c["is_correct_abstention"])
        abstention_recall_rate = round(correct_abstentions / unanswerable_count, 4)
        abstention_recall_str = f"{round(abstention_recall_rate * 100, 1)}%"
        false_answers = sum(1 for c in unanswerable if c["is_false_answer"])
        false_answer_rate = round(false_answers / unanswerable_count, 4)
        false_answer_str = f"{round(false_answer_rate * 100, 1)}%"
    else:
        correct_abstentions = 0
        abstention_recall_rate = None
        abstention_recall_str = "N/A"
        false_answers = 0
        false_answer_rate = None
        false_answer_str = "N/A"

    fully_answerable = [c for c in cases if c.get("expected_behavior") == "answer"]
    fully_answerable_count = len(fully_answerable)
    if fully_answerable_count > 0:
        false_abstentions = sum(1 for c in fully_answerable if c["is_false_abstention"])
        false_abstention_rate = round(false_abstentions / fully_answerable_count, 4)
        false_abstention_str = f"{round(false_abstention_rate * 100, 1)}%"
    else:
        false_abstentions = 0
        false_abstention_rate = None
        false_abstention_str = "N/A"

    # Behavioral alignment
    behavior_matched_count = sum(1 for c in cases if c["status_matches_behavior"])
    behavior_alignment_rate = round(behavior_matched_count / total_cases, 4)

    # Red flags
    forbidden_detected_count = sum(1 for c in cases if len(c.get("forbidden_claims_detected", [])) > 0)

    # Latencies
    retrieval_latencies = sorted([c["timings"]["retrieval_ms"] for c in cases if "timings" in c])
    generation_latencies = sorted([c["timings"]["generation_ms"] for c in cases if "timings" in c])
    total_latencies = sorted([c["timings"]["total_ms"] for c in cases if "timings" in c])

    def calc_percentiles(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"median_ms": None, "p95_ms": None}
        n = len(vals)
        med = vals[n // 2]
        p95_idx = min(int(n * 0.95), n - 1)
        p95 = vals[p95_idx]
        return {"median_ms": round(med, 1), "p95_ms": round(p95, 1)}

    # Manual review stats
    reviewed_count = sum(1 for c in cases if c.get("manual_review", {}).get("reviewed", False))
    pending_count = total_cases - reviewed_count

    return {
        "total_cases": total_cases,
        "api_failures": api_errors,
        "pipeline_errors": pipeline_errors,
        "status_distribution": status_counts,
        "retrieval": {
            "evaluated_cases": len(answerable_cases),
            "mean_source_recall": round(mean_source_recall, 4) if mean_source_recall is not None else None,
            "mean_source_recall_pct": source_recall_str,
            "perfect_source_recall_count": perfect_source_recall_count,
        },
        "citations": {
            "total_emitted": total_emitted,
            "total_valid": total_valid,
            "total_invalid": total_invalid,
            "validity_rate": overall_citation_validity,
            "validity_rate_pct": citation_validity_str,
        },
        "behavior": {
            "behavior_matched_count": behavior_matched_count,
            "behavior_alignment_rate": behavior_alignment_rate,
            "behavior_alignment_pct": f"{round(behavior_alignment_rate * 100, 1)}%",
            "unanswerable_cases": unanswerable_count,
            "correct_abstentions": correct_abstentions,
            "abstention_recall": abstention_recall_rate,
            "abstention_recall_pct": abstention_recall_str,
            "answerable_cases": fully_answerable_count,
            "false_abstentions": false_abstentions,
            "false_abstention_rate": false_abstention_rate,
            "false_abstention_pct": false_abstention_str,
            "false_answers": false_answers,
            "false_answer_rate": false_answer_rate,
            "false_answer_pct": false_answer_str,
            "forbidden_claims_detected_count": forbidden_detected_count,
        },
        "latency": {
            "sample_size": total_cases,
            "retrieval": calc_percentiles(retrieval_latencies),
            "generation": calc_percentiles(generation_latencies),
            "total": calc_percentiles(total_latencies),
            "note": "Latency percentiles (p95) are descriptive and sample-sensitive for N <= 20.",
        },
        "manual_review": {
            "reviewed_count": reviewed_count,
            "pending_count": pending_count,
            "status": "Pending manual review per evaluation/rubric.md" if pending_count > 0 else "Fully reviewed",
        },
    }


def make_mock_pipeline(
    mock_mode: str = "nominal",
    storage_dir: Optional[Path] = None,
) -> PipelineCallable:
    """Construct an offline pipeline runner that executes local retrieval but mocks generation.

    Args:
        mock_mode:
            'nominal': Returns realistic answered/abstain responses with real retrieved chunks.
            'api_error': Simulates an API network/timeout failure.
            'invalid_citations': Injects a fabricated citation ID.
    """
    def _mock_pipeline_call(question: str, top_k: int = 4) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Step 1: Real local retrieval (offline, 0 API calls)
        try:
            passages = retrieve(question=question, top_k=top_k, storage_dir=storage_dir)
        except Exception as e:
            passages = []

        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        if mock_mode == "api_error":
            return {
                "question": question,
                "answer": "I do not have sufficient information in the provided documentation to answer this question.",
                "status": "error",
                "missing_information": None,
                "citations": [],
                "invalid_citations": [],
                "citation_disclaimer": "Simulated API timeout error.",
                "retrieved_evidence": passages,
                "timings": {
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": 10.0,
                    "total_ms": round(retrieval_ms + 10.0, 2),
                },
                "model": "gpt-4o-mini-mocked",
                "error": "OpenAI API request timed out after 30 seconds (APITimeoutError).",
            }

        # Determine realistic mock behavior based on question text
        q_lower = question.lower()
        if "salary" in q_lower or "address" in q_lower or "hobby" in q_lower or "award" in q_lower or "weather" in q_lower:
            # Out of scope / injection questions
            return {
                "question": question,
                "answer": "I do not have sufficient information in the provided documentation to answer this question.",
                "status": "abstain",
                "missing_information": None,
                "citations": [],
                "invalid_citations": [],
                "citation_disclaimer": "Citations verified.",
                "retrieved_evidence": passages,
                "timings": {
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": 50.0,
                    "total_ms": round(retrieval_ms + 50.0, 2),
                },
                "model": "gpt-4o-mini-mocked",
                "error": None,
            }

        if "compare the latency" in q_lower:
            # Partial case (q18)
            resolved_cits = [
                {"chunk_id": passages[0]["chunk_id"], "source": passages[0]["source"], "page": passages[0].get("page")}
            ] if passages else []
            return {
                "question": question,
                "answer": "Project Aurora has a documented p99 latency under 18ms. Project Beacon has no documented latency benchmarks.",
                "status": "partial",
                "missing_information": "Project Beacon latency benchmarks are missing because it is in proposal stage.",
                "citations": resolved_cits,
                "invalid_citations": [],
                "citation_disclaimer": "Citations verified.",
                "retrieved_evidence": passages,
                "timings": {
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": 65.0,
                    "total_ms": round(retrieval_ms + 65.0, 2),
                },
                "model": "gpt-4o-mini-mocked",
                "error": None,
            }

        if "currently deployed to production" in q_lower:
            # Conflict case (q17)
            resolved_cits = [
                {"chunk_id": p["chunk_id"], "source": p["source"], "page": p.get("page")}
                for p in passages[:2]
            ]
            return {
                "question": question,
                "answer": "Sources conflict: The status report states Project Nexus is an internal staging pilot, while the newsletter claims it is fully deployed.",
                "status": "conflict",
                "missing_information": None,
                "citations": resolved_cits,
                "invalid_citations": [],
                "citation_disclaimer": "Citations verified.",
                "retrieved_evidence": passages,
                "timings": {
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": 70.0,
                    "total_ms": round(retrieval_ms + 70.0, 2),
                },
                "model": "gpt-4o-mini-mocked",
                "error": None,
            }

        # Nominal answerable case
        resolved_cits = [
            {"chunk_id": p["chunk_id"], "source": p["source"], "page": p.get("page")}
            for p in passages[:2]
        ]
        invalid_cits = ["chunk_fabricated_999"] if mock_mode == "invalid_citations" else []

        mock_ans = f"Based on the provided documentation, the requested information is documented. Evidence references: {passages[0]['chunk_id'] if passages else 'none'}."
        return {
            "question": question,
            "answer": mock_ans,
            "status": "answered",
            "missing_information": None,
            "citations": resolved_cits,
            "invalid_citations": invalid_cits,
            "citation_disclaimer": "Citations verified from storage.",
            "retrieved_evidence": passages,
            "timings": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": 60.0,
                "total_ms": round(retrieval_ms + 60.0, 2),
            },
            "model": "gpt-4o-mini-mocked",
            "error": None,
        }

    return _mock_pipeline_call


def run_evaluation(
    questions: List[Dict[str, Any]],
    pipeline_fn: PipelineCallable,
    split_filter: str = "all",
    top_k: int = DEFAULT_TOP_K,
    question_ids: Optional[List[str]] = None,
    storage_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute evaluation across questions filtered by split and optional question IDs."""
    env_meta = capture_environment_metadata(storage_dir)

    # Filter questions
    if split_filter in {"development", "dev"}:
        target_questions = [q for q in questions if q.get("split") == "development"]
    elif split_filter in {"holdout"}:
        target_questions = [q for q in questions if q.get("split") == "holdout"]
    else:
        target_questions = questions

    # Filter by specific question IDs if requested (strictly respecting split)
    if question_ids:
        qid_set = {qid.strip() for qid in question_ids if qid.strip()}
        target_questions = [q for q in target_questions if q.get("id") in qid_set]

    case_results: List[Dict[str, Any]] = []
    print(f"\n[INFO] Starting evaluation: {len(target_questions)} questions selected (split: {split_filter}, top_k: {top_k}).")

    for i, q in enumerate(target_questions, 1):
        q_id = q["id"]
        split = q["split"]
        cat = q["category"]
        print(f"  [{i:02d}/{len(target_questions):02d}] Evaluating {q_id} ({split} | {cat})...", end="", flush=True)

        case_res = evaluate_single_case(q, pipeline_fn, top_k=top_k)
        case_results.append(case_res)

        st = case_res["status"]
        recall = case_res["source_recall"]
        recall_display = f"{recall:.2f}" if recall is not None else "N/A"
        print(f" -> status: {st}, source_recall: {recall_display}")

    # Segregate results by split
    dev_cases = [c for c in case_results if c["split"] == "development"]
    holdout_cases = [c for c in case_results if c["split"] == "holdout"]

    dev_metrics = aggregate_metrics(dev_cases)
    holdout_metrics = aggregate_metrics(holdout_cases)
    combined_metrics = aggregate_metrics(case_results)

    return {
        "metadata": env_meta,
        "evaluation_settings": {
            "split_filter": split_filter,
            "top_k": top_k,
            "question_ids_filter": question_ids,
            "total_questions_run": len(target_questions),
        },
        "summary": {
            "development_split": dev_metrics,
            "holdout_split": holdout_metrics,
            "combined": combined_metrics,
        },
        "case_results": case_results,
    }


def format_markdown_report(eval_data: Dict[str, Any]) -> str:
    """Format evaluation results into a comprehensive Markdown report."""
    meta = eval_data.get("metadata", {})
    idx = meta.get("index_configuration", {})
    summary = eval_data.get("summary", {})
    dev_s = summary.get("development_split", {})
    hold_s = summary.get("holdout_split", {})
    comb_s = summary.get("combined", {})
    cases = eval_data.get("case_results", [])

    lines = []
    lines.append("# Evaluation Benchmark Report: Personal AI Knowledge Assistant")
    lines.append("")
    lines.append(f"**Execution Timestamp (UTC):** `{meta.get('timestamp')}`  ")
    lines.append(f"**Generation Model:** `{meta.get('generation_model')}`  ")
    lines.append(f"**Embedding Model:** `{idx.get('embedding_model')}` (dim: {idx.get('embedding_dimension')})  ")
    lines.append(f"**Indexed Chunks:** `{idx.get('collection_chunk_count')}` chunks (size: {idx.get('chunk_size')}, overlap: {idx.get('chunk_overlap')})  ")
    lines.append(f"**Platform / Python:** `{meta.get('platform')}` / Python `{meta.get('python_version')}`  ")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Manual Review Status:** Automated metrics track Source Recall, Citation Validity, and Behavioral Alignment.")
    lines.append("> **Factual correctness and semantic claim support require human inspection per `evaluation/rubric.md`.**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Split-by-Split Performance Summary")
    lines.append("")
    lines.append("| Metric | Development Split (14) | Holdout Split (6) | Combined Benchmark (20) |")
    lines.append("|---|:---:|:---:|:---:|")

    def get_split_cell(s_dict: Dict[str, Any], key: str) -> str:
        if s_dict.get("total_cases", 0) == 0:
            if key == "total_cases":
                return "0"
            return "N/A"
        if key == "total_cases":
            return str(s_dict.get("total_cases", 0))
        if key == "api_failures":
            return str(s_dict.get("api_failures", 0))
        if key == "pipeline_errors":
            return str(s_dict.get("pipeline_errors", 0))
        if key == "source_recall":
            return str(s_dict.get("retrieval", {}).get("mean_source_recall_pct", "N/A"))
        if key == "citation_validity":
            return str(s_dict.get("citations", {}).get("validity_rate_pct", "N/A"))
        if key == "abstention_recall":
            return str(s_dict.get("behavior", {}).get("abstention_recall_pct", "N/A"))
        if key == "false_abstention":
            return str(s_dict.get("behavior", {}).get("false_abstention_pct", "N/A"))
        if key == "behavior_alignment":
            return str(s_dict.get("behavior", {}).get("behavior_alignment_pct", "N/A"))
        if key == "forbidden_claims":
            return str(s_dict.get("behavior", {}).get("forbidden_claims_detected_count", 0))
        if key == "median_latency":
            med = s_dict.get("latency", {}).get("total", {}).get("median_ms")
            return f"{med} ms" if med is not None else "N/A"
        if key == "p95_latency":
            p95 = s_dict.get("latency", {}).get("total", {}).get("p95_ms")
            return f"{p95} ms" if p95 is not None else "N/A"
        if key == "manual_review":
            return str(s_dict.get("manual_review", {}).get("status", "Pending"))
        return "N/A"

    # Table rows
    lines.append(f"| **Evaluated Cases** | {get_split_cell(dev_s, 'total_cases')} | {get_split_cell(hold_s, 'total_cases')} | {get_split_cell(comb_s, 'total_cases')} |")
    lines.append(f"| **API Failures** | {get_split_cell(dev_s, 'api_failures')} | {get_split_cell(hold_s, 'api_failures')} | {get_split_cell(comb_s, 'api_failures')} |")
    lines.append(f"| **Pipeline Errors** | {get_split_cell(dev_s, 'pipeline_errors')} | {get_split_cell(hold_s, 'pipeline_errors')} | {get_split_cell(comb_s, 'pipeline_errors')} |")
    lines.append(f"| **Source Recall@k** | {get_split_cell(dev_s, 'source_recall')} | {get_split_cell(hold_s, 'source_recall')} | {get_split_cell(comb_s, 'source_recall')} |")
    lines.append(f"| **Citation Validity Rate** | {get_split_cell(dev_s, 'citation_validity')} | {get_split_cell(hold_s, 'citation_validity')} | {get_split_cell(comb_s, 'citation_validity')} |")
    lines.append(f"| **Abstention Recall** | {get_split_cell(dev_s, 'abstention_recall')} | {get_split_cell(hold_s, 'abstention_recall')} | {get_split_cell(comb_s, 'abstention_recall')} |")
    lines.append(f"| **False Abstention Rate** | {get_split_cell(dev_s, 'false_abstention')} | {get_split_cell(hold_s, 'false_abstention')} | {get_split_cell(comb_s, 'false_abstention')} |")
    lines.append(f"| **Behavior Alignment Rate** | {get_split_cell(dev_s, 'behavior_alignment')} | {get_split_cell(hold_s, 'behavior_alignment')} | {get_split_cell(comb_s, 'behavior_alignment')} |")
    lines.append(f"| **Forbidden Claims Flagged** | {get_split_cell(dev_s, 'forbidden_claims')} | {get_split_cell(hold_s, 'forbidden_claims')} | {get_split_cell(comb_s, 'forbidden_claims')} |")
    lines.append(f"| **Median Latency (Total)** | {get_split_cell(dev_s, 'median_latency')} | {get_split_cell(hold_s, 'median_latency')} | {get_split_cell(comb_s, 'median_latency')} |")
    lines.append(f"| **p95 Latency (Total)** | {get_split_cell(dev_s, 'p95_latency')} | {get_split_cell(hold_s, 'p95_latency')} | {get_split_cell(comb_s, 'p95_latency')} |")
    lines.append(f"| **Manual Review State** | {get_split_cell(dev_s, 'manual_review')} | {get_split_cell(hold_s, 'manual_review')} | {get_split_cell(comb_s, 'manual_review')} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Detailed Per-Case Evaluation Log")
    lines.append("")
    lines.append("| ID | Split | Category | Expected Behavior | Observed Status | Source Recall | Citations (Valid / Emitted) | Red Flags |")
    lines.append("|---|:---:|---|---|---|:---:|:---:|:---:|")

    for c in cases:
        q_id = c["question_id"]
        split = c["split"]
        cat = c["category"]
        exp_beh = c["expected_behavior"]
        st = c["status"]
        recall = f"{c['source_recall']:.2f}" if c["source_recall"] is not None else "N/A"
        cits = f"{c['citations_valid_count']} / {c['citations_emitted_count']}"
        flags = ", ".join(c.get("forbidden_claims_detected", [])) or "None"
        lines.append(f"| `{q_id}` | `{split}` | {cat} | `{exp_beh}` | `{st}` | {recall} | {cits} | {flags} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Methodological Notes & Limitations")
    lines.append("1. **Holdout Split Integrity**: Holdout questions (`q05`, `q09`, `q12`, `q15`, `q16`, `q20`) are measured to assess out-of-sample generalization. They must never be used to tune retrieval or prompt instructions.")
    lines.append("2. **Separation of Citation IDs from Factual Support**: An answer with 100% valid citation IDs may still misrepresent what the cited passage states. Ground truth correctness requires human scoring against `evaluation/rubric.md`.")
    lines.append("3. **Small Sample Latency Caution**: With N=20, p95 is sample-sensitive. Latency figures should be treated as descriptive estimates.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 Evaluation Runner for Personal AI Knowledge Assistant"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live evaluation using real OpenAI API requests (incurs API cost).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute full evaluation flow using offline mocked responses (0 API calls).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval against expected sources using local Chroma index (0 API calls).",
    )
    parser.add_argument(
        "--split",
        choices=["all", "development", "dev", "holdout"],
        default="all",
        help="Evaluation split to run: 'all' (20 questions), 'dev' (14 questions), or 'holdout' (6 questions).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve per question (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path override to benchmark questions.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path override for output JSON results.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Path override for output Markdown report.",
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        default=None,
        help="Optional comma-separated question IDs to evaluate (e.g. 'q04,q06,q11,q17,q18'). Respects split filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process (useful for rapid offline checks).",
    )

    args = parser.parse_args()

    questions = load_benchmark_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]

    parsed_qids = None
    if args.question_ids:
        parsed_qids = [qid.strip() for qid in args.question_ids.split(",") if qid.strip()]

    # Target question count based on split and optional question_ids
    if args.split in {"development", "dev"}:
        planned_questions = [q for q in questions if q.get("split") == "development"]
    elif args.split in {"holdout"}:
        planned_questions = [q for q in questions if q.get("split") == "holdout"]
    else:
        planned_questions = questions

    if parsed_qids:
        planned_questions = [q for q in planned_questions if q.get("id") in set(parsed_qids)]

    planned_count = len(planned_questions)

    # GUARD: Ensure user has explicitly passed --live or --dry-run or --retrieval-only
    if not args.live and not args.dry_run and not args.retrieval_only:
        print("=" * 70)
        print("PHASE 6 LIVE EVALUATION SAFETY GUARD")
        print("=" * 70)
        print(f"Target Benchmark Questions: {planned_count} questions (split: '{args.split}')")
        if parsed_qids:
            print(f"Target Question IDs:        {', '.join(parsed_qids)}")
        print(f"Target Generation Model:    {OPENAI_MODEL}")
        print(f"Estimated Token Volume:     ~{planned_count * 1200:,} tokens across prompt and completion")
        print()
        print("LIVE EVALUATION WAS NOT EXECUTED.")
        print("To prevent accidental paid API charges, live evaluation requires the explicit '--live' flag:")
        print(f"  python evaluation/evaluate.py --live --split {args.split}" + (f" --question-ids {args.question_ids}" if args.question_ids else ""))
        print()
        print("For zero-cost offline verification:")
        print("  1. Dry-Run Mode (offline mocked generation):")
        print(f"     python evaluation/evaluate.py --dry-run --split {args.split}" + (f" --question-ids {args.question_ids}" if args.question_ids else ""))
        print("  2. Retrieval-Only Mode (offline Chroma retrieval benchmark):")
        print(f"     python evaluation/evaluate.py --retrieval-only --split {args.split}" + (f" --question-ids {args.question_ids}" if args.question_ids else ""))
        print("=" * 70)
        sys.exit(0)

    # Select pipeline implementation
    if args.live:
        print("\n" + "!" * 70)
        print(f"WARNING: LIVE EVALUATION RUNNING — {planned_count} OpenAI API requests will be issued.")
        print(f"Model: {OPENAI_MODEL} | Split: {args.split}")
        if parsed_qids:
            print(f"Question IDs: {', '.join(parsed_qids)}")
        print("!" * 70 + "\n")
        pipeline_fn = lambda q, k: query_pipeline(q, top_k=k)
    elif args.retrieval_only:
        print("\n[INFO] Running RETRIEVAL-ONLY evaluation against persistent local Chroma index (0 API calls).")
        def _retrieval_only_pipeline(q: str, k: int) -> Dict[str, Any]:
            t0 = time.perf_counter()
            passages = retrieve(question=q, top_k=k)
            ret_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "question": q,
                "answer": "[RETRIEVAL-ONLY EVALUATION: Generation skipped]",
                "status": "answered" if passages else "abstain",
                "missing_information": None,
                "citations": [{"chunk_id": p["chunk_id"], "source": p["source"], "page": p.get("page")} for p in passages],
                "invalid_citations": [],
                "retrieved_evidence": passages,
                "timings": {"retrieval_ms": round(ret_ms, 2), "generation_ms": 0.0, "total_ms": round(ret_ms, 2)},
                "model": "retrieval-only-local",
                "error": None,
            }
        pipeline_fn = _retrieval_only_pipeline
    else:  # --dry-run
        print("\n[INFO] Running DRY-RUN evaluation using offline mocked pipeline (0 API calls).")
        pipeline_fn = make_mock_pipeline(mock_mode="nominal")

    # Run evaluation
    results = run_evaluation(
        questions=questions,
        pipeline_fn=pipeline_fn,
        split_filter=args.split,
        top_k=args.top_k,
        question_ids=parsed_qids,
    )

    # Generate Markdown report
    md_report = format_markdown_report(results)

    # Save results to disk
    results_dir = PATHS.evaluation_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = args.output or (results_dir / f"eval_results_{timestamp_slug}.json")
    md_path = args.markdown or (results_dir / f"eval_report_{timestamp_slug}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    # Display console output
    print("\n" + "=" * 70)
    print("EVALUATION RUN COMPLETE")
    print("=" * 70)
    print(f"JSON Results saved:     {json_path}")
    print(f"Markdown Report saved: {md_path}")
    print("\n" + md_report)


if __name__ == "__main__":
    main()
