"""End-to-end RAG query pipeline for the Personal AI Knowledge Assistant.

Coordinates question retrieval and evidence-grounded answer generation:
1. Validates user question string.
2. Embeds query and retrieves ranked evidence chunks from persistent Chroma storage.
3. Formats candidate evidence as untrusted reference material.
4. Generates an evidence-grounded answer with citations or explicit abstention via OpenAI.
5. Resolves citations strictly to stored metadata (never trusting LLM source paths).
6. Captures separate retrieval and generation latency timings.
7. Keeps similarity metrics labeled strictly as 'distance' (never confidence or probability).
8. Returns a clean structured dictionary consumable by both CLI and Streamlit UI.
"""

from __future__ import annotations

import time
from typing import Optional
from pathlib import Path
from openai import OpenAI

from src.retrieval import retrieve, validate_query
from src.generation import (
    generate_answer,
    INSUFFICIENT_EVIDENCE_PHRASE,
    CITATION_VALIDITY_DISCLAIMER,
)
from src.config import OPENAI_MODEL, DEFAULT_TOP_K


def query_pipeline(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage_dir: Optional[Path] = None,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> dict:
    """Execute the full RAG pipeline: retrieval followed by evidence-grounded generation.

    Args:
        question: User query string.
        top_k: Number of evidence chunks to retrieve (default DEFAULT_TOP_K).
        storage_dir: Optional directory override for persistent Chroma index.
        client: Optional OpenAI client instance (e.g. for offline mocking).
        model: Optional OpenAI model identifier override.
        timeout: Finite network request timeout in seconds (default 30.0).
        max_retries: Maximum number of SDK retries on transient errors (default 2).

    Returns:
        Structured result dictionary:
            question: str
            answer: str
            status: "answered" | "abstain" | "partial" | "conflict" | "error"
            missing_information: Optional[str]
            citations: list[dict] (chunk_id, source, page resolved from storage)
            invalid_citations: list[str] (unrecognized IDs filtered out)
            citation_disclaimer: str
            retrieved_evidence: list[dict] with distance, text, source, page
            timings: dict with retrieval_ms, generation_ms, total_ms
            model: str
            error: Optional[str]
    """
    t0 = time.perf_counter()
    target_model = (model or OPENAI_MODEL).strip()

    # 1. Validate query string
    try:
        clean_question = validate_query(question)
    except ValueError as e:
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "question": question,
            "answer": INSUFFICIENT_EVIDENCE_PHRASE,
            "status": "error",
            "missing_information": None,
            "citations": [],
            "invalid_citations": [],
            "citation_disclaimer": CITATION_VALIDITY_DISCLAIMER,
            "retrieved_evidence": [],
            "timings": {
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
                "total_ms": round(total_ms, 2),
            },
            "model": target_model,
            "error": str(e),
        }

    # 2. Retrieval step
    t_retrieval_start = time.perf_counter()
    try:
        passages = retrieve(
            question=clean_question,
            top_k=top_k,
            storage_dir=storage_dir,
        )
    except Exception as e:
        retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "question": clean_question,
            "answer": INSUFFICIENT_EVIDENCE_PHRASE,
            "status": "error",
            "missing_information": None,
            "citations": [],
            "invalid_citations": [],
            "citation_disclaimer": CITATION_VALIDITY_DISCLAIMER,
            "retrieved_evidence": [],
            "timings": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": 0.0,
                "total_ms": round(total_ms, 2),
            },
            "model": target_model,
            "error": f"Retrieval failed: {str(e)}",
        }
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    # 3. Generation step
    t_gen_start = time.perf_counter()
    gen_result = generate_answer(
        question=clean_question,
        passages=passages,
        client=client,
        model=target_model,
        timeout=timeout,
        max_retries=max_retries,
    )
    generation_ms = (time.perf_counter() - t_gen_start) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Format evidence with explicit distance labeling (never confidence)
    formatted_evidence = [
        {
            "chunk_id": p.get("chunk_id", ""),
            "text": p.get("text", ""),
            "source": p.get("source", ""),
            "page": p.get("page"),
            "chunk_index": p.get("chunk_index"),
            "distance": p.get("distance"),  # Cosine distance: smaller means more similar
        }
        for p in passages
    ]

    return {
        "question": clean_question,
        "answer": gen_result["answer"],
        "status": gen_result["status"],
        "missing_information": gen_result.get("missing_information"),
        "citations": gen_result["citations"],
        "invalid_citations": gen_result["invalid_citations"],
        "citation_disclaimer": CITATION_VALIDITY_DISCLAIMER,
        "retrieved_evidence": formatted_evidence,
        "timings": {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "model": gen_result["model"],
        "error": gen_result.get("error"),
    }


def main() -> None:
    """CLI runner for testing the RAG pipeline on a question."""
    import sys

    default_question = "What embedding model and vector database were used in Project Nexus?"
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else default_question

    print("=" * 70)
    print("PERSONAL AI KNOWLEDGE ASSISTANT — PIPELINE QUERY")
    print("=" * 70)
    print(f"Question: {question}\n")

    result = query_pipeline(question)

    print(f"Status: {result['status']}")
    print(f"Model:  {result['model']}")
    print(
        f"Timings: Retrieval={result['timings']['retrieval_ms']}ms | "
        f"Generation={result['timings']['generation_ms']}ms | "
        f"Total={result['timings']['total_ms']}ms\n"
    )
    print("ANSWER:")
    print(result["answer"])
    print()

    if result.get("missing_information"):
        print(f"Missing Information: {result['missing_information']}\n")

    if result["citations"]:
        print("CITATIONS (Resolved from Storage Metadata):")
        for cit in result["citations"]:
            page_str = f", Page {cit['page']}" if cit.get("page") else ""
            print(f"  - [{cit['chunk_id']}] -> Source: {cit['source']}{page_str}")
        print()

    if result["invalid_citations"]:
        print(f"INVALID / UNRESOLVED CITATIONS: {result['invalid_citations']}\n")

    print("RETRIEVED EVIDENCE:")
    for i, ev in enumerate(result["retrieved_evidence"], 1):
        page_str = f", Page {ev['page']}" if ev.get("page") else ""
        dist_str = f"{ev['distance']:.4f}" if ev.get("distance") is not None else "N/A"
        print(f"  {i}. [{ev['chunk_id']}] (Distance: {dist_str}, Source: {ev['source']}{page_str})")
        print(f"     Excerpt: {ev['text'][:120]}...\n")

    if result.get("error"):
        print(f"ERROR: {result['error']}\n")

    print(f"NOTE: {result['citation_disclaimer']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
