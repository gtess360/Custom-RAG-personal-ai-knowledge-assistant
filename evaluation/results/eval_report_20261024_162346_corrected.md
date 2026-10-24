# Evaluation Benchmark Report: Personal AI Knowledge Assistant

**Execution Timestamp (UTC):** `2026-10-24T20:23:26.247220+00:00`  
**Generation Model:** `gpt-4o-mini`  
**Embedding Model:** `all-MiniLM-L6-v2` (dim: 384)  
**Indexed Chunks:** `10` chunks (size: 120, overlap: 25)  
**Platform / Python:** `Windows-10-10.0.19045-SP0` / Python `3.12.10`  

> [!NOTE]
> **Manual Review Status:** Automated metrics track Source Recall, Citation Validity, and Behavioral Alignment.
> **Factual correctness and semantic claim support require human inspection per `evaluation/rubric.md`.**

---

## 1. Split-by-Split Performance Summary

| Metric | Development Split (14) | Holdout Split (6) | Combined Benchmark (20) |
|---|:---:|:---:|:---:|
| **Evaluated Cases** | 5 | 0 | 5 |
| **API Failures** | 0 | N/A | 0 |
| **Pipeline Errors** | 0 | N/A | 0 |
| **Source Recall@k** | 100.0% | N/A | 100.0% |
| **Citation Validity Rate** | 100.0% | N/A | 100.0% |
| **Abstention Recall** | N/A | N/A | N/A |
| **False Abstention Rate** | 0.0% | N/A | 0.0% |
| **Behavior Alignment Rate** | 80.0% | N/A | 80.0% |
| **Forbidden Claims Flagged** | 0 | N/A | 0 |
| **Median Latency (Total)** | 1768.0 ms | N/A | 1768.0 ms |
| **p95 Latency (Total)** | 12493.8 ms | N/A | 12493.8 ms |
| **Manual Review State** | Pending manual review per evaluation/rubric.md | N/A | Pending manual review per evaluation/rubric.md |

---

## 2. Detailed Per-Case Evaluation Log

| ID | Split | Category | Expected Behavior | Observed Status | Source Recall | Citations (Valid / Emitted) | Red Flags |
|---|:---:|---|---|---|:---:|:---:|:---:|
| `q04` | `development` | direct_facts | `answer` | `answered` | 1.00 | 1 / 1 | None |
| `q06` | `development` | multi_document | `answer` | `partial` | 1.00 | 4 / 4 | None |
| `q11` | `development` | analytical | `answer` | `answered` | 1.00 | 1 / 1 | None |
| `q17` | `development` | conflict_partial | `report_conflict` | `conflict` | 1.00 | 2 / 2 | None |
| `q18` | `development` | conflict_partial | `partial` | `partial` | 1.00 | 2 / 2 | None |

---

## 3. Methodological Notes & Limitations
1. **Holdout Split Integrity**: Holdout questions (`q05`, `q09`, `q12`, `q15`, `q16`, `q20`) are measured to assess out-of-sample generalization. They must never be used to tune retrieval or prompt instructions.
2. **Separation of Citation IDs from Factual Support**: An answer with 100% valid citation IDs may still misrepresent what the cited passage states. Ground truth correctness requires human scoring against `evaluation/rubric.md`.
3. **Small Sample Latency Caution**: With N=20, p95 is sample-sensitive. Latency figures should be treated as descriptive estimates.
