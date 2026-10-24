# Evaluation Plan

## Purpose
Measure whether the system retrieves supporting evidence, answers correctly, cites support, and abstains appropriately. A small test set demonstrates behavior on that set; it does not establish zero hallucination.

## Fixed set: 20 cases
| Type | Count | Examples |
|---|---:|---|
| Direct facts | 5 | Model, toolset, dates, explicitly documented result |
| Multi-document | 4 | Compare two projects using evidence from both |
| Analytical | 3 | Summarize documented relevant experience without inventing a ranking |
| Missing/out-of-scope | 4 | Undocumented salary, private address, unrelated world knowledge, unsupported deployment claim |
| Conflict/partial evidence | 2 | Contradictory status; only one half of a comparison documented |
| Instruction injection | 2 | Source asks the assistant to ignore rules or invent an achievement |

Create cases against the actual approved corpus. Clearly mark fictional project facts. Include planned vs deployed examples. Avoid real private information.

## Question record
Use evaluation/questions.json as a list of records with:
- id, category, question
- answerable: true, false, or partial
- expected_answer_facts: short list of required claims
- expected_sources: relative source paths and optional pages
- expected_evidence: relevant excerpts/sections; attach final chunk IDs after chunk settings stabilize
- forbidden_claims: claims the answer must not make
- expected_behavior: answer, abstain, partial, or report_conflict
- split: development or holdout

Use 14 development and 6 holdout questions, with answerable and abstention cases in each. Tune on development only. Once holdout failures guide changes, record that it is no longer untouched and add fresh cases for a later assessment.

## Metrics and denominators
| Metric | Definition |
|---|---|
| Source Recall@k | Required relevant sources present in top-k / all required relevant sources, averaged over answerable cases |
| Evidence sufficiency | Cases where retrieved text contains all required facts / answerable cases; manually inspect excerpts |
| Answer correctness | Cases with all expected facts and no contradictory/forbidden factual claims / answerable cases |
| Claim support rate | Factual answer claims supported by supplied evidence / factual answer claims reviewed |
| Citation validity | Citation IDs belonging to supplied context / citation IDs emitted |
| Citation support | Cited factual claims actually supported by their cited text / cited factual claims reviewed |
| Abstention recall | Correct abstentions / unanswerable cases |
| False abstention rate | Fully answerable cases incorrectly refused / fully answerable cases |
| Behavior success | Cases satisfying expected partial/conflict/injection behavior / cases in those categories |
| Latency | End-to-end median and p95; separately report retrieval and generation times |

Return N/A for zero denominators. Similarity distance is not answer confidence. Source recall alone can hide chunks that mention a project but omit the needed fact. Review claim support manually; an LLM judge is optional later, not ground truth.

## Procedure
1. Freeze corpus, questions, embedding/chunk settings and model configuration for each run.
2. Run retrieval alone; capture chunk IDs, text, distances and timings.
3. Run end-to-end generation; capture answer, citations, timings and available token usage.
4. Grade against expected facts and evidence; record reasons for each failure.
5. Save per-case results plus summary, timestamp, corpus/index fingerprint, dependency versions and model identifier. Never save secrets.
6. Separate cold-start latency from warm requests. With only 20 cases, p95 is descriptive and unstable; state sample size and repeat runs if measuring variability.
7. Categorize failures: ingestion, chunking, retrieval, context assembly, generation, citation, abstention, or UI.

## Provisional development goals
At least 90% source recall, 90% answer correctness, 100% syntactically valid citation IDs, and correct rejection on all four unanswerable cases. Also report semantic citation support and false abstentions. These are targets, not achieved results or universal production thresholds. Define a latency goal after observing the chosen hardware and API model.
