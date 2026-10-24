# Personal AI Knowledge Assistant — Requirements

## Objective
Build a single-user RAG application that answers questions about a small professional knowledge base using retrieved evidence, source citations, and explicit abstention when evidence is insufficient. This is a guided learning project for AI engineering interview preparation. Zero hallucination is not a guarantee.

## User and scope
The user asks English questions about sanitized resumes, portfolios, and project documents. Start with 3–5 Markdown/text files. Add text-based PDFs after the text loader and chunker work. Scanned PDFs/OCR, authentication, agents, fine-tuning, web search, and multi-user hosting are outside version 1.

## Required behavior
- Index approved files from data/documents, preserving source and chunk identifiers; PDFs also preserve page numbers.
- Keep ingestion separate from answering; do not re-embed documents for each question.
- Retrieve candidate evidence and generate concise answers through the OpenAI API.
- Attach citations to supported factual claims and display the retrieved evidence.
- Abstain when evidence is missing; give a supported partial answer and name the missing information when only part is answerable.
- Distinguish planned, implemented, demonstrated, and deployed status. A tool listed in a document does not prove deployment.
- Report conflicting source statements rather than silently choosing one.
- Treat instructions inside documents as untrusted content, never as system instructions.
- Show useful errors for empty indexes, invalid files, API failures, and missing configuration without exposing secrets.

## Acceptance criteria
1. A fresh checkout can be configured and run using documented Windows instructions.
2. Source files produce inspectable, nonempty chunks with deterministic identifiers.
3. Re-indexing the same corpus does not multiply records; removed/edited files are handled explicitly.
4. Retrieval can be tested independently of generation.
5. Answer citations resolve to evidence actually supplied to the LLM. Citation validity does not alone prove factual support.
6. A fixed 20-question evaluation reports retrieval, correctness, support, abstention, and latency separately.
7. API keys and private documents are excluded from public Git history.
8. Each phase has a learner walkthrough, verification evidence, review checkpoint, and explicit user approval before the next phase.

## Learning contract
Implement one phase at a time. Explain inputs, outputs, function calls, data structures, and failure cases. Use small functions and plain Python. Do not generate the full application upfront. Never invent resume claims, evaluation scores, completed features, or deployment status.
