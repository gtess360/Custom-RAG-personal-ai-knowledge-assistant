# Project Nexus: Enterprise Knowledge Assistant (Sanitized Sample)

> **Notice**: Fictional project portfolio document for the Personal AI Knowledge Assistant.

## Project Overview
Project Nexus is an internal Retrieval-Augmented Generation (RAG) assistant designed to help customer support agents query technical documentation, policy manuals, and release notes rapidly.

## System Architecture & Technologies
- **Ingestion Pipeline**: Python parsing service that converts markdown, HTML, and text-based PDFs into chunks with deterministic IDs.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` running locally on CPU.
- **Vector Store**: ChromaDB running in persistent local mode.
- **LLM Generation**: OpenAI GPT-4o-mini using strict evidence-bounded system prompts.
- **User Interface**: Streamlit dashboard with expandable evidence panels and source attribution.

## Implementation & Deployment Status (Documented Conflict)

### Section A — Central Engineering Status Report (November 2024)
- **Status: DEMONSTRATED (Internal Staging Pilot only)**.
- Piloted with 200 technical support agents across 3 regional teams during Q4 2024.
- Handled an average of 1,200 support queries per day during the trial.
- Explicit note from lead engineer: **"Project Nexus has NOT been approved or deployed to public production. It remains an internal demonstration prototype pending security review."**

### Section B — Regional Operations Newsletter (January 2025)
- An internal newsletter article states: **"Project Nexus is now fully deployed across all global operations and serves as the official enterprise support engine."**
- Note: This statement contradicts the Central Engineering Status Report in Section A.
