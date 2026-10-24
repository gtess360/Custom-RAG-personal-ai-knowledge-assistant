export interface CitationItem {
  chunk_id: string;
  source: string;
  page?: number | null;
}

export interface EvidenceItem {
  chunk_id: string;
  text: string;
  source: string;
  page?: number | null;
  chunk_index?: number | null;
  distance?: number | null;
}

export interface TimingsItem {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export type QueryStatus = "answered" | "abstain" | "partial" | "conflict" | "error";

export interface QueryResponse {
  question: string;
  answer: string;
  status: QueryStatus;
  missing_information?: string | null;
  citations: CitationItem[];
  invalid_citations: string[];
  citation_disclaimer: string;
  retrieved_evidence: EvidenceItem[];
  timings: TimingsItem;
  model: string;
  error?: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
}
