import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import Home from "../app/page";
import * as apiModule from "../lib/api";
import { QueryResponse } from "../types/api";

describe("Frontend UI Application (app/page.tsx)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    // Default mock health check
    vi.spyOn(apiModule, "checkHealth").mockResolvedValue({
      status: "ok",
      service: "personal-ai-knowledge-assistant",
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const baseResponse: QueryResponse = {
    question: "What embedding model was used in Project Nexus?",
    answer: "Project Nexus utilized sentence-transformers/all-MiniLM-L6-v2 [nexus_chunk_0].",
    status: "answered",
    missing_information: null,
    citations: [
      { chunk_id: "nexus_chunk_0", source: "data/documents/project_nexus.md", page: 1 },
    ],
    invalid_citations: [],
    citation_disclaimer:
      "Citation validity indicates that cited chunk IDs exist in the retrieved evidence.",
    retrieved_evidence: [
      {
        chunk_id: "nexus_chunk_0",
        text: "Project Nexus specifications: all-MiniLM-L6-v2 model with ChromaDB persistent vector storage.",
        source: "data/documents/project_nexus.md",
        page: 1,
        chunk_index: 0,
        distance: 0.4125,
      },
      {
        chunk_id: "nexus_chunk_1",
        text: "Project Nexus deployment status report.",
        source: "data/documents/project_nexus.md",
        page: 2,
        chunk_index: 1,
        distance: 0.7289,
      },
    ],
    timings: { retrieval_ms: 150, generation_ms: 850, total_ms: 1000 },
    model: "gpt-4o-mini",
    error: null,
  };

  it("renders initial clean state with hero title and search input", () => {
    render(<Home />);
    expect(screen.getByTestId("hero-title")).toHaveTextContent(
      "Evidence-Grounded Knowledge Assistant"
    );
    expect(screen.getByTestId("search-input")).toBeInTheDocument();
    expect(screen.getByTestId("empty-placeholder")).toBeInTheDocument();
  });

  it("validates blank or whitespace-only queries and displays warning without calling API", async () => {
    const querySpy = vi.spyOn(apiModule, "queryAssistant");
    render(<Home />);

    const submitBtn = screen.getByTestId("submit-btn");
    // Initially empty input, button is disabled
    expect(submitBtn).toBeDisabled();

    // Type spaces
    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "    " } });

    // Submit form directly
    fireEvent.submit(input.closest("form")!);

    expect(screen.getByTestId("validation-error")).toHaveTextContent(
      "Please enter a question to search and answer."
    );
    expect(querySpy).not.toHaveBeenCalled();
  });

  it("updates top_k via slider and passes to queryAssistant", async () => {
    const querySpy = vi.spyOn(apiModule, "queryAssistant").mockResolvedValue(baseResponse);
    render(<Home />);

    const slider = screen.getByTestId("top-k-slider");
    fireEvent.change(slider, { target: { value: "8" } });

    expect(screen.getByTestId("top-k-badge")).toHaveTextContent("8");

    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Tell me about Nexus" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(querySpy).toHaveBeenCalledWith("Tell me about Nexus", 8);
    });
  });

  it("prevents duplicate submissions and displays loading indicator while query is pending", async () => {
    let resolveQuery: (res: QueryResponse) => void;
    const queryPromise = new Promise<QueryResponse>((resolve) => {
      resolveQuery = resolve;
    });
    const querySpy = vi.spyOn(apiModule, "queryAssistant").mockReturnValue(queryPromise);

    render(<Home />);

    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Pending query?" } });

    const submitBtn = screen.getByTestId("submit-btn");
    fireEvent.click(submitBtn);

    // Button should now be disabled and loading indicator visible
    expect(screen.getByTestId("loading-indicator")).toBeInTheDocument();
    expect(submitBtn).toBeDisabled();

    // Secondary click attempt should NOT trigger another call
    fireEvent.click(submitBtn);
    expect(querySpy).toHaveBeenCalledTimes(1);

    // Resolve the promise
    resolveQuery!(baseResponse);

    await waitFor(() => {
      expect(screen.getByTestId("answer-card")).toBeInTheDocument();
    });
  });

  it("renders answered state with accurate wording and raw cosine distance", async () => {
    vi.spyOn(apiModule, "queryAssistant").mockResolvedValue(baseResponse);
    render(<Home />);

    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "What embedding model was used?" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("status-pill")).toHaveTextContent("Answered");
    });

    // Check accurate citations wording (no "verified" claims)
    expect(screen.getByTestId("citations-section")).toHaveTextContent(
      "CITATIONS MATCHED TO RETRIEVED EVIDENCE:"
    );

    // Check raw cosine distance & disclaimer
    expect(screen.getByTestId("distance-disclaimer")).toHaveTextContent(
      "Lower distance means a closer vector match; it does not establish answer correctness."
    );
    expect(screen.getByTestId("dist-nexus_chunk_0")).toHaveTextContent("Cosine Distance: 0.4125");
    expect(screen.getByTestId("dist-nexus_chunk_1")).toHaveTextContent("Cosine Distance: 0.7289");

    // Ensure no percentage-style distance meters or arbitrary labels
    expect(screen.queryByText(/Strong Match/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Moderate Match/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Weak Match/i)).not.toBeInTheDocument();

    // Check disclaimer note
    expect(screen.getByTestId("citation-disclaimer")).toHaveTextContent(
      "Citation-ID validation does not prove factual support."
    );
  });

  it("highlights corresponding evidence card when citation button is clicked", async () => {
    vi.spyOn(apiModule, "queryAssistant").mockResolvedValue(baseResponse);
    render(<Home />);

    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Citation click test" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("citation-btn-nexus_chunk_0")).toBeInTheDocument();
    });

    const card = screen.getByTestId("evidence-card-nexus_chunk_0");
    expect(card).not.toHaveClass("highlighted");

    // Click citation button
    fireEvent.click(screen.getByTestId("citation-btn-nexus_chunk_0"));

    expect(card).toHaveClass("highlighted");
  });

  it("renders abstain state properly when evidence is insufficient", async () => {
    const abstainResponse: QueryResponse = {
      ...baseResponse,
      status: "abstain",
      answer: "I do not have sufficient information in the provided documents to answer this question.",
      missing_information: "The documents contain no information regarding Mars rovers.",
      citations: [],
    };
    vi.spyOn(apiModule, "queryAssistant").mockResolvedValue(abstainResponse);

    render(<Home />);
    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Tell me about Mars rovers" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("status-pill")).toHaveTextContent(
        "Abstained (Insufficient Evidence)"
      );
    });

    expect(screen.getByTestId("missing-info-box")).toHaveTextContent(
      "The documents contain no information regarding Mars rovers."
    );
  });

  it("renders partial state with missing information banner", async () => {
    const partialResponse: QueryResponse = {
      ...baseResponse,
      status: "partial",
      answer: "Project Nexus handled 1,200 support queries [nexus_chunk_0].",
      missing_information: "Quarterly budget allocations were not documented in the excerpts.",
    };
    vi.spyOn(apiModule, "queryAssistant").mockResolvedValue(partialResponse);

    render(<Home />);
    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "What were the query volume and budget?" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("status-pill")).toHaveTextContent("Partial Answer");
    });
    expect(screen.getByTestId("missing-info-box")).toHaveTextContent(
      "Quarterly budget allocations were not documented in the excerpts."
    );
  });

  it("renders conflict state when contradictory documents are detected", async () => {
    const conflictResponse: QueryResponse = {
      ...baseResponse,
      status: "conflict",
      answer: "Section A states pilot staging only, whereas Section B asserts full deployment.",
    };
    vi.spyOn(apiModule, "queryAssistant").mockResolvedValue(conflictResponse);

    render(<Home />);
    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Is Nexus deployed to production?" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("status-pill")).toHaveTextContent(
        "Contradiction / Conflict Detected"
      );
    });
  });

  it("renders error state with pipeline error details", async () => {
    const errorResponse: QueryResponse = {
      ...baseResponse,
      status: "error",
      answer: "An error occurred during retrieval.",
      error: "Embedding service temporarily unavailable.",
      citations: [],
    };
    vi.spyOn(apiModule, "queryAssistant").mockResolvedValue(errorResponse);

    render(<Home />);
    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Trigger error" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("status-pill")).toHaveTextContent("Pipeline Error");
    });
    expect(screen.getByTestId("pipeline-error-detail")).toHaveTextContent(
      "Embedding service temporarily unavailable."
    );
  });

  it("handles unreachable backend network errors and retries ONLY on explicit user action", async () => {
    const querySpy = vi
      .spyOn(apiModule, "queryAssistant")
      .mockRejectedValueOnce(
        new apiModule.ApiError("Network error connecting to http://localhost:8000.", 0)
      )
      .mockResolvedValueOnce(baseResponse);

    render(<Home />);
    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Retry test question" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("error-banner")).toBeInTheDocument();
    });
    expect(screen.getByTestId("error-message")).toHaveTextContent(
      "Network error connecting to http://localhost:8000."
    );

    // Ensure it did NOT automatically retry
    expect(querySpy).toHaveBeenCalledTimes(1);

    // Retries strictly through explicit user action on the Retry button
    const retryBtn = screen.getByTestId("retry-button");
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(querySpy).toHaveBeenCalledTimes(2);
      expect(screen.getByTestId("answer-card")).toBeInTheDocument();
    });
  });

  it("safely renders model and document text containing HTML/script tags without injection", async () => {
    const maliciousResponse: QueryResponse = {
      ...baseResponse,
      answer: "Safe answer containing <script>alert('xss')</script> and <b>bold text</b>.",
      retrieved_evidence: [
        {
          chunk_id: "xss_chunk",
          text: "<img src='x' onerror='alert(1)'> document content",
          source: "data/<test>.md",
          distance: 0.1234,
        },
      ],
    };
    vi.spyOn(apiModule, "queryAssistant").mockResolvedValue(maliciousResponse);

    render(<Home />);
    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Security test" } });
    fireEvent.click(screen.getByTestId("submit-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("answer-card")).toBeInTheDocument();
    });

    // The script tag should be rendered safely as text content, NOT as an executable script element
    expect(document.querySelector("script[src='x']")).toBeNull();
    const evidenceText = screen.getByTestId("evidence-text-xss_chunk");
    expect(evidenceText.textContent).toContain("<img src='x' onerror='alert(1)'>");
  });
});
