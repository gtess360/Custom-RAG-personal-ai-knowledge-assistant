import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { checkHealth, queryAssistant, ApiError } from "../lib/api";
import { QueryResponse } from "../types/api";

describe("API Client (lib/api.ts)", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  describe("checkHealth", () => {
    it("returns health response on HTTP 200", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "ok", service: "personal-ai-knowledge-assistant" }),
      } as Response);

      const res = await checkHealth();
      expect(res.status).toBe("ok");
      expect(res.service).toBe("personal-ai-knowledge-assistant");
    });

    it("throws ApiError when backend returns non-ok status", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
      } as Response);

      await expect(checkHealth()).rejects.toThrow(ApiError);
    });

    it("throws ApiError with actionable message when network fails (unreachable backend)", async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error("Failed to fetch"));

      await expect(checkHealth()).rejects.toThrow(
        /Unable to reach backend service/
      );
    });
  });

  describe("queryAssistant", () => {
    const mockSuccessResponse: QueryResponse = {
      question: "What embedding model was used in Project Nexus?",
      answer: "The embedding model was all-MiniLM-L6-v2.",
      status: "answered",
      missing_information: null,
      citations: [
        { chunk_id: "project_nexus_p0_c000", source: "data/documents/project_nexus.md", page: null },
      ],
      invalid_citations: [],
      citation_disclaimer: "Citation validity indicates that cited chunk IDs exist.",
      retrieved_evidence: [
        {
          chunk_id: "project_nexus_p0_c000",
          text: "Project Nexus used all-MiniLM-L6-v2.",
          source: "data/documents/project_nexus.md",
          page: null,
          chunk_index: 0,
          distance: 0.354,
        },
      ],
      timings: { retrieval_ms: 120, generation_ms: 800, total_ms: 920 },
      model: "gpt-4o-mini",
      error: null,
    };

    it("rejects blank or whitespace-only inputs without calling fetch", async () => {
      const fetchSpy = vi.fn();
      global.fetch = fetchSpy;

      await expect(queryAssistant("")).rejects.toThrow("Please enter a question");
      await expect(queryAssistant("   ")).rejects.toThrow("Please enter a question");
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("clamps and validates top_k to integer between 1 and 10", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockSuccessResponse,
      } as Response);

      // top_k > 10 should be clamped to 10
      await queryAssistant("Valid question?", 15);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/query"),
        expect.objectContaining({
          body: JSON.stringify({ question: "Valid question?", top_k: 10 }),
        })
      );

      // top_k < 1 should be clamped to 1
      await queryAssistant("Valid question?", -2);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/query"),
        expect.objectContaining({
          body: JSON.stringify({ question: "Valid question?", top_k: 1 }),
        })
      );
    });

    it("parses valid QueryResponse on HTTP 200", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockSuccessResponse,
      } as Response);

      const res = await queryAssistant("What embedding model was used in Project Nexus?", 4);
      expect(res.status).toBe("answered");
      expect(res.citations.length).toBe(1);
      expect(res.retrieved_evidence[0].distance).toBe(0.354);
    });

    it("extracts error details when backend returns HTTP 422 or 500 with detail payload", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        json: async () => ({ detail: "Top-K must be between 1 and 10." }),
      } as unknown as Response);

      await expect(queryAssistant("Test question?", 4)).rejects.toThrow(
        "Top-K must be between 1 and 10."
      );
    });

    it("handles malformed JSON responses from backend gracefully", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        json: async () => {
          throw new Error("Invalid JSON");
        },
      } as unknown as Response);

      await expect(queryAssistant("Test question?", 4)).rejects.toThrow(
        "Backend returned HTTP 502"
      );
    });

    it("handles unreachable backend with actionable network error message", async () => {
      global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));

      await expect(queryAssistant("Test question?", 4)).rejects.toThrow(
        /Network error connecting to/
      );
    });
  });
});
