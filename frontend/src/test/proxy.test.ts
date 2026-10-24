import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { NextRequest } from "next/server";
import { POST } from "../app/api/query/route";

describe("Next.js Server Proxy (app/api/query/route.ts)", () => {
  const originalEnv = process.env;
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
    global.fetch = originalFetch;
  });

  const createRequest = (
    body: Record<string, unknown>,
    headers: Record<string, string> = {}
  ): NextRequest => {
    return new NextRequest("http://localhost:3000/api/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...headers,
      },
      body: JSON.stringify(body),
    });
  };

  const mockBackendSuccess = {
    question: "What is Project Aurora?",
    answer: "Project Aurora is a real-time detection system.",
    status: "answered",
    citations: [{ chunk_id: "aurora_00", source: "project_aurora.md", page: 1 }],
    invalid_citations: [],
    citation_disclaimer: "Valid citations",
    retrieved_evidence: [],
    timings: { retrieval_ms: 10, generation_ms: 200, total_ms: 210 },
    model: "gpt-4o-mini",
    error: null,
  };

  // ---------------------------------------------------------------------------
  // 1. Authenticated Reviewer Access
  // ---------------------------------------------------------------------------
  describe("Reviewer Access Authentication", () => {
    it("returns 401 when REVIEWER_ACCESS_TOKEN is configured but request lacks token", async () => {
      process.env.APP_ENV = "development";
      process.env.REVIEWER_ACCESS_TOKEN = "secret-reviewer-token";
      process.env.BACKEND_API_KEY = "backend-secret";

      const req = createRequest({ question: "Valid question?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(401);
      const json = await res.json();
      expect(json.detail).toContain("Unauthorized reviewer access");
    });

    it("returns 401 when reviewer token does not match REVIEWER_ACCESS_TOKEN", async () => {
      process.env.APP_ENV = "development";
      process.env.REVIEWER_ACCESS_TOKEN = "secret-reviewer-token";

      const req = createRequest(
        { question: "Valid question?", top_k: 4 },
        { Authorization: "Bearer wrong-token" }
      );
      const res = await POST(req);

      expect(res.status).toBe(401);
      const json = await res.json();
      expect(json.detail).toContain("Unauthorized reviewer access");
    });

    it("accepts valid reviewer token via Bearer authorization and proxies to backend", async () => {
      process.env.APP_ENV = "development";
      process.env.REVIEWER_ACCESS_TOKEN = "secret-reviewer-token";
      process.env.BACKEND_API_KEY = "backend-key-123";
      process.env.BACKEND_URL = "http://mock-backend:8000";

      const fetchSpy = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockBackendSuccess,
      } as Response);
      global.fetch = fetchSpy;

      const req = createRequest(
        { question: "What is Project Aurora?", top_k: 4 },
        { Authorization: "Bearer secret-reviewer-token" }
      );
      const res = await POST(req);

      expect(res.status).toBe(200);
      const data = await res.json();
      expect(data.status).toBe("answered");

      // Verify proxy sent backend key to backend, NOT the reviewer token
      expect(fetchSpy).toHaveBeenCalledWith(
        "http://mock-backend:8000/query",
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: "Bearer backend-key-123",
          }),
        })
      );
    });

    it("accepts valid reviewer token via x-reviewer-token custom header", async () => {
      process.env.APP_ENV = "development";
      process.env.REVIEWER_ACCESS_TOKEN = "secret-reviewer-token";
      process.env.BACKEND_API_KEY = "backend-key-123";

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockBackendSuccess,
      } as Response);

      const req = createRequest(
        { question: "What is Project Aurora?", top_k: 4 },
        { "x-reviewer-token": "secret-reviewer-token" }
      );
      const res = await POST(req);

      expect(res.status).toBe(200);
      const data = await res.json();
      expect(data.status).toBe("answered");
    });
  });

  // ---------------------------------------------------------------------------
  // 2. Production Configuration Enforcement
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // 2. Production Configuration Enforcement (Fail Closed)
  // ---------------------------------------------------------------------------
  describe("Production Configuration (Fail Closed)", () => {
    it("returns 500 when NODE_ENV=production and BACKEND_URL is missing", async () => {
      (process.env as Record<string, string | undefined>).NODE_ENV = "production";
      delete process.env.APP_ENV;
      process.env.BACKEND_URL = "";
      process.env.BACKEND_API_KEY = "backend-key-present";
      process.env.REVIEWER_ACCESS_TOKEN = "reviewer-token-set";

      const fetchSpy = vi.fn();
      global.fetch = fetchSpy;

      const req = createRequest({ question: "Valid question?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(500);
      const json = await res.json();
      expect(json.detail).toContain("BACKEND_URL must be configured");
      // Safe configuration error returned before forwarding query
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("returns 500 when NODE_ENV=production and BACKEND_API_KEY is missing", async () => {
      (process.env as Record<string, string | undefined>).NODE_ENV = "production";
      delete process.env.APP_ENV;
      process.env.BACKEND_URL = "http://internal-backend:8000";
      process.env.BACKEND_API_KEY = "";
      process.env.REVIEWER_ACCESS_TOKEN = "reviewer-token-set";

      const fetchSpy = vi.fn();
      global.fetch = fetchSpy;

      const req = createRequest({ question: "Valid question?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(500);
      const json = await res.json();
      expect(json.detail).toContain("BACKEND_API_KEY must be configured");
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("returns 500 when NODE_ENV=production and REVIEWER_ACCESS_TOKEN is missing", async () => {
      (process.env as Record<string, string | undefined>).NODE_ENV = "production";
      delete process.env.APP_ENV;
      process.env.BACKEND_URL = "http://internal-backend:8000";
      process.env.BACKEND_API_KEY = "backend-key-present";
      process.env.REVIEWER_ACCESS_TOKEN = "";

      const fetchSpy = vi.fn();
      global.fetch = fetchSpy;

      const req = createRequest({ question: "Valid question?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(500);
      const json = await res.json();
      expect(json.detail).toContain("REVIEWER_ACCESS_TOKEN must be configured");
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("fails closed: requires production settings when NODE_ENV=production even if APP_ENV is mistakenly set to development", async () => {
      (process.env as Record<string, string | undefined>).NODE_ENV = "production";
      process.env.APP_ENV = "development"; // Mistaken setting in prod environment
      process.env.BACKEND_URL = "http://internal-backend:8000";
      process.env.BACKEND_API_KEY = ""; // Missing
      process.env.REVIEWER_ACCESS_TOKEN = "reviewer-token-set";

      const fetchSpy = vi.fn();
      global.fetch = fetchSpy;

      const req = createRequest({ question: "Valid question?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(500);
      const json = await res.json();
      expect(json.detail).toContain("BACKEND_API_KEY must be configured");
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("fails closed: requires reviewer token when NODE_ENV=production even if APP_ENV=development", async () => {
      (process.env as Record<string, string | undefined>).NODE_ENV = "production";
      process.env.APP_ENV = "development";
      process.env.BACKEND_URL = "http://internal-backend:8000";
      process.env.BACKEND_API_KEY = "backend-key-present";
      process.env.REVIEWER_ACCESS_TOKEN = "required-reviewer-token";

      const fetchSpy = vi.fn();
      global.fetch = fetchSpy;

      // Request without reviewer token
      const req = createRequest({ question: "Valid question?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(401);
      const json = await res.json();
      expect(json.detail).toContain("Unauthorized reviewer access");
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("preserves open local development when NODE_ENV!=production and APP_ENV=development with unset REVIEWER_ACCESS_TOKEN", async () => {
      (process.env as Record<string, string | undefined>).NODE_ENV = "development";
      process.env.APP_ENV = "development";
      delete process.env.REVIEWER_ACCESS_TOKEN;
      process.env.BACKEND_API_KEY = "";

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockBackendSuccess,
      } as Response);

      const req = createRequest({ question: "Local dev test?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(200);
      const data = await res.json();
      expect(data.status).toBe("answered");
    });
  });

  // ---------------------------------------------------------------------------
  // 3. Safe Errors & Backend Failure Propagation
  // ---------------------------------------------------------------------------
  describe("Safe Errors & Backend Status Propagation", () => {
    it("propagates 429 Rate Limit from backend with sanitized error message", async () => {
      process.env.APP_ENV = "development";
      delete process.env.REVIEWER_ACCESS_TOKEN;

      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 429,
        json: async () => ({ detail: "Rate limit exceeded. Maximum 15 queries per minute." }),
      } as unknown as Response);

      const req = createRequest({ question: "Burst query?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(429);
      const json = await res.json();
      expect(json.detail).toContain("Rate limit exceeded");
    });

    it("returns 502 with safe error when backend service is unreachable (network error)", async () => {
      process.env.APP_ENV = "development";
      delete process.env.REVIEWER_ACCESS_TOKEN;

      global.fetch = vi.fn().mockRejectedValue(new Error("ECONNREFUSED 127.0.0.1:8000"));

      const req = createRequest({ question: "Unreachable test?", top_k: 4 });
      const res = await POST(req);

      expect(res.status).toBe(502);
      const json = await res.json();
      expect(json.detail).toContain("Unable to reach backend RAG service");
      // Must not leak secret keys or sensitive configuration
      expect(JSON.stringify(json)).not.toContain("BACKEND_API_KEY");
    });

    it("rejects invalid request body with 400 Bad Request", async () => {
      process.env.APP_ENV = "development";
      delete process.env.REVIEWER_ACCESS_TOKEN;

      const req = createRequest({ question: "   " });
      const res = await POST(req);

      expect(res.status).toBe(400);
      const json = await res.json();
      expect(json.detail).toContain("Question must be a non-empty string");
    });
  });
});
