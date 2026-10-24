import { NextRequest, NextResponse } from "next/server";

/**
 * Next.js Server-Side Proxy Route Handler for RAG Query Pipeline.
 *
 * Security & Architectural Boundaries:
 * 1. Server-Only Secrets: BACKEND_API_KEY and BACKEND_URL are read strictly on the server.
 *    They are never prefixed with NEXT_PUBLIC_ and are never exposed to browser client bundles.
 * 2. Authenticated Reviewer Access: The proxy is NOT an open relay. In production (or when
 *    REVIEWER_ACCESS_TOKEN is set), incoming requests must provide valid reviewer credentials
 *    (via `Authorization: Bearer <token>` or `x-reviewer-token: <token>`).
 * 3. Production Configuration: When APP_ENV=production, both REVIEWER_ACCESS_TOKEN and
 *    BACKEND_API_KEY must be configured. The server refuses to proxy requests if either is missing.
 * 4. Local Development Convenience: When APP_ENV=development (default) and REVIEWER_ACCESS_TOKEN
 *    is unset, open access is permitted for seamless local testing.
 * 5. Safe Error Sanitization: Internal network errors or backend secrets are never leaked in error messages.
 */

function extractReviewerToken(request: NextRequest): string | null {
  const authHeader = request.headers.get("authorization");
  if (authHeader) {
    const parts = authHeader.trim().split(/\s+/);
    if (parts.length === 2 && parts[0].toLowerCase() === "bearer") {
      return parts[1];
    }
  }
  const customHeader = request.headers.get("x-reviewer-token");
  if (customHeader) {
    return customHeader.trim();
  }
  return null;
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  const nodeEnv = (process.env.NODE_ENV || "").trim().toLowerCase();
  const appEnv = (process.env.APP_ENV || "").trim().toLowerCase();

  // Fail closed: If NODE_ENV is production, enforce production even if APP_ENV
  // is missing or mistakenly set to development. Also enforce if APP_ENV=production.
  const isProduction = nodeEnv === "production" || appEnv === "production";

  const backendUrl = process.env.BACKEND_URL?.trim();
  const backendApiKey = process.env.BACKEND_API_KEY?.trim();
  const reviewerAccessToken = process.env.REVIEWER_ACCESS_TOKEN?.trim();

  // 1. Production Configuration Check (Fails closed before reading or forwarding any query)
  if (isProduction) {
    if (!backendUrl) {
      return NextResponse.json(
        { detail: "Production configuration error: BACKEND_URL must be configured on the server." },
        { status: 500 }
      );
    }
    if (!backendApiKey) {
      return NextResponse.json(
        { detail: "Production configuration error: BACKEND_API_KEY must be configured on the server." },
        { status: 500 }
      );
    }
    if (!reviewerAccessToken) {
      return NextResponse.json(
        { detail: "Production configuration error: REVIEWER_ACCESS_TOKEN must be configured on the server." },
        { status: 500 }
      );
    }
  }

  // 2. Authenticated Reviewer Access Check
  if (reviewerAccessToken) {
    const providedToken = extractReviewerToken(request);
    if (!providedToken || providedToken !== reviewerAccessToken) {
      return NextResponse.json(
        { detail: "Unauthorized reviewer access. A valid reviewer access token is required." },
        {
          status: 401,
          headers: { "WWW-Authenticate": 'Bearer realm="Reviewer Access"' },
        }
      );
    }
  }

  // 3. Request Body Validation
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { detail: "Invalid JSON request body." },
      { status: 400 }
    );
  }

  if (typeof body !== "object" || body === null || !("question" in body)) {
    return NextResponse.json(
      { detail: "Missing required 'question' field in request body." },
      { status: 400 }
    );
  }

  const { question, top_k } = body as { question: unknown; top_k?: unknown };
  if (typeof question !== "string" || !question.trim()) {
    return NextResponse.json(
      { detail: "Question must be a non-empty string." },
      { status: 400 }
    );
  }

  // 4. Proxy to Backend Service
  const resolvedBackendUrl = backendUrl || "http://127.0.0.1:8000";
  const targetUrl = `${resolvedBackendUrl.replace(/\/+$/, "")}/query`;
  const backendHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };

  if (backendApiKey) {
    backendHeaders["Authorization"] = `Bearer ${backendApiKey}`;
  }

  // Forward client IP safely for backend rate-limiting
  const clientIp =
    request.headers.get("x-forwarded-for")?.split(",")[0].trim() ||
    request.headers.get("x-real-ip") ||
    "127.0.0.1";
  backendHeaders["X-Forwarded-For"] = clientIp;

  try {
    const backendRes = await fetch(targetUrl, {
      method: "POST",
      headers: backendHeaders,
      body: JSON.stringify({
        question: question.trim(),
        // Fallback to top_k=5 if omitted.
        // NOTE: Separately configured proxy fallback; not dynamically shared from backend DEFAULT_TOP_K.
        top_k: typeof top_k === "number" ? top_k : 5,
      }),
      cache: "no-store",
    });

    const data = await backendRes.json().catch(() => null);

    if (!backendRes.ok) {
      const errorDetail =
        data && typeof data.detail === "string"
          ? data.detail
          : `Backend service responded with HTTP ${backendRes.status}`;
      return NextResponse.json(
        { detail: errorDetail },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown network error";
    // Sanitize network errors without exposing internal host/port or secret configuration
    return NextResponse.json(
      {
        detail: `Unable to reach backend RAG service. Ensure FastAPI backend is running and reachable. (${message})`,
      },
      { status: 502 }
    );
  }
}
