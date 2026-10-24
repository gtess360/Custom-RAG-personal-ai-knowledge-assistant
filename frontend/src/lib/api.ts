import { HealthResponse, QueryResponse } from "../types/api";

const REVIEWER_TOKEN_KEY = "reviewer_access_token";

/**
 * Retrieve the reviewer access token from session storage (if set by the reviewer in the browser).
 */
export function getReviewerToken(): string | null {
  if (typeof window !== "undefined") {
    try {
      return sessionStorage.getItem(REVIEWER_TOKEN_KEY) || null;
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Persist or clear the reviewer access token in session storage for the current browser session.
 */
export function setReviewerToken(token: string | null): void {
  if (typeof window !== "undefined") {
    try {
      if (token && token.trim()) {
        sessionStorage.setItem(REVIEWER_TOKEN_KEY, token.trim());
      } else {
        sessionStorage.removeItem(REVIEWER_TOKEN_KEY);
      }
    } catch {
      // Ignore storage errors in restricted environments
    }
  }
}

export class ApiError extends Error {
  public statusCode?: number;
  public details?: unknown;

  constructor(message: string, statusCode?: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.details = details;
  }
}

/**
 * Check backend liveness via Next.js proxy route (/api/health).
 */
export async function checkHealth(timeoutMs = 4000): Promise<HealthResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch("/api/health", {
      method: "GET",
      signal: controller.signal,
      headers: {
        Accept: "application/json",
      },
      cache: "no-store",
    });

    if (!res.ok) {
      throw new ApiError(`Health check failed with status ${res.status}`, res.status);
    }

    const data: HealthResponse = await res.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    if (err instanceof Error && err.name === "AbortError") {
      throw new ApiError("Health check request timed out", 408);
    }
    throw new ApiError(
      "Unable to reach backend service. Ensure FastAPI is running.",
      0
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Execute RAG query via Next.js server-side proxy route (/api/query).
 *
 * Security & Token Flow:
 * - Browser requests use relative /api/query.
 * - If a reviewerToken is provided (or stored in sessionStorage), it is passed in the
 *   Authorization header to satisfy proxy reviewer-access checks.
 * - The backend API key is NEVER exposed in the browser.
 */
export async function queryAssistant(
  question: string,
  // NOTE: Separately configured client default; not dynamically shared from backend DEFAULT_TOP_K.
  top_k: number = 5,
  timeoutMs = 60000,
  reviewerToken?: string
): Promise<QueryResponse> {
  const trimmed = question.trim();
  if (!trimmed) {
    throw new ApiError("Please enter a question before submitting.", 400);
  }

  const token = reviewerToken || getReviewerToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      signal: controller.signal,
      headers,
      body: JSON.stringify({
        question: trimmed,
        top_k: Math.max(1, Math.min(10, Math.round(top_k))),
      }),
      cache: "no-store",
    });

    if (!res.ok) {
      let errorDetail = `Backend returned HTTP ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === "string" 
            ? errorJson.detail 
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Fallback to default message
      }
      throw new ApiError(errorDetail, res.status);
    }

    const data: QueryResponse = await res.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    if (err instanceof Error && err.name === "AbortError") {
      throw new ApiError("The query request timed out. Please try again.", 408);
    }
    throw new ApiError(
      "Network error connecting to /api/query. Please verify the server and FastAPI backend are running.",
      0
    );
  } finally {
    clearTimeout(timer);
  }
}
