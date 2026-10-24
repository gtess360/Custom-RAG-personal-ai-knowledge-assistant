import { NextResponse } from "next/server";

/**
 * Next.js Server-Side Health Check Proxy.
 *
 * Forwards health probe to the backend to verify end-to-end service connectivity
 * without exposing the private backend URL to browser clients.
 */

export async function GET(): Promise<NextResponse> {
  const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";
  const targetUrl = `${backendUrl.replace(/\/+$/, "")}/health`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);

    const res = await fetch(targetUrl, {
      method: "GET",
      signal: controller.signal,
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    clearTimeout(timeout);

    if (res.ok) {
      const data = await res.json().catch(() => ({ status: "ok" }));
      return NextResponse.json(data, { status: 200 });
    }

    return NextResponse.json(
      { status: "unhealthy", detail: `Backend returned status ${res.status}` },
      { status: 503 }
    );
  } catch {
    return NextResponse.json(
      { status: "offline", detail: "Backend service unreachable" },
      { status: 503 }
    );
  }
}
