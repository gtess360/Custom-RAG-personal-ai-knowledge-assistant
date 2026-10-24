"use client";

import React, { useEffect, useState } from "react";
import { checkHealth, getReviewerToken, setReviewerToken } from "../lib/api";

export const Navbar: React.FC = () => {
  const [healthStatus, setHealthStatus] = useState<"checking" | "online" | "offline">("checking");
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [reviewerToken, setTokenState] = useState<string>(() => getReviewerToken() || "");
  const [isTokenModalOpen, setIsTokenModalOpen] = useState(false);

  const performHealthCheck = async () => {
    try {
      const res = await checkHealth(3000);
      if (res.status === "ok") {
        setHealthStatus("online");
      } else {
        setHealthStatus("offline");
      }
    } catch {
      setHealthStatus("offline");
    }
    setLastChecked(new Date());
  };

  useEffect(() => {
    let isMounted = true;
    checkHealth(3000)
      .then((res) => {
        if (isMounted) {
          setHealthStatus(res.status === "ok" ? "online" : "offline");
          setLastChecked(new Date());
        }
      })
      .catch(() => {
        if (isMounted) {
          setHealthStatus("offline");
          setLastChecked(new Date());
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleManualRefresh = () => {
    setHealthStatus("checking");
    performHealthCheck();
  };

  const handleSaveToken = (newToken: string) => {
    const trimmed = newToken.trim();
    setReviewerToken(trimmed || null);
    setTokenState(trimmed);
    setIsTokenModalOpen(false);
  };

  return (
    <header className="navbar">
      <div className="container navbar-inner">
        <div className="brand-group">
          <div className="brand-icon">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 2a8 8 0 0 0-8 8c0 3.31 2.02 6.16 4.9 7.37L9 22h6l.1-4.63C17.98 16.16 20 13.31 20 10a8 8 0 0 0-8-8z" />
              <path d="M9 10h.01" />
              <path d="M15 10h.01" />
              <path d="M9.5 14a3.5 3.5 0 0 0 5 0" />
            </svg>
          </div>
          <div>
            <div className="brand-title">Personal AI Knowledge Assistant</div>
          </div>
          <span className="brand-badge">RAG Pipeline</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {/* Reviewer Access Token Button */}
          <button
            type="button"
            onClick={() => setIsTokenModalOpen(!isTokenModalOpen)}
            className="reviewer-token-btn"
            title="Configure reviewer access token for demonstration queries"
            data-testid="reviewer-access-btn"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "6px 12px",
              fontSize: "0.8rem",
              borderRadius: "8px",
              background: reviewerToken ? "rgba(16, 185, 129, 0.15)" : "rgba(255, 255, 255, 0.05)",
              border: reviewerToken ? "1px solid rgba(16, 185, 129, 0.35)" : "1px solid rgba(255, 255, 255, 0.1)",
              color: reviewerToken ? "#6ee7b7" : "var(--text-muted)",
              cursor: "pointer",
              fontWeight: 500,
            }}
          >
            <span>🔑</span>
            <span>{reviewerToken ? "Reviewer Token: Set" : "Reviewer Access"}</span>
          </button>

          {/* Backend Health Pill */}
          <button
            type="button"
            className="health-pill"
            onClick={handleManualRefresh}
            title={
              lastChecked
                ? `Backend via Proxy: ${healthStatus} (Last checked ${lastChecked.toLocaleTimeString()}) - Click to refresh`
                : "Checking backend health..."
            }
          >
            <span className={`status-dot ${healthStatus}`} />
            <span style={{ color: "var(--text-muted)" }}>
              Backend:{" "}
              <strong
                style={{
                  color:
                    healthStatus === "online"
                      ? "#34d399"
                      : healthStatus === "offline"
                      ? "#f87171"
                      : "#fbbf24",
                  fontWeight: 600,
                }}
              >
                {healthStatus === "online"
                  ? "Connected (via Proxy)"
                  : healthStatus === "offline"
                  ? "Offline"
                  : "Checking..."}
              </strong>
            </span>
          </button>
        </div>
      </div>

      {/* Reviewer Access Token Modal */}
      {isTokenModalOpen && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.7)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
            backdropFilter: "blur(4px)",
          }}
          onClick={() => setIsTokenModalOpen(false)}
        >
          <div
            style={{
              background: "var(--bg-card, #0f172a)",
              border: "1px solid rgba(255, 255, 255, 0.15)",
              borderRadius: "12px",
              padding: "24px",
              maxWidth: "420px",
              width: "90%",
              boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.5)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ margin: "0 0 8px 0", fontSize: "1.1rem", color: "var(--text-primary, #f8fafc)" }}>
              Reviewer Access Token
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted, #94a3b8)", marginBottom: "16px", lineHeight: 1.4 }}>
              When deployed, the server-side proxy requires an authorized reviewer token to execute RAG queries.
              The token is stored in your current browser session only and never hardcoded in source.
            </p>
            <input
              type="password"
              placeholder="Enter reviewer token..."
              value={reviewerToken}
              onChange={(e) => setTokenState(e.target.value)}
              data-testid="reviewer-token-input"
              style={{
                width: "100%",
                padding: "8px 12px",
                borderRadius: "6px",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                background: "rgba(0, 0, 0, 0.3)",
                color: "#fff",
                fontSize: "0.9rem",
                marginBottom: "16px",
                outline: "none",
                boxSizing: "border-box",
              }}
            />
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px" }}>
              <button
                type="button"
                onClick={() => {
                  handleSaveToken("");
                }}
                style={{
                  padding: "6px 12px",
                  borderRadius: "6px",
                  background: "transparent",
                  border: "1px solid rgba(255, 255, 255, 0.15)",
                  color: "#94a3b8",
                  cursor: "pointer",
                  fontSize: "0.85rem",
                }}
              >
                Clear
              </button>
              <button
                type="button"
                onClick={() => handleSaveToken(reviewerToken)}
                data-testid="save-token-btn"
                style={{
                  padding: "6px 16px",
                  borderRadius: "6px",
                  background: "var(--primary-color, #3b82f6)",
                  border: "none",
                  color: "#fff",
                  fontWeight: 600,
                  cursor: "pointer",
                  fontSize: "0.85rem",
                }}
              >
                Save Token
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};

