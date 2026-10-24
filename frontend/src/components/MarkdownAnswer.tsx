"use client";

import React from "react";

interface MarkdownAnswerProps {
  content: string;
  onCitationClick: (chunkId: string) => void;
  validCitationIds?: Set<string>;
}

/**
 * Clean markdown-aware renderer that detects [chunk_id] citation tokens
 * and renders them as interactive, clickable badges.
 */
export const MarkdownAnswer: React.FC<MarkdownAnswerProps> = ({
  content,
  onCitationClick,
  validCitationIds = new Set(),
}) => {
  // Regex to match citation tokens like [chunk_id] or standard text
  // Chunks typically look like [filename_pX_cY] or alphanumeric with underscores/hyphens
  const citationRegex = /\[([a-zA-Z0-9_\-\.]+)\]/g;

  // Split content into paragraphs
  const paragraphs = content.split(/\n\s*\n/);

  const renderInline = (text: string) => {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    // Reset regex state
    citationRegex.lastIndex = 0;

    while ((match = citationRegex.exec(text)) !== null) {
      const matchIndex = match.index;
      const fullMatch = match[0];
      const chunkId = match[1];

      // Add text before match
      if (matchIndex > lastIndex) {
        parts.push(renderFormattedText(text.substring(lastIndex, matchIndex), `text-${lastIndex}`));
      }

      // Check if this looks like a citation chunk ID (contains underscore or in validCitationIds)
      const isCitation = validCitationIds.has(chunkId) || chunkId.includes("_");

      if (isCitation) {
        parts.push(
          <button
            key={`cite-${matchIndex}-${chunkId}`}
            type="button"
            className="citation-badge"
            title={`View evidence chunk: ${chunkId}`}
            onClick={() => onCitationClick(chunkId)}
          >
            <span>📄</span>
            <span>[{chunkId}]</span>
          </button>
        );
      } else {
        parts.push(fullMatch);
      }

      lastIndex = matchIndex + fullMatch.length;
    }

    if (lastIndex < text.length) {
      parts.push(renderFormattedText(text.substring(lastIndex), `text-${lastIndex}`));
    }

    return parts;
  };

  const renderFormattedText = (raw: string, keyPrefix: string) => {
    // Basic bold parsing (**bold**)
    const boldParts = raw.split(/(\*\*[^*]+\*\*)/g);
    return (
      <React.Fragment key={keyPrefix}>
        {boldParts.map((part, idx) => {
          if (part.startsWith("**") && part.endsWith("**")) {
            return <strong key={idx}>{part.slice(2, -2)}</strong>;
          }
          return part;
        })}
      </React.Fragment>
    );
  };

  return (
    <div className="answer-body">
      {paragraphs.map((p, idx) => {
        // Handle bullet items
        if (p.trim().startsWith("- ") || p.trim().startsWith("* ")) {
          const items = p.split(/\n(?=[-*] )/);
          return (
            <ul key={idx} style={{ paddingLeft: "24px", marginBottom: "14px" }}>
              {items.map((item, itemIdx) => (
                <li key={itemIdx} style={{ marginBottom: "6px" }}>
                  {renderInline(item.replace(/^[-*]\s+/, ""))}
                </li>
              ))}
            </ul>
          );
        }

        return <p key={idx}>{renderInline(p)}</p>;
      })}
    </div>
  );
};
