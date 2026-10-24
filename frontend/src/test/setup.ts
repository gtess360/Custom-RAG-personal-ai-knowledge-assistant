import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// Polyfill scrollIntoView for jsdom
if (typeof window !== "undefined") {
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
}
