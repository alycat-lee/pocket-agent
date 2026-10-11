import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { KnowledgeCard } from "./KnowledgeCard";

vi.mock("../api/client", () => ({
  fetchKnowledgeStatus: () =>
    Promise.resolve({
      available: true,
      reason: "",
      embedder: "hash",
      doc_count: 1,
    }),
  fetchKnowledgeDocuments: () =>
    Promise.resolve({
      available: true,
      reason: "",
      documents: [
        {
          id: "doc1",
          name: "手册.md",
          source: "手册.md",
          chunks: 2,
          created_at: 1,
        },
      ],
    }),
  uploadKnowledgeFile: () => Promise.resolve({ ok: true, id: "x", name: "x", chunks: 1 }),
  deleteKnowledgeDocument: () => Promise.resolve({ ok: true, id: "doc1" }),
  searchKnowledge: () =>
    Promise.resolve({ query: "协议", text: "[来源：手册.md] 星核协议", hits: [] }),
}));

describe("KnowledgeCard", () => {
  it("shows knowledge group with document list", async () => {
    render(<KnowledgeCard />);
    await waitFor(() => {
      expect(screen.getByText("手册.md")).toBeInTheDocument();
    });
    expect(screen.getByText(/知识库/)).toBeInTheDocument();
    expect(screen.getByText("hash")).toBeInTheDocument();
    expect(screen.getByText(/命中测试/)).toBeInTheDocument();
  });
});
