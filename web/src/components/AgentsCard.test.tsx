import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AgentsCard } from "./AgentsCard";

vi.mock("../api/client", () => ({
  fetchAgents: () =>
    Promise.resolve({
      root: "agents",
      agents: [
        {
          name: "default",
          description: "默认助手",
          provider_id: null,
          tools: null,
          permission: "workspace",
          prompt: "你是默认助手",
          root: "agents/default",
        },
      ],
    }),
  createAgent: () => Promise.resolve({ ok: true, agent: {} }),
  updateAgent: () => Promise.resolve({ ok: true, agent: {} }),
  deleteAgent: () => Promise.resolve({ ok: true, name: "x" }),
}));

describe("AgentsCard", () => {
  it("lists agents and hides delete on default", async () => {
    render(<AgentsCard />);
    await waitFor(() => {
      expect(screen.getByText("default")).toBeInTheDocument();
    });
    expect(screen.getByText(/不可删除/)).toBeInTheDocument();
    expect(screen.getByText(/根目录/)).toBeInTheDocument();
  });
});
