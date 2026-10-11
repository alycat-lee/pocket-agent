import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PromptsCard } from "./PromptsCard";

vi.mock("../api/client", () => ({
  fetchPrompts: () =>
    Promise.resolve({
      variant: "local",
      prompts: [
        {
          id: "choose_tool",
          template: "选编号；够了输出 -1。",
          variants: { local: "选编号；够了输出 -1。", cloud: "下一步应该使用哪个工具？" },
          version: "1.0.0",
        },
      ],
    }),
}));

describe("PromptsCard", () => {
  it("shows active variant, version and template", async () => {
    render(<PromptsCard />);
    await waitFor(() => {
      expect(screen.getByText("choose_tool")).toBeInTheDocument();
    });
    expect(screen.getByText(/当前变体/)).toBeInTheDocument();
    expect(screen.getByText("local")).toBeInTheDocument();
    expect(screen.getByText("v1.0.0")).toBeInTheDocument();
    expect(screen.getByText(/选编号/)).toBeInTheDocument();
  });
});
