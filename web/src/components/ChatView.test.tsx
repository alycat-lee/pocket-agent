import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ChatMessage } from "../api/types";
import { ChatView } from "./ChatView";

function assistant(intentMode: "qa" | "agent" | "writer"): ChatMessage {
  return {
    id: "a1",
    role: "assistant",
    content: "回复",
    turnId: "turn-1",
    settlement: { total_steps: 0, intercepts: 0, cache_hit_rate: 0 },
    intent: {
      mode: intentMode,
      confidence: 1,
      source: intentMode === "agent" ? "llm" : "rule",
      reason: intentMode === "writer" ? "写一篇" : "短句闲聊",
      turn: 1,
    },
  };
}

describe("ChatView user bubble", () => {
  it("applies msg-user class so short bubbles shrink to the right", () => {
    render(
      <ChatView
        messages={[{ id: "u1", role: "user", content: "短消息" }]}
        onSettlementClick={vi.fn()}
        onCorrectIntent={vi.fn()}
      />,
    );
    const bubble = screen.getByText("短消息").closest(".msg");
    expect(bubble?.className).toContain("msg-user");
    expect(bubble?.className).toContain("user");
  });
});

describe("ChatView intent badge", () => {
  it("renders qa badge on the settlement row", () => {
    render(
      <ChatView
        messages={[assistant("qa")]}
        onSettlementClick={vi.fn()}
        onCorrectIntent={vi.fn()}
      />,
    );
    expect(screen.getByText("直接问答 · 未调用工具")).toBeInTheDocument();
    expect(screen.queryByText(/自动识别/)).not.toBeInTheDocument();
  });

  it("renders agent badge and confirms correction rerun", () => {
    const onCorrect = vi.fn();
    render(
      <ChatView
        messages={[assistant("agent")]}
        onSettlementClick={vi.fn()}
        onCorrectIntent={onCorrect}
      />,
    );
    expect(
      screen.getByText("自动识别：工具调用 · LLM 分类 · 点击纠正"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /自动识别：工具调用/ }));
    fireEvent.click(screen.getByRole("menuitem", { name: "长文写作" }));
    expect(onCorrect).toHaveBeenCalledWith("turn-1", "writer");
  });
});
