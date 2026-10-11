import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { TrajectoryRecord } from "../api/types";
import { TrajectoryView } from "./TrajectoryView";

function routeRecord(): TrajectoryRecord {
  return {
    id: "r1",
    seq: 1,
    event: "intent.detected",
    turn: 1,
    stepIndex: null,
    kind: "route",
    title: '意图：长文写作 ← 规则命中"写一篇"',
    ts: null,
    elapsedMs: null,
    ttftMs: null,
    promptTokensEst: null,
    currentTool: null,
    badges: [],
    errorCode: null,
    inProgress: false,
    payload: {
      mode: "writer",
      source: "rule",
      reason: "写一篇",
      confidence: 1,
      turn: 1,
      query: "写一篇关于春天的散文",
    },
    input: "写一篇关于春天的散文",
  };
}

describe("TrajectoryView route record", () => {
  it("renders route as the first row and shows details in inspector", () => {
    render(
      <TrajectoryView
        records={[
          {
            ...routeRecord(),
            id: "u1",
            kind: "user",
            event: "user.message",
            title: "用户消息",
            payload: {},
          },
          routeRecord(),
        ]}
        hasMore={false}
        loadingEarlier={false}
        followTail
        focusId={null}
        onFollowChange={vi.fn()}
        onLoadEarlier={vi.fn()}
        onExport={vi.fn()}
      />,
    );
    const titles = screen.getAllByText(/意图：长文写作/);
    expect(titles.length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: /意图：长文写作/ }));
    expect(screen.getByText("写一篇")).toBeInTheDocument();
    expect(screen.getByText("写一篇关于春天的散文")).toBeInTheDocument();
    expect(screen.getByText("规则命中")).toBeInTheDocument();
  });

  it("renders task card plan with checkboxes in inspector", () => {
    const task: TrajectoryRecord = {
      ...routeRecord(),
      id: "task1",
      kind: "task",
      event: "task.planned",
      title: "任务卡：写一篇两万字小说",
      payload: {
        type: "writer",
        goal: "写一篇两万字小说",
        constraints: { total_chars: 20000 },
        plan: ["开篇", "发展"],
        plan_done: [true, false],
        model: "deepseek-chat",
        provider_name: "DeepSeek",
        provider_type: "deepseek",
      },
    };
    render(
      <TrajectoryView
        records={[task]}
        hasMore={false}
        loadingEarlier={false}
        followTail
        focusId={null}
        onFollowChange={vi.fn()}
        onLoadEarlier={vi.fn()}
        onExport={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /任务卡：写一篇两万字小说/ }));
    expect(screen.getByText("写一篇两万字小说")).toBeInTheDocument();
    expect(screen.getByText("开篇")).toBeInTheDocument();
    expect(screen.getByText("发展")).toBeInTheDocument();
    expect(document.querySelector(".plan-list li.done")).not.toBeNull();
    expect(screen.getByText(/deepseek-chat/)).toBeInTheDocument();
  });

  it("shows thinking block in inspector, distinct from answer body", () => {
    const thinking: TrajectoryRecord = {
      ...routeRecord(),
      id: "t1",
      kind: "thinking",
      event: "step.thinking",
      title: "思维链",
      payload: { text: "先构思情节再落笔" },
      output: "先构思情节再落笔",
      input: undefined,
    };
    render(
      <TrajectoryView
        records={[thinking]}
        hasMore={false}
        loadingEarlier={false}
        followTail
        focusId={null}
        onFollowChange={vi.fn()}
        onLoadEarlier={vi.fn()}
        onExport={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /思维链/ }));
    expect(screen.getByText("先构思情节再落笔")).toBeInTheDocument();
    expect(document.querySelector(".thinking-block")).not.toBeNull();
  });
});
