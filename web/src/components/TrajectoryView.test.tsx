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

  it("nests worker steps behind an expand control", () => {
    const orch: TrajectoryRecord = {
      ...routeRecord(),
      id: "o1",
      kind: "orchestrator",
      event: "orchestrator.start",
      title: "编排 4 个节点",
      payload: {
        strategy: "synthesize",
        workers: [
          { id: "w0", title: "查北京天气" },
          { id: "m3", title: "写对比分析" },
        ],
      },
    };
    const worker: TrajectoryRecord = {
      ...routeRecord(),
      id: "w0",
      kind: "worker",
      event: "worker.start",
      title: "子代理 查北京天气",
      payload: { worker_id: "w0", title: "查北京天气" },
    };
    const step: TrajectoryRecord = {
      ...routeRecord(),
      id: "s1",
      kind: "tool",
      event: "step.tool_call",
      title: "工具 get_weather",
      payload: { worker_id: "w0", tool: "get_weather" },
      currentTool: "get_weather",
    };
    render(
      <TrajectoryView
        records={[orch, worker, step]}
        hasMore={false}
        loadingEarlier={false}
        followTail
        focusId={null}
        onFollowChange={vi.fn()}
        onLoadEarlier={vi.fn()}
        onExport={vi.fn()}
      />,
    );
    expect(screen.getByText("编排 4 个节点")).toBeInTheDocument();
    expect(screen.getByText("子代理 查北京天气")).toBeInTheDocument();
    expect(screen.queryByText("工具 get_weather")).toBeNull();
    fireEvent.click(document.querySelector(".expand-caret") as HTMLElement);
    expect(screen.getByText("工具 get_weather")).toBeInTheDocument();
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

  it("renders turn headers in ascending order", () => {
    render(
      <TrajectoryView
        records={[
          { ...routeRecord(), id: "t2", turn: 2, seq: 2, ts: 200, title: "意图：第二轮" },
          { ...routeRecord(), id: "t1", turn: 1, seq: 1, ts: 100, title: "意图：第一轮" },
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
    const headers = screen.getAllByText(/轮次 \d/);
    expect(headers.map((el) => el.textContent)).toEqual(["轮次 1", "轮次 2"]);
  });

  it("renders HH:MM:SS from a unix timestamp on the record row", () => {
    render(
      <TrajectoryView
        records={[{ ...routeRecord(), ts: 1_710_000_000 }]}
        hasMore={false}
        loadingEarlier={false}
        followTail
        focusId={null}
        onFollowChange={vi.fn()}
        onLoadEarlier={vi.fn()}
        onExport={vi.fn()}
      />,
    );
    expect(screen.getByText(/\d{2}:\d{2}:\d{2}/)).toBeInTheDocument();
  });

  it("renders planner source without leaking undefined", () => {
    const rec: TrajectoryRecord = {
      ...routeRecord(),
      title: "意图：工具调用 ← 任务卡",
      payload: {
        mode: "agent",
        source: "planner",
        reason: "任务卡",
        confidence: 0.85,
        turn: 1,
      },
    };
    const { container } = render(
      <TrajectoryView
        records={[rec]}
        hasMore={false}
        loadingEarlier={false}
        followTail
        focusId={null}
        onFollowChange={vi.fn()}
        onLoadEarlier={vi.fn()}
        onExport={vi.fn()}
      />,
    );
    expect(screen.getByText(/任务卡/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /任务卡/ }));
    expect(container.textContent).not.toContain("undefined");
    expect(screen.getByText("规划器生成")).toBeInTheDocument();
  });

  it("renders llm.call as a muted internal-call row with previews", () => {
    const rec: TrajectoryRecord = {
      ...routeRecord(),
      id: "llm1",
      kind: "other",
      event: "llm.call",
      title: "内部调用：标题生成",
      elapsedMs: 12,
      payload: {
        purpose: "rename",
        prompt_preview: "为对话生成不超过 12 字标题",
        response_preview: "北京天气查询",
        elapsed_ms: 12,
        usage: { prompt_tokens: 24, completion_tokens: 165, total_tokens: 189 },
      },
      input: "为对话生成不超过 12 字标题",
      output: "北京天气查询",
    };
    render(
      <TrajectoryView
        records={[rec]}
        hasMore={false}
        loadingEarlier={false}
        followTail
        focusId={null}
        onFollowChange={vi.fn()}
        onLoadEarlier={vi.fn()}
        onExport={vi.fn()}
      />,
    );
    const btn = screen.getByRole("button", { name: /内部调用：标题生成/ });
    expect(btn.className).toContain("llm-call");
    expect(btn.querySelector(".record-title")?.className).toContain("status-meta");
    fireEvent.click(btn);
    expect(screen.getByText("为对话生成不超过 12 字标题")).toBeInTheDocument();
    expect(screen.getByText("北京天气查询")).toBeInTheDocument();
    expect(screen.getByText(/prompt_tokens/)).toBeInTheDocument();
  });
});
