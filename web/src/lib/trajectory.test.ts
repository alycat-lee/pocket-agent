import { describe, expect, it } from "vitest";
import type { TrajectoryRecord } from "../api/types";
import {
  belongsToSession,
  orderRecordsForDisplay,
  projectEvents,
  upsertLiveEvent,
} from "./trajectory";

function rec(
  partial: Partial<TrajectoryRecord> &
    Pick<TrajectoryRecord, "id" | "kind" | "title">,
): TrajectoryRecord {
  return {
    seq: 1,
    event: "other",
    turn: 1,
    stepIndex: null,
    ts: null,
    elapsedMs: null,
    ttftMs: null,
    promptTokensEst: null,
    currentTool: null,
    badges: [],
    errorCode: null,
    inProgress: false,
    payload: {},
    ...partial,
  };
}

describe("projectEvents route", () => {
  it("maps intent.detected to route kind and title", () => {
    const rows = projectEvents([
      {
        seq: 1,
        event: "intent.detected",
        data: {
          mode: "writer",
          source: "rule",
          reason: "写一篇",
          confidence: 1,
          turn: 1,
          query: "写一篇关于春天的散文",
        },
      },
    ]);
    expect(rows[0]?.kind).toBe("route");
    expect(rows[0]?.title).toBe('意图：长文写作 ← 规则命中"写一篇"');
    expect(rows[0]?.input).toBe("写一篇关于春天的散文");
  });

  it("merges run.start model into the task card payload", () => {
    const rows = projectEvents([
      {
        seq: 1,
        event: "run.start",
        data: {
          run_id: "r1",
          provider_name: "DeepSeek",
          provider_type: "deepseek",
          model: "deepseek-chat",
        },
      },
      {
        seq: 2,
        event: "task.planned",
        data: {
          type: "writer",
          goal: "写一篇长文",
          plan: ["开篇"],
        },
      },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe("task");
    expect(rows[0]?.payload.model).toBe("deepseek-chat");
    expect(rows[0]?.payload.provider_name).toBe("DeepSeek");
    expect(rows[0]?.payload.provider_type).toBe("deepseek");
  });

  it("maps task.planned to task kind with goal title", () => {
    const rows = projectEvents([
      {
        seq: 1,
        event: "task.planned",
        data: {
          type: "writer",
          goal: "写一篇两万字小说",
          constraints: { total_chars: 20000 },
          plan: ["开篇", "发展"],
          plan_done: [false, false],
          turn: 1,
        },
      },
    ]);
    expect(rows[0]?.kind).toBe("task");
    expect(rows[0]?.title).toBe("任务卡：写一篇两万字小说");
  });

  it("maps answer.rejected to rejected badge", () => {
    const rows = projectEvents([
      {
        seq: 3,
        event: "answer.rejected",
        data: { reasons: ["思维链泄漏"] },
      },
    ]);
    expect(rows[0]?.badges).toContain("rejected");
    expect(rows[0]?.title).toBe("验收未通过");
  });

  it("maps orchestrator and worker events into nested kinds", () => {
    const rows = projectEvents([
      {
        seq: 1,
        event: "orchestrator.start",
        data: {
          strategy: "synthesize",
          workers: [
            { id: "w0", title: "查北京天气", kind: "worker" },
            { id: "w1", title: "查上海天气", kind: "worker" },
            { id: "w2", title: "查广州天气", kind: "worker" },
            { id: "m3", title: "写对比分析", kind: "merge" },
          ],
          turn: 1,
        },
      },
      {
        seq: 2,
        event: "worker.start",
        data: { worker_id: "w0", title: "查北京天气", turn: 1 },
      },
      {
        seq: 3,
        event: "step.tool_call",
        data: { worker_id: "w0", tool: "get_weather", step_index: 1 },
      },
      {
        seq: 4,
        event: "merger.start",
        data: { worker_id: "m3", strategy: "synthesize", title: "写对比分析" },
      },
    ]);
    expect(rows.map((r) => r.kind)).toEqual([
      "orchestrator",
      "worker",
      "tool",
      "merge",
    ]);
    expect(rows[0]?.title).toContain("4 个节点");
    expect(rows[1]?.title).toContain("查北京天气");
    expect(rows[3]?.title).toContain("synthesize");
  });

  it("maps step.thinking to thinking kind and text output", () => {
    const rows = projectEvents([
      {
        seq: 2,
        event: "step.thinking",
        data: { text: "先列大纲再写", step_index: 1 },
      },
    ]);
    expect(rows[0]?.kind).toBe("thinking");
    expect(rows[0]?.title).toBe("思维链");
    expect(rows[0]?.output).toBe("先列大纲再写");
  });

  it("skips answer.delta so the list stays quiet", () => {
    const rows = projectEvents([
      {
        seq: 1,
        event: "answer.delta",
        data: { delta: "字" },
      },
      {
        seq: 2,
        event: "answer.done",
        data: { answer: "完整回答", turn: 1 },
      },
    ]);
    expect(rows.some((row) => row.event === "answer.delta")).toBe(false);
    expect(rows.filter((row) => row.kind === "assistant")).toHaveLength(1);
  });

  it("folds task.step_done into the task card plan_done", () => {
    const rows = projectEvents([
      {
        seq: 1,
        event: "task.planned",
        data: {
          type: "agent",
          goal: "三步任务",
          plan: ["一", "二", "三"],
          plan_done: [false, false, false],
        },
      },
      { seq: 2, event: "task.step_done", data: { index: 0 } },
      { seq: 3, event: "task.step_done", data: { index: 1 } },
      { seq: 4, event: "task.step_done", data: { index: 2 } },
    ]);
    expect(rows.some((row) => row.event === "task.step_done")).toBe(false);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe("task");
    expect(rows[0]?.payload.plan_done).toEqual([true, true, true]);
  });

  it("maps llm.call to a muted other record with purpose title", () => {
    const rows = projectEvents([
      {
        seq: 1,
        event: "llm.call",
        data: {
          purpose: "rename",
          prompt_preview: "为对话生成不超过 12 字标题……",
          response_preview: "北京天气查询",
          elapsed_ms: 8,
          turn: 1,
          usage: { prompt_tokens: 24, completion_tokens: 165, total_tokens: 189 },
        },
      },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe("other");
    expect(rows[0]?.event).toBe("llm.call");
    expect(rows[0]?.title).toBe("内部调用：标题生成");
    expect(rows[0]?.input).toBe("为对话生成不超过 12 字标题……");
    expect(rows[0]?.output).toBe("北京天气查询");
    expect(rows[0]?.elapsedMs).toBe(8);
  });

  it("maps rag.empty to other kind with fallback title", () => {
    const rows = projectEvents([
      {
        seq: 1,
        event: "rag.empty",
        data: { query: "今年最新的汇率是多少", turn: 1 },
      },
    ]);
    expect(rows[0]?.kind).toBe("other");
    expect(rows[0]?.title).toBe("未检索到资料，由模型自身作答");
  });
});

describe("belongsToSession", () => {
  it("drops other sessions and keeps own or missing sid", () => {
    expect(belongsToSession({ session_id: "b" }, "a")).toBe(false);
    expect(belongsToSession({ session_id: "a" }, "a")).toBe(true);
    expect(belongsToSession({}, "a")).toBe(true);
    expect(belongsToSession({ session_id: "" }, "a")).toBe(true);
  });

  it("handleSse path leaves records unchanged for a foreign session", () => {
    const list = [rec({ id: "u", kind: "user", title: "用户", turn: 1 })];
    const data = { session_id: "other", model: "leak" };
    const next = belongsToSession(data, "mine")
      ? upsertLiveEvent(list, "run.start", data)
      : list;
    expect(next).toBe(list);
  });
});

describe("orderRecordsForDisplay", () => {
  it("pins task card then route as the first records of each turn", () => {
    const ordered = orderRecordsForDisplay([
      rec({ id: "u", kind: "user", title: "用户", turn: 1 }),
      rec({ id: "r", kind: "route", title: "意图", turn: 1, event: "intent.detected" }),
      rec({ id: "t", kind: "task", title: "任务卡", turn: 1, event: "task.planned" }),
      rec({ id: "a", kind: "assistant", title: "回答", turn: 1 }),
    ]);
    expect(ordered.map((r) => r.id)).toEqual(["t", "r", "u", "a"]);
  });
});
