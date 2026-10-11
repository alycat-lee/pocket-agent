import { describe, expect, it } from "vitest";
import type { TrajectoryRecord } from "../api/types";
import { orderRecordsForDisplay, projectEvents } from "./trajectory";

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
