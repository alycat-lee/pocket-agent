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
  it("pins route as the first record of each turn", () => {
    const ordered = orderRecordsForDisplay([
      rec({ id: "u", kind: "user", title: "用户", turn: 1 }),
      rec({ id: "r", kind: "route", title: "意图", turn: 1, event: "intent.detected" }),
      rec({ id: "a", kind: "assistant", title: "回答", turn: 1 }),
    ]);
    expect(ordered.map((r) => r.id)).toEqual(["r", "u", "a"]);
  });
});
