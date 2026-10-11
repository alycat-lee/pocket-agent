import { describe, expect, it } from "vitest";
import type { ChatMessage } from "../api/types";
import { projectEvents } from "./trajectory";
import {
  applyAnswerDelta,
  applyAnswerDone,
  markStreamError,
  streamMessageId,
} from "./streaming";

describe("streaming bubble", () => {
  it("grows through deltas then replaces with the final answer", () => {
    const runId = "run-1";
    let messages: ChatMessage[] = [
      { id: "u1", role: "user", content: "你是谁" },
    ];
    messages = applyAnswerDelta(messages, runId, "我");
    expect(messages.at(-1)?.content).toBe("我");
    expect(messages.at(-1)?.id).toBe(streamMessageId(runId));
    messages = applyAnswerDelta(messages, runId, "是");
    expect(messages.at(-1)?.content).toBe("我是");
    messages = applyAnswerDelta(messages, runId, "助手");
    expect(messages.at(-1)?.content).toBe("我是助手");
    messages = applyAnswerDone(messages, runId, "我是 pocket-agent 的助手", {
      settlement: { total_steps: 0 },
      turnId: "turn-1",
    });
    expect(messages.filter((m) => m.role === "assistant")).toHaveLength(1);
    expect(messages.at(-1)?.content).toBe("我是 pocket-agent 的助手");
    expect(messages.at(-1)?.settlement?.total_steps).toBe(0);
    expect(messages.at(-1)?.turnId).toBe("turn-1");
  });

  it("keeps partial stream text and marks error", () => {
    const runId = "run-2";
    let messages: ChatMessage[] = applyAnswerDelta([], runId, "已写出");
    messages = markStreamError(messages, runId);
    expect(messages[0]?.content).toBe("已写出");
    expect(messages[0]?.error).toBe(true);
  });
});

describe("trajectory skips answer.delta", () => {
  it("does not project answer.delta into the list", () => {
    const rows = projectEvents([
      { seq: 1, event: "answer.delta", data: { delta: "我" } },
      { seq: 2, event: "answer.delta", data: { delta: "是" } },
      {
        seq: 3,
        event: "answer.done",
        data: { answer: "我是助手", turn: 1 },
      },
    ]);
    expect(rows.every((row) => row.event !== "answer.delta")).toBe(true);
    expect(rows.filter((row) => row.event === "answer.done")).toHaveLength(1);
  });
});
