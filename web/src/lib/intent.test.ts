import { describe, expect, it } from "vitest";
import { badgeLabel, routeTitle } from "./intent";

describe("intent labels", () => {
  it("formats qa settlement badge", () => {
    expect(
      badgeLabel({
        mode: "qa",
        confidence: 0.9,
        source: "rule",
        reason: "短句闲聊",
        turn: 1,
      }),
    ).toBe("直接问答 · 未调用工具");
  });

  it("formats agent auto badge", () => {
    expect(
      badgeLabel({
        mode: "agent",
        confidence: 0.7,
        source: "llm",
        reason: "LLM 分类",
        turn: 2,
      }),
    ).toBe("自动识别：工具调用 · LLM 分类 · 点击纠正");
  });

  it("formats writer route title with keyword", () => {
    expect(
      routeTitle({
        mode: "writer",
        source: "rule",
        reason: "写一篇",
      }),
    ).toBe('意图：长文写作 ← 规则命中"写一篇"');
  });
});
