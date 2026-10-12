import { describe, expect, it } from "vitest";
import { formatCacheHitRate, formatUsageTokens } from "./format";

describe("formatCacheHitRate", () => {
  it("shows 100% only for true full hit", () => {
    expect(formatCacheHitRate(1)).toBe("100%");
    expect(formatCacheHitRate(0.995)).toBe("99.5%");
    expect(formatCacheHitRate(0.999)).toBe("99.9%");
    expect(formatCacheHitRate(0)).toBe("0%");
  });
});

describe("formatUsageTokens", () => {
  it("appends completion up and prompt down when both present", () => {
    expect(
      formatUsageTokens({ prompt_tokens: 24, completion_tokens: 165 }),
    ).toBe(" · tokens 165↑/24↓");
  });

  it("returns empty when usage is missing", () => {
    expect(formatUsageTokens(undefined)).toBe("");
    expect(formatUsageTokens(null)).toBe("");
    expect(formatUsageTokens({ prompt_tokens: 24 })).toBe("");
  });
});
