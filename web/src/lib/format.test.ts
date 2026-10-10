import { describe, expect, it } from "vitest";
import { formatCacheHitRate } from "./format";

describe("formatCacheHitRate", () => {
  it("shows 100% only for true full hit", () => {
    expect(formatCacheHitRate(1)).toBe("100%");
    expect(formatCacheHitRate(0.995)).toBe("99.5%");
    expect(formatCacheHitRate(0.999)).toBe("99.9%");
    expect(formatCacheHitRate(0)).toBe("0%");
  });
});
