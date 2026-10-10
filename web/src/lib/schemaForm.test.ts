import { describe, expect, it } from "vitest";
import {
  buildFormControls,
  coerceFormValue,
  groupControls,
  type SchemaField,
} from "./schemaForm";

const MOCK_SCHEMA: SchemaField[] = [
  {
    key: "base_url",
    type: "string",
    group: "模型",
    description: "LLM 地址",
  },
  {
    key: "api_key",
    type: "string",
    group: "模型",
    description: "密钥",
    secret: true,
  },
  {
    key: "temperature",
    type: "number",
    group: "模型",
    description: "温度",
    min: 0,
    max: 2,
  },
  {
    key: "max_steps",
    type: "integer",
    group: "护栏",
    description: "最大步数",
    min: 1,
    max: 100,
  },
  {
    key: "use_llm_summarize",
    type: "boolean",
    group: "上下文预算",
    description: "LLM 摘要",
  },
  {
    key: "route_mode",
    type: "string",
    group: "路由",
    description: "路由模式",
    enum: ["local", "cloud"],
  },
];

describe("buildFormControls", () => {
  it("renders controls from schema without hardcoded keys", () => {
    const controls = buildFormControls(MOCK_SCHEMA);
    expect(controls.map((c) => c.key)).toEqual([
      "base_url",
      "api_key",
      "temperature",
      "max_steps",
      "use_llm_summarize",
      "route_mode",
    ]);
    expect(controls.find((c) => c.key === "api_key")?.kind).toBe("password");
    expect(controls.find((c) => c.key === "temperature")?.kind).toBe("number");
    expect(controls.find((c) => c.key === "max_steps")?.step).toBe(1);
    expect(controls.find((c) => c.key === "use_llm_summarize")?.kind).toBe(
      "boolean",
    );
    expect(controls.find((c) => c.key === "route_mode")?.kind).toBe("select");
    expect(controls.find((c) => c.key === "route_mode")?.options).toEqual([
      "local",
      "cloud",
    ]);
  });

  it("groups by schema group order", () => {
    const groups = groupControls(buildFormControls(MOCK_SCHEMA));
    expect(groups.map((g) => g.group)).toEqual([
      "模型",
      "护栏",
      "上下文预算",
      "路由",
    ]);
    expect(groups[0].controls).toHaveLength(3);
  });

  it("coerces values by control kind", () => {
    const maxSteps = buildFormControls(MOCK_SCHEMA).find(
      (c) => c.key === "max_steps",
    )!;
    expect(coerceFormValue(maxSteps, "12")).toBe(12);
    const flag = buildFormControls(MOCK_SCHEMA).find(
      (c) => c.key === "use_llm_summarize",
    )!;
    expect(coerceFormValue(flag, true)).toBe(true);
  });
});
