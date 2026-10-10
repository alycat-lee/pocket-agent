import { describe, expect, it } from "vitest";
import {
  buildFormControls,
  groupControls,
  type SchemaField,
} from "./schemaForm";

const MOCK_SCHEMA: SchemaField[] = [
  {
    key: "base_url",
    type: "string",
    group: "模型",
    description: "LLM 地址",
    deprecated: true,
  },
  {
    key: "temperature",
    type: "number",
    group: "模型",
    description: "温度",
    min: 0,
    max: 2,
    deprecated: true,
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

describe("buildFormControls filters deprecated", () => {
  it("includes all controls when not filtered", () => {
    const controls = buildFormControls(MOCK_SCHEMA);
    expect(controls.map((c) => c.key)).toEqual([
      "base_url",
      "temperature",
      "max_steps",
      "use_llm_summarize",
      "route_mode",
    ]);
  });

  it("filters deprecated fields", () => {
    const controls = buildFormControls(MOCK_SCHEMA);
    const filtered = controls.filter(
      (c) => !MOCK_SCHEMA.find((s) => s.key === c.key)?.deprecated,
    );
    expect(filtered.map((c) => c.key)).toEqual([
      "max_steps",
      "use_llm_summarize",
      "route_mode",
    ]);
  });
});

describe("groupControls", () => {
  it("groups non-deprecated only", () => {
    const controls = buildFormControls(MOCK_SCHEMA);
    const filtered = controls.filter(
      (c) => !MOCK_SCHEMA.find((s) => s.key === c.key)?.deprecated,
    );
    const groups = groupControls(filtered);
    expect(groups.map((g) => g.group)).toEqual([
      "护栏",
      "上下文预算",
      "路由",
    ]);
  });
});
