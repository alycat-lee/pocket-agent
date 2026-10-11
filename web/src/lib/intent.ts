import type { IntentInfo, IntentMode, IntentSource } from "../api/types";

export type { IntentInfo, IntentSource };

export const MODE_LABEL: Record<IntentMode, string> = {
  qa: "直接问答",
  agent: "工具调用",
  writer: "长文写作",
};

export const SOURCE_LABEL: Record<IntentSource, string> = {
  rule: "规则命中",
  llm: "LLM 分类",
  fallback: "回退默认",
  manual: "手动纠正",
};

export function sourcePhrase(source: IntentSource, reason: string): string {
  if (source === "rule" && reason && reason !== "短句闲聊") {
    return `${SOURCE_LABEL.rule}"${reason}"`;
  }
  if (source === "rule") return SOURCE_LABEL.rule;
  if (reason) return reason;
  return SOURCE_LABEL[source];
}

export function routeTitle(info: Pick<IntentInfo, "mode" | "source" | "reason">): string {
  return `意图：${MODE_LABEL[info.mode]} ← ${sourcePhrase(info.source, info.reason)}`;
}

export function badgeLabel(info: IntentInfo): string {
  if (info.mode === "qa") {
    return "直接问答 · 未调用工具";
  }
  const prefix = info.source === "manual" ? "手动纠正" : "自动识别";
  return `${prefix}：${MODE_LABEL[info.mode]} · ${SOURCE_LABEL[info.source]} · 点击纠正`;
}
