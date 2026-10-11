import type { IntentMode, IntentSource, TrajectoryRecord } from "../api/types";
import { routeTitle } from "./intent";

type RawEvent = {
  seq: number;
  event: string;
  ts?: number;
  data: Record<string, unknown>;
};

function kindOf(event: string, data: Record<string, unknown>): TrajectoryRecord["kind"] {
  if (event === "user.message" || data.role === "user") return "user";
  if (event === "answer.done") return "assistant";
  if (event === "step.tool_call") return "tool";
  if (event === "step.intercepted") return "intercept";
  if (event === "step.compress" || event === "session.compress") return "compress";
  if (event === "approval.required") return "approval";
  if (event === "intent.detected") return "route";
  if (event === "step.thinking") return "thinking";
  if (event === "step.start" || event === "step.cache_hit") return "tool";
  return "other";
}

function titleOf(
  event: string,
  data: Record<string, unknown>,
  kind: TrajectoryRecord["kind"],
): string {
  if (kind === "user") return String(data.message || data.query || "用户消息");
  if (kind === "assistant") return "助手回答";
  if (event === "error" || event === "step.error") {
    const msg = String(data.message || data.error || "运行错误");
    if (/502|Bad Gateway/i.test(msg)) return "LLM 502 Bad Gateway";
    if (/503|504|server error\s*5/i.test(msg)) return `LLM 服务错误`;
    return msg.length > 80 ? `${msg.slice(0, 80)}…` : msg;
  }
  if (data.tool_name === "(format_error)" || data.tool === "(format_error)") {
    return `格式错误 ${data.error || data.reason || ""}`.trim();
  }
  if (event === "step.cache_hit") {
    return `缓存命中 ${data.tool || data.current_tool || ""}`.trim();
  }
  if (kind === "tool") {
    const name = String(data.tool || data.current_tool || data.name || "").trim();
    return name ? `工具 ${name}` : "工具步骤";
  }
  if (kind === "intercept") {
    return `拦截 ${data.reason || ""}`.trim();
  }
  if (kind === "compress") return "上下文压缩";
  if (kind === "approval") return `审批 ${data.tool || ""}`.trim();
  if (kind === "thinking") return "思维链";
  if (kind === "route") {
    const mode = String(data.mode || "agent") as IntentMode;
    const source = String(data.source || "fallback") as IntentSource;
    const reason = String(data.reason || "");
    return routeTitle({ mode, source, reason });
  }
  if (event === "step.start") return `步骤 ${data.step_index ?? ""}`.trim();
  return event;
}

export function projectEvents(raw: RawEvent[]): TrajectoryRecord[] {
  let turn = 0;
  const out: TrajectoryRecord[] = [];
  for (const ev of raw) {
    const data = ev.data || {};
    if (ev.event === "user.message" || data.role === "user") {
      turn += 1;
    } else if (ev.event === "answer.done") {
      // turn 已在本轮用户消息时递增
    } else if (typeof data.turn === "number") {
      turn = Math.max(turn, Number(data.turn));
    }
    const kind = kindOf(ev.event, data);
    const badges: TrajectoryRecord["badges"] = [];
    if (ev.event === "step.cache_hit" || data.from_cache === true) {
      badges.push("cache_hit");
    }
    if (ev.event === "step.intercepted") {
      badges.push("intercepted");
    }
    const err =
      typeof data.error === "string"
        ? data.error
        : typeof data.error_code === "string"
          ? data.error_code
          : null;
    if (err || data.ok === false) {
      badges.push("tool_error");
    }
    const elapsed =
      typeof data.elapsed_ms === "number" ? (data.elapsed_ms as number) : null;
    const inProgress = Boolean(data.in_progress) || (ev.event === "step.start" && elapsed == null && !data.result);
    out.push({
      id: `${ev.seq}-${ev.event}`,
      seq: ev.seq,
      event: ev.event,
      turn: typeof data.turn === "number" ? Number(data.turn) : Math.max(turn, 1),
      stepIndex:
        typeof data.step_index === "number" ? Number(data.step_index) : null,
      kind,
      title: titleOf(ev.event, data, kind),
      ts: typeof ev.ts === "number" ? ev.ts : typeof data.ts === "number" ? Number(data.ts) : null,
      elapsedMs: inProgress ? null : elapsed,
      ttftMs: typeof data.ttft_ms === "number" ? Number(data.ttft_ms) : null,
      promptTokensEst:
        typeof data.prompt_tokens_est === "number"
          ? Number(data.prompt_tokens_est)
          : null,
      currentTool:
        typeof data.current_tool === "string"
          ? data.current_tool
          : typeof data.tool === "string"
            ? data.tool
            : null,
      badges,
      errorCode: err,
      inProgress,
      payload: data,
      input:
        typeof data.prompt === "string"
          ? data.prompt
          : typeof data.input === "string"
            ? data.input
            : typeof data.query === "string"
              ? data.query
              : undefined,
      output:
        ev.event === "step.thinking" && typeof data.text === "string"
          ? data.text
          : typeof data.result === "string"
            ? data.result
            : typeof data.answer === "string"
              ? data.answer
              : typeof data.output === "string"
                ? data.output
                : typeof data.text === "string"
                  ? data.text
                  : undefined,
      args: data.args,
    });
  }
  return out;
}

/** 将实时 SSE 事件并入轨迹（按 seq 去重）。 */
export function upsertLiveEvent(
  list: TrajectoryRecord[],
  event: string,
  data: Record<string, unknown>,
  seqHint?: number,
): TrajectoryRecord[] {
  const seq =
    typeof data.seq === "number"
      ? Number(data.seq)
      : seqHint ?? (list.length ? list[list.length - 1].seq + 1 : 1);
  const projected = projectEvents([{ seq, event, data, ts: Date.now() / 1000 }]);
  const next = projected[0];
  if (!next) return list;
  const idx = list.findIndex((r) => r.seq === next.seq && r.event === next.event);
  if (idx >= 0) {
    const copy = list.slice();
    copy[idx] = next;
    return copy;
  }
  return [...list, next];
}

/** 每轮把 route 记录排到该轮第一条，其余保持原序。 */
export function orderRecordsForDisplay(
  records: TrajectoryRecord[],
): TrajectoryRecord[] {
  const groups = new Map<number, TrajectoryRecord[]>();
  const order: number[] = [];
  for (const record of records) {
    if (!groups.has(record.turn)) {
      groups.set(record.turn, []);
      order.push(record.turn);
    }
    groups.get(record.turn)?.push(record);
  }
  const out: TrajectoryRecord[] = [];
  for (const turn of order) {
    const group = groups.get(turn) ?? [];
    out.push(
      ...group.filter((r) => r.kind === "route"),
      ...group.filter((r) => r.kind !== "route"),
    );
  }
  return out;
}
