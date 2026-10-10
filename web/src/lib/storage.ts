import type { ChatMode, PermissionMode, SessionView } from "../api/types";

const VIEW_KEY = "pa.sessionView.";
const STATUS_EXPANDED = "pa.statusExpanded";
const MODE_KEY = "pa.mode.";
const PERM_KEY = "pa.perm.";

export function getSessionView(sessionId: string): SessionView {
  const v = localStorage.getItem(VIEW_KEY + sessionId);
  return v === "trajectory" ? "trajectory" : "chat";
}

export function setSessionView(sessionId: string, view: SessionView): void {
  localStorage.setItem(VIEW_KEY + sessionId, view);
}

export function getStatusExpanded(): boolean {
  return localStorage.getItem(STATUS_EXPANDED) === "1";
}

export function setStatusExpanded(v: boolean): void {
  localStorage.setItem(STATUS_EXPANDED, v ? "1" : "0");
}

export function getMode(sessionId: string): ChatMode {
  const v = localStorage.getItem(MODE_KEY + sessionId);
  if (v === "tools" || v === "writer" || v === "qa") return v;
  return "tools";
}

export function setMode(sessionId: string, mode: ChatMode): void {
  localStorage.setItem(MODE_KEY + sessionId, mode);
}

export function getPermission(sessionId: string): PermissionMode {
  const v = localStorage.getItem(PERM_KEY + sessionId);
  if (v === "readonly" || v === "confirm_write" || v === "auto") return v;
  return "confirm_write";
}

export function setPermission(sessionId: string, mode: PermissionMode): void {
  localStorage.setItem(PERM_KEY + sessionId, mode);
}

/** 模式/权限 → 单次 overrides */
export function buildOverrides(
  mode: ChatMode,
  permission: PermissionMode,
): Record<string, unknown> {
  const overrides: Record<string, unknown> = {};
  if (mode === "qa") {
    // 问答面：收紧步数，偏向直接回答
    overrides.max_steps = 1;
  } else if (mode === "writer") {
    overrides.max_steps = 12;
  }
  if (permission === "auto") {
    // 自动放行：前端侧仅作标记；实际审批由 gate 控制
    overrides._permission = "auto";
  } else if (permission === "readonly") {
    overrides._permission = "readonly";
  } else {
    overrides._permission = "confirm_write";
  }
  return overrides;
}
