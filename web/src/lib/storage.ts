import type { ChatMode, PermissionMode, SessionView } from "../api/types";

const VIEW_KEY = "pa.sessionView.";
const STATUS_EXPANDED = "pa.statusExpanded";
const MODE_KEY = "pa.mode.";
const PERM_KEY = "pa.perm.";
const AGENT_KEY = "pa.agent";

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
  if (v === "readonly" || v === "workspace" || v === "full") return v;
  return "workspace";
}

export function setPermission(sessionId: string, mode: PermissionMode): void {
  localStorage.setItem(PERM_KEY + sessionId, mode);
}

export function getActiveAgent(): string {
  return localStorage.getItem(AGENT_KEY) || "default";
}

export function setActiveAgent(name: string): void {
  localStorage.setItem(AGENT_KEY, name);
}

/** 权限 → 单次 overrides */
export function buildOverrides(
  permission: PermissionMode,
): Record<string, unknown> {
  const overrides: Record<string, unknown> = {};
  overrides._permission = permission;
  return overrides;
}
