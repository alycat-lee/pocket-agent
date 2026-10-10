export type SessionView = "chat" | "trajectory";

export type ChatMode = "qa" | "tools" | "writer";
export type IntentMode = "qa" | "agent" | "writer";
export type PermissionMode = "readonly" | "confirm_write" | "auto";

export interface SessionInfo {
  session_id: string;
  title?: string;
  turn_count?: number;
  updated_at?: string;
}

export interface Settlement {
  total_steps: number;
  llm_ms?: number;
  tool_ms?: number;
  cache_hit_rate?: number;
  real_calls?: number;
  intercepts?: number;
  prompt_tokens_max?: number;
  elapsed_ms?: number;
  stop_reason?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  settlement?: Settlement;
  turnId?: string;
  error?: boolean;
}

export interface TrajectoryRecord {
  id: string;
  seq: number;
  event: string;
  turn: number;
  stepIndex: number | null;
  kind: "user" | "assistant" | "tool" | "intercept" | "compress" | "approval" | "other";
  title: string;
  ts: number | null;
  elapsedMs: number | null;
  ttftMs: number | null;
  promptTokensEst: number | null;
  currentTool: string | null;
  badges: Array<"cache_hit" | "intercepted" | "tool_error">;
  errorCode: string | null;
  inProgress: boolean;
  payload: Record<string, unknown>;
  input?: string;
  output?: string;
  args?: unknown;
}

export interface PluginInfo {
  name: string;
  requires: string[];
  provides: string[];
  enabled: boolean;
  active: boolean;
}

export interface EventsPage {
  events: Array<{
    seq?: number;
    id?: number;
    event: string;
    ts?: number;
    data?: Record<string, unknown>;
    payload?: Record<string, unknown>;
  }>;
  next_before: number | null;
}

export interface StatusSnapshot {
  running: boolean;
  phase: string;
  stepIndex: number;
  maxSteps: number;
  elapsedSec: number;
  promptTokensEst: number;
  promptCharBudget: number;
  currentTool: string | null;
  timeline: Array<{
    id: string;
    label: string;
    kind: string;
    active: boolean;
  }>;
  settlement: Settlement | null;
  settlementUntil: number | null;
}

export interface ApprovalRequest {
  approval_id: string;
  tool?: string;
  args?: unknown;
  reason?: string;
  expiresAt: number;
}
