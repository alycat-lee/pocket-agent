import type { EventsPage, PluginInfo, SessionInfo } from "./types";
import type { SchemaField } from "../lib/schemaForm";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type") && init?.body != null) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`.trim();
    try {
      const body = (await res.json()) as { error?: string; detail?: unknown };
      if (typeof body.error === "string") detail = body.error;
      else if (body.detail != null) detail = JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export function fetchSessions(): Promise<SessionInfo[]> {
  return json("/sessions");
}

export function fetchConfigSchema(): Promise<SchemaField[]> {
  return json("/api/config/schema");
}

export function fetchConfig(): Promise<Record<string, unknown>> {
  return json("/api/config");
}

export function putConfig(
  body: Record<string, unknown>,
): Promise<{ ok: boolean; config: Record<string, unknown> }> {
  return json("/api/config", { method: "PUT", body: JSON.stringify(body) });
}

export function resetConfig(): Promise<{
  ok: boolean;
  config: Record<string, unknown>;
}> {
  return json("/api/config/reset", { method: "POST", body: "{}" });
}

export function fetchPlugins(): Promise<PluginInfo[]> {
  return json("/api/plugins");
}

export function enablePlugin(
  name: string,
): Promise<{ ok: boolean; error?: string; status?: string }> {
  return json(`/api/plugins/${encodeURIComponent(name)}/enable`, {
    method: "POST",
  });
}

export function disablePlugin(
  name: string,
): Promise<{ ok: boolean; error?: string; status?: string }> {
  return json(`/api/plugins/${encodeURIComponent(name)}/disable`, {
    method: "POST",
  });
}

export function fetchEvents(
  sessionId: string,
  before?: number | null,
  limit = 50,
): Promise<EventsPage> {
  const q = new URLSearchParams({ limit: String(limit) });
  if (before != null) q.set("before", String(before));
  return json(
    `/api/sessions/${encodeURIComponent(sessionId)}/events?${q.toString()}`,
  );
}

export function exportSession(sessionId: string): Promise<Record<string, unknown>> {
  return json(`/api/sessions/${encodeURIComponent(sessionId)}/export`);
}

export function stopRun(runId: string): Promise<{ ok: boolean }> {
  return json(`/api/chat/${encodeURIComponent(runId)}/stop`, { method: "POST" });
}

export function approve(
  approvalId: string,
  approved: boolean,
): Promise<{ ok: boolean }> {
  return json("/approve", {
    method: "POST",
    body: JSON.stringify({ approval_id: approvalId, approved }),
  });
}

export function fetchHealth(): Promise<{
  ok: boolean;
  llm: boolean;
  base_url?: string;
  message?: string;
  hint?: string;
}> {
  return json("/api/health");
}

export function fetchHealthLlm(): Promise<{
  ok: boolean;
  base_url: string;
  latency_ms: number;
  error: string;
}> {
  return json("/api/health/llm");
}

export function testProvider(body: {
  base_url: string;
  api_key?: string;
  model?: string;
  timeout?: number;
}): Promise<{
  ok: boolean;
  connect_ms: number;
  generate_ms: number;
  model_list: string[];
  error: string;
}> {
  return json("/api/providers/test", { method: "POST", body: JSON.stringify(body) });
}

/** EventSource 消费 SSE；overrides 以 query JSON 传递。 */
export function openChatEventSource(opts: {
  message: string;
  sessionId: string;
  runId: string;
  overrides?: Record<string, unknown>;
}): EventSource {
  const q = new URLSearchParams({
    message: opts.message,
    session_id: opts.sessionId,
    run_id: opts.runId,
  });
  if (opts.overrides && Object.keys(opts.overrides).length > 0) {
    q.set("overrides", JSON.stringify(opts.overrides));
  }
  return new EventSource(`/chat?${q.toString()}`);
}
