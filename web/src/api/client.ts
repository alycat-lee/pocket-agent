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

export async function stopRun(
  runId: string,
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`/api/chat/${encodeURIComponent(runId)}/stop`, {
    method: "POST",
  });
  if (res.status === 404) {
    return { ok: false, error: "run not found" };
  }
  if (!res.ok) {
    return { ok: false, error: `${res.status} ${res.statusText}`.trim() };
  }
  return res.json() as Promise<{ ok: boolean; error?: string }>;
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

export interface ProviderPreset {
  id: string;
  label: string;
  base_url?: string;
  requires_key?: boolean;
  trust_env?: boolean;
  thinking?: string;
}

export function fetchProviderPresets(): Promise<{ presets: ProviderPreset[] }> {
  return json("/api/providers/presets");
}

export function fetchProviders(): Promise<{
  providers: Array<{
    id: string;
    name: string;
    type: string;
    base_url: string;
    api_key: string;
    model: string;
    sampling: Record<string, unknown> | null;
    thinking?: string | null;
    trust_env?: boolean | null;
  }>;
  default_provider_id: string;
  routing: { choose_fill: string; generate: string };
}> {
  return json("/api/providers");
}

export function createProvider(body: {
  name: string;
  type: string;
  base_url: string;
  api_key?: string;
  model?: string;
  sampling?: Record<string, unknown> | null;
  thinking?: string;
  trust_env?: boolean;
}): Promise<{ ok: boolean; providers: unknown[] }> {
  return json("/api/providers", { method: "POST", body: JSON.stringify(body) });
}

export function updateProvider(
  id: string,
  body: Partial<{
    name: string;
    type: string;
    base_url: string;
    api_key: string;
    model: string;
    sampling: Record<string, unknown> | null;
    thinking: string;
    trust_env: boolean;
  }>,
): Promise<{ ok: boolean; providers: unknown[] }> {
  return json(`/api/providers/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export function deleteProvider(id: string): Promise<{ ok: boolean; providers: unknown[] }> {
  return json(`/api/providers/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function setDefaultProvider(id: string): Promise<{ ok: boolean; default_provider_id: string }> {
  return json(`/api/providers/${encodeURIComponent(id)}/default`, { method: "POST" });
}

export function updateRouting(body: {
  choose_fill: string;
  generate: string;
}): Promise<{ ok: boolean; routing: { choose_fill: string; generate: string } }> {
  return json("/api/routing", { method: "PUT", body: JSON.stringify(body) });
}

export function deleteSession(
  sessionId: string,
): Promise<{ ok: boolean; session_id: string }> {
  return json(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

export function fetchWorkspaces(): Promise<{
  available: boolean;
  workspaces: Array<{ id: string; name: string; root: string }>;
}> {
  return json("/api/workspaces");
}

export function createWorkspace(body: {
  name: string;
  root: string;
}): Promise<{ ok: boolean; workspace: { id: string; name: string; root: string } }> {
  return json("/api/workspaces", { method: "POST", body: JSON.stringify(body) });
}

export function bindSessionWorkspace(
  sessionId: string,
  workspaceId: string,
): Promise<{ ok: boolean; session_id: string; workspace_id: string }> {
  return json(`/api/sessions/${encodeURIComponent(sessionId)}/workspace`, {
    method: "PUT",
    body: JSON.stringify({ workspace_id: workspaceId }),
  });
}

/** EventSource 消费 SSE；overrides 以 query JSON 传递。 */
export function openChatEventSource(opts: {
  message: string;
  sessionId: string;
  runId: string;
  overrides?: Record<string, unknown>;
  workspaceId?: string | null;
  permission?: string;
}): EventSource {
  const q = new URLSearchParams({
    message: opts.message,
    session_id: opts.sessionId,
    run_id: opts.runId,
    permission: opts.permission || "workspace",
  });
  if (opts.workspaceId) q.set("workspace_id", opts.workspaceId);
  if (opts.overrides && Object.keys(opts.overrides).length > 0) {
    q.set("overrides", JSON.stringify(opts.overrides));
  }
  return new EventSource(`/chat?${q.toString()}`);
}
