import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  approve,
  exportSession,
  fetchConfig,
  fetchEvents,
  fetchHealthLlm,
  fetchSessions,
  openChatEventSource,
  stopRun,
} from "./api/client";
import type {
  ApprovalRequest,
  ChatMessage,
  ChatMode,
  PermissionMode,
  SessionInfo,
  SessionView,
  Settlement,
  StatusSnapshot,
  TrajectoryRecord,
} from "./api/types";
import { ApprovalDialog } from "./components/ApprovalDialog";
import { ChatView } from "./components/ChatView";
import { Composer } from "./components/Composer";
import { GuidePage } from "./components/GuidePage";
import { PluginsPage } from "./components/PluginsPage";
import { SettingsPage } from "./components/SettingsPage";
import { Sidebar, type PageId } from "./components/Sidebar";
import { StatusBar } from "./components/StatusBar";
import { TrajectoryView } from "./components/TrajectoryView";
import {
  buildOverrides,
  getMode,
  getPermission,
  getSessionView,
  getStatusExpanded,
  setMode as persistMode,
  setPermission as persistPermission,
  setSessionView,
  setStatusExpanded,
} from "./lib/storage";
import { projectEvents, upsertLiveEvent } from "./lib/trajectory";

function newId(): string {
  return crypto.randomUUID();
}

const EMPTY_STATUS: StatusSnapshot = {
  running: false,
  phase: "",
  stepIndex: 0,
  maxSteps: 0,
  elapsedSec: 0,
  promptTokensEst: 0,
  promptCharBudget: 5200,
  currentTool: null,
  timeline: [],
  settlement: null,
  settlementUntil: null,
};

export default function App() {
  const [page, setPage] = useState<PageId>("session");
  const [sessions, setSessions] = useState<SessionInfo[]>([
    { session_id: "default", title: "default" },
  ]);
  const [sessionId, setSessionId] = useState("default");
  const [view, setView] = useState<SessionView>(() => getSessionView("default"));
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [records, setRecords] = useState<TrajectoryRecord[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [nextBefore, setNextBefore] = useState<number | null>(null);
  const [loadingEarlier, setLoadingEarlier] = useState(false);
  const [followTail, setFollowTail] = useState(true);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<ChatMode>(() => getMode("default"));
  const [permission, setPermission] = useState<PermissionMode>(() =>
    getPermission("default"),
  );
  const [status, setStatus] = useState<StatusSnapshot>(EMPTY_STATUS);
  const [statusExpanded, setStatusExpandedState] = useState(getStatusExpanded);
  const [approval, setApproval] = useState<ApprovalRequest | null>(null);
  const [softError, setSoftError] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const [offlineDetail, setOfflineDetail] = useState<string | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | undefined>();
  const [promptBudget, setPromptBudget] = useState(5200);
  const [runId, setRunId] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const esGenRef = useRef(0);
  const tickRef = useRef<number | null>(null);
  const runStarted = useRef<number>(0);
  const turnCounter = useRef(0);

  const reloadSessions = useCallback(async () => {
    try {
      const rows = await fetchSessions();
      if (rows.length) setSessions(rows);
    } catch {
      /* ignore */
    }
  }, []);

  const checkHealth = useCallback(async () => {
    try {
      const h = await fetchHealthLlm();
      setBaseUrl(h.base_url);
      setOffline(!h.ok);
      setOfflineDetail(h.ok ? null : h.error || "LLM 不可用");
      return h.ok;
    } catch (e) {
      setOffline(true);
      setOfflineDetail(e instanceof Error ? e.message : String(e));
      return false;
    }
  }, []);

  const loadTrajectory = useCallback(async (sid: string) => {
    const page = await fetchEvents(sid, null, 50);
    const projected = projectEvents(
      page.events.map((e) => ({
        seq: Number(e.id ?? e.seq ?? 0),
        event: e.event,
        ts: e.ts,
        data: (e.data ?? e.payload ?? {}) as Record<string, unknown>,
      })),
    );
    setRecords(projected);
    setHasMore(page.next_before != null);
    setNextBefore(page.next_before);
    setFollowTail(true);
  }, []);

  const applyConfigToUi = useCallback((c: Record<string, unknown>) => {
    if (typeof c.prompt_char_budget === "number") {
      setPromptBudget(c.prompt_char_budget);
    }
    if (typeof c.base_url === "string") setBaseUrl(c.base_url);
    if (typeof c.max_steps === "number") {
      setStatus((s) => ({ ...s, maxSteps: c.max_steps as number }));
    }
  }, []);

  useEffect(() => {
    void reloadSessions();
    void checkHealth();
    fetchConfig()
      .then((c) => applyConfigToUi(c))
      .catch(() => undefined);
  }, [reloadSessions, checkHealth, applyConfigToUi]);

  useEffect(() => {
    // 切会话：切断上一会话 SSE，避免 EventSource 自动重连再跑一轮
    esGenRef.current += 1;
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (tickRef.current) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
    setSoftError(null);
    setRunId(null);
    setStatus(EMPTY_STATUS);
    setApproval(null);
    setView(getSessionView(sessionId));
    setMode(getMode(sessionId));
    setPermission(getPermission(sessionId));
    setMessages([]);
    setRecords([]);
    setFocusId(null);
    setFollowTail(true);
    void loadTrajectory(sessionId).catch(() => undefined);
  }, [sessionId, loadTrajectory]);

  useEffect(() => {
    if (!status.settlementUntil) return;
    const t = window.setTimeout(() => {
      setStatus((s) =>
        s.settlementUntil && Date.now() >= s.settlementUntil
          ? { ...s, settlement: null, settlementUntil: null }
          : s,
      );
    }, Math.max(0, status.settlementUntil - Date.now()));
    return () => window.clearTimeout(t);
  }, [status.settlementUntil]);

  const closeEs = () => {
    // 递增世代，丢弃关闭/重连带来的迟到 onerror / 事件
    esGenRef.current += 1;
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (tickRef.current) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
  };

  const handleSse = useCallback(
    (event: string, data: Record<string, unknown>) => {
      setRecords((prev) => upsertLiveEvent(prev, event, data));

      if (event === "step.start") {
        const step = Number(data.step_index ?? 0);
        const tool =
          typeof data.current_tool === "string" ? data.current_tool : null;
        setStatus((s) => ({
          ...s,
          running: true,
          phase: tool ? `正在调用 ${tool}` : "正在选择工具…",
          stepIndex: step,
          promptTokensEst: Number(data.prompt_tokens_est ?? s.promptTokensEst),
          currentTool: tool,
          timeline: [
            ...s.timeline.filter((n) => n.id !== `step-${step}`),
            {
              id: `step-${step}`,
              label: `S${step}`,
              kind: "choose",
              active: true,
            },
          ].map((n) => ({ ...n, active: n.id === `step-${step}` })),
        }));
      } else if (event === "step.tool_call") {
        const tool = String(data.tool || data.current_tool || "tool");
        setStatus((s) => ({
          ...s,
          phase: `正在调用 ${tool}`,
          currentTool: tool,
          timeline: [
            ...s.timeline,
            {
              id: `tool-${data.step_index}-${tool}`,
              label: tool,
              kind: "tool",
              active: true,
            },
          ].map((n, i, arr) => ({ ...n, active: i === arr.length - 1 })),
        }));
      } else if (event === "step.cache_hit") {
        setStatus((s) => ({
          ...s,
          phase: `缓存命中 ${data.tool || ""}`.trim(),
          timeline: [
            ...s.timeline,
            {
              id: `cache-${data.step_index}`,
              label: "cache",
              kind: "cache",
              active: true,
            },
          ].map((n, i, arr) => ({ ...n, active: i === arr.length - 1 })),
        }));
      } else if (event === "step.intercepted") {
        setStatus((s) => ({
          ...s,
          phase: `已拦截 ${data.reason || ""}`.trim(),
          timeline: [
            ...s.timeline,
            {
              id: `intercept-${data.step_index}`,
              label: "拦截",
              kind: "intercept",
              active: true,
            },
          ].map((n, i, arr) => ({ ...n, active: i === arr.length - 1 })),
        }));
      } else if (event === "approval.required") {
        const approvalId = String(data.approval_id || "");
        setStatus((s) => ({ ...s, phase: "等待审批…" }));
        setApproval({
          approval_id: approvalId,
          tool: typeof data.tool === "string" ? data.tool : undefined,
          args: data.args,
          reason: typeof data.reason === "string" ? data.reason : undefined,
          expiresAt: Date.now() + 60_000,
        });
      } else if (event === "answer.done") {
        const settlement: Settlement = {
          total_steps: Number(data.total_steps ?? 0),
          llm_ms: Number(data.llm_ms ?? 0),
          tool_ms: Number(data.tool_ms ?? 0),
          cache_hit_rate: Number(data.cache_hit_rate ?? 0),
          real_calls: Number(data.real_calls ?? 0),
          intercepts: Number(data.intercepts ?? 0),
          prompt_tokens_max: Number(data.prompt_tokens_max ?? 0),
          elapsed_ms: Number(data.elapsed_ms ?? 0),
          stop_reason: String(data.stop_reason || ""),
        };
        const turnId = `turn-${turnCounter.current}`;
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            role: "assistant",
            content: String(data.answer || ""),
            settlement,
            turnId,
          },
        ]);
        setStatus((s) => ({
          ...s,
          running: false,
          phase: "",
          settlement,
          settlementUntil: Date.now() + 5000,
          timeline: s.timeline.map((n) => ({ ...n, active: false })),
        }));
        setRunId(null);
        closeEs();
        void reloadSessions();
      } else if (event === "error") {
        const msg = String(data.message || "error");
        const conn =
          /connect|connection|llama|ECONNREFUSED|超时|unreachable|502|503|504|Bad Gateway|server error\s*5\d\d/i.test(
            msg,
          );
        if (conn) {
          setOffline(true);
          setOfflineDetail(msg);
          setSoftError(null);
          setView("chat");
          setSessionView(sessionId, "chat");
        } else {
          setSoftError("运行出错，请查看轨迹视图中的错误记录。");
          setMessages((prev) => [
            ...prev,
            {
              id: newId(),
              role: "assistant",
              content: "本轮未能完成。请切换到轨迹视图检查详情。",
              error: true,
            },
          ]);
        }
        setStatus((s) => ({ ...s, running: false, phase: "" }));
        setRunId(null);
        closeEs();
      }
    },
    [reloadSessions, sessionId],
  );

  const send = async () => {
    const text = draft.trim();
    if (!text || status.running) return;
    const ok = await checkHealth();
    if (!ok) {
      setOffline(true);
      return;
    }
    setOffline(false);
    setSoftError(null);
    turnCounter.current += 1;
    const rid = newId();
    setRunId(rid);
    setDraft("");
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", content: text, turnId: `turn-${turnCounter.current}` },
    ]);
    setRecords((prev) =>
      upsertLiveEvent(prev, "user.message", {
        message: text,
        role: "user",
        turn: turnCounter.current,
      }),
    );
    runStarted.current = Date.now();
    setStatus({
      ...EMPTY_STATUS,
      running: true,
      phase: "正在选择工具…",
      maxSteps: status.maxSteps || 8,
      promptCharBudget: promptBudget,
      timeline: [],
    });
    tickRef.current = window.setInterval(() => {
      setStatus((s) =>
        s.running
          ? {
              ...s,
              elapsedSec: (Date.now() - runStarted.current) / 1000,
            }
          : s,
      );
    }, 250);

    closeEs();
    const gen = ++esGenRef.current;
    const es = openChatEventSource({
      message: text,
      sessionId,
      runId: rid,
      overrides: buildOverrides(mode, permission),
    });
    esRef.current = es;

    const names = [
      "step.start",
      "step.tool_call",
      "step.cache_hit",
      "step.intercepted",
      "answer.done",
      "approval.required",
      "error",
    ];
    for (const name of names) {
      es.addEventListener(name, (ev) => {
        if (gen !== esGenRef.current || esRef.current !== es) return;
        try {
          const payload = JSON.parse((ev as MessageEvent).data) as Record<
            string,
            unknown
          >;
          handleSse(name, payload);
        } catch {
          /* ignore */
        }
      });
    }
    es.onerror = () => {
      // 立即关闭，禁止浏览器对同一 /chat URL 自动重连（会重复跑任务）
      if (esRef.current === es) {
        es.close();
        esRef.current = null;
      }
      if (gen !== esGenRef.current) return;
      if (tickRef.current) {
        window.clearInterval(tickRef.current);
        tickRef.current = null;
      }
      setStatus((s) => {
        if (!s.running) return s;
        void checkHealth().then((llm) => {
          if (gen !== esGenRef.current) return;
          if (!llm) setOffline(true);
          else setSoftError("连接中断，请重试。");
        });
        return { ...s, running: false, phase: "" };
      });
      setRunId(null);
    };
  };

  const segments = useMemo(
    () => [
      { label: "system", chars: 800 },
      { label: "状态摘要", chars: 400 },
      { label: "历史", chars: Math.min(3000, messages.length * 180) },
      { label: "本步信息", chars: 200 },
    ],
    [messages.length],
  );

  const newSession = () => {
    const id = `s-${Date.now().toString(36)}`;
    setSoftError(null);
    setSessions((prev) => [{ session_id: id, title: id }, ...prev]);
    setSessionId(id);
    setPage("session");
  };

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        activeSession={sessionId}
        page={page}
        onSelectSession={setSessionId}
        onNewSession={newSession}
        onPage={setPage}
      />
      <main className="main">
        <div className="mobile-nav" aria-label="移动导航">
          <button type="button" className="btn-ghost bordered" onClick={newSession}>
            新建
          </button>
          <button
            type="button"
            className={`btn-ghost bordered ${page === "session" ? "active" : ""}`}
            onClick={() => setPage("session")}
          >
            会话
          </button>
          <button
            type="button"
            className={`btn-ghost bordered ${page === "settings" ? "active" : ""}`}
            onClick={() => setPage("settings")}
          >
            设置
          </button>
          <button
            type="button"
            className={`btn-ghost bordered ${page === "plugins" ? "active" : ""}`}
            onClick={() => setPage("plugins")}
          >
            插件
          </button>
        </div>
        {page === "settings" ? (
          <>
            <header className="session-header">
              <div className="session-title">设置</div>
            </header>
            <SettingsPage onConfigSaved={applyConfigToUi} />
          </>
        ) : null}
        {page === "plugins" ? (
          <>
            <header className="session-header">
              <div className="session-title">插件</div>
            </header>
            <PluginsPage />
          </>
        ) : null}
        {page === "session" ? (
          <>
            <header className="session-header">
              <div className="session-title">{sessionId}</div>
              <div className="view-tabs" role="tablist">
                <button
                  type="button"
                  role="tab"
                  aria-selected={view === "chat"}
                  className={`tab ${view === "chat" ? "active" : ""}`}
                  onClick={() => {
                    setView("chat");
                    setSessionView(sessionId, "chat");
                  }}
                >
                  对话
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={view === "trajectory"}
                  className={`tab ${view === "trajectory" ? "active" : ""}`}
                  onClick={() => {
                    setView("trajectory");
                    setSessionView(sessionId, "trajectory");
                  }}
                >
                  轨迹
                </button>
              </div>
            </header>
            <div className="workspace">
              {view === "chat" ? (
                <StatusBar
                  status={status}
                  expanded={statusExpanded}
                  onToggle={() => {
                    const next = !statusExpanded;
                    setStatusExpandedState(next);
                    setStatusExpanded(next);
                  }}
                  onJump={(id) => {
                    setView("trajectory");
                    setSessionView(sessionId, "trajectory");
                    setFocusId(id);
                  }}
                />
              ) : null}
              {view === "chat" && offline ? (
                <GuidePage
                  baseUrl={baseUrl}
                  detail={offlineDetail}
                  onRetry={() => {
                    void checkHealth();
                  }}
                />
              ) : null}
              {view === "chat" && !offline ? (
                <ChatView
                  messages={messages}
                  errorHint={softError}
                  onSettlementClick={(turnId) => {
                    setView("trajectory");
                    setSessionView(sessionId, "trajectory");
                    const hit = records.find(
                      (r) =>
                        r.event === "answer.done" &&
                        (turnId == null || `turn-${r.turn}` === turnId),
                    );
                    setFocusId(hit?.id ?? records[records.length - 1]?.id ?? null);
                  }}
                />
              ) : null}
              {view === "trajectory" ? (
                <TrajectoryView
                  records={records}
                  hasMore={hasMore}
                  loadingEarlier={loadingEarlier}
                  followTail={followTail}
                  focusId={focusId}
                  onFollowChange={setFollowTail}
                  onLoadEarlier={async () => {
                    if (nextBefore == null) return;
                    setLoadingEarlier(true);
                    try {
                      const page = await fetchEvents(sessionId, nextBefore, 50);
                      const older = projectEvents(
                        page.events.map((e) => ({
                          seq: Number(e.id ?? e.seq ?? 0),
                          event: e.event,
                          ts: e.ts,
                          data: (e.data ?? e.payload ?? {}) as Record<
                            string,
                            unknown
                          >,
                        })),
                      );
                      setRecords((prev) => {
                        const seen = new Set(prev.map((r) => r.id));
                        return [
                          ...older.filter((r) => !seen.has(r.id)),
                          ...prev,
                        ];
                      });
                      setHasMore(page.next_before != null);
                      setNextBefore(page.next_before);
                    } finally {
                      setLoadingEarlier(false);
                    }
                  }}
                  onExport={async () => {
                    const ok = window.confirm(
                      "日志可能包含路径、prompt 与工具参数，分享前请脱敏。是否继续导出？",
                    );
                    if (!ok) return;
                    const data = await exportSession(sessionId);
                    const blob = new Blob([JSON.stringify(data, null, 2)], {
                      type: "application/json",
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `${sessionId}-session-log.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                />
              ) : null}
              <Composer
                mode={mode}
                permission={permission}
                value={draft}
                running={status.running}
                promptCharBudget={promptBudget}
                segments={segments}
                onMode={(m) => {
                  setMode(m);
                  persistMode(sessionId, m);
                }}
                onPermission={(p) => {
                  setPermission(p);
                  persistPermission(sessionId, p);
                }}
                onChange={setDraft}
                onSend={() => void send()}
                onStop={() => {
                  if (runId) void stopRun(runId);
                }}
              />
            </div>
          </>
        ) : null}
      </main>
      {approval ? (
        <ApprovalDialog
          request={approval}
          onResolve={(approved) => {
            const id = approval.approval_id;
            setApproval(null);
            void approve(id, approved);
          }}
        />
      ) : null}
    </div>
  );
}
