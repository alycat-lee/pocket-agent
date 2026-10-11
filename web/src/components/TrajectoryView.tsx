import {
  Brain,
  CaretDown,
  CaretRight,
  CheckSquare,
  ClipboardText,
  GitMerge,
  Path,
  Square,
  TreeStructure,
  Users,
} from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { TrajectoryRecord } from "../api/types";
import { formatMs } from "../lib/format";
import { MODE_LABEL, SOURCE_LABEL, type IntentInfo } from "../lib/intent";
import { orderRecordsForDisplay } from "../lib/trajectory";

interface Props {
  records: TrajectoryRecord[];
  hasMore: boolean;
  loadingEarlier: boolean;
  followTail: boolean;
  focusId: string | null;
  onFollowChange: (v: boolean) => void;
  onLoadEarlier: () => void;
  onExport: () => void;
}

const BADGE_LABEL: Record<string, string> = {
  cache_hit: "cache_hit",
  intercepted: "已拦截",
  tool_error: "工具错误",
  rejected: "验收未通过",
};

function formatClock(ts: number | null): string | null {
  if (ts == null || !Number.isFinite(ts)) return null;
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleTimeString("en-GB", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function routeInfoOf(r: TrajectoryRecord): IntentInfo | null {
  if (r.kind !== "route") return null;
  return {
    mode: String(r.payload.mode || "agent") as IntentInfo["mode"],
    confidence: Number(r.payload.confidence ?? 0),
    source: String(r.payload.source || "fallback") as IntentInfo["source"],
    reason: String(r.payload.reason || ""),
    turn: r.turn,
    query: typeof r.payload.query === "string" ? r.payload.query : r.input,
  };
}

export function TrajectoryView({
  records,
  hasMore,
  loadingEarlier,
  followTail,
  focusId,
  onFollowChange,
  onLoadEarlier,
  onExport,
}: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const displayRecords = useMemo(
    () => orderRecordsForDisplay(records),
    [records],
  );

  const selected =
    displayRecords.find((r) => r.id === (selectedId || focusId)) || null;

  const overviewClass = useMemo(
    () =>
      displayRecords.map((r) => {
        if (r.kind === "route") return "route";
        if (r.kind === "task") return "task";
        if (r.kind === "orchestrator") return "task";
        if (r.kind === "worker" || r.kind === "merge") return "route";
        if (r.kind === "thinking") return "thinking";
        if (r.badges.includes("rejected")) return "rejected";
        if (r.badges.includes("tool_error")) return "tool_error";
        if (r.badges.includes("intercepted")) return "intercepted";
        if (r.badges.includes("cache_hit")) return "cache_hit";
        return "";
      }),
    [displayRecords],
  );

  // 锚定底部 & 跟随尾部
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (followTail) {
      el.scrollTop = el.scrollHeight;
    }
  }, [records.length, followTail]);

  // 检测上滚 → 暂停跟随 + 显示回到底部按钮
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const atBottom =
        el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      if (!atBottom && followTail) {
        onFollowChange(false);
      }
      setShowScrollBtn(!atBottom);
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, [followTail, onFollowChange]);

  // 切到轨迹标签时锚定底部
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [focusId]);

  return (
    <div className="trajectory">
      <div className="traj-toolbar">
        <span className="status-meta">
          {records.length} 条记录 · {followTail ? "跟随尾部" : "已暂停跟随"}
        </span>
        <button type="button" className="btn-ghost bordered" onClick={onExport}>
          导出 session log
        </button>
      </div>
      <div className="overview-bar" aria-label="时间概览条">
        {overviewClass.length === 0 ? (
          <div className="overview-seg" style={{ opacity: 0.3 }} />
        ) : (
          overviewClass.map((cls, i) => (
            <div key={displayRecords[i].id} className={`overview-seg ${cls}`} />
          ))
        )}
      </div>
      <div className="traj-body">
        <div className="traj-list-flow" ref={scrollRef}>
          {hasMore ? (
            <div className="traj-load-earlier">
              <button
                type="button"
                className="btn-ghost bordered"
                disabled={loadingEarlier}
                onClick={onLoadEarlier}
              >
                {loadingEarlier ? "加载中…" : "加载更早"}
              </button>
            </div>
          ) : null}
          {displayRecords.map((r, index) => {
            const parentId =
              r.kind !== "worker" && r.kind !== "merge"
                ? typeof r.payload.worker_id === "string"
                  ? r.payload.worker_id
                  : null
                : null;
            if (parentId && !expanded.has(parentId)) {
              return null;
            }
            const prevTurn = index > 0 ? displayRecords[index - 1]?.turn : null;
            const showTurn = r.turn !== prevTurn;
            const info = routeInfoOf(r);
            const workerId =
              typeof r.payload.worker_id === "string"
                ? r.payload.worker_id
                : r.kind === "worker" || r.kind === "merge"
                  ? String(r.payload.worker_id || r.id)
                  : "";
            const childCount = displayRecords.filter(
              (c) =>
                c.kind !== "worker" &&
                c.kind !== "merge" &&
                c.payload.worker_id === workerId,
            ).length;
            const canExpand =
              (r.kind === "worker" || r.kind === "merge") && childCount > 0;
            const isOpen = Boolean(workerId && expanded.has(workerId));
            const clock = formatClock(r.ts);
            return (
              <div key={r.id}>
                {showTurn ? (
                  <div className="turn-header">轮次 {r.turn}</div>
                ) : null}
                <button
                  type="button"
                  className={`record ${r.kind === "route" ? "route" : ""} ${r.kind === "task" || r.kind === "orchestrator" ? "task" : ""} ${r.kind === "thinking" ? "thinking" : ""} ${r.kind === "worker" || r.kind === "merge" ? "worker" : ""} ${parentId ? "nested" : ""} ${selected?.id === r.id ? "selected" : ""}`}
                  onClick={() => setSelectedId(r.id)}
                >
                  <span className="record-step">
                    {canExpand ? (
                      <span
                        className="expand-caret"
                        onClick={(e) => {
                          e.stopPropagation();
                          setExpanded((prev) => {
                            const next = new Set(prev);
                            if (next.has(workerId)) next.delete(workerId);
                            else next.add(workerId);
                            return next;
                          });
                        }}
                      >
                        {isOpen ? (
                          <CaretDown size={14} weight="bold" />
                        ) : (
                          <CaretRight size={14} weight="bold" />
                        )}
                      </span>
                    ) : r.kind === "route" ? (
                      <Path size={16} weight="bold" />
                    ) : r.kind === "task" ? (
                      <ClipboardText size={16} weight="bold" />
                    ) : r.kind === "orchestrator" ? (
                      <TreeStructure size={16} weight="bold" />
                    ) : r.kind === "worker" ? (
                      <Users size={16} weight="bold" />
                    ) : r.kind === "merge" ? (
                      <GitMerge size={16} weight="bold" />
                    ) : r.kind === "thinking" ? (
                      <Brain size={16} weight="duotone" />
                    ) : r.stepIndex != null ? (
                      `S${r.stepIndex}`
                    ) : (
                      "—"
                    )}
                  </span>
                  <span>
                    <div className="record-title-row">
                      <div className="record-title">{r.title}</div>
                      {clock ? <span className="rec-ts">{clock}</span> : null}
                    </div>
                    <div className="record-meta">
                      {r.kind === "task"
                        ? `计划 ${Array.isArray(r.payload.plan) ? r.payload.plan.length : 0} 步`
                        : r.kind === "orchestrator"
                          ? `${String(r.payload.strategy || "collect")} · ${Array.isArray(r.payload.workers) ? r.payload.workers.length : 0} 节点`
                          : r.kind === "worker" || r.kind === "merge"
                            ? canExpand
                              ? `${isOpen ? "收起" : "展开"} ${childCount} 步`
                              : String(r.payload.strategy || "子任务")
                        : r.kind === "route" && info
                        ? `${SOURCE_LABEL[info.source] ?? String(info.source ?? "")} · 置信度 ${Math.round(info.confidence * 100)}%`
                        : r.inProgress
                          ? "进行中 · 开始标记"
                          : `耗时 ${formatMs(r.elapsedMs)} · TTFT ${formatMs(r.ttftMs)}`}
                      {r.errorCode ? ` · ${r.errorCode}` : ""}
                    </div>
                  </span>
                  <span className="badges">
                    {r.badges.map((b) => (
                      <span key={b} className={`badge ${b}`}>
                        {BADGE_LABEL[b]}
                      </span>
                    ))}
                  </span>
                </button>
              </div>
            );
          })}
          {showScrollBtn ? (
            <button
              type="button"
              className="scroll-to-bottom"
              onClick={() => {
                const el = scrollRef.current;
                if (el) el.scrollTop = el.scrollHeight;
                onFollowChange(true);
              }}
            >
              回到底部
            </button>
          ) : null}
        </div>
        <aside className="inspector">
          <h3>记录检查器</h3>
          {!selected ? (
            <p className="status-meta">点击左侧记录查看详情</p>
          ) : (
            <InspectorBody record={selected} />
          )}
        </aside>
      </div>
    </div>
  );
}

function InspectorBody({ record }: { record: TrajectoryRecord }) {
  const info = routeInfoOf(record);
  const thinking =
    record.kind === "thinking"
      ? String(record.output || record.payload.text || "")
      : "";
  if (record.kind === "task") {
    return <TaskCardInspector record={record} />;
  }
  return (
    <dl>
      <div>
        <dt>事件</dt>
        <dd>{record.event}</dd>
      </div>
      {thinking ? (
        <div>
          <dt>思维链</dt>
          <dd className="thinking-block">{thinking}</dd>
        </div>
      ) : null}
      {info ? (
        <>
          <div>
            <dt>意图</dt>
            <dd>{MODE_LABEL[info.mode] ?? String(info.mode ?? "")}</dd>
          </div>
          <div>
            <dt>来源</dt>
            <dd>{SOURCE_LABEL[info.source] ?? String(info.source ?? "")}</dd>
          </div>
          <div>
            <dt>依据</dt>
            <dd>{info.reason || "—"}</dd>
          </div>
          <div>
            <dt>置信度</dt>
            <dd>{Math.round(info.confidence * 100)}%</dd>
          </div>
          <div>
            <dt>轮次</dt>
            <dd>{info.turn}</dd>
          </div>
        </>
      ) : (
        <>
          <div>
            <dt>Tokens (est)</dt>
            <dd>
              {record.promptTokensEst == null
                ? "不可用"
                : String(record.promptTokensEst)}
            </dd>
          </div>
          <div>
            <dt>耗时</dt>
            <dd>
              {record.inProgress ? "—" : formatMs(record.elapsedMs)}
            </dd>
          </div>
          <div>
            <dt>TTFT</dt>
            <dd>{formatMs(record.ttftMs)}</dd>
          </div>
        </>
      )}
      <div>
        <dt>输入</dt>
        <dd>{record.input || info?.query || "—"}</dd>
      </div>
      {thinking ? null : (
        <div>
          <dt>输出</dt>
          <dd>{record.output || "—"}</dd>
        </div>
      )}
      <div>
        <dt>参数</dt>
        <dd>
          {record.args == null ? "—" : JSON.stringify(record.args, null, 2)}
        </dd>
      </div>
    </dl>
  );
}

function TaskCardInspector({ record }: { record: TrajectoryRecord }) {
  const plan = Array.isArray(record.payload.plan)
    ? record.payload.plan.map((item) => String(item))
    : [];
  const rawDone = record.payload.plan_done;
  const done = Array.isArray(rawDone) ? rawDone.map(Boolean) : plan.map(() => false);
  const constraints = record.payload.constraints;
  return (
    <dl>
      <div>
        <dt>事件</dt>
        <dd>{record.event}</dd>
      </div>
      <div>
        <dt>类型</dt>
        <dd>{String(record.payload.type || "—")}</dd>
      </div>
      <div>
        <dt>目标</dt>
        <dd>{String(record.payload.goal || "—")}</dd>
      </div>
      <div>
        <dt>模型</dt>
        <dd>
          {record.payload.model
            ? `${String(record.payload.model)}${
                record.payload.provider_name
                  ? `（${String(record.payload.provider_name)} · ${String(record.payload.provider_type || "")}）`
                  : ""
              }`
            : "—"}
        </dd>
      </div>
      <div>
        <dt>约束</dt>
        <dd>
          {constraints && typeof constraints === "object"
            ? JSON.stringify(constraints, null, 2)
            : "—"}
        </dd>
      </div>
      <div>
        <dt>计划</dt>
        <dd className="plan-list">
          {plan.length === 0 ? (
            "—"
          ) : (
            <ol>
              {plan.map((step, i) => (
                <li key={`${i}-${step}`} className={done[i] ? "done" : ""}>
                  {done[i] ? (
                    <CheckSquare size={14} weight="fill" />
                  ) : (
                    <Square size={14} />
                  )}
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          )}
        </dd>
      </div>
    </dl>
  );
}
