import { useEffect, useMemo, useRef, useState } from "react";
import type { TrajectoryRecord } from "../api/types";
import { formatMs } from "../lib/format";

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
};

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
  const scrollRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const selected =
    records.find((r) => r.id === (selectedId || focusId)) || null;

  const overviewClass = useMemo(
    () =>
      records.map((r) => {
        if (r.badges.includes("tool_error")) return "tool_error";
        if (r.badges.includes("intercepted")) return "intercepted";
        if (r.badges.includes("cache_hit")) return "cache_hit";
        return "";
      }),
    [records],
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
            <div key={records[i].id} className={`overview-seg ${cls}`} />
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
          {records.map((r, index) => {
            const prevTurn = index > 0 ? records[index - 1]?.turn : null;
            const showTurn = r.turn !== prevTurn;
            return (
              <div key={r.id}>
                {showTurn ? (
                  <div className="turn-header">轮次 {r.turn}</div>
                ) : null}
                <button
                  type="button"
                  className={`record ${selected?.id === r.id ? "selected" : ""}`}
                  onClick={() => setSelectedId(r.id)}
                >
                  <span className="record-step">
                    {r.stepIndex != null ? `S${r.stepIndex}` : "—"}
                  </span>
                  <span>
                    <div className="record-title">{r.title}</div>
                    <div className="record-meta">
                      {r.inProgress
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
            <dl>
              <div>
                <dt>事件</dt>
                <dd>{selected.event}</dd>
              </div>
              <div>
                <dt>Tokens (est)</dt>
                <dd>
                  {selected.promptTokensEst == null
                    ? "不可用"
                    : String(selected.promptTokensEst)}
                </dd>
              </div>
              <div>
                <dt>耗时</dt>
                <dd>
                  {selected.inProgress
                    ? "—"
                    : formatMs(selected.elapsedMs)}
                </dd>
              </div>
              <div>
                <dt>TTFT</dt>
                <dd>{formatMs(selected.ttftMs)}</dd>
              </div>
              <div>
                <dt>输入</dt>
                <dd>{selected.input || "—"}</dd>
              </div>
              <div>
                <dt>输出</dt>
                <dd>{selected.output || "—"}</dd>
              </div>
              <div>
                <dt>参数</dt>
                <dd>
                  {selected.args == null
                    ? "—"
                    : JSON.stringify(selected.args, null, 2)}
                </dd>
              </div>
            </dl>
          )}
        </aside>
      </div>
    </div>
  );
}
