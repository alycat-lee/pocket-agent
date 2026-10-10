import { CircleNotch, Lightning } from "@phosphor-icons/react";
import type { StatusSnapshot } from "../api/types";
import { formatCacheHitRate, formatElapsed, formatMs } from "../lib/format";

interface Props {
  status: StatusSnapshot;
  expanded: boolean;
  onToggle: () => void;
  onJump: (recordId: string) => void;
}

export function StatusBar({ status, expanded, onToggle, onJump }: Props) {
  if (!status.running && !status.settlement) return null;

  const budget = status.promptCharBudget || 1;
  const pct = Math.min(100, (status.promptTokensEst / budget) * 100);
  const fillClass =
    pct >= 95 ? "danger" : pct >= 80 ? "warn" : "";

  return (
    <div className="status-bar">
      {status.running ? (
        <button
          type="button"
          className="status-overview"
          onClick={onToggle}
          style={{ width: "100%", cursor: "pointer" }}
        >
          <span className="status-phase">
            {status.phase.includes("审批") ? (
              <Lightning size={14} weight="fill" />
            ) : (
              <CircleNotch size={14} className="spin" />
            )}
            {status.phase}
          </span>
          <span className="status-meta">
            第 {status.stepIndex}/{status.maxSteps} 步 ·{" "}
            {formatElapsed(status.elapsedSec)}
          </span>
          <div className="budget-track" title="上下文预算占用">
            <div
              className={`budget-fill ${fillClass}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="status-meta">
            {status.promptTokensEst} / {status.promptCharBudget}
          </span>
        </button>
      ) : null}

      {status.running && expanded ? (
        <div className="status-process">
          {status.timeline.map((n) => (
            <button
              key={n.id}
              type="button"
              className={`timeline-node ${n.active ? "active" : ""}`}
              onClick={() => onJump(n.id)}
            >
              {n.label}
            </button>
          ))}
        </div>
      ) : null}

      {status.settlement ? (
        <div className="settlement-strip">
          {status.settlement.total_steps} 步 · LLM{" "}
          {formatMs(status.settlement.llm_ms)} · 工具{" "}
          {formatMs(status.settlement.tool_ms)} · 缓存命中{" "}
          {formatCacheHitRate(status.settlement.cache_hit_rate)} · 真实调用{" "}
          {status.settlement.real_calls ?? 0} 次 · 拦截{" "}
          {status.settlement.intercepts ?? 0} 次
        </div>
      ) : null}
    </div>
  );
}
