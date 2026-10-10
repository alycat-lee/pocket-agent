import type { IntentMode, PermissionMode } from "../api/types";
import { formatChars } from "../lib/format";

interface Props {
  intent: IntentMode | null;
  intentConfidence: number;
  permission: PermissionMode;
  value: string;
  running: boolean;
  promptCharBudget: number;
  segments: { label: string; chars: number }[];
  onCorrectIntent: (mode: IntentMode) => void;
  onPermission: (p: PermissionMode) => void;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
}

const INTENT_LABEL: Record<IntentMode, string> = {
  qa: "问答",
  agent: "工具",
  writer: "长文",
};

export function Composer({
  intent,
  intentConfidence,
  permission,
  value,
  running,
  promptCharBudget,
  segments,
  onCorrectIntent,
  onPermission,
  onChange,
  onSend,
  onStop,
}: Props) {
  const used = segments.reduce((a, s) => a + s.chars, 0) + value.length;

  return (
    <div className="composer">
      <div className="composer-toolbar">
        {intent ? (
          <span
            className="intent-badge"
            title={`意图: ${INTENT_LABEL[intent]} (置信度 ${Math.round(intentConfidence * 100)}%)。点击纠正`}
            onClick={() => {
              const modes: IntentMode[] = ["qa", "agent", "writer"];
              const next = modes[(modes.indexOf(intent) + 1) % modes.length];
              onCorrectIntent(next);
            }}
          >
            {INTENT_LABEL[intent]}
            {intentConfidence < 1.0 ? ` ${Math.round(intentConfidence * 100)}%` : ""}
          </span>
        ) : null}
        <select
          className="select"
          value={permission}
          onChange={(e) => onPermission(e.target.value as PermissionMode)}
          aria-label="权限"
        >
          <option value="readonly">只读</option>
          <option value="confirm_write">仅写操作需确认</option>
          <option value="auto">自动放行</option>
        </select>
      </div>
      <div className="composer-box">
        <textarea
          value={value}
          placeholder="输入消息…（Enter 发送 / Shift+Enter 换行）"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!running) onSend();
            }
          }}
        />
        {running ? (
          <button type="button" className="btn-danger" onClick={onStop}>
            停止
          </button>
        ) : (
          <button
            type="button"
            className="btn-primary"
            onClick={onSend}
            disabled={!value.trim()}
          >
            发送
          </button>
        )}
      </div>
      <details className="ctx-usage">
        <summary>
          上下文用量 {formatChars(used)} / {formatChars(promptCharBudget)} 字符
        </summary>
        <div className="ctx-detail">
          {segments.map((s) => (
            <div key={s.label}>
              {s.label}: {formatChars(s.chars)}
            </div>
          ))}
          <div>本步输入: {formatChars(value.length)}</div>
        </div>
      </details>
    </div>
  );
}
