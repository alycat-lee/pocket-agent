import type { IntentMode, PermissionMode } from "../api/types";
import { formatChars } from "../lib/format";

interface Props {
  intent: IntentMode | null;
  intentConfidence: number;
  permission: PermissionMode;
  workspaces: { id: string; name: string; root: string }[];
  workspaceId: string | null;
  value: string;
  running: boolean;
  promptCharBudget: number;
  segments: { label: string; chars: number }[];
  onCorrectIntent: (mode: IntentMode) => void;
  onPermission: (p: PermissionMode) => void;
  onWorkspace: (id: string) => void;
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
  workspaces,
  workspaceId,
  value,
  running,
  promptCharBudget,
  segments,
  onCorrectIntent,
  onPermission,
  onWorkspace,
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
        {workspaces.length > 0 ? (
          <select
            className="select"
            value={workspaceId || workspaces[0]?.id || ""}
            onChange={(e) => onWorkspace(e.target.value)}
            aria-label="工作区"
          >
            {workspaces.map((w) => (
              <option key={w.id} value={w.id}>
                {w.name}
              </option>
            ))}
          </select>
        ) : null}
        <select
          className="select"
          value={permission}
          onChange={(e) => onPermission(e.target.value as PermissionMode)}
          aria-label="权限"
        >
          <option value="readonly">只读</option>
          <option value="workspace">工作区</option>
          <option value="full">完全</option>
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
