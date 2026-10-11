import type { PermissionMode } from "../api/types";
import { formatChars } from "../lib/format";

interface Props {
  permission: PermissionMode;
  workspaces: { id: string; name: string; root: string }[];
  workspaceId: string | null;
  value: string;
  running: boolean;
  promptCharBudget: number;
  segments: { label: string; chars: number }[];
  onPermission: (p: PermissionMode) => void;
  onWorkspace: (id: string) => void;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  stopping?: boolean;
}

export function Composer({
  permission,
  workspaces,
  workspaceId,
  value,
  running,
  promptCharBudget,
  segments,
  onPermission,
  onWorkspace,
  onChange,
  onSend,
  onStop,
  stopping = false,
}: Props) {
  const used = segments.reduce((a, s) => a + s.chars, 0) + value.length;

  return (
    <div className="composer">
      <div className="composer-toolbar">
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
          <button
            type="button"
            className="btn-danger"
            disabled={stopping}
            onClick={onStop}
          >
            {stopping ? "正在停止…" : "停止"}
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
