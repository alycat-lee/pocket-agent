import type { ChatMode, PermissionMode } from "../api/types";
import { formatChars } from "../lib/format";

interface Props {
  mode: ChatMode;
  permission: PermissionMode;
  value: string;
  running: boolean;
  promptCharBudget: number;
  segments: { label: string; chars: number }[];
  onMode: (m: ChatMode) => void;
  onPermission: (p: PermissionMode) => void;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
}

export function Composer({
  mode,
  permission,
  value,
  running,
  promptCharBudget,
  segments,
  onMode,
  onPermission,
  onChange,
  onSend,
  onStop,
}: Props) {
  const used = segments.reduce((a, s) => a + s.chars, 0) + value.length;

  return (
    <div className="composer">
      <div className="composer-toolbar">
        <select
          className="select"
          value={mode}
          onChange={(e) => onMode(e.target.value as ChatMode)}
          aria-label="模式"
        >
          <option value="qa">问答</option>
          <option value="tools">工具调用</option>
          <option value="writer">长文写作</option>
        </select>
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
          placeholder="输入消息… 使用 @文件 引用（预注入 read_file）"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
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
