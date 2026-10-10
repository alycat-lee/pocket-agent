import type { ChatMessage } from "../api/types";
import { formatCacheHitRate } from "../lib/format";

interface Props {
  messages: ChatMessage[];
  errorHint?: string | null;
  onSettlementClick: (turnId?: string) => void;
}

export function ChatView({ messages, errorHint, onSettlementClick }: Props) {
  return (
    <div className="messages">
      {errorHint ? <div className="toast-error">{errorHint}</div> : null}
      {messages.length === 0 ? (
        <div className="empty-hint">发送一条消息开始对话。步骤细节请切换到「轨迹」。</div>
      ) : null}
      {messages.map((m) => (
        <div
          key={m.id}
          className={`msg ${m.role}${m.error ? " error" : ""}`}
        >
          <div className="msg-bubble">{m.content}</div>
          {m.role === "assistant" && m.settlement ? (
            <button
              type="button"
              className="settlement-line"
              onClick={() => onSettlementClick(m.turnId)}
            >
              {m.settlement.total_steps} 步 · 缓存命中{" "}
              {formatCacheHitRate(m.settlement.cache_hit_rate)} · 拦截{" "}
              {m.settlement.intercepts ?? 0}
            </button>
          ) : null}
        </div>
      ))}
    </div>
  );
}
