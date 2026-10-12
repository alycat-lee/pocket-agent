import type { ChatMessage, IntentMode } from "../api/types";
import { formatCacheHitRate, formatUsageTokens } from "../lib/format";
import { IntentBadge } from "./IntentBadge";

interface Props {
  messages: ChatMessage[];
  errorHint?: string | null;
  onSettlementClick: (turnId?: string) => void;
  onCorrectIntent: (turnId: string, mode: IntentMode) => void;
}

export function ChatView({
  messages,
  errorHint,
  onSettlementClick,
  onCorrectIntent,
}: Props) {
  return (
    <div className="messages">
      {errorHint ? <div className="toast-error">{errorHint}</div> : null}
      {messages.length === 0 ? (
        <div className="empty-hint">发送一条消息开始对话。步骤细节请切换到「轨迹」。</div>
      ) : null}
      {messages.map((m, index) => {
        const streaming =
          m.role === "assistant" &&
          !m.settlement &&
          !m.error &&
          index === messages.length - 1;
        return (
        <div
          key={m.id}
          className={`msg ${m.role}${m.role === "user" ? " msg-user" : ""}${m.error ? " error" : ""}`}
        >
          <div className="msg-bubble">
            {m.content}
            {streaming ? <span className="stream-cursor">▍</span> : null}
          </div>
          {m.role === "assistant" && m.settlement ? (
            <div className="settlement-row">
              {m.intent && m.turnId ? (
                <IntentBadge
                  intent={m.intent}
                  onCorrect={(mode) => onCorrectIntent(m.turnId as string, mode)}
                />
              ) : null}
              <button
                type="button"
                className="settlement-line"
                onClick={() => onSettlementClick(m.turnId)}
              >
                {m.settlement.total_steps} 步 · 缓存命中{" "}
                {formatCacheHitRate(m.settlement.cache_hit_rate)} · 拦截{" "}
                {m.settlement.intercepts ?? 0}
                {formatUsageTokens(m.settlement.usage)}
              </button>
            </div>
          ) : null}
        </div>
        );
      })}
    </div>
  );
}
