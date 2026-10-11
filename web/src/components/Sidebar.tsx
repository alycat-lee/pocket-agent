import { GearSix, Plus, PuzzlePiece, Trash } from "@phosphor-icons/react";
import type { AgentInfo, SessionInfo } from "../api/types";

export type PageId = "session" | "settings" | "plugins";

interface Props {
  sessions: SessionInfo[];
  activeSession: string;
  agents: AgentInfo[];
  activeAgent: string;
  page: PageId;
  onSelectSession: (id: string) => void;
  onSelectAgent: (name: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onPage: (p: PageId) => void;
}

export function Sidebar({
  sessions,
  activeSession,
  agents,
  activeAgent,
  page,
  onSelectSession,
  onSelectAgent,
  onNewSession,
  onDeleteSession,
  onPage,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="mark" aria-hidden />
        pocket-agent
      </div>
      <div className="sidebar-agent">
        <label className="sidebar-label" htmlFor="agent-select">
          Agent
        </label>
        <select
          id="agent-select"
          value={activeAgent}
          onChange={(ev) => onSelectAgent(ev.target.value)}
        >
          {(agents.length ? agents : [{ name: "default", description: "" }]).map(
            (item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ),
          )}
        </select>
      </div>
      <div className="sidebar-section">
        <div className="sidebar-label">会话</div>
        <button
          type="button"
          className="nav-item"
          onClick={onNewSession}
          style={{ marginBottom: 6 }}
        >
          <Plus size={14} /> 新建会话
        </button>
        <div className="session-list">
          {sessions.map((s) => (
            <div
              key={s.session_id}
              className={`session-row ${
                page === "session" && activeSession === s.session_id
                  ? "active"
                  : ""
              }`}
            >
              <button
                type="button"
                className="session-item"
                onClick={() => {
                  onPage("session");
                  onSelectSession(s.session_id);
                }}
              >
                {s.title || s.session_id}
              </button>
              <button
                type="button"
                className="session-delete"
                title="删除会话"
                aria-label={`删除会话 ${s.title || s.session_id}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(s.session_id);
                }}
              >
                <Trash size={12} />
              </button>
            </div>
          ))}
        </div>
      </div>
      <div className="sidebar-foot">
        <button
          type="button"
          className={`nav-item ${page === "settings" ? "active" : ""}`}
          onClick={() => onPage("settings")}
        >
          <GearSix size={14} /> 设置
        </button>
        <button
          type="button"
          className={`nav-item ${page === "plugins" ? "active" : ""}`}
          onClick={() => onPage("plugins")}
        >
          <PuzzlePiece size={14} /> 插件
        </button>
      </div>
    </aside>
  );
}
