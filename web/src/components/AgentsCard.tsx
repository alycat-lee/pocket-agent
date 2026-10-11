import { useCallback, useEffect, useState } from "react";
import {
  createAgent,
  deleteAgent,
  fetchAgents,
  updateAgent,
} from "../api/client";
import type { AgentInfo } from "../api/types";

export function AgentsCard() {
  const [root, setRoot] = useState("");
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [prompt, setPrompt] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    const data = await fetchAgents();
    setRoot(data.root);
    setAgents(data.agents);
  }, []);

  useEffect(() => {
    void reload().catch((e) =>
      setError(e instanceof Error ? e.message : String(e)),
    );
  }, [reload]);

  return (
    <div className="prompts-card knowledge-card">
      <h3>Agent</h3>
      <p className="status-meta">
        根目录：<strong>{root || "agents"}</strong>
        。修改路径只影响新创建的 agent，不会搬迁已有目录。
      </p>
      {error ? <div className="toast-error">{error}</div> : null}

      <h4>新建</h4>
      <div className="kb-search">
        <input
          type="text"
          value={name}
          placeholder="名称，如 写手"
          onChange={(ev) => setName(ev.target.value)}
        />
        <button
          type="button"
          className="btn-primary"
          disabled={busy || !name.trim()}
          onClick={() => {
            setBusy(true);
            setError(null);
            void createAgent({
              name: name.trim(),
              description,
              prompt,
            })
              .then(() => {
                setName("");
                setDescription("");
                setPrompt("");
                return reload();
              })
              .catch((e) =>
                setError(e instanceof Error ? e.message : String(e)),
              )
              .finally(() => setBusy(false));
          }}
        >
          创建
        </button>
      </div>
      <input
        type="text"
        value={description}
        placeholder="描述（可选）"
        onChange={(ev) => setDescription(ev.target.value)}
      />
      <textarea
        className="agent-prompt"
        rows={4}
        value={prompt}
        placeholder="prompt.md（可选，写入人格）"
        onChange={(ev) => setPrompt(ev.target.value)}
      />

      <h4>已有 Agent</h4>
      {agents.length === 0 ? (
        <div className="empty-hint">还没有 agent</div>
      ) : (
        <ul className="kb-doc-list">
          {agents.map((item) => (
            <li key={item.name} className="kb-doc-item">
              <div>
                <strong>{item.name}</strong>
                <span className="status-meta">
                  {" "}
                  · {item.description || "无描述"}
                </span>
              </div>
              {item.name === "default" ? (
                <span className="status-meta">不可删除</span>
              ) : (
                <button
                  type="button"
                  className="btn-ghost bordered"
                  disabled={busy}
                  onClick={() => {
                    setBusy(true);
                    setError(null);
                    void deleteAgent(item.name, true)
                      .then(() => reload())
                      .catch((e) =>
                        setError(e instanceof Error ? e.message : String(e)),
                      )
                      .finally(() => setBusy(false));
                  }}
                >
                  删除
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      {agents.map((item) => (
        <details key={`p-${item.name}`} className="agent-details">
          <summary>编辑 {item.name} 的 prompt.md</summary>
          <textarea
            className="agent-prompt"
            rows={6}
            defaultValue={item.prompt}
            onBlur={(ev) => {
              const next = ev.target.value;
              if (next === item.prompt) return;
              void updateAgent(item.name, { prompt: next }).catch((e) =>
                setError(e instanceof Error ? e.message : String(e)),
              );
            }}
          />
        </details>
      ))}
    </div>
  );
}
