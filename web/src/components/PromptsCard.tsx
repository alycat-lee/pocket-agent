import { useEffect, useState } from "react";
import { fetchPrompts, type PromptEntry } from "../api/client";

export function PromptsCard() {
  const [variant, setVariant] = useState("");
  const [prompts, setPrompts] = useState<PromptEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void fetchPrompts()
      .then((data) => {
        setVariant(data.variant);
        setPrompts(data.prompts);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="prompts-card">
      <h3>生效提示词</h3>
      <p className="status-meta">
        只读调试入口。当前变体：<strong>{variant || "—"}</strong>
        （llamacpp → local，deepseek → cloud）
      </p>
      {error ? <div className="toast-error">{error}</div> : null}
      <div className="prompt-list">
        {prompts.map((p) => (
          <article key={p.id} className="prompt-item">
            <header className="prompt-item-header">
              <strong>{p.id}</strong>
              <span className="badge cache_hit">v{p.version}</span>
            </header>
            <pre className="prompt-template">{p.template}</pre>
          </article>
        ))}
      </div>
    </div>
  );
}
