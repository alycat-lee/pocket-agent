import { useCallback, useEffect, useState } from "react";
import {
  disablePlugin,
  enablePlugin,
  fetchPlugins,
} from "../api/client";
import type { PluginInfo } from "../api/types";

export function PluginsPage() {
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    const rows = await fetchPlugins();
    setPlugins(rows);
  }, []);

  useEffect(() => {
    reload().catch((e) =>
      setError(e instanceof Error ? e.message : String(e)),
    );
  }, [reload]);

  return (
    <div className="plugins-layout">
      {error ? <div className="toast-error">{error}</div> : null}
      {plugins.map((p) => (
        <article key={p.name} className="plugin-card">
          <div>
            <h3>{p.name}</h3>
            <div className="plugin-meta">
              提供: {(p.provides || []).join(", ") || "—"}
              <br />
              依赖: {(p.requires || []).join(", ") || "—"}
              <br />
              状态: {p.enabled ? "已挂载" : "未挂载"}
            </div>
          </div>
          <button
            type="button"
            className={`toggle ${p.enabled ? "on" : ""}`}
            aria-pressed={p.enabled}
            title={p.enabled ? "禁用" : "启用"}
            onClick={async () => {
              try {
                setError(null);
                if (p.enabled) {
                  const res = await disablePlugin(p.name);
                  if (!res.ok) {
                    setError(res.error || "禁用失败");
                    return;
                  }
                } else {
                  const res = await enablePlugin(p.name);
                  if (!res.ok) {
                    setError(res.error || "启用失败");
                    return;
                  }
                }
                await reload();
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
              }
            }}
          />
        </article>
      ))}
      {plugins.length === 0 && !error ? (
        <div className="empty-hint">暂无插件目录信息</div>
      ) : null}
    </div>
  );
}
