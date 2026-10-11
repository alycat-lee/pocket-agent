import { useEffect, useMemo, useState } from "react";
import {
  fetchConfig,
  fetchConfigSchema,
  putConfig,
  resetConfig,
} from "../api/client";
import {
  buildFormControls,
  groupControls,
  type FormControl,
} from "../lib/schemaForm";
import { KnowledgeCard } from "./KnowledgeCard";
import { PromptsCard } from "./PromptsCard";
import { ProvidersCard } from "./ProvidersCard";
import { SchemaForm } from "./SchemaForm";

function sanitizeConfig(
  values: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(values)) {
    if (typeof value === "number" && Number.isNaN(value)) continue;
    out[key] = value;
  }
  return out;
}

interface Props {
  onConfigSaved?: (config: Record<string, unknown>) => void;
}

export function SettingsPage({ onConfigSaved }: Props) {
  const [groups, setGroups] = useState<
    { group: string; controls: FormControl[] }[]
  >([]);
  const [active, setActive] = useState("");
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"save" | "reset" | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [schema, config] = await Promise.all([
          fetchConfigSchema(),
          fetchConfig(),
        ]);
        if (cancelled) return;
        const controls = buildFormControls(schema);
        // 过滤掉 deprecated 组（模型组由 ProvidersCard 定制渲染）
        const grouped = groupControls(
          controls.filter(
            (c) => !schema.find((s) => s.key === c.key)?.deprecated,
          ),
        );
        setGroups(grouped);
        setActive("模型");
        setValues({ ...config });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const current = useMemo(
    () => groups.find((g) => g.group === active)?.controls || [],
    [groups, active],
  );

  return (
    <div className="settings-layout">
      <nav className="settings-nav" aria-label="设置分组">
        <button
          type="button"
          className={`nav-item ${active === "模型" ? "active" : ""}`}
          onClick={() => setActive("模型")}
        >
          模型
        </button>
        <button
          type="button"
          className={`nav-item ${active === "提示词" ? "active" : ""}`}
          onClick={() => setActive("提示词")}
        >
          提示词
        </button>
        <button
          type="button"
          className={`nav-item ${active === "知识库" ? "active" : ""}`}
          onClick={() => setActive("知识库")}
        >
          知识库
        </button>
        {groups
          .filter((g) => g.group !== "模型")
          .map((g) => (
          <button
            key={g.group}
            type="button"
            className={`nav-item ${active === g.group ? "active" : ""}`}
            onClick={() => setActive(g.group)}
          >
            {g.group}
          </button>
        ))}
      </nav>
      <div className="settings-form">
        <div className="settings-form-body">
          {error ? <div className="toast-error">{error}</div> : null}
          {active === "模型" ? (
            <ProvidersCard />
          ) : active === "提示词" ? (
            <PromptsCard />
          ) : active === "知识库" ? (
            <KnowledgeCard />
          ) : (
            <>
              {groups.length === 0 && !error ? (
                <div className="empty-hint">正在加载配置 schema…</div>
              ) : null}
              <SchemaForm
                controls={current}
                values={values}
                onChange={(key, value) => {
                  setStatus(null);
                  setValues((prev) => ({ ...prev, [key]: value }));
                }}
              />
            </>
          )}
        </div>
        {active === "提示词" || active === "知识库" ? null : (
        <div className="form-actions">
          {status ? <div className="form-status">{status}</div> : null}
          <button
            type="button"
            className="btn-primary"
            disabled={busy != null || groups.length === 0}
            onClick={() => {
              void (async () => {
                setBusy("save");
                setError(null);
                try {
                  const res = await putConfig(sanitizeConfig(values));
                  setValues({ ...res.config });
                  setStatus("已保存，即时生效");
                  onConfigSaved?.(res.config);
                } catch (e) {
                  setError(e instanceof Error ? e.message : String(e));
                  setStatus(null);
                } finally {
                  setBusy(null);
                }
              })();
            }}
          >
            {busy === "save" ? "保存中…" : "保存"}
          </button>
          <button
            type="button"
            className="btn-ghost bordered"
            disabled={busy != null}
            onClick={() => {
              void (async () => {
                if (
                  !window.confirm(
                    "将清空 settings.json，恢复为 pocket.yml / 内置默认值。继续？",
                  )
                ) {
                  return;
                }
                setBusy("reset");
                setError(null);
                try {
                  const res = await resetConfig();
                  setValues({ ...res.config });
                  setStatus("已恢复默认，即时生效");
                  onConfigSaved?.(res.config);
                } catch (e) {
                  setError(e instanceof Error ? e.message : String(e));
                  setStatus(null);
                } finally {
                  setBusy(null);
                }
              })();
            }}
          >
            {busy === "reset" ? "恢复中…" : "恢复默认"}
          </button>
        </div>
        )}
      </div>
    </div>
  );
}
