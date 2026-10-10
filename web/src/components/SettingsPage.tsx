import { useEffect, useMemo, useState } from "react";
import { fetchConfig, fetchConfigSchema, putConfig } from "../api/client";
import {
  buildFormControls,
  groupControls,
  type FormControl,
} from "../lib/schemaForm";
import { SchemaForm } from "./SchemaForm";

export function SettingsPage() {
  const [groups, setGroups] = useState<
    { group: string; controls: FormControl[] }[]
  >([]);
  const [active, setActive] = useState("");
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [defaults, setDefaults] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
        const grouped = groupControls(controls);
        setGroups(grouped);
        setActive(grouped[0]?.group || "");
        setValues({ ...config });
        setDefaults({ ...config });
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
      <nav className="settings-nav">
        {groups.map((g) => (
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
        {error ? <div className="toast-error">{error}</div> : null}
        {status ? <div className="settlement-strip">{status}</div> : null}
        <SchemaForm
          controls={current}
          values={values}
          onChange={(key, value) =>
            setValues((prev) => ({ ...prev, [key]: value }))
          }
        />
        <div className="form-actions">
          <button
            type="button"
            className="btn-primary"
            onClick={async () => {
              try {
                const res = await putConfig(values);
                setValues(res.config);
                setDefaults(res.config);
                setStatus("已保存，即时生效");
                setError(null);
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
              }
            }}
          >
            保存
          </button>
          <button
            type="button"
            className="btn-ghost bordered"
            onClick={() => {
              setValues({ ...defaults });
              setStatus("已恢复为当前已保存值");
            }}
          >
            恢复默认
          </button>
        </div>
      </div>
    </div>
  );
}
