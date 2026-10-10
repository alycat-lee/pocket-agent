import type { FormControl } from "../lib/schemaForm";

interface Props {
  controls: FormControl[];
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}

export function SchemaForm({ controls, values, onChange }: Props) {
  return (
    <>
      {controls.map((c) => (
        <div className="field" key={c.key}>
          <label htmlFor={`f-${c.key}`}>
            {c.key}
            <span className="hint" title={c.description}>
              (?)
            </span>
          </label>
          {c.kind === "boolean" ? (
            <button
              type="button"
              id={`f-${c.key}`}
              className={`toggle ${values[c.key] ? "on" : ""}`}
              aria-pressed={Boolean(values[c.key])}
              aria-label={c.key}
              title={c.description}
              onClick={() => onChange(c.key, !values[c.key])}
            />
          ) : null}
          {c.kind === "select" ? (
            <select
              id={`f-${c.key}`}
              className="select"
              value={String(values[c.key] ?? "")}
              title={c.description}
              onChange={(e) => onChange(c.key, e.target.value)}
            >
              {(c.options || []).map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          ) : null}
          {c.kind === "number" ? (
            <input
              id={`f-${c.key}`}
              type="number"
              min={c.min}
              max={c.max}
              step={c.step}
              value={values[c.key] == null ? "" : String(values[c.key])}
              title={c.description}
              onChange={(e) => {
                const n = Number(e.target.value);
                onChange(c.key, c.step === 1 ? Math.trunc(n) : n);
              }}
            />
          ) : null}
          {c.kind === "text" || c.kind === "password" ? (
            <input
              id={`f-${c.key}`}
              type={c.kind === "password" ? "password" : "text"}
              value={String(values[c.key] ?? "")}
              title={c.description}
              autoComplete={c.secret ? "off" : undefined}
              onChange={(e) => onChange(c.key, e.target.value)}
            />
          ) : null}
          <div className="status-meta" style={{ marginTop: 4 }}>
            {c.description}
          </div>
        </div>
      ))}
    </>
  );
}
