export type SchemaFieldType = "string" | "number" | "integer" | "boolean";

export interface SchemaField {
  key: string;
  type: SchemaFieldType;
  group: string;
  description: string;
  min?: number;
  max?: number;
  enum?: unknown[];
  secret?: boolean;
}

export type FormControlKind = "text" | "number" | "boolean" | "select" | "password";

export interface FormControl {
  key: string;
  kind: FormControlKind;
  group: string;
  description: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  secret?: boolean;
}

/** 由 schema 元数据生成表单控件描述——禁止硬编码字段列表。 */
export function buildFormControls(schema: SchemaField[]): FormControl[] {
  return schema.map((field) => {
    if (field.enum && field.enum.length > 0) {
      return {
        key: field.key,
        kind: "select" as const,
        group: field.group,
        description: field.description,
        options: field.enum.map(String),
        secret: field.secret,
      };
    }
    if (field.type === "boolean") {
      return {
        key: field.key,
        kind: "boolean" as const,
        group: field.group,
        description: field.description,
      };
    }
    if (field.type === "number" || field.type === "integer") {
      return {
        key: field.key,
        kind: "number" as const,
        group: field.group,
        description: field.description,
        min: field.min,
        max: field.max,
        step: field.type === "integer" ? 1 : 0.01,
      };
    }
    return {
      key: field.key,
      kind: field.secret ? ("password" as const) : ("text" as const),
      group: field.group,
      description: field.description,
      secret: field.secret,
    };
  });
}

export function groupControls(
  controls: FormControl[],
): { group: string; controls: FormControl[] }[] {
  const order: string[] = [];
  const map = new Map<string, FormControl[]>();
  for (const c of controls) {
    if (!map.has(c.group)) {
      order.push(c.group);
      map.set(c.group, []);
    }
    map.get(c.group)!.push(c);
  }
  return order.map((group) => ({ group, controls: map.get(group)! }));
}

export function coerceFormValue(
  control: FormControl,
  raw: string | boolean,
): unknown {
  if (control.kind === "boolean") return Boolean(raw);
  if (control.kind === "number") {
    const n = Number(raw);
    if (Number.isNaN(n)) return raw;
    if (control.step === 1) return Math.trunc(n);
    return n;
  }
  return String(raw);
}
