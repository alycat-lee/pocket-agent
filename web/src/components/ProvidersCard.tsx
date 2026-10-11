import { useCallback, useEffect, useState } from "react";
import {
  createProvider,
  deleteProvider,
  fetchProviderPresets,
  fetchProviders,
  setDefaultProvider,
  testProvider,
  updateProvider,
  updateRouting,
  type ProviderPreset,
} from "../api/client";

interface Provider {
  id: string;
  name: string;
  type: string;
  base_url: string;
  api_key: string;
  model: string;
  sampling: Record<string, unknown> | null;
}

interface TestResult {
  ok: boolean;
  connect_ms: number;
  generate_ms: number;
  model_list: string[];
  error: string;
}

export function ProvidersCard() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [defaultId, setDefaultId] = useState("");
  const [routing, setRouting] = useState({ choose_fill: "", generate: "" });
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({});
  const [presets, setPresets] = useState<ProviderPreset[]>([]);
  const [draftType, setDraftType] = useState("llamacpp");
  const [draftName, setDraftName] = useState("");
  const [draftUrl, setDraftUrl] = useState("http://localhost:8000/v1");
  const [draftModel, setDraftModel] = useState("");
  const [draftKey, setDraftKey] = useState("");
  const [draftThinking, setDraftThinking] = useState("off");
  const [draftTrustEnv, setDraftTrustEnv] = useState(false);

  const reload = useCallback(async () => {
    try {
      const data = await fetchProviders();
      setProviders(data.providers as Provider[]);
      setDefaultId(data.default_provider_id);
      setRouting(data.routing);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  useEffect(() => {
    void fetchProviderPresets()
      .then((data) => {
        setPresets(data.presets);
        const first = data.presets.find((p) => p.id === "llamacpp") || data.presets[0];
        if (first) applyPreset(first);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const applyPreset = (preset: ProviderPreset) => {
    setDraftType(preset.id);
    setDraftUrl(preset.base_url || "");
    setDraftThinking(preset.thinking || "strip");
    setDraftTrustEnv(Boolean(preset.trust_env));
    if (!draftName) {
      setDraftName(preset.label);
    }
  };

  const ids = providers.map((p) => p.id);

  return (
    <div className="providers-card">
      <h3>模型 Providers</h3>
      {error ? <div className="toast-error">{error}</div> : null}
      {status ? <div className="form-status">{status}</div> : null}

      <div className="provider-list">
        {providers.map((p) => (
          <div key={p.id} className="provider-item">
            <div className="provider-item-header">
              <strong>{p.name}</strong>
              <span className="status-meta"> {p.type}</span>
              {p.id === defaultId ? (
                <span className="badge cache_hit">默认</span>
              ) : null}
            </div>
            <div className="provider-field">
              <label>base_url</label>
              <input
                type="text"
                value={p.base_url}
                onChange={(e) => {
                  setProviders((prev) =>
                    prev.map((x) =>
                      x.id === p.id ? { ...x, base_url: e.target.value } : x,
                    ),
                  );
                }}
              />
            </div>
            <div className="provider-field">
              <label>model</label>
              <input
                type="text"
                value={p.model}
                onChange={(e) => {
                  setProviders((prev) =>
                    prev.map((x) =>
                      x.id === p.id ? { ...x, model: e.target.value } : x,
                    ),
                  );
                }}
              />
            </div>
            <div className="provider-field">
              <label>api_key</label>
              <input
                type="password"
                value={p.api_key}
                placeholder={p.api_key ? "已设置（只写不读）" : ""}
                onChange={(e) => {
                  setProviders((prev) =>
                    prev.map((x) =>
                      x.id === p.id ? { ...x, api_key: e.target.value } : x,
                    ),
                  );
                }}
              />
            </div>
            <div className="provider-actions">
              <button
                type="button"
                className="btn-ghost bordered"
                disabled={testing === p.id}
                onClick={() => {
                  void (async () => {
                    setTesting(p.id);
                    setError(null);
                    try {
                      const res = await testProvider({
                        base_url: p.base_url,
                        api_key: p.api_key.includes("***") ? undefined : p.api_key || undefined,
                        model: p.model || undefined,
                        timeout: 10,
                      });
                      setTestResults((prev) => ({ ...prev, [p.id]: res }));
                    } catch (e) {
                      setTestResults((prev) => ({
                        ...prev,
                        [p.id]: {
                          ok: false,
                          connect_ms: 0,
                          generate_ms: 0,
                          model_list: [],
                          error: e instanceof Error ? e.message : String(e),
                        },
                      }));
                    } finally {
                      setTesting(null);
                    }
                  })();
                }}
              >
                {testing === p.id ? "测试中…" : "测试"}
              </button>
              <button
                type="button"
                className="btn-ghost bordered"
                onClick={() => {
                  void (async () => {
                    try {
                      await updateProvider(p.id, {
                        name: p.name,
                        base_url: p.base_url,
                        model: p.model,
                        api_key: p.api_key.includes("***") ? undefined : p.api_key || undefined,
                      });
                      setStatus(`已保存 ${p.name}`);
                      await reload();
                    } catch (e) {
                      setError(e instanceof Error ? e.message : String(e));
                    }
                  })();
                }}
              >
                保存
              </button>
              <button
                type="button"
                className="btn-ghost bordered"
                onClick={() => {
                  void (async () => {
                    try {
                      await setDefaultProvider(p.id);
                      setDefaultId(p.id);
                      setStatus(`已设为默认：${p.name}`);
                    } catch (e) {
                      setError(e instanceof Error ? e.message : String(e));
                    }
                  })();
                }}
              >
                设默认
              </button>
              <button
                type="button"
                className="btn-ghost bordered"
                onClick={() => {
                  void (async () => {
                    try {
                      await deleteProvider(p.id);
                      setStatus(`已删除 ${p.name}`);
                      await reload();
                    } catch (e) {
                      setError(e instanceof Error ? e.message : String(e));
                    }
                  })();
                }}
              >
                删除
              </button>
            </div>
            {testResults[p.id] ? (
              <div className="provider-test-result">
                {testResults[p.id].ok ? (
                  <span className="badge cache_hit">
                    连通 {testResults[p.id].connect_ms}ms · 生成{" "}
                    {testResults[p.id].generate_ms}ms
                  </span>
                ) : (
                  <span className="badge tool_error">
                    失败：{testResults[p.id].error}
                  </span>
                )}
                {testResults[p.id].model_list.length > 0 ? (
                  <span className="status-meta">
                    {" "}模型：{testResults[p.id].model_list.join(", ")}
                  </span>
                ) : null}
              </div>
            ) : null}
          </div>
        ))}
      </div>

      <div className="provider-add">
        <h4>添加 Provider</h4>
        <div className="provider-add-row">
          <select
            aria-label="预设模板"
            value={draftType}
            onChange={(e) => {
              const preset = presets.find((p) => p.id === e.target.value);
              if (preset) applyPreset(preset);
              else setDraftType(e.target.value);
            }}
          >
            {(presets.length > 0
              ? presets
              : [
                  { id: "llamacpp", label: "llama.cpp（本地）" },
                  { id: "deepseek", label: "DeepSeek" },
                  { id: "openai_compatible", label: "OpenAI 兼容" },
                ]
            ).map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="名称"
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
          />
          <input
            type="text"
            placeholder="base_url"
            value={draftUrl}
            onChange={(e) => setDraftUrl(e.target.value)}
          />
          <input
            type="text"
            placeholder="model"
            value={draftModel}
            onChange={(e) => setDraftModel(e.target.value)}
          />
          <input
            type="password"
            placeholder="api_key（可选）"
            value={draftKey}
            onChange={(e) => setDraftKey(e.target.value)}
          />
          <select
            aria-label="thinking"
            value={draftThinking}
            onChange={(e) => setDraftThinking(e.target.value)}
          >
            <option value="off">thinking: off</option>
            <option value="strip">thinking: strip</option>
            <option value="keep">thinking: keep</option>
          </select>
          <label className="status-meta">
            <input
              type="checkbox"
              checked={draftTrustEnv}
              onChange={(e) => setDraftTrustEnv(e.target.checked)}
            />{" "}
            trust_env
          </label>
          <button
            type="button"
            className="btn-primary"
            onClick={() => {
              void (async () => {
                if (!draftName || !draftUrl) return;
                try {
                  await createProvider({
                    name: draftName,
                    type: draftType,
                    base_url: draftUrl,
                    model: draftModel,
                    api_key: draftKey || undefined,
                    thinking: draftThinking,
                    trust_env: draftTrustEnv,
                  });
                  setStatus(`已添加 ${draftName}`);
                  setDraftName("");
                  setDraftKey("");
                  await reload();
                } catch (e) {
                  setError(e instanceof Error ? e.message : String(e));
                }
              })();
            }}
          >
            添加
          </button>
        </div>
      </div>

      <div className="routing-config">
        <h4>路由分配</h4>
        <div className="routing-row">
          <label>
            choose/fill
            <select
              value={routing.choose_fill}
              onChange={(e) => {
                const r = { ...routing, choose_fill: e.target.value };
                setRouting(r);
                void updateRouting(r).catch((e) =>
                  setError(e instanceof Error ? e.message : String(e)),
                );
              }}
            >
              {ids.map((id) => (
                <option key={id} value={id}>
                  {providers.find((p) => p.id === id)?.name || id}
                </option>
              ))}
            </select>
          </label>
          <label>
            generate
            <select
              value={routing.generate}
              onChange={(e) => {
                const r = { ...routing, generate: e.target.value };
                setRouting(r);
                void updateRouting(r).catch((e) =>
                  setError(e instanceof Error ? e.message : String(e)),
                );
              }}
            >
              {ids.map((id) => (
                <option key={id} value={id}>
                  {providers.find((p) => p.id === id)?.name || id}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
    </div>
  );
}
