import { useCallback, useEffect, useState } from "react";
import {
  deleteKnowledgeDocument,
  fetchKnowledgeDocuments,
  fetchKnowledgeStatus,
  searchKnowledge,
  uploadKnowledgeFile,
  type KnowledgeDocument,
  type KnowledgeHit,
} from "../api/client";

export function KnowledgeCard() {
  const [available, setAvailable] = useState(false);
  const [reason, setReason] = useState("");
  const [embedder, setEmbedder] = useState("");
  const [docs, setDocs] = useState<KnowledgeDocument[]>([]);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<KnowledgeHit[]>([]);
  const [hitText, setHitText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    const [status, listing] = await Promise.all([
      fetchKnowledgeStatus(),
      fetchKnowledgeDocuments(),
    ]);
    setAvailable(status.available);
    setReason(status.reason || listing.reason);
    setEmbedder(status.embedder);
    setDocs(listing.documents);
  }, []);

  useEffect(() => {
    void reload().catch((e) =>
      setError(e instanceof Error ? e.message : String(e)),
    );
  }, [reload]);

  return (
    <div className="prompts-card knowledge-card">
      <h3>知识库</h3>
      <p className="status-meta">
        嵌入后端：<strong>{embedder || "—"}</strong>
        {available ? " · 已启用" : " · 未启用"}
      </p>
      {!available && reason ? (
        <div className="toast-error">{reason}</div>
      ) : null}
      {error ? <div className="toast-error">{error}</div> : null}

      <div className="kb-upload">
        <label className="btn-ghost bordered kb-file-label">
          上传文档
          <input
            type="file"
            accept=".txt,.md,.pdf,text/plain,text/markdown,application/pdf"
            disabled={busy || !available}
            onChange={(ev) => {
              const file = ev.target.files?.[0];
              ev.target.value = "";
              if (!file) return;
              setBusy(true);
              setError(null);
              void uploadKnowledgeFile(file)
                .then(() => reload())
                .catch((e) =>
                  setError(e instanceof Error ? e.message : String(e)),
                )
                .finally(() => setBusy(false));
            }}
          />
        </label>
        <span className="status-meta">支持 txt / md / pdf</span>
      </div>

      <h4>文档列表</h4>
      {docs.length === 0 ? (
        <div className="empty-hint">还没有入库文档</div>
      ) : (
        <ul className="kb-doc-list">
          {docs.map((doc) => (
            <li key={doc.id} className="kb-doc-item">
              <div>
                <strong>{doc.name}</strong>
                <span className="status-meta">
                  {" "}
                  · {doc.chunks} 块 · {doc.source}
                </span>
              </div>
              <button
                type="button"
                className="btn-ghost bordered"
                disabled={busy}
                onClick={() => {
                  setBusy(true);
                  setError(null);
                  void deleteKnowledgeDocument(doc.id)
                    .then(() => reload())
                    .catch((e) =>
                      setError(e instanceof Error ? e.message : String(e)),
                    )
                    .finally(() => setBusy(false));
                }}
              >
                删除
              </button>
            </li>
          ))}
        </ul>
      )}

      <h4>命中测试</h4>
      <div className="kb-search">
        <input
          type="text"
          value={query}
          placeholder="输入 query 看召回结果"
          onChange={(ev) => setQuery(ev.target.value)}
        />
        <button
          type="button"
          className="btn-primary"
          disabled={busy || !query.trim()}
          onClick={() => {
            setBusy(true);
            setError(null);
            void searchKnowledge(query.trim())
              .then((res) => {
                setHits(res.hits);
                setHitText(res.text);
              })
              .catch((e) =>
                setError(e instanceof Error ? e.message : String(e)),
              )
              .finally(() => setBusy(false));
          }}
        >
          检索
        </button>
      </div>
      {hitText ? <pre className="prompt-template kb-hit-text">{hitText}</pre> : null}
      {hits.length > 0 ? (
        <ul className="kb-hit-list">
          {hits.map((hit) => (
            <li key={`${hit.doc_id}-${hit.source}-${hit.score}`}>
              <span className="badge cache_hit">{hit.source}</span>
              <span className="status-meta">
                {" "}
                {hit.score.toFixed(3)}
              </span>
              <div>{hit.text}</div>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
