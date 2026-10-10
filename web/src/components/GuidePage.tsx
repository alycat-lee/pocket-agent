interface Props {
  baseUrl?: string;
  onRetry: () => void;
}

export function GuidePage({ baseUrl, onRetry }: Props) {
  const url = baseUrl || "http://127.0.0.1:8080";
  return (
    <div className="guide">
      <div className="guide-card">
        <h2>llama-server 未连接</h2>
        <p>
          本地模型服务不可达。请先启动 llama-server（或其它 OpenAI 兼容接口），再重试。
        </p>
        <pre>{`# 示例：启动 llama-server
./llama-server -m model.gguf --port 8080
# 当前配置 base_url
${url}`}</pre>
        <button type="button" className="btn-primary" onClick={onRetry}>
          重试连接
        </button>
      </div>
    </div>
  );
}
