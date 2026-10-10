interface Props {
  baseUrl?: string;
  detail?: string | null;
  onRetry: () => void;
}

export function GuidePage({ baseUrl, detail, onRetry }: Props) {
  const url = baseUrl || "http://127.0.0.1:8000/v1";
  const is502 = Boolean(detail && /502|Bad Gateway|推理服务异常/i.test(detail));
  return (
    <div className="guide">
      <div className="guide-card">
        <h2>{is502 ? "llama-server 推理失败 (502)" : "llama-server 未就绪"}</h2>
        <p>
          {is502
            ? "端口可能仍在监听，但模型推理已不可用（常见于显存不足、模型加载失败或服务僵死）。请查看 llama-server 终端日志并重启。"
            : "本地模型服务不可达。请先启动 llama-server（或其它 OpenAI 兼容接口），再重试。"}
        </p>
        {detail ? <p className="status-meta">详情：{detail}</p> : null}
        <pre>{`# 示例：重启 llama-server（窗口需足够大）
./llama-server -m your-model.gguf -c 8192 --port 8000
# 当前配置 base_url
${url}
# 自检
curl ${url.replace(/\/v1\/?$/, "")}/health
curl -X POST ${url}/chat/completions -H "Content-Type: application/json" -d "{\\"model\\":\\"local-model\\",\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"hi\\"}],\\"max_tokens\\":8}"`}</pre>
        <button type="button" className="btn-primary" onClick={onRetry}>
          重试连接
        </button>
      </div>
    </div>
  );
}
