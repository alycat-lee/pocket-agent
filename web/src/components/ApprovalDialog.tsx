import { useEffect, useState } from "react";
import type { ApprovalRequest } from "../api/types";

interface Props {
  request: ApprovalRequest;
  onResolve: (approved: boolean) => void;
}

export function ApprovalDialog({ request, onResolve }: Props) {
  const [left, setLeft] = useState(() =>
    Math.max(0, Math.ceil((request.expiresAt - Date.now()) / 1000)),
  );

  useEffect(() => {
    const t = window.setInterval(() => {
      const sec = Math.max(0, Math.ceil((request.expiresAt - Date.now()) / 1000));
      setLeft(sec);
      if (sec <= 0) {
        window.clearInterval(t);
        onResolve(false);
      }
    }, 250);
    return () => window.clearInterval(t);
  }, [request, onResolve]);

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <div className="modal">
        <h3>需要审批</h3>
        <p>
          工具 <strong>{request.tool || "unknown"}</strong> 请求执行
          {request.reason ? `（${request.reason}）` : ""}。
          {left > 0 ? ` ${left}s 后自动拒绝。` : " 已超时，将拒绝。"}
        </p>
        <pre>{JSON.stringify(request.args ?? {}, null, 2)}</pre>
        <div className="modal-actions">
          <button
            type="button"
            className="btn-ghost bordered"
            onClick={() => onResolve(false)}
          >
            拒绝
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={() => onResolve(true)}
          >
            批准
          </button>
        </div>
      </div>
    </div>
  );
}
