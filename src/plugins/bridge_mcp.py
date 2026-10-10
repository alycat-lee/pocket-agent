from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, ClassVar
from urllib.parse import urljoin, urlparse

import httpx
from pocket_agent.plugins._config import agent_config_from
from pocket_agent.runtime import Context, PluginError
from pocket_agent.tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)

_WRITE_HINTS = (
    "send",
    "create",
    "delete",
    "update",
    "write",
    "remove",
    "put",
    "patch",
    "drop",
    "insert",
    "rename",
    "move",
    "mkdir",
    "unlink",
    "truncate",
    "upload",
    "destroy",
)
_PROTOCOL_VERSION = "2024-11-05"
_NAME_SAFE = re.compile(r"[^\w.\-]+")


def is_write_tool(name: str) -> bool:
    """启发式：名称含写类动词 → cacheable=False。"""
    lower = name.lower()
    return any(hint in lower for hint in _WRITE_HINTS)


def mcp_tool_name(remote_name: str) -> str:
    safe = _NAME_SAFE.sub("_", remote_name.strip()) or "tool"
    if safe.startswith("mcp_"):
        return safe
    return f"mcp_{safe}"


def content_to_str(result: Any) -> str:
    """把 tools/call 的 result 序列化为 str。"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item["text"]))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            text = "\n".join(parts)
            if result.get("isError"):
                return f"[MCP错误] {text}"
            return text
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def truncate_result(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit]


class McpClientError(PluginError):
    """MCP 协议 / 传输错误。"""


class StdioMcpClient:
    """JSON-RPC 2.0 MCP client（stdio，newline-delimited）。"""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
        name: str = "mcp",
    ) -> None:
        self.name = name
        self._timeout = timeout
        self._lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, Any]] = {}
        self._closed = False
        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        try:
            self._proc = subprocess.Popen(
                [command, *(args or [])],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=merged_env,
                cwd=cwd,
            )
        except OSError as exc:
            raise McpClientError(f"failed to start MCP server {name!r}: {exc}") from exc
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"mcp-stdio-{name}",
            daemon=True,
        )
        self._reader.start()
        self._initialize()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise McpClientError(f"{self.name}: tools/list missing tools array")
        return [t for t in tools if isinstance(t, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return self._request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        # 给进程一个从 stdin EOF 自行退出的窗口
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "MCP server %s (pid=%s) did not exit", self.name, proc.pid
                    )
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except OSError:
                pass
        deadline = time.time() + 3
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pocket-agent", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._write(msg)

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            if self._closed or not self.alive():
                raise McpClientError(f"{self.name}: client closed")
            req_id = self._next_id
            self._next_id += 1
            event = threading.Event()
            self._pending[req_id] = event
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": method,
                    "params": params,
                }
            )
        if not event.wait(self._timeout):
            with self._lock:
                self._pending.pop(req_id, None)
                self._responses.pop(req_id, None)
            raise McpClientError(f"{self.name}: timeout waiting for {method}")
        with self._lock:
            response = self._responses.pop(req_id, None)
            self._pending.pop(req_id, None)
        if response is None:
            raise McpClientError(f"{self.name}: empty response for {method}")
        if "error" in response:
            err = response["error"]
            raise McpClientError(f"{self.name}: {method} error: {err}")
        return response.get("result")

    def _write(self, msg: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise McpClientError(f"{self.name}: stdin closed")
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except OSError as exc:
            raise McpClientError(f"{self.name}: write failed: {exc}") from exc

    def _read_loop(self) -> None:
        stdout = self._proc.stdout
        if stdout is None:
            return
        try:
            for line in stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("MCP %s: skip non-json line: %s", self.name, line[:120])
                    continue
                if not isinstance(msg, dict):
                    continue
                if "id" not in msg:
                    continue  # notification / server→client request without handling
                req_id = msg["id"]
                if not isinstance(req_id, int):
                    continue
                with self._lock:
                    self._responses[req_id] = msg
                    event = self._pending.get(req_id)
                if event is not None:
                    event.set()
        except Exception:
            logger.debug("MCP %s reader stopped", self.name, exc_info=True)


class SseMcpClient:
    """可选 SSE 传输：GET 建立事件流，POST JSON-RPC 到 endpoint。"""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        name: str = "mcp-sse",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self._timeout = timeout
        self._url = url
        self._headers = dict(headers or {})
        self._lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, Any]] = {}
        self._closed = False
        self._endpoint: str | None = None
        # MCP server 通常是本地 stdio/sse，强制直连避免代理劫持
        self._client = httpx.Client(
            timeout=timeout, headers=self._headers, trust_env=False
        )
        self._stop = threading.Event()
        self._reader = threading.Thread(
            target=self._sse_loop,
            name=f"mcp-sse-{name}",
            daemon=True,
        )
        self._reader.start()
        deadline = time.time() + timeout
        while self._endpoint is None and not self._closed and time.time() < deadline:
            time.sleep(0.02)
        if self._endpoint is None:
            self.close()
            raise McpClientError(f"{self.name}: SSE endpoint not received from {url}")
        self._initialize()

    @property
    def pid(self) -> int | None:
        return None

    def alive(self) -> bool:
        return not self._closed and self._endpoint is not None

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise McpClientError(f"{self.name}: tools/list missing tools array")
        return [t for t in tools if isinstance(t, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return self._request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self._client.close()
        except Exception:
            logger.debug("MCP SSE %s client close failed", self.name, exc_info=True)
        self._reader.join(timeout=3)

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pocket-agent", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            if self._closed:
                raise McpClientError(f"{self.name}: client closed")
            req_id = self._next_id
            self._next_id += 1
            event = threading.Event()
            self._pending[req_id] = event
            self._post(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": method,
                    "params": params,
                }
            )
        if not event.wait(self._timeout):
            with self._lock:
                self._pending.pop(req_id, None)
                self._responses.pop(req_id, None)
            raise McpClientError(f"{self.name}: timeout waiting for {method}")
        with self._lock:
            response = self._responses.pop(req_id, None)
            self._pending.pop(req_id, None)
        if response is None:
            raise McpClientError(f"{self.name}: empty response for {method}")
        if "error" in response:
            raise McpClientError(f"{self.name}: {method} error: {response['error']}")
        return response.get("result")

    def _post(self, msg: dict[str, Any]) -> None:
        if not self._endpoint:
            raise McpClientError(f"{self.name}: no SSE endpoint")
        try:
            res = self._client.post(self._endpoint, json=msg)
            res.raise_for_status()
        except httpx.HTTPError as exc:
            raise McpClientError(f"{self.name}: POST failed: {exc}") from exc

    def _sse_loop(self) -> None:
        try:
            with self._client.stream(
                "GET",
                self._url,
                headers={"Accept": "text/event-stream", **self._headers},
            ) as response:
                response.raise_for_status()
                event_name = "message"
                data_lines: list[str] = []
                for line in response.iter_lines():
                    if self._stop.is_set():
                        break
                    if line is None:
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif line == "":
                        self._handle_sse_event(event_name, "\n".join(data_lines))
                        event_name = "message"
                        data_lines = []
                if data_lines:
                    self._handle_sse_event(event_name, "\n".join(data_lines))
        except Exception:
            if not self._closed:
                logger.debug("MCP SSE %s reader stopped", self.name, exc_info=True)
            self._closed = True

    def _handle_sse_event(self, event_name: str, data: str) -> None:
        if not data:
            return
        if event_name == "endpoint":
            endpoint = data.strip()
            if endpoint.startswith("/"):
                parsed = urlparse(self._url)
                origin = f"{parsed.scheme}://{parsed.netloc}"
                endpoint = urljoin(origin + "/", endpoint.lstrip("/"))
            self._endpoint = endpoint
            return
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict) or "id" not in msg:
            return
        req_id = msg["id"]
        if not isinstance(req_id, int):
            return
        with self._lock:
            self._responses[req_id] = msg
            event = self._pending.get(req_id)
        if event is not None:
            event.set()


def connect_mcp_server(spec: dict[str, Any]) -> StdioMcpClient | SseMcpClient:
    """根据 config 条目建立 MCP 连接。"""
    name = str(spec.get("name") or "mcp")
    transport = str(spec.get("transport") or "stdio").lower()
    timeout = float(spec.get("timeout") or 30.0)
    if transport == "stdio":
        command = spec.get("command")
        if not command:
            raise PluginError(f"mcp server {name!r}: stdio requires command")
        args = spec.get("args") or []
        if not isinstance(args, list):
            raise PluginError(f"mcp server {name!r}: args must be a list")
        env = spec.get("env") if isinstance(spec.get("env"), dict) else None
        cwd = spec.get("cwd")
        return StdioMcpClient(
            str(command),
            [str(a) for a in args],
            env=env,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            name=name,
        )
    if transport == "sse":
        url = spec.get("url")
        if not url:
            raise PluginError(f"mcp server {name!r}: sse requires url")
        headers = spec.get("headers") if isinstance(spec.get("headers"), dict) else None
        return SseMcpClient(
            str(url),
            timeout=timeout,
            name=name,
            headers={str(k): str(v) for k, v in (headers or {}).items()},
        )
    raise PluginError(f"mcp server {name!r}: unknown transport {transport!r}")


def adapt_mcp_tool(
    client: StdioMcpClient | SseMcpClient,
    mcp_tool: dict[str, Any],
    *,
    per_result_limit: int,
) -> Tool:
    remote_name = str(mcp_tool.get("name") or "tool")
    local_name = mcp_tool_name(remote_name)
    desc = str(mcp_tool.get("description") or remote_name).strip() or remote_name
    if len(desc) > 20:
        desc = desc[:20]
    schema = mcp_tool.get("inputSchema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    cacheable = not is_write_tool(remote_name)

    def _fn(**kwargs: Any) -> str:
        try:
            result = client.call_tool(remote_name, dict(kwargs))
        except McpClientError as exc:
            return f"[MCP错误] {exc}"
        return truncate_result(content_to_str(result), per_result_limit)

    _fn.__name__ = local_name
    return Tool(
        name=local_name,
        short_desc=desc,
        schema=schema,
        fn=_fn,
        cacheable=cacheable,
    )


class BridgeMcpPlugin:
    """MCP 桥接：stdio/sse → tools（requires: tools）。"""

    name = "bridge-mcp"
    requires: ClassVar[list[str]] = ["tools"]
    provides: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.clients: list[StdioMcpClient | SseMcpClient] = []

    def apply(self, ctx: Context, config: dict) -> None:
        registry: ToolRegistry = ctx.require("tools")
        ac = agent_config_from(config)
        limit = int(config.get("per_result_limit") or ac.per_result_limit)
        servers = config.get("mcp_servers") or []
        if not isinstance(servers, list):
            raise PluginError("bridge-mcp config.mcp_servers must be a list")
        if not servers:
            logger.info("bridge-mcp: no mcp_servers configured; skip")
            return

        self.clients = []

        def _close_all() -> None:
            for client in list(self.clients):
                try:
                    client.close()
                except Exception:
                    logger.debug("error closing MCP client", exc_info=True)
            self.clients.clear()

        ctx.on_dispose(_close_all)

        for spec in servers:
            if not isinstance(spec, dict):
                raise PluginError(f"invalid mcp server entry: {spec!r}")
            if spec.get("disabled"):
                continue
            client = connect_mcp_server(spec)
            self.clients.append(client)
            try:
                remote_tools = client.list_tools()
            except McpClientError:
                client.close()
                self.clients.remove(client)
                raise
            for mcp_tool in remote_tools:
                tool = adapt_mcp_tool(client, mcp_tool, per_result_limit=limit)
                disposer = registry.register_with_disposer(tool)
                ctx.on_dispose(disposer)
            logger.info(
                "bridge-mcp: connected %s (%s tools)",
                client.name,
                len(remote_tools),
            )
