#!/usr/bin/env python3
"""Minimal MCP stdio echo server for bridge-mcp tests / demos.

Protocol: newline-delimited JSON-RPC 2.0
Methods: initialize / tools/list / tools/call
Tools: echo (read) / write_note (write heuristic)
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _send(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle(req: dict[str, Any]) -> None:
    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        _send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-echo", "version": "0.1.0"},
                },
            }
        )
        return

    if method == "notifications/initialized":
        return

    if method == "tools/list":
        _send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "回显输入文本",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                },
                                "required": ["text"],
                            },
                        },
                        {
                            "name": "write_note",
                            "description": "模拟写笔记",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                },
                                "required": ["text"],
                            },
                        },
                    ]
                },
            }
        )
        return

    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            text = f"echo:{args.get('text', '')}"
        elif name == "write_note":
            text = f"wrote:{args.get('text', '')}"
        else:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"unknown tool: {name}"},
                }
            )
            return
        _send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                },
            }
        )
        return

    if req_id is not None:
        _send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        )


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(req, dict):
            _handle(req)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
