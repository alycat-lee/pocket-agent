# 30 — 内部调用可观测与真实 Token 计量（Phase 43）

> 编号接 Phase 42。起因：用户开了服务端详细日志后抓到一次"标题生成"请求，轨迹里却看不到；同时日志暴露 thinking off 对 R1 类模型无效、token 用量有真实值我们却用估算。

---

## 一、诊断（基于用户抓取的 llama-server 日志）

1. **观测盲区**：该请求是 `_maybe_rename` 的标题生成（prompt=「为对话生成不超过 12 字标题…」）。改名/指代补全/摘要/planner/验收重试这些内部 LLM 调用只发业务事件或不发事件，**轨迹里完全不可见**——用户看到模型在跑却不知道在干什么。
2. **thinking off 假关闭**：请求带 `"chat_template_kwargs":{"enable_thinking":false}`，但 DeepSeek-R1-Distill 模板不支持该开关，照样输出 165 token 的 `reasoning_content`。客户端目前把它丢弃（无泄漏），但**每次调用白烧一倍 token、延迟翻倍**。
3. **真实 usage 被浪费**：响应里有完整 `usage`（prompt_tokens=24, completion_tokens=165, cached_tokens=2），llm_client 完全不解析，轨迹结算行还在用字符数/1.5 的估算。

---

## 二、Phase 43 提示词（贴给 Cursor）

```text
你是一名资深全栈工程师，继续开发 pocket-agent（本地小模型 agent 框架）。本期三件事：内部 LLM 调用进轨迹、真实 token 计量、thinking 假关闭自动降级。红线：内部调用事件只放摘要（prompt/response 各取前 120 字符），全文仍在 Phase 42 的调试日志；不改 build 脚本；所有新行为必须有测试。

═══ 任务 A：内部 LLM 调用进轨迹 ═══
1. session.py：_maybe_rename、_rewrite_if_needed、_maybe_roll_summary（summarize）、_verify_answer 重试、planner.plan 调用处，成功后 emit：
       self._events.emit("llm.call", {
           "session_id": sid, "turn": 当前轮次,
           "purpose": "rename" | "rewrite" | "summarize" | "verify_retry" | "planner",
           "prompt_preview": prompt[:120],
           "response_preview": 结果[:120],
           "elapsed_ms": 计时,
           "usage": 客户端 last_usage（见任务 B，可能为 None 则不携带该键）,
       })
   异常路径不发（保持现有静默兜底语义）。
2. ui_web.py：_BUS_EVENTS 与 SSE 名单加 "llm.call"（可落盘，单条很小）。
3. 前端 trajectory.ts：kindOf 加 llm.call → "other"；titleOf → `内部调用：${{rename:"标题生成",rewrite:"指代补全",summarize:"摘要压缩",verify_retry:"验收重试",planner:"任务规划"}[purpose] || purpose}`；记录行弱化显示（复用 status-meta 样式），detail 里展示 prompt_preview/response_preview/elapsed_ms/usage。
4. 测试：fake LLM 跑一轮触发改名，断言事件含 purpose="rename" 与 preview；前端 projectEvents 渲染为单行弱化记录。

═══ 任务 B：真实 token 计量 ═══
1. llm_client：解析非流式响应的 usage 字段（prompt_tokens/completion_tokens/total_tokens，prompt_tokens_details.cached_tokens 有则收），存 self.last_usage: dict | None；流式路径若响应含 usage 块同样解析，没有则为 None。
2. RoutedLLM 透传 last_usage（_remember 时同步）。
3. runner/session 结算：answer.done payload 增加 "usage" 字段（取自 last_usage，无则不携带）；prompt_tokens_max 仍保留估算值，不改名（前端兼容）。
4. 前端结算行：usage 存在时追加显示 `· tokens {completion_tokens}↑/{prompt_tokens}↓`，不存在不显示。
5. 测试：mock 响应带 usage → last_usage 解析正确 → answer.done 携带；无 usage 字段的旧服务不报错。

═══ 任务 C：thinking 假关闭自动降级 ═══
1. llm_client：thinking=="off" 的响应若仍带非空 reasoning_content → 自动按 strip 处理（思维链进 last_thinking 不进正文——现有逻辑已是），并记 warning 日志一次（每进程一次）："模型模板不支持 enable_thinking，已按 strip 处理"；同时把 self._thinking 改为 "strip"（后续请求不再发假开关）。
2. PROVIDER_PRESETS 的 llamacpp 预设默认 thinking 改 "strip"（off 对 R1 类模型是假的，strip 是安全默认）。
3. 测试：fake 响应带 reasoning_content + thinking=="off" → 正文不含思维链、_thinking 降级为 strip、日志有 warning。

═══ 验收（必须全绿再交付）═══
1. pytest -q && cd web && pnpm test && pnpm build
2. grep -rn "MockLLM" src/ 无输出
3. 手测清单写进交付说明：新会话发"你是谁"→ 轨迹除问答记录外出现"内部调用：标题生成"弱化行，detail 可见 prompt 摘要与真实 token 数；结算行显示真实 tokens；R1 模型下 thinking=off 自动降级且回答无思维链泄漏。
```

---

## 三、使用方式

1. 本文档放入 `pocket-agent-docs/`。
2. 贴"二、Phase 43 提示词"整段给 Cursor，一次完成 A–C。
3. 前端有改动，`pnpm build` + Ctrl+F5 强刷再测。
4. 验收结果贴回来。完成后建议 Phase 44 进入真实基准（D1-D9 三轨出数字）。
