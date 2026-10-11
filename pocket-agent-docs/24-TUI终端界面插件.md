# 24 — TUI 终端界面插件（Phase 38）

> 编号接 Phase 37。目标：按"一切皆插件"哲学，为 pocket-agent 新增本地终端界面（TUI），与 Web UI 平级、共用同一份会话数据。

---

## 一、调研结论：TUI 是什么，DSH 的 TUI 怎么做

**TUI（Terminal User Interface）**：终端内的全屏交互界面——有布局、快捷键、滚动区、状态行，但不用浏览器。Claude Code、opencode 是代表。

**DSH 的情况**：官方界面只有 Web UI，终端交互面是空白的；社区插件 dsh-TUI 补上了这块。它的机制正是我们"一切皆插件"要对标的：

| dsh-TUI 机制 | 我们的映射 |
|---|---|
| 独立 `tui` profile 挂载，纯插件、零核心改动，卸载不留补丁 | pocket.yml 加 `tui` profile，`ui-tui` 插件走同一装配机制，不碰 runner/session |
| 会话日志是唯一真源，TUI 只做交互与呈现投影；与 Web/headless 共用同一份会话数据 | 复用 SessionManager + EventStore：TUI 里聊的会话 Web 能看到，反之亦然 |
| 事件驱动渲染（订阅 session/event 增量渲染） | 订阅事件总线（intent/task/step/answer.delta/answer.done）本地投影 |
| 斜杠命令体系（/new /resume /model /export /quit） | 一期实现 /help /new /sessions /resume /agent /model /export /stop /quit |
| inline 审批（y/n，Esc 取消） | ui-tui 自设 `approval_gate` 为终端问答（无 ui-web 时 gate 为 None 会跳过审批，必须补） |
| 状态行（轮次/步数/耗时/缓存命中率） | 用 answer.done 的 settlement 字段打印单行结算 |
| 流式逐字渲染 | 消费 Phase 37 的 answer.delta |
| Ctrl+C 中断当轮 | 调 request_stop（已有） |

**技术选型**：Python 实现，用 **prompt_toolkit**（纯 Python、跨平台、Windows 友好，处理按键/多行输入/历史），作为可选依赖 `pocket-agent[tui]`，未安装时插件给出明确安装指引。不引入 Textual 等重型框架——本地小工具保持轻量。

---

## 二、Phase 38 提示词（贴给 Cursor）

```text
你是一名资深 Python 工程师，继续开发 pocket-agent（本地小模型 agent 框架）。本期新增 TUI 终端界面插件。红线：一切皆插件——不得改动 runner/session/verifier 等引擎代码；TUI 只消费 ctx 服务与事件总线；禁止 MockLLM 进 src/；与 Web UI 共用同一份会话存储；所有新行为必须有测试。

═══ 任务 A：profile 与插件骨架 ═══
1. src/pocket_agent/plugins/ui_tui.py 新建 `UiTuiPlugin`：
   - name = "ui-tui"；requires = ["sessions"]（session 插件已依赖 agent/writer）；provides = ["tui"]。
   - apply(ctx, config)：注册 TuiApp 实例为 "tui" 服务（含 run() 方法）；设置审批 gate（见任务 C）；订阅事件做渲染投影（任务 D）。
   - 未安装 prompt_toolkit 时 raise PluginError("TUI 需要 prompt_toolkit：pip install 'pocket-agent[tui]' 或 pip install prompt_toolkit")。
2. src/pocket_agent/runtime/__init__.py：_HEADLESS_SKIP 集合加 "ui-tui"（headless 跳过）；新增 profile 语义 "tui"：跳过 ui-web、挂载 ui-tui。实现方式：把 _HEADLESS_SKIP 改为按 profile 的映射——headless 跳过 {ui-web, ui-tui}；web 跳过 {ui-tui}；tui 跳过 {ui-web}。
3. src/pocket_agent/plugins/__init__.py builtin_plugins 注册 "ui-tui"。
4. pocket.yml：plugins 列表追加 `- name: ui-tui`（默认不 disabled；headless/web profile 下会被跳过，不影响现状），顶部注释补一行：`# tui：python cli.py serve --profile tui 进入终端界面`。
5. pyproject.toml 加可选依赖：`[project.optional-dependencies] tui = ["prompt_toolkit>=3.0"]`。
6. cli.py：serve --profile tui 装配完成后，若 ctx.has("tui") 则调用 ctx.require("tui").run() 进入交互循环（阻塞至退出后正常 unmount）。

═══ 任务 B：交互循环（prompt_toolkit）═══
TuiApp.run()：
- 布局：上方滚动消息区 + 底部多行输入框（Enter 发送，Esc+Enter 或 Ctrl+J 换行）；顶部一行标题栏显示 `pocket-agent · {provider}·{model} · agent:{当前agent}`。
- 发送：调 sessions.chat(session_id, text, agent=当前agent)——必须在工作线程执行，主线程保持 UI 响应；运行中状态行显示"运行中 · S{step} · {tool}…"（消费 step.* 事件）。
- 渲染：用户消息前缀 `❯`；助手回答支持基础 Markdown（标题/粗体/代码块用 prompt_toolkit 的 formatted text 即可，不做完整渲染器）；流式 answer.delta 增量追加当前回答；answer.done 定稿并打印结算行：`轮次 N · M 步 · 缓存命中 xx% · 拦截 K · 停止原因`。
- Ctrl+C：运行中 → request_stop（保留已生成内容）；空闲连按两次退出。/quit 与 Ctrl+D 退出。
- 启动时恢复最近会话列表最后活跃会话（sessions.list_sessions 取 updated_at 最大者），无则新建临时会话 id（t-<时间戳>）。
- 会话互通验收：TUI 聊过的会话，启动 web profile 后能在会话列表看到且内容完整（events 落盘 + turns 持久化已有，不许另起存储）。

═══ 任务 C：inline 审批 ═══
- apply 时设置 agent.approval_gate（若 ui-web 未挂载、gate 为 None）：终端阻塞式问答——打印 `审批 · {tool} · {args摘要} · {reason}`，读单键 y（允许一次）/ n（拒绝）/ Esc（拒绝）；60 秒超时按拒绝处理并提示"审批超时已拒绝"。
- 测试：fake gate 场景单测审批问答函数的三分支（y/n/超时），用 monkeypatch 替代真实按键。

═══ 任务 D：事件投影与斜杠命令 ═══
1. 事件订阅（渲染投影，全部只读）：
   - intent.detected → 打印 `意图：{mode}（{source}）` 弱化色
   - task.planned → 打印 `任务卡：{goal}` + plan 步骤列表
   - step.tool_call → `● 工具 {name} {args单行摘要}`，完成时同行更新 `✓/✖`
   - step.intercepted → `已拦截：{reason}`
   - rag.empty → `未检索到资料，由模型自身作答`
   - answer.rejected → `验收未通过：{reasons}，正在重试`
   - answer.delta / answer.done 见任务 B
2. 斜杠命令（输入 / 开头时拦截，不进模型）：
   - /help 列出全部命令；/new 新会话；/sessions 列出当前 agent 会话（序号+标题+更新时间）；/resume <序号|id> 恢复并从 GET 详情重建消息显示；/agent <名> 切换 agent（无参列出 agents）；/model 列出 providers（序号+name+model+默认标记），/model <序号> 切换默认路由（复用 providers 模块读写 settings，与 Web 设置页同一份数据）；/export 导出当前会话到 ./<sid>.md；/stop 同 Ctrl+C；/quit 退出。
   - 未知命令提示 /help。
3. 测试：命令解析器纯函数单测（每个命令的解析与分派）；事件投影函数单测（给定 payload 产出预期文本行）。

═══ 任务 E：测试与验收 ═══
1. 事件投影、命令解析、审批问答全部为纯函数单测（不启动真 TUI）。
2. TuiApp 装配测试：headless profile 不含 tui 服务；tui profile（monkeypatch prompt_toolkit 存在）装配后有 tui 服务且无 ui-web 路由。
3. pytest -q 全绿；grep -rn "MockLLM" src/ 无输出。
4. 手测清单写进交付说明：python cli.py serve --profile tui 启动→输入"你是谁"流式回答→/model 切换→写文件触发审批 y 通过→Ctrl+C 中断保留已生成→/quit；转 web profile 同会话内容完整。
```

---

## 三、使用方式

1. 本文档放入 `pocket-agent-docs/`。
2. 贴"二、Phase 38 提示词"整段给 Cursor，一次完成 A–E。
3. 本期纯后端/终端，不涉及 `pnpm build`。
4. 跑完先 `pip install prompt_toolkit`，再 `python cli.py serve --profile tui` 实测，验收四条结果贴回来。
