# 09 · Web 界面升级与配置外化（Phase 17~18）

> 前置条件：Phase 13~15 完成（插件运行时 + ui-web 插件已存在）。本文档对齐 DeepSeek Harness **官方原生** Web UI 的信息架构重做界面（第三方插件界面一律不参考），并由界面元素反推配置外化需求。Phase 18 要求 Cursor 调用 `/ui-ux-pro-max` 与 `/impeccable` 两个 skill 完成界面生成与审查。

## 1. 界面信息架构（对齐 DSH 官方原生 Web UI）

DSH 原生界面最核心的结构是**会话页头的视图标签环**：「对话」和「轨迹」是两个并列标签，点击切换——对话区只展示问答，轨迹是独立视图，把模型请求和工具调用按时间排开。步骤信息绝不内嵌进消息流。pocket-agent 的差异化素材（账本拦截、缓存命中、预算控制）全部放进轨迹视图呈现。

```
┌────────────┬───────────────────────────────────────────┐
│ 左侧边栏    │ 会话页头：标题 │ 视图标签 [对话] [轨迹]     │
│            ├───────────────────────────────────────────┤
│            │ ⚡实时状态条（运行中才显示，仅对话视图）     │
│            ├───────────────────────────────────────────┤
│ 会话列表    │                                           │
│ + 新建会话  │ 【对话视图 · 默认】纯净问答：              │
│            │   用户消息 / 助手回答（流式）              │
│            │   审批弹窗、错误提示                       │
│            │   —— 无任何步骤卡片 ——                    │
│            │                                           │
│            │ 【轨迹视图 · 点击标签切换】                 │
│            │   顶部：时间概览条（横向时间投影）          │
│            │   主体：按轮次组织的事件记录表              │
│            │   点击记录 → 检查器（tokens/耗时/输入/输出）│
│ ──────────│   右上：导出 session log                   │
│ ⚙ 设置     ├───────────────────────────────────────────┤
│ 🧩 插件     │ 模式选择器 │ 权限选择器（输入框上方）       │
│            │ 输入框 + @文件引用 + 发送/停止              │
│            │ 输入框底部：上下文用量（点击展开明细）       │
└────────────┴───────────────────────────────────────────┘
```

### 1.1 视图标签环（对话 | 轨迹）

- 标签位于会话页头，按会话记住上次所在视图
- 标签环可扩展：后续视图（如长文写作预览）以新标签注册进环——这是官方"对话视图环"的开放结构
- 运行中无论停在哪个视图，事件都持续落盘，切换视图不丢数据

### 1.2 对话视图（默认）

- 只有：用户消息、助手回答（流式渲染）、审批弹窗、错误提示、实时状态条
- 助手回答旁显示一行轻量结算（`N 步 · 缓存命中 X · 拦截 Y`），点击即切换到轨迹视图定位本轮——这是消息流与轨迹之间唯一的桥梁
- llama-server 未连接时显示引导页（启动命令 + 重试按钮），不显示裸报错

### 1.3 轨迹视图（差异化核心，逐条对齐官方 ui-trajectory 包行为）

**事件记录表**：
- 按轮次组织，分组渲染：用户 / 助手 / 工具调用 / 拦截事件 / 压缩事件，明确标示轮次边界与步骤边界
- 徽章体系落在记录上：`cache_hit`（绿）、`已拦截`（黄）、`工具错误`（红）；失败记录显示错误代码
- 点击记录打开**检查器**：token 用量、耗时、输入 prompt、输出结果、参数全文
- 长历史行为：打开时定位到当前尾部；点击顶部"加载更早"按需翻页；虚拟滚动只渲染可见行
- 流式期间视图跟随尾部，用户向上滚动即暂停跟随（新记录不打断回看），回到底部恢复跟随
- **进行中的记录只显示开始标记，不虚构耗时**——没有真实数据就保持空白
- 助手记录的时间条区分 TTFT（步骤开始到首个 token）与解码时间；若后端流中没有 token 时间戳，对应指标显示"不可用"，不估算充数

**时间概览条**（固定在记录表上方，P0 静态投影 / P1 交互）：
- P0：从左到右投影各记录的真实开始时间与耗时，徽章着色
- P1：拖选区间聚焦记录表、滚轮缩放时间域、右键清除选择、悬停 500ms 显示精确时刻

**纯投影原则**：轨迹视图从 session 事件流投影组装，只读不改写对话数据——对话视图与轨迹视图共享同一份事件真相源，但各自独立渲染。

**导出**：右上角"导出 session log"按钮，导出前弹提示"日志可能包含路径、prompt 与工具参数，分享前请脱敏"。

### 1.4 实时状态条（仅对话视图，用户定制保留）

位置在会话页头与消息流之间，**运行中才渲染、空闲时完全收起不占布局**。双行结构：

- 速览行（常显）：阶段图标 + 当前活动文案（`正在选择工具…` → `正在调用 get_weather` → `等待审批…`）、第 N/max 步、运行秒表、上下文预算占用条（`prompt_tokens_est / prompt_char_budget` 百分比，超 80% 变黄、超 95% 变红）
- 过程行（点击展开）：本 run 已完成步骤的迷你时间线（choose/fill/工具/拦截），当前节点呼吸动画；节点点击 = 切换到轨迹视图并定位对应记录
- 轮次结算条（run 结束后原地停留 5 秒再收起）：`N 步 · LLM Xm · 工具 Xm · 缓存命中 99% · 真实调用 X 次 · 拦截 Y 次`；**命中率不谎报**——99.5% 不许四舍五入成 100%，只有真满命中才显示 100%

数据全部来自事件总线经 SSE 推送的扩展字段（见 §2），事件驱动刷新，禁止前端轮询。状态条展开/收起偏好存 localStorage，不进后端配置。

### 1.5 输入区（对齐官方布局）

- 输入框**上方**：模式选择器（问答 / 工具调用 / 长文写作，按会话切换工具面）、权限选择器（只读 / 仅写操作需确认 / 自动放行——映射审批策略）
- 输入框：@ 文件引用（映射 read_file 预注入）、发送/停止按钮
- 输入框**底部**：上下文用量一行（`≈2.1k / 5.2k 字符`），点击展开明细（system / 状态摘要 / 历史 / 本步信息各区段占用）

## 2. 配置外化反推（界面元素 → 后端需求）

从上述界面反推：**设置页每个表单项都对应一个不许再硬编码的参数**。全部外化项如下：

| 设置分组 | 外化参数 | 来源 |
|---|---|---|
| 模型 | base_url / model / api_key / temperature / repeat_penalty / top_p / max_tokens / timeout | AgentConfig |
| 上下文预算 | prompt_char_budget / per_result_limit / summarize_input_limit / use_llm_summarize | AgentConfig |
| 护栏 | max_steps / max_consecutive_intercepts | AgentConfig |
| 会话 | keep_full_turns / session_summary_limit | AgentConfig |
| 长文写作 | section_char_target / seam_tail_chars / overlap_max | AgentConfig |
| 工具 | tool_recall_threshold / tool_recall_top_n | AgentConfig |
| 模型路由 | simple_tasks 走本地 / complex_tasks 走云端的路由开关 | Phase 14 llm-deepseek |
| 实时状态条与轨迹数据源 | SSE 事件扩展字段：step_index / elapsed_ms / current_tool / prompt_tokens_est / ttft_ms / 轮次结算数据 | AgentRunner 事件总线 |

配套架构改造（Phase 17 的核心）——**三层配置优先级**：

```
单次请求覆盖（前端临时调整）> 用户设置（settings.json，设置页持久化）> pocket.yml 默认值
```

后端必须新增的 API：

| API | 用途 | 驱动界面 |
|---|---|---|
| `GET /api/config/schema` | 全部配置项元数据（key/类型/范围/分组/说明） | **设置页表单由 schema 自动渲染**，新增配置项零前端改动 |
| `GET /api/config` / `PUT /api/config` | 读写用户设置层 | 设置页保存 |
| `POST /api/chat` 增加 `overrides` 字段 | 单次请求覆盖 | 模式/权限/模型临时切换 |
| `GET /api/sessions/{id}/events?before={cursor}&limit=N` | 事件历史分页（从尾部向前），轨迹视图数据源 | 轨迹视图 + 新打开页面回放 |
| `GET /api/sessions/{id}/export` | 导出 session log（JSON） | 轨迹视图导出按钮 |
| `GET /api/plugins` | 已挂载插件与状态 | 插件页 |
| `POST /api/plugins/{name}/enable\|disable` | 运行时挂载/卸载 | 插件页开关 |
| `POST /api/chat/{run_id}/stop` | 中断运行 | 停止按钮 |

两项硬约束：
- **事件持久化**：事件总线的全部步骤事件必须同步落盘（复用 checkpoint 事件流），轨迹视图才能在新打开页面时回放历史——SSE 只负责实时推送，不是存储
- **api_key 只写不读**：GET /api/config 永远返回脱敏值（`sk-***`），对齐 DSH 凭证独立存放、不回传明文的做法

## 3. 设置页与插件页规格

**设置页**：左侧分组导航（模型/上下文/护栏/会话/长文/工具/路由），右侧表单区。表单控件由 `/api/config/schema` 的字段元数据驱动：数值型用带范围限制的 number input、布尔用开关、枚举用下拉。每个字段带说明 tooltip（文案来自 schema 的 description）。底部：保存（写 settings.json，即时生效不重启）、恢复默认。

**插件页**：卡片列表，每个插件显示名称、提供的服务、依赖、状态；开关切换调 enable/disable，卸载被 PluginInUse 拒绝时弹出提示说明被谁依赖。

---

## Phase 17 · 配置外化与运行时覆盖

````
@docs/03-模块实现规格.md @pocket-agent-docs/08-插件化架构改造.md @pocket-agent-docs/09-Web界面与配置外化.md

实现三层配置体系与配套 API（09 文档 §2）：

1. AgentConfig 增加 from_layers()：pocket.yml 默认 + settings.json 用户层
   + 请求级 overrides 字典，三层合并，后者优先
2. 新增 config_schema.py：以声明式结构描述全部配置项
   （key/type/min/max/enum/group/description），是设置页的数据源，
   新增配置项必须在此登记
3. ui-web 插件新增 API：GET /api/config/schema、GET/PUT /api/config
   （api_key 脱敏返回）、POST /api/chat 支持 overrides 字段、
   GET /api/plugins、POST /api/plugins/{name}/enable|disable、
   POST /api/chat/{run_id}/stop、
   GET /api/sessions/{id}/events（游标分页，从尾部向前）、
   GET /api/sessions/{id}/export
4. AgentRunner 支持中断：每步开头检查 stop 标志，
   触发后走 final_answer 优雅收尾，stop_reason 增加 STOPPED
5. 事件持久化：全部步骤事件同步落盘（复用 checkpoint 事件流），
   events API 从落盘数据分页读取；SSE 只做实时推送
6. SSE 事件 payload 扩展（状态条与轨迹视图的数据源）：
   step.start / step.tool_call / step.cache_hit / step.intercepted
   各事件携带 step_index、elapsed_ms、current_tool、prompt_tokens_est、
   ttft_ms（无 token 时间戳时为 null，前端显示"不可用"）；
   answer.done 携带整轮结算
   {llm_ms, tool_ms, cache_hit_rate, real_calls, intercepts, prompt_tokens_max}
7. settings.json 写入做原子操作（写临时文件再 rename），
   损坏时回退 pocket.yml 默认并告警
8. 测试：三层优先级合并正确性、overrides 不污染 settings.json、
   schema 完整性（AgentConfig 每个字段都在 schema 中登记）、
   stop 中断路径、api_key 脱敏、SSE 事件字段完整性、
   events 分页游标正确性（顺序、边界、空页）

验收：pytest 全绿；curl PUT /api/config 改 temperature 后
下一次对话即时生效，无需重启服务；
curl GET /api/sessions/default/events 能分页拉取落盘事件。
````

---

## Phase 18 · Web 界面重做（使用 skill）

````
@pocket-agent-docs/09-Web界面与配置外化.md

重做 ui-web 插件的前端，严格按 09 文档 §1 的信息架构（官方原生
DSH 结构：视图标签环 + 纯净消息流 + 独立轨迹视图）与 §3 的
设置页/插件页规格。工作要求：

1. 首先调用 /ui-ux-pro-max skill 生成界面设计方案与页面代码，
   信息架构以 09 文档 §1 的 ASCII 布局为准，skill 负责
   视觉层级、配色、间距、交互细节的专业化
2. 技术约束：单页应用，允许引入轻量构建（Vite + React），
   构建产物内嵌进 ui-web 插件包；SSE 用 EventSource 消费；
   设置页表单必须由 GET /api/config/schema 驱动渲染，
   禁止硬编码表单字段；轨迹历史走 events API 分页加载
3. 必须实现的结构与元素（09 §1）：
   - 会话页头视图标签环：对话 / 轨迹切换，按会话记住所在视图
   - 对话视图纯净：只有问答、审批弹窗、错误提示、实时状态条；
     回答旁轻量结算行，点击切轨迹定位本轮
   - 轨迹视图：按轮次分组的事件记录表（用户/助手/工具/拦截/压缩，
     标示轮次与步骤边界）、记录检查器（tokens/耗时/输入/输出/参数）、
     顶部时间概览条（P0 静态投影即可）、流式跟随尾部且上滚暂停、
     进行中记录只显示开始标记不虚构耗时、虚拟滚动、
     "加载更早"分页、导出 session log（带脱敏提示）
   - 徽章体系落在轨迹记录上：cache_hit 绿 / 已拦截 黄 / 工具错误 红
   - 实时状态条（09 §1.4 完整规格，仅对话视图）
   - 输入区：上方模式选择器 + 权限选择器，底部上下文用量行
     （点击展开区段明细），发送/停止按钮
   - 审批弹窗 60s 超时拒绝
   - 缓存命中率不谎报（99.5% 不显示 100%）
4. 空态与错误态：llama-server 未连接时显示引导页
   （启动命令 + 重试按钮），不显示裸报错文本
5. 全部页面完成后，调用 /impeccable skill 对界面做一轮
   完整审查，按审查意见修正后再交付
6. 测试：保留并更新 SSE 端点测试；新增 schema 驱动渲染的
   前端逻辑测试（mock schema 验证表单生成）

验收：pnpm build 通过；启动 web profile 后浏览器完成
"多工具对话 → 消息流只见问答与结算行 → 点结算行切到轨迹视图
逐步检查 → 设置页改 max_steps → 再对话即时生效"全流程；
/impeccable 审查无 blocker 级遗留问题。
````

---

## 执行顺序与注意事项

1. Phase 17 先行（后端 API 是界面的数据源），Phase 18 依赖它的 schema、overrides 与 events 分页
2. Phase 18 的提示词里已明确要求 Cursor 先调 `/ui-ux-pro-max`、完成后调 `/impeccable`——如果 Cursor 没主动调，手动输入 `/ui-ux-pro-max` 触发
3. 视图标签环是本次修正的核心：步骤信息一律不进消息流，轨迹视图通过 events API + SSE 投影组装，纯投影不改写对话数据
4. 设置页 schema 驱动渲染是底线要求：它保证以后每加一个配置项，设置页自动出现对应控件，这是"配置外化"不烂尾的关键
5. 两条诚实性红线：命中率不谎报 100%；没有 token 时间戳就显示"不可用"，不虚构 TTFT 与耗时
