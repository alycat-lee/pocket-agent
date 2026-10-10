# 05 · Cursor 开发提示词（分阶段执行）

## 使用方法

1. 在 Cursor 打开空项目目录 `pocket-agent`，先把 `01`、`02`、`03` 三份文档放到 `docs/` 目录
2. **严格按 Phase 0 → 6 顺序执行**，每阶段把对应提示词整段粘贴给 Cursor Agent（Composer）
3. 每阶段末尾有验收命令，跑通后再进入下一阶段；失败就把报错贴回让 Cursor 修
4. 提示词中 `@docs/xx` 表示在 Cursor 对话中 @ 引用该文档

## 全局规则（建议先写入 `.cursor/rules/project.mdc` 或直接在第一条对话中给出）

````
本项目是 pocket-agent：面向 8G 显存本地小模型的工具调用 Agent 框架。

铁律：
- Python ≥ 3.10，全部代码带类型注解，ruff 通过
- 核心依赖只允许 httpx、pydantic；测试用 pytest + pytest-cov；不引入其他依赖
- 所有 LLM 交互必须经过 LLMClientProtocol 抽象，禁止在业务模块直接发 HTTP
- 接口签名必须与 docs/03-模块实现规格.md 完全一致
- 提示词模板必须与 docs/04-运行时提示词模板.md 逐字一致，不得改写
- 目录结构遵循 docs/02-技术架构设计.md §5
- 每写一个模块同步写对应单元测试
````

---

## Phase 0 · 项目骨架

````
@docs/02-技术架构设计.md @docs/03-模块实现规格.md

初始化 pocket-agent 项目骨架：

1. 按架构文档 §5 的目录结构创建全部目录和空文件（含 __init__.py）
2. 编写 pyproject.toml：项目名 pocket-agent，Python>=3.10，
   依赖 httpx、pydantic；dev 依赖 pytest、pytest-cov、ruff；
   src 布局（[tool.setuptools] package-dir）
3. 实现 src/config/：严格按规格文档 §1 的 AgentConfig
4. 实现 src/llm_client/ 中的三个异常类和
   LLMClientProtocol（Protocol 定义，不含实现）
5. 配置 pytest（pyproject 内 [tool.pytest.ini_options]，testpaths=tests）
6. pyproject 将 src/ 映射为包名 pocket_agent（package-dir），src 下按模块分目录、不再套 pocket_agent/

验收：pip install -e ".[dev]" 成功，pytest 空跑通过，ruff check src 无错误。
````

---

## Phase 1 · 工具注册模块

````
@docs/03-模块实现规格.md

实现 src/tools/ 和 tests/test_tools.py：

1. Tool dataclass、ToolRegistry、@tool 装饰器，接口严格按规格 §3
2. schema 自动生成：从函数签名的类型注解映射 JSON Schema 类型
   （str/int/float/bool），参数默认全部 required；
   docstring 的 Args 段描述解析为参数 description（可选实现）
3. recall() 关键词召回：任务文本与 name+short_desc 做字符级重叠打分，
   按分数降序返回 top_n；工具总数 ≤ AgentConfig.tool_recall_threshold 时返回全部
4. 测试覆盖：schema 生成正确性（四种类型）、装饰器注册、
   recall 排序正确性、召回阈值行为

验收：pytest tests/test_tools.py -v 全部通过。
````

---

## Phase 2 · 调用账本与循环护栏

````
@docs/03-模块实现规格.md

实现 ledger.py、guard.py 及对应测试：

1. CallLedger（规格 §4）：指纹 = md5(tool_name + canonical_json)，
   canonical_json 要求 sort_keys=True、ensure_ascii=False、紧凑分隔符；
   check/save 语义与 real_call_count/hit_count 计数器
2. SemanticDeduper 占位实现：归一化（去空白、 lowercase）后的
   字符串相似度，接口预留真实 embedding 替换
3. LoopGuard（规格 §6）：三种终止条件——
   max_steps、连续拦截 ≥ max_consecutive_intercepts（非拦截步清零）、
   相邻两步原始输出完全相同
4. 测试重点：
   - 参数键序不同但语义相同的 dict 必须生成相同指纹
   - 连续拦截计数在被拦截 2 次后触发终止，中间插入正常步后计数清零
   - 三种 GuardVerdict 各自的 reason 值正确

验收：pytest tests/test_ledger.py tests/test_guard.py -v 全部通过。
````

---

## Phase 3 · 上下文压缩器

````
@docs/02-技术架构设计.md @docs/03-模块实现规格.md

实现 compressor.py 和 tests/test_compressor.py：

1. estimate_tokens：len(text)/1.5 向上取整
2. compress_result 规则截断：头部 70% + "\n[...截断 N 字...]\n" + 尾部 20%，
   总长 ≤ per_result_limit；use_llm_summarize=True 时走 LLM 摘要路径
   （输入先截断到 summarize_input_limit，输出超限再规则截断），
   llm 为 None 时强制回退规则截断
3. build_prompt 预算拼装：system + task + state + step_info 为固定区段，
   history 从最新条目往回塞入直到 prompt_char_budget 耗尽，
   返回完整 prompt 文本
4. summarize_state：生成"已完成 N 步：<最近 keep_last 步>"格式摘要，
   超 500 字符截断中间
5. 测试重点：截断后长度与格式、history 逆向塞入顺序、
   超预算时最旧历史被丢弃、固定区段永远保留

验收：pytest tests/test_compressor.py -v 全部通过。
````

---

## Phase 4 · llama.cpp server 客户端

````
@docs/03-模块实现规格.md

实现 llm_client.py 的 LlamaCppClient（规格 §2）：

1. 基于 httpx，POST {base_url}/chat/completions
2. 三个方法：generate / generate_int / generate_json，
   采样参数从 AgentConfig 读取（temperature/repeat_penalty/top_p/max_tokens）
3. 消息固定为单轮 system+user 两条，绝不构造多轮历史
4. generate_int：response_format 用 json_schema 约束
   {"type":"object","properties":{"value":{"type":"integer",
   "minimum":minimum,"maximum":maximum}},"required":["value"]}，
   解析 value 字段；服务返回无法解析时退化正则提取 + 范围校验，
   仍失败抛 LLMFormatError
5. generate_json：response_format 携带传入 schema，解析失败抛 LLMFormatError
6. 连接错误/超时/5xx 重试 2 次（指数退避 1s、2s），最终抛 LLMConnectionError
7. tests/conftest.py 实现 MockLLM：实现 LLMClientProtocol，
   构造时接收预设响应队列（支持 str/int/dict 混合），
   每次调用弹出并记录收到的 prompt，队列耗尽抛 AssertionError
8. 用 respx 或直接 mock httpx 写客户端单元测试：
   正常响应解析、重试逻辑、LLMFormatError 路径

验收：pytest tests/test_llm_client.py -v 全部通过。
````

---

## Phase 5 · 状态机执行器（核心）

````
@docs/02-技术架构设计.md @docs/03-模块实现规格.md @docs/04-运行时提示词模板.md

实现 prompts.py、runner.py 和 tests/test_runner.py：

1. prompts.py：五个模板常量，与文档 04 逐字一致
2. AgentRunner.run 主循环，严格按规格 §8 的 7 步流程：
   - recall 候选 → build_prompt(CHOOSE_TOOL) → generate_int(-1, len-1)
   - -1 跳出；否则 build_prompt(FILL_ARGS) → generate_json(tool.schema)
   - ledger.check 命中则记 from_cache 记录不执行；
     未命中则 try 执行工具，异常捕获为 "[工具错误] {e}" 记入历史继续循环
   - compress_result → save → append → guard.on_step
   - 所有终止路径都走到 FINAL_ANSWER 生成回答，返回 RunResult
   - 仅 LLMConnectionError 允许上抛
3. 每步打印结构化日志（logging，DEBUG 级）：
   步骤号/工具/参数/缓存命中/prompt 估算 token/耗时
4. RunResult.stats 汇总：total_steps/real_calls/cache_hits/
   intercepts/prompt_tokens_max
5. 用 MockLLM 写四个端到端剧本（规格 §10）：
   正常剧本、重复剧本（验证拦截与连续拦截终止）、
   失控剧本（max_steps 优雅终止有回答）、异常剧本（工具抛错循环继续）

验收：pytest tests/test_runner.py -v 全部通过，
pytest --cov=pocket_agent 核心模块覆盖率 ≥80%。
````

---

## Phase 6 · CLI 演示与集成验证

````
@docs/02-技术架构设计.md @docs/03-模块实现规格.md

实现 demo_tools.py、cli.py 和 README.md：

1. demo_tools.py 四个 mock 工具（规格 §9）：
   get_weather（内置城市字典，含北京晴25℃、上海多云18℃等至少5个城市）、
   calculator（用 ast 模块安全求值，禁止 eval）、
   read_file（前 2000 字）、get_current_time
2. cli.py：argparse 实现
   python cli.py demo "任务文本" [--base-url ...] [--verbose]
   每步打印 [Step N] choose=... args=... cache=hit/miss tokens≈N，
   结尾打印最终回答和汇总统计
3. README.md：安装步骤、llama-server 启动命令
   （-c 8192 --flash-attn --cache-reuse 256）、demo 使用示例
4. 最后跑全量：pytest --cov=pocket_agent --cov-report=term-missing，
   ruff check src tests，修复所有问题

验收命令：
pytest 全绿；ruff 无错误；
启动 llama-server 后运行
python cli.py demo "查询北京和上海的天气并告诉我温差" --verbose
预期 ≥3 步、无重复真实调用、正常输出温差回答。
````

---

## 常见问题处理

| 问题 | 处理 |
|---|---|
| Cursor 某阶段产出与规格不符 | 回复"接口必须与 @docs/03-模块实现规格.md §X 一致，请修正"，不要接受偏离 |
| 集成阶段模型乱选工具 | 先检查 llama-server 是否以 `-c 8192` 启动；再检查日志中 prompt 是否超预算 |
| response_format 报错 | 旧版本 llama.cpp 不支持 json_schema，升级 llama.cpp；客户端已带正则兜底 |
| demo 仍出现重复选择 | 属于预期内（模型层），确认账本日志显示 `cache=hit` 拦截——框架层拦截即设计成功 |
