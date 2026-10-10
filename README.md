# pocket-agent

面向 8G 显存本地小模型的工具调用 Agent 框架。

模型只做两种原子任务：**单步选择（选择题）** 与 **参数填充（填空题）**；流程、记忆、去重、格式与护栏由代码保证。

## 环境要求

- Python ≥ 3.10
- [llama.cpp](https://github.com/ggerganov/llama.cpp) 的 `llama-server`
- 推荐模型：Qwen3-4B-Instruct（Q4_K_M）

## 安装

```bash
pip install -e ".[dev]"
```

核心依赖仅 `httpx`、`pydantic`、`pyyaml`。Web UI：

```bash
pip install -e ".[web]"
# 或开发环境已含 fastapi/uvicorn：
python cli.py serve --profile web
```

浏览器打开 `http://127.0.0.1:3000`（需 llama-server）。`profile: headless` 时不加载 `ui-web`、不监听端口。

### MCP 桥接（可选）

在 `pocket.yml` 启用 `bridge-mcp`，把外部 MCP server 的工具注册为 `mcp_*`：

```yaml
- name: bridge-mcp
  config:
    mcp_servers:
      - name: echo
        transport: stdio
        command: python
        args: ["scripts/mock_mcp_echo.py"]
      # 真实 filesystem MCP 示例（需本机已安装对应 server）：
      # - name: fs
      #   transport: stdio
      #   command: npx
      #   args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
```

然后 `python cli.py demo "用 mcp_echo 回显 hello"`（需 llama-server）。

## 启动 llama-server

上下文窗口必须足够大（`-c 8192`），否则压缩与预算管理失去意义：

```bash
llama-server -m qwen3-4b-instruct-q4_k_m.gguf \
  -c 8192 --flash-attn \
  --cache-reuse 256 \
  --port 8000
```

服务默认监听 `http://localhost:8000`，框架使用 OpenAI 兼容接口：`http://localhost:8000/v1`。

## Demo

```bash
python cli.py demo "查询北京和上海的天气并告诉我温差"
```

指定服务地址并打印详细日志：

```bash
python cli.py demo "查询北京和上海的天气并告诉我温差" \
  --base-url http://localhost:8000/v1 \
  --verbose
```

断点续跑（中途 Ctrl+C 后用同一 `run_id` 恢复，已完成步骤走账本缓存）：

```bash
python cli.py demo "查询北京和上海的天气并告诉我温差" --resume demo-1
# 中断后：
python cli.py demo "查询北京和上海的天气并告诉我温差" --resume demo-1
```

预期：

- 至少 3 步（查北京、查上海、计算温差，或等价路径）
- 无重复真实工具调用（重复请求由账本拦截，`cache=hit`）
- 正常输出温差相关最终回答

每步日志格式：

```text
[Step 1] choose=get_weather args={"city": "北京"} cache=miss tokens≈820
```

## 内置演示工具

| 工具 | 说明 |
|---|---|
| `get_weather` | mock 城市天气（北京/上海/广州/深圳/成都等） |
| `calculator` | `ast` 安全四则运算（禁止 `eval`） |
| `read_file` | 读取本地文件前 2000 字 |
| `get_current_time` | 当前时间字符串 |

## 源码布局

`src/` 按功能模块分目录（`config/`、`llm_client/`、`tools/`、`runner/` …），经 `pyproject.toml` 的 `package-dir` 映射为导入路径 `pocket_agent.*`。

## 库用法

```python
from pocket_agent import AgentConfig, AgentRunner, tool
from pocket_agent.demo_tools import register_demo_tools
from pocket_agent.llm_client import LlamaCppClient
from pocket_agent.tools import ToolRegistry

config = AgentConfig()
registry = ToolRegistry(config)
register_demo_tools(registry)
runner = AgentRunner(config, LlamaCppClient(config), registry)
result = runner.run("查询北京天气")
print(result.answer)
```


## 测试

```bash
pytest --cov=pocket_agent --cov-report=term-missing
ruff check src tests
```

Windows 上若默认 `py -3` 指向 free-threading 构建，请使用：

```bash
py -3.13 -m pytest --cov=pocket_agent --cov-report=term-missing
py -3.13 -m ruff check src tests
```

跑完后清理缓存与测试临时文件：

```bash
py -3.13 scripts/clean.py
```

pytest 已默认 `--cache-clear`；Agent 跑完测试/检查后也会自动执行上述清理。

### 长文本生成

```bash
py -3.13 scripts/write.py "新能源汽车行业分析" --chars 3000 \
  --material tests/fixtures/report.txt --output article.md
py -3.13 scripts/eval.py --writer
```

### 真机集成与评测

需先启动 llama-server（`-c 8192 --flash-attn --cache-reuse 256 --port 8000`）：

```bash
py -3.13 -m pytest tests/test_integration.py -v --integration
py -3.13 scripts/eval.py --runs 3
py -3.13 scripts/eval.py --runs 3 --baseline
py -3.13 scripts/eval.py --writer
```

集成测试默认跳过，必须显式加 `--integration`。

### 专项压测

```bash
py -3.13 -m pytest --stress -v
py -3.13 scripts/stress_test.py
```

`scripts/stress_test.py` 会打印长结果压缩耗时表，并给出 `-c 4096` 窗口边界真机验证指引。

## 文档

- `docs/01-项目需求文档.md`
- `docs/02-技术架构设计.md`
- `docs/03-模块实现规格.md`
- `pocket-agent-docs/04-运行时提示词模板.md`
- `pocket-agent-docs/05-Cursor开发提示词.md`
