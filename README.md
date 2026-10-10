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

核心依赖仅 `httpx`、`pydantic`；开发依赖含 `pytest`、`pytest-cov`、`ruff`。

## 启动 llama-server

上下文窗口必须足够大（`-c 8192`），否则压缩与预算管理失去意义：

```bash
llama-server -m qwen3-4b-instruct-q4_k_m.gguf \
  -c 8192 --flash-attn \
  --cache-reuse 256 \
  --port 8080
```

服务默认监听 `http://localhost:8080`，框架使用 OpenAI 兼容接口：`http://localhost:8080/v1`。

## Demo

```bash
python cli.py demo "查询北京和上海的天气并告诉我温差"
```

指定服务地址并打印详细日志：

```bash
python cli.py demo "查询北京和上海的天气并告诉我温差" \
  --base-url http://localhost:8080/v1 \
  --verbose
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

## 文档

- `docs/01-项目需求文档.md`
- `docs/02-技术架构设计.md`
- `docs/03-模块实现规格.md`
- `pocket-agent-docs/04-运行时提示词模板.md`
- `pocket-agent-docs/05-Cursor开发提示词.md`
