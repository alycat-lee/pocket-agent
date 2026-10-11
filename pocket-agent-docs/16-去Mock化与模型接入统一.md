# 16 · 去 Mock 化与模型接入统一（Phase 29）

> 立项原则：本地小模型 Agent 的运行时，本地模型是必然存在的依赖——连不上就明确报错引导，**运行时禁止出现任何 mock 数据与 mock 模型**；模型接入统一为 OpenAI 兼容协议一种实现，本地能跑通，外部大模型就能跑通，不设特例。

## 1. 去 Mock 化（运行时红线）

**现状问题**：`demo_tools.py` 的假天气、假时间等 mock 工具挂在默认 profile 里（小说事故里模型对着"上海天气 18℃"打转，素材就来自这些假数据）；测试基建的 MockLLM 与运行时边界需明确隔离。

**规格**：

1. `demo_tools.py` 整体移入 `tests/fixtures/`，重命名为 `fixture_tools.py`——它是测试道具，不是产品功能
2. `pocket.yml` 的默认 profile 移除全部 mock 工具；`tools-core` 插件只保留真实可用的工具（calculator、read_file、get_current_time 这类真实执行的保留；返回编造数据的天气工具删除——真实天气要么接真实 API，要么不存在）
3. **禁止静默降级**：LLM 连接失败时，运行时必须抛出明确错误并进入"模型未连接"引导态（Phase 19 的 health 引导页已有），严禁回退到任何内置假数据、假回答、缓存冒充
4. MockLLM 只允许存在于 `tests/` 目录——新增 CI 检查：`grep -r "MockLLM" src/` 必须为空，防止测试桩泄漏进产品代码
5. CLI demo 命令调整：demo 不再预置假工具，改为引导用户连接真实模型跑真实任务

## 2. 模型接入统一（一种协议，零特例）

**规格**：

1. providers 的 `type` 字段收敛为**唯一实现** `openai_compatible`；`llamacpp` / `deepseek` / `openai` 降级为**预设模板**（仅是默认值的组合）：

```python
PROVIDER_PRESETS = {
    "llamacpp": {          # 本地 llama.cpp server
        "base_url": "http://localhost:8000/v1",
        "requires_key": False,
        "trust_env": False,           # 本地直连，防代理劫持（502 教训）
        "thinking": "off",            # chat_template_kwargs enable_thinking=false
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "requires_key": True,
        "trust_env": True,
        "thinking": "strip",          # 剥离 reasoning_content
    },
    "openai_compatible": {            # 通用：任何兼容端点
        "requires_key": True,
        "trust_env": True,
        "thinking": "strip",
    },
}
```

2. 客户端统一为一个 `OpenAICompatibleClient`（基于现有 LlamaCppClient 改造）：`trust_env`、`thinking` 策略、api_key 全部由 provider 配置驱动；现有 `llm_deepseek.py` 独立实现删除，合并进统一客户端
3. `trust_env` 自动判定兜底：base_url 的 host 为 localhost / 127.0.0.1 / 内网网段时强制 `trust_env=False`（即使用户选了云端预设填了本地地址也安全）
4. thinking 三档：`off`（请求关闭思考）/ `strip`（响应剥离思考块）/ `keep`（保留并经 `step.thinking` 事件进轨迹）——对齐 Phase 26-① 的规格
5. 设置页添加 provider 时，类型下拉变为预设模板选择（选模板自动填默认值，全部可改）

## 3. 验收逻辑

统一后的检验标准很简单：**同一个 providers 配置结构，添加一个本地 llama.cpp 和一个 DeepSeek，两张卡片"测试"都通、路由切换正常、对话都正常**——证明协议层无特例。

---

## Phase 29 · 去 Mock 化与模型接入统一

````
@docs/03-模块实现规格.md @pocket-agent-docs/16-去Mock化与模型接入统一.md

实施去 Mock 化与模型接入统一（文档 16 全文规格）：

1. 去 Mock：demo_tools.py 移入 tests/fixtures/ 并改名 fixture_tools.py；
   pocket.yml 默认 profile 移除全部编造数据的 mock 工具，
   真实执行的工具（calculator/read_file/get_current_time）保留在
   tools-core；LLM 连接失败一律进入"模型未连接"引导态，
   全面排查并删除任何静默降级到假数据的路径；
   CLI demo 改为真实模型引导流程
2. CI 红线：新增脚本/测试断言 grep -r "MockLLM" src/ 为空，
   grep -r "fixture_tools\|demo_tools" src/ 为空
3. 模型接入统一：新建统一 OpenAICompatibleClient
   （基于 LlamaCppClient 改造，trust_env/thinking/api_key 全部由
   provider 配置驱动）；PROVIDER_PRESETS 三个预设模板按文档 §2 定义；
   删除 llm_deepseek.py 独立实现，deepseek 迁移为预设；
   trust_env 内网/localhost 强制 False 的兜底判定；
   thinking 三档（off/strip/keep）按 Phase 26-① 规格实现
4. 设置页：添加 provider 改为"选预设模板 → 自动填默认值 → 可改"流程
5. 测试：
   - 统一客户端的 trust_env 自动判定（localhost/内网/公网三种地址）
   - thinking 三档行为（mock transport 验证请求体与响应处理）
   - 预设模板迁移：旧 providers 配置自动升级不丢数据
   - CI 红线检查脚本本身
6. 真机验收：providers 同时配置本地 llama.cpp 与 DeepSeek，
   两张卡片测试都通；路由分别指向两者各跑一轮对话；
   断开 llama-server 后界面进入"模型未连接"引导态而非假数据

验收：pytest 全绿 + 上述真机验收全过；
grep 红线检查通过；pnpm build 后界面实测。
````

---

## 注意事项

1. 执行顺序：本 Phase 与 Phase 26 的思考模型适配（①）有交集——thinking 三档是 26-① 的完整版，若 26 已先做完，本 Phase 把它收敛进 provider 预设即可，别重复实现两套
2. 测试不受影响：MockLLM 仍然是单元测试的合法基建，红线只管 src/ 运行时
3. 删 mock 天气后，工具调用类冒烟用例改走 calculator / read_file / get_current_time 这类真实工具（"算 25 减 18"、"读 README 前两段"）
