# 06 · 测试方案与 Cursor 提示词（Phase 7~9）

> 前置条件：Phase 0~6 已全部完成并验收通过。本文件延续《05-Cursor开发提示词》的执行方式：逐阶段粘贴、跑验收、再进下一阶段。

## 测试分层总览

| 层 | 内容 | 依赖真实模型 | 对应阶段 |
| --- | --- | --- | --- |
| 1 | 场景剧本测试（长结果、多调用链、对抗） | 否（MockLLM） | Phase 7 |
| 2 | 真机集成测试（难度阶梯任务集） | 是（llama-server） | Phase 8 |
| 3 | 评测脚本（N 次成功率统计 + 裸 Agent 基线对比） | 是 | Phase 8 |
| 4 | 专项压测（超长结果、长链条、窗口边界） | 部分是 | Phase 9 |

**断言原则**：真机测试不断言"回答内容写得好不好"（小模型生成质量有随机性），只断言**框架行为指标**——停止原因、真实调用次数、缓存命中、Prompt 预算。

---

## Phase 7 · 场景剧本测试（MockLLM，CI 可跑）

````
@docs/03-模块实现规格.md

新增 tests/test_scenarios.py，补三个场景剧本（全部用 conftest.py 的 MockLLM，不起真实服务）：

1. 长结果压缩剧本 test_long_result_compressed：

- 注册一个 mock 工具 report_tool，返回 "数据报告。" * 1000（5000 字）
- MockLLM 队列：choose→0，fill→{"path": "report.txt"}，choose→-1，final→"回答"
- 断言：steps[0].result 长度 ≤ AgentConfig.per_result_limit；
result 含 "[...截断" 标记；stats["prompt_tokens_max"] ≤ 3500 换算字符预算；
stop_reason == MODEL_DONE

2. 多调用链剧本 test_six_step_chain：

- MockLLM 排 6 次 choose+fill + 1 次 choose→-1 + final
- 断言：stats["real_calls"] == 6；hit_count == 0；
steps 中无重复指纹（用 CallLedger.fingerprint 校验）；stop_reason == MODEL_DONE

3. 越界选择兜底剧本 test_out_of_range_choice：

- MockLLM 第一次 choose 抛出 LLMFormatError（模拟约束解码失效且正则兜底失败）
- 断言：AgentRunner 捕获后记入历史并继续循环或优雅终止，
不向上抛 LLMFormatError；若规格 §8 未覆盖此路径，
在 runner.py 补上：LLMFormatError 按"本步作废"处理，计 1 次拦截计入护栏

运行 pytest tests/test_scenarios.py -v 全部通过后，
再跑全量 pytest --cov=pocket_agent --cov-report=term-missing 确认覆盖率不掉。

验收：上述命令全绿。
````

---

## Phase 8 · 真机集成测试 + 评测脚本

````
@docs/02-技术架构设计.md @docs/03-模块实现规格.md

新增真机集成测试与评测脚本：

1. tests/fixtures/report.txt：生成一个 5000 字的测试文件
（重复的段落文本，但第 100 行附近埋入关键句
"核心结论：本季度增长率 37.2%"，供长结果用例验证关键数据保留）
2. tests/test_integration.py，全部用例标记 @pytest.mark.integration，
通过 fixture 创建连接 http://localhost:8080/v1 的真实 AgentRunner，
服务不可达时 pytest.skip。任务集按难度阶梯：

| 用例 | 任务 | 断言 |
|---|---|---|
| test_one_step | "现在几点了" | real_calls == 1，MODEL_DONE |
| test_two_city | "查询北京和上海的天气并告诉我温差" | 无重复真实调用；answer 含 "温" 或 "°C" |
| test_three_city | "查北京、上海、广州天气，算最高最低温差" | real_calls ≤ total_steps；MODEL_DONE |
| test_long_file | "读取 tests/fixtures/report.txt 并总结要点" | 每步 result ≤ per_result_limit；answer 含 "37.2"（验证压缩保住关键数据，若失败则记录为已知问题不阻塞） |
| test_many_tools | 注册 15 个工具后问 "北京天气怎么样" | 选中 get_weather，recall 生效 |

所有用例通用断言：stop_reason 合法、prompt_tokens_max ≤ 预算、
real_calls == 去重后指纹数（零重复真实调用）。
conftest 加 pytest_addoption 支持 --integration 开关，默认跳过。

3. scripts/eval.py：评测脚本

- 任务集 TASKS 含上述 5 个任务，每个跑 10 次（--runs 可调）
- 汇总输出表格：success_rate / dup_call_rate（必须为 0）/
avg_intercepts / avg_steps / avg_prompt_tokens
- --baseline 模式：内置一个 30 行的裸自由 Agent Loop
（直接多轮对话让模型自主调工具，无任何框架机制），
跑同一任务集，输出两组指标对比表
- 结果落盘 eval_results.json

验收：llama-server 运行中执行
pytest tests/test_integration.py -v --integration 全部通过；
python scripts/eval.py --runs 3 能输出指标表格。
````

---

## Phase 9 · 专项压测

````
@docs/02-技术架构设计.md @docs/03-模块实现规格.md

新增 scripts/stress_test.py 与对应单元测试：

1. 长结果压测 stress_long_result：
mock 工具分别返回 1 万、5 万字，用 MockLLM 跑，
断言 prompt_tokens_max ≤ 预算，记录压缩耗时，
打印耗时表验证压缩器性能线性（5 万字压缩 < 2 秒）
2. 长链条压测 stress_long_chain：
MockLLM 排 10 步不同工具调用（max_steps 临时调 15），
断言 summarize_state 截断后循环仍正常终止、状态摘要 ≤ 500 字符
3. 窗口边界验证（真机，手动执行，脚本输出指引）：
提示用户用 -c 4096 重启 llama-server 后跑
pytest tests/test_integration.py --integration，
脚本最后打印："对比 -c 8192 的 eval 结果，
success_rate 下降即反向验证上下文压缩机制的价值"
4. tests/test_stress.py：前两项的 pytest 化（MockLLM 版本），
标记 @pytest.mark.stress，默认跳过，--stress 开启

验收：pytest --stress 全绿；scripts/stress_test.py 输出完整耗时表。
````

---

## 执行顺序与注意事项

1. Phase 7 纯本地，随时可做；**Phase 8、9 前必须先启动 llama-server**（`-c 8192 --flash-attn --cache-reuse 256`）
2. 每个 Phase 新开 Cursor 对话，@ 引用对应文档
3. Phase 8 的 `test_long_file` 如果断言"37.2"失败：这是规则截断丢中间内容的已知边界，让 Cursor 把该断言降级为 warning 并记录 issue——后续可开 `use_llm_summarize=True` 对比两档压缩策略的关键数据保留率
4. eval 的基线对比数据建议截图存档——"裸 Agent 重复率 vs pocket-agent 重复率"就是这套方案最直观的价值证明