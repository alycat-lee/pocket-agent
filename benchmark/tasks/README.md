# 基准任务集

每题一张任务卡：`{id, dimension, input, rubric, expected_constraints?}`。

`rubric.pass_criteria` 给人与 judge 看；`score_fields` 决定自动指标或云端打分项。

| 文件 | 维度 | 客观主指标 |
|---|---|---|
| d1_format.json | 指令遵循 | 格式解析率 |
| d2_memory.json | 记忆 | needle 命中（含 10 轮取回） |
| d3_tools.json | 工具 | 是否调用 / 重复调用率 |
| d4_planning.json | 规划 | 类型与约束提取（含事故原句） |
| d5_verify.json | 校验 | 思维链 / 模板泄漏 |
| d6_longform.json | 长文 | 字数比 / 数字一致性 |
| d7_knowledge.json | 知识 | 时效题（多依赖 judge） |
| d8_reasoning.json | 推理 | 答案命中 + judge |
| d9_multiturn.json | 多轮 | 指代跟随 |

纪律：任务只增不改。事故原句「给我生成一篇2万字的小说 内容随意」在 d4-accident-novel。
