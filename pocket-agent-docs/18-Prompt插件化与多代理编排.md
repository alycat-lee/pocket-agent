# 18 · Prompt 插件化与多代理编排（Phase 31~32）

> 立项依据（文档 15 能力矩阵）：D4 任务理解与规划、D8 推理深度的工程缓解。本阶段把"一切皆插件"推进到两个新领域：Prompt 与 Agent 本身——这是"自动切分任务 + 合并结果"的实现载体。

## 1. 设计原则

**切分-合并主义**：小模型等效大模型的路径 = 把大模型"一口气想完"的事，拆成"小模型一次只想一件小事"的流水线。多代理不是目的，是切分的载体；合并不是拼接，是带验收的组装。

**切分的四个维度**（现有能力各就各位）：
- 任务切分：Planner 任务卡（Phase 27）
- 上下文切分：预算拼装（Phase 3）
- 生成切分：writer 分段（Phase 12）
- 注意力切分：每步只见必要信息（状态机）
- **本期新增：角色切分——orchestrator / worker / merger 各司其职，各自独立的短上下文**

**合并的三个防线**：一致性卡（事实不矛盾）、Verifier（不合格打回）、接缝处理（重叠裁剪）。

## 2. Prompt 即插件（Phase 31）

现状：`prompts.py` 九个常量硬编码，措辞不可换、不可按模型适配、改动无法测量。

规格：

1. `prompt_registry` 服务：每个原子任务的提示词登记为
   `{id, template, variants: {local: ..., cloud: ...}, version}`
2. 提示词插件：第三方可注册/覆盖任意原子任务提示词（disposer 回收）
3. 变体选择：按当前 provider 的 preset 自动选用 local/cloud 变体——
   本地 4B 用超短指令版（≤150 字），云端用完整版
4. 与基准联动：`scripts/eval.py --benchmark` 支持 `--prompt-variant` 参数，
   提示词改动 = 跑一次对应维度基准对比，分数不升不许合入
5. 设置页只读展示当前生效提示词与版本（调试入口）
6. 迁移：prompts.py 全部常量搬进内置 prompt 插件，外部接口不变

## 3. 多代理编排（Phase 32）

```
┌────────────────────────────────────────────┐
│ orchestrator（编排代理）                     │
│  输入：TaskCard.plan                        │
│  职责：子任务依赖图 → 调度 → 汇总            │
└───┬──────────────┬──────────────┬──────────┘
    ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ worker  │  │ worker  │  │ worker  │  ← 每个 = 独立 AgentRunner
│ 子任务1  │  │ 子任务2  │  │ 子任务3  │    独立账本/短上下文/单任务
└────┬────┘  └────┬────┘  └────┬────┘
     └─────────────┼─────────────┘
                   ▼
            ┌─────────────┐
            │ merger 合并  │ → Verifier 验收 → 交付
            └─────────────┘
```

规格：

1. **worker = 独立 AgentRunner 实例**：独立 CallLedger、独立护栏、
   输入只有"本子任务 + 必要的上游摘要"（≤ 预算），绝不传其他
   子任务的全文——子代理间通信只传摘要，这是预算纪律
2. **调度器**：按依赖拓扑序执行；并发度按 provider 区分——
   本地 llama.cpp 单模型串行（并发=1），云端 provider 可并发
   （默认 3）；panic 隔离：单个子任务失败不拖垮全局，
   标记失败后由 orchestrator 决定重试/跳过/终止
3. **merger 合并策略**（按任务类型选）：
   - collect：并列结果拼接（多城市天气类）
   - chain：顺序依赖传递（前者输出是后者输入）
   - synthesize：一次 LLM 调用做归并总结（输入先压缩）
   - vote：同任务多跑几路取一致结果（D8 推理维度的自洽投票）
4. **护栏**：子代理嵌套深度 ≤ 2（worker 不得再生 worker 的 worker）；
   全局步数预算（默认 30，各 worker 的 max_steps 独立计算）；
   账本指纹全局共享——不同 worker 重复调同一工具同参数照样命中缓存
5. **轨迹视图**：子代理渲染为嵌套记录——任务卡 → 子任务节点
   （可展开看该 worker 的完整步骤），对齐 DSH 侧边栏子代理会话的
   可观测性，但收进轨迹页
6. 适用判定：TaskCard.plan ≤ 2 步时不开多代理（杀鸡不用牛刀，
   直接单 runner 执行）；≥ 3 步且步骤间有明确产物传递才启用

## 4. 诚实边界（写进文档给 Cursor 和人看）

多代理放大优点也放大错误：**子任务出错会在合并处放大**。所以 merger 之后必须过 Verifier，关键子任务可用 vote 策略冗余。编排层的目标是"任务完成率追平成熟 Agent"，单点内容质量上限仍是模型内力——创作类 worker 可单独路由云端。

---

## Phase 31 · Prompt 即插件

````
@docs/03-模块实现规格.md @pocket-agent-docs/18-Prompt插件化与多代理编排.md

实现 Prompt 插件化（文档 18 §2）：

1. prompt_registry 服务与提示词插件机制
   （{id, template, variants, version} + 注册 disposer）
2. prompts.py 全部常量迁移为内置 prompt 插件，外部接口不变；
   每个原子任务补齐 local/cloud 双变体（local 版 ≤150 字超短指令）
3. 变体自动选择：按当前 provider preset（llamacpp→local，
   deepseek→cloud）
4. eval.py --benchmark 增加 --prompt-variant 参数支持 A/B 对比
5. 设置页只读展示生效提示词与版本
6. 测试：注册/覆盖/卸载回收、变体选择正确性、
   迁移后全部既有测试零回归

验收：pytest 全绿；注册一个自定义 choose 提示词插件后
轨迹可见措辞变化，卸载后恢复默认。
````

---

## Phase 32 · 多代理编排（切分-合并引擎）

````
@docs/03-模块实现规格.md @pocket-agent-docs/18-Prompt插件化与多代理编排.md

实现多代理编排（文档 18 §3 全文规格）：

1. orchestrator 插件（requires: agent, llm）：
   TaskCard.plan → 子任务依赖图 → 拓扑调度 → 汇总
2. worker：独立 AgentRunner 实例（独立账本与护栏），
   输入 = 子任务 + 上游摘要（≤ 预算），禁传其他子任务全文
3. 调度器：本地 provider 串行、云端并发 ≤3；
   子任务失败隔离（重试 1 次 / 跳过 / 终止三档由 orchestrator 决策）
4. merger 四种策略（collect / chain / synthesize / vote）
   按任务类型选择，synthesize 输入先压缩
5. 护栏：嵌套深度 ≤2；全局步数预算 30；
   全局共享账本指纹（跨 worker 去重）
6. 适用判定：plan ≤2 步走单 runner，≥3 步才开编排
7. 轨迹视图：子代理嵌套记录（任务卡 → 子任务节点可展开
   看 worker 完整步骤）
8. 测试（MockLLM 剧本）：
   - 依赖拓扑调度顺序正确
   - 跨 worker 账本去重命中
   - 子任务失败三档处理路径
   - 深度上限拦截 worker 再生 worker
   - vote 策略多数一致
   - ≤2 步任务不开编排的快速路径

验收：pytest 全绿；真机跑"查北京上海广州天气，写一段
300 字对比分析"——轨迹可见 orchestrator 拆出 3 个天气 worker
+ 1 个合并节点，全程零重复真实调用。
````

---

## 注意事项

1. Phase 31 先行：多代理的每个角色（orchestrator/merger）都是原子提示词任务，提示词插件化后它们才能按模型调优
2. Phase 32 依赖 Phase 27 的 TaskCard（plan 字段是编排输入）——若 27 未完成先补
3. 本地单模型下"多代理"是逻辑并发、物理串行——价值在上下文隔离与角色聚焦，不在速度；要速度需云端 provider
4. 编排不是银弹：plan 质量决定一切，Planner 的基准成绩（D4 维度）是 Phase 32 效果的前置指标
