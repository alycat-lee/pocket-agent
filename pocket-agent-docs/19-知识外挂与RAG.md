# 19 · 知识外挂与 RAG（Phase 33）

> 立项依据（文档 15 能力矩阵）：D7 知识获取维度。核心认知：知识不是智能，是数据——大模型靠背，小模型靠查，外挂知识源反而更可靠（可溯源、可更新、可审计）。对小模型的附加价值：检索约束 + 引用标注能显著压制 8B 的幻觉倾向。

## 1. 架构：三个挂载点

```
文档/网页/Wiki ──入库──► 知识库（sqlite 向量存储）
                            │ 检索 top-k
                            ▼
① 工具层：search_knowledge / wiki_lookup / web_search（标准工具注册）
② 上下文层：检索结果源头压缩 → 预算拼装；事实性内容进一致性卡
③ 验收层：Verifier 引用校验——事实性断言必须可追溯到检索结果
```

## 2. 知识库插件（knowledge-base）

1. **嵌入模型**：本地小 embedding 模型（bge-small-zh / m3e-small 级别，
   ~100MB，CPU 可跑）——不进 GPU，不占 8G 显存预算；
   也可配置云端 embedding API（OpenAI 兼容）
2. **向量存储**：sqlite + 余弦相似度起步（numpy 计算），
   不引入 faiss/chroma 等重型依赖；文档量 < 10 万 chunk 性能足够
3. **入库流水线**：文档上传（txt/md/pdf 先解析为文本）→
   切分（500 字块、50 字重叠）→ embedding → 落盘
4. **检索**：query embedding → top-k（默认 5）→ 源头压缩
   （每块 ≤ per_result_limit）→ 返回紧凑文本
5. **API**：POST /api/knowledge/documents（上传）、
   GET /api/knowledge/documents、DELETE /api/knowledge/documents/{id}、
   POST /api/knowledge/search（调试直查）
6. **界面**：设置页新增"知识库"分组——文档列表、上传、删除、
   命中测试（输入 query 看召回结果）

## 3. 工具层

| 工具 | 说明 | cacheable |
|---|---|---|
| `search_knowledge` | 检索私有知识库，返回 top-k 压缩文本（含来源标记） | 是 |
| `wiki_lookup` | 维基百科 API：搜索条目 + 返回摘要（源头压缩） | 是 |
| `web_search` | 真实网页搜索（SearXNG 自建或 Serper API），返回前 3 条标题+摘要 | 是 |

工具描述保持 ≤20 字铁律；检索类工具结果必须带**来源标记**（文档名/URL），供引用校验使用。

## 4. 与执行器的集成

1. **qa 模式**：知识型问题（Planner 任务卡新增 `needs_knowledge: bool` 字段）先强制检索再回答——qa 不再等于"裸生成"
2. **writer 模式**：检索到的事实性内容进一致性卡（ConsistencyCard.facts），分段写作全程携带，数字/术语不打架
3. **agent 模式**：走标准工具流程，无需特殊处理
4. **诚实兜底**：检索无结果时，工具返回"未找到相关资料"——回答必须如实说明，禁止编造（写进 FINAL_ANSWER 的 qa 变体提示词）

## 5. Verifier 增加引用校验

任务卡 `needs_knowledge=true` 的回答：

- 回答中的数字/日期/专名应能在当轮检索结果中找到出处（字符串匹配级校验）
- 找不到出处的比例 > 30% → 不通过，打回并附"请仅基于检索资料回答"
- 回答末尾自动附来源列表（代码拼装，不靠模型自觉）

## 6. 与基准联动

D7 维度任务集（benchmark/tasks/d7_knowledge.json）：
- 私有知识题（答案只在入库文档里）
- 时效题（"最新""今年"类）
- 幻觉陷阱题（问一个不存在的东西，期望回答"未找到资料"而非编造）

---

## Phase 33 · 知识外挂与 RAG

````
@docs/03-模块实现规格.md @pocket-agent-docs/19-知识外挂与RAG.md

实现知识库与检索工具（文档 19 全文规格）：

1. knowledge-base 插件：embedding（本地 bge-small-zh 级别，
   CPU 运行，sentence-transformers 或 onnxruntime 轻量实现，
   进 extras 不进核心依赖）+ sqlite 向量存储 + 入库流水线
   （解析/切分 500 字块 50 字重叠/embedding/落盘）+
   top-k 检索 + 源头压缩
2. 三个检索工具按 §3 表格注册（search_knowledge / wiki_lookup /
   web_search），结果必带来源标记
3. 集成：TaskCard 增加 needs_knowledge 字段（Planner 提示词同步
   更新，录入 prompt 插件）；qa 知识型问题强制先检索；
   writer 事实进一致性卡；检索空结果的诚实兜底提示词
4. Verifier 引用校验（§5：出处匹配 >30% 无出处打回，
   回答末尾代码拼装来源列表）
5. API 与界面：知识库 CRUD API + 设置页"知识库"分组
   （文档列表/上传/删除/命中测试）
6. 基准：benchmark/tasks/d7_knowledge.json 收录三类任务
   （私有知识/时效/幻觉陷阱）
7. 测试：
   - 入库→检索回环（固定 embedding stub 验证流水线）
   - 切分边界与重叠正确性
   - 引用校验的正反用例
   - 幻觉陷阱：检索为空时回答必须声明"未找到资料"
   - embedding 缺失依赖时插件优雅降级（禁用并提示，不崩溃）

验收：pytest 全绿；真机上传一份文档后问答命中其内容且附来源；
幻觉陷阱题回答"未找到相关资料"；
python scripts/eval.py --benchmark --dims d7 产出三跑道对比。
````

---

## 注意事项

1. embedding 模型选型的硬约束：CPU 可跑、体积 ≤ 200MB、中文效果好——bge-small-zh 是当前最优解；不要为了效果上大模型挤占显存，主模型的 8G 是红线
2. web_search 接 SearXNG 时注意 SSRF 防护（拦截内网地址），这是 DSH 踩过的坑
3. 幻觉陷阱题是本 Phase 的灵魂用例——"让 8B 学会说不知道"比"让 8B 答对"更有产品价值
4. 与 Phase 16 的 MCP 桥接不冲突：MCP 生态里也有检索类 server，知识库插件是内置一等公民，二者可共存
