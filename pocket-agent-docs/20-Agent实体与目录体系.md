# 20 · Agent 实体与目录体系（Phase 34）

> 两个问题一个解法：会话丢失的根因是 `pocket.yml` 从未给 session 插件配置 `sessions_dir`（`config.get()` 拿到 None → 静默跳过落盘）；而 Agent 目录体系落地后，会话天然应该存进 agent 自己的目录——两个问题在同一处收敛。

## 1. 目录结构（Agent 一等公民）

```
项目目录/
└── agents/                     # agent 根目录（默认，设置页可改）
    ├── shared/                 # 首次启动自动创建：所有 agent 的共享区
    │   ├── memory/             # 共享记忆（跨 agent 事实、用户偏好）
    │   ├── sessions/           # 共享会话（跨 agent 协作的对话）
    │   └── files/              # 共享文件（所有 agent 可读）
    ├── default/                # 内置默认 agent（存量会话迁移到这里）
    └── <agent名称>/            # 每个 agent 一个目录，以名称命名
        ├── agent.yml           # 定义：名称/描述/绑定 provider/工具子集/默认权限
        ├── prompt.md           # 人格与系统提示词（注入 system 区段）
        ├── memory/
        │   └── entities.json   # 长期记忆：实体槽位（needle 的正解）
        ├── skills/             # 技能包目录（每个技能一个含 SKILL.md 的子目录）
        ├── sessions/           # 该 agent 的会话记录（.json）
        └── workspace/          # 该 agent 的默认工作目录（文件工具根）
```

**铁律**：目录即真相源——agent 的一切状态都在自己目录里，可查看、可备份、可 git 管理、可手动编辑。删除 agent 目录 = 彻底删除该 agent。

## 2. 会话持久化修复（并入本 Phase）

1. **修复接线**：session 插件的 `sessions_dir` 不再读裸配置，改由 agents 服务按当前 agent 提供 `agents/<name>/sessions/`；shared 会话存 `agents/shared/sessions/`
2. **消灭静默跳过**：`sessions_dir=None` 时启动日志打 WARNING "会话未持久化"；web 模式强制必须有持久化目录
3. **存量迁移**：旧 `.sessions/` 目录的 json 文件自动迁移到 `agents/default/sessions/`，迁移后旧目录重命名为 `.sessions.migrated` 保留备查

## 3. agents 服务插件（provides=["agents"]）

```python
@dataclass
class AgentProfile:
    name: str                   # 目录名（合法文件名校验）
    description: str
    provider_id: str | None     # 绑定模型（None = 跟随默认）
    tools: list[str] | None     # 工具子集（None = 全部）
    permission: str = "workspace"

class AgentService:
    def create(self, name: str, ...) -> AgentProfile:
        """创建 agents/<name>/ 完整脚手架（agent.yml/prompt.md/
        memory/skills/sessions/workspace 六个成员）"""
    def list(self) -> list[AgentProfile]: ...
    def get(self, name: str) -> AgentProfile: ...
    def delete(self, name: str, keep_files: bool = True) -> None: ...
    def root(self) -> Path: ...            # agents 根目录（设置可改）
    def shared_dir(self) -> Path: ...      # agents/shared/
```

**系统提示词装配**：`prompt.md` 内容注入 Planner / final_answer 的 system 区段——agent 人格从此是文件，不是代码。

**记忆**：`memory/entities.json` 实体槽位（键值对：工号、偏好、姓名……），会话滚动摘要时实体槽位永不压缩、永远注入——这是 needle 测试的正解；`shared/memory/` 对所有 agent 可读，私有记忆优先于共享记忆。

**技能**：`skills/` 下每个含 SKILL.md 的子目录为一个技能，命中任务时把技能正文注入当轮上下文（Phase 31 的 prompt 插件机制承载）。

## 4. API 与界面

- API：`GET/POST/DELETE /api/agents`、`GET/PUT /api/agents/{name}`（编辑 agent.yml 字段与 prompt.md）、`POST /api/chat` 增加 `agent` 字段（默认 default）
- 界面：侧边栏顶部加 **agent 选择器**（对齐 DSH 的工作区选择位置），会话列表按当前 agent 过滤；设置页新增"Agent"分组：agent 列表管理 + agents 根目录路径配置（外化进 config_schema）
- 权限联动：agent 的 workspace/ 目录绑定为文件工具根；`shared/` 对文件工具只读开放

---

## Phase 34 · Agent 实体与目录体系

````
@docs/03-模块实现规格.md @pocket-agent-docs/20-Agent实体与目录体系.md

实现 Agent 目录体系并顺带修复会话持久化（文档 20 全文规格）：

1. agents 服务插件：AgentService + AgentProfile 按 §3 规格；
   首次启动自动创建 agents/ 根目录、shared/ 三个子目录、
   default agent 脚手架；目录名合法性校验（防路径注入）
2. 会话持久化修复：sessions_dir 改由 agents 服务按 agent 提供
   （agents/<name>/sessions/）；sessions_dir 缺失时启动 WARNING，
   web 模式强制要求；旧 .sessions/ 自动迁移到 agents/default/
3. prompt.md 装配进 Planner 与 final_answer 的 system 区段；
   memory/entities.json 实体槽位永不压缩永远注入；
   shared/memory 只读共享、私有优先
4. skills/ 目录扫描：含 SKILL.md 的子目录注册为技能，
   命中任务时注入当轮上下文（走 prompt 插件机制）
5. API：agents CRUD + chat 带 agent 字段；
   前端侧边栏顶部 agent 选择器、会话按 agent 过滤、
   设置页 Agent 分组（含根目录路径配置项）
6. 测试：
   - 脚手架创建完整性（六成员齐全）
   - 会话落盘到正确 agent 目录、重启后恢复
   - 旧 .sessions 迁移幂等（迁移两次不重复不丢失）
   - 实体槽位不被滚动摘要压缩
   - shared 目录跨 agent 可读、私有记忆优先
   - 目录名路径注入攻击拦截（../ 等）
   - 删除 agent 保留文件（keep_files）与彻底删除两路径

验收：pytest 全绿；pnpm build 后实测——重启服务会话列表
保留历史；创建 agent "写手" 后 agents/写手/ 六成员齐全；
该 agent 会话与 default 会话互相隔离；shared/memory 写入后
两个 agent 都能读到。
````

---

## 注意事项

1. 本 Phase 的隐藏价值：agent.yml + prompt.md + skills/ 的结构就是 DSH Skills 与 Agent 预设的对应物——目录约定先行，生态后至
2. agents 根目录路径改非默认位置时，已存在的 agents 不自动搬迁（只对新创建生效），文档和设置页都要写清楚
3. 会话持久化修复的回归验证要覆盖"升级用户"：旧版本启动过、有 .sessions/ 的用户升级后历史不能丢
4. default agent 不可删除（UI 隐藏其删除按钮，后端拦截）
