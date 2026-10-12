# 27 — Shell 工具与权限规则引擎（Phase 40）

> 编号接 Phase 39。写能力已就位，本期放开执行能力，同时把"三档太粗"的权限补上规则引擎——**shell 不受路径围栏约束，规则引擎是它唯一的闸门，两者必须同 phase 上线。**

---

## 一、设计要点（先说清边界）

1. **shell 工具绕过路径围栏**：`run_command("cat /etc/passwd")` 物理上拦不住。防线是：规则引擎默认 ask（弹审批）+ 内置灾难命令 deny + readonly 权限下 shell 不进候选列表。文档和交付说明必须如实写明这一点。
2. **规则引擎替换现有审批判断**：runner:412 目前只有"非 cacheable 且 permission≠full 就弹审批"一条逻辑，本期改为规则求值三态：allow（直接跑）/ ask（走审批）/ deny（直接拒绝）。
3. 用户环境是 Windows：subprocess 用 `shell=True`（Windows 走 cmd、POSIX 走 sh），不指定具体解释器。

---

## 二、Phase 40 提示词（贴给 Cursor）

```text
你是一名资深 Python 工程师，继续开发 pocket-agent（本地小模型 agent 框架）。本期目标：shell 执行工具 + 权限规则引擎。红线：一切皆插件；shell 上线必须同批带规则引擎；不破坏现有 cacheable/审批语义（无规则配置时行为与现状完全一致）；所有新行为必须有测试。

═══ 任务 A：权限规则引擎（新模块 src/pocket_agent/permissions/__init__.py）═══
1. 数据模型：
       @dataclass
       class PermissionRule:
           effect: str      # "allow" | "ask" | "deny"
           tool: str = "*"  # fnmatch 模式，匹配工具名
           pattern: str = ""  # 正则，匹配 json.dumps(args, ensure_ascii=False)；空=不约束
           note: str = ""   # 拒绝/询问时展示给用户
   求值：按列表顺序首个命中者生效；无命中 → 回退默认：cacheable 工具 allow，非 cacheable 工具 ask。
2. 内置兜底规则（追加在用户规则之后，可被用户规则抢先命中）：
   - deny 灾难命令：tool=run_command，pattern 分别覆盖 `rm\s+-[a-z]*r[a-z]*f`、`del\s+/[fsq]`、`format\s+[a-zA-Z]:`、`:\(\)\s*\{`、`mkfs`、`dd\s+if=`（逐个建规则，note 用中文说明原因）
   - ask：tool=run_command（其余一切命令）
   - ask：tool 匹配 write_file/edit_file（保持现状语义）
3. AgentConfig 增加字段 `permission_rules: list[dict] = []`，三层合并沿用 from_layers（settings.json 的 permission_rules 覆盖 pocket.yml 默认层；request overrides 不开放此键，防止请求层越权）。
4. 单元测试：顺序短路、tool glob、pattern 正则、无命中回退、用户规则抢先于内置 deny、overrides 注入 permission_rules 被忽略。

═══ 任务 B：runner 接入规则求值 ═══
src/pocket_agent/runner/__init__.py，替换现有审批判断（约 :412）：
    verdict, rule = evaluate(self._config.permission_rules, tool.name, args, cacheable=tool.cacheable)
    if verdict == "deny":
        result = f"[已拒绝] 命中权限规则：{rule.note or rule.pattern or rule.tool}"
        记录 StepRecord(error=result)，emit step.intercepted reason="permission_rule"，guard.on_step 正常计数，continue
    elif verdict == "ask" and self.approval_gate is not None and permission != "full":
        走现有 approval.required 流程（payload 里附 rule note）
    else: 直接执行
注意：deny 不得计入"连续拦截停止"之外的副作用，被拒步骤要像普通拦截一样写轨迹。
测试：deny 规则下文件未被创建且轨迹含 permission_rule；ask 触发审批；allow 白名单命令（如 rule: allow/run_command/^echo ）跳过审批直接执行；无规则配置时行为与现状一致（非 cacheable 仍弹审批）。

═══ 任务 C：run_command 工具（新插件 src/pocket_agent/plugins/shell_plugin.py）═══
1. 插件名 "shell"，requires=["tools", "workspace"]，pocket.yml 在 workspace 之后加 `- name: shell`。
2. 工具（cacheable=False，描述 ≤20 字）：
       def run_command(command: str) -> str
   - cwd 锚定会话工作区根（manager.resolve_path(sid, ".")）；
   - subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60, cwd=...)；timeout 与输出上限走插件 config（timeout_sec 默认 60、max_output_chars 默认 4000）；
   - 返回格式：`退出码 0\n{stdout+stderr 合并}`；超时 → `[工具错误] 命令超时（60s），进程已终止`；非零退出码正常返回文本（让模型自纠），不抛异常；
   - 输出超上限截断并附 `…（输出过长已截断，共 N 字符）`。
3. readonly 权限下 shell 不进候选：runner 候选裁剪处（Phase 39 已为写工具加过过滤）把 run_command 一并排除，补测试。
4. 测试（全部用真实 subprocess，不 mock）：echo hello 成功；非零退出码透传；超时终止（用 sleep/timeout 命令按平台选择，Windows 用 ping -n）；输出截断；cwd 在工作区（命令打印当前目录断言）；readonly 时工具不在候选。

═══ 任务 D：意图与界面 ═══
1. intent_router _TOOL_HINTS 确认含：运行、执行、命令（无则补）。
2. 审批弹窗/轨迹已能展示 run_command 的 args（command 字段），确认 Web 审批弹窗对长命令换行显示正常（CSS word-break），不改逻辑。
3. settings 的 permission_rules 本期只支持编辑 settings.json 生效，不做 UI（文档注明）。

═══ 验收（必须全绿再交付）═══
1. pytest -q
2. grep -rn "MockLLM" src/ 无输出
3. 手测清单写进交付说明：
   - "运行 echo hello" → 弹审批 → 允许 → 输出 hello
   - settings.json 配 {"permission_rules":[{"effect":"allow","tool":"run_command","pattern":"^echo ","note":"echo 白名单"}]} 后 "运行 echo hi" 不弹审批直接执行
   - "运行 rm -rf /" → 直接被 deny，不弹审批，轨迹可见 permission_rule
   - permission=readonly 时让 agent 执行命令 → 候选无 run_command
4. 交付说明必须写明：shell 不受路径围栏约束，防线 = 规则引擎 + 审批 + readonly 排除。
```

---

## 三、使用方式

1. 本文档放入 `pocket-agent-docs/`。
2. 贴"二、Phase 40 提示词"整段给 Cursor，一次完成 A–D。
3. 本期纯后端，不涉及 `pnpm build`。
4. 验收三条 + 手测四条结果贴回来。之后 Phase 41 建议跑真实基准（D1-D9 三轨出数字）——工具齐了，该交卷了。

## 四、配置说明（无 UI）

`permission_rules` 本期只支持编辑 `settings.json` 生效，设置页不做表单。示例：

```json
{
  "permission_rules": [
    {"effect": "allow", "tool": "run_command", "pattern": "^echo ", "note": "echo 白名单"}
  ]
}
```

三层合并：pocket.yml 默认 < settings.json < 请求 overrides；**overrides 不开放 `permission_rules`**，防止请求层越权。

`run_command` **不受路径围栏约束**。防线 = 规则引擎（内置灾难 deny + 默认 ask）+ 审批门 + readonly 下不进候选。
