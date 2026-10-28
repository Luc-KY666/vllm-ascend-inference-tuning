# Goal Plus 搜索流程

本文件定义一次 vLLM-Ascend 参数搜索在已安装 Goal Plus 公开接口上的通用调用顺序。
模型、指标、参数、资源、预算和任务阶段均由当前 task 与外部 SearchSpec 冻结；本文不
包含任何任务级合同。工具参数以当前注册 schema 为准，冻结前可用公开 schema 查询接口复核。

## 参与角色

- Main：读取 task 的 `goal.md`、`goal-pi.md`、`workflow.md`、`input.md`，准备 origin，
  冻结 SearchSpec、独立 verifier、reference、edit surface 和必要依赖。
- worker：在被授权的 edit surface 内提出候选字段和可检验假设，不启动验收服务，不修改
  workload、verifier、阈值、资源或全局排名。
- 独立 verifier：执行预检、部署、服务身份验收、固定 workload、官方测量、证据导出和清理，
  只把官方 raw result 中的冻结指标交还 Goal Plus。
- Goal Plus：持有 Goal、Search、candidate、Evidence、selection 和 promotion 的权威状态。

没有有效 baseline、独立 verifier 或完整评价合同时，不开始声称收益；只记录阻塞并继续可
执行的诊断工作。

一次 Search 只建立一份 Search 级环境 probe。probe 必须在首个 origin/baseline verifier
之前完成，并将 `environment_probe.json`、sandbox 证据和两个完整 vLLM help artifact 的
路径与 SHA256 写入冻结合同。后续所有 deployment、benchmark、candidate 和 promotion
attempt 只能读取这份 artifact；不得再次调用环境 probe、CANN/sandbox/端口发现或 vLLM
help。任何需要确认 `vllm serve` 或 `vllm bench serve` 参数、能力或默认语义的步骤，都
必须直接查阅首次 probe 的 `vllm-serve-help-all.txt` 或 `bench-serve-help-all.txt`，不得执行
`vllm serve --help`、`vllm serve --help=all`、`vllm bench serve --help`、
`vllm bench serve --help=all` 或其他 help 变体。上下文或 fingerprint 变化时关闭当前
Search 并创建新的 Search。

如果 deployment 或 benchmark Skill 的普通流程要求在 attempt 内“调用 probe 一次”，该要求
在 Search 绑定上下文中由 Provider 注入的 Search 级 artifact 满足；verifier 必须跳过该次
本地调用，改为校验并引用既有 artifact，不得按 attempt 重新执行 probe。

## 冻结前 clean-source 门禁

每一次 `goal_plus_search_freeze_spec` 都必须先执行一次 `clean-source`，包括修复 verifier
后的重新 freeze。检查对象是独立的 Search source，不是候选 worktree 或运行日志目录。门禁
至少扫描以下内容，并把 source HEAD、source digest、扫描清单和结果写入 Provider 指定的
diagnostics Evidence：

- `git status --porcelain=v1 --untracked-files=all` 的 tracked 修改和未跟踪文件；
- 递归出现的 `__pycache__`、`*.pyc` 和临时符号链接；
- `.git/MERGE_HEAD`、`CHERRY_PICK_HEAD`、`REVERT_HEAD`、`BISECT_LOG`、`rebase-merge/`、
  `rebase-apply/`、`sequencer/`、`index.lock` 等未完成 Git 操作；
- source 目录中不属于冻结合同的临时链接、临时文件和其他未登记生成物。

门禁失败时不得直接生成 FrozenSpec。尤其是 source 中发现未跟踪 `*.pyc` 或
`__pycache__` 时，当前 freeze 必须拒绝；它们不能因被 `.gitignore` 忽略或看似由本轮生成而被
当作 clean。先按以下分级策略修复，再完整重扫：

1. 只有能证明由本轮生成、且位于预先登记 allowlist 的 `__pycache__`、`*.pyc` 或临时链接
   才能自动清理；修复前保存路径和摘要，修复后记录动作、结果和新摘要。清理后必须重新
   执行 freeze 前的完整扫描，不能在同一个被拒绝的 freeze 中继续。
2. 未跟踪业务文件、tracked 修改、未知链接和未完成 Git 操作必须保留现场，记录 owner、
   原因和恢复要求；不得自动执行 `reset`、`clean`、`stash`、Git abort 或覆盖文件。
3. 每次修复后都必须重新执行完整扫描。相同问题未变化或出现新问题时直接阻塞，不循环重试；
   只有最终扫描为空且 source HEAD/digest 稳定时才继续 freeze。

修复过程本身不得写入 source/worktree；修复 Evidence 必须写入
`GOAL_PLUS_VERIFIER_DIAGNOSTICS_DIR`。该目录缺失或无法写入也属于 freeze 阻塞。

## 0. 分流

只有同时满足以下条件才创建 Search：

- 存在数值指标和明确优化方向；
- verifier 能独立完成部署、压测、证据导出和清理；
- candidate 只能修改外部提供的白名单和 edit surface；
- 候选空间存在多个互不重复且有证据支撑的探索方向。

调用 `goal_plus_record_triage` 记录是否进入 Search、置信度、建议阶段、原因和缺失项。
合同不完整时进入 `spec_discovery`，不得依赖宿主默认值跳过冻结。

## 1. 冻结 SearchSpec

调用 `goal_plus_search_freeze_spec`，至少冻结：

- 当前 active Goal、workspace provider、目标和唯一指标字段及方向；
- 已通过 clean-source 的 source path、source HEAD/digest、candidate edit surface 和不可修改的
  verifier/workload 文件；
- candidate 并发、累计预算、超时和停止条件；
- 已注册资源池、设备组、端口、物理锁、Goal Plus 注入的 deployment/benchmark 输出目录和
  verifier 持久化存放契约；其中每个 attempt 的临时输出由
  `GOAL_PLUS_VERIFIER_TMPDIR` 给定，持久化 Evidence 由
  `GOAL_PLUS_VERIFIER_DIAGNOSTICS_DIR` 给定，两个目录都必须位于 source/worktree 外；
- 一个用于排名的 process verifier 以及 promotion verifier；
- Search 级唯一环境 probe 及其 artifact 路径、SHA256 和执行上下文；
- verifier artifact 的相对路径和稳定输出格式。

冻结前用公开模型接口确认 worker 模型，并用隔离样本检查 verifier 对
`valid`、`invalid`、`error` 的判定。冻结失败不生成 FrozenSpec。

## 2. 创建 run

调用 `goal_plus_search_create(goal_plus_id, frozen_spec_id, request_id)`。只有回执
`state=linked` 才能规划 candidate；相同 `request_id` 重试必须返回原 run，不创建重复状态。
资源锁从服务启动前一直覆盖到清理结束。

## Attempt 状态机与操作前门禁

每个 attempt 的公开状态必须按以下生命周期解释，Goal Plus/runtime 的状态是唯一真相：

状态序列固定为 `queued -> running -> settling -> valid|invalid|error`，不得跳过 `settling`。

| 状态 | 含义 | 允许的下一状态 |
| --- | --- | --- |
| `queued` | 已接受派发但 verifier 尚未开始 | `running` |
| `running` | verifier 已启动并可能持有服务、设备、端口或锁 | `settling` |
| `settling` | verifier 已返回或进入收尾，正在清理、持久化 Evidence 或等待公开结算 | `valid`、`invalid`、`error` |
| `valid` | 报告、评分字段、Evidence、清理和状态核对均成功 | 终态 |
| `invalid` | 候选、质量或确定性容量边界不满足合同 | 终态 |
| `error` | 环境、基础设施、未知故障或清理失败，无法证明候选结果 | 终态 |

客户端查询超时只意味着“继续观察”：不取消、不重试、不重新派发，也不得把远端 attempt
改成失败或终态，不得替换 `call_id`。继续前使用公开 wait、recovery
或 status 读取真实状态。

### Verifier 长耗时等待与健康轮询

模型部署和压测可能持续很久。派发 Verifier 时必须根据冻结 workload 计算完整等待预算；未有
更严格的合同覆盖时，`verifier_wait_timeout_s` 至少预留 `1800s`，不能使用几秒或几十秒的
客户端默认超时代替该预算。这个预算是允许 Verifier 完成的总时间，不是把一次短查询超时当作
取消信号；到达预算前不得取消、重试或重新派发同一 attempt。

每个 attempt 必须在 `GOAL_PLUS_VERIFIER_TMPDIR` 建立唯一的 `verifier.log`（或等价的 Verifier
执行日志），并在启动、
预检、部署、benchmark、清理和结算阶段写入带时间戳的结构化状态/心跳。日志至少包含
`attempt_id`、`call_id`、阶段、进程身份、最近一次成功检查和错误分类；日志持久化副本及其
摘要 hash 写入 `GOAL_PLUS_VERIFIER_DIAGNOSTICS_DIR`，不得写回 source/worktree。

Main 必须以不大于 `log_poll_interval_s` 的短周期（默认建议 `15s`）轮询该日志，同时读取
公开 wait/recovery/status，而不是一次阻塞到 `1800s`。每次轮询都要确认：

- 最近心跳仍在 `heartbeat_interval_s` 容忍窗口内（长 benchmark 也必须由 wrapper 继续发心跳）；
- 公开状态仍显示调用和 Verifier 活跃，且没有终态或清理失败；
- 日志没有 `error`/`failed`/进程退出/无法解析的终止记录，并且资源健康检查仍在更新。

日志出现明确错误、进程已退出、公开状态已进入错误终态，或连续超过心跳容忍窗口无法证明
健康执行时，立即停止等待并进入 `settling -> error` 收尾；保留日志和最后状态，不再消耗剩余
等待预算。仅仅没有新的 benchmark 进度但心跳、进程和公开状态均健康时继续观察。日志轮询只
用于发现 liveness/错误，最终 `valid|invalid|error` 仍以公开 Verifier 结果和收尾 Evidence 为准。

### Attempt 运行时与清理

每个 attempt 必须使用统一的服务启动 wrapper，让 probe、serve、benchmark 和清理处于同一执行
上下文。若 Goal Plus verifier 用 `env["PYTHONPATH"] = str(cwd)` 覆盖环境，wrapper 必须从仍
存活的 `ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_LATEST_HOME` 重建 CANN 的
`python/site-packages` 与 `opp/built-in/op_impl/ai_core/tbe`，再与候选源路径组成服务子进程的
`PYTHONPATH`。

wrapper 必须设置 `PYTHONDONTWRITEBYTECODE=1`。服务停止时除进程组清理外，还必须遍历 `/proc`
清理 engine/worker 后代，并在 attempt 结算前确认没有孤儿进程、NPU 残留或未释放锁。任何
清理失败都必须保留真实 Evidence，并按本流程的 `settling -> error` 规则收尾；不得把未清理
现场交给下一个 attempt。

Main 在以下每个动作前都必须先读取公开 recovery/status：派发 candidate、steering、停止
session、`session_run`、`select` 和 promotion。只有同时确认以下条件才可继续：

- `active_invocation=false`；
- `running_verification=false`，且没有 `queued`、`running` 或未结算的 `settling` attempt；
- `unsettled_workspace_changes=false`，服务、后代进程、端口和设备锁均已完成收尾。

任一条件不满足时只允许等待、读取公开状态或恢复原 attempt；不得执行下一次 mutation。

### Main 固定续跑循环

Main 续跑不能用新的 Goal 身份覆盖现场，固定按下面的循环执行：

1. 保留原 `attempt_id`，调用 `goal_plus_search_run_verifier(attempt_id=原_attempt_id)` 查询/等待
   该 attempt 的公开结算。查询超时只继续这个循环；不取消、不重试、不重新派发，也不改变原
   `attempt_id` 或原 verifier 日志。
2. 每次返回都重新读取公开 recovery/status。只有原 attempt 已进入
   `valid|invalid|error`，并且同时满足 `active_invocation=false`、`running_verification=false`、
   `unsettled_workspace_changes=false` 三个门禁，才允许离开循环。
3. 需要继续探索时调用 `goal_plus_session_run`，必须生成全新的 `call_id`（新的
   `session_run` call，即新 call_id）；绝不拼接旧 Goal ID、旧 attempt ID 或旧 call_id 生成；不得覆盖活跃 attempt。
   续跑后回到第 1 步处理新 attempt。

任一门禁为 true，或仍有 `queued`、`running`、`settling` attempt 时，只能等待、查询或恢复原
attempt；不得调用新的 `session_run`、steering、停止、select 或 promotion。

## 3. 规划和物化 candidate

调用 `goal_plus_search_plan_next(run_id, requested_k=budget.max_parallel)`，确认规划数量
等于冻结并发。随后一次性调用 `goal_plus_search_start_batch` 提交完整 proposals；并发大于
一时，所有 proposals 必须位于同一 batch。

每个 proposal 至少包含 intent，最好包含 hypothesis、instructions 和 expected tradeoff。
重复调用只读取既有 plan/batch，不追加或替换候选。

## 4. 运行 candidate session

对每个 candidate 调用 `goal_plus_session_run`，同一个请求的幂等重试使用稳定 `call_id`；恢复
或继续已有 session 时遵守上面的固定续跑循环，不重放旧 session 历史。worker 只能交付候选字段
和假设，不能自行运行独立验收或嵌套搜索。

### Attempt 阶段与结果归因

| 阶段 | 可变内容 | 目的 |
| --- | --- | --- |
| baseline | 推荐值或冻结默认 | 建立可比较 origin |
| candidate | edit surface 内的候选值 | 搜索候选空间 |
| promotion | 获选值，可使用新 attempt 目录 | 独立复验与交付 |

baseline 失败、环境故障和候选失败必须分开归因。每个候选配置只提交一次；重复测量只能作为
诊断，不能重试到通过。

## 5. 独立 verifier 与结算

verifier 必须只读候选字段和受信任的 provider 注入，并在一次 attempt 内完成：

1. 读取配置、设备组、端口、锁和输出目录。
   输出目录必须来自 `GOAL_PLUS_VERIFIER_TMPDIR`，且本次 deployment、benchmark、raw result、
   日志、脚本和 `.work` 均不得写入 source/worktree；持久化报告和 Evidence 只能写入
   `GOAL_PLUS_VERIFIER_DIAGNOSTICS_DIR`。
2. 复用 Search 级唯一 probe，校验路径、schema、上下文和 SHA256；不得重新探测。需要
   CLI 能力或参数信息时读取 probe 保存的 help artifact，不得重新执行任何 help 命令。
3. 调用部署 Skill 完成资源、配置、静态检查、服务启动、健康、身份和最小请求验收。
4. 将部署交还的服务事实传给压测 Skill，执行冻结 workload 和官方命令。
5. 从官方 raw result 读取唯一评分字段，输出最后一行 JSON，包含字段、方向、attempt 身份
   和 `outcome=valid|invalid|error`。
6. 导出部署/压测 artifact，停止本次服务并核对资源清理。

部署或压测的长阶段必须由 wrapper 记录心跳并由 Main 持续轮询执行日志；不能只等待一个长
客户端请求返回。Verifier 日志、公开状态和进程健康三者不一致时，以更保守的错误路径收尾，
不得为了等满预算而掩盖已经发生的失败。

资源、配置、服务身份、请求完整性、raw result、指标缺失、超时和清理问题必须保留真实
失败分类；不得把连接失败转换为零分，也不得重试到通过。

静态 OOM 预检若确定模型配置超出冻结资源容量，必须在本次 attempt 内记录
`capacity_boundary`/`static_oom` Evidence，完成清理后将状态结算为 `invalid`。不得启动服务、
不生成有效性能分数、不转成 transient `error`，也不消耗重试来“验证”同一边界；只有无法
确认原因的环境或基础设施 OOM 才进入 `error`；未知的环境或基础设施 OOM 不得伪装成容量边界。

新 attempt 的 verifier 由 `goal_plus_search_run_verifier` 完成结算。子 agent 调用带
`agent_session_id` 和一句话 hypothesis；Main 的 promotion 复验不带该字段，并只在 promotion
scope 使用复验参数。恢复或续跑已有 attempt 时统一回到上面的固定循环，不能把恢复当成新的
verifier 派发。

## 6. 观察与续跑

- 可使用 `goal_plus_session_wait_any` 获取任意完成 lane 的提示，避免逐 lane 轮询；它不能替代
  原 attempt 的 `goal_plus_search_run_verifier(attempt_id=…)` 结算循环。
- 使用 `goal_plus_search_list_history` 读取排名、scores、key metrics 和失败记录。
- 只有 SearchSpec 明确启用自适应分配时，才查询和应用 allocation decision。
- 预算、候选空间、收益阈值或环境阻塞满足后停止派发，否则以新 call 继续已有 session 或规划下一批。

attempt 中断或客户端超时时回到上面的固定续跑循环；该循环负责原 `attempt_id` 的观察/恢复、
锁与清理核对，以及新 `session_run` call 的生成。

## 7. 选择与提升

`goal_plus_search_select` 和 `goal_plus_search_promote` 同样受上述操作前门禁约束；任何活跃
调用、运行中的 verifier 或未结算 workspace changes 都必须先收尾，不能覆盖活跃 attempt。

调用 `goal_plus_search_select` 获取与实测配置和 artifact 一致的获选 candidate。随后调用
`goal_plus_search_promote` 执行独立复验；`goal_plus_search_apply_promotion` 按冻结 mode
交付 artifact 或写回源码。没有有效收益时仍必须复验实际选择的回退配置。

## 8. 终审与报告

依次调用：

- `goal_plus_record_search_result`
- `goal_plus_run_final_check`
- `goal_plus_search_report`
- `goal_plus_set_status`

最终 Search report/Evidence 至少包含：

- baseline 与 final 的同口径原始样本、唯一指标、seed、workload digest、重复次数、预热和
  统计公式；baseline/final 必须使用同一比较合同，不能只给两个汇总值；
- baseline/final 的模型、服务和候选配置摘要和 hash，实际物理设备/lane 身份、端口、资源锁
  和服务生命周期；
- 与指标方向一致的收益公式和代入值。最小化指标使用
  `(baseline - final) / baseline`，最大化指标使用 `(final - baseline) / baseline`；缺失
  有效两端样本时不得计算收益；
- 重复/成对测量策略、离散度、设备轮换、并发争用、调度抖动等噪声说明和解释边界；
- 所有失败候选、attempt 状态、失败分类、原因、是否可重试和未覆盖搜索空间；
- 服务/子进程、端口、设备锁、`GOAL_PLUS_VERIFIER_TMPDIR`、
  `GOAL_PLUS_VERIFIER_DIAGNOSTICS_DIR` 和 worktree 的清理状态，以及相关 Evidence hash；
- 明确声明：本次 Search 结果不是用户业务仓的最终 E2E 证明，仍需用户在真实业务仓、目标
  设备和正式发布方式下独立完成业务精度、部署和性能验收。

## 9. 失败隔离与恢复

verifier 不充分时调用 `goal_plus_search_invalidate_run` 原子阻止新规划、session、结算、
选择和提升；停止并关闭候选后修复 verifier，使用新 run 继续。只读诊断使用公开 monitor
snapshot；恢复状态使用 recovery status/prepare/recover 接口，不修改 Goal Plus 私有状态。
输出目录和持久化 Evidence 遵循 Provider 输入契约；清理或修复失败时保留 source/worktree
现场并阻塞 freeze/select/promotion；task 和 bridge 不推断其他输出路径，也不自动 reset/clean/stash。
