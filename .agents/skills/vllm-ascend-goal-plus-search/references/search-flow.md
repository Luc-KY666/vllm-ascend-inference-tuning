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
- clean source path、candidate edit surface 和不可修改的 verifier/workload 文件；
- candidate 并发、累计预算、超时和停止条件；
- 已注册资源池、设备组、端口、物理锁、Goal Plus 注入的 deployment/benchmark 输出目录和
  verifier 持久化存放契约；
- 一个用于排名的 process verifier 以及 promotion verifier；
- Search 级唯一环境 probe 及其 artifact 路径、SHA256 和执行上下文；
- verifier artifact 的相对路径和稳定输出格式。

冻结前用公开模型接口确认 worker 模型，并用隔离样本检查 verifier 对
`valid`、`invalid`、`error` 的判定。冻结失败不生成 FrozenSpec。

## 2. 创建 run

调用 `goal_plus_search_create(goal_plus_id, frozen_spec_id, request_id)`。只有回执
`state=linked` 才能规划 candidate；相同 `request_id` 重试必须返回原 run，不创建重复状态。
资源锁从服务启动前一直覆盖到清理结束。

## 3. 规划和物化 candidate

调用 `goal_plus_search_plan_next(run_id, requested_k=budget.max_parallel)`，确认规划数量
等于冻结并发。随后一次性调用 `goal_plus_search_start_batch` 提交完整 proposals；并发大于
一时，所有 proposals 必须位于同一 batch。

每个 proposal 至少包含 intent，最好包含 hypothesis、instructions 和 expected tradeoff。
重复调用只读取既有 plan/batch，不追加或替换候选。

## 4. 运行 candidate session

对每个 candidate 调用 `goal_plus_session_run`，为每次幂等重试使用稳定 `call_id`；继续探索
使用新 `call_id`，不重放旧 session 历史。worker 只能交付候选字段和假设，不能自行运行
独立验收或嵌套搜索。

## 5. 独立 verifier 与结算

verifier 必须只读候选字段和受信任的 provider 注入，并在一次 attempt 内完成：

1. 读取配置、设备组、端口、锁和输出目录。
2. 复用 Search 级唯一 probe，校验路径、schema、上下文和 SHA256；不得重新探测。需要
   CLI 能力或参数信息时读取 probe 保存的 help artifact，不得重新执行任何 help 命令。
3. 调用部署 Skill 完成资源、配置、静态检查、服务启动、健康、身份和最小请求验收。
4. 将部署交还的服务事实传给压测 Skill，执行冻结 workload 和官方命令。
5. 从官方 raw result 读取唯一评分字段，输出最后一行 JSON，包含字段、方向、attempt 身份
   和 `outcome=valid|invalid|error`。
6. 导出部署/压测 artifact，停止本次服务并核对资源清理。

资源、配置、服务身份、请求完整性、raw result、指标缺失、超时和清理问题必须保留真实
失败分类；不得把连接失败转换为零分，也不得重试到通过。

调用 `goal_plus_search_run_verifier` 完成结算。子 agent 调用带 `agent_session_id` 和
一句话 hypothesis；Main 的 promotion 复验不带该字段，并只在 promotion scope 使用复验参数。

## 6. 观察与续跑

- 使用 `goal_plus_session_wait_any` 等待任意完成 lane，避免逐 lane 轮询。
- 使用 `goal_plus_search_list_history` 读取排名、scores、key metrics 和失败记录。
- 只有 SearchSpec 明确启用自适应分配时，才查询和应用 allocation decision。
- 预算、候选空间、收益阈值或环境阻塞满足后停止派发，否则以新 call 继续已有 session 或规划下一批。

attempt 中断时先确认锁已释放，再使用公开 recovery 接口恢复原 attempt；不得拼接旧 Goal
身份或覆盖活跃日志。

客户端超时不表示 verifier 已取消。下一轮 session 或 steering 前必须通过 wait/recovery/status
确认前一 attempt 已结算、锁已释放且服务后代进程已清理；否则不得派发新的 verifier。

## 7. 选择与提升

调用 `goal_plus_search_select` 获取与实测配置和 artifact 一致的获选 candidate。随后调用
`goal_plus_search_promote` 执行独立复验；`goal_plus_search_apply_promotion` 按冻结 mode
交付 artifact 或写回源码。没有有效收益时仍必须复验实际选择的回退配置。

## 8. 终审与报告

依次调用：

- `goal_plus_record_search_result`
- `goal_plus_run_final_check`
- `goal_plus_search_report`
- `goal_plus_set_status`

交付至少包含 baseline、候选和最终配置摘要，所有报告/raw result 的路径和 hash，资源/端口/
服务生命周期，promotion 结果，失败候选和未验证项。

## 9. 失败隔离与恢复

verifier 不充分时调用 `goal_plus_search_invalidate_run` 原子阻止新规划、session、结算、
选择和提升；停止并关闭候选后修复 verifier，使用新 run 继续。只读诊断使用公开 monitor
snapshot；恢复状态使用 recovery status/prepare/recover 接口，不修改 Goal Plus 私有状态。
部署和压测只能在 Goal Plus/provider 注入的本次输出目录下写入文件；持久化 Verifier 按
Goal Plus 的 verifier 存放契约处理。task 和 bridge 不推断其他输出路径。
