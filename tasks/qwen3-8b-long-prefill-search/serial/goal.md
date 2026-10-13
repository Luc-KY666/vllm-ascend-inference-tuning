$goal-plus mode=autonomous max_parallel=1 workspace_provider=git_worktree promotion_mode=artifact_only strategy=agent_guided 请仅使用本任务指定的 vLLM-Ascend Skill，完成 Qwen3-8B long-prefill 的 baseline、串行参数搜索、独立复验和停服交付。

## 用户参数

以下三项是本任务唯一需要用户直接修改的输入，每项在本文件中只出现一次：

- `model_path`: <模型路径>
- `dataset_root`: <数据集路径>
- `max_attempts`: <最大尝试次数>

## 目标

模型身份固定为 `qwen3-8b`。只允许读取和调用本仓库
`.agents/skills/` 下的以下三个 Skill：

1. `.agents/skills/vllm-ascend-deployment/SKILL.md`
2. `.agents/skills/vllm-ascend-benchmark/SKILL.md`
3. `.agents/skills/vllm-ascend-inference-tuning/SKILL.md`

先读取当前宿主能够提供的根 `AGENTS.md`、`TASKS.md`，以及本目录的
`README.md`、`workflow.md`、`inputs.md`、`search.md`、`modules.md` 和上述三个 Skill。
若宿主缺少必要文件或公开 Goal Plus provider 能力，记录阻塞并停止相应阶段，等待补齐
设备、端口、路径、版本、锁或凭据事实。

## 固定任务合同

- 负载选择固定为 `selection=preset`、`scenario=long_prefill`。
- 数据集使用 `long_prefill` 预设的 `likaixin/InstructCoder`、`train` 和
  `hf_output_len=128`，数据集根目录来自本节用户参数。
- 压测模式固定为 `benchmark_mode=production`。
- 最大并发固定为 `64`，请求到达速率固定为 `5 req/s`。
- 未由本任务明确指定的服务和 benchmark 参数，必须通过目标环境的
  `vllm serve --help=all`、`vllm bench serve --help=all` 或官方模型配置解析实际
  默认值后冻结；冻结记录以目标环境解析值为准。
- goodput SLO 状态固定为 `not_requested`，benchmark 命令保持无 `--goodput` 约束。
- 唯一优化指标为官方 raw result 中的 `TTFT P95`，方向为最小化。
- 唯一可调参数为 `TP`、`DP`、`max_num_seqs`、`max_num_batched_tokens`。模型、
  tokenizer、dtype、量化、上下文、采样、负载、统计口径、provider 和验收程序属于
  固定合同。

## Baseline 与搜索

先按部署 Skill 的推荐模式完成静态 OOM 和最小 TP 诊断，使用
`recommendation.recommended_tp` 作为 baseline 的 TP。baseline 的 DP 固定为 `1`，
`max_num_seqs` 和 `max_num_batched_tokens` 使用目标 vLLM 当前 CLI 解析出的默认值。
之后用完整四参数重新生成部署输入，完成服务健康、`/v1/models`、模型身份和最小请求
验收，再执行官方 long-prefill benchmark。推荐值、默认值和 provider 事实无法确认时
进入阻塞状态。

Main 必须依据当前模型配置、CLI 能力、授权设备数量和真实部署预检冻结有限候选空间，
设备范围和固定参数以冻结合同为准。candidate 的 edit surface 固定为四个可调参数；
失败 attempt 保留真实 artifact 并计入候选预算。

本任务的 `SearchSpec.budget.max_attempts` 取本节用户参数，表示整个串行搜索的候选
attempt 总数；候选预算范围限定为搜索阶段，baseline、最终独立复验和清理属于独立阶段。
入口预算小于完成最低两份
不同候选配置所需的数量时，selection 前报告 `incomplete`。

## 执行顺序

1. 核验宿主、provider、模型配置、CLI、环境 wrapper、设备、端口、锁和输出 surface。
2. 解析默认值，执行 deployment Skill 推荐模式和完整 baseline。
3. 冻结 SearchSpec、long-prefill workload、TTFT P95 指标和候选 edit surface。
4. 在一个固定 lane 上逐个执行候选：部署预检、服务身份验收、官方 benchmark、artifact
   导出和清理。
5. 每次结算后使用真实 benchmark artifact 分析下一份候选假设；Main 单写搜索编排。
6. 达到预算或停止条件后，通过公开 selection 获取最佳 artifact。
7. 在同一 provider 合同下独立执行 baseline/最佳配置对照复验，核对配置摘要一致，
   导出报告后停服并确认端口、设备和进程清理。

静态 OOM `reject`、数据集解析失败、CLI 不兼容、服务身份失败、正式请求不完整、
raw result 缺失或不唯一、TTFT P95 缺失以及清理失败都必须按 Skill 合同报告真实状态，
通过判定只接受 Skill 合同下的完整独立 Evidence。
