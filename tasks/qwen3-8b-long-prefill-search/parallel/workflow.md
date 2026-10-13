# 并行 Long-Prefill 工作流

本文件定义 `parallel/` 任务的执行顺序。入口中的用户参数集中维护本任务的模型路径、
数据集根目录和候选预算；本文件引用该冻结来源。

## 1. 准备与身份确认

Main 首先读取宿主根规则、`TASKS.md`（若宿主提供）、本任务资料和三个指定 Skill。
确认 Goal Plus 的公开 provider 已绑定 `git_worktree`、`direct runtime` 和
`local verifier`，并确认 `artifact_only` 可用。Goal、candidate、设备租约、物理锁和
output surface 以公开 provider 回执为准。

部署输入必须重新核对：

- 模型路径可读，served model name 为 `qwen3-8b`；
- 官方 `config.json`、tokenizer、dtype 和上下文事实可解析；
- 任务 Python、vLLM、vLLM-Ascend、CANN/ATB wrapper 和 ABI 一致；
- provider 注入两条彼此独立的授权设备 lane；
- 两个端口、共享资源锁和两个 attempt 输出目录由 provider 提供，且各自位于授权边界内。

任一事实无法确认时，记录 `unknown` 或并行接线阻塞，停在准备阶段。

## 2. 默认值与 baseline

Main 使用目标环境的 `vllm serve --help=all` 和 `vllm bench serve --help=all` 解析
任务未固定的参数。解析结果和版本摘要写入本次冻结 Evidence，实际冻结值以目标环境
解析结果为准。

先生成 recommendation-mode 的部署输入，执行部署 Skill 的静态 OOM 和最小 TP 诊断。
推荐模式只执行诊断；推荐状态可用并返回 TP 后，Main 生成完整部署输入。完整
baseline 使用推荐 TP、DP=1 和 CLI 默认的 `max_num_seqs`、
`max_num_batched_tokens`，重新执行静态预检、服务身份验收和官方 benchmark。

baseline 必须包含可核对的 deployment report、benchmark result、raw result 路径、
配置摘要、workload 摘要和清理状态，以及完整 baseline artifact。

## 3. 冻结 SearchSpec

冻结前记录：

- 模型、served model name、tokenizer 和实际配置摘要；
- 四个可调参数的白名单和 edit surface；
- long-prefill 预设、数据集解析结果、请求速率和并发；
- 官方 raw result 的 TTFT P95 字段映射；
- candidate 总预算、单次超时、设备池、两条端口、锁和两个输出目录；
- 清理责任、Evidence 导出方式和最终复验次数。

候选空间必须由当前 CLI、模型 attention heads、两条 lane 的设备数量和预检事实过滤。
TP 必须能均匀切分 attention heads，DP 与 TP 的设备乘积上限为当前 candidate 交付的
设备数量，两个批处理参数必须是正整数且满足输入契约。若剩余空间不足两份不同于
baseline 的配置，报告 `incomplete`，并保留既有 workload 和约束。

## 4. 双 lane candidate

Search 使用两个独立 candidate lane，`max_parallel=2`。一次 batch 最多派发两个
尚未评估的参数组合。每个 lane 必须固定自己的设备、端口、锁、临时目录和服务进程，
并在独立 verifier 生命周期内完成：

1. 生成本 candidate 的 deployment input。
2. 执行静态 OOM 预检。
3. 启动服务并验收 health、`/v1/models`、served model identity 和最小请求。
4. 使用官方 `vllm bench serve` 执行 production long-prefill。
5. 保存唯一 raw result、规范化 benchmark report 和 hash。
6. 按 provider 合同停止服务并确认清理。

所有 lane 共享一个 `budget.max_attempts` 总数。一次派发两个 candidate 消耗两个
attempt；失败候选也计数，同一候选配置只提交一次。至少完成两批候选
并有完整 Evidence 后才进入公开 selection。若某 lane 清理、资源或服务状态未知，
停止继续派发并保留在途和失败证据。

candidate 的修改范围限定为四个可调部署参数和允许的输出。批次全部结算后 Main 只读取
独立 verifier artifact，比较 TTFT P95 和绝对测量，形成下一批有证据的配置假设。
Main 单写 `TASKS.md` 和汇总。

## 5. 选择、复验与收尾

满足以下条件后才允许进入公开 selection：

- 两条 lane 的授权、端口、锁和 output surface 均可核对；
- 至少两批候选完成独立测量；
- 至少一批中两份候选的正式 benchmark 时间区间存在重叠；
- 选择所需的 raw result、配置摘要、清理和 Evidence 均完整。

预算耗尽、候选空间耗尽或环境错误时停止派发。选择结果以 GP 公开 artifact 为准。

最终 verifier 在同一 provider 资源合同下独立测量 baseline 和获选配置，使用相同的
long-prefill、production、64 并发和 5 req/s 合同。复验后导出报告，停止服务，确认
两条 lane 的端口、设备、锁和临时目录均按合同释放。存在独立复验或清理缺口时报告
`incomplete`。
