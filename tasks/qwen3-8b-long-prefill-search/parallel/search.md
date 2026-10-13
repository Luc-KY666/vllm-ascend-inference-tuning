# 并行搜索合同

本文件定义 `parallel/` 任务冻结给 Goal Plus Main 的 SearchSpec 内容。模型路径、数据集
根目录和候选预算集中维护于入口参数区，本文件引用该冻结来源。

## SearchSpec

```json
{
  "workspace": {
    "provider": "git_worktree",
    "max_parallel": 2
  },
  "model": {
    "served_model_name": "qwen3-8b",
    "model_path": "来自 goal.md 用户参数",
    "config_digest": "由目标模型配置解析后冻结"
  },
  "deployment": {
    "fixed": {
      "dp": 1,
      "model": "qwen3-8b",
      "dtype": "由模型配置或 CLI 默认值冻结",
      "quantization": "由模型配置或 CLI 默认值冻结"
    },
    "searchable": [
      "tp",
      "dp",
      "max_num_seqs",
      "max_num_batched_tokens"
    ],
    "edit_surface": [
      "tensor parallel size",
      "data parallel size",
      "max number of sequences",
      "max number of batched tokens"
    ]
  },
  "workload": {
    "selection": "preset",
    "scenario": "long_prefill",
    "benchmark_mode": "production",
    "max_concurrency": 64,
    "request_rate": 5,
    "dataset": {
      "name": "likaixin/InstructCoder",
      "split": "train",
      "output_len": 128,
      "root": "来自 goal.md 用户参数"
    }
  },
  "metric": {
    "name": "TTFT_P95",
    "direction": "minimize",
    "source": "vllm_bench_serve raw result"
  },
  "budget": {
    "max_attempts": "来自 goal.md 用户参数",
    "max_parallel": 2
  },
  "resources": {
    "device_pool": "由 Goal Plus provider 冻结",
    "ports": "由 Goal Plus provider 分配",
    "resource_lock": "由 Goal Plus provider 冻结"
  }
}
```

`deployment.fixed.dp=1` 表示 baseline 的固定值。由于 `dp` 属于用户要求的可调参数，
candidate 可以在真实设备和后端约束允许时覆盖该 baseline 值；任何覆盖都必须满足
`tp * dp` 的上限为 `device_count`，并通过部署输入契约。

## Baseline

1. 以 recommendation-mode 运行部署 Skill。
2. 读取 `recommendation.recommended_tp`。
3. 生成完整 deployment input：推荐 TP、DP=1、vLLM 默认批处理参数。
4. 重新完成静态 OOM、服务身份和官方 benchmark。
5. 保存 baseline 的真实 TTFT P95 和所有原始 artifact。

缺少 recommendation、默认值或 provider 事实时，baseline 状态记为 `blocked` 或
`incomplete`，等待事实补齐后再进入排名。

## 候选空间

Main 根据目标环境的 `vllm serve --help=all`、模型 attention heads、两条 lane 的设备
数量和静态预检结果冻结有限离散空间：

- TP 的候选值来自 CLI 支持集合，能够均匀切分 attention heads，且上限为授权设备数量。
- DP 只允许 CLI 支持、正整数且满足设备乘积约束的值。
- `max_num_seqs` 和 `max_num_batched_tokens` 只允许 CLI 支持的正整数。
- 至少保留两份与 baseline 区分的可测配置；首批优先一次只改变一个参数。
- 后续组合必须引用已经结算的 benchmark Evidence。

候选 attempt 预算是入口 `max_attempts` 的单一总数。一次派发两个 candidate 消耗两个
attempt；失败候选也计数。attempt 计数范围限定为候选搜索，baseline、最终复验和清理
属于独立阶段。至少两批候选完成独立测量并有完整 Evidence 后才进入 selection。

## 负载与测量

`selection=preset` 且 `scenario=long_prefill` 时，benchmark Skill 派生 HF adapter、
`likaixin/InstructCoder`、`train`、预设相对路径和 `hf_output_len=128`。数据集 root
来自入口参数，数据集格式和解析流程固定由 benchmark Skill 提供。

production 测量冻结最大并发 64、请求速率 5 req/s。请求数、warmup、rounds、
burstiness、temperature、stream、usage 和其他未指定字段必须从当前
`vllm bench serve --help=all` 及任务当前版本的官方默认值解析，并记录实际生效值。
goodput SLO 状态固定为 `not_requested`，benchmark 命令保持默认 goodput 关闭状态。

官方 raw result 必须唯一、可解析，且包含 TTFT P95。服务身份、数据集解析、请求
完整性、失败数、指标字段和清理状态必须全部通过；任一项未通过时，候选状态记为
`invalid`、`blocked` 或 `error`，并保留完整 artifact。

## 停止与最终复验

满足任一条件后停止派发：

- 候选总预算用尽；
- 候选空间耗尽；
- 已完成最低两批候选且后续假设集合为空；
- 两条 lane 的环境、设备、服务或清理故障要求停止。

选择必须通过 Goal Plus 公开 selection。最终在同一 provider 资源合同下独立测量
baseline 和获选配置，核对 selection、deployment、benchmark 和 publication 摘要
一致，然后停服。独立复验和清理均完成后才能发布；任一缺口都将状态记为
`incomplete`。
