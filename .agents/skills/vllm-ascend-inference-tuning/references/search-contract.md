# Goal Plus 搜索合同

本文件定义顶层 skill 在进入参数搜索前必须核对的合同字段。它不实现 Goal Plus API，
也不规定某个任务的具体指标或数值。

## 最小 SearchSpec 内容

```json
{
  "workspace": {
    "provider": "git_worktree",
    "max_parallel": 1
  },
  "model": {
    "model_path": "<任务输入>",
    "served_model_name": "<任务输入>",
    "config_digest": "<冻结摘要>"
  },
  "deployment": {
    "fixed": {},
    "searchable": {
      "parameter_name": ["task-defined"]
    },
    "edit_surface": ["task-defined"]
  },
  "workload": {
    "scenario": "<冻结场景>",
    "digest": "<冻结摘要>",
    "requests": 0,
    "warmups": 0,
    "rounds": 0
  },
  "metric": {
    "name": "<冻结指标>",
    "direction": "minimize",
    "absolute_gates": {},
    "relative_gates": {}
  },
  "budget": {
    "max_candidates": 0,
    "max_attempts": 0,
    "timeout_seconds": 0
  },
  "resources": {
    "device_pool": "<GP 冻结引用>",
    "ports": ["<GP 分配值>"],
    "resource_lock": "<GP 冻结引用>"
  }
}
```

占位符不是默认值。任务必须提供真实值，且 Main 在冻结前检查：

- 可调参数来自当前 vLLM 版本实际支持的 CLI；
- candidate 不能修改模型、workload、verifier、reference、阈值和资源；
- `direction`、硬门槛、停止条件和预算能由独立 verifier 执行；
- 每个 candidate 的设备、端口、缓存、日志、临时目录和服务进程隔离；
- baseline 与 candidate 使用同一输入、工具链、统计口径和资源预算。

## 指标和结果

搜索指标必须与 Goal Plus 的 `metric_name` 对齐。部署和压测 artifact 只提供原始事实，
不能自行产生最终排名。

候选结果至少分为：

| 类型 | 处理 |
| --- | --- |
| `invalid` | 配置、协议、质量、绝对 SLO 或相对门槛失败；保留真实数值和原因 |
| `error` | 环境、设备、服务、证据导出或清理失败；不能伪造零分 |
| `valid` | 独立 verifier 完整通过，可进入 GP 公开排名 |

同一配置的波动用于诊断，不能在候选失败时重试到通过。最终获选配置必须在任务要求
的 provider 和设备上复验，且 selection、实测参数和 publication 摘要一致。

## Provider 边界

### git_worktree/direct

- GP 创建独立 candidate worktree 和 local verifier。
- 服务启动、健康、正式请求、benchmark、清理和 artifact 导出在同一次持锁执行内完成。
- verifier 使用 `GOAL_PLUS_VERIFIER_TMPDIR` 或任务明确的隔离目录。
- 设备和端口由 GP 注入，skill 不手工改写。

### ThinkThread 路径

- worker 在授权设备上预检、启动服务并交付参数和服务接口。
- snapshot/fs.run verifier 只请求、采集、分析和导出证据。
- 短 verifier 不启动、重启或停止服务。
- Main 在最终交付前核对当前服务实际配置，必要时在 worker 生命周期外按公开流程重启并
  复验。

## 交付清单

Main 收尾前核对：

- baseline、每个 candidate 和 promotion attempt 的配置摘要及 hash；
- 原始 SSE/raw JSON 只保存一份，引用路径和 hash 可读；
- TTFT、TPOT、E2EL 的统计样本数、P50/P95 和缺失规则；
- 设备、锁、端口、服务启动/测量/清理时间区间；
- selection 与最终配置一致；
- 无收益时 baseline 经过真实独立复验；
- 未覆盖边界和用户必须在真实环境完成的 E2E 已写入 TASKS。
