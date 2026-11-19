# Manifest 契约

所有 manifest 使用 `schema_version: 1`。共享 validator 位于
`pypto-common/pypto_contract.py`。模型特有字段、ABI 和 feature 规则放在
`references/<model>-adapter.md`，不写入通用 validator。

## Profile Manifest

必需顶层字段：

```json
{
  "schema_version": 1,
  "producer": "pypto-profile",
  "model": {"family": "example", "model_id": "example-1"},
  "source": {
    "kind": "npu_chip_swimlane",
    "npu_measured": true,
    "workload_id": "owner:run:dispatch:digest"
  },
  "tasks": []
}
```

Native 测量必须包含 `source.workload_id`，表示一次独立 clean timing dispatch。同一 dispatch
中的多个 task/layer 共用该 identity。Fixture 可以省略，但不得作为硬件性能证据。

每个 task 表示一个稳定 graph node，而不是 DSL statement 或 physical SPMD block。规范化后的
必需字段：

| 字段 | 含义 |
| --- | --- |
| `task_id` | capture 内唯一 node identity |
| `task_name` | 尽可能稳定的 source/name-hint identity |
| `phase` | `prefill`、`decode` 或其他明确 phase |
| `kind` | `compute`、`communication`、`memory`、`synchronization`、`control`、`metadata` |
| `resource_shape` | `AIC`、`AIV`、`MIX`、`HOST`、`NONE` |
| `block_num` | logical SPMD block count |
| `active_subslots` | 活跃 `AIC`、`AIV0`、`AIV1` 成员 |
| `dependencies` | predecessor `task_id` |
| `tensor_roles` | `weight`、`kv_cache`、`window`、`control` 等 semantic access |
| `features` | batch、active token、KV length、dtype 等动态查找 feature |
| `duration_samples_us` | 真实 task duration；inventory-only node 可为空 |
| `subslot_duration_samples_us` | 可选 AIC/AIV0/AIV1 task-completion observation |
| `subslot_block_duration_samples_us` | 可选、按epoch保存的排序physical-block duration distribution；每组长度必须等于`block_num` |
| `outputs` | control-output provider 使用的命名输出 descriptor |

`kernel_ids` 和 `metadata` 是可选透传对象。Runtime ID 只用于诊断，不得单独作为稳定 latency key。

Exact latency signature 包含 model family/model ID、task name、phase、kind、resource shape、
block count、active subslot、kernel name 和完整规范化 feature。任何会改变延迟的动态 feature 都
必须在拟合前由 exporter 提供证据并导出。

## Model Manifest

`pypto-latency-model` 按 exact latency signature 聚合 measured sample。每个 group 记录 task count、
median、min/max、confidence、signature、key，以及 profile 提供时的 per-subslot median。若profile
提供block distribution，subslot model还记录每个排序rank跨独立workload的中位duration。排序rank
用于恢复occupancy distribution，不代表原始logical block identity。当前实现是 exact median
baseline和测量域内IDW，不允许外推。

Point 的 raw sample 数或独立 workload 数任一低于配置阈值时均为 `low_confidence`。生产模型
必须保留 observed range 和独立 validation metric，并在 strict mode 拒绝 unsupported
extrapolation。

## Memory Profile

Memory profile与task profile分离，仍使用`schema_version: 1`。`samples`按独立native serving
workload记录`weight_materialize`、`kv_materialize`和`runtime_reserved_memory`；同一路径只提供一个
`workload_id`。Duration model示例：

```json
{
  "operation": "kv_materialize",
  "estimator": "exact_median_or_linear_interpolation_within_measured_bytes",
  "extrapolation": "reject",
  "points": [
    {"bytes": 19209912320, "duration_us": 32632000, "workload_count": 3}
  ]
}
```

`weight_materialize`的bytes必须来自目标 runtime tensor exact sum；`weights (estimated)`日志不能
作为exact feature。KV bytes由exact page count和target runtime page shape计算。每个可用bytes point都必须
达到最少独立workload数，不能合并相邻page count冒充重复样本。

`footprint_model`记录native exact `total_bytes`、`free_bytes`、driver-visible `peak_non_kv`和simpler
committed，以及两者之差的reserved bytes。`a2a3sim`和native device-delay首次sizing都记录自身
non-KV committed baseline，将后续committed增量作为逻辑KV allocation加入native `peak_non_kv`。
Native device-delay还取该logical used与实时physical driver used的最大值：虚拟weight/KV不会放大
capacity，外部进程占用仍会收紧capacity。这样free-memory、allocation和lifetime保持一致。
只有duration models和footprint model全部strict-ready时，顶层`strict_ready`才为true。运行时只允许
exact point或测量域内线性插值，任何外推立即失败；exact footprint也不允许环境变量覆盖capacity。

## Simulation Plan

Offline plan 把每个 graph task 分类为：

- `simulate`：替换执行并调度预测 duration；
- `preserve`：保留依赖/控制语义，不做设备替换；
- `missing_model`：必须模拟但无延迟条目。

Plan 保留 task decision 用于审计，但 CPU-sim/native-delay 的实际执行 surface 是每个
`program + func_id` 的 child callable。共享该 child 的task site和AIC/AIV subslot prediction必须
显式聚合。跨生命周期feature复用时，可显式选择首选`capture_role`；plan必须记录输入候选、实际
选择和排除的role。Device-delay可携带`block_duration_us_by_subslot`，缺失或维度冲突时回退task/
subslot duration。Control semantic冲突会写入coverage，并令plan不能runtime-stub-ready。

Control output 使用 adapter 定义的 provider/directive。Runtime adapter 必须明确有效字段和 ABI padding；控制流常量只用于维持 serving 生命周期，不具备数值正确性含义。

### Timewarp Plan

Timewarp plan从已通过`runtime_stub_ready`的`backend=a2a3sim` sleep plan派生，保持原
`kernel_models`、task decision、control output和source manifest不变，只把`execution_mode`设为
`timewarp`并增加：

```json
{
  "timewarp": {
    "clock": "CLOCK_MONOTONIC+non_decreasing_global_offset",
    "sleep_oracle_required": true,
    "graph_models": [{
      "program": "decode_fwd",
      "duration_us": 29179.79,
      "task_count": 11117,
      "modeled_task_count": 10636,
      "modeled_block_count": 18783,
      "aic_clusters": 20,
      "scheduler": "deterministic_task_dag_resource_list_v1"
    }]
  }
}
```

每个runtime program必须恰有一个graph model。`duration_us`来自source profile DAG、plan中的
callable duration和block-rank vector，不是E2E拟合值。Capture role歧义、独立DAG不一致、缺失
predecessor、cycle、func model缺失或block维度冲突均拒绝生成。Timewarp plan不能交给sleep或
native-delay launcher；timewarp launcher也拒绝sleep plan。

## Coverage 契约

`strict_ready=true` 至少要求：无 missing model、无非法外推、满足可选 confidence 门禁、无
callable control conflict。`runtime_stub_ready=true` 还要求无 runtime-unmapped task，且所有
control output 都有可执行 ABI 映射。运行时 telemetry 必须证明至少一个 modeled stub 执行，
并且不存在 unmodeled execution。
