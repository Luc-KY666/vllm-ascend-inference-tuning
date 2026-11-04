# EP 负载分析

本文件从 Ascend profile 判断 EP rank 的 token 工作量是否偏斜；token 偏斜且性能恶化经上层确认后，再按第 4 节生成 EPLB 配置候选。本文件不直接判断性能瓶颈。

## 1. 生成分析数据库

对同一次 benchmark 的完整多卡 profile 执行：

```bash
msprof-analyze -m ep_load_balance \
  -d <cluster_or_profile_parent> \
  -o <output_dir> \
  --force --agent
```

成功后读取命令返回的：

```text
<output_dir>/cluster_analysis_output/cluster_analysis.db
```

先确认返回状态为 success、数据库存在，并且包含 EPTokensSummary 和 TopEPTokensInfo。没有有效数据库或缺少任一表时，结论为“无法判定”；不要把终端返回的状态 JSON 当成统计明细。

## 2. 读取两张表

```bash
sqlite3 <output_dir>/cluster_analysis_output/cluster_analysis.db \
  "SELECT rank, epRanks, inputShapesSummary FROM EPTokensSummary ORDER BY epRanks, rank;"

sqlite3 <output_dir>/cluster_analysis_output/cluster_analysis.db \
  "SELECT epRanks, tokensDiff FROM TopEPTokensInfo ORDER BY tokensDiff DESC;"
```

字段含义：

- EPTokensSummary.rank：global rank；
- EPTokensSummary.epRanks：该 EP 组的 rank 集合；
- EPTokensSummary.inputShapesSummary：GroupedMatmul 输入 shape 第一维之和，作为该 rank 的 token 工作量代理；
- TopEPTokensInfo.tokensDiff：同一 EP 组内代理值的最大值与最小值之差。

## 3. 得出 token 负载结论

按 epRanks 分组，对 inputShapesSummary 记录最大值、最小值、差值和相对差异（最小值大于 0 时可报告 max/min）。

- tokensDiff 有记录，或同组 rank 的代理值明显偏斜：结论为“存在 token 工作量偏斜”；
- 同组代理值接近且 TopEPTokensInfo 无记录：结论为“未观察到明显 token 偏斜”，不等于端到端性能一定均衡；
- rank 不全、EP 组映射不清、采集区间不一致或 EPTokensSummary 为空：结论为“无法判定”。TopEPTokensInfo 为空只表示工具没有筛出明显偏斜，仍需看 EPTokensSummary 的原始值。

该工具没有适用于所有模型和 workload 的固定百分比阈值，报告必须保留原始值和 EP 组映射，不能自行发明统一阈值。

## 4. 生成 EPLB 配置候选

前提：第 3 节结论为存在 token 偏斜，且上层已结合同一次 profile 的 step/op 数据和 benchmark 指标确认该偏斜对应性能恶化。

### 先确认当前 vLLM-Ascend 策略

定位实际使用的 vLLM-Ascend 源码、分支/commit 和版本，检查：

- vllm/config/parallel.py：EP 拓扑、EP size 和 eplb_config 默认值；
- vllm/engine/arg_utils.py：--additional-config 和环境变量入口；
- EPLB 实现：dynamic_eplb、expert_heat_collection_interval、algorithm_execution_interval、num_redundant_experts、eplb_policy_type 等字段的语义、取值范围和生效条件。

只有源码确认支持时，才能推荐 DYNAMIC_EPLB、--enable-expert-parallel 或具体 eplb_config 字段。不要默认套用未在当前源码中找到的环境变量。EP size 按源码和实际 TP/DP/EP 拓扑计算。

### 根据当前配置选择候选

- **当前没有 EPLB 配置**：生成“首次启用动态 EPLB”候选；
- **当前已经启用 EPLB**：保持 TP/DP/EP 拓扑，一次只调整一个 EPLB 字段或明确耦合字段组；
- **源码不支持或当前配置无法确认**：只报告缺失信息，不生成配置。

首次启用候选的形态如下，字段和值必须先用当前源码、模型专家数和显存约束复核：

```bash
export DYNAMIC_EPLB=true
vllm serve <model> --tensor-parallel-size <tp> --data-parallel-size <dp> --enable-expert-parallel --additional-config '{
  "eplb_config": {
    "dynamic_eplb": true,
    "expert_heat_collection_interval": 600,
    "algorithm_execution_interval": 50,
    "num_redundant_experts": <ep_size>,
    "eplb_policy_type": 2
  }
}'
```

600、50、ep_size 和策略类型只是当前源码分析对应的首个候选，不是所有版本的固定值。若实现、显存或模型约束不满足，停止或改用源码允许的值。
