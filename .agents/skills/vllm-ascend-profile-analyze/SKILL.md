---
name: inference-profile-analyze
description: >-
  分析已采集的昇腾 NPU 推理 profile（torch_npu.profiler / msprof 产出的 ASCEND_PROFILER_OUTPUT 目录），
  定位推理性能瓶颈。从 step_trace_time 定性瓶颈类型（compute-bound / comm-bound / free-bound），
  对 op_statistic 取 top 算子并按算子族聚合，从 api_statistic 归因 device Free，
  从 communication_matrix 判断通信是带宽瓶颈还是开销瓶颈，产出瓶颈判定报告与优化方向建议。
  触发：用户提供推理 profile 目录、要求分析推理性能瓶颈、解读 op_statistic / step_trace_time /
  communication_matrix、定位推理慢 / device 空闲高 / 通信占比高 / 算子热点时使用。
  不负责 profile 采集、模型部署、精度评测或算子级代码优化落地。
---

## Overview

针对**已采集**的昇腾 NPU 推理 profile 数据，按"能否定位瓶颈"的优先级逐层分析，产出一份可复核的瓶颈判定报告。核心产物是 **瓶颈类型判定 + Free 归因 + 优化方向建议**，而不是罗列 top 算子。

**前提条件**：用户已有一份或多份 profile，可能有两种形态：
- **单卡**：一个 `<run>_ascend_pt/` 目录，其下 `ASCEND_PROFILER_OUTPUT/` 含 `op_statistic.csv`、`step_trace_time.csv`、`api_statistic.csv`（`communication_matrix.json` 可能缺失，见数据文件约定）。
- **多卡**：一个**父目录**，其下有多个 `<run>_ascend_pt/` 子目录（通常按 rank 命名，如 `..._rank0_..._ascend_pt/`、`..._rank1_..._ascend_pt/`），每个子目录结构同单卡。

每个 `<run>_ascend_pt/` 至少含 `op_statistic.csv`、`step_trace_time.csv`、`api_statistic.csv`。**本 skill 只做分析与报告，不负责 profile 采集。** 若数据文件缺失，先让用户补采或补传，不在本 skill 内启动采集。

## 数据文件约定

profile 目录的标准结构（`<run>_ascend_pt/`）：

| 文件 | 作用 | 本 skill 是否必须 |
|------|------|-------------------|
| `ASCEND_PROFILER_OUTPUT/op_statistic.csv` | 算子级耗时统计（device 侧） | 必须 |
| `ASCEND_PROFILER_OUTPUT/step_trace_time.csv` | step 级计算/通信/空闲占比 | 必须 |
| `ASCEND_PROFILER_OUTPUT/api_statistic.csv` | host 侧 API 耗时 | 必须（Free 归因用） |
| `ASCEND_PROFILER_OUTPUT/communication_matrix.json` | 通信 op 带宽/耗时 | 条件使用（存在有效 HCCS 数据时） |
| `ASCEND_PROFILER_OUTPUT/analysis.db` | msprof 解析结果库 | 可选（深度下钻用） |
| `profiler_metadata.json` | 采集环境/并行组信息 | 辅助 |
| `PROF_*/host/`、`PROF_*/device_*/` | 原始 trace 切片 | 一般不直接读 |

字段含义、计算口径与误判规避见 [reference/analysis-guide.md](reference/analysis-guide.md)。

## Workflow

1. 定位 profile 目录并校验数据完整性
2. 算占比并定性（step_trace_time → Computing / Communication / Free 占比 + 分流）
3. Free 归因（区分"请求间隔 Free" vs "实现缺陷 Free"）
4. Top-N 算子与算子族聚合（op_statistic）
5. 通信分析（是不是瓶颈 / 开销型 vs 带宽型 / 多卡快慢 rank）
6. 输出瓶颈判定报告与优化方向

---

## Step 1: 定位 profile 目录并校验完整性

向用户确认 profile 的路径，先判断形态：

- **单卡**：用户给出的是 `<run>_ascend_pt/` 目录，内含 `ASCEND_PROFILER_OUTPUT/`。
- **多卡**：用户给出的是父目录，内含多个 `<run>_ascend_pt/` 子目录（按 rank 命名）。先列出子目录确认 rank 范围。

```bash
# 多卡：先列出所有 rank 子目录
ls -d "<父目录>"/*_ascend_pt 2>/dev/null

# 单卡时 PROF_DIRS=("<run>_ascend_pt")；多卡时收集父目录下全部 rank
PROF_DIRS=("<一个或多个 <run>_ascend_pt 目录>")
for PROF_DIR in "${PROF_DIRS[@]}"; do
  ls "$PROF_DIR/ASCEND_PROFILER_OUTPUT/op_statistic.csv" \
     "$PROF_DIR/ASCEND_PROFILER_OUTPUT/step_trace_time.csv" \
     "$PROF_DIR/ASCEND_PROFILER_OUTPUT/api_statistic.csv" 2>&1
done
```

任一必须文件缺失 → 让用户补传或重新采集，不继续。`communication_matrix.json` 的存在性与有效项判别见 Step 5b，缺失或仅 LOCAL 自环时不阻断分析。

**多卡必须分析全部 rank**：对每个 rank 完整执行 Step 2～4，先从目录名或 `profiler_info_*.json` 取得 global rank，再记录 `step_trace_time.csv` 中的 `Device_id`；报告同时给出逐 rank 结果和跨 rank 差异，主瓶颈判定以最差 rank、离群 rank 及其证据为依据，不能只看算术平均值。

读取 `profiler_metadata.json` 记录并行组信息（`parallel_group_info`）和环境变量（`PYTORCH_NPU_ALLOC_CONF`、`ASCEND_LAUNCH_BLOCKING`、`HCCL_ALGO` 等），写入报告的"采集信息"段。

---

## Step 2: 算占比并定性（step_trace_time）

逐 rank 按 [analysis-guide.md](reference/analysis-guide.md) 的占比口径计算 `Computing%`、`Communication%` 和 `Free%`，并记录 `Overlapped`、`Bubble`。多卡同时输出各指标的 `min / median / max` 和离群 rank。

**定性分流（按占比高低，非单一路由）**：

- `Free%` 最高或明显偏高（如 `Free% > 30%` 或 `Free% > 2×Computing%`）→ **主方向 Step 3**。
- `Computing%` 最高 → **主方向 Step 4**。
- `Communication%` 最高 → **主方向 Step 5**。
- 多项都高 → 混合，按各占比逐项分析。

分流只是"先看哪一步"，**不是只做那一步**：Compute-bound 仍要看 Step 5 的 overlap，Comm-bound 仍要看 Step 4 算子画像，Free-bound 仍要看 Step 4 描述 device 忙时画像。

> 注：占比只是"device 时间花在哪"的粗分，不是最终瓶颈判定。最终瓶颈类型（compute-bound / 开销型通信 / 带宽型通信 / 请求间隔 Free / 实现缺陷 Free）在 Step 3/4/5 完成后综合判定，写入 Step 6 报告。

---

## Step 3: Free 归因（推理 profile 的核心一步）

> 只要 `Free` 明显偏高就执行本步；`Free` 很低时跳过。

逐 rank 读取 `api_statistic.csv`（host 侧），按下面两类**分流**。分流结论直接决定优化方向。

### 3a. 请求间隔 Free（负载特征，非缺陷）

按 [analysis-guide.md](reference/analysis-guide.md) 的“请求间隔 Free 判据”检查。未排除请求间隔时，将其标为首要候选并给出确认方法，不进入实现缺陷归因。

### 3b. 实现缺陷 Free（device 真被饿死）

排除请求间隔后，按 [analysis-guide.md](reference/analysis-guide.md) 的 `api_statistic.csv` 归因口径分析，在报告中按 rank 列出 top3 成因，并比较同类 API 的耗时与调用次数。

---

## Step 4: Top-N 算子与算子族聚合（op_statistic）

逐 rank 按 [analysis-guide.md](reference/analysis-guide.md) 的 `op_statistic.csv` 口径输出 Top-5、`Count × Avg` 结论和算子族聚合。多卡保留 rank 维度，并标出同名算子的耗时差异。

---

## Step 5: 通信分析

先判断通信是否为瓶颈，再区分开销型与带宽型；多 rank 额外检查快慢 rank。

### 5a. 通信是不是瓶颈？——看占比与 overlap

结合 `Communication%`、`Overlapped` 和 `Bubble` 判断通信是否挤占计算。通信是瓶颈时进入 5b；否则跳过 5b。多 rank 场景均执行 5c。

### 5b. 是哪一类通信瓶颈？——开销型 vs 带宽型

按 [analysis-guide.md](reference/analysis-guide.md) 的 `communication_matrix.json` 口径分析。无有效 HCCS 数据时注明无法区分；有数据时输出开销型或带宽型结论及对应优化方向。

### 5c. 多卡快慢 rank 检查

按 [analysis-guide.md](reference/analysis-guide.md) 的多卡快慢 rank 判据检查。工作量不一致或只有单 rank 时注明无法比较。

---

## Step 6: 输出瓶颈判定报告

报告模板见 [reference/report-template.md](reference/report-template.md)。报告保存目录默认为**当前 profile 文件的同级目录**（即 `<run>_ascend_pt/` 所在目录；用户指定目录时使用用户指定的），文件名 `profile_report.md`。报告必须包含：

1. 采集信息（profile 目录、rank、并行组、环境变量）
2. 各 rank 三项占比 + 跨 rank 差异 + 定性（computing / comm / free 谁为主 + 最终瓶颈类型综合判定）
3. 各 rank Free 归因（**明确区分请求间隔 vs 实现缺陷**，列出 top3 host 成因）
4. 各 rank Top-5 算子 + 算子族聚合
5. 通信分析（是不是瓶颈 / 开销型 vs 带宽型 / 多卡快慢 rank 检查）
6. 优化方向建议（按预期收益排序，标注"需进一步验证" vs "可直接实施"）

**报告纪律**：所有结论必须可追溯到具体数据文件与数值，禁止凭印象下结论。Free-bound 且未排除请求间隔时，优化建议段必须先写"建议确认请求间隔影响"，再写其它方向。

## 参考文档

| 文档 | 说明 |
|------|------|
| [reference/analysis-guide.md](reference/analysis-guide.md) | 核心数据文件字段含义、计算口径、误判规避（含请求间隔 Free 与多卡快慢 rank 判据） |
| [reference/report-template.md](reference/report-template.md) | 瓶颈判定报告模板 |
