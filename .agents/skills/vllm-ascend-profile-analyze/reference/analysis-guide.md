# 数据文件分析指南

本 skill 分析 `torch_npu.profiler` / `msprof` 产出的 `ASCEND_PROFILER_OUTPUT` 目录。本文给出核心文件的字段含义、计算口径和误判规避，重点是推理 profile 特有的“请求间隔 Free”和多卡快慢 rank 判据。

---

## step_trace_time.csv

### 字段

| 字段 | 含义 | 本 skill 用法 |
|------|------|---------------|
| `Device_id` | 当前进程使用的设备号 | 与 global rank 分开记录；多机或设备重映射时不能用它代替 rank |
| `Step` | step 标识 | **推理 profile 常为空且只有 1 行**（多 step 聚合成一行），此时无法从本文件看 Free 的时序位置 |
| `Computing` | 计算时间(us) | 计算 |
| `Communication(Not Overlapped)` | 未与计算重叠的通信时间 | 与 `Communication` 相等说明 `Overlapped=0` |
| `Overlapped` | 计算与通信重叠时间 | `=0` → 计算通信完全串行 |
| `Communication` | 通信总时间 | 计算 |
| `Free` | device 空闲时间(us) | **推理 profile 的判断核心** |
| `Stage` | 一个 step 的总跨度 | 占比分母首选 |
| `Bubble` | 流水气泡 | 判流水是否拉满 |
| `Communication(Not Overlapped and Exclude Receive)` | 排除接收的未重叠通信 | 深度分析用 |
| `Preparing` | 准备时间 | 一般很小 |

### 占比口径

- 首选分母 `Stage`。
- `Stage` 缺失/为 0 → 用 `Computing + Communication + Free`（少数情况下三者之和略小于 `Stage`，因 `Preparing` 等列未计入，取大者作分母即可）。
- 三项占比：`Computing% = Computing / 分母`，`Communication%`、`Free%` 同理。

### ⚠️ 请求间隔 Free 判据（推理 profile 专属陷阱）

推理 profile 通常是连续发多个推理请求采的，**两次请求之间的 host 调度间隔会让 device Free 爆表**。判定信号（满足任一即把"请求间隔"列为首要候选）：

1. 本文件只有 1 行且 `Step` 列为空 → Free 的时序位置不可见，**无法排除**请求间隔。
2. `Free%` 远高于 `Computing%`（如 Free > 2×Computing），且 `api_statistic` 中没有主导的 host 同步/下发开销。
3. `Free%` 高但 host 侧（`api_statistic`）工作量也低 → host 多数时间在等下一请求，device 自然空转。
4. `Overlapped=0` 是 Free 偏高的**协同证据**，但不是请求间隔的决定性证据——它说明计算通信零重叠，需在实现缺陷归因里单独处理。

**确认方法**（择一）：

- 用**单请求**重新采集（仅一次推理），看 Free 是否消失。
- 用**无间隔 / 固定高 QPS 连续请求**重新采集对照。
- 看 trace 时间线（`PROF_*/` 原始数据或 MindStudio）确认大段 Free 是否落在两次请求边界。

**纪律**：未排除请求间隔前，禁止把高 Free 直接结论为"host 下发慢、需优化"。

---

## op_statistic.csv

### 字段

| 字段 | 含义 |
|------|------|
| `Device_id` | 设备号 |
| `OP Type` | 算子类型名 |
| `Core Type` | 执行核类型：`AI_CORE` / `AI_VECTOR_CORE` / `MIX_AIC` / `MIX_AIV` / `AI_CPU` / `DSA_SQE` 等 |
| `Count` | 调用次数 |
| `Total Time(us)` | 总耗时（排序主键之一） |
| `Min/Avg/Max Time(us)` | 单次耗时区间 |
| `Ratio(%)` | 占总算子总耗时的百分比 |

### 分析口径

- **Top-N 用 `Ratio%` 或 `Total Time` 排序**，但必须同时列出 `Count` 与 `Avg`，不能只看 `Total`。
- **Count × Avg 拆解**决定优化路径：
  - 高 Count + 低 Avg → 高频小算子，优化 = **reduce dispatch**（融合 / 图模式 / 减少调用次数）。
  - 低 Count + 高 Avg/Max → 单 kernel 慢，优化 = **kernel/tiling 优化**（走算子调优 skill）。
- **按算子族聚合**（GEMM / Attention / MoE 路由 / Norm / 采样）比单 OP 更能对应模型结构。
- **Core Type 分布**：
  - `AI_CPU` → host fallback 串行执行，融合消除对象。
  - `MIX_AIC` / `MIX_AIV` → 已走融合核，无需再单独融合。
  - `AI_CORE` / `AI_VECTOR_CORE` → 常规核执行。
  - `DSA_SQE` → 调度核（如本例 `DSARandomUniform`），通常非热点。

### Top-5 不能直接当瓶颈

op_statistic 把所有 step 的算子混在一起，是"device 干活时谁吃时间多"。当 device 有大段 Free（Free-bound）时，top-5 优化得再快也吃不掉 Free——必须先做 Free 归因（Step 3）。top-5 真正的作用是在 **compute-bound** 或确认 device 在干活时段，给出算子级优化方向。

---

## api_statistic.csv

### 字段

| 字段 | 含义 |
|------|------|
| `Device_id` | `host` 或设备号；`host` 行是 host 侧 |
| `Level` | `acl` / `node` / `communication` 等 |
| `API Name` | host API 名 |
| `Time(us)` | 累计耗时（Free 归因的主键） |
| `Count` | 调用次数 |
| `Avg/Min/Max(us)` | 单次耗时 |
| `Variance` | 耗时方差（高方差 → 有离群慢调用） |

### Free 归因口径（实现缺陷类，排除请求间隔后用）

逐 rank 读取文件，先筛选 `Device_id=host`，再按 `Level` 和 `API Name` 汇总 `Time(us)`。`Level` 通常为 `acl`、`node`、`communication`，不能按 `Level=host` 过滤。重点看以下类别的累计耗时与占比：

| 成因类别 | 信号 API | 优化方向 |
|----------|----------|----------|
| host 下发饿死 | `host,node,launch` | reduce dispatch（融合/图模式/减调用次数） |
| 同步阻塞打断流水 | `aclrtSynchronizeStream`、`aclrtSynchronizeStreamWithTimeout`、`aclrtSynchronizeEvent` | 去冗余 sync、改异步、多流 overlap |
| 纯 CPU 开销 | `*GetWorkspaceSize` 系列 | 开 workspace 缓存（`ACLNN_CACHE_LIMIT`） |
| 数据搬运 | `aclrtMemcpyAsync`、`aclrtMemcpy` | 减少不必要 H2D/D2D、复用显存 |
| 反复建/销 | `aclrtCreateEventExWithFlag`、`aclrtDestroyEvent` | 复用 event/stream |

每类算占总 host 耗时比例，取 top3 写入报告。`Variance` 高的 API 提示有离群慢调用（如本例 `aclnnEqScalar` avg 110us 但 max 2119us），可单独标注为"偶发慢点"。

### 注意

- `*GetWorkspaceSize` 与对应执行 API 常成对出现，归因时把二者作为同一组看（如本例 `aclnnInnerFusedInferAttentionScoreGetWorkspaceSize=105374us` 几乎与算子本身 `102082us` 等量，是可消除的纯开销）。
- `aclrtSynchronizeStreamWithTimeout` 高 Count（如本例 3332 次）且总时长高 → 推理 step 级同步频繁，是 overlap 优化的目标。

---

## communication_matrix.json

### 存在性与有效项判别（先做）

该文件在**有通信 op 被采集到时**才生成，**不是"多卡才存在"**。先判断是否含 `HCCS` 项决定是否做带宽分析：

| 情况 | 信号 | 处理 |
|------|------|------|
| 文件不存在 | 无通信 op 被采集 | 5b 带宽分析跳过，仅看 step_trace 串统计 |
| 仅 `LOCAL` 项（单卡） | 索引 `0-0`、Transit Size 接近 0（如 0.008MB） | 卡内 allreduce 自环，**无分析价值**，5b 跳过 |
| 含 `HCCS` 项（多卡） | 索引形如 `1-0`、Transit Size 有量级 | 有跨卡链路，做带宽利用率分析 |

> ⚠️ 单卡 profile 的 `LOCAL` 自环项（`0-0`）不是真通信，禁止拿它的带宽做"通信慢/快"判断。

### 结构

顶层 `step` → `collective`（集合通信）/ `p2p`。每个通信 op 下按 `<rank>-<link>` 索引：

| 字段 | 含义 |
|------|------|
| `Transport Type` | `LOCAL`（卡内自环，不占链路）/ `HCCS`（卡间链路） |
| `Transit Size(MB)` | 传输数据量 |
| `Transit Time(ms)` | 传输时间 |
| `Bandwidth(GB/s)` | 实测带宽 |
| `Op Name` | hccl op 名 |

### 分析口径（仅对含 HCCS 项的情况）

- 取 `*-total` 聚合项（`allreduce-total` / `allgather-total` 等）看整体带宽。
- 把实测带宽与链路上限对比：HCCS 单 link ~39 GB/s；多 link 聚合更高。
- **远低于上限**（如本例 allreduce-total 5.1 GB/s、allgather HCCS 18.3 GB/s）→ 通信慢是 **launch/同步/小消息开销** 导致，**不是带宽瓶颈** → 优化方向 = 减少通信次数（通信融合），而非换更快链路。
- **接近上限** → 真带宽瓶颈，优化方向 = 减少通信数据量（梯度压缩 / 分组）。

### 多卡快慢 rank 检查

先确认各 rank 的采集区间和工作量一致，再比较 `Computing` 与 `Communication(Not Overlapped)`。差异率按 `(最大值 - 最小值) / 最小值 × 100%` 计算，报告最快、最慢的 rank 和 Device ID。本节只判断是否存在快慢 rank，不分析原因。
