# Free 时间成因与 host 下发分析

本文件从 Ascend profile 判定单卡或某 rank 的 Device Free 时间偏高的成因；落到 host 下发（Host Bound）时进一步拆 Free、查慢 API、根因归类并给候选。只产候选不执行，落地由 `vllm-ascend-profiling` 调优迭代编排。

## 1. 判 Free 成因：排查表 + Host Bound 确认

Free 偏高不一定是 host 下发问题。先按下表排查：请求稀疏、异常频率命中的直接结案，不出 host 下发候选；800MHz 空闲自动降频是 Free 偏长的结果而非独立成因，不结案。都排除后按三条件确认 host 下发。

数据准备（本文件所有 msprof-analyze 模式共用的命令与产物链路）：

```bash
msprof-analyze -m <mode> -d <cluster_or_profile_parent> -o <output_dir> --force --agent
```

- 路径限制按子命令区分：`-m` 集群分析传 cluster 数据路径或单卡父目录，不传单卡目录；advisor 子命令单卡传 `*_ascend_pt` 目录、多卡传父目录；
- 产物在 `<output_dir>/cluster_analysis_output/`：`cluster_analysis.db`（各模式表）及模式子目录（free_analysis 的 CSV 在 `FreeAnalysis/free_analysis.csv`）；
- 查询前先 `sqlite3 <db> ".schema <表名>"` 确认表名；
- 全 rank 耗时拆解可用 `-m cluster_time_summary`（`ClusterTimeSummary` 表，单位 us；不可用时以各 rank `step_trace_time.csv` 为准）。可用时 `free` 是逐 rank Free 基线；`taskLaunchDelayAvgTime`（host 侧 api 开始到 device 侧 task 开始的平均耗时）跨 rank 对比可量化下发慢的 rank。

| 成因 | 多卡判据 | 单卡判据 | 验证 | 出口 |
|---|---|---|---|---|
| 请求稀疏 | 全组 Free 同步偏高、形态一致，与请求到达节奏对得上 | 本卡 Free 集中在请求间隙，与负载到达节奏对齐 | 各 rank Free 形态对比（`step_trace_time.csv`，`ClusterTimeSummary.free` 可用时同）+ `-m free_analysis` 的 Reason 分布（成因/次数/总时长/占比）；单卡看 Free 出现时刻与请求时序 | 非问题，记录后结案 |
| 空闲自动降频 | Free 偏长时设备自动降到 800MHz | 同左 | `-m freq_analysis`，查 db：`FreeFrequencyRanks`（rankId/aicoreFrequency [800,1800]） | 不结案：降频是 Free 长的结果，Free 的成因仍按其余行判 |
| 异常频率 | 出现 1800/800 以外的频率 | 同左 | `-m freq_analysis`，查 db：`AbnormalFrequencyRanks`（异常频率列表，如 [800,1150,1450,1800]） | 硬件/频率，移交，不出调优候选 |

请求稀疏、异常频率都排除 → 按三条件确认 host 下发（Host Bound，同时成立）：

1. Free% > 10%（profile 窗口占比），或远大于同组 rank baseline；
2. Free 区间内不含计算任务也不含通信任务；
3. dispatch 或 host 侧 gap 与 Free 区间对齐。

出口（对应 §5 的结论枚举）：

- 三条件同时成立 → Host Bound，进 §2；
- 不成立或数据缺失无法验证 → 非 Host Bound，不进 §2：API 慢但 device 已有队列工作记 device-bound queueing，其余记 `insufficient-evidence`。

仅凭 API 延迟高不能判 Host Bound。

## 2. 拆 Free：逐条区间判读 + dispatch 几何 + 算子下发分布统计

free_analysis 直接输出逐条 Free 区间，“拆”即逐条判读。`FreeAnalysis/free_analysis.csv` 每行一段：Rank ID/Start Time/End Time/Duration/Pytorch Idle Time/Cann Idle Time/Reason（按 Reason 聚合成因/次数/总时长/占比）。

判读动作（对 Free 偏高的 rank）：

1. 取大 Duration 区间；
2. 区间位置：落在请求间隙（对照请求到达时序）→ 回 §1 请求稀疏；落在请求处理中 → host 下发断点候选，进 dispatch 几何；
3. Reason 分布辅助归类。

dispatch 几何：`call ops → kernel launch → kernel run`。PYTORCH_API 是框架/算子调用入队，CANN 层 `acl_thread` 异步出队并把 launch 落到 device。

实现：对选定的 Free 区间时间窗，从 rank 级 `ascend_pytorch_profiler_{rank_id}.db` 取三层时刻对齐——PYTORCH_API 调用时刻、CANN_API launch 时刻、device task 起始时刻——看 gap 卡在哪层。判读：

- dispatch-to-run 近垂直（launch 后 task 很快起）且对上 Free → Host Bound；
- launch 后 device 延迟启动但期间有别的 device 活 → 健康的 device 侧排队，非 Host Bound；
- 关注斜率突然变垂直的段——host 下发藏不住的断点多在此。

数据前提：三层对齐需要逐次时间戳明细（rank 级 db 的 API 表）。profiler level 或 with_stack 不足时只剩聚合数据（`api_statistic.csv`），几何对齐做不了，改用聚合对比并在结论里记 `insufficient-evidence`。

算子下发分布统计（多卡时在断点定位后补全量视角——不看单个断点，统计 main 下发线程全部 enqueue 事件的落点）：

```bash
python scripts/dispatch_distribution.py <profile 父目录或单个 *_ascend_pt 目录> [--main-tids <tid,...>]
```

脚本读各 rank 的 `trace_view.json`，对每个 rank 统计 main 下发线程的 enqueue 事件（`cat=='enqueue'`；main tid 取该类事件最多的线程，输出 top3 供核对，`--main-tids` 显式指定）落在 device `Communication` / `Free` / `Computing` 区间内和区间外的次数与时长占比。单卡无对照 rank，只看落点结构。

判读（跨 rank 对比）：

- 先比每 rank 的 enqueue 次数与总时长：跨 rank 一致 → main 下发量不慢，差异在时机；某 rank 明显更长 → 回 §3 查该 rank 的慢 API；
- enqueue 大比例落在 Communication → 预取：通信等待窗口里提前下发下一层（先到的 rank 才有这个窗口），device 不空转；
- enqueue 大比例落在 Free → 锁步：没有等待窗口，device 空了才现场下发，下发耗时暴露成 Free；
- 次数/总时长一致 + 落点相反 → 两卡 host 下发时机错位（一张预取、一张锁步），锁步侧的 Free 即其下发暴露，“main 下发慢”不成立（根因归类见 §4）。

## 3. 查异常慢 API

数据来源：rank 级 `ascend_pytorch_profiler_{rank_id}.db` 的 `PYTORCH_API` / `CANN_API` 明细表；`depth` 字段可用时优先 `depth=1`（`depth=0` 常是外层包装，太粗），是否可用取决于采集配置。profiler level 不够、只有 `api_statistic.csv` 聚合时，退化为同名 API 的 Avg/Max/Variance 对比（无逐次时间戳），时间戳对齐部分记 `insufficient-evidence`。

范围：选定 Free 区间时间窗内的 API。同类相比：`aten::` 比 `aten::`、`npu::` 比 `npu::`、`launch` 比 `launch`、`Enqueue@`/`Dequeue@` 比同队列操作，看 count/total/avg/p95/p99/max，标出 max 或高分位远超同 rank 其他时间窗、或集群短 Free rank 的项，再把时间戳与 Free 区间和 dispatch 几何对齐。

判读：

- `launch` 时间含在 `Dequeue@` 内，异常 launch 归到出队侧 launch 路径，不重复累加；
- 长 `Dequeue@` → 二级管线 `acl_thread` 被干扰或延迟；
- 长 `Enqueue@` → `task_queue` 满、框架分发线程在等出队消费，或分发线程自身被扰；
- 长 `aten::`/`npu::` → 框架分发线程被干扰，看频率和间隔形态是否对上 Free；
- API 慢但 device 已有队列活 → 优化机会，非 Host Bound 证据。

## 4. 根因分类与候选

工具化先行：`msprof-analyze advisor schedule -d <ascend_pt 目录或集群父目录> --agent` 只覆盖部分类别——Operator Dispatch Issues（算子下发问题）、Fusible Operator Analysis（Host/MTE 瓶颈算子序列）对应模型侧下发压力，Synchronize Stream Issues（流同步检测）对应同步打断，GC Analysis（垃圾回收事件检测）对应 GC/锁/allocator；CPU-only、调度干扰与 host 下发时机错位三类无自动检测，手工判。

六类根因可同时命中，每类自带候选/方向：

- **模型侧下发压力**——找什么：高频小算子、launch 开销与 device 运行同量级、可融合的 Cast/TransData/Transpose 序列、低占用的零碎 kernel。候选：
  - 融合/减少 launch 次数：转本 skill 的算子融合分支（SKILL.md 2.5）；
  - 图模式（转 SKILL.md 2.1）：capture/replay 消除逐算子下发，优先于 `TASK_QUEUE_ENABLE=2`（后者限 eager）；
  - `TASK_QUEUE_ENABLE=2`：`ASCEND_LAUNCH_BLOCKING=1` 时失效；Level2 增 NPU 显存峰值；eager 模式前提；须先有下发/aclnn 在关键路径的证据，非通用解药；
  - `ACLNN_CACHE_LIMIT=100000`（默认 10000，范围 [1,10000000]）：动态 shape 一级下发瓶颈时；增 Host 内存；
- **同步打断**——找什么：间隙内有 `aclrtSynchronizeStream`、`aten::item`、`.cpu()`、标量提取、数据依赖控制流。方向（代码侧）：确认同步是否语义必需，移出热路径、batch 或值留 device。
- **CPU-only**——找什么：分发线程长时间做 Python 活不下发（预处理、日志、指标、shape/控制逻辑、host 张量）。方向（代码侧）：减少热路径 Python 工作、预计算或缓存 shape/控制结果、CPU 准备与 device 执行重叠。
- **调度干扰**——找什么：分发线程被抢占、跨核/跨 NUMA 迁移、无记录空泡；`acl_thread` 一并查。判据边界：空泡在 profile 内无法区分被抢占（R）与睡眠等队列（S）——S 是队列空、睡在 eventfd 等 main 投递，属下发节奏不是调度，须 OS 级证据（sched_switch 的 prev_state）确认 R 才动绑核。候选：vLLM 侧 `enable_cpu_binding`（`additional_config`，默认 True 开启，vllm-ascend 的 `cpu_binding.py` 绑定 `acl_thread`/`release_thread`；关闭用 `--additional-config '{"enable_cpu_binding": false}'`）。
- **GC/锁/allocator**——找什么：周期性暂停、enqueue/dequeue 锁、内存分配抖动。候选：tcmalloc（`LD_PRELOAD="/usr/local/lib/libtcmalloc.so" ...`）：仅 host 分配开销或锁争用可见时；不要仅因 Host Bound 就换 allocator。
- **host 下发时机错位（预取/锁步）**——找什么：§2 算子下发分布统计显示 enqueue 次数/总时长跨 rank 一致、落点相反——一侧大比例落 Communication（预取），一侧大比例落 Free（锁步），锁步侧 Free 即其下发暴露。机制是自反馈环：先到 → 等待窗口预填 → device 不空 → 又先到；后到 → 无等待窗口 → 实时下发 → device 空 → 又后到，方向一旦确定被锁死。种子（最初谁先到）profile 内不可判：框架 collective root 偏置或启动抖动，验证方法是多复测看方向固定还是随机。方向：缩短每层下发耗时等比缩短锁步侧 Free 暴露（锁步侧 Free 总量 ≈ 每层下发耗时 × 层数），引用模型侧下发压力类的候选。

各候选改后须对比同 rank/时间窗的耗时、Free 占比和 dispatch 几何，单变量复测由 `vllm-ascend-profiling` 调优迭代编排。

## 5. 输出

1. 结论：Host Bound / device-bound queueing / `insufficient-evidence`，附一句话依据；
2. 证据：rank/时间窗、Free 占比、关键区间、API/task 名、dispatch 几何、下发分布（次数/总时长/落点占比）；
3. 根因类别（§4）附判据；
4. 候选（§4，仅证据支撑的）；
5. 缺什么证据，具体到补什么（如 prev_state 区分 R/S、复测看错位方向）。
