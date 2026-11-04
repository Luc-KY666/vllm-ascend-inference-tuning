---
name: vllm-ascend-profile-analyze
description: >-
  分析已采集的昇腾 NPU 推理 profile，提供两条路径：正常 profile 分析（目录完整性、计算/通信占比定性、
  Top-N 算子统计和基础通信摘要）以及调优相关分析（EP 负载、启动配置、通信、Free 成因/host 下发和算子融合候选）。不负责 profile
  采集、benchmark 编排、模型部署、参数修改或代码落地。
---

# vLLM-Ascend Profile 分析

本 skill 只读取已经生成的 profile，按照用户选择的路径输出可追溯的分析报告。它是
`vllm-ascend-profiling` 调优迭代使用的分析工具，不负责冻结 benchmark、执行变更或验证收益。

## 输入和数据边界

输入可以是单卡 `<run>_ascend_pt/` 目录，或包含多个 rank 目录的父目录。每个 profile 至少需要：

- `ASCEND_PROFILER_OUTPUT/step_trace_time.csv`
- `ASCEND_PROFILER_OUTPUT/op_statistic.csv`

`api_statistic.csv`、`communication_matrix.json`、`profiler_metadata.json` 和 `analysis.db` 按实际分析需要使用。
必需文件缺失、rank 不全或目录混入其他采集批次时，停止并要求补充数据；本 skill 不在缺失时启动采集。

## 开始时选择分析能力

收到用户请求后，先根据提示选择一个或多个分析能力，只执行与请求相关的分支：

| 用户提示/目标 | 选择的分析能力 | 主要输入 |
|---|---|---|
| “分析这个 profile”“看计算/通信占比”“找热点算子” | 路径 1：正常 Profile 分析 | profile 目录 |
| “EP 不均衡”“MoE 专家负载”“路由不均” | EP 负载和启动配置分析 | profile、`cluster_analysis.db`、实际启动命令 |
| “启动参数怎么调”“配置怎么改” | 配置建议分析 | profile、实际启动命令、版本 help |
| “通信瓶颈”“通信是否均衡”“快慢 rank 通信差异”“通信 overlap”“HCCS/带宽” | 通信分析 | step trace、集群分析 DB、通信矩阵、rank 数据 |
| “Free 偏高”“device 空转”“Host Bound”“下发慢” | Free 时间与 host 下发分析 | profile、msprof-analyze 产物、`trace_view.json` |
| “小算子融合”“融合算子”“结合源码优化” | Profile 驱动的算子融合分析 | profile、vLLM/vllm-ascend 源码、模型 `config.json` |

如果用户要求“完整调优分析”，先执行路径 1，再按 profile 证据执行相关调优分支。提示无法唯一确定分支时，先说明可选能力并要求用户选择；不要默认把所有分支都执行一遍。

## 两条分析路径

### 路径 1：正常 Profile 分析

适用于用户只要求“看看 profile”“分析计算/通信占比”“统计热点算子”或需要一份基础 profile 报告的场景。

#### 1. 定位 profile 目录并校验完整性

- 单卡：确认输入目录本身是 `<run>_ascend_pt/`。
- 多卡：列出父目录下全部 `*_ascend_pt/`，确认 rank 范围并分析全部 rank。
- 对每个 rank 检查 `step_trace_time.csv` 和 `op_statistic.csv`；记录 `Device_id`，但不要用它替代 global rank。
- 记录 profile 路径、rank/Device 映射、并行信息和数据文件缺失情况。

#### 2. 计算占比并定性，包含基础通信摘要

对每个 rank 使用 `step_trace_time.csv` 计算：

```text
denominator = Stage（缺失或为 0 时使用 Computing + Communication + Free）
Computing% = Computing / denominator
Communication% = Communication / denominator
Free% = Free / denominator
```

根据占比给出基础定性：

- `Computing%` 最高：compute-bound 倾向；
- `Communication%` 最高：comm-bound 倾向；
- `Free%` 最高或明显偏高：free-bound 倾向，并注明请求间隔可能造成 Free 偏高；深入成因判定转 2.4；
- 多项接近：标记为混合型，不强行选择单一瓶颈。

同时记录 `Overlapped`、`Bubble` 和多 rank 的 min/median/max。若存在 `communication_matrix.json`，只做基础摘要：是否有有效 HCCS 项、通信时间/带宽概况和是否存在明显 rank 差异；单卡 LOCAL 自环不能当作跨卡通信瓶颈。

#### 3. Top-N 算子统计

从 `op_statistic.csv` 输出每个 rank 的 Top-5（或用户指定 N），至少包含：

- `OP Type`、`Core Type`、`Count`；
- `Total Time`/`Ratio`；
- `Avg`、必要时 `Max`；
- 按 GEMM、Attention、Norm、MoE、采样等算子族的简单聚合。

高 Count + 低 Avg 只能标记为高频小算子候选，低 Count + 高 Avg/Max 只能标记为单 kernel 候选；Top-N 本身不等于最终瓶颈。

#### 正常分析输出

报告文件默认为 `profile_report.md`，包含：采集信息、逐 rank 计算/通信/Free 占比和定性、基础通信摘要、Top-N 算子表、数据缺失和无法判定项。正常路径不执行启动参数修改、EP 配置尝试或融合 patch。

### 路径 2：调优相关分析能力

适用于用户明确提出配置、通信、EP、算子融合、Free 偏高或 host 下发等调优问题，或要求基于 profile 给出调优方向的场景。本路径只产出候选和证据，不执行 benchmark 或变更。

#### 2.1 EP 负载分析与配置建议

EP 分支只按下面顺序执行，不把 profile 分析和配置推荐混在一起：

1. **分析 token 负载**：读取 [reference/ep-balance.md](reference/ep-balance.md)，执行 `msprof-analyze -m ep_load_balance`，从 `cluster_analysis.db` 的 `EPTokensSummary` 和 `TopEPTokensInfo` 得出每个 EP 组的 token 负载结论；
2. **交叉验证性能影响**：用同一次 profile 的 step_trace、op statistic 和 benchmark 指标判断 token 偏斜是否对应 rank Computing 或端到端性能恶化；
3. **决定是否给配置**：
   - 未发现 token 偏斜：报告 EP token 负载无明显异常，不给 EPLB 配置；
   - 有 token 偏斜但没有性能恶化：报告偏斜，转查通信、Free 或动态 Shape，不给 EPLB 配置；
   - token 偏斜且性能恶化：读取 [reference/ep-balance.md](reference/ep-balance.md) 第 4 节，先检查当前 vLLM-Ascend EPLB 源码和实际配置，再生成首次启用或单变量调整候选；
4. **输出复测条件**：记录当前命令、源码版本、profile 证据、唯一配置变更和固定 benchmark 复测要求。源码未确认支持的字段不得推荐。

#### 2.2 Profile 驱动的算子融合分析

当用户提示选择“算子融合”且 profile 显示高 Count 低 Avg 小算子、AI_CPU fallback 或 host dispatch 序列，并且 device 忙时段确实受其影响时，读取
[reference/operator-fusion.md](reference/operator-fusion.md)。先确认 profile 类型，再决定结论强度：

- **单请求 profile：用于候选定位。** 应只有一个正式请求，其他请求和并发压测停止，warmup 在 profile 外完成；profile 窗口应覆盖该请求从开始处理到完成的全过程，或明确记录实际覆盖的阶段。它用于还原一次请求的算子顺序、shape/dtype、host dispatch 和模块边界。
- **benchmark profile：用于聚合证据和收益验证。** 它可以用于统计热点算子的 Count、Avg、Total/Ratio 和并发场景影响，但多个请求可能交错，不能据此声称看到了某一个请求的完整调用链或连续算子序列。
- 如果只有 benchmark profile，先输出“聚合热点候选”和证据缺口；在提交融合 patch 或声称调用链已确认前，要求补采单请求 profile。单请求 profile 不能替代固定 benchmark 的最终收益验证。

算子融合不是只根据 profile 算子名给建议，必须联合分析：

1. profile 中的算子 Count、Avg、Total/Ratio、Core Type、host dispatch 和临时张量迹象；
2. vLLM 与 vllm-ascend 源码中模型 decoder layer `forward` 的真实调用链、模块边界和条件分支；
3. 模型 `config.json` 中的 hidden size、head/head_dim、rotary、top-k、专家数、dtype、TP/EP 等 shape 和并行约束。

`reference/operator-fusion.md` 已集中必要的融合模式、源码追踪、接口约束和报告要求，当前 skill 不需要运行时读取其他仓库的 skill。结合上述三类证据输出：

4. 原始源码路径、逻辑位置和调用片段；
5. 连续算子序列、输入 shape/dtype/layout 和 TP/EP 属性；
6. 已有融合接口或需要实现的接口；
7. 融合候选、预期收益、风险和回退方式。

没有源码、shape 或真实接口证据时，只输出候选设计，不声称已经完成替换。

#### 2.3 通信分析

该分支只负责通信问题的入口和结果路由，不分析 CPU 绑核、Host binding、通用 Host dispatch 或其他非通信调优问题。该分支回答多卡通信是否均衡、不均衡的根因（等待型/传输型/负载型）；具体的输入资格判断、数据准备、通信占比粗筛、热点定位、根因分流（等待判定、传输分类）和输出格式，读取 [reference/communication-analysis.md](reference/communication-analysis.md)。

输入为 profile 或分析数据库时，按 reference 的 DB 优先级和 profiler level 规则执行；输出必须区分 `wait-caused`、`real-transfer-slow-suspected` 和 `insufficient-evidence`，并引用具体通信组、op、rank 和表字段。结案表的标签按下面的映射转成调优建议：

| 根因标签 | 调优建议/移交 |
|---|---|
| `wait-caused`（伪快卡） | host 下发侧候选，取证与候选见 [reference/free-analysis.md](reference/free-analysis.md) §4（`TASK_QUEUE_ENABLE=2` 含 eager 前提与显存风险、`enable_cpu_binding` 须 R 证据、`ACLNN_CACHE_LIMIT`） |
| `wait-caused`（计算算子种类不一致） | 移交算子/EP 分支查种类分叉原因（动态 shape、EP 路由、条件分支）；区别于负载型——那是同算子、次数/shape 更大 |
| `wait-caused`（纯计算慢卡） | 计算瓶颈，转路径 1 的 Top-N 算子热点分析 |
| `wait-caused`（频率异常） | 硬件/频率检查，不属调优候选 |
| `real-transfer-slow-suspected`（小包模式） | 通信/调用模式候选 |
| `real-transfer-slow-suspected`（慢链路/retry/relay/对齐） | 链路排查/稳定性检查；对齐只报 suspicion |
| `real-transfer-slow-suspected`（计算-通信带宽争用） | 先补无重叠基线，再评估通信/计算重叠编排 |
| 负载型 | EP 分支 → [reference/ep-balance.md](reference/ep-balance.md) 第 4 节 EPLB 配置候选 |
| `insufficient-evidence` | 输出缺什么证据和补采要求 |

当前 skill 产出通信证据、根因判断和对应调优候选方向，不执行通信参数修改或 patch。

#### 2.4 Free 时间与 host 下发分析

该分支判定单卡或某 rank 的 Device Free 偏高的成因，落到 host 下发（Host Bound）时给候选；只产候选不执行。入口有三类：用户直接提出 Free/host 下发问题、路径 1 定性 free-bound 倾向后的深入判定、通信分析结案表 `wait-caused`（伪快卡）路由来的 Free 偏高 rank。

流程：先按排查表排除请求稀疏、异常频率（空闲自动降频是 Free 长的结果，不结案），再按三条件确认 Host Bound；不成立或无法验证时按 device-bound queueing / `insufficient-evidence` 收口，不出下发候选。成立后拆 Free（逐条区间判读、dispatch 几何、算子下发分布统计）、查异常慢 API、根因分类，只给证据支撑的候选。判读口径与候选明细读取 [reference/free-analysis.md](reference/free-analysis.md)，输出区分 Host Bound / device-bound queueing / `insufficient-evidence`，附根因类别、候选和证据缺口。

#### 调优分析输出

调优调用方应额外记录：当前启动命令和环境、分析命令、关键 profile 数值、候选变更、风险、验证条件和“需要固定 benchmark 复测”标记。配置尝试和 patch 落地由 `vllm-ascend-profiling` 的调优迭代编排。

## 参考文档

| 文档 | 用途 |
|---|---|
| [reference/analysis-guide.md](reference/analysis-guide.md) | 各数据文件的结构、字段含义与单位；按需读取 |
| [reference/communication-analysis.md](reference/communication-analysis.md) | 多卡通信均衡性判定与不均衡根因分析（等待型/传输型/负载型） |
| [reference/report-template.md](reference/report-template.md) | 正常 profile 报告模板 |
| [reference/ep-balance.md](reference/ep-balance.md) | 生成和读取 EP 数据库，得出 token 负载结论；偏斜确认后生成 EPLB 配置候选 |
| [reference/operator-fusion.md](reference/operator-fusion.md) | profile 热点到源码融合候选的分析规则 |
| [reference/free-analysis.md](reference/free-analysis.md) | Free 时间成因判定与 host 下发（Host Bound）分析，落到 host 下发时给候选 |
