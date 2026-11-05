# 通信均衡性分析

本文件是 SKILL.md 通信分析分支的执行模块，回答两个问题：**多卡通信是否均衡**；**不均衡的根因是什么**。

分析流程即章节号，粒度逐级下钻（整卡 → 单 op → 链路），每一步都可能提前结案：

1. **输入资格判断**：只有完整多卡 profile 才做均衡分析；
2. **通信是否值得分析**：占比和 rank 差异都不明显就直接结束；
3. **热点定位**：把差异落到具体通信组、collective 和 rank/step/op；
4. **根因分流**：先判等待（`wait-caused`），再查真实传输（`real-transfer-slow-suspected`），计算量或通信量不对称定性为负载型。

判不出的一律记 `insufficient-evidence`，只产出通信证据和根因判断，必要时可以分析当前环境下 vllm 和 vllm-aascend 源码。

## 1. 输入资格判断

| 输入 | 资格判定 |
|---|---|
| 完整多卡（全部 rank 的 `*_ascend_pt/`，或已有 `cluster_analysis.db`） | **通过**：按下文生成/确认分析 db，进入 §2 |
| 单卡部署（world_size=1，用 `profiler_metadata.json` 或启动命令确认） | **不做均衡判断**：没有跨卡通信，profile 里的通信 op 是 LOCAL 自环（launch/同步开销），无跨卡分析价值。用户问的是该卡通信开销或故障时，走 `msprof-analyze advisor all -d <rank_ascend_pt> --agent`，不属于均衡性分析 |
| 多卡部署但 rank 不全 | **`insufficient-evidence`**：缺跨 rank 对比，停止并要求补齐其余 rank 的 profile。单 rank 的 wait 占比高只作方向性线索，不得基于单 rank 硬套 §2-§4 |

资格通过后的数据准备：

```bash
msprof-analyze -m all -d <cluster_or_profile_parent> -o <output_dir> --force --agent
```

- `-d` 指到包含全部 rank 目录的父目录。
- 返回 error 或 JSON message 为空：停止，提示用户去掉 `--agent` 自行执行看日志；命令失败后不要退回逐个 rank 读数据。
- 查询前先 `sqlite3 <db> ".schema <表名>"` 确认实际列名；缺必需表时该项记 `insufficient-evidence`，不中止整个分析。

DB 优先级：`cluster_analysis.db`（集群诊断首选）> rank 级 `analysis.db`（单卡诊断）> rank 级 `ascend_pytorch_profiler_{rank_id}.db`（通信小任务/retry/relay 证据）> `profiler_info.json`（判定 level）。

核心表：

| 表 | 关键字段 | 用途 |
|---|---|---|
| `CommunicationGroupMapping` | `group_name`（组哈希）、`group_id`、`pg_name`、`type`、`rank_set` | 把组哈希映射到真实通信域，解读 group 前必须先查 |
| `ClusterCommunicationTime` | `hccl_op_name`、`group_name`、`rank_id`、`step`、`start_timestamp`、`elapsed_time`、`wait_time`、`transit_time` | 跨 rank 时序，等待判定主表 |
| `ClusterCommunicationMatrix` | `hccl_op_name`、`src_rank`、`dst_rank`、`transport_type`、`transit_size`、`transit_time`、`bandwidth` | 链路路由、带宽、慢链路定位 |
| `ClusterCommunicationBandwidth` | op、rank、链路类型、`transit_size`、`bandwidth` | per-rank 数据量和带宽细节 |

profiler level（从 `profiler_info.json` 判定）：**level1+** 可用小任务表 `COMMUNICATION_TASK_INFO` 的 Notify Wait、size、链路、src/dst 字段；**level0** 下这些字段可能是 DB 默认值，一律视为未知，不得据此推断对端方向或链路类型。最终报告必须注明 level。

rank 口径：表中 rank 可能是通信域本地 rank，报告须写明 global 还是本地，用 `rank_set` 换算，换算不可靠时只报本地 rank 并注明。

单位纪律：时间/数据量字段先确认单位再换算（`elapsed_time`/`wait_time`/`transit_time` 通常为 ms，`start_timestamp` 为 µs，`transit_size` 可对照 `communication_matrix.json` 的 MB 口径），报告统一用 ms 并标注。

## 2. 通信是否值得分析

整卡粗筛：通信整体占比高不高、各 rank 通信总时间有没有明显差异。两者都没有，后面的分析不用做。

逐 rank 读 `step_trace_time.csv` 记录 Computing/Communication/Free/Overlapped（占比分母用 Stage，缺失或为 0 时用 Computing + Communication + Free）。有集群 db 时补充 wait 构成：

```bash
sqlite3 -header <db> "
SELECT rank_id, COUNT(*) AS op_rows,
       SUM(elapsed_time) AS total_ms, SUM(wait_time) AS wait_ms,
       SUM(transit_time) AS transit_ms,
       ROUND(SUM(wait_time)*1.0/SUM(elapsed_time), 3) AS wait_ratio
FROM ClusterCommunicationTime
WHERE hccl_op_name != 'Total Op Info'
GROUP BY rank_id ORDER BY total_ms DESC;"
```

判定（不用固定阈值，报告原始值和差异率 `(max-min)/min`）：

- **各 rank 总时间接近且 Communication% 不高** → 通信不是当前问题，结束；
- **均衡但 Communication% 偏高** → 属整体开销而非均衡性问题，结束并附一句方向（小包/调用模式/串行；小包判定见 §4.3）；
- **存在 rank 差异** → 进入 §3，记下离群 rank 名单和各 rank 的 Free/Computing 基线（4.2 定性要用）。

## 3. 热点定位

把 §2 的差异落到具体位置：哪个通信组、哪个 collective、哪些 rank/step/op。

```bash
sqlite3 -header <db> "
SELECT t.hccl_op_name,
       COALESCE(g.pg_name, '') AS pg_name, COALESCE(g.type, '') AS group_type,
       COUNT(DISTINCT t.rank_id) AS rank_count,
       SUM(t.elapsed_time) AS total_ms, MAX(t.elapsed_time) AS max_ms,
       MIN(t.elapsed_time) AS min_ms, SUM(t.wait_time) AS total_wait_ms
FROM ClusterCommunicationTime t
LEFT JOIN CommunicationGroupMapping g ON COALESCE(t.group_name, '') = g.group_name
WHERE t.hccl_op_name != 'Total Op Info'
GROUP BY t.hccl_op_name, t.group_name, g.pg_name, g.type, g.rank_set
ORDER BY total_ms DESC LIMIT 20;"
```

判定：`max_ms/min_ms` 倍数大的 op 是跨 rank 不均衡候选，`Total Op Info` 等聚合行必须排除。

§2 的离群 rank 在候选中不明显时，按该 rank 过滤复查（该 rank 的 top op 对比同 op 其他 rank）：该 rank 的 op 数量或通信总量本身不同 → 4.4 负载不均；单 op 差异都不明显 → 记 `insufficient-evidence`。

**选定深挖 op**：从候选中取 2~5 个具体 `hccl_op_name`（可限定 step），优先总耗时高、`max_ms/min_ms` 倍数大、wait 占比高的，报告写明选择理由 → §4。

## 4. 根因分流：先判等待，再查传输

对 §3 选定的每个 op 依次过一道是非题：**该 op 耗时长，是不是在等别的 rank？**等待成因未排除前，不得对该 op 下带宽、链路、retry 结论。

### 4.1 等待判定

**等待者/迟到者反转**：集合通信要同组 rank 到齐。先到的在 op 内部等待，通信耗时长；最后到的 rank 一进来就完成，通信耗时短。因此**耗时最长的 rank 通常是等待者，开始最晚或耗时最短的 rank 才是迟到者（问题 rank）**。禁止把"通信时间最长的 rank"直接当成问题 rank。

跨 rank 对齐（对每个选定 op）：

```bash
sqlite3 -header <db> "
SELECT step, rank_id, start_timestamp, elapsed_time, wait_time, transit_time
FROM ClusterCommunicationTime
WHERE hccl_op_name = '<selected_op>'
ORDER BY step, start_timestamp;"
```

对每个 (step, op) 算：最早/最晚开始 rank、耗时最长/最短 rank、`start_span`（最晚开始 − 最早开始）、`end_span`（最晚结束 − 最早结束，结束 = start + elapsed，先统一单位）：

| 特征 | 判定 |
|---|---|
| `start_span` 大、`end_span` ≤ max(1 ms, 10% × `start_span`)、最早开始 rank == 耗时最长 rank | **强等待特征**：早到的在等，全组几乎同时结束 |
| `start_span` > `end_span` 且耗时偏斜明显 | 弱等待特征 |
| `end_span` 与 `start_span` 同量级 | 时序定不了等待，需 Notify Wait 或更多证据 |

`end_span` 不稳定或无法对齐到同一通信实例时，禁止仅凭 start/耗时偏斜下等待结论。

**Notify Wait 佐证（level1+）**：在 rank 级 `ascend_pytorch_profiler_{rank_id}.db` 的 `COMMUNICATION_TASK_INFO` 查选定 op 的大 `Notify Wait`；其 src/peer 字段可指认被等 rank（level0 下是默认值，不得使用）。

**判定出口**：

- 等待成立（强/弱特征，Notify Wait 可佐证）→ 结论 `wait-caused`，进 4.2 定性迟到 rank；
- 等待不成立（`end_span` 与 `start_span` 同量级，或迟到 rank 自身 transit 明显更高）→ 进 4.3 传输分析；
- op 只有聚合耗时无法对齐、或特征不明确且无 Notify Wait → 该 op 记 `insufficient-evidence`。

### 4.2 等待成立：迟到 rank 定性

对最晚开始或耗时最短的 rank，用 §2 记下的 Free/Computing 基线对表定性（判据和验证手段都在本节）：

| 迟到 rank 特征 | 定性 | 验证 |
|---|---|---|
| `Free` 极高（远超均值）且 Compute/Comm 偏短 | **伪快卡**：host 下发慢饿死 device，它是让全组等它的慢卡 | `free_analysis`；推理 profile 先排除请求间隔 Free（全组 Free 同步偏高、形态一致属请求间隔；仅该 rank 远超同组才成立），排除前不得定性；成因判定与 host 下发候选见 [free-analysis.md](free-analysis.md) |
| `Computing` 显著高于同组均值，Free 均匀 | 纯计算慢卡 | `cluster_time_summary` |
| 频率 800MHz 或非 1800/800 | 频率异常 | `freq_analysis` |
| 计算量/通信量本身更大 | 负载型 | 按 4.4 取证定性 |

`slow_rank` 模式可交叉验证慢卡结论。伪快卡/计算慢卡/频率的深入优化不在通信分支展开。

**伪快卡定性前置**：对比迟到 rank 与同组 rank 的**计算算子**种类、调用次数、shape（各 rank `op_statistic.csv`，计算算子而非通信算子）。算子种类不一致 → 根因记为计算算子不一致；种类一致但调用次数/shape 明显更大 → 计算量更大，按 4.4 定性；均一致而 Free 仍高 → 伪快卡成立。

**停止规则**：判为 `wait-caused` 的 op 输出等待链（谁等谁、start/end 数值、迟到 rank 定性）后停止，不得再给该 op 下 retry、relay、带宽或链路结论。

### 4.3 等待不成立：真实传输分析

传输证据都来自**通信算子**（hccl op）在各链路上的传输表现，分两层：链路总览（确认慢的是链路——途经的多条 op 都慢——还是只有个别 op 慢）和选定 op 的 rank 对明细。

```bash
sqlite3 -header <db> "
SELECT COALESCE(g.pg_name, '') AS pg_name, m.transport_type,
       COUNT(DISTINCT CAST(m.src_rank AS TEXT) || '->' || CAST(m.dst_rank AS TEXT)) AS rank_pair_count,
       SUM(m.transit_size) AS total_size, AVG(m.bandwidth) AS avg_bw,
       MIN(m.bandwidth) AS min_bw, MAX(m.bandwidth) AS max_bw
FROM ClusterCommunicationMatrix m
LEFT JOIN CommunicationGroupMapping g ON m.group_name = g.group_name
WHERE m.hccl_op_name != 'Total Op Info'
GROUP BY m.group_name, g.pg_name, m.transport_type
ORDER BY SUM(m.transit_time) DESC;"
```

```bash
sqlite3 -header <db> "
SELECT src_rank, dst_rank, transport_type,
       SUM(transit_size) AS total_size, SUM(transit_time) AS total_ms,
       AVG(bandwidth) AS avg_bw
FROM ClusterCommunicationMatrix
WHERE hccl_op_name = '<selected_op>'
GROUP BY src_rank, dst_rank, transport_type
ORDER BY avg_bw ASC;"
```

判定纪律：只在**同类型链路内**比较带宽（LOCAL > HCCS/PCIE > RDMA，跨类型禁止比较）；单独的 `MIN(bandwidth)` 低不能定故障，payload 小（SDMA < 16 MB / RDMA < 1 MB）也会拉低，命中即小包模式。再从 rank 级 `ascend_pytorch_profiler_{rank_id}.db` 的 `COMMUNICATION_OP` 对应行取 retry/relay 类指标（先 `.schema` 确认列名）。

| 证据组合 | 根因 | 注意 |
|---|---|---|
| retry 类指标非零 | 通信重传 | 有直接计数器；大算子常表现为 4 s 量级耗时突增 |
| relay 类指标非零 | 通信中继 | 优先用显式指标，不要仅凭带宽推断；矩阵路由可佐证 |
| payload 小（SDMA < 16 MB / RDMA < 1 MB）且次数多 | 小包/建链开销，通信模式问题 | 小包低带宽是预期行为，不定硬件故障 |
| 非小包、带宽持续低于同类 rank/链路 | 慢链路/拥塞候选 | 同类型内比较，用矩阵确认 src/dst 和 transport |
| 与 MatMul、FA 等 memory-heavy 计算重叠且带宽同步下降 | 计算-通信带宽争用 | 无无重叠基线时只作候选 |
| 非小包、带宽极低且 size 不能被 512 整除 | 地址对齐疑点 | 只报 suspicion，不得说已确认 |

带宽参考锚点（经验值非阈值）：SDMA memcpy > 16 MB 约 19 GB/s，SDMA inline-reduce > 16 MB 约 17 GB/s，RDMA > 1 MB 约 21 GB/s。小包、overlap、拓扑都会拉低表观带宽。

### 4.4 负载不均判定（计算量/通信量不对称）

来路：§3 复查发现该 rank 的 op 数量或通信总量不同、4.2 发现计算算子调用次数/shape 明显更大、或 4.3 传输证据正常但各 rank 通信耗时仍不对称。

- **计算量取证**：对比该 rank 与同组 rank 的计算算子调用次数和输入 shape（各 rank `op_statistic.csv`），明显更大 → 计算量不对称（如：EP token 偏斜等）；
- **通信量取证**：对选定 op 按 rank 对比 `ClusterCommunicationBandwidth` 的 `transit_size`，各 rank 差异大而带宽接近 → 通信量不对称。

任一不对称 → 负载型。allreduce/allgather 各 rank 传输量天然对称，负载型高发于 all-to-all（EP dispatch/combine）等路由相关通信；DP batch 不均同属负载型。op 级结论仍按 §4 的标签给出，根因字段标注负载不均。

## 5. 输出

- **总结论**：均衡 → 一句话加 §2 各 rank 占比/总时间/差异率、最快/最慢 rank；不均衡 → 点名哪些 op/链路、各挂什么标签。
- **逐热点结案表**（主体，每个选定 op/链路一行）：op/链路（`hccl_op_name`、通信组 pg_name、rank/链路对及选择理由，含 4.3 的慢链路 rank 对）、标签 · 定性/根因模式（标签三选一，定性/根因模式按下表取值，与 `SKILL.md` 2.3 映射表的行一一对应）、定案证据、缺什么证据（`insufficient-evidence` 时）。

| 标签 · 定性/根因模式 | 定案证据 |
|---|---|
| `wait-caused` · 伪快卡 | 谁等谁、start/end span 数值、迟到 rank 的 Free 形态和算子一致性对比 |
| `wait-caused` · 计算算子种类不一致 | 计算算子种类差异 |
| `wait-caused` · 纯计算慢卡 | Computing 显著高于同组均值 |
| `wait-caused` · 频率异常 | 频率值 |
| `real-transfer-slow-suspected` · 小包模式 | payload、bandwidth、调用次数 |
| `real-transfer-slow-suspected` · 慢链路/retry/relay/对齐 | bandwidth、transport、retry/relay 计数 |
| `real-transfer-slow-suspected` · 计算-通信带宽争用 | 与 memory-heavy 计算重叠且带宽同步下降 |
| 负载型 · 计算量/通信量不对称 | 计算算子次数/shape 对比、transit_size 对比 |
| `insufficient-evidence` | 缺什么证据 |
- **报告头口径**：profiler level、rank 口径（global/本地）、时间单位（ms）。

用语纪律：实测字段用事实措辞；`likely` 需至少两个证据源一致；`possible` 用于单源启发式；不够就写 `insufficient-evidence`，不猜。
