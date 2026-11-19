# Native Qwen 验证状态

初始日期：2026-11-11
最新更新：2026-11-17

## 环境

- 模型：`/data/disk1/liangkaiyi/Qwen3-14B`
- 初始设备：physical NPU 4，直接使用 `--device 4`
- 最新 baseline 设备：physical NPU 6，直接使用 `--device 6`
- PyPTO：`fc50efe71ab1e6adbfd7703d3f6ac8b209d85c0b`
- simpler：`355ce3015e3e344d1fc3a61c206ee01823ce292c`
- pypto-lib：`ec58b8ad7454b321f379260e95a0ae4431a5ae57`
- PTOAS：`v0.72`

所有有效 native run 都没有使用 `ASCEND_RT_VISIBLE_DEVICES` 重映射。

## 健康设备对照

`pypto-lib/examples/beginner/hello_world.py -p a2a3 -d 4` 成功编译、运行，并在 5.10 秒内
通过数值验证。这证明 NPU 4 和基础 PyPTO/simpler onboard 路径当时正常。

## 初始 Qwen 失败与根因

未修改的 Qwen3-14B serving callable 无法完成 native warmup：

- `prefill_fwd` 在 request admission 前失败；
- 诊断性旁路 `prefill_fwd` 后，`decode_fwd` 仍失败；
- device strace 显示 `decode_fwd` 已 dispatch，随后 AICPU stream 报
  `ACL_ERROR_RT_AICPU_EXCEPTION`（`507018`），且没有 terminal record；
- 表面上的数分钟延迟来自异常后的 runtime force-reset，不是模型 latency。

`QWEN3_14B_20CORE_RUNBOOK.md` 最终定位到设备 occupancy：Bin6 只有 20 个 AIC cluster，而
pinned prefill/decode attention kernel 固定启动 24-block hard-sync cohort。Hard `syncall` 要求
整个 cohort 同时 resident，因此 runtime 报
`require_sync_start block_num=24 > limit=20`，host 侧表现为 `507018`。

扩大 watchdog 或 ring 无法修复 occupancy mismatch。已按 runbook 应用源码修复：

- 同步 attention launch 使用 `pl.system.available_cluster_count()`；
- kernel 用 `pl.tile.get_block_num()` 读取实际宽度，并执行 grid-stride loop；
- decode 中常量 24 只保留为 transfer scratch allocation 上界。

修复后 Qwen contract 为 `27 passed`。Native-A 随后在 NPU 4 使用以下配置成功：

- `max_model_len=512`、`max_num_seqs=16`；
- dependency/task ring 为 262144，ring heap 为 1 GiB；
- prompt 为 `Huawei is`，greedy 输出 5 token。

输出 token 为 `[264, 8453, 67926, 5440, 2813]`，与 runbook 一致。Warmup prefill 约
1.37 秒，warmup decode 约 0.04 秒，5-token request E2E 约 0.25 秒。

## 首轮 Native Profile 与 Sleep

后续 native run 对 `prefill_fwd`、`decode_fwd`、`topk_select_fwd` 启用无旁路 level-4
profiling，并产生相同 token。Clean capture 位于 `/tmp/pypto-qwen-profile-20core-A`，每个
program 都有配对的 `deps.json`、`chip_swimlane_records.json`、name map 和 dispatch identity。

Exporter 严格校验 20-AIC/40-AIV blocked core topology。MIX work 的 physical vector core 按
cluster 交错为 AIV0/AIV1；pure AIV work 使用 `deps.json` 中唯一 logical slot。因此可以保留
AIV0/AIV1 独立 timing，而不伪造 HBG execution identity。

| Program | Task 数 | 已测 compute | 保留 sync | 不完整行 |
| --- | ---: | ---: | ---: | ---: |
| `prefill_fwd` | 1124 | 1044 | 80 | 0 |
| `decode_fwd` | 11117 | 10636 | 481 | 0 |
| `topk_select_fwd` | 16 | 16 | 0 | 0 |

Decode graph 有 40 个 MIX task，均保留 AIC、AIV0、AIV1 sample。另有 81 个 predecessor
identity 来自当前 dispatch 外的 TensorMap creator；它们记录为 external dependency，而不是
当前 graph 中可执行 task。该 capture 未开启 args dump，不能证明 dispatch 的动态 sequence
length，只能保守标记 selected serving dispatch 和固定 kernel/config dimension。

首版拟合得到 54 个 feature-point group，其中 9 个 single-occurrence group 为 low confidence。
Combined plan 覆盖 11,696 个 compute task，保留 561 个 synchronization task，并可生成 stub。

完整 Qwen sleep run 通过未修改 serving request 生命周期和 PyPTO/simpler CPU-simulator
scheduler。虚拟设备内存为 64 GiB，沿用原生 KV sizing，分配 1082 个 KV page，并返回：

```text
token_ids: [2813, 2813, 2813, 2813, 2813]
finish_reason: length
```

请求 E2E 为 1.43 秒。该结果只证明功能可运行；CPU scheduling、host sleep resolution 和 9 个
low-confidence group 使它不能用于硬件精度结论。

## 2026-11-16 严格证据链重验

流水线按最新版 `vllm-ascend-simulator` 方法更新：

- confidence 必须由独立 `source.workload_id` 构成，不能只依赖同图重复 layer/task；
- 实际 replacement surface 按 `program + func_id` 校验，共享 task/subslot prediction 显式聚合；
- unmodeled compiled child 执行即 abort，不再使用零时长 stub；
- 首次执行 telemetry 证明真正运行的 modeled callable；
- 相邻验证 runner 分离 request warmup 和 measurement，并报告分布与 native drift。

在 physical NPU 2/3 完成三次完整 request capture（`C/D/E`），保留 request prefill `d4`、
sampling `d6` 和 decode step 4 `d16`。NPU 0/1 的并行尝试 `A/B` 在外部 VLLM 占卡后于
weight upload 失败，`collection_run.json` 为 `returncode=1`，已排除。

严格拦截发现 prefill func ID 25/26 只在 warmup graph 执行。随后通过
`--capture-dispatch prefill_fwd=1` 完成三次 ordinal capture（`F/G/H`），避免 request prefill
覆盖初始化 DFX 目录。

最终模型 `/tmp/pypto-qwen-model-v3` 包含 55 groups、76 feature points；每点有 3 个独立
workload，`status=baseline`，无 low-confidence point/group。

最终 plan `/tmp/pypto-qwen-plan-v3/simulation_plan.json` 覆盖 44,996 个 compute task instance，
保留 3,121 个 synchronization node，并聚合成 58 个实际 runtime callable。Missing model、
extrapolation、callable conflict、runtime-unmapped 均为 0，`strict_ready=true` 且
`runtime_stub_ready=true`。

单请求严格拦截 telemetry 执行 58/58 callable：24 prefill、32 decode、2 top-k，无 unmodeled
execution。相邻验证完成了旧 NPU 2 native-A 和 sleep 阶段：

| 运行 | Mean ms | P50 ms | P95 ms | 结果 |
| --- | ---: | ---: | ---: | --- |
| native-A | 215.992 | 213.446 | 227.482 | 5 次 native token 序列一致 |
| sleep | 1242.238 | 1243.972 | 1260.079 | 5 次 `[2813] * 5`，58 条 callable telemetry |

Sleep mean 相对该单边 native-A 暂高 475.13%。这不是正式 midpoint error，因为同卡 native-B
尚未完成。Coverage 与控制流通过，但墙钟精度未通过。

## 2026-11-16 Sleep 精度优化

逐轮实验详见 `../OPTIMIZATION_LOG.md`。关键结论：

1. 保持 task graph 和 58 callable 不变、把所有 kernel duration 置零后，官方 4-thread sim
   mean 仍为 `275.299 ms`；该值是 sleep 模式的 host floor。
2. 禁用 `sched_yield`、NUMA 限制和按角色 CPU affinity 均无改善。
3. 独立 6-thread sim runtime 曾将零时长 mean 降至 `253.349 ms`，但范围为
   `181.498–288.674 ms`，抖动不合格；该运行改变 scheduler 拓扑，只能用于 contention 诊断，
   不能作为 acceptance baseline。
4. 5-thread 和 7-thread 诊断 mean 分别为 `293.795 ms` 和 `307.530 ms`；16-thread custom
   runtime 因共享原子竞争恶化到 `316.808 ms`。线程数搜索至此停止。
5. `SIMPLER_SCHED_PROFILING` 显示 decode 图中 dispatch、complete/fanin/fanout、idle 扫描以及
   sim 线程生命周期共同构成 host floor；单个零时长 stub 不是主因。
6. 在 NPU 6 重测的当前 native-A 非常稳定：

```text
213.631, 210.660, 211.881, 213.808, 212.077 ms
mean = 212.411 ms
```

五次输出 token 都是 `[264, 8453, 67926, 5440, 2813]`。正式验收仍以 pinned 4-thread
runtime 的零时长 mean `275.299 ms` 为 host floor；它相对 native mean 高 `29.61%`。改变 AICPU
线程数会改变 cluster 归属和执行时序，因此所有 custom-thread 数据只作诊断，不能通过选择最接近
native 的配置完成 5% 验收。任何非负 per-kernel sleep 目前也无法仅靠 duration scaling 达标。

正式 bracket runner 已移除 `--sim-aicpu-threads`；后续优化必须保持 task graph、依赖、默认
scheduler 拓扑和请求计时不变。若 `a2a3sim` 等价平台原语优化仍无法降低 host floor，则转向原生
离散事件 `LATENCY_SIM`。

NPU 4 和 NPU 5 的两次重测在编译/初始化期间分别被外部进程抢占并 OOM，均未产生请求数据，
不得作为 baseline。NPU 6 native-A 成功后又被外部 `VLLM::EngineCore` 占用；native-B 必须
等待同一 physical NPU 6 空闲后重跑，不能换卡拼接。

## 2026-11-17 Persistent Runtime、Memory Profile 与当前下限

后续等价平台优化保留固定4-thread AICPU scheduler、原task graph和completion语义，同时恢复
profile对应的20-cluster AICore topology并复用OS worker。零时长固定3+5降至
mean/p50/p95 `186.819/190.840/197.161 ms`，证明存在正duration预算。

Native critical path为prefill `39.170 ms`、decode `29.803 ms`、top-k `0.248 ms`。60-role
profiling中5个decode scheduler window为`29.151/28.609/27.699/32.580/27.734 ms`，均值仅比
native decode低`2.18%`，因此剩余误差主要在graph startup、retirement、staging和request外壳，
不是duration简单缩放问题。

AICore executor SO现与persistent worker同生命周期；相同binary保持驻留，binary变化或finalize
才替换/关闭。该修复消除了prefill切换decode时的`Executor functions not loaded`。最终cache-only
固定3 warmup + 5 measured为mean/p50/p95 `229.768/227.470/239.165 ms`，执行58/58 callable并
返回`[2813] * 5`。相对NPU 6 native-A分别为`+8.17%/+7.26%/+11.86%`，仍未达到5%，且该结果
使用non-strict整体memory profile，不能替代正式midpoint验收。

新增memory profile和Qwen weight/KV虚拟materialization。8个native workload给出weight phase
中位数`38.4595 s`；原生static materialization参数面的exact tensor sum为
`29,540,610,048 bytes`，日志中的`27.98 GB`只是未padding模型参数估算。KV测量域为
`908–921 pages`，runtime reserved-memory中位数为`510 MB`。诊断运行在不搬运约49 GB真实
weight/KV数据的情况下得到918 pages、logical used memory `58.90/65.45 GB`，且weight/KV phase
分别补时到`38.462 s`和`33.346 s`。当前manifest
`/tmp/pypto-qwen-memory-profile-v1.json`整体仍为non-strict：weight点已有exact bytes和8个独立
duration workload，已strict-ready；相同KV page point未达到每点3个独立workload，reserved bytes
也是两个四舍五入值之差。正式runner会在native-A开始前拒绝它。

统一graph epoch会让60个role在相近deadline同步唤醒。原hybrid、低spin和无spin单请求分别为
`634.685/429.581/476.631 ms`，均明显回归；该路线已淘汰。Host实际允许`0-191`共192个CPU，
`nproc=1`来自`OMP_NUM_THREADS=1`，不是cgroup单CPU限制，因此回归原因仍需在timer、线程落点、
共享状态和NUMA之间继续审计。当前最佳仍使用每role
cumulative deadline，host scheduler已消耗的wall time会抵扣modeled预算。这是明确语义偏离，
不能表述为“每个stub独立等待完整duration”。

NPU 6当前仍被外部PID `1752482`的`VLLM::EngineCore`占用。未终止外部进程，正式同卡
`native-A -> simulator -> native-B`仍待设备释放。若graph外壳和request staging优化后仍不能达到
5%，应实现原生`ExecutionMode.LATENCY_SIM`，而不是缩放duration或修改请求计时。

## A2/A3 与 A2/A3Sim 控制面 Bracket

为避免用完整Qwen graph混淆平台固定开销，新增同图zero-work benchmark：`chain2_aiv`、
`spmd8_aiv`和`mix_aic_aiv0_aiv1`。每段排除编译、注册和worker初始化，在physical NPU 0完成
`native-A -> sim -> native-B`，每个case warmup 20、测量200轮。Sim orchestration/runtime仍为
pinned PyPTO路径；因sim环境缺pinned GCC 15，空child使用系统GCC 11，报告中显式标记。

| Case | Native midpoint p50 ms | Sim p50 ms | P50误差 | P95误差 |
| --- | ---: | ---: | ---: | ---: |
| `chain2_aiv` | 0.538 | 1.195 | +122.25% | +154.17% |
| `spmd8_aiv` | 0.536 | 1.220 | +127.55% | +156.77% |
| `mix_aic_aiv0_aiv1` | 0.540 | 1.244 | +130.60% | +143.07% |

Native A/B p50 drift为`16.33%–17.40%`，明显小于sim/native差异；mean也高
`124.56%–134.18%`。因此确认`a2a3sim`不是`a2a3`的性能等价平台。汇总为
`/tmp/pypto-platform-overhead-bracket-summary.json`。

## Native Device-Delay Zero Gate

已生成独立native plan `/tmp/pypto-qwen-plan-v3-native/simulation_plan.json`，SHA256为
`e1a79462de90e603019a7dc6b5abbb3850ab73f15e22b9a8ca1bcf19ed23b55a`。它包含58个runtime
callable model、44,996个simulated task，0 missing/conflict/unmapped，并声明`backend=a2a3`。

在physical NPU 1完成zero-duration gate。Hook保留真实parent orchestration、AICPU scheduler、
TensorMap、dispatch和completion，只注册24个prefill、32个decode和2个top-k delay child。Qwen
单请求返回`[2813] * 5`，wall为`165.577 ms`，日志无scheduler/device错误。产物为
`/tmp/pypto-qwen-native-delay-zero-npu1`。该结果证明58/58 replacement和控制输出可达，不是
延迟精度结果。

非零3+5第一次在NPU 1因验证命令只传1条prompt而在请求前退出；第二次发现NPU 2/3已有外部进程
各占约33.8 GB，并在真实weight materialization时OOM。两者都没有measured request，不能进入
精度统计。未终止或干预外部进程。

## NPU 1 非零 Device-Delay 收敛

在同一physical NPU 1固定Qwen3-14B、`Huawei is`、5 token、`max_num_seqs=16`，完成多轮相邻
bracket。原V3把warmup-prefill和request-prefill按`program + func_id`取全局中位数；35,860个
warmup task压过1,124个request task，导致request prefill使用错误feature duration。首轮稳定误差
约为mean/p50/p95 `+8%/+8%/+8%`，不是timer loop误差。

单task硬件诊断得到`observed = fixed + slope * target`，slope为`0.9966`、intercept为`0.067 us`。
Level-4 device-delay timeline进一步定位：V3 prefill makespan为`57.852 ms`，native约`39.2 ms`；
decode和top-k device compute已基本闭合。V3 prefill static CPM为`30.011 ms`，证明主要问题是
callable feature聚合，而非request外壳或weight/KV初始化。

Planner新增可重复`--prefer-capture-role`。同一callable存在指定role时只聚合该role，独占warmup
func ID仍保留；plan记录input/selected candidate count和selected/excluded role。V5将prefill static
CPM降到`20.240 ms`，同卡bracket结果如下：

| Case | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| native-D | 214.835 | 214.988 | 215.515 |
| V5 device-delay | 223.863 | 223.603 | 225.883 |
| native-E | 217.406 | 216.777 | 220.765 |
| native midpoint | 216.121 | 215.882 | 218.140 |
| V5误差 | +3.582% | +3.576% | +3.549% |

Native D/E drift为`+1.197%/+0.832%/+2.436%`。V5满足5%，但level-4仍显示prefill observed
compute `44.560 ms`，高于native约`35.46 ms`。原因是profile只保留每个task/subslot的最慢block，
stub让所有physical block都等待该最大值，延长了非关键block的core occupancy。

V6 profile保留每个subslot、每个epoch的排序physical-block duration distribution；latency model对
相同rank跨J/K/L三个独立request workload取中位数。Device child通过`get_block_idx(args)`选rank，
MIX AIV通过`get_sub_block_id(args)`区分AIV0/AIV1。排序分布不伪造原始block identity，但保留
block count、duration distribution和最慢block完成时延。V6模型为55 groups、76 feature points，
每点3个独立workload，0 low-confidence；58/58 callable都有block model。

首版连续`switch(block_idx)`被CCEC优化成未注册的`.rodata` jump table，warmup首次dispatch触发
`507901`并由runtime force-reset。该失败无request样本，未进入bracket。实现改为`.text`内二叉比较
分支后，完整smoke通过。reset后重新执行的正式bracket为：

| Case | Mean ms | P50 ms | P95 ms | Stdev ms |
| --- | ---: | ---: | ---: | ---: |
| native-G | 214.813 | 215.204 | 215.349 | 0.858 |
| V6 block-rank | 215.867 | 215.734 | 216.554 | 0.439 |
| native-H | 214.649 | 215.484 | 215.538 | 1.422 |
| native midpoint | 214.731 | 215.344 | 215.443 | - |
| V6误差 | +0.529% | +0.181% | +0.515% | - |

Native G/H drift为`-0.076%/+0.130%/+0.088%`。58个callable全部替换，8个simulator请求均返回
`[2813] * 5`，日志无scheduler/device错误。正式bracket后补做V6 level-4归因：prefill makespan/
observed compute为`39.803/36.195 ms`，native约`39.1–39.2/35.46 ms`；decode为
`29.948/29.193 ms`，native约`29.79/28.80 ms`；top-k makespan为`0.254 ms`，native约
`0.248 ms`。V3/V5 prefill makespan分别为`57.852/48.542 ms`，因此改进来自block occupancy闭合，
不是E2E偶合。

Plan SHA256为`508b8228de14649c07f12d3aff63db178707d6a8e4740d87b5289b534d847273`；正式产物位于
`/tmp/pypto-qwen-native-delay-v6-block-bracket2-npu1`，level-4产物位于
`/tmp/pypto-qwen-native-delay-v6-block-profile-npu1`。逐callable分析器对58个callable、42,593个
block observation完成严格比较，0 missing：overall MAE/MAPE为`0.184 us/2.78%`，decode为
`0.113 us/0.50%`，prefill为`0.241 us/4.57%`。Top-k为`0.624 us/22.82%`；高百分比来自约
1 us小block的固定计时开销，最大绝对误差仅`5.48 us`，top-k graph makespan仍只高约`6 us`。
报告位于同目录`operator_accuracy.json`。

该精度只覆盖固定request feature。Request kernel中的weight/KV/HBM访问耗时已内含在native
physical-block duration中；它不是独立可审计的`{operation, bytes}`内存模型。高层weight/KV
capacity、allocation、materialization和lifetime由memory sidecar负责；V6 bracket时旧manifest仍为
non-strict，后续exact evidence闭合见下一节。两类证据不能互相替代，多卡HCCL也仍未覆盖。

## Strict Memory Evidence 与虚拟生命周期

在空闲physical NPU 1连续采集M1/M2/M3三个独立native workload，均成功并得到相同point：

- static weight exact sum：`29,540,610,048 bytes`；
- KV：`921 pages * 20,971,520 bytes = 19,314,769,920 bytes`；
- exact total memory：`65,452,113,920 bytes`；
- exact simpler committed：`39,155,604,908 bytes`；
- runtime reserved：`422,531,668 / 421,958,228 / 421,950,036 bytes`。

新manifest `/tmp/pypto-qwen-memory-profile-v2-strict.json` 的weight、KV、footprint和顶层均为
`strict_ready=true`，SHA256为
`8667f25f0305e7e28ee05eaad88a7f4771134284e2ee604712e656a14cb73c78`。Weight中位duration为
`38.375 s`，KV中位duration为`31.008 s`。

首个strict smoke仍按旧reserved近似算出922 pages，单点模型拒绝外推；修复后footprint直接保存
native exact `total_bytes`和`driver_visible_non_kv_bytes`。第二个smoke精确命中921 pages，但在KV
分配后的breakdown查询中把总committed误与non-KV footprint比较，再次fail closed。最终实现记录
首次sizing的simulator non-KV baseline，并把后续committed增量作为逻辑KV allocation计入free
memory，borrow/free生命周期不被旁路。

最终strict smoke位于`/tmp/pypto-qwen-strict-memory-smoke-v3`：921 pages、逻辑used/free
`58.89/6.56 GB`，weight/KV phase分别约`38.376 s/31.008 s`，58/58 telemetry和`[2813] * 5`
通过。Exact footprint存在时，`PYPTO_SLEEP_DEVICE_MEMORY_BYTES`不能静默覆盖native capacity。

## Native Device-Delay 低HBM状态

历史V6 bracket的weight/KV仍有真实HBM backing，因此其`<1%`时延只证明child block-rank模型，
不能证明低内存高精度路径。现已加入native `rtMalloc/rtFree` interposer：weight、profile scratch KV
和final KV由exact one-shot token分类，返回`aclrtReserveMemAddress`保留的有效device VA，不调用
`rtMalloc`或`aclrtMallocPhysical`提交payload HBM。原allocator仍记录完整逻辑extent、caller borrow、
canonical identity、free和committed bytes。

Launcher强制要求strict memory profile和memory audit。Audit证明weight
`29,540,610,048 bytes`、final KV `19,314,769,920 bytes`以及scratch KV `20,971,520 bytes`
全部虚拟化且`physical_backing_bytes=0`。首个硬件smoke被运行中途抢卡的外部57.4 GB VLLM任务污染，
在4 GiB arena增长时OOM；保留证据且未干预外部PID。第二个smoke因logical memory view缺421.96 MB
native driver/runtime reserved而算出941 pages，strict exact point正确拒绝。

修复后`device_memory_info`取exact logical footprint与physical driver used的最大值，既恢复921 pages，
也保留外部占用收紧容量。第三个NPU 2 zero-duration smoke完成58/58 replacement和`[2813] * 5`；
100 ms采样只有一个NPU PID，物理peak为`9,312 MB`，结束后无残留进程。产物位于
`/tmp/pypto-qwen-native-delay-v7-vmem-zero3-npu2`。

同一physical NPU 2的新正式bracket为：

| Case | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| native-A | 216.528 | 216.576 | 218.354 |
| low-HBM device-delay | 217.503 | 217.122 | 219.319 |
| native-B | 218.848 | 218.569 | 219.913 |
| native midpoint | 217.688 | 217.572 | 219.134 |
| error | -0.085% | -0.207% | +0.085% |

Native drift为`+1.071%/+0.920%/+0.714%`。所有15个measure均返回5 token和length finish；simulator
请求均为`[2813] * 5`。Plan/memory SHA256保持为
`508b8228de14649c07f12d3aff63db178707d6a8e4740d87b5289b534d847273`和
`8667f25f0305e7e28ee05eaad88a7f4771134284e2ee604712e656a14cb73c78`。正式产物位于
`/tmp/pypto-qwen-native-delay-v7-vmem-bracket-npu2`，summary SHA256为
`3d7948a97977d4afb7823e0a0d669783bb4aac55c3213270659a23dfa4efa709`。最终回归为`46 passed`。

## 有效产物

- request capture：`/tmp/pypto-qwen-profile-v2-{C,D,E}`；
- warmup-prefill capture：`/tmp/pypto-qwen-profile-v2-{F,G,H}`；
- export：`/tmp/pypto-qwen-export-v2`；
- model：`/tmp/pypto-qwen-model-v3`；
- strict plan：`/tmp/pypto-qwen-plan-v3`；
- request-role plan：`/tmp/pypto-qwen-plan-npu1-v5-native`；
- block-rank model/plan：`/tmp/pypto-qwen-model-npu1-v6-block`、
  `/tmp/pypto-qwen-plan-npu1-v6-block-native`；
- V5 bracket：`/tmp/pypto-qwen-native-delay-v5-bracket-npu1`；
- V6正式bracket：`/tmp/pypto-qwen-native-delay-v6-block-bracket2-npu1`；
- V6 level-4归因：`/tmp/pypto-qwen-native-delay-v6-block-profile-npu1`；
- strict smoke：`/tmp/pypto-qwen-sim-diagnose-v3`；
- 旧 partial matrix：`/tmp/pypto-qwen-validation-v2`；
- 当前 NPU 6 native-A：`/tmp/pypto-qwen-opt-native-a-npu6`；
- memory profile：`/tmp/pypto-qwen-memory-profile-v1.json`（历史non-strict）、
  `/tmp/pypto-qwen-memory-profile-v2-strict.json`（当前strict）；
- exact memory evidence：`/tmp/pypto-qwen-memory-evidence-npu1`；
- strict memory smoke：`/tmp/pypto-qwen-strict-memory-smoke-v3`；
- 最终cache-only runtime：`/tmp/pypto-pool60-cache-final-runtime/lib`；
- cache-only固定3+5：`/tmp/pypto-qwen-sim-v3-pool60-cache-final2-w3`；
- 统一epoch失败实验：`/tmp/pypto-qwen-sim-v3-pool60-memorydiag2-r1`、
  `/tmp/pypto-qwen-sim-v3-pool60-memorydiag-epochsleep-r1`、
  `/tmp/pypto-qwen-sim-v3-pool60-memorydiag-epochnospin-r1`；
- 精度实验：`/tmp/pypto-qwen-sim-zero-*`。

早期 `507018` capture、OOM capture A/B，以及 NPU 4/5 抢占失败日志都不是 latency 数据。
