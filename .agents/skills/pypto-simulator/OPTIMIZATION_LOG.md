# PyPTO Simulator 精度优化日志

本文档按时间顺序记录每一次性能精度优化。每轮必须写明假设、改动、相同 workload 下的结果、
结论和产物。失败方案同样保留，防止重复试验或选择性报告。

## 固定验证配置

- 模型：`/data/disk1/liangkaiyi/Qwen3-14B`
- prompt：`Huawei is`
- 输出：5 token，greedy，`ignore_eos=true`
- serving：`max_model_len=512`、`max_num_seqs=16`、`max_num_batched_tokens=4096`
- ring：dep/task `262144`，heap `1 GiB`
- 请求：3 次 request warmup + 5 次 measured request
- simulator plan：`/tmp/pypto-qwen-plan-v3/simulation_plan.json`
- 验收：同一 physical NPU 的 `native-A -> simulator -> native-B`；mean/p50/p95 相对 native
  midpoint 的绝对误差均不超过 5%

## 基线：严格覆盖修复

**改动**：加入独立 `workload_id`、warmup ordinal capture、`program + func_id` callable 聚合、
unmodeled child fail-closed 和首次执行 telemetry。

**结果**：55 groups、76 feature points，每点 3 个独立 workload；44,996 simulated task、3,121
preserved sync node，聚合为 58 个 runtime callable；0 missing、0 extrapolation、0 conflict。严格
simulator 执行 58/58 callable。

**墙钟**：旧 NPU 2 native-A mean `215.992 ms`；原始 sleep mean `1242.238 ms`，相对单边
native-A 为 `+475.13%`。Native-B 因同卡被占用未完成。

**结论**：功能 coverage 通过，性能不通过。产物：`/tmp/pypto-qwen-validation-v2`。

## 轮次 1：零时长 Plan

**假设**：区分 latency model 与 CPU simulator 固有 host floor。

**改动**：保持 58 个 callable、task graph、控制输出和 scheduler 不变，仅将所有
`kernel_models[].duration_us` 置零。

**结果**：measured request 为 `304.972/264.902/260.812/253.773/292.035 ms`，mean
`275.299 ms`；58 个 callable 全部执行。

**效果**：相对旧 native-A 仍高约 `27.46%`。非负 sleep 不可能低于该 floor。

**结论**：当前误差首先是 host scheduler/dispatch 问题，不是模型 duration scaling 问题。产物：
`/tmp/pypto-qwen-sim-zero`。

## 轮次 2：禁用 `sched_yield`

**假设**：sim scheduler 空轮询中的 `sched_yield()` 系统调用占主导。

**改动**：仅在 simulator 进程临时 interpose `sched_yield`，改为 CPU yield hint，不修改 task。

**结果**：零时长 mean `277.024 ms`，范围 `240.769–296.687 ms`。

**效果**：相对轮次 1 变慢 `0.63%`，无统计改善。

**结论**：poll syscall 不是主因，方案淘汰。产物：`/tmp/pypto-qwen-sim-zero-noyield`。

## 轮次 3：AICPU 线程 16，官方 Runtime

**假设**：增加 scheduler 线程可以减少 10k 级 task 的 dispatch/completion overhead。

**改动**：增加显式 `--sim-aicpu-threads` 实验入口。

**结果**：官方 pinned runtime 在首个 task 前拒绝：`launch_aicpu_num (16) must be ... [2, 4]`。

**结论**：无 timing 数据；官方 ABI 上限为 4，不能直接调高。失败产物：
`/tmp/pypto-qwen-sim-zero-t16`。

## 轮次 4：NUMA/CPU 限制与 48 GiB 虚拟显存

**假设**：跨 NUMA 迁移和 61.8 GB host-backed 虚拟设备 allocation 导致抖动。

**改动**：CPU 限定为 `72-191`，内存 interleave 到 NUMA `3,5,7`；虚拟设备内存降为
48 GiB，KV page 从 1082 降到 345，仍覆盖 16×512 token。

**结果**：零时长 mean `325.506 ms`，范围 `283.427–369.264 ms`。

**效果**：相对轮次 1 恶化 `18.24%`。

**结论**：整体 affinity 增加 worker 竞争；缩小 KV 只减少初始化时间，不改善请求延迟。方案
淘汰。产物：`/tmp/pypto-qwen-sim-zero-numa48`。

## 轮次 5：当前 Native-A 重测

**改动**：在空闲 physical NPU 6 使用固定配置重跑 native-A。

**结果**：5 次 native token 均为 `[264, 8453, 67926, 5440, 2813]`；measured request 为
`213.631/210.660/211.881/213.808/212.077 ms`，mean `212.411 ms`，范围仅 3.15 ms。

**结论**：旧 215.992 ms baseline 有效，当前更稳定目标约 212 ms。产物：
`/tmp/pypto-qwen-opt-native-a-npu6`。

## 轮次 6：独立 6-thread Sim Runtime

**假设**：官方 sim 已启动 6 个 physical AICPU host thread，只需让 6 个全部参与调度。

**改动**：从同一 simpler revision `355ce3015` 建独立 worktree，把 sim runtime active 上限由 4
提高到 launch 上限 6；不覆盖 venv 或硬件 runtime。Sleep hook 增加显式 custom runtime 目录。

**结果**：零时长 mean `253.349 ms`，范围 `181.498–288.674 ms`。

**效果**：相对轮次 1 改善 `7.97%`，但相对当前 native-A mean 仍高 `19.27%`，且抖动过大。

**结论**：scheduler 并行度有影响，但 6 线程不足且稳定性不合格。产物：
`/tmp/pypto-qwen-sim-zero-fast6`、`/tmp/pypto-fast-runtime/lib`。

## 轮次 7：独立 16-thread Sim Runtime

**假设**：15 scheduler + 1 orchestrator 可进一步降低 host floor。

**改动**：custom sim runtime 同时把 active 和 launch 上限扩为 16。

**结果**：零时长 mean `316.808 ms`，范围 `281.317–358.543 ms`。

**效果**：原子竞争超过并行收益，相对 6-thread 恶化 `25.04%`。

**结论**：线程越多并非越快，方案淘汰。产物：`/tmp/pypto-qwen-sim-zero-fast16`。

## 轮次 8：6-thread 按角色固定物理 CPU

**假设**：为 60 个 core worker 和 6 个 AICPU 分配独立 CPU 可消除迁移抖动。

**改动**：临时按 `sim-aic-*`、`sim-aiv-*`、`sim-aicpu-*` thread name 设置 affinity。

**结果**：零时长 mean `326.995 ms`，范围 `255.248–385.624 ms`。

**效果**：相对未绑定 6-thread 恶化 `29.07%`。

**结论**：固定到后半区 CPU 造成更大共享数据/NUMA 成本，方案淘汰。产物：
`/tmp/pypto-qwen-sim-zero-fast6-affinity`。

## 轮次 9：Scheduler Profiling

**假设**：用 runtime 自带 `SIMPLER_SCHED_PROFILING` 分解零时长 host floor，定位真实热点。

**改动**：构建 profiling-only 6-thread runtime；修复日志门禁，launcher 通过
`PYPTO_SIMPLER_LOG_LEVEL=INFO` 在 worker 初始化前设置 C++ runtime 阈值。该 build 和 INFO 日志
只用于诊断，不作为 timing baseline。

**结果**：profiling 单请求为 `539.250 ms`，日志开销显著，不能与 native 比较。最后一个 decode
图的 5 个 scheduler thread 各运行约 `30.1–30.3 ms`，合计完成 17,824 个 subtask；其中每线程
`dispatch` 约占 `40.7–50.0%`，`complete` 约占 `17.6–23.8%`，`idle` 约占 `26.3–41.6%`。
Device phase 同时记录 scheduler window `38.104 ms`、run wall `68.304 ms`，说明线程启动、core
retirement 和 runtime teardown 也占有显著成本。

**结论**：热点是原 scheduler 对大量 block/subslot 的 dispatch、依赖 fanin/fanout 与 sim 线程
生命周期成本，不是单个 sleep stub。产物：`/tmp/pypto-qwen-sim-zero-fast6-prof-info2`。

## 轮次 10：5/7-thread 诊断与方法纠偏

**假设**：仅用于判断 scheduler contention 随线程数的方向，不作为精度校准。

**结果**：5-thread measured mean/p50/p95 为 `293.795/297.805/302.140 ms`；7-thread 为
`307.530/304.831/329.267 ms`，均劣于 6-thread 诊断结果。

**方法纠偏**：改变 AICPU 线程数会改变 cluster 到 scheduler 的归属和执行时序。选择一个最接近
native 的线程数违背“沿原执行链路模拟”的要求，即使结果碰巧进入 5% 也不能验收。因此：

- 5/6/7/16-thread custom runtime 全部只保留为 host contention 诊断证据；
- 正式 `native-A -> simulator -> native-B` runner 移除 `--sim-aicpu-threads`；
- acceptance 必须使用 pinned runtime 的默认 4-thread scheduler 拓扑；
- 后续只优化 `a2a3sim` 平台原语的实现成本，不能改变 task graph、依赖、scheduler 拓扑或请求计时。

产物：`/tmp/pypto-qwen-sim-zero-fast5`、`/tmp/pypto-qwen-sim-zero-fast7`。

## NPU 占用记录

- NPU 4 native-A 在编译期间被外部 PID `2525646` 抢占，首次 runtime allocation OOM；无请求数据。
- NPU 5 随后被外部 PID `2749836` 抢占，warmup resource growth OOM；无请求数据。
- 两轮均不得作为 native baseline；NPU 6 释放后整组重新开始并成功得到轮次 5。
- Native-B 必须等待 NPU 6 再次空闲，不能换卡与该 native-A 拼接。

## 轮次 11：Persistent Worker 与低扰动 Stub

**改动**：复用 4-thread AICPU 和 AICore 逻辑角色对应的 OS worker；每轮仍创建原生逻辑角色、
task、依赖和 completion。首次执行 telemetry 改为只读 fast path + CAS，零 duration 不读取时钟。

**结果**：固定 3 warmup + 5 measured 的零时长 mean/p50/p95 为
`186.819/190.840/197.161 ms`。

**结论**：OS thread 创建与回收是主要固定成本之一。该优化不删除逻辑角色，可保留；零时长已
低于 native，存在正的 modeled duration 预算。产物：
`/tmp/pypto-qwen-sim-zero-pool4-faststub-w3`。

## 轮次 12：逐 Stub Sleep、Hybrid Delay 与 24-role Deadline

**结果**：

- 每个 stub 独立 `sleep_until`：`1173.009/1172.534/1184.963 ms`；
- 短时长 spin、长时长 sleep + spin：`556.845/555.471/574.475 ms`；
- 24-role cumulative deadline：`503.478/503.255/509.058 ms`。

**结论**：微秒 sleep oversleep 会沿依赖链累积。Cumulative deadline 让同一role已经消耗的
host wall time抵扣后续modeled预算，性能更好，但它不再是“每个stub独立等待完整duration”，是
必须审计的语义偏离。产物分别为 `/tmp/pypto-qwen-sim-v3-pool4-faststub-w3`、
`/tmp/pypto-qwen-sim-v3-pool4-hybrid-w3` 和
`/tmp/pypto-qwen-sim-v3-pool4-deadline-w3`。

## 轮次 13：Duration 层级与 Native Critical Path 闭合

**测量**：native critical path 为 prefill `39.170 ms`、decode `29.803 ms`、top-k
`0.248 ms`。decode plan 静态关键路径为 `8.371 ms`，C/D/E raw profile 为
`8.776–8.876 ms`。`sum(duration x block_num)=549.854 ms`，native physical work 为
`590.807–596.567 ms`。

**结论**：exporter 的 duration 是physical block耗时，logical task span可能包含多波执行；模型
不存在简单的block重复放大。剩余误差不能通过猜测倍率修改duration，重点应转向graph外壳、
staging和runtime生命周期。

## 轮次 14：恢复 20-cluster Physical Topology

**发现**：官方 `SIM_AUTO_BLOCKDIM=8` 只启动 `8 AIC + 16 AIV`，与profile使用的20-cluster
NPU不一致。

**改动**：sim恢复 `20 AIC + 40 AIV`，persistent pool复用对应60个AICore逻辑role；AICPU仍为
正式固定4-thread拓扑。

**结果**：60-role单请求 `349.088 ms`；profiling单请求 `317.028 ms`。5个decode scheduler
window为 `29.151/28.609/27.699/32.580/27.734 ms`，均值相对native decode为 `-2.18%`。

**结论**：decode scheduler内部已基本闭合。剩余主要在prefill graph、top-k staging、启动、
retirement和request外壳。稳定产物：`/tmp/pypto-pool60-deadline-runtime/lib`。

## 轮次 15：AICore SO 生命周期修复

**故障**：旧 `drain_execution()` 每轮 `dlclose` AICore executor，但binary identity cache仍认为
相同binary已加载；prefill后decode报 `Executor functions not loaded` 和
`RunPreparationRefused: code -1000`。

**修复**：所有逻辑role完成并join后保留executor SO；只有下一程序binary bytes变化或worker
finalize时替换/关闭。空handle无条件触发重载。

**结果**：prefill -> decode -> top-k多程序切换成功，58/58 callable和合法token均通过；不再出现
函数表丢失。cache-only冷态单请求为 `282.124 ms`。固定3 warmup + 5 measured诊断为
mean/p50/p95 `229.768/227.470/239.165 ms`，较旧24-role正式mean `503.478 ms`减少
`54.36%`。产物：`/tmp/pypto-pool60-cache-final-runtime/lib` 和
`/tmp/pypto-qwen-sim-v3-pool60-cache-final2-w3`。

## 轮次 16：Weight/KV 虚拟内存与 Memory Profile

**改动**：新增 `export_memory_profile.py`，从独立native serving日志导出
`weight_materialize`、`kv_materialize` 和 `runtime_reserved_memory`。模拟时仍执行原生
allocation、地址登记、borrow/free和KV sizing，只对大tensor省略真实init/copy，返回稳定逻辑地址，
并在原phase内部补足测得duration。小于阈值的控制tensor和scratch保持真实backing。

**证据**：8个独立native workload中，weight phase中位数 `38.4595 s`；原生static
materialization参数面的exact tensor sum为 `29,540,610,048 bytes`，而日志中的`27.98 GB`只是
未padding模型参数估算。KV测量域为`908–921 pages`。driver-visible `peak_non_kv` 与simpler
committed的差值中位数为 `510 MB`。

**门禁**：当前manifest `/tmp/pypto-qwen-memory-profile-v1.json` 整体仍为non-strict。weight点已有
8个独立duration workload和exact bytes，已strict-ready；KV相同page point未达到每点3个独立
workload，reserved bytes仍是两个四舍五入日志值之差。正式runner拒绝整体manifest，只有
diagnostic launcher可显式允许。

**结果**：虚拟weight phase为 `38.462 s`，918-page KV phase为 `33.346 s`；logical used memory
为 `58.90/65.45 GB`，不发生约47 GB真实weight/KV搬运。请求热路径对照仍为约350 ms，说明初始化
虚拟化没有偷改请求计时。

## 轮次 17：统一 Graph Epoch 失败实验

**假设**：所有AICore role从同一graph `sim_t0`累计deadline，可消除persistent worker分别唤醒的
epoch偏斜。

**环境事实**：进程允许使用`0-191`共192个CPU。`nproc`显示1是因为环境中
`OMP_NUM_THREADS=1`，不是cgroup CPU配额；persistent worker使用`std::thread`，不能据此推断只有一个
CPU。58个kernel model中46个小于`50 us`，60个AICore role在相近deadline同步唤醒时仍可能产生
共享状态、timer、NUMA和run-queue开销，但具体占比尚未闭合。

**结果**：

- 统一epoch + 原100 us spin margin：单请求 `634.685 ms`；
- 超过10 us即sleep、末尾2 us spin：`429.581 ms`；
- 完全绝对sleep、无spin：`476.631 ms`。

无spin版本request graph为 prefill `53.583 ms`、top-k `8.098 ms`、5次decode
`30.344/51.016/80.382/60.709/75.447 ms`。绝对timer和同步run queue造成依赖链上的role迟到，
同role cumulative deadline不能抵扣其他role的迟到。

**结论**：统一wall-clock epoch在当前host环境下显著回归，方案淘汰；不能把回归归因于单CPU
配额，后续需单独审计线程落点、共享锁/原子、timer和NUMA。主工作树恢复为每role执行起点；这是
已声明的sleep语义偏离，不得表述为理想的每stub完整sleep。失败产物：
`/tmp/pypto-pool60-epoch-cache-runtime-v2/lib`、
`/tmp/pypto-pool60-epoch-sleep-runtime/lib`、
`/tmp/pypto-pool60-epoch-nosspin-runtime/lib`。

## 轮次 18：Host 环境纠偏、Graph Telemetry 与固定 Slot 失败实验

**环境纠偏**：进程的`Cpus_allowed_list`为`0-191`。`nproc=1`来自环境变量
`OMP_NUM_THREADS=1`，不是cgroup单CPU配额；persistent executor使用`std::thread`，此前把统一epoch
回归解释为“60个role争用一个CPU”不成立。统一epoch回归数据仍有效，但原因需在timer、线程落点、
共享锁/原子和NUMA之间继续审计。

**Graph telemetry**：首版诊断在每个graph后打开并追加JSONL。固定3+5被扰动到
`489.760–702.963 ms`，且当时host load average约`103–116`，不得用于性能结论。实现已改为进程内
缓冲、退出时一次写盘，并增加单元测试证明flush前不创建文件。

**固定slot假设**：让60个AICore和4个AICPU role跨graph固定到同一个OS worker，移除共享deque
领取竞争。

**结果**：固定3+5 measured为`229.477/304.124/450.999/370.251/371.395 ms`，mean约
`345.249 ms`，明显劣于共享队列候选。该轮host load仍约90以上，环境与实现变量混杂，不能量化
固定slot自身成本；但它没有提供改善证据，且固定随机创建线程会引入未证明的NUMA/locality语义，
因此回退，不进入主候选。产物：`/tmp/pypto-pool60-slots-runtime/lib`、
`/tmp/pypto-qwen-sim-v3-pool60-slots-w3`。

## 轮次 19：Exact Native Memory Evidence 链路

新增`install_memory_evidence_hook.py`和`run_memory_evidence.py`。该sidecar只允许native `a2a3`，
不替换allocation/copy，记录static tensor exact sum、KV实际page bytes/count、`free_bytes`、
`total_bytes`和`simpler_committed_bytes`原始整数。公式保持与serving一致：
`runtime_reserved_bytes = max(total_bytes - free_bytes - simpler_committed_bytes, 0)`。

`export_memory_profile.py`新增可重复的`--evidence`。Duration仍来自`--log` clean运行；weight feature
和footprint来自sidecar。Footprint必须有至少3个独立workload，clean KV duration仍必须在相同bytes
point有至少3个独立workload，整体才会`strict_ready=true`。Sidecar间weight bytes不一致、KV bytes
算术不一致或workload ID无效均fail closed。

当前Python完整回归为`32 passed`；Ruff、compileall、`diff --check`和回退固定slot后的C++ runtime
构建均通过。由于NPU 6在约25分钟只读轮询中始终由外部PID `1752482`占用`57029 MB`，sidecar尚未
做native smoke，不能把单元测试结果当硬件证据，也未生成strict memory manifest或native-B。

## 当前判断与下一步

1. 当前最佳固定3+5 simulator诊断为mean/p50/p95
   `229.768/227.470/239.165 ms`；相对native-A分别为`+8.17%/+7.26%/+11.86%`，仍未达到5%。
   它使用non-strict整体memory profile，不能替代A/B midpoint验收。
2. Memory virtualization已覆盖Qwen weight/KV高层phase，weight exact bytes和独立duration
   coverage已strict-ready；仍需补相同KV page point至少3个独立native workload和精确
   reserved-memory读数。
3. 60-role decode scheduler内部接近native；主要剩余成本是prefill约14 ms、top-k约8 ms、graph
   startup/retirement/future join和request外壳。应先做per-graph零时长与cache-only 3+5分解。
4. 当前host允许192个CPU，但统一epoch同步唤醒仍产生明显且不稳定的额外开销；根因尚未在
   timer、线程落点、共享状态和NUMA之间闭合。继续微调sleep threshold不能作为校准方法；若平台原语和staging优化后仍超过5%，应转向原生
   `ExecutionMode.LATENCY_SIM`离散事件推进。
5. NPU 6当前被外部`VLLM::EngineCore`占用。不得终止外部进程；空闲后需重跑完整同卡
   `native-A -> simulator -> native-B`，并以midpoint同时检查mean/p50/p95。

## 轮次 20：同图 Zero-Work 平台 Bracket

**假设**：当前剩余误差可能来自`a2a3sim`与真实`a2a3`控制面的结构性性能差异，而不是Qwen
latency model。

**改动**：新增`chain2_aiv`、`spmd8_aiv`和`mix_aic_aiv0_aiv1`三个同图zero-work case；排除
编译、注册和worker初始化，在physical NPU 0执行`native-A -> sim -> native-B`，每段warmup 20、
测量200轮。

**结果**：sim p50相对native midpoint分别高`122.25%/127.55%/130.60%`，p95高
`154.17%/156.77%/143.07%`；native A/B p50 drift为`16.33%–17.40%`。

**结论**：确认两平台存在显著控制面性能差异。`a2a3sim`继续作为无卡CI/control-flow smoke；
高精度路线优先转向隔离的`a2a3` device-delay。产物：
`/tmp/pypto-platform-overhead-bracket-summary.json`。

## 轮次 21：A2/A3 Device-Delay Zero Gate

**改动**：planner新增`--backend a2a3`；新增native-delay hook和launcher。保留真实parent、AICPU、
TensorMap、dispatch/ACK/FIN和completion，仅将modeled AIC/AIV child替换为CCEC同ABI kernel。
非零kernel按50 MHz `get_sys_cnt()`等待完整duration；未建模备用func_id不注册到callable table，
意外dispatch在scheduler查址阶段失败。常规native路径完全旁路。

**结果**：native plan包含58个callable、44,996个simulated task，0 missing/conflict/unmapped。
Physical NPU 1上的zero gate精确替换`24 prefill + 32 decode + 2 top-k`，返回
`[2813] * 5`，单请求wall为`165.577 ms`，无scheduler/device错误。产物：
`/tmp/pypto-qwen-native-delay-zero-npu1`。

**结论**：device-delay replacement surface、真实设备控制流和Qwen控制输出通过可达性门禁；
zero-duration不是精度通过。非零3+5尚无有效请求样本：一次命令prompt数量错误，在请求前退出；
另一次因NPU 2/3外部进程各占约33.8 GB，在weight materialization时OOM。必须等待空闲卡后完成
同卡`native-A -> device-delay -> native-B`，不能复用旧native-A或跨卡拼接。

## 轮次 22：Nonzero Bracket 与 Device Timer 排除

**结果**：NPU 1首轮V3 bracket的simulator mean/p50/p95为
`242.729/242.819/243.256 ms`；native midpoint为`222.895/218.871/239.317 ms`，误差
`+8.90%/+10.94%/+1.65%`。Native p95 drift为`-14.85%`，因此p95表面通过不稳定。第二轮稳定端点
仍复现约8%正偏。

**诊断**：单task device-delay sweep拟合为`slope=0.9966`、`intercept=0.067 us`，排除50 MHz
`get_sys_cnt()` loop倍率。Graph timing和level-4显示偏差发生在task graph内部：V3 prefill makespan
`57.852 ms`，native约`39.2 ms`；decode为`30.671 ms`对native约`29.79 ms`；top-k device compute
约`0.255 ms`对native`0.248 ms`。

## 轮次 23：Capture Role 隔离

**根因**：V3 plan把`initialization_warmup_prefill`的35,860个task和`request_prefill`的1,124个
task按同一`program + func_id`取中位数。Warmup以约32:1数量优势覆盖request feature，例如QK/PV
partial request约`17 us`，stub却执行约`131 us`。这违反feature-point隔离。

**修复**：Planner新增可重复`--prefer-capture-role`。同一callable存在首选role时只聚合该role；
warmup独占func ID继续保留。Kernel model记录input/selected candidate count、selected/excluded role。
默认行为不变，新增回归覆盖per-callable fallback。

**V5 bracket**：native midpoint mean/p50/p95为`216.121/215.882/218.140 ms`；simulator为
`223.863/223.603/225.883 ms`，误差`+3.582%/+3.576%/+3.549%`。Native drift为
`+1.197%/+0.832%/+2.436%`。V5首次稳定满足5%。

## 轮次 24：Physical Block Distribution

**剩余问题**：V5 level-4 prefill static CPM已从V3的`30.011 ms`降至`20.240 ms`，接近native
约`18.7–19.0 ms`；但observed compute仍为`44.560 ms`，高于native约`35.46 ms`。Exporter只保留
每个task/subslot最慢block，导致所有stub block都占用最大duration。

**修复**：

- profile新增`subslot_block_duration_samples_us`，每个epoch保存排序physical-block distribution；
- contract严格校验slot、block_num、非负有限值和排序；
- latency model对同rank跨独立workload取中位数，并支持测量域内插值；
- plan按subslot保留block duration，维度冲突时回退task/subslot duration；
- device child用`get_block_idx(args)`选rank，MIX AIV用`get_sub_block_id(args)`选lane；
- audit记录每个subslot的block duration count。

V6模型包含55 groups、76 feature points，每点3个独立workload、0 low-confidence，58/58 callable
都有block model。排序分布保留occupancy distribution和最大完成时延，但不声称恢复原始block
identity。

## 轮次 25：Jump Table 失败与 V6 正式通过

**失败实验**：首版用连续`switch(block_idx)`选择duration。CCEC将其优化为`.rodata` jump table，
hook只注册ELF `.text`，warmup首次dispatch触发`507901`。Runtime完成force-reset；该次无request
样本，native端点作废。未干预外部任务。

**修复**：改为二叉比较分支，duration立即数和控制流全部位于`.text`。最大50-block stub为700-byte
`.text`，完整warmup/request smoke通过。

**正式 bracket**：reset后在physical NPU 1执行`native-G -> V6 -> native-H`：

| Case | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| native-G | 214.813 | 215.204 | 215.349 |
| V6 block-rank | 215.867 | 215.734 | 216.554 |
| native-H | 214.649 | 215.484 | 215.538 |
| midpoint | 214.731 | 215.344 | 215.443 |
| error | +0.529% | +0.181% | +0.515% |

Native drift为`-0.076%/+0.130%/+0.088%`。58/58 callable、8次`[2813] * 5`和returncode 0均通过。
补充V6 level-4得到prefill makespan/compute `39.803/36.195 ms`，decode
`29.948/29.193 ms`，top-k makespan `0.254 ms`；native对应约`39.1–39.2/35.46 ms`、
`29.79/28.80 ms`和`0.248 ms`。V3/V5 prefill makespan为`57.852/48.542 ms`，证明block-rank
改进发生在graph内部occupancy，不是E2E偶合。产物：
`/tmp/pypto-qwen-native-delay-v6-block-bracket2-npu1`、
`/tmp/pypto-qwen-native-delay-v6-block-profile-npu1`；plan SHA256
`508b8228de14649c07f12d3aff63db178707d6a8e4740d87b5289b534d847273`。

新增`analyze_native_delay_profile.py`逐`program + func_id + subslot + block-rank`比较plan target与
level-4 observed duration。V6共58 callable、42,593 block sample、0 missing；overall MAE/MAPE
`0.184 us/2.78%`，decode `0.113 us/0.50%`，prefill `0.241 us/4.57%`。Top-k MAPE为
`22.82%`，但MAE仅`0.624 us`、最大绝对误差`5.48 us`；约1 us小block的固定开销会放大百分比，
应与top-k graph仅约`6 us` makespan差异一起解释。

**内存结论**：Request kernel内部weight/KV/HBM访问已包含在physical-block duration中，但尚无独立
`{operation, bytes}`热路径模型。高层weight/KV capacity/materialization/lifetime仍由memory
sidecar负责；V6 bracket时旧memory manifest仍non-strict。不得用本轮E2E通过替代strict memory
验收，也不得外推到未测shape、KV point或HCCL通信。

## 轮次 26：Exact Memory Capacity 与 Strict Smoke

**Native evidence**：在physical NPU 1连续执行M1/M2/M3三个独立workload，均得到weight
`29,540,610,048 bytes`、KV `921 * 20,971,520 = 19,314,769,920 bytes`、total
`65,452,113,920 bytes`和simpler committed `39,155,604,908 bytes`。Reserved bytes为
`422,531,668/421,958,228/421,950,036`。未发生外部占卡或错误。

**模型**：`/tmp/pypto-qwen-memory-profile-v2-strict.json`的weight、KV、footprint和顶层均为
`strict_ready=true`。Weight中位duration `38.375 s`，KV中位duration `31.008 s`，SHA256
`8667f25f0305e7e28ee05eaad88a7f4771134284e2ee604712e656a14cb73c78`。

**失败门禁 1**：旧footprint只保存`reserved = peak_non_kv - simpler_committed`，再加到sim当前
committed，因虚拟materialization的committed差异算出922 pages。单点921-page模型正确拒绝外推。

**修复 1**：Manifest保存exact `native_total_bytes`、`native_free_bytes`、
`driver_visible_non_kv_bytes`和`native_simpler_committed_bytes`。Sizing直接返回native exact
`(free,total)`；exact profile存在时环境变量不能覆盖capacity。

**失败门禁 2**：921-page分配完成后，breakdown查询把含KV的总committed与分配前non-KV footprint
直接比较并拒绝。

**修复 2**：首次sizing记录simulator non-KV baseline；后续`committed - baseline`作为逻辑KV
allocation delta，device memory info返回`used = native_non_kv + delta`。这样allocation、free-memory和
lifetime都保留，不隐藏KV占用。

**最终结果**：`/tmp/pypto-qwen-strict-memory-smoke-v3`成功，921 pages，逻辑used/free
`58.89/6.56 GB`，weight/KV phase约`38.376/31.008 s`，58/58 telemetry，输出`[2813] * 5`。
Request kernel内存访问仍内含于block latency，不等于独立热路径bytes模型；HCCL仍未覆盖。

## 轮次 27：Native Device-Delay 虚拟 Weight/KV 地址

**缺口**：V6高精度device-delay只替换child kernel，weight和KV仍由native `rtMalloc`提交HBM；
`58.89 GB`是该旧路径的物理占用，未满足“高精度控制面和大内存虚拟化必须同时成立”的契约。
历史V6 `<1%`结果因此不能作为低HBM路径的验收结果。

**实现**：新增`virtual_memory_interposer.cpp`，由launcher作为`LD_PRELOAD`库构建。Python hook在每个
已分类的static weight、profile scratch KV和final KV allocation前写入one-shot
`{operation, exact_bytes, request_id, forbidden_pid}` token。Forked chip worker的`rtMalloc`只有精确
命中token才调用`aclrtReserveMemAddress`保留2 MiB对齐VA，不调用`rtMalloc`或
`aclrtMallocPhysical`；`rtFree`调用`aclrtReleaseMemAddress`。Parent PID不能消费token，任一错位、
未消费、bytes不匹配或reserve错误均fail closed。

原PyPTO MemoryAllocator仍以完整逻辑bytes登记该VA，caller allocation table、canonical identity、
provenance、borrow、early dispatch、in-flight free和`committed_device_memory`均保留。Weight/KV不再
H2D/zero materialize，但materialization phase仍等待strict profile中的`38.375/31.008 s`。
Launcher强制传入strict memory profile和memory audit；audit要求weight/KV exact aggregate且每个
allocation的`physical_backing_bytes=0`。

**无卡验证**：interposer以真实动态链接边界完成fake-runtime smoke：`rtMalloc(4096)`返回
`0xffff9b040000`保留VA、reservation为2 MiB、`rtFree`成功，control record记录正确consumer PID。
实际`libhost_runtime.so`已确认动态引用`rtMalloc/rtFree`。

**硬件失败门禁 1**：首次NPU 2 smoke启动时为空，但外部`VLLM::Worker_TP`随后抢占57.4 GB；本进程
增长4 GiB arena时OOM。Telemetry记录两个NPU PID和合计61.9 GB peak，该结果作废且未干预外部PID。
失败前16个weight allocation和2个scratch KV已经全部虚拟化。

**硬件失败门禁 2**：第二次无污染smoke完整通过profile warmup，但driver只看到10.02 GB physical
non-KV，而logical committed 39.16 GB未包含421.96 MB driver/runtime reserved，KV sizing算出941 pages。
Strict单点正确拒绝`19,734,200,320 != 19,314,769,920 bytes`。

**修复**：native device-delay的`device_memory_info`使用双视图。Physical driver used继续用于检测外部
占用；logical used取exact `driver_visible_non_kv_bytes + (committed - baseline)`与physical used的最大值。
这保留外部占用收紧能力，同时恢复exact 921-page capacity。

**Zero-duration通过**：第三次NPU 2 smoke完成58/58 replacement、921 pages和`[2813] * 5`。
Weight `29,540,610,048`、scratch KV `20,971,520`、final KV `19,314,769,920 bytes`均为
`physical_backing_bytes=0`。100 ms `npu-smi`采样只有一个PID，无污染，物理peak为`9,312 MB`，结束后
NPU无残留进程。产物：`/tmp/pypto-qwen-native-delay-v7-vmem-zero3-npu2`。

**正式V7 bracket**：同一physical NPU 2按3 warmup + 5 measure执行
`native-A -> low-HBM device-delay -> native-B`：

| Case | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| native-A | 216.528 | 216.576 | 218.354 |
| low-HBM device-delay | 217.503 | 217.122 | 219.319 |
| native-B | 218.848 | 218.569 | 219.913 |
| midpoint | 217.688 | 217.572 | 219.134 |
| error | -0.085% | -0.207% | +0.085% |

Native drift为`+1.071%/+0.920%/+0.714%`。15个measure均为5 token且finish reason为length；simulator
5个请求均为`[2813] * 5`。产物：`/tmp/pypto-qwen-native-delay-v7-vmem-bracket-npu2`；
`bracket_summary.json` SHA256为`3d7948a97977d4afb7823e0a0d669783bb4aac55c3213270659a23dfa4efa709`。
Plan和memory SHA256仍为
`508b8228de14649c07f12d3aff63db178707d6a8e4740d87b5289b534d847273`和
`8667f25f0305e7e28ee05eaad88a7f4771134284e2ee604712e656a14cb73c78`。最终全量回归`46 passed`，
Ruff、尾随空白、artifact算术和interposer `-Wall -Wextra -Werror`均通过；NPU 2无残留进程。

## 轮次 28：独立 A2/A3Sim Timewarp V1

**动机**：Sleep保持完整simpler控制流，但每个微秒级child使用wall sleep，CPU scheduler和wakeup floor
会主导无卡墙钟。Timewarp必须作为sleep之上的独立加速层，不能改变sleep plan、hook或验收结果。

**实现**：新增`pypto-timewarp`。Plan builder从sleep plan引用的normalized profile读取task predecessor、
`block_num`、subslot和block-rank duration，按AIC/AIV/MIX资源做deterministic list scheduling，生成
program frontier。独立installer复用sleep已验证的callable替换、strict coverage、控制输出和内存契约，
但生成无wall delay的child stub。Worker按`dp_rank:step_id`记录L3 program，encoded `StepResult` seal后，
engine在处理token前通过Unix socket TimeKeeper推进共享
`CLOCK_MONOTONIC + non_decreasing_global_offset`；producer-to-observer gate阻止后续jump越过当前token
时间戳读取。Streaming benchmark只在显式clock环境变量存在时读取虚拟时钟，默认和sleep仍使用
`time.perf_counter()`。

**隔离证明**：`install_sleep_hook.py`和`run_sleep_simulator.py`未修改；sleep正duration stub的wall-sleep
测试继续通过。Timewarp plan拒绝sleep launcher语义，sleep plan输入保持`execution_mode=sleep`不变。

**无卡验证**：新增8个timewarp单测，覆盖dependency/resource schedule、block count、pipelined step
frontier、delivery jump、producer-to-observer gate、显式benchmark clock、no-wait stub和plan隔离；全skill
回归为`59 passed`，Ruff、`py_compile`和`diff --check`通过。旧Qwen a2a3sim plan经显式request role编译
得到decode/prefill/top-k frontier `29.180/53.802/0.247 ms`，task数`11117/1124/16`，plan SHA256为
`0a29769fdf3e1c4758f10bf088b2f4c2f4d2cc8b189f3189d040d0c1c3085403`。这些是旧V3模型的离线
schedule审计，不是sleep-vs-timewarp或native精度证据。

**待验证**：NPU 2仍被外部任务占用；本轮没有中断后台空卡轮询，也没有执行hardware bracket。后续先做
`sleep-A -> timewarp -> sleep-B`流式矩阵，验证每case/global TTFT、TPOT、E2EL mean/p50/p95与completion
order，再继续native bracket。V1尚未把virtual predecessor frontier在线写回simpler scheduler slot，
因此必须以sleep bracket而不是plan makespan自证正确。

## 轮次 29：Ordinal Bootstrap 与 Batch16 Profile 补点

**问题**：`--capture-dispatch prefill_fwd=2` 的 bootstrap dispatch 只有 `.pto/.cpp`，没有可执行
child `kernel_config.py`，worker 在目标 ordinal 前失败。

**修复**：profile hook 在显式 profile 进程内 patch `KernelCompiler.compile`，使保存到 artifact root
的子构建使用 `codegen_only=False`；首次 dispatch 仍关闭 DFX，目标 ordinal 才启用 level-4。普通
serving、native、sleep 和 timewarp 不经过该 patch。新增回归验证 executable compile config，profile
launcher 单测和完整相关测试通过。

**结果**：NPU 2 在连续空闲门禁后完成三份独立 batch16 prefill capture，均为 `returncode=0`，每份有
`deps.json`、level-4 swimlane 和 dispatch identity。每份为 `active_batch=16,total_tokens=256`，对应
16 个并发、16 input token 的 request feature。三份合并后 latency model 为 76 groups、97 feature
points、0 low-confidence，batch16 point 每点 3 个独立 workload；model SHA256 为
`9504b76a243aca35da71792a1ca4f9c3574a02441ab0108820ac589045a23a07`。由于旧三份采集没有运行期 PID
监测，暂只能作为候选证据；profile launcher 已增加可选 `--npu-device` 运行期 `proc-mem` 监测，发现
多 PID 或 telemetry error 时自然结束后将 collection 标为失败，不使用 host PID namespace 归属判断。

## 轮次 30：Timewarp Runtime Smoke 与 Strict Deadline Gate

**结果**：旧 serial plan 的无卡 Qwen timewarp smoke 在 pool runtime 下执行 `58/58` modeled stubs，
0 unmodeled，6 graph events、4 completed steps，输出 `[2813,2813]`；cache-only pool runtime 的
2-token diagnostic 为 `TTFT=89.419 ms, TPOT=79.684 ms, E2EL=169.104 ms`。对应 trace 显示部分
step deadline 已被 host control plane 越过，不能解释为已消除 wall overhead。16-thread runtime 更慢，
`TTFT=219.060 ms, TPOT=252.761 ms, E2EL=471.821 ms`。

**修复**：TimeKeeper 新增 `deadline_overrun_count/total_ns/max_ns`。strict timewarp launcher 发现任一
带 graph step 的 modeled deadline overrun 就保留 snapshot/trace 并失败；`--allow-deadline-overrun` 仅
用于诊断，不能进入精度验收。共享时钟仍严格遵循
`CLOCK_MONOTONIC + non_decreasing_global_offset`，没有负 offset、固定 E2E 减法或请求时间戳缩放。

**Batch16 结论**：使用三份新 profile 生成的 batch16 plan 通过 strict coverage，prefill/decode/top-k
frontier 为 `368.150/28.392/0.247 ms`。真实 warmup 在 17,939-task prefill graph 处触发 custom
sim runtime child `SIGABRT`，没有 request step 或输出，因此 batch16 timewarp 仍未通过；该失败保留在
`/tmp/pypto-qwen-timewarp-batch16-v1`，不能用 serial plan 或固定倍率替代。完整 skill 回归为
`62 passed`，Ruff、`py_compile` 和 `diff --check` 通过。
