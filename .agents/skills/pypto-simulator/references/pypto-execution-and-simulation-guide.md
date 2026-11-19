# PyPTO 执行链深度参考与 Qwen 案例

本文是面向首次接触 PyPTO、Ascend 和本项目读者的深度解释及当前 Qwen 案例记录，不是通用
schema、launcher 或验收规则的唯一来源。通用边界以根目录 `README.md`、`references/architecture.md`
和 `references/manifest-contract.md` 为准；模型特有内容以对应 `references/<model>-adapter.md`
为准。

目标是回答四个问题：

1. 用户提交一个文本请求后，系统依次做了什么？
2. PyPTO、simpler、AICPU、AIC/AIV 分别负责什么？
3. 真实 NPU 执行和 `a2a3sim` 执行在哪里分叉？
4. 当前 sleep simulator 保留了什么、替换了什么，误差又来自哪里？

文中的英文名称多数是代码、ABI 或命令中的原名，因此保留英文并在首次出现时解释。

## 1. 先建立一个最小心智模型

可以先把整个系统看成一家执行计算订单的工厂：

```text
用户请求
  -> pypto-serving：决定这次要做 prefill 还是 decode，准备输入
  -> PyPTO program：描述模型计算和并行方式
  -> PyPTO 编译器：把 program 拆成控制程序和很多可调度 task
  -> simpler runtime：根据依赖关系调度 task
  -> AIC/AIV：执行真正的矩阵或向量计算
  -> completion：通知后继 task 可以继续
  -> sampling：选择下一个 token
  -> 返回生成结果
```

各部分最重要的职责是：

| 部分 | 最简单的解释 | 不负责什么 |
| --- | --- | --- |
| `pypto-serving` | 接收请求、组batch、管理KV cache、调用模型 | 不直接调度每个NPU task |
| PyPTO DSL/program | 用Python描述模型计算、循环和设备并行 | 不直接决定每个task何时运行 |
| PyPTO编译器 | 把program变成host控制程序和设备kernel | 不处理在线请求队列 |
| simpler runtime | 建依赖、调度、dispatch、等待completion | 不决定LLM业务语义 |
| AIC | 适合矩阵乘等Cube计算的执行槽 | 不负责请求调度 |
| AIV | 适合逐元素、归一化等Vector计算的执行槽 | 不负责请求生命周期 |
| AICPU | 运行设备侧控制和task scheduler | 通常不执行大矩阵计算 |
| `a2a3` | 在真实Ascend NPU上执行上述链路 | 不是CPU模拟器 |
| `a2a3sim` | 在CPU进程中模拟A2/A3 runtime和task调度 | 不是性能等同的真实NPU |

当前模拟器不是“收到请求后直接sleep一次”。它仍让原始PyPTO控制程序建立原始task图，再在每个
被调度的计算task进入执行体时，用延迟stub代替真实计算。

## 2. 一次LLM请求包含哪些阶段

以prompt `Huawei is`、生成5个token为例。

### 2.1 模型初始化

请求到来前，serving先做一次初始化：

1. 加载Qwen权重到CPU内存。
2. 编译或读取PyPTO的prefill、decode和top-k程序。
3. 创建共享的L3 worker。这里的L3指PyPTO较高层的整图调用层，不是CPU缓存。
4. 把静态模型权重上传到设备内存。
5. 运行一次warmup，让runtime arena、ring和临时buffer完成分配。
6. 根据剩余设备内存计算可分配多少KV page。
7. 分配KV cache。

权重保存模型参数。KV cache保存attention已经计算过的Key和Value，避免生成每个新token时重新计算
全部历史token。

### 2.2 Prefill

Prefill一次处理prompt中的全部token。它主要完成：

- token embedding；
- 多层attention和MLP；
- 把prompt对应的K/V写入KV cache；
- 生成最后位置的logits。

`logits`可以理解为“词表中每个候选token的分数”。

### 2.3 Sampling

Sampling根据logits选择下一个token。当前固定workload使用greedy，即选择分数最大的token；实际
Qwen路径使用`topk_select_fwd`完成设备侧候选选择。

### 2.4 Decode循环

生成剩余token时，每轮decode只输入刚生成的一个token，并读取已有KV cache：

```text
第1轮decode -> 生成token 2
第2轮decode -> 生成token 3
第3轮decode -> 生成token 4
第4轮decode -> 生成token 5
```

对于本项目的5-token请求，执行链大致是一次request prefill、一次top-k，以及若干次decode。具体
次数由serving的集成sampling流程决定，验证telemetry以实际dispatch为准。

## 3. PyPTO program是什么

PyPTO允许开发者用Python DSL描述NPU计算。代码看起来像Python，但其中的tensor操作、循环和
设备调用不是立即用普通Python算出结果，而是被记录和编译。

一个完整program通常同时包含两类逻辑：

- **Host orchestration**：控制任务提交、循环、设备选择和tensor连接。
- **InCore kernel**：真正运行在AIC或AIV上的计算函数。

编译器会把InCore函数从控制代码中“outline”出来。Outline可以理解为：把一段设备计算抽成一个
可独立调度的函数，并给它一个`func_id`。

因此编译后不是一个不可分割的大kernel，而是：

```text
一个 orchestration function
  + 多个 outlined kernel
  + 每个kernel的参数签名和binary
  + runtime所需的metadata
```

Qwen当前有三个主要program：

| Program | 用途 |
| --- | --- |
| `prefill_fwd` | 处理prompt并写入初始KV cache |
| `decode_fwd` | 每轮读取KV cache并生成新logits |
| `topk_select_fwd` | 从logits中选择候选token |

## 4. 编译链做了什么

从Python program到可运行产物，大致经过以下步骤：

```text
PyPTO Python DSL
  -> PyPTO IR（中间表示）
  -> 分离host orchestration和InCore kernel
  -> 生成task/callable信息
  -> PTOAS编译AIC/AIV kernel
  -> 生成simpler runtime产物
```

几个术语：

- **IR**：Intermediate Representation，中间表示。编译器内部使用的结构化程序。
- **PTOAS**：把PyPTO的设备计算编译成Ascend目标代码的工具链。
- **binary**：编译后的可执行字节。在native路径中供NPU执行，在sim路径中可能是CPU共享库。
- **callable**：runtime可以调用的编译产物。一个parent callable带有多个child kernel。
- **`func_id`**：child kernel在一个program中的数字标识。

当前simulation plan按`program + func_id`匹配runtime执行体。例如：

```text
decode_fwd + func_id 17 -> 对应一个确定的delay stub
```

只用`func_id`不够，因为不同program中可能存在相同数字。

## 5. Tensor、地址和内存

### 5.1 Host tensor和device tensor

- **Host tensor**：数据在CPU可访问内存中。
- **Device tensor**：数据在NPU设备内存中，host通常只持有地址、shape和dtype等描述。
- **Static tensor**：权重等初始化一次、反复复用的数据。
- **Transient tensor**：只在一次dispatch附近临时存在的数据。
- **Control tensor**：sequence length、block table、slot mapping、sampled ID等控制执行的数据。

### 5.2 Tensor descriptor

Runtime传递kernel参数时，不只传一个裸地址，还要传tensor描述信息，例如：

```text
数据基地址
shape
stride
dtype
offset
```

Sleep stub虽然不做真实模型计算，但sampling producer必须写合法输出。因此它会读取同一份descriptor，
找到`sampled_ids[row][0]`，写入合法常量token，并清零padding。

### 5.3 Weight和KV cache

真实NPU路径会：

```text
分配真实设备内存 -> 搬运或初始化全部字节 -> kernel读写这些字节
```

当前simulator的Qwen虚拟memory路径会：

```text
执行原生allocation调用
  -> 获得唯一、稳定、可做指针偏移的逻辑地址
  -> 保留地址登记、borrow/free和内存预算
  -> 大weight/KV不触碰全部真实backing，也不做全量copy
  -> 在原materialization phase内部补足测得延迟
```

小型控制tensor和scratch仍有真实backing，因为host控制程序或stub可能需要读写它们。

当前pinned Qwen实际static materialization参数总计`29,540,610,048 bytes`。Serving日志中的
`27.98 GB`只是按模型结构估算的未padding参数量，不能作为精确上传字节数。

## 6. Task graph是什么

### 6.1 Task

Task是simpler scheduler看到的可调度工作单元。它比“整个模型”小，但比kernel内部的一条加法指令
大。当前拦截边界就在outlined simpler task对应的child kernel执行体。

一个task通常包含：

- 稳定task identity；
- 调用哪个`func_id`；
- 输入和输出tensor；
- 前驱task；
- 需要AIC还是AIV；
- 需要多少block；
- 是否允许early dispatch。

### 6.2 依赖

如果task B读取task A刚写的数据，B必须等A完成。这就是依赖边：

```text
A -> B
```

常见依赖类型包括：

- RAW：Read After Write，先写后读；
- WAW：Write After Write，两次写同一位置必须有顺序；
- 显式`deps=`：程序员或编译器明确指定的依赖。

### 6.3 TensorMap

TensorMap是runtime用于连接tensor生产者、消费者和依赖关系的重要结构。可以把它理解为一张
“哪块数据由谁产生、谁要读取”的账本。它参与依赖建立和完成传播，不是模型权重本身。

当前simulator保留TensorMap操作。否则即使总sleep时间相同，也无法模拟并行、fanin/fanout、
early dispatch和资源冲突。

### 6.4 Block和SPMD

一个logical task可以拆成多个block并行执行。SPMD表示多个执行实例运行同一份程序，但处理不同
数据分片。

例如一个task有20个block，并不表示有20个不同算法；通常表示同一个kernel由20个执行位置共同
处理不同tile。

### 6.5 MIX task

MIX task同时需要Cube和Vector侧工作。Profile会分别记录：

```text
AIC subslot
AIV0 subslot
AIV1 subslot
```

它们有各自的执行时间和完成条件。当前计划保留这些subslot信息，而不是把MIX task粗暴压成一个
不可解释的整图sleep。

## 7. 谁在调度task

simpler runtime负责把已经满足依赖的task送到合适执行资源。

### 7.1 AICPU角色

本项目正式拓扑使用4个AICPU scheduler角色。它们运行设备侧控制逻辑，主要负责：

- 检查依赖是否满足；
- 从队列或ring中取task；
- 把block dispatch到AIC/AIV角色；
- 接收completion；
- 唤醒后继task；
- 判断整张graph是否完成。

AICPU不是当前大矩阵计算的主要执行者。

### 7.2 60个AICore角色

Pinned NPU和profile使用20-cluster拓扑。每个cluster有三个被runtime区分的执行slot：

```text
20 × AIC  = 20
20 × AIV0 = 20
20 × AIV1 = 20
合计       = 60个AICore执行角色
```

这里的role是一个逻辑执行位置，不是请求、task或scheduler线程。它接收scheduler分配的block，执行
child kernel，然后发布completion。

CPU simulator还为4个AICPU角色提供worker，所以persistent worker pool通常有64个OS worker：

```text
60个AICore worker + 4个AICPU worker
```

Persistent只表示OS线程跨graph复用；每轮graph中的逻辑role、task、依赖和completion没有删除。

## 8. 一次graph如何执行

把前面的概念连起来，一次`decode_fwd`大致经历：

1. `pypto-serving`根据active request构造decode batch。
2. Model runner更新token ID、sequence length、block table和slot mapping等控制tensor。
3. Runner调用编译后的`decode_fwd` callable。
4. Runtime绑定host tensor、device tensor、weight和KV地址。
5. Host orchestration开始运行并提交task。
6. TensorMap和显式依赖形成task graph。
7. AICPU scheduler寻找依赖已经满足的task。
8. Scheduler把task block dispatch给AIC/AIV角色。
9. AIC/AIV执行kernel。
10. 每个block或subslot完成后发布completion。
11. Scheduler更新依赖计数，继续dispatch后继task。
12. 所有terminal条件满足后，`run()`返回。
13. Serving读取小型sampling结果，更新request状态。
14. 如果还需token，进入下一轮decode。

这里的`run()`边界只是等待整张graph自然完成。当前simulator没有把`run()`替换成一次固定sleep。

## 9. `a2a3`和`a2a3sim`

### 9.1 `a2a3`

`a2a3`是真实Ascend A2/A3硬件backend：

```text
AIC/AIV kernel -> 真实NPU执行
设备内存       -> 真实HBM
completion     -> 真实设备执行完成后产生
```

Native profile和正式A/B baseline必须使用该路径，且不得加载任何delay hook。显式device-delay模式
也使用`a2a3`平台，但它是独立的hardware-in-the-loop实验，不是native真值。

### 9.2 `a2a3sim`

`a2a3sim`是simpler的CPU simulator backend。它在host进程中模拟A2/A3 runtime：

- host orchestration仍执行；
- TensorMap和task graph仍建立；
- AICPU scheduler仍运行；
- AIC/AIV角色由CPU线程表示；
- task仍按依赖dispatch和complete；
- kernel binary是CPU可加载形式，而不是真实NPU指令执行。

因此它更接近“CPU上运行一套NPU调度模型”，而不是“模拟每个晶体管”或“整图估时器”。

### 9.3 为什么CPU simulator本身也会耗时

Native NPU可以真正并行执行很多AIC/AIV block。当前host允许进程使用`0-191`共192个CPU；环境中的
`OMP_NUM_THREADS=1`只影响OpenMP默认并行度，不能解读为cgroup只给一个CPU。即便CPU数量足够，
60个AICore角色和4个AICPU角色仍会产生以下host墙钟开销：

- scheduler扫描；
- 原子变量和队列操作；
- OS线程唤醒；
- task dispatch和completion；
- graph startup和retirement；
- timer sleep和唤醒误差。

这就是为什么“所有kernel duration设为0”时，simulator仍然不是0毫秒。

### 9.4 `a2a3` device-delay是什么

Device-delay继续使用真实NPU的AICPU scheduler、TensorMap、dispatch、ACK、FIN和completion，只把
最终AIC/AIV child binary替换为同ABI delay kernel。Delay kernel由CCEC编译，运行在真实AIC/AIV
上，并通过设备系统计数器等待目标tick。

它避免真实业务数学计算，也避免`a2a3sim`的CPU线程调度误差，但仍需要真实NPU、设备内存和执行
槽。因此它适合作为高保真参考，不是最终无NPU simulator。

## 10. 当前simulation pipeline

当前方案分成离线和在线两部分。

### 10.1 Native profile

先在真实NPU上采集：

```text
同一PyPTO/simpler/PTOAS版本
  -> dep-gen导出静态task图
  -> level-4 timing记录真实task/subslot时间
  -> dispatch identity提供独立workload ID
```

`deps.json`是task和依赖的静态真源；timing记录提供duration。每个feature point要求独立native
workload，不能用同一图中的重复layer冒充重复实验。

### 10.2 Latency model

Fitter按task signature聚合样本。Signature包含program、task name、资源类型、block数量和已有证据
支持的动态feature。

当前使用：

- exact point取独立workload中位数；
- 只允许测量域内插值；
- 禁止外推；
- low-confidence点不能进入strict plan。

### 10.3 Simulation plan

Planner把每个graph node分类：

- `simulate`：真实计算、通信或weight/KV访问，需要替换执行体；
- `preserve`：同步、metadata和控制节点，保留原语义；
- `missing_model`：应模拟但没有可信模型，strict模式直接失败。

最终runtime替换面按`program + func_id`聚合，同时保留task级预测供审计。

### 10.4 Host Sleep Hook

Hook只在`a2a3sim`安装。它保留parent orchestration binary，把child `CoreCallable`换成stub：

```text
原始scheduler dispatch child kernel
  -> stub记录首次执行telemetry
  -> 等待modeled duration
  -> 必要时写sampling控制输出
  -> 正常返回
  -> 原始scheduler处理completion
```

Plan外child一旦执行会`abort()`，没有静默零时长fallback。普通硬件`a2a3`命令完全旁路该hook。

### 10.5 Native Device-Delay Hook

该hook只由独立launcher在`backend=a2a3`的strict plan上安装。原始`ChipCallable` parent、签名、
config和runtime保持不变；58个modeled child替换为设备delay binary。编译但未建模的备用func_id不
注册到callable table，若意外dispatch会在scheduler查址阶段失败，不会执行真实业务kernel。

Qwen zero-duration gate在physical NPU 1完成58/58 replacement并返回5个合法常量token。Zero gate
只证明可达性和控制输出，不能替代非零延迟精度验证。

## 11. Duration是如何等待的

### 11.1 理想语义

最容易理解的理想方案是：每个stub被调用时独立等待完整duration。

问题是大多数kernel只有几微秒到几十微秒。普通OS sleep经常比目标多睡几十或几百微秒，这些误差
沿task依赖链累加，最终比native慢很多倍。

### 11.2 当前per-role cumulative deadline

当前最佳runtime为每个AICore role维护累计deadline：

```text
role开始执行时：deadline = 当前时间
每个stub：       deadline += modeled duration
如果host已经晚于deadline：不再额外等待
否则：           等到deadline
```

这可以避免把同一role上的host scheduler开销和NPU modeled duration重复相加。

但它是明确的语义偏离：stub不一定独立等待完整duration，已经消耗的host wall time会抵扣预算。因此
结果必须标注为cumulative deadline simulation，不能宣称是严格逐stub sleep。

### 11.3 为什么不用统一graph epoch

曾尝试让60个role从同一个graph起点累计deadline。实验中许多role在相近时刻唤醒，decode出现
30到80毫秒的剧烈抖动。三种等待实现均明显回归，因此当前恢复为每role实际开始执行时建立epoch。
Host允许192个CPU，回归不能归因于“只有一个CPU”；timer、线程落点、共享状态和NUMA的占比仍需
继续审计。

### 11.4 Device-Delay等待语义

Device-delay不使用host sleep，也不使用cumulative deadline。每个AIC/AIV child读取50 MHz
`get_sys_cnt()`，独立自旋到profile tick后返回。Task/subslot提供完成上界，block-rank model再通过
`get_block_idx(args)`恢复排序physical-block occupancy distribution；MIX AIV按
`get_sub_block_id(args)`区分lane。这恢复了逐stub完整duration语义并保留真实设备scheduler；代价是
运行期间真实占用NPU执行槽。

## 12. Memory simulation

Memory profile独立记录三个高层操作：

| 操作 | 含义 |
| --- | --- |
| `weight_materialize` | 静态权重逻辑分配和上传phase |
| `kv_materialize` | KV page分配和初始化phase |
| `runtime_reserved_memory` | driver可见但simpler allocator未计入的保留内存 |

Runtime仍按逻辑分配量计算committed和free memory，因此原生KV sizing公式继续运行。任一bytes超出
测量域时立即失败，不能按带宽随意外推。

当前strict manifest由NPU 1的3个独立921-page workload生成：

- weight exact bytes为`29,540,610,048`，中位duration约`38.375 s`；
- KV exact bytes为`19,314,769,920`，中位duration约`31.008 s`；
- native exact total为`65,452,113,920 bytes`；
- driver-visible non-KV为`39,577,563,136 bytes`；
- runtime reserved中位数为`421,958,228 bytes`。

Simulator首次sizing记录自身non-KV baseline，后续committed增量作为逻辑KV allocation计入used/
free。Strict smoke精确得到921 pages和`58.89/6.56 GB` used/free。正式runner仍会拒绝non-strict
manifest；单点bytes不匹配也立即拒绝外推。

## 13. 哪些东西被保留，哪些被替换

| 层面 | 当前处理 |
| --- | --- |
| serving请求队列和生命周期 | 保留 |
| prefill/decode/top-k调用顺序 | 保留 |
| PyPTO host orchestration | 保留 |
| TensorMap、TaskId、依赖 | 保留 |
| MIX/SPMD、block dispatch | 保留 |
| AICPU scheduler拓扑 | 保留固定4-thread |
| 20 AIC + 40 AIV角色 | 保留 |
| startup、completion、retirement | 保留 |
| AIC/AIV真实数学计算 | 替换为delay stub |
| 大weight/KV真实搬运和初始化 | 替换为逻辑分配加phase delay |
| 小型控制buffer | 保留真实读写 |
| sampled token数值语义 | 不保留；写合法常量token维持控制流 |
| 普通native profile/A/B路径 | 完全不修改 |
| 显式`a2a3` device-delay路径 | 保留设备控制面，只替换modeled AIC/AIV child |

## 14. 当前结果应该怎样解读

Physical NPU 1上固定Qwen3-14B workload的正式V6 bracket为：

```text
native midpoint mean/p50/p95 = 214.731/215.344/215.443 ms
device-delay mean/p50/p95    = 215.867/215.734/216.554 ms
error                         = +0.529%/+0.181%/+0.515%
native drift                  = -0.076%/+0.130%/+0.088%
```

V3约8%的正偏来自两个模型错误：warmup task数量覆盖request feature，以及所有physical block统一等待
最慢block时延。Capture-role隔离和block-rank distribution修复后，58/58 callable与8次
`[2813] * 5`均通过。

这项通过只适用于固定request feature和真实NPU device-delay路线。Request kernel中的HBM/weight/KV
访问已内含在physical-block latency中，不是独立的`{operation, bytes}`热路径模型；多卡HCCL也未
覆盖。无NPU的`a2a3sim`仍存在结构性host控制面开销，不能引用device-delay精度作为自身精度。

同图zero-work bracket中，`a2a3sim` p50比`a2a3` native midpoint高
`122.25%–130.60%`。因此`a2a3sim`继续用于control-flow/CI，最终无NPU高精度目标仍需要原生虚拟
时间runtime。

## 15. 看到一个耗时时，先问它属于哪一层

| 现象 | 优先检查 |
| --- | --- |
| 模型加载慢 | 编译、weight materialization、KV materialization |
| 单个graph内部慢 | task duration、依赖、scheduler window、role等待 |
| graph很快但请求慢 | serving队列、staging、future join、request lifecycle |
| top-k明显慢 | logits staging、控制buffer copy、graph startup |
| 第一次快慢不同 | JIT、SO加载、worker启动、cache warmup |
| p95很高 | CPU拓扑/线程落点、线程唤醒、timer oversleep、外部负载 |
| token错误 | sampled output ABI、control tensor、未建模child |
| KV page数不同 | total memory、committed memory、reserved footprint |

不要把所有差值都归到kernel duration，也不要用固定端到端减法掩盖不同层的开销。

## 16. 术语表

| 术语 | 本文中的含义 |
| --- | --- |
| ABI | 两段代码约定的参数布局、符号和调用方式 |
| AIC | 主要执行Cube/矩阵计算的逻辑资源 |
| AIV | 主要执行Vector计算的逻辑资源 |
| AICPU | 运行设备控制和scheduler逻辑的CPU型资源 |
| AICore role | simulator中代表一个AIC或AIV执行位置的逻辑角色 |
| block | 同一kernel的一个并行执行实例 |
| callable | runtime可调用的编译产物 |
| child kernel | parent callable中被outline出的设备计算函数 |
| completion | task或subslot完成的通知 |
| decode | 每次基于一个新token和已有KV cache继续生成 |
| dependency | task之间必须满足的先后关系 |
| dispatch | scheduler把可运行task交给执行资源 |
| duration | 模型预测的设备执行时间 |
| early dispatch | 在允许条件满足时提前提交部分工作 |
| feature point | 一组确定输入特征对应的测量点 |
| `func_id` | 一个program内child kernel的数字标识 |
| graph | 一次program调用形成的task和依赖集合 |
| HBM | NPU的高带宽设备内存 |
| host | 运行Python、serving和host runtime的CPU侧 |
| InCore kernel | 在AIC/AIV执行的计算函数 |
| IR | 编译器内部的中间程序表示 |
| KV cache | attention历史Key/Value缓存 |
| latency model | 根据task signature预测duration的模型 |
| MIX | 同一task同时包含AIC和AIV侧工作 |
| orchestration | 创建、连接和提交task的控制程序 |
| prefill | 一次处理prompt全部token的阶段 |
| PTOAS | PyPTO设备kernel编译工具链 |
| role | 一个逻辑执行位置，不等于task或请求 |
| sampling | 根据logits选择下一个token |
| scheduler | 根据依赖和资源决定task何时、在哪里运行 |
| SPMD | 多个执行实例运行同一程序处理不同数据分片 |
| staging | 在调用边界准备、绑定或搬运tensor数据 |
| stub | 替代真实kernel的简化执行体 |
| subslot | MIX task中的AIC、AIV0或AIV1成员 |
| task | simpler scheduler的可调度工作单元 |
| TensorMap | 跟踪tensor生产消费和依赖关系的runtime结构 |
| telemetry | 证明哪些stub实际执行过的运行记录 |
| worker | 持有runtime、设备资源或执行线程的进程/对象 |

## 17. 对应代码入口

从上到下阅读时可参考：

- Serving Qwen runner：`pypto_serving/model/qwen/npu_runner.py`
- L3 dispatch：`pypto_serving/model/common/runner/l3_dispatch.py`
- PyPTO worker：`pypto/runtime/distributed_runner.py`
- Host graph参数绑定：`runtime/src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp`
- Simulator device runner：`runtime/src/a2a3/platform/sim/host/device_runner.cpp`
- TensorMap scheduler：`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/`
- Sleep hook：`pypto-emulate/scripts/install_sleep_hook.py`
- Profile exporter：`pypto-profile/scripts/export_runtime_profile.py`
- Memory exporter：`pypto-profile/scripts/export_memory_profile.py`
- Plan builder：`pypto-emulate/scripts/build_simulation_plan.py`

更严格的schema和实验数据分别见`manifest-contract.md`和`native-qwen-validation.md`。
