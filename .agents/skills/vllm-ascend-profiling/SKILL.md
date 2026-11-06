---
name: vllm-ascend-profiling
description: >-
  指导 vLLM-Ascend 推理的 profile 采集与调优迭代：采集覆盖单请求或连续多请求场景下的服务端
  --profiler-config 配置、/start_profile 与 /stop_profile 采集窗口、torch_npu 解析和 .csv
  结果校验（benchmark 场景的采集范式见 references）；调优迭代基于 vllm-ascend-profile-analyze
  给出的建议，一次只更改一个配置，重新部署并按冻结 benchmark 复测、采集下一轮 profile。
---

# vLLM-Ascend Profiling

本 skill 指导 vLLM-Ascend 推理的 profile 采集，并把 profile 调优组织成可复测的闭环：
固定 benchmark，采集并分析 baseline profile，基于
[vllm-ascend-profile-analyze](../vllm-ascend-profile-analyze/SKILL.md) 给出的建议一次只
更改一个配置，重新部署、用同一 benchmark 复测并采集下一轮 profile。

benchmark 场景（含调优复测）的采集窗口范式见
[references/benchmark-profiling.md](references/benchmark-profiling.md)。

每次调用只针对一个服务实例。PD 分离、多机等多实例场景由调用方编排多次调用，每次使用独立的
`profile_dir`。

## profile 采集

### 适用范围

- 服务端采集配置：启动命令中的 `--profiler-config` 和提前创建的 `profile_dir`；
- 调用 `/start_profile`；
- 调用 `/stop_profile`；
- 在本次 profile 目录中找到新生成的 `*_ascend_pt` 目录；
- 调用 `torch_npu.profiler.profiler.analyse` 完成解析；
- 验证解析之后存在 `.csv` 结果文件。

### 必需输入

调用方必须提供本次服务实例的实际值：

```text
service_url: http://<host>:<port>
served_model_name: <name>
profile_dir: <absolute path created before service startup>
```

其中：

- `service_url` 必须是本次服务实例的地址；
- `served_model_name` 标识本次被采集的模型；
- `profile_dir` 必须是启动命令中 `torch_profiler_dir` 使用的目录；
- `profile_dir` 必须属于本次 profiling，不能使用旧的日期目录。

### 执行流程

#### 1. 服务端采集配置

启动服务前创建本次 `profile_dir`，并在服务启动命令中加入：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"<profile_dir>","torch_profiler_with_stack":false}'
```

`torch_profiler_with_stack=false` 关闭 vLLM-Ascend 默认开启的 stack 采集，减少额外开销；
除非用户明确要求，不要打开。

#### 2. 开始采集

服务通过基本验收后，开始采集：

```bash
curl --fail --silent --show-error \
  -X POST <service_url>/start_profile
```

#### 3. 停止采集

采集窗口内的真实推理负载由调用方发送；没有负载时可能不会生成有效数据。负载完成后停止采集：

```bash
curl --fail --silent --show-error \
  -X POST <service_url>/stop_profile
```

`/stop_profile` 失败时，本次 profiling 失败，保留 profile 目录供诊断。

#### 4. 定位和解析结果

停止采集后，只在本次 `profile_dir` 下查找新生成的 `*_ascend_pt` 目录。不要使用其他日期目录或历史解析结果。

找到目录后执行：

```bash
python -c 'from torch_npu.profiler.profiler import analyse; analyse("<ascend_pt_dir>")'
```

`analyse` 成功完成后，验证解析之后存在 `.csv` 结果文件：

```bash
find <ascend_pt_dir> -name '*.csv'
```

不存在任何 `.csv` 文件时，本次 profiling 失败。

### 成功条件

只有以下条件全部满足时，本次 profiling 才算成功：

- `/start_profile` 成功；
- `/stop_profile` 成功；
- 本次 `profile_dir` 下存在新生成的 `*_ascend_pt` 目录；
- `analyse` 成功完成，且 `*_ascend_pt` 目录下存在 `.csv` 结果文件。

profiling 产物不是服务验收证据。profiling 失败时，不修改调用方报告，也不把已成功的服务验收改成失败。

未启用 profiling 时，调用方不创建 `profile_dir`，也不调用本 skill。

## 调优迭代

按 [vllm-ascend-profile-analyze](../vllm-ascend-profile-analyze/SKILL.md) 给出的建议迭代调优：
固定 benchmark，每次只改一个配置，重新部署、复测并采集下一轮 profile。服务部署使用
[vllm-ascend-deployment](../vllm-ascend-deployment/SKILL.md)，benchmark 测量使用
[vllm-ascend-benchmark](../vllm-ascend-benchmark/SKILL.md)。结论必须能追溯到 profile
文件、实际命令和 benchmark 指标；没有证据时只输出待验证假设。

### 输入契约

开始前收集并原样记录，缺少关键项时先补齐，不用宿主默认值猜测：

1. **benchmark 配置**：本次会话内冻结的 benchmark 参数（数据集、请求数、warmup、并发或到达率、轮次、seed、生成参数、主要关注性能指标项、profile 触发方式和采集时长）；只有待验证的启动参数或代码 patch 可变，其余任何变化都必须经过用户审核；
2. **服务与启动事实**：实际 `vllm serve` 启动命令、模型路径、可见设备、TP/DP/EP 设置、环境变量和服务 URL；
3. **代码与模型**（融合方向必需）：vLLM 和 vllm-ascend 源码根目录、模型 `config.json`；
4. **输出目录**：本次调优专属绝对路径，每个迭代使用独立子目录，禁止混用历史 profile 或 benchmark 结果。

信息不全时先生成“待确认 benchmark 配置草案”，未确认前不得采集正式 profile 或宣布收益。

### 迭代流程

1. **固定 benchmark（第 0 步，强制）**：第一段输出必须是“Benchmark 调优配置（待确认）”，
   不得先给 profile 结论或先改启动参数。按输入建立 `<tuning_dir>/benchmark_config.json`，
   用户确认后计算摘要哈希——它是本次会话唯一的测量合同；先按该配置完成 baseline benchmark
   并保存指标，不合格时停止调优。
2. **采集 baseline profile**：按 [references/benchmark-profiling.md](references/benchmark-profiling.md)
   的范式绑定采集窗口（warmup 在 profile 外、首个正式请求触发、固定时长），按上文的采集流程
   完成服务端配置、开始/停止采集和解析校验。
3. **调用 profile-analyze**：把 profile 交给 profile-analyze 完成分析并拿回建议，只记录其
   输入、输出和选定的单一候选，不重复解释分析规则。
4. **单变量变更与复测**：从建议中只取一个参数（或一个紧密耦合参数组，或一个融合 patch）修改，
   重新部署服务，严格复用同一 benchmark、同一触发范式和同一采集时长复测并采集下一轮 profile；
   主指标（按输入契约冻结的主要关注性能指标项）改善，但其余指标出现较为严重的劣化时，候选失败。
5. 回到第 3 步，同一合同下继续取候选，直到试过的候选达到用户指定次数或建议耗尽。

### 结果判定与交付

每个迭代目录保存：实际启动命令、profile 目录、profile 分析报告、候选变更/patch、benchmark 结果和结论。最终报告 `tuning_report.md` 汇总每次迭代的唯一变更和指标对比、最终保留的配置/patch、回退步骤和未解决风险。候选在质量不回退且主指标改善明确超过用户阈值或测量噪声时保留；用户未给阈值时，报告绝对值和相对变化，不自行宣称“有效”。

### 必须停止的情况

benchmark 合同被破坏（配置未确认、摘要或 baseline 不可复现、请求质量校验失败）或 profile 采集不满足上文的成功条件时停止；停止时交付已收集证据和下一步验证条件，不通过猜测补齐结论。
