---
name: vllm-ascend-profiling
description: >-
  指导 vLLM-Ascend 推理时的 profiling 采集：服务端采集配置（启动命令加入 --profiler-config、
  提前创建 profile_dir）、/start_profile 与 /stop_profile 采集窗口、torch_npu 解析和
  .csv 结果验证；单机、PD 分离、多机等多实例场景均可复用。
---

# vLLM-Ascend Profiling

本 skill 指导 vLLM-Ascend 推理时的 profiling 采集，覆盖服务端采集配置、采集窗口、解析和
结果验证；每次调用只针对一个服务实例。PD 分离、多机等多实例场景由调用方编排多次调用，
每次使用独立的 `profile_dir`。

## 适用范围

本 skill 负责：

- 服务端采集配置：启动命令中的 `--profiler-config` 和提前创建的 `profile_dir`；
- 调用 `/start_profile`；
- 调用 `/stop_profile`；
- 在本次 profile 目录中找到新生成的 `*_ascend_pt` 目录；
- 调用 `torch_npu.profiler.profiler.analyse` 完成解析；
- 验证解析之后存在 `.csv` 结果文件。

## 必需输入

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

## 执行流程

### 1. 服务端采集配置

启动服务前创建本次 `profile_dir`，并在服务启动命令中加入：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"<profile_dir>","torch_profiler_with_stack":false}'
```

`torch_profiler_with_stack=false` 关闭 vLLM-Ascend 默认开启的 stack 采集，减少额外开销；
除非用户明确要求，不要打开。

### 2. 开始采集

服务通过基本验收后，开始采集：

```bash
curl --fail --silent --show-error \
  -X POST <service_url>/start_profile
```

### 3. 停止采集

采集窗口内的真实推理负载由调用方发送；没有负载时可能不会生成有效数据。负载完成后停止采集：

```bash
curl --fail --silent --show-error \
  -X POST <service_url>/stop_profile
```

`/stop_profile` 失败时，本次 profiling 失败，保留 profile 目录供诊断。

### 4. 定位和解析结果

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

## 成功条件

只有以下条件全部满足时，本次 profiling 才算成功：

- `/start_profile` 成功；
- `/stop_profile` 成功；
- 本次 `profile_dir` 下存在新生成的 `*_ascend_pt` 目录；
- `analyse` 成功完成，且 `*_ascend_pt` 目录下存在 `.csv` 结果文件。

profiling 产物不是服务验收证据。profiling 失败时，不修改调用方报告，也不把已成功的服务验收改成失败。

未启用 profiling 时，调用方不创建 `profile_dir`，也不调用本 skill。
