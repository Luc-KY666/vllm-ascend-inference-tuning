---
name: vllm-serve-bench
description: Use when user requests vLLM online serving performance benchmark on Ascend, covering long-prefill / long-decode / balanced workloads with real datasets
---

## Overview

针对已部署的 vLLM-Ascend OpenAI 兼容服务，使用 `vllm bench serve` 做在线服务性能压测。本 skill 聚焦三种负载场景，每种场景对应一个真实存在的数据集（均来自 vllm / vllm-ascend 官方支持列表）：

- **长 prefill** → `hf` 数据集（`likaixin/InstructCoder`）：真实代码 + 编辑指令，代码天然偏长输入，短输出，压测 prefill 与 TTFT
- **长 decode** → `hf` 数据集（`AI-MO/NuminaMath-CoT`）：真实数学题 + 长 CoT 解答，短输入长输出，压测 decode 与 TPOT/ITL
- **均衡负载** → `sharegpt` 数据集：真实多轮对话，长度按真实分布，压测综合服务能力

**前提条件**：推理服务已部署并运行。**本 skill 只做 vllm bench serve 压测，不负责环境准备、模型部署、精度评测或离线 throughput/latency 测试。** CANN/ATB 加载、vllm-ascend 安装、NPU 可见性等假定已就绪，不在本 skill 范围内。

## 数据集选择依据

| 场景 | `--dataset-name` | 数据来源 | 真实文本 | 长度特征 |
|------|------------------|----------|----------|----------|
| 长 prefill | `hf` | `likaixin/InstructCoder`（HuggingFace） | 真实代码 | 代码天然偏长输入，输出短（默认 200） |
| 长 decode | `hf` | `AI-MO/NuminaMath-CoT`（HuggingFace） | 真实数学题+CoT解答 | 短输入（题）+长输出（解答） |
| 均衡负载 | `sharegpt` | `ShareGPT_V3_unfiltered_cleaned_split.json` | 真实对话 | 真实分布 |

## Workflow

1. 收集服务连接信息 + 验证服务可用
2. 确定负载场景并收集参数
3. 准备数据集
4. 执行 `vllm bench serve`
5. 收集指标并生成报告

---

## Step 1: 收集服务连接信息

**评测执行位置（使用 AskUserQuestion 询问）：**

- header: "执行位置"
- question: "vllm bench 命令在哪台机器上执行？"
- options:
  - "本机执行" — 评测工具装在本机，通过 API 调用推理服务
  - "推理服务所在机器" — 在部署推理服务的同一台机器上执行

选择后：
- **本机执行**：直接运行评测命令
- **推理服务所在机器**：后续命令通过 `ssh <username>@<ip>` 包装执行，需追问 SSH 连接信息（格式：`<username>@<ip>`）

**服务连接信息（必须询问用户确认，不可假设默认值）：**

| 参数 | 说明 | 备注 |
|------|------|------|
| 服务 IP | 推理服务所在机器 IP | 用户说"本地"则为 127.0.0.1 |
| 服务端口 | 推理服务端口号 | 用户说"默认"则为 8000 |
| `model` | 本地模型路径（加载 tokenizer） | 必须是本地路径或 HF ID，不能是 served-model-name；vllm bench 需要本地有 tokenizer 文件 |
| `served_model_name` | API 请求中的模型名 | 必须与部署时 `vllm serve` 的 `--served-model-name` 一致，否则找不到模型 |
| `endpoint` | API 路径 | chat/instruct 模型用 `/v1/chat/completions`；base 模型用 `/v1/completions` |

询问后用以下命令验证服务可访问并确认模型名：

```bash
curl --fail --silent http://<IP>:<PORT>/v1/models
```

返回 JSON 中的 `id` 字段应与 `served_model_name` 一致；不一致时以服务端返回为准或让用户确认。远程服务需确认网络连通性（防火墙、SSH 隧道等）。

**tokenizer 问题处理**：如果 vllm bench 报 tokenizer 相关错误，询问用户本地是否有对应模型路径。如果没有，让用户提供一个本地可用的模型路径（仅需 tokenizer 文件）。

**路径规范化**：结果统一写入 `./running-time/<served_model_name>/bench/`。把 `served_model_name` 作为目录名使用前，先把 `/`、`\\`、`..` 等路径字符替换为 `_`。

---

## Step 2: 确定负载场景并收集参数

**每次执行都必须重新询问场景和参数，禁止复用之前的输入。** 服务连接信息（IP、端口、模型路径、served_model_name）可以复用。

**首先使用 AskUserQuestion 确定负载场景：**

- header: "负载场景"
- question: "请选择压测的负载场景："
- options:
  - "长 prefill" — 大输入小输出，压测 prefill 与 TTFT（InstructCoder 数据集）
  - "长 decode" — 短输入长输出，压测 decode 与 TPOT/ITL（NuminaMath-CoT 数据集）
  - "均衡负载" — 真实对话分布，压测综合服务能力（sharegpt 数据集）

**通用参数（所有场景都询问，使用 AskUserQuestion 展示选项）：**

| 参数 | 说明 | 选项 |
|------|------|------|
| `num_prompts` | 测试请求数量 | 16 / 32 / 64 / 100 / 200 |
| `max_concurrency` | 最大并发数 | 16 / 32 / 64 / 128 |
| `request_rate` | 请求速率（QPS），inf=同时发送所有 | inf / 1 / 4 / 16 |
| `temperature` | 采样温度 | 0（确定性） / 0.1 / 0.7 |

> 快速验证用 16~32；正式基准测试用 100~200。`request_rate=inf` 表示所有请求在 t=0 同时发出，用于压满服务；设具体 QPS 用于固定速率服务测试。

### 长 prefill 场景 → InstructCoder（hf 数据集）

InstructCoder 输入 = 真实代码文件 + 编辑指令，代码天然偏长，输出短（默认 `--hf-output-len 200`）。

| 参数 | 说明 | 推荐值 / 选项 |
|------|------|---------------|
| `--hf-output-len` | 输出长度（tokens） | 64 / 128 / 200 / 256 |

### 长 decode 场景 → NuminaMath-CoT（hf 数据集）

不需指定输入/输出长度——输入是数学题、输出是 CoT 解答，长度由真实数据决定（解答通常数百到数千 tokens，天然长 decode）。可选 `--hf-output-len <n>` 覆盖输出长度上限。

### 均衡负载场景 → sharegpt 数据集

ShareGPT 长度由真实对话决定，不需指定 input/output 长度，只收集通用参数。vllm 默认过滤 `max_prompt_len=1024, max_total_len=2048`，对应均衡负载区间。可选 `--sharegpt-output-len <n>` 覆盖输出长度（一般不设，保留真实长度）。

---

## Step 3: 准备数据集

下载地址、镜像配置、缓存路径、校验方式等细节见 [reference/datasets-guide.md](reference/datasets-guide.md)。要点：

- **InstructCoder / NuminaMath-CoT**：vllm 经 `huggingface_hub` 自动拉取，无需手动下载。HF 不可达时 `export HF_ENDPOINT=https://hf-mirror.com`；如需集中缓存 `export HF_HOME=./running-time/<served_model_name>/bench/dataset/hf_cache`。
- **sharegpt**：必须先用 wget 下载到 `./running-time/<served_model_name>/bench/dataset/`，缺失会报错。执行时 `--dataset-path` 指向该文件。

---

## Step 4: 执行 vllm bench serve

完整命令模板与参数表见 [reference/command-examples.md](reference/command-examples.md)。要点：

- **backend / endpoint**：chat/instruct 模型用 `--backend openai-chat --endpoint /v1/chat/completions`；base 模型用 `--backend openai --endpoint /v1/completions`。不要用 `--backend vllm`（那是直连本地引擎、需重载模型，不适合已部署 HTTP 服务）。
- **结果保存**：固定 `--save-result --result-dir ./running-time/<served_model_name>/bench`。
- **HF 数据集环境变量**（InstructCoder / NuminaMath-CoT 场景前执行一次）：`HF_ENDPOINT`（镜像）与 `HF_HOME`（集中缓存到 `./running-time/<served_model_name>/bench/dataset/hf_cache`）。
- **`--served-model-name`** 必须与部署时一致，执行前询问用户确认。

各场景命令骨架（`...` 处为通用参数 `--model/--served-model-name/--base-url/--num-prompts/--max-concurrency/--request-rate/--temperature`，见 reference）：

```bash
# 长 prefill（InstructCoder）
vllm bench serve --backend openai-chat ... --endpoint /v1/chat/completions \
  --dataset-name hf --dataset-path likaixin/InstructCoder --hf-split train --hf-output-len 128 \
  --save-result --result-dir ./running-time/<served_model_name>/bench

# 长 decode（NuminaMath-CoT）
vllm bench serve --backend openai-chat ... --endpoint /v1/chat/completions \
  --dataset-name hf --dataset-path AI-MO/NuminaMath-CoT --hf-split train \
  --save-result --result-dir ./running-time/<served_model_name>/bench

# 均衡负载（sharegpt）
vllm bench serve --backend openai-chat ... --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path ./running-time/<served_model_name>/bench/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
  --save-result --result-dir ./running-time/<served_model_name>/bench
```

> vllm bench serve 通常几分钟内完成（num_prompts=200 或长 decode 场景可能十几分钟），不需要 `setsid`；若预期超时可加 `> ./running-time/<served_model_name>/bench/bench.log 2>&1 &` 后台运行并轮询日志。

---

## Step 5: 收集指标并生成报告

> 报告模板见 [reference/report-template.md](reference/report-template.md)

vllm bench serve 完成后终端直接打印结果块，同时 `--save-result` 写入 JSON。读取最新结果：

```bash
ls -t ./running-time/<served_model_name>/bench/*.json | head -1
```

**关键指标：**

| 指标 | 说明 | 场景侧重 |
|------|------|----------|
| `Mean/P99 TTFT (ms)` | 首 Token 时间 | 长 prefill |
| `Mean/P99 TPOT (ms)` | 每输出 Token 时间（不含首 Token） | 长 decode |
| `Mean/P99 ITL (ms)` | Token 间隔延迟 | 长 decode |
| `Output token throughput (tok/s)` | 输出吞吐 | 长 decode |
| `Request throughput (req/s)` | 请求吞吐 | 均衡负载 |
| `Total Token throughput (tok/s)` | 总吞吐 | 均衡负载 |
| `Successful / Failed requests` | 请求计数 | 全场景必看 |

报告写入 `./running-time/<served_model_name>/bench/`，包含压测配置、执行命令、指标表。失败请求数 > 0 时在报告中标注原因（常见：长度超 `max_model_len`、并发过高 OOM、网络超时）。

## 参考文档

| 文档 | 说明 |
|------|------|
| [reference/datasets-guide.md](reference/datasets-guide.md) | 三个数据集的来源、下载与校验 |
| [reference/command-examples.md](reference/command-examples.md) | 完整命令模板与参数表 |
| [reference/report-template.md](reference/report-template.md) | 性能压测报告模板 |
