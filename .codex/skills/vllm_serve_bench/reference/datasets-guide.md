# 压测数据集指南

本 skill 使用三个真实数据集，均来自 vllm / vllm-ascend 官方支持列表。不使用 `random` 合成数据集（prompt 是随机 token，EOS 不可控，尤其 decode 场景）。

---

## 数据集总览

| 场景 | `--dataset-name` | 来源 | 真实文本 | 需下载 | 长度控制 |
|------|------------------|------|----------|--------|----------|
| 长 prefill | `hf` | `likaixin/InstructCoder`（HuggingFace） | 真实代码 | vllm 自动拉取 | 输入由代码决定，`--hf-output-len` 控输出 |
| 长 decode | `hf` | `AI-MO/NuminaMath-CoT`（HuggingFace） | 真实数学题+CoT解答 | vllm 自动拉取 | 真实长度，可选 `--hf-output-len` 覆盖 |
| 均衡负载 | `sharegpt` | `ShareGPT_V3_unfiltered_cleaned_split.json` | 真实对话 | wget 下载 | 真实分布，可选 `--sharegpt-output-len` 覆盖 |

---

## InstructCoder 数据集（长 prefill）

HuggingFace 真实数据集，`likaixin/InstructCoder`，通用代码编辑数据集，114k 条 (instruction, input=代码, output=编辑后代码) 三元组。vllm 的 `InstructCoderDataset` 用 `{input}\n\n{instruction}...` 作输入，input 是真实代码文件、天然偏长，无输入长度过滤。

**使用：**

```bash
vllm bench serve ... --dataset-name hf --dataset-path likaixin/InstructCoder --hf-split train ...
```

- vllm 通过 `huggingface_hub` 自动拉取，下载到 HF cache（默认 `~/.cache/huggingface/datasets`）。
- HuggingFace 不可达时设镜像（对 `hf` 数据集加载生效）：`export HF_ENDPOINT=https://hf-mirror.com`
- 如需集中 HF 缓存到本次压测目录：`export HF_HOME=./running-time/<served_model_name>/bench/dataset/hf_cache`
- `--hf-output-len <n>` 控输出长度（默认 200，长 prefill 设 128）。

**特征：**
- 输入是真实代码、天然偏长。长输入 = 大量 prefill 计算。
- `--dataset-path` 填的是 HF ID，不是本地目录；数据由 `load_dataset()` 管理。

---

## NuminaMath-CoT（长 decode，hf 数据集）

HuggingFace 真实数据集，`AI-MO/NuminaMath-CoT`，数学竞赛题 + 带 chain-of-thought 的解答。vllm 的 `AIMODataset` 用 `problem` 字段作输入、`solution` 字段作期望输出，输出长度 = 解答 token 数（CoT 解答通常数百到数千 tokens，天然长 decode）。过滤上限放宽到 `max_prompt_len=2048, max_total_len=32000`。

**使用：**

```bash
vllm bench serve ... --dataset-name hf --dataset-path AI-MO/NuminaMath-CoT --hf-split train ...
```

- vllm 通过 `huggingface_hub` 自动拉取，下载到 HF cache（默认 `~/.cache/huggingface/datasets`）；集中缓存设 `HF_HOME=./running-time/<served_model_name>/bench/dataset/hf_cache`。
- HuggingFace 不可达时设镜像（对 `hf` 数据集加载生效）：`export HF_ENDPOINT=https://hf-mirror.com`
- 可选 `--hf-output-len <n>` 覆盖输出长度上限（一般不设，保留真实解答长度）。

**约束/注意：**
- 依赖模型能产出 CoT 推理；非推理模型输出可能偏短。
- 数据集较大，首次加载需联网拉取；streaming 模式下按 `num_prompts` 采样。

---

## sharegpt 数据集（均衡负载）

真实多轮对话数据集，输入输出长度由真实对话决定。vllm 默认过滤 `max_prompt_len=1024, max_total_len=2048`，对应均衡负载区间。

**官方下载地址（与 vllm-ascend 文档一致）：**

```bash
mkdir -p ./running-time/<served_model_name>/bench/dataset

# 官方源
wget -P ./running-time/<served_model_name>/bench/dataset \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# HuggingFace 不可达时用镜像（vllm-ascend 官方脚本 ensure_sharegpt_downloaded 使用此镜像）
wget -P ./running-time/<served_model_name>/bench/dataset \
  https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

**镜像说明：**
- `HF_ENDPOINT=https://hf-mirror.com` 只对 `huggingface_hub` / `datasets` 库生效，**不重写** `wget`/`curl` 直连 URL。用 wget 时必须显式把 `huggingface.co` 换成 `hf-mirror.com`。
- `VLLM_USE_MODELSCOPE=True` 只控制模型/tokenizer 加载，**不影响** benchmark 数据集下载。

**下载后校验：**

```bash
ls -lh ./running-time/<served_model_name>/bench/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
python -c "import json; d=json.load(open('./running-time/<served_model_name>/bench/dataset/ShareGPT_V3_unfiltered_cleaned_split.json')); print(len(d), 'entries')"
```

文件约 200MB+，条目数约 90k+。校验失败需重新下载。

**使用：**

```bash
vllm bench serve ... --dataset-name sharegpt \
  --dataset-path ./running-time/<served_model_name>/bench/dataset/ShareGPT_V3_unfiltered_cleaned_split.json ...
```

- `--dataset-path` 必填，指向本地下载的 JSON 文件。
- 可选 `--sharegpt-output-len <n>` 覆盖输出长度（一般不设，保留真实长度）。

---

---
