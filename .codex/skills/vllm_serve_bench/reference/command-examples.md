# vllm bench serve 命令示例

## 完整参数表

**服务连接：**

| 参数 | 说明 |
|------|------|
| `--backend <type>` | 后端类型：`openai`(completions) / `openai-chat`(chat) |
| `--base-url <url>` | 服务地址 http://host:port |
| `--endpoint <path>` | API 路径：`/v1/completions` 或 `/v1/chat/completions` |
| `--model <path>` | 本地模型路径（加载 tokenizer），不是 served-model-name |
| `--served-model-name <name>` | API 请求中的模型名（与服务端一致） |
| `--tokenizer <path>` | 可选，tokenizer 路径（与 model 不同时指定） |

**测试控制：**

| 参数 | 说明 |
|------|------|
| `--num-prompts <n>` | 测试请求数 |
| `--request-rate <rate>` | 请求速率 QPS，inf=同时发送所有 |
| `--max-concurrency <n>` | 最大并发数 |
| `--burstiness <f>` | 突发因子，默认 1（泊松），<1 更突发，>1 更均匀 |

**hf 数据集（长 prefill，InstructCoder / 长 decode，NuminaMath-CoT）：**

| 参数 | 说明 |
|------|------|
| `--dataset-name hf` | 固定为 hf |
| `--dataset-path <id>` | HuggingFace 数据集 ID，如 `AI-MO/NuminaMath-CoT` |
| `--hf-split <split>` | 数据集 split，如 `train` |
| `--hf-output-len <n>` | 可选，覆盖输出长度上限 |

**sharegpt 数据集（均衡负载）：**

| 参数 | 说明 |
|------|------|
| `--dataset-name sharegpt` | 固定为 sharegpt |
| `--dataset-path <file>` | 本地 ShareGPT JSON 路径 |
| `--sharegpt-output-len <n>` | 可选，覆盖输出长度 |

**采样与结果：**

| 参数 | 说明 |
|------|------|
| `--temperature <f>` | 温度，0=确定性 |
| `--seed <n>` | 随机种子，默认 0 |
| `--save-result` | 保存结果 JSON |
| `--result-dir <path>` | 结果目录 |
| `--result-filename <name>` | 可选，指定结果文件名 |

---

## 结果目录

所有场景结果统一保存到 `./running-time/<served_model_name>/bench/`（`served_model_name` 中的 `/`、`\\`、`..` 先替换为 `_`）：

```bash
RESULT_DIR=./running-time/<served_model_name>/bench
mkdir -p $RESULT_DIR
```

---

## 命令模板

> 以下 `<model_path>` 是本地模型路径，`<served_model_name>` 是 API 模型名，二者不同。`<IP>:<PORT>` 是服务地址。

### 长 prefill（InstructCoder，真实代码长输入）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # HF 不可达时设，可达可省
export HF_HOME=./running-time/<served_model_name>/bench/dataset/hf_cache   # 集中 HF 缓存
RESULT_DIR=./running-time/<served_model_name>/bench
mkdir -p $RESULT_DIR

vllm bench serve \
  --backend openai-chat \
  --model <model_path> \
  --served-model-name <served_model_name> \
  --base-url http://<IP>:<PORT> \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path likaixin/InstructCoder --hf-split train --hf-output-len 128 \
  --num-prompts 100 --max-concurrency 32 --request-rate inf \
  --temperature 0 \
  --save-result --result-dir $RESULT_DIR
```

### 长 decode（NuminaMath-CoT，真实数学题长解答）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # HF 不可达时设，可达可省
export HF_HOME=./running-time/<served_model_name>/bench/dataset/hf_cache   # 集中 HF 缓存
RESULT_DIR=./running-time/<served_model_name>/bench
mkdir -p $RESULT_DIR

vllm bench serve \
  --backend openai-chat \
  --model <model_path> \
  --served-model-name <served_model_name> \
  --base-url http://<IP>:<PORT> \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path AI-MO/NuminaMath-CoT --hf-split train \
  --num-prompts 100 --max-concurrency 32 --request-rate inf \
  --temperature 0 \
  --save-result --result-dir $RESULT_DIR
```

### 均衡负载（sharegpt，真实对话）

```bash
RESULT_DIR=./running-time/<served_model_name>/bench
mkdir -p $RESULT_DIR

vllm bench serve \
  --backend openai-chat \
  --model <model_path> \
  --served-model-name <served_model_name> \
  --base-url http://<IP>:<PORT> \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path $RESULT_DIR/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200 --max-concurrency 32 --request-rate inf \
  --temperature 0 \
  --save-result --result-dir $RESULT_DIR
```

### base 模型（completions 端点）

把 `--backend openai-chat` 换成 `--backend openai`，`--endpoint` 换成 `/v1/completions`，其余不变。

### 固定 QPS 服务测试

把 `--request-rate inf` 换成具体 QPS（如 `4`），可加 `--burstiness`：

```bash
... --request-rate 4 --burstiness 1.0 ...
```

### 多组并发对比

固定场景与长度，仅改 `--max-concurrency` 多次执行：

```bash
for c in 16 32 64 128; do
  vllm bench serve ... --max-concurrency $c \
    --save-result --result-dir ./running-time/<served_model_name>/bench/concurrency_$c
done
```

---

## 注意事项

| 项 | 说明 |
|----|------|
| `--model` vs `--served-model-name` | 前者是本地路径（加载 tokenizer），后者是 API 模型名，两者不同 |
| `--backend vllm` 不要用 | 直连本地引擎、需重新加载模型，不适合已部署 HTTP 服务 |
| `--save-result` 必加 | 不加只在终端打印，无法事后复核；加后写 JSON 到 `--result-dir` |
| `max_model_len` 校验 | 请求总长不得超过服务部署时的 max_model_len |
| chat 模型必须用 chat 端点 | instruct/chat 模型用 `/v1/completions` 会报错或输出异常 |
| 远程服务 | `--base-url http://IP:PORT`，确认网络连通、端口开放 |
| HF 数据集缓存 | `HF_HOME=./running-time/<served_model_name>/bench/dataset/hf_cache` 集中 HF 缓存；`HF_ENDPOINT` 对 hf 加载生效，不重写 wget URL |

## 结果读取

```bash
# 最新结果文件
ls -t ./running-time/<served_model_name>/bench/*.json | head -1

# 查看内容（Python 格式化）
python -m json.tool $(ls -t ./running-time/<served_model_name>/bench/*.json | head -1) | head -60
```
