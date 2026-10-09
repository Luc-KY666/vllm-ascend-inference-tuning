# 模型拉起与压测

本节根据八个必填输入、可选 goodput 输入、数据集配置和其他显式覆盖值，执行动态环境检查、模型服务启动、健康检查、在线 HTTP `vllm bench serve`、失败诊断以及最终 JSON 写入。

本 Skill 启动服务后通过 OpenAI 兼容 HTTP 接口压测该服务。默认使用 `openai-chat` backend 和 `/v1/chat/completions`；base 模型可以显式覆盖为 `openai` backend 和 `/v1/completions`。不要使用 `--backend vllm`，因为该模式会绕过已启动的 HTTP 服务并重新加载本地引擎。

服务启动前必须已经完成[静态 OOM 容量检查](static-oom-precheck.md)。检查结果为 `reject` 时不得进入本节的服务启动和 benchmark 流程；检查结果为 `pass` 或 `unknown` 时按本节继续执行，并将该检查结果原样写入最终 JSON。

## 1. 服务配置默认值

以下配置可以使用默认值；模型名称和模型路径不在此处提供默认值，必须由主入口显式传入：

```text
HOST=127.0.0.1
PORT=8117
BENCH_BASE_URL=http://127.0.0.1:${PORT}
BENCH_BACKEND=openai-chat
BENCH_ENDPOINT=/v1/chat/completions
STARTUP_TIMEOUT_SEC=1800
POLL_INTERVAL_SEC=10
TRUST_REMOTE_CODE=true
VLLM_USE_V1=1
DOWNLOAD_SOURCE=huggingface
RESULT_ROOT=./results
SUMMARY_FILENAME=benchmark_result.json
ENVIRONMENT_PROFILE_PATH=${RESULT_ROOT%/}/environment_profile.json
REFRESH_ENVIRONMENT_PROFILE=false
```

base 模型或服务只提供 completions 接口时，可显式覆盖非必填配置：

```text
BENCH_BACKEND=openai
BENCH_ENDPOINT=/v1/completions
```

默认 host 只绑定本机回环地址，适用于本机在线 benchmark。只有明确需要外部机器访问服务时，才显式覆盖 `HOST=0.0.0.0` 或其他监听地址；该覆盖必须写入最终 JSON 的 `overrides`。无论服务监听地址是什么，健康检查、身份校验和 benchmark 默认仍通过 `BENCH_BASE_URL=http://127.0.0.1:${PORT}` 访问本机服务，除非显式覆盖 benchmark base URL。

`MODEL_NAME` 和 `MODEL_PATH` 是必填输入：

```text
MODEL_NAME=<必填的 model_name，服务对外暴露的模型名>
MODEL_PATH=<必填的 model_path，服务模型来源和 benchmark tokenizer 来源>
```

`MODEL_NAME` 是 API 请求中的 served model name；`MODEL_PATH` 是服务加载模型的路径或 ID，也是 `vllm bench serve --tokenizer` 用于加载 tokenizer 的来源。两者不能因为看起来相似而互相替代。

实际端口、模型来源、benchmark backend/endpoint、结果目录和本次任务实际使用的 NPU 卡号必须写入最终 `benchmark_result.json`。NPU 卡号记录在 `model_config.npu_devices` 中。不得在最终 JSON 或必需的用户可见诊断工件中写入 NPU 健康状态、HBM 占用、运行进程或资源快照。端口 `8117` 被占用时，可以选择一个可用端口进行一次安全修复，但不得改变八个必填输入。

## 2. 本机环境画像缓存

为支持同一机器上频繁执行多组配置压测，可复用的环境探测必须从单次运行流程中提取出来，写入本机环境画像文件。默认画像路径为 `${RESULT_ROOT%/}/environment_profile.json`；显式覆盖 `ENVIRONMENT_PROFILE_PATH` 时必须写入 `overrides`。

环境画像只缓存跨运行稳定、且与八个必填输入无关的信息，例如：

- `schema_version`：固定为 `vllm-ascend-benchmark-env-v1`。
- `created_at`、`updated_at` 和生成画像时使用的时区或 UTC offset。
- `commands`：`python3`、`vllm`、`curl`、`npu-smi` 的可执行路径及轻量文件指纹。
- `versions`：Python、Torch、vLLM 和 vLLM-Ascend 版本。
- `vllm_cli`：`vllm serve` 和 `vllm bench serve` 的完整帮助摘要、已确认支持的关键参数、是否支持 `--served-model-name`、`--goodput` 和显式结果文件名。
- `npu_baseline`：Torch NPU 是否可用、生成画像时可见设备数量和 `ASCEND_RT_VISIBLE_DEVICES` 值；不得记录设备健康状态、HBM 占用、运行进程或启动/清理资源快照。
- `datasets.local_paths`：本地数据集目录或文件的可读性、JSON 可解析性、样本数或条目数等校验结果，以及路径类型、大小、mtime、inode 等用于判断缓存是否过期的指纹。

以下内容不得放入环境画像并复用：

- 八个必填输入及其派生值。
- 本次服务端口占用、服务 PID、服务健康状态和 `/v1/models` 身份校验结果。
- 当前 NPU 上的运行进程、实时 HBM 占用、设备健康状态或清理前后资源快照。
- 本次模型路径的权重、tokenizer 和配置检查结果，除非后续文档显式引入独立的模型画像缓存。

加载环境画像时必须先验证：

- JSON 可解析且 `schema_version` 匹配。
- 关键命令路径仍存在，轻量文件指纹未变化。
- Python/Torch/vLLM/vLLM-Ascend 模块路径指纹未变化；无法确认时必须刷新画像，而不是继续信任旧画像。
- 当前 `ASCEND_RT_VISIBLE_DEVICES` 与画像中的值一致；不一致时至少刷新 NPU baseline，不能用旧的可见设备数做本次 TP/DP 判断。
- 本次需要使用的本地数据集路径在画像中存在且路径指纹一致；不存在或已变化时，只重新校验该路径并以原子写入方式更新画像。
- 如果本次启用了 goodput，画像必须确认 `vllm bench serve` 支持 `--goodput`；画像缺少该结论或记录为不支持时，必须刷新 CLI 检查，不能静默忽略 goodput。

当 `REFRESH_ENVIRONMENT_PROFILE=true`、画像不存在、画像校验失败或关键路径指纹变化时，才执行完整环境探测。刷新失败时，在启动服务前写出环境检查失败结果 JSON；不要退回到未经校验的旧画像。画像写入必须先写临时文件，校验 JSON 可解析后再原子替换目标文件。

每次运行的 `COMMANDS_LOG` 必须记录本次使用的环境画像路径、画像是复用还是刷新、刷新原因和画像 schema version。最终 `benchmark_result.json` 的 `artifacts` 可以记录环境画像路径，但不得把画像内容内联到最终 JSON。

## 3. 动态环境检查

首次生成或刷新环境画像时执行以下检查：

```bash
command -v vllm
command -v curl
command -v npu-smi
vllm serve --help=all
vllm bench serve --help=all
```

必须确认当前安装版本同时支持服务启动和在线 HTTP benchmark 所需的参数，并把结论写入环境画像。至少检查以下选项是否能被解析：

```text
vllm serve: --host --port --served-model-name --tensor-parallel-size
            --data-parallel-size --max-num-seqs --max-num-batched-tokens
vllm bench serve: --backend --model --tokenizer --base-url --endpoint
                  --dataset-name --hf-name --dataset-path
                  --num-prompts --request-rate --max-concurrency
                  --num-warmups --burstiness --goodput
                  --save-result --result-dir
```

如果普通帮助只显示配置分组，不能将其视为参数兼容性检查通过；应使用当前版本的完整帮助或实际参数解析结果。检查失败时，在启动服务前写出环境检查失败结果。启用 goodput 但当前 CLI 不支持 `--goodput` 时，在启动服务前写出环境检查失败结果，不得退回到不带 goodput 的 benchmark。画像有效且已包含这些检查结果时，后续运行不得重复执行完整帮助和模块导入探测。

确认 vLLM-Ascend 可以使用 NPU，并记录版本：

```bash
python3 - <<PY
import torch
import vllm
import vllm_ascend

assert torch.npu.is_available(), "No Ascend NPU detected"
print("vllm:", vllm.__version__)
print("vllm-ascend: ready")
PY
```

TP、DP、可见 NPU 和本地路径的静态检查由主入口负责；本节负责检查当前 CLI、运行环境画像、服务端口和实际启动行为。

启动前必须在可见设备范围内确认 NPU 可用设备数量；画像中的 NPU baseline 可以避免重复导入 vLLM/Torch 做版本探测，但不能替代本次设备选择、占用排除和服务启动后的实际使用卡号确认。不预设固定 HBM 阈值、拓扑或设备选择算法。资源检查结果仅用于内部决策，不得写入最终 JSON 或必需的用户可见诊断工件。服务启动后，必须确定并记录本次任务实际使用的 NPU 卡号。

如果设置了 `ASCEND_RT_VISIBLE_DEVICES`，所有设备统计、资源检查和服务使用记录都只能针对可见设备。不能把宿主机全量设备数直接当成可用设备数。`model_config.npu_devices` 使用实际任务可识别的 NPU 卡号表示。

## 4. 模型权重规则

- `random`：必须追加 `--load-format dummy`，不加载 checkpoint 权重；输入的 `MODEL_PATH` 仍需提供配置和 tokenizer 信息。
- `hf` 和 `sharegpt`：不得追加 `--load-format dummy`，必须使用真实权重。
- 默认优先使用可读的本地权重；没有本地权重时使用默认的 Hugging Face 来源。显式选择 ModelScope 时才设置 ModelScope 来源和 `VLLM_USE_MODELSCOPE=True`。
- 下载、认证或网络失败直接写入错误 JSON，不在来源之间静默切换。

启动前的模型路径静态检查：

- 当 `MODEL_PATH` 解析为本地目录或配置文件时，必须检查其存在且可读；本地模型目录或配置文件必须提供可解析的 `config.json`（或明确指定的配置文件），并提供当前模型和 vLLM 可识别且可读的 tokenizer 必要资产，例如 `tokenizer.json`、`tokenizer.model` 或所需的 vocab/merges 文件组合。
- `random` 模式不要求 checkpoint 权重；`hf` 和 `sharegpt` 模式的本地模型路径必须至少包含一个可读的真实权重文件或权重索引。
- 如果 `MODEL_PATH` 不是本地路径而是模型 ID，不因本地没有同名目录判定输入无效；按 `DOWNLOAD_SOURCE` 在启动加载阶段验证模型来源、认证和下载，失败时直接写入错误 JSON，不在来源之间静默切换。
- 本地路径或必要文件检查失败时，不启动服务或 benchmark，按 `error.type=invalid_model_path` 写入具体缺失、不可读或无法解析的路径。

## 5. 服务启动、健康检查和 Valid

每次运行创建独立目录：

```bash
RUN_DIR="${RESULT_ROOT%/}/${RUN_ID}"
SERVER_LOG="${RUN_DIR}/vllm_server.log"
BENCH_LOG="${RUN_DIR}/benchmark.log"
COMMANDS_LOG="${RUN_DIR}/commands.txt"
mkdir -p "${RUN_DIR}"
: > "${SERVER_LOG}"
: > "${BENCH_LOG}"
: > "${COMMANDS_LOG}"
```

服务启动、健康检查、benchmark 和清理必须在同一个持续存活的父进程或执行上下文中完成。不得在即将结束的短命命令会话中启动后台服务后离开该会话，否则执行环境可能回收后台进程。服务命令应直接作为后台子进程启动，启动后立即捕获 `$!` 到 `SERVER_PID`；不要通过会改变 PID 归属的额外 daemon 或 wrapper 启动服务。父进程必须保持运行，直到本次流程完成或触发清理。

服务命令范式：

```bash
SERVE_ARGS=(
    "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${MODEL_NAME}"
    --tensor-parallel-size "${TP}"
    --data-parallel-size "${DP}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)

if [ "${DATASET_MODE}" = "random" ]; then
    SERVE_ARGS+=(--load-format dummy)
fi
if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
    SERVE_ARGS+=(--trust-remote-code)
fi

vllm serve "${SERVE_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
```

每次启动尝试都必须在 `COMMANDS_LOG` 中记录 attempt 编号、完整命令、`SERVER_PID`、启动时间和结果。若 `/health` 成功前 `SERVER_PID` 已退出，先检查 `SERVER_LOG`：日志为空或没有 vLLM 启动标记且父执行上下文已结束时，按启动生命周期问题处理；已有模型、路径、OOM 或配置错误日志时，按对应失败类型处理。启动生命周期问题可以用完全相同的 `SERVE_ARGS` 做一次有记录的安全重试，重试不得修改八个必填输入；重试时保留前一次命令和失败原因，并将当前有效 PID 更新为新的 `SERVER_PID`。

启动后按 `POLL_INTERVAL_SEC` 轮询：

```bash
HEALTH_URL="http://127.0.0.1:${PORT}/health"
START_TIME=$(date +%s)

while true; do
    if curl -fsS --connect-timeout 2 --max-time 5 "${HEALTH_URL}" >/dev/null; then
        MODEL_HEALTHY=true
        break
    fi

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        MODEL_HEALTHY=false
        break
    fi

    ELAPSED=$(( $(date +%s) - START_TIME ))
    if (( ELAPSED >= STARTUP_TIMEOUT_SEC )); then
        MODEL_HEALTHY=false
        break
    fi
    sleep "${POLL_INTERVAL_SEC}"
done
```

`/health` 返回成功后，不能仅凭该响应认定服务就绪。必须优先请求 `GET /v1/models`，确认响应可解析且包含与 `MODEL_NAME` 完全一致的服务模型名：

```bash
curl --fail --silent "http://127.0.0.1:${PORT}/v1/models"
```

模型名不匹配时不得进入 benchmark。若 `/v1/models` 不可用或无法完成身份校验，不能仅凭 `/health` 继续，应使用实际 benchmark endpoint 以 `MODEL_NAME` 发起一次最小请求预检；预检失败不得进入正式 benchmark，预检本身不计入正式指标。例如 chat endpoint 使用 `max_tokens=1` 的最小请求，completions endpoint 使用 `max_tokens=1` 的最小 prompt。

应在可行时核对 `${PORT}` 的监听者或等价进程标识与当前 `SERVER_PID` 或服务进程一致；无法获取进程身份时，至少保留 `/v1/models` 校验或 endpoint 预检结果作为服务身份依据。

`Valid` 的判定规则：

- `/health` 成功前，`Valid=false`。
- 模型进程退出、健康检查超时、服务身份验证失败、OOM、路径错误或静态检查失败，`Valid=false`。
- `/health` 和服务身份验证均成功后，`Valid=true`；即使后续 benchmark 失败，也保持 `Valid=true`，并将 `status` 写为 `benchmark_failed`。
- OOM 必须识别为 `error.type=oom`，且 `Valid=false`。不要自动调整 TP、DP、`max_num_seqs` 或 `max_num_batched_tokens`。

## 6. benchmark 命令

服务健康和身份验证通过后，使用在线 OpenAI 兼容 backend 执行。`NUM_PROMPTS`、`REQUEST_RATE`、`MAX_CONCURRENCY` 和 `BURSTINESS` 必须使用[数据集配置](dataset-config.md)按 `benchmark_mode` 派生并完成样本数截断后的最终值：

```bash
BENCH_ARGS=(
    --backend "${BENCH_BACKEND}"
    --model "${MODEL_NAME}"
    --tokenizer "${MODEL_PATH}"
    --base-url "${BENCH_BASE_URL}"
    --endpoint "${BENCH_ENDPOINT}"
    --num-prompts "${NUM_PROMPTS}"
    --request-rate "${REQUEST_RATE}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --num-warmups "${NUM_WARMUPS}"
    --temperature "${TEMPERATURE}"
    --percentile-metrics "${PERCENTILE_METRICS}"
    --metric-percentiles "${METRIC_PERCENTILES}"
    --save-result
    --result-dir "${RUN_DIR}"
    --disable-tqdm
)
if [ -n "${BURSTINESS:-}" ]; then
    BENCH_ARGS+=(--burstiness "${BURSTINESS}")
fi

GOODPUT_ARGS=()
if [ -n "${GOODPUT_TTFT_MS:-}" ]; then
    GOODPUT_ARGS+=("ttft:${GOODPUT_TTFT_MS}")
fi
if [ -n "${GOODPUT_TPOT_MS:-}" ]; then
    GOODPUT_ARGS+=("tpot:${GOODPUT_TPOT_MS}")
fi
if [ -n "${GOODPUT_E2EL_MS:-}" ]; then
    GOODPUT_ARGS+=("e2el:${GOODPUT_E2EL_MS}")
fi
if [ "${#GOODPUT_ARGS[@]}" -gt 0 ]; then
    BENCH_ARGS+=(--goodput "${GOODPUT_ARGS[@]}")
fi
```

`GOODPUT_TTFT_MS`、`GOODPUT_TPOT_MS` 和 `GOODPUT_E2EL_MS` 由可选 goodput 输入归一化得到；至少一个非空时才追加 `--goodput`。不要为了“启用 goodput”填充未提供的约束，也不要把用户的 `TPOP` 直接传给 CLI，应归一化为 `tpot`。

这里的 `--model` 是 API 请求使用的模型名；未显式传入服务模型名时，`vllm bench serve` 默认使用 `--model` 的值。`--tokenizer` 是 tokenizer 名称或路径，通常传入 `MODEL_PATH`，用于加载服务模型对应的 tokenizer。不要使用 `--backend vllm`、`--host`/`--port` 直连本地引擎的旧命令形态。

正式执行前记录 `BENCH_START_TIME`，并记录 `${RUN_DIR}` 中已有的原始结果文件集合：

```bash
BENCH_START_TIME=$(date +%s)
vllm bench serve "${BENCH_ARGS[@]}" >"${BENCH_LOG}" 2>&1
BENCH_EXIT_CODE=$?
```

如果当前 CLI 帮助提供显式结果文件名或路径参数，优先将结果文件固定到本次 `${RUN_DIR}` 下的唯一文件，并记录实际路径；不得在未通过 CLI 兼容性检查时擅自追加版本不确定的参数。

## 7. 四种场景的命令分支

### 6.1 Random

```bash
BENCH_ARGS+=(
    --dataset-name random
    --random-input-len "${RANDOM_INPUT_LEN}"
    --random-output-len "${RANDOM_OUTPUT_LEN}"
    --random-range-ratio "${RANDOM_RANGE_RATIO}"
    --seed 0
    --ignore-eos
)
```

### 6.2 HF：长 Prefill 和长 Decode

根据输入 `SCENARIO` 使用对应的 canonical 数据集：

```bash
case "${SCENARIO}" in
    long_prefill)
        HF_DATASET="likaixin/InstructCoder"
        HF_OUTPUT_LEN="${HF_OUTPUT_LEN:-128}"
        ;;
    long_decode)
        HF_DATASET="AI-MO/NuminaMath-CoT"
        ;;
    *)
        # 生成 input_invalid，不启动服务或 benchmark。
        ;;
esac

if [ -n "${HF_HOME:-}" ]; then
    export HF_HOME
fi
if [ -n "${HF_ENDPOINT:-}" ]; then
    export HF_ENDPOINT
fi

test -n "${HF_DATASET:-}" || {
    echo "HF_DATASET 不能为空" >&2
    exit 2
}
test -n "${HF_DATASET_PATH:-}" || {
    echo "HF_DATASET_PATH 不能为空" >&2
    exit 2
}

BENCH_ARGS+=(
    --dataset-name hf
    --hf-split "${HF_SPLIT}"
    --hf-name "${HF_DATASET}"
    --dataset-path "${HF_DATASET_PATH}"
)

if [ -e "${HF_DATASET_PATH}" ]; then
    test -r "${HF_DATASET_PATH}" || {
        echo "HF_DATASET_PATH 不存在或不可读: ${HF_DATASET_PATH}" >&2
        exit 2
    }
fi

if [ -n "${HF_OUTPUT_LEN:-}" ]; then
    BENCH_ARGS+=(--hf-output-len "${HF_OUTPUT_LEN}")
fi
```

长 Prefill 使用 `likaixin/InstructCoder`，默认 `--hf-output-len 128`；长 Decode 使用 `AI-MO/NuminaMath-CoT`，默认不设置输出上限，以保留真实 CoT 解答长度。HF 数据集首次加载可能需要联网；`HF_HOME` 应指向本次运行目录下的缓存位置，避免不同运行隐式共享不明缓存。

### 6.3 ShareGPT：Prefill Decode 均衡

```bash
if [ "${SCENARIO}" = "prefill_decode_balance" ]; then
    test -r "${SHAREGPT_DATASET_PATH:-}" || {
        echo "SHAREGPT_DATASET_PATH 不存在或不可读: ${SHAREGPT_DATASET_PATH:-<empty>}" >&2
        exit 2
    }

    BENCH_ARGS+=(
        --dataset-name sharegpt
        --dataset-path "${SHAREGPT_DATASET_PATH}"
    )
    if [ -n "${SHAREGPT_OUTPUT_LEN:-}" ]; then
        BENCH_ARGS+=(--sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}")
    fi
fi
```

ShareGPT 文件必须在 benchmark 前准备并通过 JSON 可解析性检查。通常不设置 `SHAREGPT_OUTPUT_LEN`，让真实多轮对话决定输出长度；当前 vLLM 的默认过滤区间为 `max_prompt_len=1024`、`max_total_len=2048`，具体参数以当前 CLI 和数据集实现为准。

原始结果无法唯一确定、无法解析或与本次请求参数不一致时，不得提取性能指标；服务已经通过健康检查和身份验证时仍保持 `Valid=true`，但写入 `status=benchmark_failed`、`error.stage=benchmark`，并将 `error.type` 设为 `raw_result_not_found`、`raw_result_ambiguous` 或 `raw_result_invalid` 之一，`artifacts.raw_benchmark_json` 写为 `null`。无论 benchmark 退出码如何，性能指标只能从通过下述校验的原始 JSON 提取，不能从日志或未验证候选补全。

## 8. 原始结果校验与性能提取

如果 CLI 没有显式结果文件路径参数，benchmark 完成后只能从本次运行目录中确定原始结果文件：

- 候选必须是本次 benchmark 开始后新建或更新的、可读且非空的 JSON 普通文件；排除 `${SUMMARY_JSON}`、临时文件和其他诊断文件。
- 候选必须能解析为 JSON，并且其中存在的模型标识（如 `model_id` 或 `served_model_name`）必须与 `${MODEL_NAME}` 完全一致；存在 `tokenizer_id`、`num_prompts`、`max_concurrency`、`request_rate` 或 `burstiness` 时，必须分别与 `${MODEL_PATH}`、`${NUM_PROMPTS}`、`${MAX_CONCURRENCY}`、`${REQUEST_RATE}` 和 `${BURSTINESS}` 一致。
- 启用 goodput 时，候选必须包含可解析的数值字段 `request_goodput`，包括合法的 `0`；缺失或为 `null` 时不得把 benchmark 当作 goodput 结果。未启用 goodput 时不从 `request_goodput` 推导任何值。
- 如果结果包含日期或时间字段，该字段必须与本次运行相符；候选还必须来自当前 `${RUN_DIR}`，不能复用上一次运行的文件。
- 经过上述校验后必须恰好剩余一个候选。不得使用宽泛 glob 后取第一个、最新一个或任意一个文件来猜测结果。

原始结果文件验证通过后，必须检查请求完成情况。若结果包含 `failed` 且大于 `0`，或包含 `completed` 且小于 `${NUM_PROMPTS}`，或包含 `num_prompts` 且与 `${NUM_PROMPTS}` 不一致，不得将本次 benchmark 判为成功。服务已经通过健康检查和身份验证时仍保持 `Valid=true`，但写入 `status=benchmark_failed`、`error.stage=benchmark`、`error.type=benchmark_incomplete`，并在 `error.message` 或诊断字段中记录 `failed`、`completed` 和期望请求数。此时性能指标写为 `null`，不能把部分完成的结果作为成功指标。

benchmark 使用 `--metric-percentiles 90,99`，必须提取 TTFT、TPOT、E2EL 三项的 P90 和 P99。直接字段优先使用：

```text
p90_ttft_ms
p99_ttft_ms
p90_tpot_ms
p99_tpot_ms
p90_e2el_ms
p99_e2el_ms
```

如果结果使用嵌套 percentile 结构，则按指标名和百分位寻找 `90`、`99`，并转换成毫秒。任何缺失的性能值写为 `null`，不要填 `0` 或猜测。启用 goodput 时，将原始 JSON 的 `request_goodput` 原样提取到最终 `goodput.request_goodput`，单位为 `req/s`；不能用请求吞吐替代 goodput。在线 benchmark 额外提供的 ITL、请求吞吐和 token 吞吐可以保存在 `dataset.parameters` 或诊断工件中，但不能替换输出契约规定的 TTFT、TPOT 和 E2EL 字段。

## 9. 写出唯一结果 JSON

benchmark 完成或失败后，写出主入口约定的 `${SUMMARY_JSON}`。结果 JSON 必须包含：

- `model_config`：八个必填输入中的模型与部署相关字段，包括模型名称、模型路径、TP、DP、`max_num_seqs`、`max_num_batched_tokens`、实际端口、`npu_devices` 和 HTTP backend/endpoint。`npu_devices` 只记录本次任务实际使用的 NPU 卡号，不记录资源信息、健康状态、HBM 占用或运行进程。
- `dataset`：数据集名称或模式、路径或 ID、HF 数据集名（HF 场景）、实际数据条数、`benchmark_mode`、最大并发、warmup、请求速率、burstiness 和数据集专用参数。`scenario` 为非 random 时记录 `dataset.scenario`；`scenario=random` 时省略该字段。
- `static_oom_check`：静态 OOM 容量检查结果。`status=unknown` 时即使 benchmark 成功也不得删除或改写为 `pass`。该字段不得记录 NPU 健康状态、HBM 占用、运行进程或资源快照。
- `Valid`：按照本节规则填写。
- `performance`：TTFT、TPOT、E2EL 的 P90/P99，单位毫秒。
- `goodput`：未启用时为 `null`；启用时记录 `constraints_ms` 和原始结果中的 `request_goodput`，单位为 `req/s`。如果 benchmark 未完成或 goodput 字段缺失，`request_goodput` 为 `null` 并按 benchmark 失败处理。
- `status`、`error`、`overrides` 和诊断工件路径；诊断工件路径可以包含 `environment_profile`，指向本次读取或刷新的本机环境画像文件。

`${SUMMARY_JSON}` 不得直接以截断方式写入。应先在同一目录生成临时文件，写完后验证文件非空、可解析为 JSON，并满足本 Skill 的最低结构和 `Valid`、`status`、`performance` 字段一致性；验证通过后再以原子方式替换 `${SUMMARY_JSON}`。写入或验证失败时，必须尽力写出一个最小且可解析的失败结果，不能留下空文件、半写文件或仅有终端输出。

输出文件只使用 JSON，不再以人类可读表格作为接口结果。控制台可以打印结果文件路径，不能只打印指标而不生成文件。

## 10. 失败诊断

明确识别以下显存错误：

```text
out of memory
OutOfMemoryError
out of HBM
Cannot allocate memory
ACLRT out of memory
MemoryError
```

非 OOM 配置问题最多进行一次有记录的修复，例如端口冲突、路径错误、ModelScope 环境变量或 CLI 参数兼容性。第二次失败后停止，并把最近日志摘要写入 `error.message`。无论失败发生在哪个阶段，都保留服务日志、benchmark 日志和命令记录。

## 11. 清理

清理前必须保留本次服务的 `SERVER_PID` 以及可用于识别其子进程的进程组或等价服务标识。只清理本次启动的服务及其子进程，不得使用 `pkill vllm` 或按进程名称终止其他服务。

使用记录的 `SERVER_PID` 清理本次服务：

```bash
cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
```

清理完成后必须验证：

- `SERVER_PID` 及其子进程已退出；
- `${PORT}` 不再由本次服务提供。
