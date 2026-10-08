---
name: minimal-tp-vllm-ascend-deploy
description: >-
  在单机昇腾 NPU 上，根据模型权重、架构和显存严格计算可用的最小 TP，启动 vLLM-Ascend OpenAI 兼容服务并完成健康、模型和对话验收；用户明确要求时，按需启用 profiling 采集与解析。适用于模型权重和运行依赖已在目标机准备好的场景；不负责模型推荐、下载、环境准备、多机部署或常规性能 benchmark。
---

# 单机最小 TP 部署

目标是让服务使用满足资源和上下文门槛的**最小允许 TP**启动，并以可复核证据结束。TP 在本 skill 中指单机使用的 NPU Device 数量；只支持单机，不能把多机节点数当成 TP。

## 适用边界与必需输入

开始前收集并记录：

- 模型目录和对外暴露的 `served_model_name`；模型权重必须已经存在目标机。
- 目标机（当前主机或一个 SSH 目标）和芯片系列（A2/A3/A5，对应 910B 系列/910C 系列/950 系列，或 310P/300I Duo）。执行命令时假定当前 shell 已经位于可用的 vLLM-Ascend 环境。
- `min_required_context_len`，默认 `4096`；用户若指定更大的业务上下文，使用该值。
- `gpu_memory_utilization`，默认 `0.9`。
- 若模型目录不能完整计算权重，要求用户提供 `weight_gib`。不要用 active params 代替 MoE 总权重。
- 若 `config.json` 缺少 KV 架构字段，要求用户提供官方 `official_context_len`；没有架构也没有官方上下文时停止，不能猜 4096。

运行产物目录为 `./running-time/<served_model_name>/deploy_benchmark/`。将模型名作为目录名使用前，先把 `/`、`\\` 和 `..` 等路径字符替换为 `_`；后续所有命令必须使用同一个规范化目录名。

不在本 skill 范围内：模型下载/转换、运行环境准备、量化产物制作、多机 HCCL/SSH 编排和性能 benchmark。用户明确要求时可以额外启用本 skill 的 profiling 采集流程；量化模型可以部署，但必须明确其量化精度并加入 `--quantization ascend`。

## 1. 目标机和依赖探测

以下命令直接在目标机当前 shell 执行。假定 vLLM-Ascend、CANN/ATB 和 Python 依赖已经准备好；若命令不存在或导入失败，停止并报告环境问题，不在本 skill 内切换或创建运行环境。

先验证：

```bash
npu-smi info
npu-smi info -t memory
vllm --version
python -c 'import vllm; print(vllm.__version__)'
```

从 `npu-smi info` 只读取 `Device` 列，得到实际 Device ID、数量、芯片型号和每 Device 显存。Device 数量必须与后续 `ASCEND_RT_VISIBLE_DEVICES` 的数量一致。检查目标 Device 是否已有进程或显存占用；已有服务不停止，需换空闲 Device 或取得用户明确授权。

310P 需按物理卡成对选择 Device，并追加 `--enforce-eager --dtype float16`；910 系列无需这两个参数。

## 2. 计算最小 TP

优先使用本目录的 `scripts/plan_min_tp.py`。它读取模型 `config.json` 和权重目录，输出 JSON 计算记录；远程目标无法直接访问脚本时，在控制端收集同样的输入后运行脚本，不能手工改写结果。

```bash
mkdir -p ./running-time/<served_model_name>/deploy_benchmark
python scripts/plan_min_tp.py \
  --config <model_dir>/config.json \
  --weight-dir <model_dir> \
  --vram-gib <每Device显存GiB> \
  --available-devices <目标机可用Device总数> \
  --min-context 4096 \
  --gpu-memory-utilization 0.9 \
  --runtime-headroom-factor 0.9 \
  --official-min-tp <没有明确官方最低要求时填1> \
  > ./running-time/<served_model_name>/deploy_benchmark/min_tp_plan.json
```

权重目录无法可靠统计时改用 `--weight-gib <权重GiB>`。工具会优先读取 `num_hidden_layers`/`num_layers`、`num_key_value_heads`/`num_kv_heads`、`head_dim`（缺失时用 `hidden_size / num_attention_heads`）和 `torch_dtype`。架构字段可位于常见的 `text_config` 中。

计算口径必须保持以下顺序和单位（先全部换成 bytes）：

```text
planning_memory_utilization = gpu_memory_utilization * runtime_headroom_factor
per_card_safe_weight_bytes = vram_bytes * planning_memory_utilization
min_cards_by_weight = ceil(weight_bytes / per_card_safe_weight_bytes)
raw_required_cards = max(min_cards_by_weight, official_min_tp, 1)
allowed_card_counts = [1, 2, 4, 8, 16]
candidate_tp = first allowed_card_count >= raw_required_cards
model_total_safe_vram_bytes = candidate_tp * vram_bytes * planning_memory_utilization
available_kv_bytes_total = model_total_safe_vram_bytes - weight_bytes
kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes
theoretical_max_model_len = floor(available_kv_bytes_total / kv_bytes_per_token)
```

默认 `gpu_memory_utilization=0.9`，`runtime_headroom_factor=0.9`。这两个系数只用于最小 TP 规划，因此规划安全预算为 `TP 总显存 × 0.9 × 0.9`，用于为图模式编译和运行时额外开销预留空间。

若候选 TP 的 KV 空间不足或理论上下文小于 `min_required_context_len`，依次尝试更大的允许值 `[1,2,4,8,16]`，直到不超过目标机可用 Device 数；第一个满足条件的值才是最小 TP。若 `config.json` 没有完整架构但有官方上下文，则只能在权重、硬件和最低 TP 校验通过后将理论值保守记为 `official_context_len`，并在报告中标记 `context_source=official_fallback`，不能伪造 KV bytes/token。

`kv_bytes_per_token < 1024` 或 `theoretical_max_model_len > 1048576` 都是参数异常，必须停止并报告，不能把异常值用于启动。

模型可部署必须同时满足：

- `weight_bytes <= model_total_safe_vram_bytes`；
- `available_kv_bytes_total > 0`，并且理论上下文不少于门槛；
- 计算出的 TP 不超过可用 Device 数，且属于 `[1,2,4,8,16]`；
- 芯片、量化和明确的官方最低 TP 约束满足。

执行前保存 JSON 计算记录，至少包含 `tp`、`weight_bytes`、`gpu_memory_utilization`、`runtime_headroom_factor`、`planning_memory_utilization`、`model_total_safe_vram_bytes`、`available_kv_bytes_total`、`kv_bytes_per_token`、`theoretical_max_model_len` 和输入参数。任何数字不能用展示后的 GB 值反算；不要把官方示例 TP 当作最低硬约束。

## 3. 启动门禁和命令

启动前必须完成以下门禁：

1. `ASCEND_RT_VISIBLE_DEVICES` 选择的 Device 数量等于计算出的 `tp`，且 310P 组合满足物理卡约束。
2. `max_model_len` 已明确，且 `min_required_context_len <= max_model_len <= theoretical_max_model_len`；没有用户要求时取 `min_required_context_len`，不要默认写入超出理论值的 32768。
3. 目标端口未占用；发现占用时换端口或询问用户，不能停止未知进程。
4. 当前工作目录可写；运行产物统一写入 `./running-time/<served_model_name>/deploy_benchmark/`。
5. 量化模型加入 `--quantization ascend`；非量化模型不要添加。

默认先加载 CANN 和 ATB 环境；只有用户明确提供了不同安装路径时才替换这两个路径：

先执行 `pwd` 记录工作目录；启动和后续验收必须保持在同一目录，否则相对日志/PID 路径会指向不同位置。

```bash
mkdir -p ./running-time/<served_model_name>/deploy_benchmark
cat > ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.command.sh <<'EOF'
#!/usr/bin/env bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export VLLM_USE_MODELSCOPE=true PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=<device_ids>
exec vllm serve <model_path> \
  --host 0.0.0.0 --port <port> \
  --tensor-parallel-size <tp> \
  --served-model-name <served_model_name> \
  --max-num-seqs 16 --max-model-len <max_model_len> \
  --trust-remote-code --gpu-memory-utilization <utilization> \
  <actual_extra_flags>
EOF
chmod 777 ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.command.sh
nohup bash -c 'exec ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.command.sh' \
  </dev/null > ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.log 2>&1 &
vllm_pid=$!
echo "$vllm_pid" > ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.pid
```

将 `<actual_extra_flags>` 直接替换为实际参数：量化模型追加 `--quantization ascend`，310P 追加 `--enforce-eager --dtype float16`；用户明确要求 profiling 时追加本 skill 后面的 `--profiler-config` 配置。不要把尖括号原样传给 shell。命令脚本本身就是此次部署的可复用命令记录。脚本中的 `exec` 保证 PID 文件对应实际 vLLM 进程；`nohup` 和 `&` 让服务脱离终端在后台运行。若环境加载或启动失败，进程会退出，直接根据日志判定失败。

本 skill 只主动创建以下运行产物，不写入模型目录：

- `./running-time/<served_model_name>/deploy_benchmark/min_tp_plan.json`：最小 TP 计算记录；
- `./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.command.sh`：本次实际部署命令；
- `./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.log`：服务日志；
- `./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.pid`：服务 PID。

不能用 `pkill` 或猜测 PID 停止不属于本次启动的服务。

## 4. 可选 profiling 采集

只有用户明确要求采集 profiling 时才启用本节；普通部署不要添加 profiler 参数。profiling 不改变最小 TP 计算规则，也不能替代进程、启动日志、模型 API 和对话 API 验收。

启动命令中的 `<actual_extra_flags>` 追加以下参数，并确保路径使用本次规范化的运行产物目录。启用 profiling 时先生成一次精确到分钟的目录名，后续启动参数、采集和报告都使用同一个已解析的 `<profile_dir>`：

```bash
profile_dir="./running-time/<served_model_name>/deploy_benchmark/vllm_profile_$(date +%Y%m%d_%H%M)"
mkdir -p "$profile_dir"
```

`torch_profiler_with_stack=false` 用于关闭 vLLM-Ascend 默认开启的 stack 采集，减少额外开销。启动前创建 profiling 目录，并在同一工作目录中完成后续采集和解析：

将 `<profile_dir>` 替换为上一步得到的实际目录路径后，在 `<actual_extra_flags>` 中加入：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"<profile_dir>","torch_profiler_with_stack":false}'
```

服务通过常规验收后开始采集。端口必须替换为本次部署端口，并且采集期间必须发送真实请求，否则可能没有有效 profiling 数据：

```bash
curl --fail --silent --show-error -X POST http://127.0.0.1:<port>/start_profile

curl --fail --silent --show-error http://127.0.0.1:<port>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served_model_name>","messages":[{"role":"user","content":"请生成一段用于 profiling 的响应。"}],"max_tokens":64}'

curl --fail --silent --show-error -X POST http://127.0.0.1:<port>/stop_profile
```

停止采集后，在本次 `<profile_dir>` 下找到新生成的实际 `*_ascend_pt` 目录，替换 `<ascend_pt_dir>` 后解析。不要使用其他日期目录或旧的解析文件：

```bash
python -c 'from torch_npu.profiler.profiler import analyse; analyse("<ascend_pt_dir>")'
```

profiling 验收必须记录：`start_profile` 和 `stop_profile` 均返回成功；本次带分钟时间戳的 `<profile_dir>` 下存在 `*_ascend_pt` 目录；`analyse` 成功完成并在该目录产生解析结果。profiling 文件不是模型服务验收证据。

启用 profiling 时，本 skill 额外产生带 `YYYYMMDD_HHMM` 时间戳的 `vllm_profile_<timestamp>/` 目录及其解析产物；未启用时不创建该目录，也不要求安装或导入 `torch_npu.profiler`。

## 5. 验收

启动后在同一目标 shell 中轮询，最多等待 10 分钟。每次检查：

```bash
test -f ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.pid && kill -0 "$(cat ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.pid)"
ps -p "$(cat ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.pid)" -o args= | grep -F 'vllm serve'
grep -E 'Application startup complete|Traceback|RuntimeError|OutOfMemory|OOM' ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.log | tail -50
curl --fail --silent http://127.0.0.1:<port>/v1/models
```

验收必须全部通过：

- PID 对应的 `vllm serve` 进程仍存活；
- 日志出现 `Application startup complete`，且没有启动阶段异常；
- `/v1/models` 返回 HTTP 200，并包含预期 `served_model_name`；
- 用同一个模型名完成一次对话请求，HTTP 200 且返回非空、有语义的文本：

```bash
curl --fail --silent http://127.0.0.1:<port>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served_model_name>","messages":[{"role":"user","content":"请用一句话说明你已启动。"}],"max_tokens":32}'
```

不要只以进程存在、HTTP 200 或端口监听判定成功。记录 `service_url`、`served_model_name`、实际 TP、Device 列表、`max_model_len`、日志路径和四项验收证据。启动失败时保留日志；若为真实显存不足，先说明实际错误，再经用户确认按下一个允许 TP 重算，不要静默扩大 TP。

## 结果格式

最终报告使用以下字段：

```text
deploy_status: success | failed
service_url: http://<host>:<port>
served_model_name: <name>
runtime_dir: ./running-time/<served_model_name>/deploy_benchmark
minimal_tp: <1|2|4|8|16>
visible_devices: <ids>
max_model_len: <value>
calculation_record: ./running-time/<served_model_name>/deploy_benchmark/min_tp_plan.json
log_path: ./running-time/<served_model_name>/deploy_benchmark/vllm_deploy.log
acceptance: process / startup log / models API / chat API
sample_request: <rendered curl command using service_url and served_model_name>
profiling: disabled | enabled
profile_dir: <only when profiling is enabled>
profile_ascend_pt_dir: <only when profiling is enabled>
profile_analysis: <only when profiling is enabled>
```

部署成功后必须额外输出一条可直接复制的样例请求，并将 `<service_url>` 和 `<served_model_name>` 替换为本次实际值：

```bash
curl --fail --silent <service_url>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served_model_name>","messages":[{"role":"user","content":"你好，请介绍一下自己。"}],"max_tokens":64}'
```
