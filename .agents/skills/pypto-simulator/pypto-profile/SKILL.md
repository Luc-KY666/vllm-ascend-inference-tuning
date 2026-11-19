---
name: pypto-profile
description: 将 PyPTO/simpler native task graph、timing 和 memory evidence 规范化为 profile manifest。
---

# PyPTO Native Profile

本 skill 负责采集和导出 native evidence。调度单位是 outlined simpler task，不是 ACLNN 算子、DSL
statement 或单个 physical SPMD block。采集必须使用与目标 serving 完全相同的 PyPTO/simpler 版本。

## Native dispatch capture

```bash
python .agents/skills/pypto-simulator/pypto-profile/scripts/run_serving_profile.py \
  --output /tmp/capture --level 4 --npu-device 4 \
  --program decode_program -- \
  pypto-serving <native serving command>
```

`--capture-dispatch PROGRAM=ORDINAL` 可按 serving L3 call boundary 选择某一次 invocation，适合
区分初始化、warmup 和 request graph：

```bash
python .agents/skills/pypto-simulator/pypto-profile/scripts/run_serving_profile.py \
  --output /tmp/capture --level 4 --npu-device 4 \
  --capture-dispatch prefill_program=2 -- \
  pypto-serving <native serving command>
```

ordinal bootstrap 会生成完整 executable child artifact，目标 ordinal 才执行 level-4 capture。
`--bypass-program` 只用于诊断；旁路 program 不属于 NPU measurement，也不能进入 latency model。

`--npu-device` 运行期间只读轮询 `npu-smi info -t proc-mem`，记录 observed PID。出现多个 NPU
PID 或 telemetry error 时将 collection 标记失败，不终止外部进程。不能使用容器 PID namespace 或
进程组归属判断 host PID。正式采集前必须保证目标卡连续 60 秒无进程。

有效 collection 会写入：

- `capture_config.json`；
- `collection_run.json`，包含 child return code、monitor error、observed PID 和污染状态；
- executable child artifacts；
- level-4 DFX output。

## Runtime profile export

```bash
python .agents/skills/pypto-simulator/pypto-profile/scripts/export_runtime_profile.py \
  --capture-dir /tmp/capture/build/program/dfx_outputs/rank0/d4 \
  --model-family example --model-id example-1 --phase decode \
  --program decode_program --features-json /tmp/features.json \
  --require-level4 --output /tmp/profile
```

导出器通过 `deps.json` 连接 task timing，保留：

- task predecessor；
- AIC/AIV/MIX subslot；
- resource shape、block count；
- callable identity；
- control output metadata；
- 每个 epoch 的 sorted physical-block duration distribution；
- source `workload_id` 和 capture role。

必须提供的 native evidence 包括 `deps.json`、`chip_swimlane_records.json`、`name_map*.json` 和
`dispatch_identity*.json` / `dispatch_program*.json`。字段不能靠字符串猜测。动态 feature 只有在
args dump 或 dispatch-specific evidence 证明后才能写入 `--features-json`。

## Exact memory profile

Task profile 和 memory profile 分开生成。memory evidence 必须来自独立 native workload，记录：

- exact static weight tensor bytes；
- exact KV page count、page bytes 和 total bytes；
- `total_bytes`、`free_bytes`、driver-visible non-KV bytes；
- simpler committed bytes；
- weight/KV/materialization phase duration。

```bash
python .agents/skills/pypto-simulator/pypto-profile/scripts/run_memory_evidence.py \
  --output /tmp/memory-evidence.jsonl --workload-id memory-a -- \
  python .agents/skills/pypto-simulator/pypto-emulate/scripts/run_validation_workload.py \
  --output /tmp/requests.jsonl --warmup-requests 0 --requests 1 -- \
  pypto-serving <native serving command>

python .agents/skills/pypto-simulator/pypto-profile/scripts/export_memory_profile.py \
  --log /tmp/native-a.log --log /tmp/native-b.log --log /tmp/native-c.log \
  --evidence /tmp/memory-evidence-a.jsonl \
  --evidence /tmp/memory-evidence-b.jsonl \
  --evidence /tmp/memory-evidence-c.jsonl \
  --kv-page-bytes <exact bytes per page> \
  --output /tmp/memory-profile.json
```

`--kv-page-bytes` 是模型适配提供的 exact page layout。当前默认值只为历史 Qwen fixture 兼容，
正式接入新模型必须显式传入。

exact bytes 不能使用参数量、日志四舍五入值或模型名推断。每个 bytes point 至少需要三个独立
workload；不同 page count 不能合并为一个 point。所有 total/free 算术、KV page 算术、workload
identity 和 weight bytes 不一致都必须失败。只有 duration model 和 footprint model 都满足 strict
条件时，profile 才能用于正式模拟。

模型特有的 page layout、allocation 分类、control output 和 tokenizer 规则放在
`references/<model>-adapter.md`，不能写入本 skill。

## 输出契约

profile manifest 使用 `schema_version: 1`，至少包含：

- `model`；
- native `source.kind`、`source.npu_measured`、独立 `source.workload_id`；
- `tasks`；
- task identity、phase、kind、resource shape、block count、dependencies、tensor roles、features；
- duration samples、subslot samples、block samples 和 outputs。

fixture 可以省略 native workload identity，但绝不能作为硬件性能证据。详见
`../references/manifest-contract.md`。

## 前置条件

- normalize/export：Python 3.10+；
- native capture：pinned PyPTO/simpler/PTOAS 和 Ascend；
- 使用真实 physical NPU ID，不能同时用 `ASCEND_RT_VISIBLE_DEVICES` 重映射；
- profile launcher、native baseline 和 simulator launcher 必须保持进程隔离。
