# Qwen3-Coder-30B 部署参数搜索报告

- 日期：2026-10-21
- Goal Plus 任务：`gp_0001`（revision 1）
- 最终状态：`complete`
- 冻结 spec：`spec_d219f52f7fcc`
- 搜索 run：`run_0002`（前身 `run_0001` 因 spec 缺陷作废）

---

## 1. 目标与合同

在 preset `long_prefill` 场景下，以官方 `vllm bench serve` raw result 的 **`TTFT_P95`（毫秒）** 为唯一评分字段，**最小化**该指标。

- 串行模式：`max_parallel=1`
- 候选预算：`max_attempts=10`（共享）
- 晋升模式：`artifact_only`（交付 `.patch` 产物，不写回主源码）
- baseline / candidate / promotion 三者保持相同模型、场景、指标口径与 workload。

### 冻结 workload（全程不变）

| 参数 | 值 |
| --- | --- |
| 模型 | `Qwen3-Coder-30B-A3B-Instruct`（served as `qwen3-coder-30b`） |
| 模型路径 | `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct` |
| 权重 | 61.07 GB（MoE，bf16，heads=32/kv=4/head_dim=128/layers=48） |
| 场景 | `long_prefill`（preset） |
| num_prompts | 500 |
| max_concurrency | 64 |
| request_rate | 4（开环到达率） |
| benchmark_mode | `production` |
| 后端 | `openai`，endpoint=`/v1/completions` |
| 数据集 | `likaixin/InstructCoder`（train），`hf-output-len=128` |

### 可编辑面与候选约束

唯一可编辑文件：`candidate_params.json`，字段 `tp / dp / max_num_seqs / max_num_batched_tokens`。

- `tp`：2（或 `null` 表示基线，由静态 OOM 最小 TP 推荐解析为 2）；`dp`：1
- `max_num_seqs ∈ [16, 256]`；`max_num_batched_tokens ∈ [64, 8192]`
- 约束：`max_num_batched_tokens >= max_num_seqs`；`tp * dp <= 2`

### verifier

- 命令：`python3 search_verifier.py`（完整流程：静态 OOM → 设备选择 → `vllm serve` 部署 → 官方 `vllm bench serve` 压测 → 解析 raw result → 完整性校验 → 输出 `TTFT_P95`）
- 角色：`ranking_signal`（评分）+ `promotion_gate`（提升复验）
- 产物：deployment v2 + benchmark v2 report

---

## 2. 运行环境

| 项 | 值 |
| --- | --- |
| vLLM / vLLM-Ascend | 0.18.0 / 0.18.0 |
| torch_npu | 2.9.0 |
| 设备 | 8× Ascend 910B3 |
| 空闲 NPU | 0,1,2,3,5（4,6,7 被其他租户占用；本任务选用 0,1） |
| 服务端口 | 8117（首选，确认可用；候选范围 8117–8120） |

设备选择由 verifier 依据 `search_config.json` 的 `authorized_devices=[0..7]` 结合 npu-smi 实时 free/busy 判定，选前 `tp*dp` 张空闲卡，并设 `ASCEND_RT_VISIBLE_DEVICES`。

---

## 3. 完整搜索路径

> 全部候选 `tp=2, dp=1`；`(mns, mnbt)` 表示 `(max_num_seqs, max_num_batched_tokens)`。

### 3.1 基线（bootstrap）

| 阶段 | 配置 | TTFT_P95 |
| --- | --- | --- |
| baseline origin | (32, 4096) | **2176.4 ms** |
| workspace 迁移复验 | (32, 4096) | 2410.4 ms |

> 前序还有一次作废 spec（`spec_4cc97de20905`）的 baseline 测量 = 1608.5 ms；该整个 spec 因 verifier 文件清单缺陷（见 §5.1）作废，未计入有效结果。

### 3.2 候选迭代（run_0002，candidate `c001`）

| # | 配置 (mns, mnbt) | TTFT_P95 (ms) | 假设 | 结果判定 |
| --- | --- | --- | --- | --- |
| 1 | (32, 8192) | 3888.9 | 增大 mnbt 允许更大 prefill batch | 更差 |
| 2 | (16, 4096) | 18857.0 | 减小 mns 降低调度竞争 | 灾难（远差） |
| 3 | (64, 4096) | **387.3** | mns 对齐 max_concurrency=64，消除排队 | 大幅改善 |
| 4 | (128, 4096) | 395.9 | mns 超过并发提供更多调度余量 | 略差 |
| 5 | (64, 8192) | 385.9 | mns=64 下增大 mnbt | 微降 |
| 6 | (64, 2048) | **360.5** | mns=64 下减小 mnbt，更短单批 prefill | **最优 / 选中** |
| 7 | (64, 1024) | 364.7 | 继续减小 mnbt | 拐点，略差 |

### 3.3 晋升复验

| 阶段 | 配置 | TTFT_P95 | 结论 |
| --- | --- | --- | --- |
| promotion 独立复验 | (64, 2048) | 401.3 ms | `outcome=valid` |

### 3.4 搜索曲线小结

```
(32,4096) ──────────────── 2176.4  ← baseline
(32,8192) ───── 3888.9
(16,4096) ─────────────────────── 18857.0（灾难）
(64,4096) ── 387.3  ← mns=64 是对齐最大并发的关键
(128,4096) ── 395.9
(64,8192) ── 385.9
(64,2048) ── 360.5  ★ 最优
(64,1024) ── 364.7  ← 拐点
```

**核心结论**：
1. `max_num_seqs=64`（对齐 workload 的 `max_concurrency=64`）是主导因素——将 baseline 的 2176 ms 降到 ~390 ms，消除并发排队等待。
2. 在 mns=64 固定后，`max_num_batched_tokens` 呈 U 型，**2048 最优**（8192→385.9、4096→387.3、2048→360.5、1024→364.7）；低于 2048 后 chunked prefill 开销回升。

---

## 4. 最终结果

| 项 | 值 |
| --- | --- |
| 获选候选 | `c001`（iteration 6） |
| 获选配置 | `{"tp": 2, "dp": 1, "max_num_seqs": 64, "max_num_batched_tokens": 2048}` |
| 候选评分 TTFT_P95 | **360.5 ms** |
| promotion 复验 TTFT_P95 | **401.3 ms**（valid） |
| 相对 baseline 改善 | **约 82%**（2176 → 360~401 ms） |
| 候选 attempt 使用 | 7 / 10 |
| 晋升模式 | `artifact_only` |

### 交付产物

| 产物 | 路径 |
| --- | --- |
| 晋升 patch | `.gp/runs/run_0002/promotion/c001.patch` |
| 搜索报告（自动生成） | `.gp/runs/run_0002/report.md`（及 `report.html`） |
| 候选结果账本 | `.gp/runs/run_0002/workspace/c001/results.tsv`（7 行） |
| 工作记录 | `TASKS.md` |

---

## 5. 遇到的问题与解决方案

### 5.1 `FrozenVerifierModified`（反作弊哈希拦截）

- **触发原因**：首次冻结把可编辑的 `candidate_params.json` 误列入 `verifier_artifact_paths`，anti-cheat 哈希校验拦截了任何候选编辑。
- **解决**：`invalidate_run`（reason=`verifier_contract_invalid`）作废 `run_0001`，重新冻结时 verifier 文件清单只保留 `search_verifier.py` + `search_config.json`。

### 5.2 `DirtySourceWorkspace` / `FrozenSourceChanged`

- **触发原因**：冻结后向已跟踪的 `TASKS.md` 追加进度并提交，导致 HEAD 前进、冻结 artifact 与 origin 不一致。
- **解决**：`git reset --hard` 回冻结提交；运行期笔记改存 gitignored 的 `search_runtime/`，Goal Plus 终态后再提交 `TASKS.md`。

### 5.3 `device_pools` 要求 trusted host 配置

- **触发原因**：首版 spec 声明了 `device_pools` 与 `resource_lock`，但宿主环境无 `GOAL_PLUS_DEVICE_POOLS`。
- **解决**：去掉两者；verifier 改用 `authorized_devices` + npu-smi 实时 free/busy 选卡。

### 5.4 Skill 目录含未跟踪 `__pycache__`

- **触发原因**：本地联调运行/编译 verifier 生成 `.pyc`，freeze 校验未跟踪文件失败。
- **解决**：清理全部 `__pycache__` 与 `.pyc` 后重新 freeze。

### 5.5 `VerifierInProgress`（候选 session 反复 failed）

- **触发原因**：`run_verifier` 单次查询等待上限仅 60s，而一次部署+压测需约 9 分钟；worker 在 verifier 未结束前重复提交。
- **解决**：框架明确客户端超时不取消验证；Main 固定"启动 → 等 failed → bash 轮询进程退出 → 读 `list_history` 取结算结果"的节奏，结算前不派发新 attempt。7 次 attempt 均正确结算。

### 5.6 `apply_promotion` 拒绝写回

- **触发原因**：`promotion_mode=artifact_only`，框架禁止写回主源码。
- **处置**：符合预期，交付 `.patch` 产物即可，无需写回。

---

## 6. 数据噪声与可复现性说明

- 本次 verifier 调用 `vllm bench serve` 时**未加 `--seed`**（默认 `None`，从系统熵取种子），导致同一配置重复测量的 TTFT_P95 存在固有方差：baseline 两次 1608.5 / 2176.4 ms（约 26%），获选配置候选/promotion 两次 360.5 / 401.3 ms（约 11%）。
- 该方差主要来自 client 侧随机性（prompt 采样顺序 + 泊松到达时序）对尾部分位指标（P95）的放大，叠加硬件/多租户环境抖动。
- 主效应结论（baseline → 候选约 5.5× 改善）远超该噪声带，稳健可信；平台区内部 1% 级别的差异（如 360.5 vs 364.7）不可过度解读。
- **可复现性改进方向**：`vllm bench serve` 支持 `--seed`，可固定数据集采样洗牌与请求到达时序；但需为 baseline/candidate/promotion 统一加同一 seed 并重新冻结 spec，且制造采样、硬件争抢等物理噪声仍不可完全消除。