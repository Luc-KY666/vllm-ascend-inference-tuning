---
name: vllm-ascend-deployment
description: 在任务和 Goal Plus 冻结的设备、环境与输出合同内，完成 vLLM-Ascend 推理服务的参数校验、静态 OOM 预检、启动和健康验收。
---

# vLLM-Ascend 部署技能

本 skill 只提供可复用的部署方法。模型、设备、端口、环境 wrapper、输出目录和服务
生命周期由当前任务及 Goal Plus provider 冻结；skill 不下载模型、不创建设备调度器、不
管理资源锁，也不替代独立 verifier。

部署成功表示当前服务通过了任务约定的环境、进程、健康、模型身份和最小请求验收。它不
表示模型质量通过、压测完成、参数最优或 Goal Plus candidate 已被选中。

## 输入与边界

开始前读取任务的 `goal.md`、`workflow.md`、`inputs.md` 和已有 `TASKS.md`，并确认：

- `model_path`：模型目录、配置来源或任务允许的模型 ID。
- `served_model_name`：服务对外暴露的模型身份。
- `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens`、显存比例、上下文长度和
  dtype 等部署参数。
- Goal Plus 已冻结的 `authorized_devices`、`visibility_env`、端口、资源锁、激活
  wrapper、绝对 Python 和 verifier 临时目录。
- 允许写入的部署报告、日志、PID、配置和诊断 artifact 路径。

输入字段、状态和报告结构见：

- [输入契约](references/input-contract.md)
- [静态 OOM 规则](references/static-oom.md)
- [部署报告](references/report-schema.md)

模型权重和 tokenizer 默认只读。不得把模型目录、凭据、场景 skill、verifier 或共享
缓存写入 candidate 的可修改范围。

## 缺少部署参数时的行为

`model_path` 和 `served_model_name` 始终必须提供。若缺少 TP/DP、批处理上限、上下文
门槛或其他启动所需参数：

1. 列出缺失字段和无法完成的校验。
2. 调用纯计算的 `scripts/static_oom_precheck.py`（若输入足够）生成最小 TP 和容量
   风险建议。
3. 输出 `status=recommendation_only` 的部署报告。
4. 不启动 vLLM，不占用设备，不修改用户输入，也不把推荐值当作冻结配置。

最小 TP 推荐是诊断结果。只有用户或 Main 将推荐值写入本轮冻结合同后，才能进入真实
部署。

## 执行顺序

### 1. 核对运行身份

- 确认当前 provider 是 `git_worktree`/direct 还是 ThinkThread。
- 核对 Goal Plus 注入的设备、可见性环境、资源锁、临时目录和输出 surface。
- 不从全机 `npu-smi` 枚举结果扩大授权设备，不手工改写
  `ASCEND_RT_VISIBLE_DEVICES`。
- 使用任务提供的激活入口，核对 vLLM、PyTorch、torch-npu 和 vLLM-Ascend 的实际
  import 路径与版本。
- 按安装版本的 `vllm serve --help=all` 检查任务需要的参数。帮助不兼容时在启动前
  失败，不靠猜测参数名继续执行。

### 2. 校验模型与参数

- 校验模型配置、tokenizer 和任务需要的权重文件；识别缺失文件、不可读文件和
  Git LFS pointer。
- 校验模型身份、TP/DP、批处理参数、显存比例、上下文预算、dtype、量化开关和任务
  允许的额外启动参数。
- 校验 `tp * dp` 不超过 GP 已授权的设备数量，并核对可见设备和任务冻结的设备分组。
- 端口只能使用任务或 GP 分配的值；冲突时报告失败，不停止未知进程，也不私自换端口。

### 3. 执行静态 OOM 预检

预检只使用模型配置、权重估算、dtype、TP、卡容量和最低上下文 KV Cache 容量。它不
读取实时 HBM 占用、不选择设备、不启动服务，也不把 `max_num_seqs * model_max_len`
作为拒绝条件。

```text
python <skill-root>/scripts/static_oom_precheck.py \
  --input <task-owned-input.json> \
  --output <task-owned-preflight.json>
```

结果含义：

- `pass`：静态容量可行，继续动态启动验证。
- `reject`：静态容量明显不足，不启动服务，不运行压测。
- `unknown`：配置或设备容量无法可靠计算，保留未知原因并继续动态验证。

预检不得静默扩大 TP/DP、替换设备或降低上下文要求。实际启动仍可能因为运行时
workspace、图编译、通信或真实 KV 使用 OOM；这类结果必须单独记录为运行时失败。

### 4. 启动服务

服务启动必须使用任务冻结的 wrapper 和参数。日志、PID、命令记录、缓存和临时文件
写入本次 invocation 获准的目录；不能使用固定的 `./results`、`./running-time`、
固定端口或模型目录。

git_worktree/direct 路径中，服务启动、健康检查、请求、压测（如同次启停合同要求）
和清理必须在同一次持锁 verifier 中完成。ThinkThread 路径中，worker 可以按任务合同
预启动并交付服务接口；短 verifier 只消费已交付接口，不启动、重启或停止服务。

服务进程必须记录可核验的 PID、启动时间、命令摘要和进程组标识。清理只允许针对本次
启动的进程，不能使用按进程名或端口的模糊终止。

### 5. 健康与身份验收

按以下顺序验收，前一步失败时停止后续副作用：

1. `/health` 或任务指定的等价就绪接口。
2. `/v1/models`，确认返回的模型名与 `served_model_name` 完全一致。
3. 使用同一模型名发送任务允许的最小对话请求，确认 HTTP 状态、响应结构、非空内容
   和正常结束。
4. 在可行时核对监听端口、服务进程和本次启动 PID 的归属。

只有健康和服务身份都通过，才记录 `service_identity_valid=true`。端口可达、进程存活、
单个 HTTP 200 或候选自己写入的 JSON 都不能单独证明部署成功。

## 报告与失败语义

输出 `deployment_report.json` 的具体路径由任务或 GP 注入。报告至少区分：

- `recommendation_only`：参数不足，只完成推荐或静态诊断。
- `input_invalid`：输入、路径、CLI 或静态容量门禁失败。
- `startup_failed`：服务进程未正常启动或启动阶段退出。
- `healthcheck_failed`：健康接口超时或返回错误。
- `service_identity_invalid`：服务健康但模型身份或最小请求不匹配。
- `service_ready`：部署阶段的独立服务验收通过。
- `cleanup_failed`：按次清理未完成；不能宣称资源已释放。

`service_ready` 只代表部署层通过。benchmark、正确性、性能收益、selection、promotion
和最终服务保留分别由压测 skill、任务 verifier 和 Goal Plus Main 判断。

报告中可以记录本次实际使用的设备 ID、端口和环境身份，但不得记录启动前或清理后的
全机资源快照、HBM 占用、进程列表来代替独立验收。

## 禁止事项

- 不复制、修改或依赖本目录之外的 `.codex/`、`.pi/` 配置作为 Playground skill 入口。
- 不创建第二套 Goal/Search/candidate 调度器、设备锁服务或 Runtime。
- 不把示例模型、数据集、端口、设备、路径和 timeout 写成公共默认值。
- 不因静态 OOM 或启动失败自动扩大 TP/DP。
- 不把 `Valid=true`、PID 文件、候选报告或单次健康请求当成最终任务通过。
- 不写只读模型目录，不覆盖已有 attempt 证据，不删除失败现场。
