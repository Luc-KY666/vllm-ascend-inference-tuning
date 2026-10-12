---
name: vllm-ascend-deployment
description: 在任务和 Goal Plus 冻结的输入、设备与输出合同内，完成 vLLM-Ascend 推理服务的预检、启动、健康验收和部署报告。
---

# vLLM-Ascend 部署技能

本 skill 负责一次部署 attempt 的编排。模型、设备、端口、环境 wrapper、输出目录和
服务生命周期由任务 Main 或 Goal Plus provider 提供；本 skill 不下载模型、不发现或
分配设备、不创建资源锁，也不替代独立 verifier。

部署报告中的 `service_ready` 只表示本次服务通过部署层验收，不表示压测成功、参数最优、
candidate promotion 或最终业务收益。

## 执行流程

1. 读取任务的 `goal.md`、`workflow.md`、`inputs.md`、`TASKS.md` 和 GP provider
   配置，确认本轮允许的模型身份、运行参数、资源、服务生命周期和输出 surface。
2. 使用任务 Main 或 provider 生成的唯一规范输入
   `deployment-input.json`。输入来源、字段归属和状态边界见
   [input-contract.md](references/input-contract.md)。缺少任务合同或 provider 事实时
   停止当前部署，不猜测默认值、设备、端口或宿主路径。
3. 按 [input-contract.md](references/input-contract.md) 执行输入契约检查，并根据
   结果区分 `recommendation_only`、`input_invalid` 和待动态确认的 `unknown`。
4. 从规范输入、官方模型 `config.json` 和 provider 注入的硬件事实派生本次 attempt
   的 `static-oom-input.json`，再按 [static-oom.md](references/static-oom.md) 的
   唯一入口执行静态 OOM 预检。
5. 静态预检为 `reject` 时不启动服务，并在部署报告中记录
   `status=static_oom_rejected`、`failure.type=static_oom_rejected` 和
   `oom.phase=static_preflight`、`oom.type=static_oom_rejected`；为 `unknown` 时保留未知
   原因，不把结果改写成 `pass`。
   只有任务/provider 合同明确允许带着该未知项进入动态确认时才继续；没有足够运行参数时，
   只输出最小 TP 推荐和 `status=recommendation_only`，不占用设备。
6. 预检和动态前置条件允许启动后，使用任务冻结的 wrapper、设备可见性、端口和 vLLM
   参数拉起服务。按任务约定记录进程、命令摘要、日志和清理边界；服务生命周期由当前
   provider 管理。
7. 依次执行任务指定的健康接口、`/v1/models`、served model identity 和最小请求验收。
   服务启动或最小请求阶段发生 OOM 时，记录
   `failure.stage=service_startup`、`failure.type=service_startup_oom` 和
   `oom.phase=service_startup`、`oom.type=service_startup_oom`；只有健康、模型身份和最小
   请求都通过时，才记录 `service_identity_valid=true`。
8. 按 [report-schema.md](references/report-schema.md) 写出
   `deployment_report.json`，保留规范输入、静态预检 artifact、服务检查、错误阶段和
   清理状态。按 provider 合同清理或交付服务，并保持失败现场可追溯。

## 参考资料

- [输入契约](references/input-contract.md)：唯一规范输入及其来源、状态和派生关系。
- [静态 OOM 规则](references/static-oom.md)：`static-oom-input.json` 的预检入口和
  容量判定口径。
- [部署报告结构](references/report-schema.md)：一次部署 attempt 的报告合同。

## 边界

- 不调用、不复制、不修改本目录外的 `.codex/` 或 `.pi/` 配置。
- 不在 skill 内实现 Goal Plus Runtime、candidate 管理、搜索调度、设备租约或第二套
  verifier。
- 不因静态 OOM、启动失败或健康检查失败自动扩大 TP/DP、替换设备或降低上下文要求。
- 不把 PID、端口可达、单个 HTTP 200、候选自报 JSON 或 `Valid=true` 当作最终任务通过。
