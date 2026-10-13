# Qwen3-8B Long-Prefill Search

本目录提供两套只使用 vLLM-Ascend Playground Skill 的参数搜索任务：

| 入口 | provider 配置 | 搜索并发 |
| --- | --- | --- |
| `serial/` | `git_worktree`、`direct runtime`、`local verifier`、`artifact_only` | `max_parallel=1` |
| `parallel/` | `git_worktree`、`direct runtime`、`local verifier`、`artifact_only` | `max_parallel=2` |

两个入口的模型、负载、指标和可调参数合同一致，只在候选调度并发、资源要求和
并发证据上不同。用户需要修改的模型路径、数据集目录和候选搜索预算只在对应
入口的 `goal.md` 与 `goal-pi.md` 中维护一次。

## Skill

任务只调用本仓库 `.agents/skills/` 下的以下 Skill：

- `vllm-ascend-deployment`
- `vllm-ascend-benchmark`
- `vllm-ascend-inference-tuning`

任务入口限定为上述三项 Skill；负载、部署资料和运行期事实由 task 文档及 provider
冻结。

## 启动入口

从 `vllm-ascend-inference-tuning` 仓库根目录进入受信宿主后，将对应文件作为第一条
Goal Plus 消息：

```text
qwen3-8b-long-prefill-search/serial/goal.md
qwen3-8b-long-prefill-search/parallel/goal.md
```

Pi 使用同目录的 `goal-pi.md`。进入目标环境后，Main 还必须按入口中的 workflow、
inputs、search 和 modules 读取顺序核验真实 provider、模型、CLI、设备和输出目录。

## 任务边界

本目录是 parent-only 文档任务。模型、数据集、vLLM、CANN/ATB、设备、端口和
Goal Plus provider 均由用户环境或公开 provider 接口提供。真实 NPU 部署、独立
benchmark 和最终复验是任务通过的证明来源。
