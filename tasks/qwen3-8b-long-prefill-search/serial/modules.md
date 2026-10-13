# 串行任务模块

本任务是 parent-only 文档任务。业务仓、workspace manifest、模型文件、数据集文件、
设备脚本、人工 workload 和自定义 verifier 均由用户环境、指定 Skill 与 Goal Plus
provider 提供。

| 模块 | 来源 | 职责 |
| --- | --- | --- |
| 部署 | `.agents/skills/vllm-ascend-deployment/` | 规范 deployment input、静态 OOM/最小 TP 预检、服务启动、健康和模型身份验收、部署报告 |
| 压测 | `.agents/skills/vllm-ascend-benchmark/` | 解析 preset long-prefill workload、检查官方 CLI、执行 `vllm bench serve`、校验 raw result 和 benchmark 报告 |
| 编排 | `.agents/skills/vllm-ascend-inference-tuning/` | 冻结 SearchSpec、建立 baseline、串行派发 candidate、读取独立 Evidence、selection 和最终复验 |
| 任务合同 | `.agents/task/` 中由本目录资料复制的文件 | 提供模型身份、用户输入、负载、指标、预算和 provider 约束 |
| 运行控制 | Goal Plus 公开接口 | 管理 Goal、Search、candidate、设备、锁、verifier、Evidence 和 promotion |

candidate 的 edit surface 固定为任务合同声明的部署参数和允许的输出字段。三个 Skill、
模型、数据集、provider 资源、verifier、阈值和统计方法由既有合同提供；Main 使用
Goal Plus 公开接口完成设备调度、搜索和 benchmark 证据归档。

本任务的串行性质只由入口命令中的 `max_parallel=1`、SearchSpec 和 provider 事实共同
决定；普通进程顺序或静态文档属于辅助信息，完成状态以 provider 回执和独立 Evidence
为准。
