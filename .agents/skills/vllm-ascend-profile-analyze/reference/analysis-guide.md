# Profile 数据文件说明

本文只说明 ASCEND_PROFILER_OUTPUT 中各 CSV 文件和 communication_matrix.json 的结构、字段含义与单位，不定义分析流程、阈值或优化建议。

## step_trace_time.csv

记录每个 Device 的 step/stage 时间组成。

| 字段 | 含义 | 单位/备注 |
|---|---|---|
| Device_id | 采集数据对应的设备号 | 不等同于 global rank |
| Step | step 标识 | 推理场景可能为空 |
| Computing | 设备计算时间 | us |
| Communication(Not Overlapped) | 未与计算重叠的通信时间 | us |
| Overlapped | 与计算重叠的时间 | us |
| Communication | 通信总时间 | us |
| Free | 设备空闲时间 | us |
| Stage | stage 或 step 的总时间跨度 | 通常为 us |
| Bubble | 流水气泡时间 | 通常为 us |
| Communication(Not Overlapped and Exclude Receive) | 排除接收后的未重叠通信时间 | us |
| Preparing | 准备阶段时间 | 通常为 us |

## op_statistic.csv

按算子统计设备侧执行信息。

| 字段 | 含义 | 单位/备注 |
|---|---|---|
| Device_id | 采集数据对应的设备号 | 不等同于 global rank |
| OP Type | 算子名称或类型 | 文本 |
| Core Type | 执行核类型 | 如 AI_CORE、AI_VECTOR_CORE、AI_CPU、MIX_AIC、MIX_AIV |
| Count | 算子调用次数 | 次 |
| Total Time(us) | 该算子的累计执行时间 | us |
| Min Time(us) | 单次执行最小时间 | us |
| Avg Time(us) | 单次执行平均时间 | us |
| Max Time(us) | 单次执行最大时间 | us |
| Ratio(%) | 该算子时间占统计范围内算子总时间的比例 | % |

## api_statistic.csv

按 API 统计 Host 或设备侧 API 执行信息。

| 字段 | 含义 | 单位/备注 |
|---|---|---|
| Device_id | API 所属设备或 Host | Host 行可能使用 host 标识 |
| Level | API 所属层级或类别 | 例如 acl、node、communication |
| API Name | API 名称 | 文本 |
| Time(us) | 该 API 的累计执行时间 | us |
| Count | API 调用次数 | 次 |
| Avg(us) | 单次调用平均时间 | us |
| Min(us) | 单次调用最小时间 | us |
| Max(us) | 单次调用最大时间 | us |
| Variance | 单次调用时间的方差或工具提供的离散度指标 | 以工具实际输出定义为准 |

## communication_matrix.json

记录通信操作的链路级数据。顶层通常按 step 组织，step 下包含 collective 或 p2p 通信项；通信项内部再按 source-rank 到 destination-rank 的链路索引组织。

| 字段 | 含义 | 单位/备注 |
|---|---|---|
| Step | 通信发生的 step | 可能是数字或字符串 |
| Collective/P2P | 通信类型或通信项分组 | 具体名称以文件结构为准 |
| Rank-Link 索引 | 源 Rank 到目的 Rank 的链路标识 | 例如 1-0 |
| Transport Type | 传输类型 | 如 LOCAL、HCCS，具体枚举以文件为准 |
| Transit Size(MB) | 该链路传输的数据量 | MB |
| Transit Time(ms) | 该链路传输耗时 | ms |
| Bandwidth(GB/s) | 该链路测得的带宽 | GB/s |
| Op Name | 对应通信算子名称 | 文本 |

文件缺失、字段缺失或字段名称变化时，以当前 profile 实际结构为准，不根据本说明补造字段。
