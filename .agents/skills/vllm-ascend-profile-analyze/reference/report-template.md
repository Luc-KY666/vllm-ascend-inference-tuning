# 推理 Profile 瓶颈判定报告模板

## 采集信息

```markdown
## 推理 Profile 瓶颈判定报告 — {run_name}

### 采集信息
| 项目 | 值 |
|------|-----|
| profile 目录 | {prof_dir} |
| rank 范围 | {rank_scope} |
| rank / Device ID 映射 | {rank_device_mapping} |
| 并行组 | {parallel_group_info} |
| 关键环境变量 | {PYTORCH_NPU_ALLOC_CONF / ASCEND_LAUNCH_BLOCKING / HCCL_ALGO / ACLNN_CACHE_LIMIT 等} |
| 数据文件 | op_statistic.csv / step_trace_time.csv / api_statistic.csv / communication_matrix.json（缺失项标注） |
```

## 瓶颈类型判定

```markdown
### 瓶颈类型判定

| Rank | Device ID | Computing% | Communication% | Free% | Overlapped(us) | Bubble(us) |
|------|-----------|------------|----------------|-------|----------------|------------|
| {rank_0} | {device_0} | ... | ... | ... | ... | ... |
| {rank_1} | {device_1} | ... | ... | ... | ... | ... |

**跨 rank 差异**：{各指标 min / median / max、最大最小差及离群 rank}

**各 rank 定性**：{rank -> compute-bound / comm-bound / free-bound / 混合型}
**整体瓶颈类型**：{综合最差 rank、离群 rank 及其证据后的结论}
**判定依据**：{各 rank 命中 Step 2 规则表哪一条}
```

## Free 归因

```markdown
### Free 归因

| Rank | 请求间隔证据 | 结论 | 确认方法 |
|------|--------------|------|----------|
| {rank} | {evidence} | {conclusion} | {confirmation_or_none} |

**实现缺陷归因**（排除请求间隔后；若未排除则标"待确认"）：

| Rank | 排名 | 成因类别 | 累计耗时(us) | 占 host 总耗时% | 代表 API |
|------|------|----------|--------------|-----------------|----------|
| {rank_0} | 1 | {category} | {time} | {pct}% | {api_name} |
| {rank_0} | 2 | ... | ... | ... | ... |
| {rank_1} | 1 | ... | ... | ... | ... |
```

## Top-5 算子与算子族聚合

```markdown
### 各 rank Top-5 算子（按 Ratio%）

| Rank | OP Type | Total(us) | Count | Avg(us) | Core Type | Ratio% |
|------|---------|-----------|-------|---------|-----------|--------|
| {rank_0} | {op1} | ... | ... | ... | ... | ... |
| {rank_0} | {op2} | ... | ... | ... | ... | ... |
| {rank_1} | {op1} | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

**Count×Avg 结论**：{summary}

### 算子族聚合

| Rank | 算子族 | Total(us) | 合计 Ratio% | 对应模型结构 |
|------|--------|-----------|-------------|--------------|
| {rank} | {family} | {time} | {ratio} | {model_component} |

**Core Type 分布**：{summary}
```

## 通信分析

```markdown
### 通信分析

**带宽利用率**（communication_matrix）：

| Rank | 通信 op | Transport Type | Transit Size(MB) | Transit Time(ms) | 带宽(GB/s) | 是否带宽瓶颈 |
|------|---------|----------------|------------------|------------------|------------|--------------|
| {rank} | {communication_op} | {transport_type} | {size} | {time} | {bandwidth} | {conclusion} |

**结论**：{communication_summary}

**多卡快慢 rank 检查**：

| 指标 | 最快 rank / Device | 最慢 rank / Device | 差异率 |
|------|--------------------|--------------------|--------|
| Computing | ... | ... | ... |
| Communication(Not Overlapped) | ... | ... | ... |

**结论**：{存在明显快慢 rank / 无明显快慢 rank / 无法比较}
```

## 优化方向建议

```markdown
### 优化方向建议（按预期收益排序）

| 优先级 | 方向 | 预期收益 | 验证状态 | 落地 skill |
|--------|------|----------|----------|------------|
| 1 | {direction} | {高/中/低} | {status} | {skill_or_none} |
| 2 | ... | ... | ... | ... |
```

---
*分析时间: {timestamp}*
*profile 目录: {prof_dir}*
