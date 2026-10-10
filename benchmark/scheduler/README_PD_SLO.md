# 根据 E2E SLO 分配 prefill / decode 时间

> 当前发布基于社区 v0.5.17 的 29481685，只包含 sjtushenhai 的优化提交；
> 同事的 14 个内部基础提交及其改动已移除。下文性能、配置和源码哈希是
> 之前组合环境的历史记录，不能据此宣称当前社区基线分支已达到 51%～52%。
> 当前提交与验证范围见 [社区基线发布说明](OPTIMIZATION_COMMITS_20261010.md)。


已验证的 cold-v4 在单卡 RTX PRO 5000 72GB、chatrl-glm-flash-0625-fp8 上，
每轮清缓存后实现达标速率 2.6 QPS。相同参数关闭 PD 时为 2.5 QPS，独立代码收益
约 4%；相对原始 running6 的 2.2 QPS，参数加代码整体提高 18.2%。
详细逐轮证据见 [优化汇总](OPTIMIZATION_SUMMARY_5000PRO.md)。3 QPS 仍未达标。
cold-v5 未提高达标 QPS，工作区已恢复 cold-v4；running24 对照因用户接手服务而中止，未作为推荐。

## 调度逻辑

pd_slo_ms=5000 显式启用 PDSLOController；默认 0，保持原调度路径。
该功能在同一 GPU 上安排 P/D 执行顺序，并非多 GPU 的 P/D 分离。

控制器在构建 prefill batch 前读取当前运行请求和可接纳的等待请求，计算请求年龄、
未缓存输入工作量、预计剩余输出。prefill 和 decode 的耗时来自异步 CUDA event，
不为决策增加设备全局同步。prefill 用实际 token 数拟合，decode 按 batch 大小学习。

cold-v4 根据完成当前 decode 所需时间与剩余 SLO 时间之比设置 decode 份额，最高 80%。
prefill 执行后按该份额累积 decode 时间额度，实际 decode GPU 时间抵扣额度。
这能在连续新请求到达时保留生成进度，而不是每轮都优先接纳 prefill。
额度之外最多评估两个 decode 步，比较先 decode 再 prefill 与立即 prefill 的
预计完成代价；评价包含请求 E2E、接近 SLO 的代价以及超时代价。
预测期间随输出完成缩小 batch，避免持续高估后续 decode 耗时。

连续推迟 prefill 的时间在调度点限制为 min(500 ms, SLO × 10%)。无 decode 工作
时立即允许 prefill，并重置额度。已有 KV 准入、请求槽位、取消和完成路径继续生效；
不会丢弃超时请求、提前结束输出或抢占已经提交的 GPU forward。

输出长度仅根据已完成请求在线学习，按“已生成到当前位置”估算剩余量；样本不足时
使用输出上限作为保守边界。不会预读未来请求或实际最终输出。API 接入时间通过 IPC
保留，以包含分词和服务端排队；客户端等待和网络耗时仍由最终 HTTP E2E 验收。

未采用的 cold-v5 验证假设：即便额外 lookahead 只有两个步骤，也可能在 CUDA graph padding
批大小边界连续偏向 decode。候选版本只允许这两个额外步骤完成某个请求时才延迟
prefill，普通 decode 进度仍由时间额度保障。单元测试复现了旧问题，但实测仍只有
2.6 QPS 达标，3 / 2.7 QPS 的 P95 为 5947 / 5075 ms，因此保留 cold-v4。

## 代码位置

| 文件 | 职责 |
| --- | --- |
| [pd_slo_controller.py](../../python/sglang/srt/managers/pd_slo_controller.py) | 估计耗时与剩余输出、时间额度、P/D 顺序及等待上限 |
| [scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 准入前决策、缓存匹配、GPU 计时、完成样本和统计 |
| [schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | 传递计时事件和完成观测标记 |
| [req_time_stats.py](../../python/sglang/srt/observability/req_time_stats.py) | 跨进程保留 API 接入时间 |
| [server_args.py](../../python/sglang/srt/server_args.py) | 参数和不兼容配置校验 |
| [控制器测试](../../test/registered/unit/managers/test_pd_slo_controller.py) | 超时、估计、时间额度和预测边界 |
| [接入测试](../../test/registered/unit/managers/test_scheduler_pd_slo.py) | 实际 scheduler 与 radix cache 路径 |

当前仅支持 CUDA、单 GPU、普通自回归生成；不与 speculative、mixed chunk、
Prefill Delayer、固定 decode interval、PD multiplexing 或多 GPU 配置叠加。

## 压测口径和证据

使用用户原 auto_qps.py HTTP runner，CHAT 数据、200 请求、两轮、非流式，
P95 E2E ≤5000 ms 且成功率 ≥99%。每个 repeat 开始前检查 /v1/loads 并调用
/flush_cache，失败即退出；首轮不达标仍按原脚本提前停止，不强行补第二轮。
外部脚本已增加 HTTP 模式的 flush-cache 支持，改动存为
[auto_qps_http_flush_cache.patch](auto_qps_http_flush_cache.patch)。
保留默认 1 条 warmup、temperature=0.6、top_p=1、输出上限 128 和输入内容。

每轮 KV cache 清空；同轮请求之间可自然命中前缀，实测约 22%。清 KV cache 不会
重置控制器学习统计，因此另做全新进程仅测试 2.6 QPS 的确认。未启用确定性推理，
逐请求 sampling seed 不生效；各轮同时记录输出 token 总量，不把输出波动当成代码收益。

[5000pro_cold_slo.json](results/5000pro_cold_slo.json) 包含通过及失败轮次、运行源码
哈希和清缓存证明。原始逐请求结果、完整源码快照、服务状态和 cache-validation.json
位于 /sgl-workspace/hyg-llm-benchmark-cases/bench-report-auto/cold-slo-5000pro/。
PD 关闭时 2.5 QPS 的证据来自 cold-v3-off-r16-1，仅未启用的控制器文件不同，
其余五份运行源码与 cold-v4 相同。cold-v4-verification.json 校验四个 v4 序列的
六份源码、benchmark 源码、请求内容和
18 项服务设置一致，仅有 pd_slo_ms 这一处有意差异。

server_info 的 internal_states[].pd_slo_stats 可核对实际决策与 GPU 时间。
fresh cold-v4 开启序列记录 decode_credit=691、额外 decode=916、prefill=302，
确认实际执行了新增代码。有限样本及 0.1 QPS 附近的边界不能作为所有流量的保证。

旧 bench_pd_slo.py、summarize_pd_slo.py 及热缓存 JSON 保留用于历史复核，
其结果不参与当前冷缓存结论。完整运行命令直接在交付消息中提供。
