# RTX PRO 5000 的冷缓存调度调优

> 当前发布基于社区 v0.5.17 的 29481685，只包含 sjtushenhai 的优化提交；
> 同事的 14 个内部基础提交及其改动已移除。下文性能、配置和源码哈希是
> 之前组合环境的历史记录，不能据此宣称当前社区基线分支已达到 51%～52%。
> 当前提交与验证范围见 [社区基线发布说明](OPTIMIZATION_COMMITS_20261010.md)。


当前已验证 cold-v4：原始 running6 的达标速率 2.2 QPS，running16/PD 关闭为
2.5 QPS，running16/PD 开启为 2.6 QPS。整体提升 18.2%，独立 PD 提升 4%。
每轮均清空 KV cache，P95 E2E ≤5000 ms、成功率 ≥99%。3 QPS 尚未达标。
cold-v5 未提高达标 QPS，已恢复 cold-v4；running24 对照因用户接手服务而中止，未作为推荐。详见 [优化汇总](OPTIMIZATION_SUMMARY_5000PRO.md)
和 [PD 实现说明](README_PD_SLO.md)。旧热缓存结果不计入当前验收。

环境为单卡 RTX PRO 5000 72GB Blackwell，模型目录
/data/models/chatrl-glm-flash-0625-fp8，TP=1。原模型、六个内核环境变量、
attention backend、显存比例和日志频率保持一致。

已验证配置为 FCFS、chunked_prefill_size=16384、max_running_requests=16、
prefill_decode_interval=0、pd_slo_ms=5000。提高运行请求数减少准入等待；
控制器分配 P/D GPU 时间，避免生成被连续 prefill 打断。
launch_5000pro.sh 的 e2e-slo 对应该配置，e2e 为 PD 关闭的参数对照，baseline
保留原 running6。默认 ServerArgs 仍为 FCFS、interval=0、PD 关闭。
完整启动与压测命令直接在交付消息中提供。

## 保留的社区策略

| 功能 | 上游来源 | 适配行为 |
| --- | --- | --- |
| prefill-decode-interval | [#35017](https://github.com/sgl-project/sglang/pull/35017)，6f69f927da9e5692bb4709821faecff6be9b5a8a | prefill 后给 decode 固定轮数；DP 使用同步的 extend 标志 |
| hrrn | [#32911](https://github.com/sgl-project/sglang/pull/32911)，4df5df911ba9f1d1ffdeb1516e106f2df691fbdf | 按等待期间处理的 prefill token 数与未缓存工作量排序 |
| shortest-prefill-first | [#40024](https://github.com/sgl-project/sglang/pull/40024)，65ef55e2a8c51be0523723fcb98d7138b5839ebb | 匹配缓存后按未缓存输入排序，为短请求预留 continuation 预算 |
| Prefill Delayer | [#23189](https://github.com/sgl-project/sglang/pull/23189) | 本分支原有，保留队列和毫秒上限扩展 |

适配保留 NextBatchPlan、extend_range 和 KV 准入接口。新请求工作量从输入、
重算输出和实时缓存命中计算；continuation 预留预算计入请求槽位，至少保留一页进度。
已有未完成 chunk 时不能创建第二个，host cache load-back 后重新核对。
HRRN 的计数不包含 decode 残留 extend 元数据、推测验证或 mixed 中的 decode token。
单卡无可 decode 请求时不空等固定 interval，DP 仍遵守全局同步顺序。

SPF 需要 prefix cache，不含等待老化；HRRN 队列超过 128 或关闭缓存时回退 FCFS。
两者均不与显式 priority scheduling 叠加。本次推荐使用 FCFS，不能将并发或 PD
控制器收益归因于 SPF/HRRN 的移植。

## 数据分布与未采用的方案

按 seed=42 选取的 200 条请求，模板编码输入均值约 3677、中位数 3931、P95 5789、
最大 7039 token；输出上限 128，实际输出随采样变化。清缓存后本轮自然 token
命中率约 22%，绝大部分输入仍需 prefill。

此前比较过 4K chunk、SPF/HRRN、固定 decode 间隔、mixed chunk 和 prefill delayer。
它们在已测组合中没有成为推荐。尤其 SPF/4K 会增加分块和长输入等待；不能把其他
硬件或负载下的收益直接套用。提高运行请求上限也有代价，必须用最终 E2E 验证。

cold-v2 的较小动态 prefill token 上限、cold-v3 的固定 decode 份额下限和更长
lookahead 均未证明独立代码收益。cold-v4 取消固定份额下限，并限制额外 lookahead；
cold-v5 进一步检查完成边界，未提高达标 QPS，未保留。

## 结果与验证

[5000pro_cold_slo.json](results/5000pro_cold_slo.json) 汇总每轮的 QPS、P95、平均
E2E、输出总量、成功数和清缓存证明。原始基线由干净 b1478c429 worktree 启动；
PD 对照使用相同配置和源码，仅切换开关。开启后另做新进程验证，避免先前学习状态
影响首次启动判断。原始数据位于外部 benchmark 的 cold-slo-5000pro 目录。

外部 auto_qps.py 已支持 HTTP 模式 flush-cache，每个 repeat 前清理缓存，
相应测试 33 passed、15 subtests。补丁存于 auto_qps_http_flush_cache.patch，
外部 benchmark 目录不是本仓库的一部分。默认首轮失败时提前停止第二轮。

bench_5000pro.sh 与 bench_pd_slo.py 是此前实验辅助脚本；旧报告里的冷/热口径
不能代替每轮清缓存的新验收。旧 JSON 保留为历史记录。非流式只验证 E2E，
不能据此声称 TTFT 或 TPOT 改善。
