# RTX PRO 5000：优化内容、实测收益与压测边界

> 当前发布基于社区 v0.5.17 的 29481685，只包含 sjtushenhai 的优化提交；
> 同事的 14 个内部基础提交及其改动已移除。下文性能、配置和源码哈希是
> 之前组合环境的历史记录，不能据此宣称当前社区基线分支已达到 51%～52%。
> 当前提交与验证范围见 [社区基线发布说明](OPTIMIZATION_COMMITS_20261010.md)。


完整交付说明与后续 50% 目标进展见 [优化实现详解](OPTIMIZATION_DELIVERY_20261009.md)。

更新日期：2026-10-09。硬件为 RTX PRO 5000 72GB Blackwell（SM120），
模型为 `chatrl-glm-flash-0625-fp8`，服务模型名为 `chat`。

## 结论

在用例版本 `202609111724`、缓存扰动目标 24%、平均 E2E ≤1500 ms 的
条件下，当前推荐的 BF16 KV 配置最高已验证达标并发由 **6 提高到 7**，对应实测吞吐由
**4.27 提高到 5.18 QPS，提升 21.31%**。独立清空 KV 缓存后的并发 7
复测为 **5.15 QPS，提升 20.61%**，两次成功率均为 100%。

主要收益来自 **SM120 上的 NVFP4 MoE 适配、FA4 前缀 attention 单次计算**，
并配合最大运行请求数从 6 调到 16。当前配置使用 FCFS，PD 关闭。
这是组合配置的收益，不能全部归因于 PD、量化或某一行参数。

并发 7 的三次测量均达标，吞吐为 5.14／5.18／5.15 QPS。并发 8 的
两次结果为 5.32／5.30 QPS，但平均 E2E 为 1503.15／1506.87 ms，
均略超 SLA。**当前已验证的达标范围到并发 7；并发 8 是临界未达标档。**
这不是所有数据和线上流量下不变的容量上限。

并发 8 **测到了吞吐收益**：5.32／5.30 QPS 相对基线最高达标档
4.27 QPS 分别提高 **24.59%／24.12%**。但平均 E2E 分别超出 SLA
3.15／6.87 ms，因此不能将约 24% 写成“达标容量提升”。这也不是
同并发 8 的 A/B 对照：基线在并发 7 不达标后停止，没有测并发 8。
同一次优化版摸高中，并发从 7 增到 8，QPS 仅从 5.18 增到 5.32
（提高 2.70%），平均 E2E 则从 1349.85 增到 1503.15 ms。

**后续实验补充**：FP8 KV + 原生 FP8 MLA decode 的 16/32 splits
试验在并发 8 分别测到 **5.42／5.43 QPS、平均 E2E
1473.59／1471.75 ms、成功率 100%**，相对基线吞吐提高
**26.93%／27.17%**，两次延迟 SLA 均达标。32 splits 继续测并发 9，
得到 5.50 QPS、平均 E2E 1636.26 ms，因 SLA 不达标停止。
两次 FP8 试验的 split 参数不同，且生成质量仍待评估，未计入
本文已验证推荐配置的收益；实验开关默认关闭。详见
[FP8 实验与质量记录](SM120_FP8_ATTENTION_ANALYSIS.md)。

换成相同并发 8 比较，FP8 的 5.42／5.43 QPS 相对 BF16 KV 独立
复测的 5.30 QPS，增量为 **2.26%／2.45%**，平均 E2E 降低
**33.28／35.12 ms**。因此大部分已测收益来自前面的 NVFP4 MoE +
FA4 组合；FP8 decode 的额外收益较小，但使并发 8 跨过延迟门槛。
后续固定参考前缀对照已验证 BF16 重复结果逐位一致，FP8 参考 NLL
接近 BF16，但仍存在概率分布差异；尚不能把参考评分替代业务质量
验收。

**最新的融合 FP8 prefill 实验**已接入隔离服务，与原生 FP8 decode
组合后，并发 8 测得 **5.78 QPS、平均 E2E 1382.72 ms、成功率
100%**。相对原始最高达标档提高 **35.36%**，相对 BF16 KV 的
并发 8 复测提高 **9.06%**。继续到并发 9 为 5.84 QPS、平均 E2E
1537.25 ms，因 SLA 不达标停止。它证明并发 8 可以继续测出收益；
但新组合只有一次容量测量，业务质量尚未验收，仍不替代上面的
推荐配置结论，也尚未达到 40% 目标。

## 主要优化了什么

| 项目 | 实际修改及作用 | 当前是否采用 |
| --- | --- | --- |
| SM120 NVFP4 MoE | 将在线 NVFP4 接到本卡可用的 FlashInfer CUTLASS 后端，加载时转换 routed MoE 权重，降低该部分计算和访存成本 | 是，主要收益来源 |
| FA4 兼容性修复 | 修复接口、共享内存、输出路径和变长 tile 越界问题，使 FA4 能在本卡正确运行 | 是，是采用 FA4 的前提 |
| FA4 单次前缀 attention | 当完整 KV 能放入临时容量时，一次计算缓存前缀与新输入，减少两段 attention、LSE 输出和合并开销 | 是，在 NVFP4 基础上进一步提高吞吐 |
| 最大运行请求数 6 → 16 | 增加服务端运行槽位，允许更大的运行 batch；属于配置调整 | 是；不代表并发 16 能满足 SLA |
| PD 剩余输出估计 | 同一输出上限的历史样本不足时，合并历史估计剩余长度，改善无显式输出上限时的学习能力 | 代码已修复；本场景未测到吞吐收益，PD 关闭 |
| 原生 FP8 MLA decode | 直接读取当前 batch 的索引 FP8 KV，QK/PV 使用 FP8，避免整池反量化；修复 CUDA graph 下 KV 偏移过期问题 | 实验开关默认关闭；并发 8 延迟达标，生成质量待验收 |
| 融合 FP8 prefill | 合并 K 拼接与 K/V 分块量化，Q 按 tile 量化，attention 复用已结束使用的 Q/K/V 共享内存；短输入回退 BF16 | 隔离原型；与 FP8 decode 组合在并发 8 达到 5.78 QPS，业务质量待验收 |
| 共享专家融合 | 试验将共享专家也并入 NVFP4 MoE | 不采用；没有超过独立 FP8 共享专家 |

### NVFP4：让本卡用上更低精度的 MoE 计算

原始 `nvfp4_online` 的参数约束只允许 SM100/SM103 上的 TRTLLM 路径，
直接在本卡打开无法运行。本次为 SM120/SM121 增加
`flashinfer_cutlass` 后端选择与校验，复用已有 ModelOpt NVFP4 权重打包及
scale 处理，实现实际模型加载、推理和 CUDA graph 捕获。

转换对象是 routed MoE：加载时从源 FP8 权重转为 NVFP4。dense 层及独立
共享专家仍走 FP8，源模型文件保持原样。CUTLASS 路径使用动态 block
activation scale 和单位 global scale，不等同于经过校准的激活量化。

并发 7 的 NVFP4 + FlashInfer 筛选结果为 4.96 QPS、平均 E2E 1409.21 ms。
它相对基线最高达标档的 4.27 QPS 提高约 16.16%，但同时使用了 running=16，
所以不能把全部 16.16% 当作量化的独立贡献。

### FA4：先保证正确运行，再减少前缀计算的额外步骤

在 CUDA 12.9、CUTLASS DSL 4.6.0 环境中修复了四个实际问题：

1. 根据实际 NVVM `fmax` Python 函数签名选择调用方式，避免用 CUDA
   toolkit 版本推断 DSL 接口。
2. SM120 复用 SM80 MMA/cp.async 实现时，明确使用普通输出路径，避免
   错误选择未实现的 TMA 输出。
3. 对本模型 head_dim=256 使用 64×64 tile，占用 96 KiB shared memory；
   原 128×64 配置需要 128 KiB，超过本卡 99 KiB 限制。
4. 变长 tile scheduler 的 grid 可能包含无效 tile；先检查有效性，再读取
   `cu_seqlens`，防止越界访问。

仅修复 FA4 后，attention kernel 微测快约 8%–10%，但整模型筛选没有
证明吞吐收益。进一步发现，SM120 原来的 FA4 路径一直拆开计算
prefix/new attention，还要输出 LSE 并合并结果。

新增分支在 SM120 的普通 extend、已知长度且完整 KV 总量不超过临时
容量时使用 `MHA_ONE_SHOT`：一次因果 attention 完成计算。长度未知、
容量不足或其他 forward mode 继续使用原有分支。

同为 running=16、独立共享专家的并发 7 筛选中，
NVFP4 + FlashInfer 为 4.96 QPS，NVFP4 + FA4 单次路径为 5.14 QPS，
提高约 **3.63%**。这是单档筛选的增量对照；最终组合另做完整摸高和
独立复测确认。各阶段百分比不能直接相加。

内部 tile 调参还比较了其他形状：128×32 更慢，32×32 的相对 L2 误差
达到 22%–27%，均未采用；生产默认保留数值通过的 64×64。

### PD：修复估计问题，但本批负载没有验证出收益

本版用例平均输入约 5000 token，平均输出约 20 token。GPU profile 中，
prefill attention 占 kernel 累计耗时 30.6%，Triton MoE 占 29.2%，
QKV-A 回退 GEMM 约 1.4%。这些比例是 kernel 时长之和，不是无重叠的
端到端耗时分解。PD 筛选日志记录 prefill 约 141.2 秒、decode 约 39.6 秒，
说明优化输入计算更值得优先投入。

PD 时间片只能改变同一 GPU 上 prefill/decode 的执行顺序，不能消除这部分
输入计算。原 PD 配置在并发 7 为 4.21 QPS、平均 E2E 1659.04 ms，未达标。

同时发现：未指定 max_tokens 时，不同请求获得不同的上下文剩余上限。
旧预测器按上限精确分组，至少需要 8 个同组样本；实测 512 个历史样本
分散到 275 种上限，最大同组只有 7 个，预测器可能一直停在冷启动。
修复后在同组样本不足时合并历史，以 Kaplan–Meier 生存估计处理截断样本，
同时保留当前请求真实的输出长度上限。

修复版筛选仍只有 4.13 QPS、平均 E2E 1694.18 ms，未证明吞吐收益。
所以保留修复代码，当前推荐配置关闭 PD。早期 P95 5000 ms 的长输出
测试中 PD 有尾延迟收益，详见[历史调度优化汇总](OPTIMIZATION_SUMMARY_5000PRO.md)；
其用例和验收口径不同，不能与本轮 QPS 直接比较。

## 涉及的代码

| 作用 | 文件 |
| --- | --- |
| SM120 NVFP4 后端选择、参数约束及说明 | [overrides.py](../../python/sglang/srt/arg_groups/overrides.py)、[server_args.py](../../python/sglang/srt/server_args.py) |
| 在线 NVFP4 与 CUTLASS 方法衔接、activation scale 处理 | [nvfp4_online.py](../../python/sglang/srt/layers/quantization/nvfp4_online.py)、[modelopt_quant.py](../../python/sglang/srt/layers/quantization/modelopt_quant.py) |
| FA4 接口、输出、tile 和越界修复 | [utils.py](../../python/sglang/kernels/ops/attention/flash_attn/cute/utils.py)、[interface.py](../../python/sglang/kernels/ops/attention/flash_attn/cute/interface.py)、[flash_fwd.py](../../python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd.py)、[flash_fwd_sm120.py](../../python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm120.py) |
| FA4 单次前缀 attention 分支 | [attention_backend_handler.py](../../python/sglang/srt/models/deepseek_common/attention_backend_handler.py) |
| PD 历史样本合并估计 | [pd_slo_controller.py](../../python/sglang/srt/managers/pd_slo_controller.py) |

使用当前工作区代码，并显式启用下面的 NVFP4／FA4 配置，才会进入本轮主要
优化路径。原来的 FP8、全 FlashInfer、running=6 启动配置不能获得这里
验证的组合收益。优化版源码哈希记录在汇总数据中；基线为干净的
`b1478c429` worktree。

## 推荐配置与原始配置

这里只记录参数，完整启动命令按用户要求在对话中提供，不写入文档。

| 参数或行为 | 原始基线 | 当前推荐 |
| --- | --- | --- |
| 源模型 | chatrl-glm-flash-0625-fp8 | 相同 |
| routed MoE 量化 | 源 FP8 | `quantization=nvfp4_online` |
| MoE 后端 | 原 FP8 路径 | `moe-runner-backend=flashinfer_cutlass` |
| Prefill attention | FlashInfer | `prefill-attention-backend=fa4` |
| Decode attention | FlashInfer | `decode-attention-backend=flashinfer` |
| 最大运行请求数 | 6 | 16 |
| 调度策略／PD | FCFS／关闭 | FCFS／关闭 |
| 固定 prefill-decode-interval | 0 | 0 |
| 共享专家 | 独立 FP8，关闭融合 | 独立 FP8，关闭融合 |
| chunked-prefill-size | 16384 | 16384 |
| TP／mem-fraction-static | 1／0.9 | 1／0.9 |

以下六个环境变量沿用原始基线，不计作本轮新增优化：

| 环境变量 | 值 |
| --- | --- |
| SGLANG_OPT_MOE_QUANT_ONCE | 1 |
| SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL | /opt/trtllm_gen_moe_cubin_pool |
| SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD | 256 |
| SGLANG_GLM_FUSED_GATE_TOPK | 1 |
| SGLANG_MOE_ROUND0_SINGLETON | 1 |
| SGLANG_GLM_QKVA_BF16 | 1 |

NVFP4 不适用原 FP8 quant-once 路径；即使保留该环境变量，逐层日志也会
标为 INELIGIBLE，因此不能将其重复计入 NVFP4 的收益。

## 压测口径与完整结果

使用本地 `bench-local-linux-amd64`，通过 HTTP 请求服务；本轮不是新建
平台任务，下面的运行标识是本地报告目录。100 条用例的 SHA256 为
`e0e3eb43e8fe6d2efe33e3f7b5e95607e605c9846eb5d20a01a952cfdab5ffd2`。

用例版本 `202609111724`，场景 default，每档 180 秒、请求超时 60 秒。
完整摸高从并发 4 开始，每档加 1、最大并发设为 0，首次 SLA 不达标即停止。
本轮 SLA 是**平均 E2E ≤1500 ms**，不使用 P95 5000 ms 的旧验收口径。
成功率另行核查；表内有效实验均为 100%，排除了快速失败造成的虚假吞吐。

请求使用流式输出、关闭 thinking、min_p=0.05；temperature、top_p 和
输出上限保留模型／工具原有行为，没有为提高 QPS 缩短输出上限。
没有启用确定性推理，输出长度存在采样波动。

缓存扰动目标为 0.24，位置 last_user、长度 8。每次独立实验先清空
KV 缓存；同一次摸高中不在档位之间额外清缓存。清空后，本轮请求间仍
可自然复用前缀。24% 是工具的前缀字节比例目标，不是承诺的 token 命中率。
基线／完整优化运行的服务日志 token 命中率分别为 23.30%／23.45%。
日志统计包含健康预检，不把它们全部当作压测请求；报告中缓存字段为 0
也不能替代服务日志所测的 token 命中率。

### 基线与优化版逐档对照

| 并发 | 基线状态 | 基线 QPS | 基线平均 E2E(ms) | 基线 P95(ms) | 优化状态 | 优化 QPS | 优化平均 E2E(ms) | 优化 P95(ms) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 4 | met | 4.04 | 988.26 | 1740.94 | met | 4.83 | 827.57 | 1417.77 |
| 5 | met | 4.14 | 1208.20 | 2066.38 | met | 4.94 | 1011.77 | 1726.01 |
| 6 | met | 4.27 | 1403.52 | 2450.34 | met | 5.02 | 1193.74 | 2102.73 |
| 7 | unmet | 4.27 | 1636.86 | 2703.72 | met | **5.18** | **1349.85** | 2377.92 |
| 8 | — | — | — | — | unmet | 5.32 | 1503.15 | 2544.63 |

两侧已测档位成功率均为 100%；“—”表示未测。完整运行状态均为
completed，stop_reason 均为 `SLA unmet: sla_unmet`。
completed 只表示任务结束，达标以每档 met/unmet 判断。

按报告显示值，最高达标吞吐提升为 `(5.18 / 4.27 - 1) × 100% = 21.31%`。
相同并发 6 下，吞吐提高 17.56%，平均 E2E 从 1403.52 降至 1193.74 ms，
下降 14.95%。两侧最高达标档平均输出为 19.39／19.88 token，
没有出现靠显著减少输出获得这部分吞吐的情况。

### 方案筛选与复测

下表除并发 8 复测外均为并发 7、每档 180 秒。各实验以清空 KV 缓存开始。

| 方案／运行 | 并发 | 状态 | QPS | 平均 E2E(ms) | P95 E2E(ms) | 成功率 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 原 PD + running16 | 7 | unmet | 4.21 | 1659.04 | 2786.41 | 100% |
| 合并历史估计 PD + running16 | 7 | unmet | 4.13 | 1694.18 | 2845.37 | 100% |
| FP8 + 修复后的 FA4，尚无单次路径 | 7 | unmet | 4.19 | 1668.37 | 3009.63 | 100% |
| NVFP4 + FlashInfer | 7 | met | 4.96 | 1409.21 | 2476.48 | 100% |
| NVFP4 + 融合共享专家 | 7 | met | 4.89 | 1431.61 | 2487.63 | 100% |
| NVFP4 + FA4 单次路径，候选筛选 | 7 | met | 5.14 | 1361.58 | 2268.38 | 100% |
| 最终组合独立复测 | 7 | met | **5.15** | **1359.69** | 2325.00 | 100% |
| 最终组合从并发 8 继续摸高 | 8 | unmet | 5.30 | 1506.87 | 2674.34 | 100% |

FA4 尚无单次路径的筛选包含约 6 秒首次 JIT，不能作为充分预热后的严格
后端比较。更早的 FA4 启动失败和服务崩溃试验已排除；其中 0% 成功率、
快速失败产生的高 QPS 不用于容量判断。

并发 7 独立复测共 927 请求，平均输出 20.12 token，日志 token 命中率
22.9728%；相对基线提升 20.61%。结合候选筛选及完整摸高，三次并发 7
吞吐范围为 5.14–5.18 QPS，提升范围为 20.37%–21.31%。

## QPS 还能不能继续压测

后续的 interval 调整、SM120 FP8 attention 兼容性和算子原型试验，
另见 [SM120 FP8 实验记录](SM120_FP8_ATTENTION_ANALYSIS.md)。实验结果
分别记录；尚未证明的收益不并入本文约 21% 的达标容量结论。

可以继续验证，但应把“符合延迟要求的容量”和“更高并发下的吞吐”分别
记录。本次已追加从并发 8 开始的摸高，参数和服务配置保持不变，先清空 KV
缓存，再按每档加 1、首次不达标停止的规则执行。

新增运行 `20261009T070008Z-2866784783` 于 07:00:08–07:03:08 UTC 完成，
共 958 请求，全部成功；平均输出 20.19 token，日志 token 命中率 22.9865%。
并发 8 为 **5.30 QPS、平均 E2E 1506.87 ms**，比 SLA 高 6.87 ms，
因此 stop_reason 为 `SLA unmet: sla_unmet`，没有进入并发 9。
本次续测自身没有 met 档，不能把早先的并发 7 当作本次新测结果。

已有两次并发 8 的平均 E2E 都略高于 1500 ms，说明它处于边界附近。
现有结果支持并发 7，尚不支持“并发 8 稳定达标”；也不应把只高几毫秒
解释为硬件绝对上限。后续值得做的验证是：

- **延长稳定性测试**：保持配置、SLA、采样和缓存规则，复测并发 7／8，
  延长单档时长，观察结果波动及慢请求分布。
- **验证固定发送 QPS**：用按固定速率到达的压测逐步增加请求发送速率，
  观察排队、平均／尾延迟和成功率，才能决定线上适合承诺的固定 QPS。
- **继续优化后再摸高**：进一步减少 prefill attention 与 MoE 耗时，再用
  相同口径从边界附近复测；不通过缩短输出或提高缓存目标制造收益。

### 最新实验：融合 FP8 prefill + 原生 FP8 decode

本地 HTTP 运行 `20261009T093220Z-3379976948`，时间为
09:32:20–09:38:21 UTC；起始并发 8、步长 1、最大并发 9。
用例版本、SHA256、24% 缓存扰动目标、180 秒时长、60 秒超时及
原始采样/输出上限保持不变。编译预热后清空 KV，档位之间不额外
清缓存。CUDA graph 和 overlap 开启，custom logit processor 关闭。

| 并发 | 状态 | QPS | 成功 QPS | P95 E2E(ms) | 平均 E2E(ms) | 成功率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | met | 5.78 | 5.78 | 2395.56 | 1382.72 | 100% |
| 9 | unmet | 5.84 | 5.84 | 2688.27 | 1537.25 | 100% |

最高满足延迟 SLA 的实验并发为 8，最高尝试并发为 9；任务 completed，
stop_reason 为 `SLA unmet: sla_unmet`。两档分别 1042/1056 请求，
全部成功；成功 QPS 在全成功时与 QPS 相同。平均输出分别为
19.71/19.81 token。服务日志统计 token 命中率为 23.3054%，包括
工具预检请求，与既有测试的缓存口径一致。

相同并发 8，相对仅 FP8 decode 的 16-split 试验，QPS 从 5.42
提高至 5.78，增量 **6.64%**。相对原始 4.27 QPS，提高 **35.36%**；
这是不同最高达标档的吞吐比较，不是原始版并发 8 的 A/B。
采样输出长度仍有波动，单次结果不能将所有差异完全归因于算子。
40% 目标为 5.978 QPS，仍比本次最高达标吞吐高约 **3.43%**。

新 prefill 路径及 hook 仅保留在 `/tmp/chat-opt-20261009/`，尚未纳入
推荐服务。固定参考评分和生成 smoke 已完成，但概率分布存在差异，
不能替代业务质量验收。完整配置、源码哈希、报告和质量记录见
[FP8 实验文档](SM120_FP8_ATTENTION_ANALYSIS.md)及其汇总 JSON。

本轮控制的是同时在途请求数，QPS 是实测完成吞吐。5.18 QPS 不等于已经
验证以固定 5.18 次/秒持续发送的流量也满足 SLA。若只为观察更高并发下的
吞吐而放宽停止规则，需要另记实验条件；超出 SLA 的结果不能计入达标容量。

## 精度、启动耗时与验证记录

NVFP4 改变数值精度。对同一份 100 条参考回答、1360 个 token 做
teacher-forcing 输入 logprob 评分：

| 配置 | 平均 NLL | 困惑度 |
| --- | ---: | ---: |
| FP8 + FA4 对照 | 2.108203 | 8.233435 |
| NVFP4 + FlashInfer | 2.126114 | 8.382231 |
| NVFP4 + 融合共享专家 | 2.130518 | 8.419228 |
| 最终 NVFP4 + FA4 单次路径 | 2.131568 | 8.428073 |

最终组合的 NLL 比 FP8 + FA4 对照增加约 1.11%。这是有限参考回答上的
数值质量检查，**不能解释成“业务准确率下降 1.11%”**；完整业务准确率
仍需单独验收。评分输入 SHA256 为
`cc23a089c191b805c48c5290d70b89c99bd48cf48bc973e83cacbbdb612eafb1`。

在线量化使模型加载由约 6.13 秒增加到 37.08 秒；模型显存从 29.25 GiB
降到 18.01 GiB，减少约 11.24 GiB。服务会继续按配置分配 KV cache，
所以不能把模型显存降幅当作服务总显存降幅。

已完成的检查：

- NVFP4 后端约束及 scale：80 项 CPU 测试、33 项 subtests 通过，
  1 项跳过；实际 SM120 模型加载、CUDA graph、推理与压测通过。
- FA4 数值与越界保护：2 项 GPU 测试、4 项 subtests 通过，覆盖
  head_dim=128/256、变长请求、不同 Q/K 长度、因果 mask、LSE 和无效 tile。
- FA4 单次分支：2 项 CPU 测试、4 项 subtests 通过，并完成实际模型
  cached prefix 路径、参考回答评分和完整压测。
- PD 估计修复：19 项 CPU 测试通过。
- 10 个本轮修改或新增的非 vendor Python 文件通过 Black、isort；
  测试注册及 Git diff 空白检查通过。CuTe vendor 按仓库规则排除格式化，
  已验证实际 GPU 路径。测试服务记录的源码哈希与工作区一致。

## 数据索引

[汇总 JSON](results/capacity_5000pro_20261009.json) 保留 10 个有效实验的
原始 summary、配置、诊断、源码哈希与质量检查汇总；失败实验另列排除原因。
下面目录均相对于
`/sgl-workspace/llm-benchmark/artifacts/bench-local/optimization-20261009/`，
每个目录内的 `summary.json` 为该次原始结果。

| 用途 | 报告目录 |
| --- | --- |
| 干净基线完整摸高 | `base-r6/20261009T053736Z-2531650138` |
| 最终组合候选筛选 | `nvfp4-fa4oneshot-r16-c7/20261009T063300Z-2718732020` |
| 最终组合完整摸高 | `nvfp4-fa4oneshot-full/20261009T063624Z-1408890550` |
| 并发 7 独立复测 | `nvfp4-fa4oneshot-repeat/20261009T065236Z-2053606025` |
| 并发 8 续测 | `nvfp4-fa4oneshot-continue-c8/20261009T070008Z-2866784783` |

原始日志、逐条质量评分和 profile 位于 `/tmp/chat-opt-20261009/`，
其中 `profile-base-summary.json` 记录 profile，
`fa4-tile-micro.json` 记录 tile 对照；微测不参与容量统计。

早期本地／平台运行也测到基线并发 6 达标、7 未达标，达标 QPS 约
4.23／4.22，但未随报告记录完整服务源码版本，故本轮百分比统一以
干净 `b1478c429` 对照的 **4.27 QPS** 为分母。
