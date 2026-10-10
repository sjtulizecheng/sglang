# RTX PRO 5000：FP8 KV 与原生 FP8 attention 实验

> 当前发布基于社区 v0.5.17 的 29481685，只包含 sjtushenhai 的优化提交；
> 同事的 14 个内部基础提交及其改动已移除。下文性能、配置和源码哈希是
> 之前组合环境的历史记录，不能据此宣称当前社区基线分支已达到 51%～52%。
> 当前提交与验证范围见 [社区基线发布说明](OPTIMIZATION_COMMITS_20261010.md)。


完整交付说明与后续 50% 目标进展见 [优化实现详解](OPTIMIZATION_DELIVERY_20261009.md)。

更新日期：2026-10-09。本文记录继续优化的实验，不替代
[已验证的容量结果](CAPACITY_OPTIMIZATION_5000PRO.md)。容量验收仍为
用例 `202609111724`、缓存扰动目标 24%、平均 E2E ≤1500 ms，基线
4.27 QPS；提升 40% 需要至少 5.978 QPS，同时保持请求成功和可接受的质量。

## 并发 8 的现有收益

已验证组合（NVFP4 routed MoE、FA4 单次 prefill、BF16 KV、FCFS、
running=16、PD 关闭、prefill-decode-interval=0）在并发 8 为
5.32／5.30 QPS，相对基线最高达标档提高 24.59%／24.12%。但平均 E2E
1503.15／1506.87 ms，两次均略超 SLA。不能将其记为达标容量提升。
基线摸高在并发 7 停止，没有相同并发 8 的基线数据。

最新隔离实验将融合 FP8 prefill 与原生 FP8 decode 组合，并发 8
达到 **5.78 QPS、平均 E2E 1382.72 ms、成功率 100%**，相对原始
最高达标吞吐提高 **35.36%**；并发 9 为 5.84 QPS、1537.25 ms，
因延迟超标停止。这是业务质量待验收的单次实验结果，尚未采用为
推荐配置，亦未达到 40% 目标。逐档结果见下文。

固定 interval=8 的独立 180 秒试验为 5.20 QPS、平均 E2E 1538.30 ms、
P95 2542.31 ms，940/940 请求成功。平均 TTFT 由 interval=0 复测的
379.17 ms 增至 654.16 ms；合并 prefill 的收益未抵消首 token 等待，
本轮不采用 interval=8。

## GPU 支持不等于当前软件路径支持

环境：RTX PRO 5000 72GB，SM120，CUDA 12.9，PyTorch 2.11.0+cu129，
Triton 3.6.0，FlashInfer 0.6.15.post1，CUTLASS DSL 4.6.0，
cuDNN 9.17.1 / frontend 1.27。当前模型使用 dense MLA，latent 512、
RoPE 64；materialized prefill Q/K/V head dimension 为 256。

| 路径 | 实际验证结果 |
| --- | --- |
| 已安装 FlashMLA dense FP8 | 实际运行报 `no kernel image is available for execution on the device`，错误来自 `sm90/dense_fp8/flash_fwd_mla_metadata.cu:76`；构建目标没有 SM120 |
| 名称包含 SM120 的稀疏 FlashMLA | 用于 DSA 等稀疏布局，不能直接代替当前模型的 dense MLA |
| 当前 FA4 SM120 | 使用 BF16/FP16；FA4 的现有 FP8 分支面向 SM100 |
| FlashInfer FA2 FP8 Q | 会转换成 FP16，不能据此声称原生 FP8×FP8 |
| cuDNN FP8 ragged/THD | 实测拒绝：`sdpa fp8 with THD not supported for sm120 yet` |
| cuDNN FP8 dense | 实测拒绝当前 head_dim=256；提示只支持 ≤128 或特定 192/128 组合，后者还要求更新版本 |
| cuDNN BF16 | 能运行，但测试形状下约 2.38–2.43 ms，慢于 FA4 的约 1.31 ms |

FlashInfer MLA 原 decode 代码在索引 attention 前对整层分配的 KV pool
调用 `.to(q.dtype)`。Q 为 BF16 时，仅设置 FP8 KV 会引入整池转换，
不代表 attention 使用原生 FP8 计算。

## 当前热点

对已优化服务采集的 40 步 profile 中，kernel 累计耗时分布为：

| 类别 | 占比 |
| --- | ---: |
| FA4 prefill attention | 34.77% |
| NVFP4 grouped GEMM | 18.73% |
| 其他 GEMM，包括 dense FP8 | 26.84% |
| FlashInfer MLA decode | 4.29% |

这是 kernel 时长之和，不能直接作为无重叠的端到端耗时分解。
即使 decode attention 微测快一倍，也不能推导出服务吞吐快一倍。
输入约 5000 token、输出约 20 token 的当前负载仍以 prefill 为重点。
profile 运行受采样开销影响，不纳入容量数据。

## 原生 FP8 原型

### Prefill

[fp8_prefill_sm120.py](../../python/sglang/kernels/ops/attention/fp8_prefill_sm120.py)
对 Q/K 动态量化，softmax 使用 FP32，QK 和 PV 的矩阵乘使用原生 FP8。
生成 PTX 中已确认 `mma.sync...f32.e4m3.e4m3.f32`。

Q=4096、K=5120、H=20、D=256 时，当前最好微测约 1.685 ms，
仍慢于 FA4 的约 1.313 ms；相对 L2 误差约 5.15%。寄存器占用较高，
部分 tile 配置发生 spill。该原型没有接入服务，也不属于推荐配置。

### Decode

[fp8_mla_decode_sm120.py](../../python/sglang/kernels/ops/attention/fp8_mla_decode_sm120.py)
直接读取当前 batch 的索引 KV，支持 unit-scale E4M3 的 512+64 dense
布局。Q 按 head 和分量动态量化，QK/PV 使用原生 FP8，softmax 和 split-K
合并使用 FP32；不接受稀疏 block-scaled KV 格式。

8 请求、上下文 2048–7093 token 的初始微测：

| 路径 | 时间(ms) | 对相同 FP8 cache 反量化后参考的相对 L2 误差 |
| --- | ---: | ---: |
| FlashInfer BF16，预先反量化缓存 | 0.0761 | — |
| 原生 FP8，16 splits | 0.0376 | 3.27% |
| 原生 FP8，32 splits | 0.0368 | 3.25% |

相对原始 BF16 cache 的总相对 L2 误差约 4.5%。短上下文初始微测较慢，
其中包含 Python/launch 开销。补充 CUDA graph 测量后，16 splits 的
长上下文时间为 **0.0326 ms**，FlashInfer BF16 为 **0.0744 ms**；
短上下文 17/127 token 分别为 **0.00768／0.00917 ms**。
2 项 GPU 测试通过，覆盖空 KV 行、变长和乱序物理索引、3/20/33 heads，
以及 CUDA graph 重放后更新 Q、索引和长度。随机张量数值检查不能代替
实际模型的生成质量验收。

新增实验开关 `SGLANG_SM120_FP8_MLA_DECODE` 默认关闭。接入范围限制为
SM120/SM121、page size 1、非 DCP、普通非推测 decode、BF16 Q 及
512+64 E4M3 KV；其他分支保持原实现。目前仅作为实验开关，生成质量验收完成前不推荐用于生产。

首次接入发现 CUDA graph 元数据错误：`fast_mla_decode_plan` 只刷新
FlashInfer 内部计划，不刷新 wrapper 的 `_kv_indptr_buf`。新 kernel
直接读取该旧缓冲区时，100 条 greedy 生成全部异常并达到 64 token
上限，已排除该次生成，未对它做容量压测。

修复为读取 indices updater 每轮写入的 `self.kv_indptr`；物理 KV
索引仍来自 wrapper 的实际索引缓冲区。新增 CPU 回归测试显式构造
“wrapper 偏移过期、实时偏移已更新”，验证选用正确缓冲区；并验证
开关关闭及推测分支继续走原后端。

该次异常运行的 teacher-forcing NLL 为 2.1273，仍接近基线，说明
只测 prefill 的参考回答评分无法发现 decode 接入错误。真实生成和
CUDA graph 元数据验证必须另外执行，不能用 NLL 正常替代。

修复偏移后的 100 条 greedy 生成中，12 条文本完全一致，平均 token
序列相似度约 64.54%；平均输出从 BF16 对照的 18.92 增至 20.23 token，
64 token 截断由 0 条变为 1 条，所有返回 logprob 均有限。重复输出
故障已消除，但生成差异仍需业务质量评估，**不能把 12% 文本一致率
当作 12% 准确率，也不能把这一版直接视为可上线的容量提升**。
这里 temperature=0、64 token 上限仅用于生成一致性检查，容量压测
仍使用原始采样参数和输出上限。

该 BF16 生成对照使用了实验 prefill tactics，FP8 修复版恢复原 tactics；
两侧虽是相同模型与输入，仍不能把文本差异全部归因于 FP8。后续若做
严格的生成质量 A/B，应同时固定 MoE tactics 和 batch 调度。

恢复 BF16 服务后，补测了使用相同原始 MoE tactic cache 的 100 条
greedy 对照：FP8 有 14 条文本完全一致，平均 token 相似度约 60.59%，
平均输出为 BF16 20.57 / FP8 20.23 token，截断为 0 / 1 条。另一个
重要对照是，两次 BF16 运行（tactics 与动态 batch 不同）也只有 17 条
文本完全一致。因此当前文本一致率不能单独量化 FP8 质量损失，业务
准确率仍需专门评分，而非将相似度直接换算成准确率。

上述容量实验结束时，服务恢复为 BF16 KV、FCFS、interval=0、PD 关闭、running=16；
两项新实验开关关闭，原始 tactic cache 内容核对一致，健康检查及
100 条生成通过，完成后再次清空 KV 缓存。

### FP8 修复版并发 8 实测

| 并发 | 状态 | QPS | 平均 E2E(ms) | P95 E2E(ms) | 成功率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | met | **5.42** | **1473.59** | 2493.81 | 100% |

180.43 秒完成 978 请求，平均输入 5080.43 token、平均输出 20.44 token，
日志 token 命中率 22.9999%。每次独立运行前清空 KV，缓存扰动目标仍为
24%。相对原始最高达标档 4.27 QPS 的吞吐提高 **26.93%**；相对
BF16 KV 并发 8 复测 5.30 QPS，增量为 **2.26%**，平均 E2E 下降
33.28 ms（2.21%），跨过了 1500 ms 阈值。

这是一次达到延迟 SLA 的**低精度实验**，生成质量尚待确认，不替代
原先已验证的 BF16 KV 推荐配置，也未达到 40% 吞吐目标。本次只测
并发 8，`stop_reason=max_concurrency_reached`，没有测试并发 9。

报告目录：
`/sgl-workspace/llm-benchmark/artifacts/bench-local/optimization-20261009/goal40-fp8decode-fixed-c8/20261009T074843Z-2411683230/`。

### 32 splits 独立运行：并发 8 达标，9 未达标

在真实输入微测后，将实验 kernel 的 split 数由 16 改为 32，保持
其他服务参数、用例、采样和 SLA 不变。GPU 数值、空 KV 行、CUDA
graph 元数据重放及分派检查共 4 项、5 个 subtests 通过后，独立清空
KV，运行并发 8→9，每档 180 秒。两档之间没有清空 KV。

| 并发 | 状态 | QPS | 成功 QPS | 平均 E2E(ms) | P95 E2E(ms) | 成功率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | met | **5.43** | 5.43 | **1471.75** | 2613.95 | 100% |
| 9 | unmet | 5.50 | 5.50 | 1636.26 | 2793.20 | 100% |

并发 8 完成 979 请求，平均输出 20.51 token；并发 9 完成 993 请求，
平均输出 19.95 token。整轮服务日志的 token 命中率为 23.2964%，
缓存扰动目标仍为 24%。成功 QPS 因全部请求成功，与报告 QPS 相同。

状态为 `completed`，`stop_reason=SLA unmet: sla_unmet`；本次
最高延迟达标并发为 8，最高尝试并发为 9。5.43 QPS 相对原始最高
达标吞吐 4.27 QPS 提高 **27.17%**，相对 BF16 KV 并发 8 的独立
复测 5.30 QPS 提高 **2.45%**。基线没有并发 8 数据，前一个比例
不是相同并发的 A/B。

32 splits 相对 16 splits 的 5.42 QPS 只多 0.01 QPS，不能证明
稳定的额外整机收益；因此保留实验 kernel 原默认 16 splits，32
仍可作为显式算子参数测试。并发 8 的两次 FP8 试验分别使用 16/32
splits，不能称为完全相同配置的重复测试。32 splits 本轮没有增加
业务质量评分，沿用实验性质，不替代 BF16 KV 推荐配置。

并发 8→9 的 QPS 只增加 1.29%，平均 E2E 增加 164.51 ms；简单
继续增加并发没有带来相称收益，也未达到 5.978 QPS 的 40% 目标。

报告目录：
`/sgl-workspace/llm-benchmark/artifacts/bench-local/optimization-20261009/goal40-fp8decode-s32-c8/20261009T081622Z-3649991418/`。

### 真实模型输入上的误差与耗时

为区分 KV 存储量化和 FP8 计算的影响，从同一批 8 个真实请求中采集
第 0、23、46 层的 Q、逻辑 KV、实际输出；上下文长度为 4112–7186。
采集使用关闭 CUDA graph 的诊断服务；离线重放的 FlashInfer BF16
输出在三层都与保存的服务输出逐位相同。以下计时另外使用 CUDA graph，
只测算子，不将诊断服务的耗时当成容量数据。

| 层 | 仅 FP8 存储的相对 L2 误差 | 原生 FP8 计算总相对 L2 误差 | BF16(ms) | FP8 16 splits(ms) | FP8 32 splits(ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.584% | 5.196% | 0.07420 | 0.04008 | 0.03173 |
| 23 | 3.071% | 6.589% | 0.07417 | 0.04016 | 0.03165 |
| 46 | 3.698% | 5.346% | 0.07419 | 0.04008 | 0.03156 |

误差均以原始 BF16 输出为参考，总误差列使用 16 splits。32 splits
总相对 L2 为 5.202%／6.586%／5.346%，与 16 splits 接近。
KV 最大绝对值为 4.875–7.875，没有超出 E4M3 范围；原生 FP8 输出
与 BF16 的平均余弦相似度约 0.9986–0.9988。这些是三层算子的数值
证据，不能换算成业务准确率，也不证明所有层或所有输入都无溢出。

原始张量保存在 `/tmp/chat-opt-20261009/real-attention/mla-{0,23,46}.pt`，
记录保存在 `goal40-fp8-real-decode-micro.json`。

### 固定参考前缀的 decode 质量对照

为排除自由生成分叉、MoE tactics 和动态 batch 的混淆，追加了串行
100 条固定参考回答对照。两侧使用原始 MoE tactic cache、开启 CUDA
graph、关闭 overlap scheduling；每一步先保存完整未修改 logits，再
强制采样相同参考 token，确保后续前缀相同。该设置只用于质量诊断，
不计入容量结果，也不改变服务压测的采样和输出上限。

共记录 1460 个参考 token：100 个来自 prefill 后的首次采样，1360 个
来自真实 decode。两次 BF16 对照的全部 1460 行 logits **逐位一致**。
输入 SHA256 为
`cc23a089c191b805c48c5290d70b89c99bd48cf48bc973e83cacbbdb612eafb1`。

| 范围 | token 数 | BF16 参考 NLL | FP8 参考 NLL | 平均 KL(BF16‖FP8) | 两侧 top-1 一致率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 1460 | 2.152681 | 2.149117 | 0.069026 | 88.36% |
| Prefill 首次采样 | 100 | 2.547257 | 2.511534 | 0.063708 | 91.00% |
| Decode | 1360 | 2.123668 | 2.122469 | 0.069417 | 88.16% |

全部 token 的 NLL 差为 -0.003564，按 case bootstrap 的 95% 区间为
[-0.028699, 0.020285]，包含 0；参考 token 排名第一的数量为 BF16
778/1460、FP8 773/1460。参考回答评分没有显示显著变化，但概率分布
存在差异，单 token 最大 KL 为 4.44065。**这不是业务准确率，也不能
据此保证自由生成质量等价。**较早的 prefill-only NLL 少计每条回答的
首个目标 token，统计 1360 个，不能直接与这里 1460 个 token 的均值混算。

补充 FP8 存储、QK/PV 使用 BF16 的受控对照：全部 token NLL 为
2.146985、decode NLL 为 2.120179，相对 BF16 的平均 KL 分别为
0.062796 / 0.062728；decode top-1 一致率为 87.43%。原生 FP8 的
decode KL 为 0.069417，说明仅 KV 存储量化已经带来明显分布差异，
不能把全部差异归因于原生 FP8 乘法。两种变体在 100 个 prefill
首 token 上的 logits 逐位一致。直接比较原生 FP8 与存储量化/BF16
计算对照，decode 平均 KL 为 0.053144、top-1 一致率为 89.34%；
KV 存储和计算两部分都存在影响，KL 不能相减或相加来分配贡献。
这种分解仍受约 0.25% 的算子实现误差影响，
不能作为完全精确的误差归因。

诊断挂载时机也单独核对：普通 forward hook 在 CUDA graph 捕获后
才注册，不能替换图里已经捕获的算子。首次采用该挂载方法的存储
对照在采集前停止，未产生有效对照或容量结果；修正后的独立启动
bootstrap 在 graph 捕获前安装指定变体，并记录进程、配置和源码
哈希。前述 BF16/原生 FP8 对照不依赖该 hook，不受此问题影响。

### Decode 精度分解与候选改进

对同一批真实第 0/23/46 层输入，在隔离原型中分别关闭 QK、RoPE、PV
的 FP8 计算。保留 64 维 RoPE 的 BF16 计算，latent QK 和 PV 仍用
FP8，微测时间维持约 0.0401 ms，同时降低数值误差：

| 层 | 全 FP8 相对 BF16 的 L2 | BF16 RoPE 相对 BF16 的 L2 | 全 FP8 相对仅存储量化参考的 L2 | BF16 RoPE 相对仅存储量化参考的 L2 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 5.196% | 4.589% | 1.973% | 0.596% |
| 23 | 6.589% | 3.363% | 5.671% | 1.158% |
| 46 | 5.346% | 3.864% | 3.776% | 1.305% |

QK/PV 全部用 BF16、只保留索引 FP8 KV 存储的变体约 0.0799 ms，
比原生 FP8 慢；它相对 FlashInfer 的相同存储参考仍有 0.224%–0.259%
实现误差。Q residual 双 FP8 dot 约 0.069–0.083 ms，没有速度优势。
这些都是算子诊断，尚未修改生产默认，也没有新增整机容量结论。

对 BF16 RoPE 变体也完成同样的 100 条固定参考对照。全部 token NLL
为 2.156405，相对 BF16 的差为 +0.003724，case bootstrap 95% 区间
[-0.020746, 0.028025]，仍包含 0。decode NLL 为 2.130293，平均
KL 为 0.063580，低于原生全 FP8 的 0.069417；但 decode top-1
一致率为 86.91%，低于全 FP8 的 88.16%。相对存储量化/BF16 计算
对照的 decode KL 为 0.048522。算子误差减小并未让所有模型指标
一致改善，因此该变体仍留在隔离实验中，没有替换默认或新增容量结论。

### Prefill 后续试验

FP8 prefill 拆分 V 输出 tile 的最好结果约 1.84 ms，同次 FA4 对照
约 1.32 ms；转置 V 布局的试验为 3.37–8.20 ms，并出现较多寄存器
spill。两种修改都没有采用，原型文件已恢复。

FA4 将 Q 常驻寄存器、复用 shared memory 的试验，数值正常的候选
为 1.404–1.789 ms，对照为 1.385 ms；其中一个双 stage 配置相对
L2 达 0.788，直接排除。没有候选同时满足正确性和性能收益要求，
未修改生产 FA4 的参数。

随后在 `/tmp` 中单独实现 CuTe FP8 prefill 原型，QK/PV 分别使用
FP8 warp MMA 的 K32/K16 指令布局，显式用两次 warp shuffle 将
softmax 概率的 accumulator 布局转成 PV 的 A operand 布局；输出
保留 BF16。直接复用 BF16 的寄存器 reshape 会产生 NaN 或 L2>1，
该错误版本已经排除。128×64 tile 配合 256 threads 避免了严重 spill。
带动态缩放的原型另外强制重新编译并保存 PTX，确认包含
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` 和
`mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32`，确实使用
SM120 的 FP8×FP8 指令。

合成 Q=4096、K=5120、H=20、D=256，128×64 tile 的 kernel 时间为
**1.0761 ms**，同次 BF16 对照为 **1.3867 ms**，算子耗时减少约
22.4%，相对 L2 为 5.304%。短 ragged 输入则约 0.0207–0.0272 ms，
慢于 BF16 的 0.0110 ms。此前共享内存 P 版本为 1.621 ms，4 次
shuffle 版本约 1.165 ms，均未作为最终候选。

这里的 Q/K/V 使用 unit-scale E4M3 转换，**合成计时没有包含量化
转换，尚未接入服务**。不能将 22.4% 算子收益当成整机 QPS 收益；
需要真实输入、量化开销、生成质量和同口径容量验证。

随后采集第 0/23/46 层实际 prefill，Q=K=5093、H=20、D=256；
离线 BF16 重放均与保存的服务输出逐位一致。128×64 FP8 原型
在三层约 1.14–1.15 ms，直接转换计入后约 1.316 ms，对照
BF16 约 1.45–1.46 ms，耗时减少约 9%–10%。但第 0 层 V 最大
绝对值仅 0.04346，unit-scale 转换使 16.76% 的元素由非零变为零，
该层 attention 相对 L2 达 7.872%，不能直接用于服务。

增加 GPU 上的 per-head 动态缩放，QK 中应用 Q/K scales，输出应用
V scale 后，第 0 层 V 的量化相对 L2 从 11.38% 降为 2.65%，
attention 总相对 L2 降为 4.615%。第 23/46 层 attention L2 为
5.793%/6.735%，并非所有层都更接近 BF16。动态 scale 的 reduction
及量化计入后，时间为 **1.361–1.372 ms**，同次 BF16 为
**1.440–1.445 ms**，净耗时减少仅约 **5%–6%**。这是三层算子
结果，尚未形成整机收益。下一步若继续该方向，需要将量化与现有
Q/K/V 生成或 gather 融合，并处理精度；仅替换 attention 不足以
支持 40% 容量提升的结论。

### 融合 K 拼接、分块量化及共享内存复用

继续检查模型准备流程后，发现 20 heads、192+64 维 K 的现有路径
先分别复制 K-nope 和 K-RoPE，再进入 attention。隔离原型将 K 拼接
与 K/V 的 FP8 量化合成一个 kernel；Q 另按 query tile 量化。
scales 按序列、head、tile 保存，避免先对整层输入计算全局最大值。
K scale 在 softmax 前应用，V scale 改变时同步重缩放累加输出，
以保持 online softmax 的量纲一致。

原先为了 BF16 输出，将 Q 的共享内存扩大了一倍。新候选在最后
一次 V 读取完成并经过线程块同步后，用整个 Q/K/V 区域存放输出，
减少同时存活的共享内存需求。64×64 候选比 128×64/128×128 更快；
更大 tile 本身没有带来额外收益。

以下计时均包含 Q 量化、K 拼接/KV 量化及 attention；BF16 对照包含
原有的两次 K 复制和 attention。输入仍为同一组真实 Q=K=5093：

| 层 | BF16 准备+attention(ms) | 融合 FP8 准备+attention(ms) | 相对 BF16 输出 L2 |
| --- | ---: | ---: | ---: |
| 0 | 1.5398 | 1.2781 | 4.309% |
| 23 | 1.5570 | 1.2889 | 6.006% |
| 46 | 1.5613 | 1.2933 | 5.602% |

这部分合计耗时减少约 **17%**。短 ragged 输入仍较慢：BF16 包含
K 拼接约 0.0169 ms，FP8 约 0.0285 ms，因此服务候选需保留短输入
回退。独立检查覆盖不对齐的序列边界、部分 tile、变化较大的 V
scale、零 V、重复运行和 CUDA graph 更新 Q 后重放；CUDA memcheck
为 0 errors，racecheck 为 0 errors / 0 warnings。上述仍是算子
结果，不能直接计入服务 QPS 或业务质量。

随后通过隔离 hook 接入实际服务：保留原有 RoPE、KV 写入、前缀
gather 和 KV projection，只替换满足条件的单次 prefill 准备及
attention。总 query token 少于 1024 或其他不满足条件的输入仍走
BF16 prefill。Decode 沿用原生 FP8 实验；prefill 为 eager 执行。
将量化 kernel 的 tile 总数改成运行时参数，避免每个请求长度触发
新的 Triton 编译。这个改动后重新通过数值及 graph smoke；上述
memcheck/racecheck 对应改动前版本，没有混称为同一源码检查。

固定参考检查覆盖 100 条用例、1460 个 token，CUDA graph 开启、
overlap 关闭；在强制参考 token 前保存完整 logits。BF16 KV、仅
FP8 decode、再加入融合 FP8 prefill 的平均 NLL 分别为
2.152681、2.149117、2.131334。新候选相对 BF16 的 NLL 差值
95% case-bootstrap 区间为 [-0.046647, 0.002527]，包含 0；
decode top-1 一致率为 86.76%，平均 KL 为 0.075180。
这些结果不能证明生成质量提高或等价。相对仅 FP8 decode，新增
prefill 的 decode top-1 一致率为 87.87%，平均 KL 为 0.076642。

另有 20 条自由生成 smoke 全部完成，平均输出 17.5 token，没有
触及诊断用 64-token 上限。该检查复用了参考运行后的 KV 前缀，
只能说明基本生成流程可运行；它不是冷缓存容量测试或业务质量
验收。容量测量保留原始采样和输出上限，不使用这套诊断限制。

### 融合 FP8 prefill 的整机容量结果

容量服务重新开启 overlap，关闭 custom logit processor，decode
CUDA graph 保持开启，原生 FP8 decode 使用默认 16 splits。
9 条预热生成覆盖单请求及并发 4，无截断；这些请求以新服务的空
KV 开始，后续自然复用前缀，完成后再次清空 KV 才开始容量测量。
日志确认真实请求使用融合 prefill，短输入保留 BF16 回退。

本地运行 `20261009T093220Z-3379976948` 沿用同版本 100 条用例，
缓存目标 24%，每档 180 秒、超时 60 秒，未改变采样或输出上限。

| 并发 | 状态 | QPS | P95 E2E(ms) | 平均 E2E(ms) | 成功/请求数 | 平均输出 token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | met | 5.78 | 2395.56 | 1382.72 | 1042/1042 | 19.71 |
| 9 | unmet | 5.84 | 2688.27 | 1537.25 | 1056/1056 | 19.81 |

最高满足延迟 SLA 的实验并发为 8，最高尝试为 9，stop_reason 为
`SLA unmet: sla_unmet`。日志 token 命中率为 23.3054%，统计包含
工具预检；没有因快速失败产生虚假吞吐。相同并发 8，相对 BF16 KV
独立复测的 5.30 QPS 提高 **9.06%**；相对仅原生 FP8 decode 的
5.42 QPS 提高 **6.64%**。相对原始最高达标 4.27 QPS 提高
**35.36%**，仍低于 40% 目标所需的 5.978 QPS。各阶段百分比不能
相加，采样长度波动及业务质量差异也尚未通过重复实验排除。

报告目录：
`/sgl-workspace/llm-benchmark/artifacts/bench-local/optimization-20261009/goal40-union-prefill-c8/20261009T093220Z-3379976948/`。
此候选通过 `/tmp` hook 接入隔离服务，没有写入生产实现或推荐启动
配置。后续采用前仍需业务质量验收与同口径容量复测。

## Dense FP8 GEMM 与另一条 MoE 路径

对真实输入的五种 dense projection，在 M=2048/4096/4652/8192/16384
下比较原调度与不使用 Stream-K 的调度。只对 K≤2048 的形状采用
后者时，五种 projection 合计加速约 0.84%–4.06%；K=5120 的
o_proj 在部分形状上仍以原调度更快。大多数短 K 输出逐位一致，
有差异的相对 L2 不超过约 3.6e-6。该结果是独立算子微测，尚未
修改生产调度策略或计入服务 QPS。

已安装 FlashInfer 另有面向 SM120/SM121 的 `B12xMoEWrapper`，
与现有仅面向 SM100 系列的 `CuteDslMoEWrapper` 不同。其 NVFP4
输入布局可接受 gate/up 拼接权重，并提供 CUDA graph workspace；
但当前环境实际构造时报错：
`b12x fused MoE requires CUDA 13 or later. Current CUDA version: 12.9.`
进一步在隔离进程中调用私有 `launch_sm120_moe`，底层 kernel 可以在
当前 CuTe DSL 环境编译和运行；没有修改已安装包或版本检查。因此
上述错误是高层 API 的实际限制，不能据此断言底层 kernel 无法运行。

合成输入使用 E=64、top-k=4、K=2048、N=1536、单位 global scale，
对相同 NVFP4 权重做 CUDA graph 微测：

| token 数 | CUTLASS(ms) | b12x(ms) | b12x 相对 CUTLASS 输出 L2 |
| --- | ---: | ---: | ---: |
| 8 | 0.13944 | 0.11631 | 3.007% |
| 64 | 0.32234 | 0.31544 | 2.765% |
| 512 | 0.34771 | 0.41064 | 2.801% |
| 4096 | 0.83923 | 1.14202 | 2.782% |

该对照没有加载服务的原始 autotune cache，输入也不是模型的真实
激活，不能用它估计服务收益。大 prefill 形状更慢，未采用。

还验证了一个接入风险：现有在线 NVFP4 权重使用非单位 global
weight scale，而 b12x 入口把 `w1_alpha` 同时传作 FC1 输入 scale。
合成的非单位权重 scale 约 4.4e-5 时，直接复用 CUTLASS combined
alpha，虽输出有限，相对参考的 L2 仍达到 **4052.44**，最大输出
3792，对照仅 0.957。单独为输入传单位 scale，并通过 weight views
保留实际 GEMM alpha 后，相对 L2 回到 **3.103%**。这说明不能仅
放宽后端约束、直接替换函数；需要单独适配 scale 合约、workspace
和路由，并做真实模型质量与服务验证。目前没有接入服务。

## MoE prefill autotune

原 FlashInfer tactic cache 只覆盖 1/2/4/8/16 token 的 decode batch。
新增实验在现有 autotune 上下文内使用一个代表层覆盖 prefill token
预算，同形状的 MoE 层复用结果，避免为每一层重复调优。

实际 SM120 启动成功，新增 tactic 覆盖到 16384 token，随后 CUDA graph
捕获通过。并发 8、180 秒实测为 **5.22 QPS、平均 E2E 1531.01 ms、
P95 2576.81 ms、942/942 请求成功**，平均输出 20.11 token，日志
token 命中率 22.9802%。没有超过原组合的 5.30–5.32 QPS，不采用为默认。

实验由 `SGLANG_SM120_NVFP4_PREFILL_AUTOTUNE` 控制，默认关闭；启用时
使用独立 tactic cache key。已备份实验 tactics 并恢复原缓存，避免后续
实验无意中沿用它。3 项 CPU 测试覆盖 token 预算、同形状层复用、
路由合法性、非目标硬件/并行配置跳过，以及实验与默认 cache 隔离。

报告目录：
`/sgl-workspace/llm-benchmark/artifacts/bench-local/optimization-20261009/goal40-prefilltune-c8/20261009T073734Z-333440171/`。

## 原始证据位置

原始记录位于 `/tmp/chat-opt-20261009/`：

[本轮汇总 JSON](results/goal40_sm120_20261009.json) 保存 5 项有效服务
实验、原始 summary、服务器配置与源码哈希、微测和质量诊断。

| 内容 | 文件 |
| --- | --- |
| 优化版 profile 汇总 | `goal40-profile-summary.json` |
| FlashMLA 失败及后端整池转换诊断 | `goal40-progress.json` |
| cuDNN ragged / dense 验证 | `goal40-cudnn-attention.json`、`goal40-cudnn-dense-attention.json` |
| FP8 prefill 微测 | `goal40-fp8-prefill-micro.json`、`goal40-fp8-prefill-vhead.json` |
| FP8 decode 初始微测 | `goal40-fp8-mla-decode-micro.json` |
| FP8 decode CUDA graph 微测 | `goal40-fp8-mla-decode-graph.json` |
| 真实 MLA 输入重放及 FP8 存储/计算误差 | `goal40-fp8-real-decode-micro.json` |
| 真实 dense GEMM 输入调度微测 | `goal40-dense-real-micro.json` |
| FA4 Q 寄存器试验 | `goal40-fa4-qregs-micro.json` |
| FP8 prefill 后续试验 | `goal40-fp8-prefill-splitv.json`、`goal40-fp8-prefill-vtrans.json` |
| SM120 b12x 高层 API、私有 kernel 与 scale 合约试验 | `goal40-b12x-moe-probe.json`、`goal40-b12x-private-micro.json`、`goal40-b12x-scale-probe.json` |
| 同原始 MoE tactics 的生成对照 | `goal40-fp8decode-fixed-original-tactics-comparison.json` |
| BF16 生成自身的变化 | `goal40-bf16-generation-variation.json` |
| 固定参考 decode 的 BF16 自身重复、原生 FP8 和存储对照 | `goal40-reference-bf16-sync-b-comparison.json`、`goal40-reference-fp8-sync-a-comparison.json`、`goal40-reference-storage-pregraph-a-comparison.json` |
| 原生 FP8 与存储对照直接比较 | `goal40-native-vs-storage-reference-comparison.json` |
| BF16 RoPE 的固定参考对照 | `goal40-reference-rope-pregraph-a-comparison.json`、`goal40-rope-vs-storage-reference-comparison.json` |
| Decode 精度分解微测 | `goal40-decode-precision-variants.json` |
| CuTe FP8 prefill 合成与真实输入 | `goal40-cute-fp8-shuffle2-micro.json`、`goal40-cute-fp8-real-prefill-micro.json`、`goal40-cute-fp8-scaled-real-prefill-micro.json` |
| CuTe FP8×FP8 PTX 指令确认 | `goal40-cute-fp8-ptx-confirmation.json` |
| 融合分块量化及共享内存复用微测 | `goal40-blockscaled-prefill-micro.json`、`goal40-union-prefill-micro.json` |
| 融合 prefill 的固定参考质量及生成 smoke | `goal40-reference-union-dynamic-a-comparison.json`、`goal40-reference-union-vs-bf16-comparison.json`、`goal40-union-prefill-generation-smoke.json` |
| 融合 prefill 的容量预热及完整汇总 | `goal40-union-prefill-capacity-warmup.json`、`goal40-union-prefill-c8-collected.json` |
| autotune 原缓存备份 | `autotune-before-prefill.json` |

interval=8 报告目录：
`/sgl-workspace/llm-benchmark/artifacts/bench-local/optimization-20261009/goal40-interval8-c8/20261009T071759Z-3704231553/`。

本轮追加试验后，服务恢复为 BF16 KV、NVFP4 routed MoE、FA4 单次
prefill、FCFS、interval=0、PD 关闭、running=16。CUDA graph 和
overlap scheduling 已开启，诊断 hooks、自定义 logit processor
已关闭，两项实验开关关闭。最新恢复服务标识为
`goal40-union-bf16-restored`，健康检查及生成健康检查通过，再次清空
KV；源码哈希、原始 MoE tactic cache 和原有 43 个文件的暂存内容
已核对一致。此次追加工作只修改文档、汇总 JSON 和 `/tmp` 诊断原型，
没有将新精度变体或 CuTe prefill 原型接入推荐服务。

文档只记录参数与结果，不保存可执行启动或压测命令。
