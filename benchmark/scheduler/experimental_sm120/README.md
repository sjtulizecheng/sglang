# SM120 GLM prefill 实验

> 发布范围（2026-10-12）：在 `b8835cd473` 之上增量提交，保留社区基线与既有分项优化。
> 本文 7.73～7.74 QPS 来自包含同事优化的原本地环境（基点 `b1478c4293`），
> 不是本次公开分支的整机实测。当前分支不包含同事的内部提交或其代码。
> 新增提交、测试范围与历史结果口径见 [本次交付索引](../OPTIMIZATION_COMMITS_20261012.md)。

这些模块通过显式 `forward-hooks` factory 启用，供本地性能与数值验证使用。
普通的 `sglang serve` 启动不会自动导入它们。它们会替换当前进程中的
SGLang 方法，尚未作为通用推理后端或生产 API 提供。

默认 factory 在 CUDA graph 捕获后安装，适用于这里的 eager prefill 优化。
需要改变 decode 图内核的实验必须在 hook spec 中显式指定
`capture_phase: "before_graph"`；新 factory 和回调必须支持图捕获。
这项显式选项保留了普通观察 hook 的默认安装时序。

## 模块与范围

| Factory / 模块 | 行为 |
| --- | --- |
| `tensorcore_router:create_hook` | 安装已有 FP8 Attention/projection hook；对 M≥32 的 BF16 Router 使用 Tensor Core GEMM，保持 FP32 累加和输出。 |
| `nvfp4_dense_prefill:create_hook` | 在上述基础上，对 M≥1024 的 Q-B、KV-B、O、共享专家 gate/up 和 down 投影使用 NVFP4。 |
| `nvfp4_qkva_prefill:create_hook` | 进一步转换 prefill 的 QKV-A；将 N=1344 补零到 1408，返回原有 1344 列的视图。 |
| `nvfp4_dispatch.py` | 复用公共 CUTLASS 调优结果，缓存 runner、workspace 和 tactic；不修改 GEMM 内核。 |

前三个 `create_hook` factory 已包含其前序 hook，一次选择一个。
INT8 LM head 和 routed MoE 的独立 factory 可与 dense factory 组合。小输入继续
执行原方法。权重在首次适用调用时转换一次并保留原张量，激活在每次调用时
量化；源模型文件不被修改。当前实现面向静态权重推理，不支持原地更新权重
后复用转换缓存。

## 已验证本地组合：真实路由 MoE 与 INT8 LM head

`nvfp4_moe_prefill.py:install_tactics` 使用真实 prefill 操作数筛选出的
GEMM1/GEMM2 tactic `(12,42)`。沿用原 NVFP4 权重、激活量化和独立
finalize，只改变 GEMM tile、操作数方向及调用分发。仅用于 SM120、当前
FlashInfer 0.6.15.post1 构建、TP=EP=1、64 专家、top-k=4、hidden=2048、
中间维度 1536、1024～16384 个输入 token；其他输入走原路径。
两次 GEMM 对应 `swap_ab=true`、`256x128x64B`。ID 与构建绑定，
启动时检查 SM120、版本和 tactic 数量，预热时逐层检查原实现与候选输出。

真实第 1/23/46 层、3737/5093/5855 token 共 9 组微测，完整 MoE
耗时降低 6.53%～9.81%，输出逐位一致。另有 12 组 1024/1025/8192/16384
token 检查，修改激活、expert IDs、路由权重后的 CUDA Graph 重放与
默认实现仍逐位一致；8192/16384 样本使用重复的真实输入行。
该 factory 在图捕获之后安装，只改变 eager prefill。

最快的融合 finalize 配置 `(12,52)` 在上述输入上约降低 11%～14%，
但输出相对 L2 约 0.33%～0.38%，本次服务候选使用 `(12,42)`。
默认 `use_fused_finalize=true` 仅允许融合 tactic；本次抓到的默认
`(-1,-1)` 实际选择独立 finalize，重复 10 次输出逐位一致。

`lm_head_int8.py:install_lm_head` 是已测 INT8 LM-head 实现的整理副本，
计算函数及 Triton 装饰器的 AST 与原实验完全相同。它须单独使用
`capture_phase="before_graph"`；dense prefill 与上述 MoE factory
保留默认 `after_graph`。INT8 权重仅用作紧凑存储，计算时精确转 BF16，
经 BF16 Tensor Core、FP32 累加后乘按行 scale。其质量检查见下文。

首轮完整服务：并发 11、7.73 QPS、平均 E2E 1422.22 ms、P95 2458.79 ms，
1395/1395 成功；平均输出 20.27 token，实际 token cache hit 23.22%。
与原始 4.27 QPS 相比约 +81.0%，与上一组 7.58 QPS 相比约 +1.98%。
独立重启复测为 7.74 QPS、平均 E2E 1420.48 ms、P95 2437.82 ms，
1395/1395 成功，平均输出 20.19 token、实际 token cache hit 23.23%。
两轮都先完成九请求预热再清 KV，每档 180 秒、并发 11；均超过
7.686 QPS（原始 ×1.8）且满足平均 E2E≤1500 ms。原始基线是并发 6
的达标吞吐，此处是达标吞吐的比较，不是相同并发下的单项 A/B。

本次证据位于 `/tmp/chat-opt-20261010-goal80/`：
`moe-routed-summary.json`、`moe-routed-graph-check.json`、
`moe-routed-service-validation.json`、`moe-routed-result.json`、
`moe-routed-repeat-result.json`。汇总副本保存于
`../results/goal80_sm120_20261011.json`。
所有 46 层在正式压测前通过输出逐位一致检查；预热后清 KV，再开始
180 秒压测。本次保留原有 Attention、dense NVFP4 和 INT8 LM-head 配置。

## 已验证环境与配置

- RTX PRO 5000 72GB，SM120，TP=1；GLM MoE Lite，47 层、20 heads、
  materialized Attention head dimension 256。
- SGLang 0.5.17 本地工作树、PyTorch 2.11.0+cu129、FlashInfer
  0.6.15.post1、CuTe DSL 4.6.0。
- 依赖已有的 `projection_tuning` 和 `materialized_fp8_*` 模块。它们位于配套 FlashInfer 的
  `flashinfer/experimental/mla_fp8_sm120/`；
  仅将该实验模块目录加入模块搜索路径，避免覆盖已安装的 FlashInfer 包。
- `nvfp4_dispatch.py` 调用 FlashInfer 的私有接口，其他版本需重新验证。
  `nvfp4_gemm_sm120_autotune.json` 是本卡五类 dense 投影的调优缓存；
  未覆盖的 QKV-A 形状使用库的默认 tactic。

完整服务同时使用 `nvfp4_online` routed MoE、`flashinfer_cutlass` MoE
runner、FA4 prefill、FlashInfer decode，以及 `fp8_e4m3` KV cache。
本地原有六个优化环境变量保持开启，额外设置
`SGLANG_SM120_FP8_MLA_DECODE=1`、
`SGLANG_SM120_NVFP4_PREFILL_AUTOTUNE=0`。
已测配置为 running=16、chunked prefill=16384、静态显存比例 0.9、
prefill/decode interval=4，并为 decode batch 1～16 各捕获一张 CUDA graph；
关闭 shared experts fusion。诊断评分使用的 custom logit processor 和
disable overlap 设置不用于性能压测。

当前实验采用的 hook 配置字段如下。这是配置说明，不是启动命令。

| 字段 | 验证值 |
| --- | --- |
| `minimum_query_tokens` | 1024 |
| `router_validation_calls` / `router_topk` | 0 / 4；独立 Router 预热验证已完成 |
| `dense_validation_calls` | 1 |
| `dense_activation_global_scale` | 16 |
| `dense_scale_overrides` | `[2048,5120]` 的 scale=512；`[2048,1536]` 的 scale=1 |
| `dense_cached_dispatch` | true |
| `dense_autotune_cache` | 本目录 `nvfp4_gemm_sm120_autotune.json` 的绝对路径 |
| `qkva_activation_global_scale` / `qkva_validation_calls` | 16 / 1，仅 QKV-A factory 使用 |

上述 global scale 来自本模型采样诊断，不是通用参数。尤其不能将共享专家
down 投影沿用为统一 scale=16：已经观察到部分激活超出其表示范围。

## 现有证据与限制

统一用例为 202609111724，缓存扰动目标 24%，每档 180 秒；先完成固定
九请求预热，再清空 KV。以成功请求和平均 E2E≤1500 ms 为约束。

- 原始已验证吞吐 4.27 QPS；本轮起点复测 6.50 QPS。
- Router 候选：同并发 9，6.50→6.62 QPS。
- dense NVFP4 与配置组合目前最好的一轮：并发 11，7.55 QPS，平均
  E2E 1453.81 ms，1364/1364 成功。相对原始约 +76.8%；尚未独立复测。
- 加入 QKV-A 后的首轮结果：并发 11，7.70 QPS，平均 E2E 1428.17 ms，
  1388/1388 成功，相对原始约 +80.3%。平均输出 token
  从 20.19 变为 19.73，不能将全部吞吐增量归因于 GEMM 提速。
  QKV-A 微测的约 54% 耗时下降使用真实权重、合成激活。
- 从本目录启动的独立摸高：并发 11 为 7.62 QPS、1398.48 ms，
  1377/1378 成功；并发 12 为 7.75 QPS、1547.46 ms，1397/1397 成功。
  最高达标并发 11，停止于 SLA 未达标；没有再次达到 80% 吞吐增幅。
- interval=2 的独立并发 11 复测最终为 6.78 QPS、平均 E2E 1419.80 ms，
  1370/1371 成功。180 秒发压后仍需等待请求结束，总计 202.17 秒；不能用
  发压窗口的进度 QPS 代替这个最终吞吐。interval=2 未带来已验证收益。
- 两次独立复测各有一条请求在约 446～520 ms 收到首 token、60 秒读流
  超时，来自同一用例；尾部日志显示服务继续生成数千 token，没有运行异常。
  原始压测未设置 max_tokens，暂不能确定是输出分布变化还是其他原因。
  限 256 token、每次清 KV 的五次重放，以及保留公共前缀的 64 次定种子
  串行诊断均正常 stop，后者最长 41 token。诊断结果不计入容量，也不能
  证明问题消失；固定参考评分不能检测自由生成中的长输出问题。
- 随后的并发 11 定种子诊断在 264 次请求中捕获 1 次持续重复音符 `♪`，
  最终触及 256-token 诊断上限。已确认该候选可以出现重复生成；尚不能仅凭
  这一结果确认两次原始超时的具体输出，也不能确定新增 QKV-A 量化是否是原因。
  仅关闭 QKV-A 量化的对照，同样 264 组输入与种子全部正常 stop、最长
  47 token；问题种子 2107 输出 19 token 后停止。异步 batch 组成未完全
  固定，且只捕获一次事件，不能据此确认因果；QKV-A 暂保留为待验证候选。
- 32 splits decode、Attention 调用方案缓存、量化 FFI 缓存、量化线程
  数调整均未纳入这里的运行路径。interval=8 的结果低于 interval=4。
- 后续按 decode batch 选择 16/32/64 splits：局部微测有收益，32/64
  splits 的变长、空行和更新索引的 CUDA graph 检查均通过。但此前服务
  factory 安装晚于图捕获，新 decode 内核没有进入图。那轮 7.52 QPS、
  1460.95 ms、1357/1357 成功以及固定参考评分，不能用于评价分片调整，
  原“无吞吐收益”的归因已撤回；需要显式提前安装后重新验证。
- 提前安装后的复测已完成，batch 1～16 的指定分片数均在捕获内部得到
  确认。并发 11：7.52 QPS、平均 E2E 1460.80 ms、P95 2543.53 ms，
  1358/1358 成功，180.57 秒，平均输出 20.24 token，服务日志实测
  token cache hit 23.21%。对比此前 dense 组合的 7.55 QPS，本次没有
  测出明确服务收益，暂不采用；这是一组有效的新证据，不改变旧实验覆盖不足
  的记录。相对 dense 的固定参考 NLL 差为 +0.007723，95% 区间
  [−0.013560, 0.029598]，top-1 一致率 88.01%；未检出显著 NLL
  退化仍不代表业务质量等价。
- 缓存 cuDNN 调用的 20 组对照与调优 CUTLASS 逐位一致，但 GPU 改进较小，
  两类共享专家投影的主机驱动耗时反而增加，暂不采用。首次微测因载入 cuDNN
  缓存时清空了 CUTLASS 缓存而不构成有效对照，已修正并将那次结果单独排除。
- LM head 按行 FP8 权重、BF16 激活候选：batch 11 微测从 0.52244 ms
  降到 0.26493 ms，但实际提前安装、确认 batch 1～16 全部进入 decode
  图后的固定参考评分未通过。相对 dense 对照，NLL 增加 0.012999，
  配对 bootstrap 95% 区间为 [0.003251, 0.022507]；相对本轮起点增加
  0.037615，区间为 [0.001402, 0.074824]。两者都检出显著退化，暂不采用。
  旧的晚安装评分没有覆盖新 decode 内核，不能作为该候选的质量证据。
- 随后的 INT8 LM head 候选保留 BF16 激活，将整数权重精确转为 BF16
  后使用 Tensor Core；按行 scale 在 FP32 累加之后应用。batch 11
  微测 0.52218→0.26436 ms，真实权重+合成激活的输出相对 L2 0.897%。
  batch 1～16 的图重放和词表尾部检查通过，PTX 确认使用 BF16 MMA。
  在实际图内路径的 100 个用例、1460 个参考 token 上，相对 dense 的
  NLL 差 +0.001354，95% 区间 [−0.006014, 0.008216]，top-1 一致率
  98.01%。未检出显著 NLL 退化，仍不能证明业务质量等价。随后容量实测：
  并发 11 为 7.58 QPS、平均 E2E 1448.95 ms、1370/1370 成功，
  平均输出 20.12 token。相对原始 +77.52%，尚未达到 80%；对比此前
  7.55 QPS 仅 +0.40%，仍需独立复测，不能将单算子 49% 当成服务增幅。
- Decode Router 的后续实验将分块 GEMM 改为 BF16 Tensor Core，并将
  FP32 归约按原来的加法顺序展开。在真实 layer 1 的 decode 输入上，
  batch 1/3/8/11/16 的耗时降低 47%～52%。20 组检查的专家选择全部
  一致，更新输入后的图重放检查通过；单独展开归约的版本保持逐位一致，
  两项组合的最大 logits 误差为 9.54e−7。batch 11/16 使用重复的
  原 batch 8 输入行，整模型质量与容量仍需验证。
- Router+INT8 组合已确认全部 46 个 MoE 层进入 batch 1～16 图。相对
  INT8 对照，固定参考 NLL 差 +0.001076，95% 区间
  [−0.016091, 0.018866]，top-1 一致率 91.51%。未检出显著 NLL
  退化，但微测专家选择一致不能扩大成整模型一致。容量为 7.57 QPS、
  平均 E2E 1451.81 ms、1367/1367 成功，平均输出 20.50 token；
  对比 INT8 的 7.58 QPS 未见 QPS 增益，输出长度变化限制了速度归因。
- 上述容量测试结束后的额外 profile 诊断发生进程 abort：
  `free(): invalid next size (fast)`，scheduler exit −6，根因尚未
  确认。只保留一段无前缀命中的 4467-token prefill trace，Attention
  占 GPU kernel 时间 24.97%、NVFP4 MoE GEMM 占 28.68%。后续连接
  快速失败形成的高 QPS 和零延迟不是容量证据。该采样不完整，不能代表
  整场压测；该轮异常退出的服务已清理，随后试验另行启动服务。

完整模型固定参考评分使用相同的 100 个 prompt、1460 个参考 token，在
强制参考 token 前记录 logits。dense+QKV-A 候选相对本轮起点：NLL 差
为 +0.01859，按用例配对 bootstrap 95% 区间为 [−0.01906, 0.05718]；
top-1 一致率 83.08%，平均 KL 0.13975，全部 logits 有限。未检出显著
NLL 退化，不代表质量等价；缩放诊断与评分使用当前数据，需要独立业务
数据验收。该评分也不是最初 FP8 权重与 routed MoE 在线 NVFP4 的独立
质量对照。

单档通过只验证该并发档位，不能宣称已经探明容量上限。不同并发下的相对
原始吞吐增幅、相同并发的增量、算子耗时降幅应分别解释，不能相加。

[Attention 流水线、逐项实验和 HTTP 结果](../ATTENTION_PIPELINE_5000PRO.html)
包含当前进展。原始日志、质量评分与微测证据保存在
`/tmp/chat-opt-20261010-goal80/`；分项提交以本次交付索引为准。

## 本次交付的回归测试

`tests/` 下保留四项独立 GPU 测试：Router 与社区 FP32 参考的 logits/top-k、
五类 dense GEMM 对公共 API 的逐位对照及图重放、INT8 LM-head 的 batch 1～16
与词表尾部、真实路由 MoE 的动态输入图重放。它们通过 unittest 执行，
需要本分支 SGLang 可导入、SM120 GPU 和 FlashInfer 0.6.15.post1。
MoE 测试通过 `SM120_MOE_CAPTURE_MANIFEST` 指定本地采集清单；模型权重
和请求张量不随代码发布，未设置时该测试明确跳过。
这些测试验证算子与适配器，不代替本次公开分支的服务性能或业务质量验收。
