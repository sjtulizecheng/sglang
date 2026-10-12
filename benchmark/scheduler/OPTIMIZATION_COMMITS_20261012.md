# 2026-10-12 优化分项交付

本次在指定远端分支的现有提交上追加，没有重写历史：

- SGLang：`sjtulizecheng/sglang` 的 `5000pro_glm4.7_flash`，起点 `b8835cd473`。
- FlashInfer：`sjtulizecheng/flashinfer` 的 `fp8_mla_sm120_dev`，起点 `f9a2ce5acf`。
- 新增 commit 的 Author / Committer 均为 `sjtushenhai <1730536718@qq.com>`。
- 本地同事提交、暂存区和工作区保留；未将内部工作分支的其他历史带入公开分支。

## 本次 SGLang 分项提交

| 优化项 | Commit |
| --- | --- |
| CUDA Graph 前后 hook 安装时序 | [cd1c2b3ee3](https://github.com/sjtulizecheng/sglang/commit/cd1c2b3ee34e3122b5eb26401a02aeffe613ab8e) |
| Prefill Router：BF16 Tensor Core、FP32 累加 | [7c010c274a](https://github.com/sjtulizecheng/sglang/commit/7c010c274ab946ca3e35466cd11a27fb32f78801) |
| Dense / shared-expert NVFP4 + CUTLASS 调优缓存 | [bb532dc247](https://github.com/sjtulizecheng/sglang/commit/bb532dc24786aa5a29090c2013946cd798e5fad3) |
| INT8 LM head：BF16 Tensor Core 计算 | [8e50fa3f72](https://github.com/sjtulizecheng/sglang/commit/8e50fa3f72f1ead428c7708254b38f96d0e1b193) |
| 真实路由 NVFP4 MoE：tactic 12 / 42 | [fbc4998e08](https://github.com/sjtulizecheng/sglang/commit/fbc4998e08ddc26045c9c631b6bdec44a9491c34) |
| QKV-A NVFP4 补齐实验（未采用） | [caa7e0af6d](https://github.com/sjtulizecheng/sglang/commit/caa7e0af6d47f4293b5a557f5f74b1034b3e471a) |

代码在 [experimental_sm120](experimental_sm120/README.md)，均需显式 factory
启用。默认启动不会自动打开这些实验。普通 hook 仍在图捕获后安装；
需要修改 decode 图的 INT8 LM head 显式选择 `before_graph`。
QKV-A 候选单独保留，不包含在最终采用组合中。

迁移仅对两个私有 dispatch 增加硬件/版本检查，并进行格式整理；没有改变
已测 Tensor Core 内核的计算逻辑。MoE tactic 编号和 dense 调优缓存依赖
FlashInfer **0.6.15.post1**，不能因研究仓库存在这些文件就覆盖安装整个包。
只把配套 FlashInfer 的 `flashinfer/experimental/mla_fp8_sm120` 实验目录
加入模块搜索路径。完整配置与精度范围见实验目录 README。

## 已有优化与 FlashInfer 本次交付

在线 NVFP4 MoE 的 SM120 CUTLASS 适配、FA4 兼容与单次前缀合并、原生
FP8 MLA decode、调度实验已经在 SGLang 起点历史中；没有重复制造 commit。
全部历史条目见 [2026-10-10 分项索引](OPTIMIZATION_COMMITS_20261010.md)。

FlashInfer 的运行源码及测试与本地逐文件比较一致。本次在其原分支追加
交付说明和社区基线 GPU 验证记录，原有分项提交保持不变：

| 优化项 | 已有 Commit |
| --- | --- |
| 融合 FP8 prefill，K16 PV | `2d81fdc43f` |
| PV K32 指令及加载布局 | `3e25c2fb95` |
| TMA 与 shared-memory swizzle | `28385bf2f0` |
| PRMT 概率寄存器重排 | `ffad90f628` |
| V scale 倒数 | `c896ad61cb` |
| log2 在线 softmax 与 scale 融合 | `7cfe4cf382` |
| head-major QKV 临时布局 | `d27368d7b6` |
| 短 K dense FP8 GEMM 调度 | `f05215d47a` |
| 完整块 mask 优化与 LPT 调度 | `438ee55eec` |

上述 FlashInfer 条目的链接、条件和历史数据见其
[MATERIALIZED_PREFILL.md](https://github.com/sjtulizecheng/flashinfer/blob/fp8_mla_sm120_dev/flashinfer/experimental/mla_fp8_sm120/MATERIALIZED_PREFILL.md)。

## 历史性能证据的范围

本次只做迁移、回归验证与提交，**没有重新发起完整服务压测**。
下列结果来自包含同事优化的原本地环境（SGLang 基点 `b1478c4293`），
不能宣称本次公开社区基线分支已经达到同样吞吐。

| 历史组合 | 并发 | QPS | 平均 E2E(ms) | P95 E2E(ms) | 成功请求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始基线 | 6 | 4.27 | 1403.52 | — | — |
| 最终组合首轮 | 11 | 7.73 | 1422.22 | 2458.79 | 1395/1395 |
| 独立重启复测 | 11 | 7.74 | 1420.48 | 2437.82 | 1395/1395 |

按显示 QPS 计算，达标吞吐增加 81.03% / 81.26%。原始 C6 与最终 C11
不是同并发单项 A/B，百分比也不是某一个 kernel 的提速。两次 C11 测试
都在最大并发档停止，没有探明容量上限。

每轮 180 秒、用例版本 202609111724，固定预热后清 KV，实际 token cache
hit 为 23.22% / 23.23%。编译缓存与 KV/prefix cache 是不同层次。
Router 的历史 C9 筛选为 6.50→6.62 QPS；INT8 LM head 的单算子约降时
49%，服务仅 7.55→7.58 QPS。真实路由 MoE 的完整算子耗时降低
6.53%～9.81%，服务从 7.58 到 7.73/7.74 QPS。dense 与调度组合没有
独立隔离每一部分的 QPS，不能把全部变化归因于 NVFP4 GEMM。

量化路径的数值/固定参考 NLL 检查不等于业务质量验收。QKV-A 复测未稳定
达标且出现罕见长输出，未采用。低成功率快速失败的异常 profile 数据不计入容量。

- [历史聚合结果](results/goal80_sm120_20261011.json)：原始指标、配置与证据哈希。
- [独立 Attention HTML](ATTENTION_PIPELINE_5000PRO.html)：流水线、分项实验、历史吞吐。

## 本次验证

验证使用本次公开基线代码，PyTorch 2.11.0+cu129、FlashInfer 0.6.15.post1、
CuTe DSL 4.6.0 和 RTX PRO 5000 72GB（SM120）。

- 4 项 CPU 测试：默认/显式 hook 时序、空列表、错误阶段拒绝。
- Router：M=1/31/32/1025/5093，与社区 FP32 logits 和 top-k 对照通过。
- Dense：五类投影在 M=1025 时与公共 NVFP4 API 逐位一致，修改量化输入后图重放通过。
- INT8 LM head：batch 1～16、词表尾部、每档 3 次修改输入图重放，对 FP32 参考相对 L2 <0.001。
- MoE：3 个真实层 × 4 种 token 数（1024/1025/8192/16384），eager 与修改激活/路由后的图重放逐位一致。
- 配套 FlashInfer：2 项 FP8 prefill 测试（变长/因果空行、动态图输入），以及 1 项投影数值测试通过。
- 所有实验 factory 在公开基线上导入与安装通过；无需导入同事的内部模块。
- Python AST、Black、isort、注册测试检查、diff 空白检查和新增文件凭据扫描通过。
- HTML 桌面 1440 / 手机 390 像素下无横向溢出，两个选择器及复位功能通过，无页面错误或外部网络请求。

MoE 真实输入文件留在本机，不提交模型权重或请求内容。
测试覆盖范围以 [本次验证清单](results/publication_validation_20261012.json) 为准；
未运行的完整服务与业务质量验收不计为通过。
