# 社区基线发布与优化提交

> 后续新增优化与本次验证见 [2026-10-12 增量交付](OPTIMIZATION_COMMITS_20261012.md)。

当前 SGLang 分支直接基于社区 v0.5.17 的
**29481685462732237d80d86076d6563e1f658102**，只叠加本次 sjtushenhai
的优化与测试补丁，以及发布文档。同事的 14 个内部提交及其代码改动已移除，
没有将它们改名后重新提交。

## 仓库与提交范围

- SGLang：[sjtulizecheng/sglang · 5000pro_glm4.7_flash](https://github.com/sjtulizecheng/sglang/tree/5000pro_glm4.7_flash)。
- FlashInfer：[sjtulizecheng/flashinfer · fp8_mla_sm120_dev](https://github.com/sjtulizecheng/flashinfer/tree/fp8_mla_sm120_dev)。
- Author 和 Committer 沿用明确指定的身份：sjtushenhai <1730536718@qq.com>。
  GitHub 仓库账号已改为 sjtulizecheng，提交身份没有擅自改为其他名字。
- 只更新上述指定目标分支；不推送内部工作分支、全部分支或内部历史备份。
- SGLang 内部 CI、内部 Dockerfile、内部部署脚本、同事的 router/MoE/QKV-A、
  MIO/Qwen 和 PD multiplexing 改动均不包含在当前代码中。

## 性能数据适用范围

**当前社区基线分支尚未重新进行 GPU 服务压测，不能宣称已经达到 51%～52% 提升。**

历史组合环境测得最高达标吞吐从 4.27 提升到 6.45～6.49 QPS，提升
51.05%～51.99%；并发 9 平均 E2E 为 1395.14 / 1385.80 ms，成功率 100%。
该环境包含现在已经移除的基础改动，数据仅作为优化过程的历史证据保留。
各子项的 3.63%、6.64%、4.67% 等累计服务增量同样适用于原实验环境，
不能直接移用到当前分支或相加。

保留的优化包括在线 NVFP4 的 SM120/CUTLASS 适配、FA4 兼容修复和单次前缀
attention、原生 FP8 KV decode，以及配套 FlashInfer 的融合 FP8 prefill、
K32 PV、TMA、PRMT/softmax/布局、短 K GEMM 和 mask/LPT 调度。
PD、HRRN、最短 prefill、MoE autotune 和早期 Triton prefill 的实验补丁
单独保留，未计为最终收益，不因提交而默认启用。

## 分项 commit

### SGLang

| 改动 | Commit |
| --- | --- |
| feat(quantization): support online NVFP4 MoE on SM120 CUTLASS | [7e9ca06d39](https://github.com/sjtulizecheng/sglang/commit/7e9ca06d3957d62a614044f12b69d7e513e9e866) |
| fix(attention): make FA4 prefill compatible with SM120 | [d03d208b7a](https://github.com/sjtulizecheng/sglang/commit/d03d208b7a0aa1d4836c06fb384eefece69847a7) |
| perf(attention): use one-shot FA4 prefix prefill on SM120 | [574b0fd85e](https://github.com/sjtulizecheng/sglang/commit/574b0fd85e081d11a358b8071812afdff3de5773) |
| perf(attention): decode directly from FP8 MLA KV on SM120 | [d2c01be623](https://github.com/sjtulizecheng/sglang/commit/d2c01be623db9aeda9e390d478135f22568c1356) |
| experiment(attention): preserve standalone Triton FP8 prefill prototype | [de701e4580](https://github.com/sjtulizecheng/sglang/commit/de701e458080a4d258ebade14ee10ba3f7f77b29) |
| experiment(moe): add opt-in SM120 NVFP4 prefill autotuning | [591755bdc2](https://github.com/sjtulizecheng/sglang/commit/591755bdc27c1df6dc6aeae1947364742915820d) |
| experiment(scheduler): add token-aged HRRN policy | [399008c9a4](https://github.com/sjtulizecheng/sglang/commit/399008c9a47613b56a073e9ccb449d48ea19e630) |
| experiment(scheduler): reserve chunk budget for short prefills | [4c1573d922](https://github.com/sjtulizecheng/sglang/commit/4c1573d922f49667bc66494a970d22a47e032d8b) |
| experiment(scheduler): add synchronized prefill decode interval | [a192d9bd61](https://github.com/sjtulizecheng/sglang/commit/a192d9bd61dde90aaea9e675afd8d885f6e63504) |
| experiment(scheduler): add opt-in SLO-aware prefill decode sharing | [50cd79a82b](https://github.com/sjtulizecheng/sglang/commit/50cd79a82b94eb657dfbfe7c038c46ef6b9985f0) |
| test(scheduler): isolate prefill admission from distributed topology | [bd3a3deb62](https://github.com/sjtulizecheng/sglang/commit/bd3a3deb62f4355950e5a612f73ad5b70edf24de) |

### FlashInfer

| 改动 | Commit |
| --- | --- |
| feat(experimental): add SM120 FP8 MLA decode and prefill study | [30a74f3196](https://github.com/sjtulizecheng/flashinfer/commit/30a74f3196c7c7705c1e6924dd441b2b2faed34b) |
| perf(experimental): reuse FP8 MLA scores and probabilities on SM120 | [b5f4f69795](https://github.com/sjtulizecheng/flashinfer/commit/b5f4f69795c1976efb8b560e8028ba7759c93878) |
| experiment(sm120): preserve opt-in sharded QK research | [6483a18e3a](https://github.com/sjtulizecheng/flashinfer/commit/6483a18e3a9cc1f361acaa45a1efd40badbbaf75) |
| perf(sm120): add fused block-scaled FP8 prefill with K16 PV | [2d81fdc43f](https://github.com/sjtulizecheng/flashinfer/commit/2d81fdc43f4d6b6dce365caa57310f524c3aa93a) |
| perf(sm120): widen FP8 prefill PV to K32 | [3e25c2fb95](https://github.com/sjtulizecheng/flashinfer/commit/3e25c2fb9533843d84b0eb747563eb621304f3f0) |
| perf(sm120): use TMA and swizzled shared storage for FP8 prefill | [28385bf2f0](https://github.com/sjtulizecheng/flashinfer/commit/28385bf2f0ff55c41ac2231ef822b66de43aed6c) |
| perf(sm120): pack FP8 probability fragments with PRMT | [ffad90f628](https://github.com/sjtulizecheng/flashinfer/commit/ffad90f628dc96b1a32283fcc1eaf7240c242211) |
| perf(sm120): reduce FP8 value-scale division overhead | [c896ad61cb](https://github.com/sjtulizecheng/flashinfer/commit/c896ad61cbbfff907f73ce7ba035a08fe57975f3) |
| perf(sm120): fuse block scales into log2 online softmax | [7cfe4cf382](https://github.com/sjtulizecheng/flashinfer/commit/7cfe4cf3821de2fed7414fe936ee602935058090) |
| perf(sm120): prepare head-major FP8 attention temporaries | [d27368d7b6](https://github.com/sjtulizecheng/flashinfer/commit/d27368d7b6cd8f58aaf8b3f73adfde205dbd96b4) |
| perf(sm120): avoid Stream-K overhead for short FP8 projections | [f05215d47a](https://github.com/sjtulizecheng/flashinfer/commit/f05215d47a4208011553fe5a93af0725d8bfd6fa) |
| perf(sm120): skip interior causal masks and schedule heavy tiles first | [438ee55eec](https://github.com/sjtulizecheng/flashinfer/commit/438ee55eecc598a842178f122429a8c3d05a7b10) |
| docs(sm120): record per-item commits and verified throughput evidence | [be72aff0fb](https://github.com/sjtulizecheng/flashinfer/commit/be72aff0fb4d2649cc0e3cec279c7b4aab871949) |
| docs(sm120): scope evidence to the historical serving baseline | [f9a2ce5acf](https://github.com/sjtulizecheng/flashinfer/commit/f9a2ce5acf8ea565311238c7a8c991072c9ff010) |

## 验证

- 社区基线上的 CPU 回归：150 passed、1 skipped、52 subtests passed。
  跳过项为该 CPU 环境未注册的 MXFP4 用例。
- 11 个迁移后的代码/测试 commit 均与原提交具有相同的 stable patch ID；
  没有额外搬入同事提交中的代码。
- 测试注册、包内测试位置及最终 diff 检查通过。
- FlashInfer 实验模块导入、显式 prefill hook 安装和社区层接口已检查。
  投影 GEMM 的导入由 CUDA 条件控制；CPU 检查不代表 GPU 执行验证。
- 内部改动独占的文件恢复为社区内容或不再存在；共享文件只保留本次优化的补丁。
- 当前清理没有启动服务或压测，也没有修改原工作区或暂存内容。

[历史实现与实验说明](OPTIMIZATION_DELIVERY_20261009.md)、
[历史服务结果](results/goal50_sm120_20261009.json)、
[当前发布检查记录](results/delivery_commits_20261010.json)。

历史报告中的本地路径、源码哈希和旧环境记录保留原值，用于追溯原实验。
它们不意味着当前分支包含对应的内部提交。本文没有新增服务或压测命令。
