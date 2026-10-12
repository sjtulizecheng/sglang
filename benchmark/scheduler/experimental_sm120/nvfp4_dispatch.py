"""Local version-specific fast dispatch for already validated NVFP4 operands."""

import torch
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm.gemm_base import (
    DEFAULT_WORKSPACE_SIZE,
    _MM_FP4_TUNING_CONFIG_128x4,
    get_cutlass_fp4_gemm_module,
)


class CachedCutlassFp4:
    def __init__(self, cache_path):
        import flashinfer

        if flashinfer.__version__ != "0.6.15.post1":
            raise RuntimeError("Cached NVFP4 tactics require FlashInfer 0.6.15.post1")
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("Cached NVFP4 tactics are validated on SM120 only")
        self.runner = get_cutlass_fp4_gemm_module(12, 0).cutlass_fp4_gemm_runner()
        self.kernel = self.runner._fp4_gemm_runner
        self.workspace = torch.empty(
            DEFAULT_WORKSPACE_SIZE, device="cuda", dtype=torch.uint8
        )
        with autotune(False, cache=cache_path):
            pass
        self.tuner = AutoTuner.get()
        self.tactics = {}

    def __call__(self, a, w, sa, sw, alpha):
        # Only used by the opted-in static model: contiguous packed rows,
        # 128x4 scale layout, FP4 E2M1 operands, BF16 output, one CUDA device.
        m, packed_k, n = a.shape[0], a.shape[1], w.shape[0]
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        key = (m, packed_k, n)
        if key not in self.tactics:
            inputs = [
                a,
                w.T,
                sa,
                sw.T,
                alpha,
                torch.bfloat16,
                out,
                16,
                True,
                self.workspace,
            ]
            _, tactic = self.tuner.choose_one(
                "fp4_gemm", [self.runner], _MM_FP4_TUNING_CONFIG_128x4, inputs
            )
            self.tactics[key] = tactic
        if a.dtype == torch.uint8 and sa.dtype == torch.float8_e4m3fn:
            sa = sa.view(torch.uint8)
        if w.dtype == torch.uint8 and sw.dtype == torch.float8_e4m3fn:
            sw = sw.view(torch.uint8)
        self.kernel(a, w, sa, sw, alpha, out, self.workspace, self.tactics[key])
        return out
