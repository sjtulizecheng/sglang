"""Cached CUTLASS dispatch must preserve public NVFP4 GEMM results."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvfp4_dispatch import CachedCutlassFp4


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCachedNvfp4Dense(unittest.TestCase):
    def test_five_projection_shapes_and_changed_graph_inputs(self):
        import flashinfer
        from flashinfer import mm_fp4, nvfp4_quantize

        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("Validated on SM120 only")
        if flashinfer.__version__ != "0.6.15.post1":
            self.skipTest("Tactic cache requires FlashInfer 0.6.15.post1")
        torch.manual_seed(126)
        dispatch = CachedCutlassFp4(
            str(Path(__file__).resolve().parents[1] / "nvfp4_gemm_sm120_autotune.json")
        )
        for n, k, activation_scale in (
            (8960, 512, 16),
            (5120, 768, 16),
            (2048, 5120, 512),
            (3072, 2048, 16),
            (2048, 1536, 1),
        ):
            with self.subTest(n=n, k=k):
                # Exercise a partial row tile and the original shape-specific scales.
                x = torch.randn(1025, k, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                wg = 2688.0 / weight.float().abs().max()
                ag = torch.tensor(float(activation_scale), device="cuda")
                w, sw = nvfp4_quantize(weight, wg)
                a, sa = nvfp4_quantize(x, ag)
                alpha = (wg * ag).reciprocal()

                def reference():
                    return mm_fp4(
                        a,
                        w.T,
                        sa,
                        sw.T,
                        alpha=alpha,
                        out_dtype=torch.bfloat16,
                        backend="cutlass",
                    )

                for _ in range(3):
                    actual = dispatch(a, w, sa, sw, alpha)
                self.assertTrue(torch.equal(actual, reference()))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = dispatch(a, w, sa, sw, alpha)
                new_a, new_sa = nvfp4_quantize(x * 0.5, ag)
                a.copy_(new_a)
                sa.copy_(new_sa)
                graph.replay()
                self.assertTrue(torch.equal(output, reference()))
                del graph


if __name__ == "__main__":
    unittest.main()
