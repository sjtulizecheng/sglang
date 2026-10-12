"""Exercise the installed router against the community FP32 gate reference."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorcore_router


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTensorCoreRouter(unittest.TestCase):
    def test_logits_and_topk_with_prefill_and_decode_shapes(self):
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteGate

        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("Validated on SM120 only")
        torch.manual_seed(126)
        original = Glm4MoeLiteGate.forward
        gate = SimpleNamespace(
            weight=torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16) / 64,
            e_score_correction_bias=torch.rand(64, device="cuda") / 10,
            _weight_fp32=None,
        )
        # Isolate this optimization from the preceding attention factory.
        dependency = SimpleNamespace(create_hook=lambda _: lambda *args: None)
        with (
            patch.dict(sys.modules, {"projection_tuning": dependency}),
            patch.object(Glm4MoeLiteGate, "forward", original),
            torch.no_grad(),
        ):
            tensorcore_router.create_hook({"router_validation_calls": 1})
            for m in (1, 31, 32, 1025, 5093):
                with self.subTest(tokens=m):
                    x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
                    expected = original(gate, x)
                    actual = Glm4MoeLiteGate.forward(gate, x)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                    bias = gate.e_score_correction_bias
                    a = (actual.sigmoid() + bias).topk(4).indices.sort(-1).values
                    b = (expected.sigmoid() + bias).topk(4).indices.sort(-1).values
                    self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
