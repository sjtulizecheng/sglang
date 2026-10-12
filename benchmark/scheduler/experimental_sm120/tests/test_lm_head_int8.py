"""Check integer-weight arithmetic, vocabulary tails and changed graph inputs."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lm_head_int8 import forward


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestInt8LMHead(unittest.TestCase):
    def test_graph_replays_all_decode_batches_against_fp32(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("Validated on SM120 only")
        torch.manual_seed(20261011)
        w = torch.randint(-127, 128, (131, 2048), device="cuda", dtype=torch.int8)
        scales = torch.rand(131, 1, device="cuda") / 100
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(
            setattr, torch.backends.cuda.matmul, "allow_tf32", previous_tf32
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        for batch in range(1, 17):
            with self.subTest(batch=batch):
                x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
                for _ in range(3):
                    forward(x, w, scales)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = forward(x, w, scales)
                for replay in range(3):
                    x.copy_(torch.randn_like(x) * (replay + 1))
                    graph.replay()
                    reference = (torch.mm(x.float(), w.float().T) * scales.T).bfloat16()
                    self.assertTrue(torch.isfinite(output).all())
                    error = (
                        output.float() - reference.float()
                    ).norm() / reference.float().norm()
                    self.assertLess(float(error), 0.001)
                del graph


if __name__ == "__main__":
    unittest.main()
