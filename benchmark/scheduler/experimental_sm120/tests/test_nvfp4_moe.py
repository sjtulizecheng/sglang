"""Replay screened MoE tactics against default tactics with changed routing.

Set SM120_MOE_CAPTURE_MANIFEST to a local real-input capture manifest. Model
weights and request tensors are deliberately not distributed with this test.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvfp4_moe_prefill import native_forward, native_runner


def to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, list):
        return [to_cuda(item) for item in value]
    return value


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestNvfp4RoutedMoe(unittest.TestCase):
    def test_changed_activations_routing_and_graph_replay(self):
        import flashinfer

        manifest_path = os.environ.get("SM120_MOE_CAPTURE_MANIFEST")
        if not manifest_path:
            self.skipTest("Set SM120_MOE_CAPTURE_MANIFEST for real-input replay")
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("Validated on SM120 only")
        if flashinfer.__version__ != "0.6.15.post1":
            self.skipTest("Tactics require FlashInfer 0.6.15.post1")
        manifest = json.loads(Path(manifest_path).read_text())
        runner = native_runner()
        selected = {}
        for row in manifest["captures"]:
            selected.setdefault(row["layer_ordinal"], row)
        self.assertTrue(selected, "Capture manifest is empty")
        for layer, row in selected.items():
            weights = torch.load(
                manifest["layers"][str(layer)]["weights"],
                weights_only=True,
                map_location="cpu",
            )
            call = torch.load(row["path"], weights_only=True, map_location="cpu")
            saved = {
                k: to_cuda(v)
                for k, v in (weights | call).items()
                if k not in ("reference_output", "tactics")
            }
            for m in (1024, 1025, 8192, 16384):
                with self.subTest(layer=layer, tokens=m):
                    args = saved.copy()
                    for name in (
                        "input",
                        "token_selected_experts",
                        "token_final_scales",
                    ):
                        t = saved[name]
                        args[name] = t.repeat((m + t.shape[0] - 1) // t.shape[0], 1)[
                            :m
                        ].contiguous()
                    out = torch.empty(m, 2048, device="cuda", dtype=torch.bfloat16)
                    ref = torch.empty_like(out)
                    native_forward(runner, args, ref)
                    native_forward(runner, args, out, (12, 42))
                    self.assertTrue(torch.equal(out, ref))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        native_forward(runner, args, out, (12, 42))
                    args["input"].mul_(0.875)
                    for name in ("token_selected_experts", "token_final_scales"):
                        args[name].copy_(args[name].roll(1, dims=0))
                    graph.replay()
                    native_forward(runner, args, ref)
                    self.assertTrue(torch.equal(out, ref))
                    del graph


if __name__ == "__main__":
    unittest.main()
