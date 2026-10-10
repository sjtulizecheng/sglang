"""Cover MoE prefill buckets without forwarding a whole synthetic model batch."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner.flashinfer_autotune import (
    autotune_sm120_nvfp4_prefill,
    flashinfer_autotune_cache_path,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class FakeMethod:
    def __init__(self):
        self.moe_runner_config = SimpleNamespace(
            hidden_size=32, top_k=2, activation="silu", is_gated=True
        )
        self.inputs = []

    def apply(self, layer, dispatch):
        self.inputs.append(dispatch)


class TestSM120PrefillAutotune(unittest.TestCase):
    def setUp(self):
        from sglang.srt.environ import envs

        self.enterContext(envs.SGLANG_SM120_NVFP4_PREFILL_AUTOTUNE.override(True))

    def runner(self):
        layers = [
            SimpleNamespace(
                quant_method=FakeMethod(),
                w13_weight=torch.empty((4, 64, 16)),
                w2_weight=torch.empty((4, 32, 16)),
            )
            for _ in range(2)
        ]
        return (
            SimpleNamespace(
                device="cuda",
                is_draft_worker=False,
                dtype=torch.bfloat16,
                server_args=SimpleNamespace(
                    quantization="nvfp4_online",
                    moe_runner_backend="flashinfer_cutlass",
                    max_prefill_tokens=128,
                    chunked_prefill_size=96,
                    max_running_requests=16,
                ),
                ps=SimpleNamespace(tp_size=1, moe_ep_size=1),
                model=SimpleNamespace(named_modules=lambda: enumerate(layers)),
            ),
            layers,
        )

    def test_prefill_budget_and_shared_weight_shapes(self):
        runner, layers = self.runner()
        with (
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            patch(
                "sglang.srt.layers.quantization.nvfp4_online.ModelOptNvFp4OnlineFusedMoEMethod",
                FakeMethod,
            ),
        ):
            autotune_sm120_nvfp4_prefill(runner)
        self.assertEqual(len(layers[0].quant_method.inputs), 1)
        self.assertEqual(len(layers[1].quant_method.inputs), 0)
        dispatch = layers[0].quant_method.inputs[0]
        self.assertEqual(dispatch.hidden_states.shape, (96, 32))
        ids = dispatch.topk_output.topk_ids
        self.assertEqual(ids.shape, (96, 2))
        self.assertEqual(ids.dtype, torch.int32)
        self.assertTrue(torch.all(ids[:, 0] != ids[:, 1]))
        torch.testing.assert_close(
            dispatch.topk_output.topk_weights.sum(dim=1), torch.ones(96)
        )

    def test_other_architectures_and_distributed_paths_do_not_tune(self):
        for change in ("arch", "tp", "ep", "draft", "quantization", "budget"):
            with self.subTest(change=change):
                runner, layers = self.runner()
                if change == "tp":
                    runner.ps.tp_size = 2
                elif change == "ep":
                    runner.ps.moe_ep_size = 2
                elif change == "draft":
                    runner.is_draft_worker = True
                elif change == "quantization":
                    runner.server_args.quantization = "fp8"
                elif change == "budget":
                    runner.server_args.chunked_prefill_size = 8
                with patch(
                    "torch.cuda.get_device_capability",
                    return_value=(10, 0) if change == "arch" else (12, 0),
                ):
                    autotune_sm120_nvfp4_prefill(runner)
                self.assertFalse(any(x.quant_method.inputs for x in layers))

    def test_experimental_tactics_do_not_replace_default_cache(self):
        from sglang.srt.environ import envs

        runner, _ = self.runner()
        runner.server_args.model_path = "/model/test"
        runner.server_args.flashinfer_autotune_skip_ops = None
        runner.model_config = SimpleNamespace(hf_config=SimpleNamespace())
        runner.ps.pp_size = runner.ps.attn_dp_size = 1
        runner.ps.tp_rank = runner.ps.pp_rank = 0
        runner.ps.dp_rank = None
        with (
            TemporaryDirectory() as cache_dir,
            envs.SGLANG_CACHE_DIR.override(cache_dir),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            experiment_path = flashinfer_autotune_cache_path(runner)
            with envs.SGLANG_SM120_NVFP4_PREFILL_AUTOTUNE.override(False):
                default_path = flashinfer_autotune_cache_path(runner)
            self.assertNotEqual(experiment_path, default_path)


if __name__ == "__main__":
    unittest.main()
